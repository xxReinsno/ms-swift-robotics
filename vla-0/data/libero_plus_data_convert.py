import json
import os
import shutil
import pickle
import numpy as np
import tensorflow_datasets as tfds
import tyro
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict

# --- 配置常量 ---
# 对应 OmniVLA 的相机顺序: [AgentView, WristView]
# 这里的 key 必须对应 RLDS 数据集 observation 中的实际 key
CAMERA_CONFIGS = [
    ("image", "agentview_rgb"),       # 第三人称
    ("wrist_image", "eye_in_hand_rgb") # 第一人称
]

@dataclass
class ActionTokenizer:
    """
    OmniVLA 风格的动作分词器
    完全对齐 rv_train/models/qwen/model.py 中的 QwenActor.get_text_action
    """
    min_act: np.ndarray
    max_act: np.ndarray
    num_bins: int = 1000

    def encode(self, actions: np.ndarray) -> str:
        """
        actions: (Chunk_Size, Action_Dim)
        """
        # 1. 归一化 (act - min) / (max - min)
        denominator = self.max_act - self.min_act
        # 防止除以0
        denominator = np.where(denominator == 0, 1.0, denominator)
        
        norm_actions = (actions - self.min_act) / denominator

        # 2. 截断 (Clip) 到 [0, 1] 范围，防止越界
        norm_actions = np.clip(norm_actions, 0, 1)
        
        # 3. 离散化 (0 ~ num_bins)
        discrete_actions = np.round(norm_actions * self.num_bins).astype(int)
        
        # 4. 转为扁平的字符串 (Space Separated)
        return " ".join(map(str, discrete_actions.flatten().tolist()))

def compute_stats(dataset) -> Dict[str, np.ndarray]:
    """
    遍历数据集计算全局 Min/Max
    """
    print("正在扫描全量数据以计算统计值 (Min/Max)...")
    min_vals = []
    max_vals = []
    
    # 遍历 dataset (tf.data.Dataset)
    # 注意：如果数据量巨大，这里可能比较耗时，但对 VLA 训练至关重要
    for episode in tqdm(dataset, desc="Computing Stats"):
        try:
            # 获取该 episode 所有动作
            # steps 是一个 generator，转为 list 或 numpy
            actions = np.array([step['action'] for step in episode['steps'].as_numpy_iterator()])
            if len(actions) > 0:
                min_vals.append(np.min(actions, axis=0))
                max_vals.append(np.max(actions, axis=0))
        except Exception as e:
            print(f"[Warning] Error reading episode during stats calc: {e}")

    if not min_vals:
        raise ValueError("未能读取到任何动作数据，无法计算统计值。")

    global_min = np.min(np.stack(min_vals), axis=0)
    global_max = np.max(np.stack(max_vals), axis=0)
    
    print(f"统计计算完成:\nMin: {global_min}\nMax: {global_max}")
    return {"min": global_min, "max": global_max}

def main(
    data_dir: str = "/home/yuquan002/ssd/xyq_ws/libero_plus_mixdata/libero_mix", # 更新后的数据源
    output_dir: str = "/home/yuquan002/ssd/xyq_ws/libero_vl_dataset/libero_plus_omnivla",
    stats_path: Optional[str] = None, 
    chunk_size: int = 8,
    action_bins: int = 1000,
    overwrite: bool = True,
):
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    jsonl_path = output_path / "train.jsonl"

    # 1. 目录清理
    if output_path.exists():
        if overwrite:
            print(f"清理旧目录: {output_path}")
            shutil.rmtree(output_path)
        else:
            print(f"目录已存在且 overwrite=False，跳过: {output_path}")
            return
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"正在加载数据集: {data_dir}")
    try:
        # [关键修改] 使用 builder_from_directory 直接加载指定路径的数据集
        # 这不需要数据集名称注册，直接读取 data_dir 下的 dataset_info.json
        builder = tfds.builder_from_directory(data_dir)
        dataset = builder.as_dataset(split='train')
    except Exception as e:
        print(f"错误: 无法从 {data_dir} 加载数据集。\n详情: {e}")
        print("请确保该目录下包含 dataset_info.json 和 .tfrecord 文件。")
        return

    # 2. 获取统计信息 (优先加载文件，否则计算)
    if stats_path and os.path.exists(stats_path):
        print(f"加载统计文件: {stats_path}")
        with open(stats_path, "rb") as f:
            loaded_stats = pickle.load(f)
            if 'out_ori_act' in loaded_stats: loaded_stats = loaded_stats['out_ori_act']
            stats = loaded_stats
    else:
        # 默认执行重新计算
        stats = compute_stats(dataset)
        # 可选：保存计算出的 stats 以备后用
        with open(output_path / "dataset_stats.pkl", "wb") as f:
            pickle.dump(stats, f)
            print(f"统计值已保存至 {output_path / 'dataset_stats.pkl'}")

    # 初始化 Tokenizer (假设动作维度为 7)
    action_dim = 7
    action_tokenizer = ActionTokenizer(
        min_act=stats['min'][:action_dim], 
        max_act=stats['max'][:action_dim], 
        num_bins=action_bins
    )

    # 3. 构造 System Prompt (完全对齐 model.py)
    total_tokens = chunk_size * action_dim
    system_prompt = (
        f"Analyze the input image and predict robot actions for the next {chunk_size} timesteps. "
        f"Each action has {action_dim} dimensions. Output a single sequence of {total_tokens} integers "
        f"(0-{action_bins} each), representing the {chunk_size} timesteps sequentially. "
        "Provide only space separated numbers. Nothing else."
    )

    total_samples = 0
    print(f"开始转换数据，写入: {jsonl_path}")

    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        # 遍历数据集
        for episode in tqdm(dataset, desc="Converting"):
            # 预读取所有 steps
            steps = list(episode["steps"].as_numpy_iterator())
            episode_len = len(steps)
            
            if episode_len == 0: continue

            # 获取该 episode 最后一步动作，用于 Padding
            last_action = steps[-1]["action"]

            for i, step in enumerate(steps):
                # --- A. 处理图像 ---
                image_paths = []
                obs = step["observation"]
                
                # 检查所有必要的相机是否存在
                has_all_cameras = True
                current_img_tokens = []
                
                for rlds_key, suffix in CAMERA_CONFIGS:
                    if rlds_key in obs:
                        img_arr = obs[rlds_key]
                        # 文件名: {global_index}_{camera_suffix}.jpg
                        fname = f"{total_samples:09d}_{suffix}.jpg"
                        save_path = images_dir / fname
                        
                        # 转换并保存
                        Image.fromarray(img_arr).save(save_path, quality=95)
                        image_paths.append(str(Path("images") / fname))
                        current_img_tokens.append("<image>")
                    else:
                        # 如果是必须的相机缺失，这里可以选择跳过或者填黑图
                        # 为了严格对齐 OmniVLA，假设数据完整
                        pass
                
                if not image_paths: continue

                # --- B. 构造 Query ---
                # 格式: <image><image>...Task Description
                # 注意: 这里的 instruction decode 需要确认数据集中 instruction 是 bytes 还是 string
                try:
                    if isinstance(step["language_instruction"], bytes):
                        instruction = step["language_instruction"].decode("utf-8")
                    else:
                        instruction = str(step["language_instruction"])
                except:
                    instruction = "Perform the task." # Fallback

                image_tokens_str = "".join(current_img_tokens)
                query = f"{image_tokens_str}{instruction}"

                # --- C. 动作 Chunking ---
                future_actions = []
                for k in range(chunk_size):
                    if i + k < episode_len:
                        future_actions.append(steps[i + k]["action"])
                    else:
                        # [关键对齐] 超出范围时，重复最后一步动作 (Repeat Padding)
                        future_actions.append(last_action)
                
                # 堆叠 & 维度截断 (确保是 action_dim)
                actions_np = np.stack(future_actions)[:, :action_dim]
                
                # 编码 Response
                response_str = action_tokenizer.encode(actions_np)

                # --- D. 写入 ---
                sample = {
                    "system": system_prompt,
                    "query": query,
                    "response": response_str,
                    "images": image_paths
                }
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total_samples += 1

    print(f"\n转换完成! 总样本: {total_samples}")
    print(f"数据输出至: {output_dir}")

if __name__ == "__main__":
    tyro.cli(main)