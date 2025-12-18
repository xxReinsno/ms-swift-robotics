import json
import shutil
import numpy as np
import tensorflow_datasets as tfds
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# ================= 配置部分 =================
# 本地 RLDS 数据集路径
DATA_DIR = "/home/yuquan002/ssd/modified_libero_rlds" 

# 输出路径 (将包含 images 文件夹和 train.jsonl)
OUTPUT_DIR = "/home/yuquan002/ssd/xyq_ws/libero_vl_dataset/libero_omni_swift"

# 数据集名称
DATASET_NAMES = [
    "libero_spatial_no_noops", 
    "libero_object_no_noops", 
    "libero_goal_no_noops", 
    "libero_10_no_noops"
]

# 核心参数 (与 OmniVLA 保持一致)
CHUNK_SIZE = 8       # 预测未来 8 步动作
HISTORY = 1          # 输入历史帧数 (OmniVLA 默认为 1)
NUM_BINS = 1000      # 动作离散化分箱数
IMG_SIZE = (224, 224) # 图片 Resize 大小
# 相机配置: RLDS 中的 key -> 保存的文件后缀
CAMERAS = {
    "image": "agent",
    "wrist_image": "wrist"
}

# System Prompt
SYSTEM_PROMPT = (
    f"Analyze the input image(s) and predict robot actions for the next {CHUNK_SIZE} timesteps. "
    f"Each action has 7 dimensions. Output a single sequence of {CHUNK_SIZE * 7} integers "
    f"(0-{NUM_BINS} each), representing the {CHUNK_SIZE} timesteps sequentially. "
    "Provide only space separated numbers. Nothing else."
)
# ===========================================

class ActionIntegrator:
    def __init__(self, min_act, max_act, num_bins=1000):
        self.min_act = np.array(min_act)
        self.max_act = np.array(max_act)
        self.num_bins = num_bins
        # 避免除以 0
        self.scale = self.max_act - self.min_act
        self.scale[self.scale == 0] = 1.0

    def encode(self, actions):
        # 归一化 [0, 1]
        norm_actions = (actions - self.min_act) / self.scale
        norm_actions = np.clip(norm_actions, 0, 1)
        # 离散化
        discrete_actions = np.round(norm_actions * self.num_bins).astype(int)
        return " ".join(map(str, discrete_actions.flatten().tolist()))

def get_dataset_stats(data_dir, dataset_names):
    """遍历数据集计算全局 Min/Max Action 并保存"""
    print("正在计算数据集统计值 (Min/Max Action)...")
    all_actions = []
    
    for ds_name in dataset_names:
        try:
            ds = tfds.load(ds_name, data_dir=data_dir, split='train')
            for episode in tqdm(ds, desc=f"Scanning {ds_name}"):
                for step in episode['steps']:
                    all_actions.append(step['action'].numpy())
        except Exception as e:
            print(f"Skipping stats for {ds_name}: {e}")

    if not all_actions:
        raise ValueError("未找到任何动作数据，请检查路径。")

    all_actions = np.array(all_actions)
    stats = {
        'min': np.min(all_actions, axis=0).tolist(),
        'max': np.max(all_actions, axis=0).tolist()
    }
    
    # 保存统计值 (重要：推理时需要！)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)
    print(f"统计值已保存至 {OUTPUT_DIR}/dataset_stats.json")
    
    return stats

def process_dataset():
    output_path = Path(OUTPUT_DIR)
    images_dir = output_path / "images"
    
    if output_path.exists():
        shutil.rmtree(output_path)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 获取并保存统计值
    stats = get_dataset_stats(DATA_DIR, DATASET_NAMES)
    integrator = ActionIntegrator(stats['min'], stats['max'], NUM_BINS)
    
    jsonl_entries = []
    global_idx = 0
    
    print("开始生成训练数据...")
    for ds_name in DATASET_NAMES:
        try:
            dataset = tfds.load(ds_name, data_dir=DATA_DIR, split="train")
        except Exception as e:
            continue

        for episode in tqdm(dataset, desc=f"Processing {ds_name}"):
            # 加载整个 episode 以处理时序窗口
            steps = list(episode['steps'].as_numpy_iterator())
            episode_len = len(steps)
            
            for i in range(episode_len):
                # --- A. 处理图像 (History Window) ---
                img_paths = []
                # 获取 t - HISTORY + 1 到 t 的帧
                for t_offset in range(1 - HISTORY, 1):
                    t = max(0, i + t_offset) # Padding: 如果越界则取第一帧
                    step_t = steps[t]
                    
                    # 遍历所有需要的相机
                    for cam_key, suffix in CAMERAS.items():
                        obs = step_t['observation']
                        # 兼容不同的 key 命名 (有些数据集是 image, 有些是 observation/image)
                        if cam_key in obs:
                            img_arr = obs[cam_key]
                        else:
                            continue # Skip missing camera
                            
                        # 保存图片
                        # 命名格式: globalIdx_timestep_camera.jpg
                        # 注意: 这里的 global_idx 是样本维度的，t_offset 区分历史帧
                        file_name = f"{global_idx:09d}_seq{t_offset}_{suffix}.jpg"
                        save_path = images_dir / file_name
                        
                        Image.fromarray(img_arr).resize(IMG_SIZE).save(save_path, quality=95)
                        img_paths.append(str(Path("images") / file_name))

                # --- B. 处理动作 (Future Chunk) ---
                action_chunk = []
                for j in range(CHUNK_SIZE):
                    t = i + j
                    if t < episode_len:
                        action_chunk.append(steps[t]['action'])
                    else:
                        # Padding: 重复最后一帧动作
                        action_chunk.append(steps[-1]['action'])
                
                response_str = integrator.encode(np.array(action_chunk))
                
                # --- C. 构造 ms-swift 样本 ---
                # <image> 占位符数量 = 图片数量
                query_prompt = f"{steps[i]['language_instruction'].decode('utf-8')}"
                query_prompt += "<image>" * len(img_paths)
                
                entry = {
                    "query": query_prompt,
                    "response": response_str,
                    "system": SYSTEM_PROMPT,
                    "images": img_paths
                }
                jsonl_entries.append(entry)
                global_idx += 1

    # 保存 JSONL
    with open(output_path / "train.jsonl", "w", encoding="utf-8") as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"完成！数据已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_dataset()