import os
import random
import torch
from typing import List, Dict, Any, Optional

# --- 1. 精确导入 ms-swift 组件 (Based on ms-swift source structure) ---

from swift.llm import get_model_tokenizer, load_dataset, get_template, EncodePreprocessor
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from swift.llm.dataset import register_dataset, DatasetMeta

# 引入 HuggingFace 的组件
from transformers import HfArgumentParser, PreTrainedTokenizerBase
from transformers import DataCollatorForSeq2Seq

logger = get_logger()

# --- 2. 自定义 DataCollator (OmniVLA 核心增强逻辑) ---
class OmniVLADataCollator(DataCollatorForSeq2Seq):
    """
    复刻 OmniVLA 的随机掩码增强：
    在训练时随机将 Action Token 替换为 '?' 并忽略其 Loss。
    """
    def __init__(self, tokenizer, model=None, padding=True, action_mask_aug_per=0.1):
        super().__init__(tokenizer, model=model, padding=padding)
        self.action_mask_aug_per = action_mask_aug_per
        # 动态获取 '?' 的 Token ID (Qwen2.5/3 通常是 30 或其他符号)
        self.mask_token_id = tokenizer.convert_tokens_to_ids("?")
        if isinstance(self.mask_token_id, list):
            self.mask_token_id = self.mask_token_id[0]
        
        logger.info(f"OmniVLA Augmentation Enabled: Mask Token='?' (ID={self.mask_token_id}), Ratio={action_mask_aug_per}")

    def __call__(self, features, return_tensors=None):
        # 1. 执行标准的 Padding (ms-swift 的 template 已经处理好了 input_ids)
        batch = super().__call__(features, return_tensors=return_tensors)
        
        # 如果增强比例为0，直接返回
        if self.action_mask_aug_per <= 0:
            return batch

        input_ids = batch['input_ids']
        labels = batch['labels']
        bs = input_ids.shape[0]
        
        for i in range(bs):
            # 10% 的概率不进行增强，保持原样
            if random.random() < 0.1:
                continue
            
            # 随机确定当前样本的 Mask 比例 (0.0 ~ max_per)
            aug_per = random.uniform(0.0, self.action_mask_aug_per)
            
            # 找到属于 Action 的部分 (labels != -100)
            # ms-swift 中，System/User Prompt 的 label 默认为 -100
            action_indices = (labels[i] != -100).nonzero(as_tuple=True)[0]
            
            if len(action_indices) == 0:
                continue
                
            # 计算需要 Mask 的 Token 数量
            mask_len = int(len(action_indices) * aug_per)
            if mask_len == 0:
                continue
            
            # 随机选择索引
            # 注意：random.sample 的范围是 action_indices 的长度，然后映射回 global 索引
            selected_indices = random.sample(range(len(action_indices)), mask_len)
            global_indices = action_indices[selected_indices]
            
            # 应用 Mask:
            # 1. 输入变为 '?' (Mask Token)
            # 2. 标签变为 -100 (忽略 Loss)
            batch['input_ids'][i, global_indices] = self.mask_token_id
            batch['labels'][i, global_indices] = -100
            
        return batch

# --- 3. 数据集注册辅助函数 ---
def register_local_libero(dataset_dir):
    """注册本地 JSONL 数据集到 ms-swift"""
    dataset_name = 'libero-swift-vla'
    # 兼容目录或文件路径
    if os.path.isdir(dataset_dir):
        jsonl_path = os.path.join(dataset_dir, 'train.jsonl')
    else:
        jsonl_path = dataset_dir

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"数据集文件未找到: {jsonl_path}，请先运行数据转换脚本！")

    register_dataset(DatasetMeta(
        dataset_name=dataset_name,
        dataset_path=jsonl_path,
        split=['train'] # 假设只有一个 train.jsonl
    ))
    return dataset_name




def train():
    # 1. 解析命令行参数
    parser = HfArgumentParser(SftArguments)
    args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    if len(remaining_args) > 0:
        logger.warning(f"Warning: The following arguments were unused: {remaining_args}")

    # 2. 注册 OmniVLA 数据集 (使用 dataset 参数中的路径作为基准)
    # 注意：我们这里玩了个 trick，假设用户传的 --dataset 是路径
    # 如果用户传的是名字 'libero-vla'，这步会跳过，假设已注册
    dataset_path = args.dataset[0]
    if '/' in dataset_path and os.path.exists(dataset_path):
        logger.info(f"Detected local dataset path: {dataset_path}, registering...")
        dataset_name = register_local_libero(dataset_path)
        args.dataset = [dataset_name] # 替换为注册名
    
    # 3. 设置环境
    seed_everything(args.seed)
    
    # 4. 加载模型 & Tokenizer
    logger.info(f"Loading model: {args.model_type}")
    model, tokenizer = get_model_tokenizer(
        args.model_type, 
        args.model_id_or_path, 
        model_kwargs={'device_map': 'auto'},
        load_model=True
    )
    model = prepare_model(model, args)

    # 5. 加载 & 预处理数据
    logger.info("Processing dataset...")
    template = get_template(args.template_type, tokenizer, args.system, args.max_length, args.truncation_strategy)
    train_dataset, val_dataset = load_dataset(args.dataset, args.dataset_test_ratio, args.dataset_seed)
    
    # 使用 ms-swift 标准编码流程
    # 注意: 为了兼容性，使用 map 手动 encode
    def _encode(example):
        return template.encode(example)

    train_dataset = train_dataset.map(
        _encode, batched=False, num_proc=args.dataset_num_proc, remove_columns=train_dataset.column_names
    )
    if val_dataset:
        val_dataset = val_dataset.map(
            _encode, batched=False, num_proc=args.dataset_num_proc, remove_columns=val_dataset.column_names
        )

    # 6. 【关键】注入自定义 OmniVLA DataCollator
    logger.info("Injecting OmniVLA DataCollator...")
    data_collator = OmniVLADataCollator(
        tokenizer=tokenizer, 
        padding=True, 
        action_mask_aug_per=0.1 # 这里硬编码了 0.1，与 OmniVLA 一致
    )

    # 7. 初始化 Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args.training_args,
        data_collator=data_collator, # <--- 注入点
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # 8. 开始训练
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Training finished. Model saved to {args.output_dir}")

if __name__ == "__main__":
    import os
    train()