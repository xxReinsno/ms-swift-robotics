#!/bin/bash
# 路径配置
CUSTOM_REGISTER_PATH=/home/yuquan002/ssd/xyq_ws/ms-swift-robotics/vla-0/data/dataset_register.py

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0,4,5
export NPROC_PER_NODE=3
export ROOT_IMAGE_DIR=/home/yuquan002/ssd/xyq_ws/libero_vl_dataset/libero_omni_swift
export WANDB_PROJECT='Qwen3-VL-Robotics'
export WANDB_RUN_NAME="Qwen3-VL-4B-VLA0-$(date +%Y%m%d_%H%M%S)"

swift sft \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --custom_register_path $CUSTOM_REGISTER_PATH \
    --dataset libero-vla0 \
    --load_from_cache_file true \
    --report_to tensorboard wandb \
    --use_hf true \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 24 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-6 \
    --lr_scheduler_type constant \
    --weight_decay 1e-10 \
    --max_grad_norm 0 \
    --warmup_ratio 0 \
    --max_length 2048 \
    --attn_impl flash_attn \
    --padding_free true \
    --gradient_checkpointing true \
    --gradient_accumulation_steps 1 \
    --eval_steps 500 \
    --save_steps 2000 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --output_dir output/omnivla \
    --loss_type vla0_loss \
    --loss_scale vla0 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 8 \
    # --deepspeed zero2
