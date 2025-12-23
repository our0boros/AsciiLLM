#!/usr/bin/env python3
"""
数据集准备脚本 - 生成用于SFT训练的ASCII艺术数据
"""

import os
import sys
import sqlite3
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.dataset_pipeline import DatasetGenerator, Config

def prepare_dataset(args):
    """准备训练数据集"""
    # 配置数据集生成器
    config = Config(
        n_prompts=args.n_prompts,
        n_images=args.n_images,
        output_dir=args.output_dir,
        llm_name=args.llm_name,
        sd_model_base=args.sd_model_base,
        sd_model_lora=args.sd_model_lora,
        inference_steps=args.inference_steps,
        llm_int4=not args.no_llm_int4,
        device=args.device,
    )
    
    # 生成数据集
    generator = DatasetGenerator(config)
    generator.run()
    
    # 统计生成的数据
    db_path = Path(args.output_dir) / 'dataset.db'
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 统计总数
            cursor.execute("SELECT COUNT(*) FROM samples")
            total = cursor.fetchone()[0]
            print(f"总共生成了 {total} 条样本")
            
            # 按尺寸和模式统计
            cursor.execute("""
                SELECT long_edge, mode, COUNT(*) 
                FROM samples 
                GROUP BY long_edge, mode 
                ORDER BY long_edge, mode
            """)
            
            print("\n按尺寸和模式统计:")
            for long_edge, mode, count in cursor.fetchall():
                print(f"  尺寸: {long_edge if long_edge > 0 else 'auto'}, 模式: {mode}, 数量: {count}")

def main():
    parser = argparse.ArgumentParser(description='准备ASCII艺术SFT训练数据集')
    parser.add_argument('--n-prompts', type=int, default=1000, help='提示词数量')
    parser.add_argument('--n-images', type=int, default=4, help='每个提示词生成的图像数量')
    parser.add_argument('--output-dir', type=str, default='data/ascii_art_dataset', help='输出目录')
    parser.add_argument('--llm-name', type=str, default='models/Qwen3-1.7B', help='LLM模型路径')
    parser.add_argument('--sd-model-base', type=str, default='models/dreamlike-diffusion-1.0', help='SD基础模型')
    parser.add_argument('--sd-model-lora', type=str, default='models/插画风格lora模型扁平插画_V2.0.safetensors', help='SD LoRA模型')
    parser.add_argument('--inference-steps', type=int, default=25, help='SD推理步数')
    parser.add_argument('--no-llm-int4', action='store_true', help='不使用LLM INT4量化')
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 准备数据集
    prepare_dataset(args)

if __name__ == '__main__':
    main()