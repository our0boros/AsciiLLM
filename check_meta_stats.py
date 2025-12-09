#!/usr/bin/env python3
"""
直接检查生成的元数据统计信息
"""
import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='检查元数据统计信息')
    parser.add_argument('--meta-file', type=str, default='data/ascii_art_dataset/meta.parquet', 
                        help='元数据文件路径')
    args = parser.parse_args()
    
    meta_file = Path(args.meta_file)
    
    if not meta_file.exists():
        print(f'错误: 元数据文件不存在: {meta_file}')
        return 1
    
    print(f'读取元数据文件: {meta_file}')
    df = pd.read_parquet(meta_file)
    
    # 统计信息
    total_samples = len(df)
    unique_prompts = df['prompt'].nunique()
    unique_images = df['orig_img'].nunique()
    
    print(f'\n--- 数据集统计信息 ---')
    print(f'总样本数: {total_samples}')
    print(f'唯一提示词数: {unique_prompts}')
    print(f'唯一原始图像数: {unique_images}')
    
    # 检查每个图像对应的样本数
    print(f'\n--- 图像样本数分布 ---')
    img_sample_counts = df['orig_img'].value_counts()
    for img_path, count in img_sample_counts.items():
        print(f'{img_path}: {count}个样本')
    
    return 0

if __name__ == '__main__':
    main()