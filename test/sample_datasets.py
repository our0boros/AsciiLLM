#!/usr/bin/env python3
"""
从dataset_pipeline.py生成的结果中统计并随机读取n个样本进行打印
"""
import os
import sys
import argparse
import logging
import random
from pathlib import Path
import pandas as pd

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='从ASCII字符画数据集中统计并随机读取样本')
    parser.add_argument('--data-dir', type=str, default='data/ascii_arts_dataset', 
                        help='数据集目录路径')
    parser.add_argument('--n', type=int, default=1, help='随机读取的样本数量')
    parser.add_argument('--show-ascii', action='store_true', default=True, 
                        help='是否显示完整的ASCII字符画内容')
    args = parser.parse_args()
    
    # 设置数据目录和元数据文件路径
    data_dir = Path(args.data_dir)
    meta_file = data_dir / 'meta.parquet'
    
    # 检查元数据文件是否存在
    if not meta_file.exists():
        logger.error(f'元数据文件不存在: {meta_file}')
        logger.error('请先运行dataset_pipeline.py生成数据集')
        return 1
    
    # 读取元数据
    logger.info(f'读取元数据文件: {meta_file}')
    try:
        df = pd.read_parquet(meta_file)
    except Exception as e:
        logger.error(f'读取元数据失败: {e}')
        return 1
    
    # 统计数据集信息
    total_samples = len(df)
    unique_prompts = df['prompt'].nunique()
    unique_images = df['orig_img'].nunique()
    
    logger.info(f'\n--- 数据集统计信息 ---')
    logger.info(f'总样本数: {total_samples}')
    logger.info(f'唯一提示词数: {unique_prompts}')
    logger.info(f'唯一原始图像数: {unique_images}')
    logger.info(f'字符画模式分布:')
    for mode, count in df['mode'].value_counts().items():
        logger.info(f'  - {mode}: {count} ({count/total_samples*100:.1f}%)')
    logger.info(f'颜色选项分布:')
    for color, count in df['color'].value_counts().items():
        color_str = {'rgb' : '彩色', 'gray' : '灰度', 'char' : '字符'} [color]
        logger.info(f'  - {color_str}: {count} ({count/total_samples*100:.1f}%)')
    logger.info(f'长边字符数分布:')
    for long_edge, count in df['long_edge'].value_counts().sort_index().items():
        logger.info(f'  - {long_edge}: {count} ({count/total_samples*100:.1f}%)')
    
    # 随机选择n个样本
    n = min(args.n, total_samples)
    logger.info(f'\n--- 随机选择 {n} 个样本 ---')
    
    # 使用pandas的sample方法进行随机采样
    sampled_df = df.sample(n=n, random_state=None)  # 使用None表示每次运行都不同
    
    # 打印每个样本的信息
    for idx, row in sampled_df.iterrows():
        logger.info(f'\n[{idx+1}/{n}] 样本信息:')
        logger.info(f'提示词: {row["prompt"]}')
        logger.info(f'原始图像: {row["orig_img"]}')
        logger.info(f'字符画尺寸: 宽={row["w_char"]} 高={row["h_char"]}')
        logger.info(f'设置: 颜色={"彩色" if row["color"] else "黑白"} 模式={row["mode"]} 长边字符数={row["long_edge"]}')
        logger.info(f'生成时间: {row["timestamp"]}')
        
        if args.show_ascii:
            logger.info(f'\nASCII字符画:')
            logger.info('\n' + row["ascii_text"])
            logger.info('-' * 50)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
