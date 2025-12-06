#!/usr/bin/env python3
# pipeline.py
import os, json, math, itertools, argparse, logging, tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from peft import PeftModel

import pandas as pd
import torch, tqdm
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入本地converter模块
import sys
import os

# 添加当前目录到Python路径，以便在直接运行脚本时也能找到模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from converter import img_to_ascii as converter_img_to_ascii

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 0. 参数配置 ==========
def parse_args():
    parser = argparse.ArgumentParser(description='批量生成ASCII字符画数据集')
    parser.add_argument('--n-prompts', type=int, default=20, help='生成的提示词总数')
    parser.add_argument('--n-images', type=int, default=10, help='每个提示词生成的图像数量')
    parser.add_argument('--output-dir', type=str, default='data/ascii_art_dataset', help='输出目录')
    parser.add_argument('--device', type=str, default=None, help='使用的设备(cuda/cpu)')
    parser.add_argument('--llm-name', type=str, default='models/Qwen3-1.7B', help='LLM模型名称或路径')
    parser.add_argument('--sd-model-base', type=str, default='models/stable-diffusion-v1-5', help='Stable Diffusion模型名称或路径')
    parser.add_argument('--sd-model-lora', type=str, default=None, help='Stable Diffusion LoRA模型名称或路径') # models/illustration_flat/插画风格lora模型扁平插画_V2.0.safetensors
    parser.add_argument('--inference-steps', type=int, default=25, help='扩散模型推理步数')
    return parser.parse_args()

# ========== 1. 加载模型 ==========
def load_models(args):
    """加载LLM和Stable Diffusion模型"""
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 1.1 LLM for prompt
    logger.info(f'加载LLM模型: {args.llm_name}')
    tok = AutoTokenizer.from_pretrained(args.llm_name)
    llm = AutoModelForCausalLM.from_pretrained(
            args.llm_name, torch_dtype=torch.float16 if device=='cuda' else torch.float32
    ).to(device)
    
    # 1.2 SD for image
    logger.info(f'加载Stable Diffusion基础模型: {args.sd_model_base}')
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model_base,
        torch_dtype=torch.float16 if device=='cuda' else torch.float32
    ).to(device)
    
    # 1.3 加载LoRA模型（如果提供）
    if args.sd_model_lora and args.sd_model_lora != 'None':
        lora_path = Path(args.sd_model_lora)
        logger.info(f'加载Stable Diffusion LoRA模型: {lora_path}')
        
        # 检查是否为单个.safetensors文件
        if lora_path.is_file() and lora_path.suffix == '.safetensors':
            # 直接加载单个.safetensors文件
            logger.info(f'直接加载.safetensors文件: {lora_path}')
            sd_pipe.load_lora_weights(
                lora_path.parent,  # 模型目录
                weight_name=lora_path.name,  # 权重文件名
                torch_dtype=torch.float16 if device=='cuda' else torch.float32
            )
        else:
            # 尝试使用PEFT方式加载（需要adapter_config.json）
            try:
                logger.info(f'尝试使用PEFT方式加载LoRA模型')
                sd_pipe.unet = PeftModel.from_pretrained(
                    sd_pipe.unet,
                    args.sd_model_lora,
                    torch_dtype=torch.float16 if device=='cuda' else torch.float32
                )
            except Exception as e:
                logger.warning(f'PEFT加载失败，尝试作为普通LoRA加载: {e}')
                sd_pipe.load_lora_weights(
                    args.sd_model_lora,
                    torch_dtype=torch.float16 if device=='cuda' else torch.float32
                )
    
    return device, tok, llm, sd_pipe

# ========== 2. 工具函数 ==========
def load_seed_concepts():
    """从文件加载种子概念列表"""
    concepts_file = Path(__file__).parent / 'seed_concepts.txt'
    if not concepts_file.exists():
        logger.error(f'种子概念文件不存在: {concepts_file}')
        # 使用默认概念作为备用
        return [
            'cyberpunk city', 'sunset beach', 'mechanical cat', 'forest spirit', 'desert pyramid'
        ]
    
    concepts = []
    with concepts_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # 跳过空行和注释
                concepts.append(line)
    
    logger.info(f'成功加载 {len(concepts)} 个种子概念')
    return concepts

# ========== 3. LLM 批量生成提示词 ==========
def generate_prompts(tok, llm, n_prompts=20, batch_size=5, device='cuda'):
    """使用LLM批量生成图像描述提示词"""
    # 从文件加载种子概念
    seed_concepts = load_seed_concepts()
    
    system = ("You are a visual artist. Write a 10-30 word English image description for the given concept, "
              "including subject, environment, lighting, tone, material, mood. Do not repeat words. "
              "Output only the description.")
    
    prompts = []
    logger.info(f'开始生成 {n_prompts} 个提示词...')
    
    for i in tqdm.tqdm(range(0, n_prompts, batch_size), desc='生成提示词'):
        batch_size_current = min(batch_size, n_prompts - i)
        concepts = np.random.choice(seed_concepts, batch_size_current)
        messages = [system + '\nConcept: ' + c for c in concepts]
        
        try:
            inputs = tok(messages, return_tensors='pt', padding=True).to(device)
            with torch.no_grad():
                out = llm.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.9)
            texts = tok.batch_decode(out, skip_special_tokens=True)
            
            for t in texts:
                # 提取描述部分
                prompt = t.split('Output only the description.')[-1].strip()
                prompts.append(prompt)
                logger.debug(f'生成提示词: {prompt}')
                
        except Exception as e:
            logger.error(f'生成提示词时出错: {e}')
            # 生成备用提示词
            for c in concepts:
                prompts.append(f'A beautiful image of {c}, highly detailed, cinematic lighting')
    
    return prompts[:n_prompts]

# ========== 4. 图像→字符画 ==========
def generate_ascii_from_converter(img: Image.Image, long_edge: int, color: bool, mode: str) -> str:
    """使用外部converter工具生成ASCII字符画"""
    # 创建临时文件保存图像
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        img.save(tmp_path)
        # 调用converter模块生成ASCII字符画
        ascii_art = converter_img_to_ascii(
            tmp_path,
            width_chars=long_edge,
            color=color,
            mode=mode
        )
        return ascii_art
    finally:
        # 删除临时文件
        if tmp_path.exists():
            tmp_path.unlink()

# ========== 5. 主循环 ==========
def main():
    # 解析参数
    args = parse_args()
    
    # 设置输出目录
    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / 'orig').mkdir(exist_ok=True)
    
    # 定义字符画参数
    LONG_EDGE_CHARS = [8, 16, 32, 64, 128]
    COLOR_OPTIONS = [False, ]  # 黑白/彩色
    MODES = ['normal', 'complex', 'braille']  # 三种模式
    
    # 检查是否可以增量保存
    meta_file = OUT_DIR / 'meta.parquet'
    existing_meta = None
    if meta_file.exists():
        try:
            existing_meta = pd.read_parquet(meta_file)
            logger.info(f'发现现有元数据，将进行增量保存。现有记录数: {len(existing_meta)}')
        except Exception as e:
            logger.error(f'读取现有元数据失败: {e}')
            existing_meta = None
    
    # 加载模型
    device, tok, llm, sd_pipe = load_models(args)
    
    # 生成提示词
    prompts = generate_prompts(tok, llm, n_prompts=args.n_prompts, device=device)
    logger.info(f'成功生成 {len(prompts)} 个提示词')
    
    # 开始生成数据
    meta_list = []
    
    # 确定起始索引
    if existing_meta is not None and not existing_meta.empty:
        # 获取现有最大索引
        max_idx = max(int(path.stem.split('_')[0]) for path in existing_meta['orig_img'].map(lambda x: Path(x)))
        count_img = max_idx + 1
        logger.info(f'从索引 {count_img} 开始增量生成')
    else:
        count_img = 0
    
    for prompt in tqdm.tqdm(prompts, desc='处理提示词'):
        logger.info(f'处理提示词: {prompt[:50]}...')
        
        # 生成图像
        try:
            images = sd_pipe([prompt] * args.n_images, num_inference_steps=args.inference_steps).images
        except Exception as e:
            logger.error(f'生成图像时出错: {e}')
            continue
        
        for img_idx, img in enumerate(images):
            # 保存原图
            orig_path = OUT_DIR / 'orig' / f'{count_img:06d}.png'
            img.save(orig_path)
            logger.debug(f'保存原图: {orig_path}')

            # 遍历所有字符画参数组合（颜色/黑白 * 3种模式 = 6种组合）
            for color in COLOR_OPTIONS:
                for mode in MODES:
                    for long in LONG_EDGE_CHARS:
                        try:
                            # 使用converter.py生成ASCII字符画
                            ascii_art = generate_ascii_from_converter(img, long, color, mode)
                            
                            # 计算字符宽高
                            lines = ascii_art.splitlines()
                            h_char = len(lines)
                            w_char = max(len(line) for line in lines)
                            
                            # 添加到元数据列表（直接存储ASCII文本，不保存到单独文件）
                            meta_list.append({
                                'prompt': prompt,
                                'orig_img': str(orig_path.relative_to(OUT_DIR)),
                                'w_char': w_char,
                                'h_char': h_char,
                                'ascii_text': ascii_art,  # 直接存储ASCII文本内容
                                'color': color,
                                'mode': mode,
                                'long_edge': long,
                                'timestamp': pd.Timestamp.now()
                            })
                            
                        except Exception as e:
                            logger.error(f'生成ASCII字符画失败 (color={color}, mode={mode}, long={long}): {e}')
            
            count_img += 1
    
    # 保存元数据（增量保存）
    if meta_list:
        new_meta = pd.DataFrame(meta_list)
        
        if existing_meta is not None and not existing_meta.empty:
            # 合并现有元数据和新元数据
            combined_meta = pd.concat([existing_meta, new_meta], ignore_index=True)
        else:
            combined_meta = new_meta
        
        # 保存为parquet格式
        combined_meta.to_parquet(meta_file, index=False)
        logger.info(f'元数据已保存: {meta_file}')
        logger.info(f'新增 {len(new_meta)} 条记录，总记录数: {len(combined_meta)}')

    logger.info(f'数据生成完成！共生成 {len(meta_list)} 个ASCII字符画')
    logger.info(f'元数据保存在: {meta_file}')
    
    # 随机采样一个样本显示
    if meta_list:
        import random
        random_sample = random.choice(meta_list)
        logger.info(f'\n--- 随机采样样本 ---')
        logger.info(f'提示词: {random_sample["prompt"]}')
        logger.info(f'图像: {random_sample["orig_img"]}')
        logger.info(f'字符画设置: 宽={random_sample["w_char"]} 高={random_sample["h_char"]} 颜色={random_sample["color"]} 模式={random_sample["mode"]}')
        logger.info(f'\nASCII字符画示例:')
        logger.info(random_sample["ascii_text"])
        logger.info('--- 样本显示结束 ---')

if __name__ == '__main__':
    main()