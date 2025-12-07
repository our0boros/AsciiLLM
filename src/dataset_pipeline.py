#!/usr/bin/env python3
# pipeline.py
import os, json, math, itertools, argparse, logging, tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from peft import PeftModel

import pandas as pd
import torch, tqdm
from diffusers import StableDiffusionPipeline, DiffusionPipeline, QuantoConfig, PipelineQuantizationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 启用CUDA内存自动分配
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # 启用CuDNN基准测试以加速重复操作

# 导入本地converter模块
import sys
import os

# 添加当前目录到Python路径，以便在直接运行脚本时也能找到模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from converter import generate_ascii_from_pil as converter_generate_ascii

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 0. 参数配置 ==========
def parse_args():
    parser = argparse.ArgumentParser(description='批量生成ASCII字符画数据集')
    parser.add_argument('--n-prompts', type=int, default=200_000, help='生成的提示词总数')
    parser.add_argument('--n-images', type=int, default=4, help='每个提示词生成的图像数量')
    parser.add_argument('--output-dir', type=str, default='data/ascii_art_dataset', help='输出目录')
    parser.add_argument('--device', type=str, default=None, help='使用的设备(cuda/cpu)')
    parser.add_argument('--llm-name', type=str, default='models/Qwen3-1.7B', help='LLM模型名称或路径')
    parser.add_argument('--sd-model-base', type=str, default=r"models/dreamlike-diffusion-1.0", help='Stable Diffusion模型名称或路径')
    parser.add_argument('--sd-model-lora', type=str, default="models/插画风格lora模型扁平插画_V2.0.safetensors", help='Stable Diffusion LoRA模型名称或路径')
    parser.add_argument('--inference-steps', type=int, default=25, help='扩散模型推理步数')
    parser.add_argument('--llm-int4', action='store_true', default=True, help='是否对LLM模型启用int4量化')
    parser.add_argument('--sd-int4', action='store_true', default=True, help='是否对Stable Diffusion模型启用int4量化')
    return parser.parse_args()

# ========== 1. 加载模型 ==========
def load_models(args):
    """加载LLM和Stable Diffusion模型"""
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 1.1 LLM for prompt（原逻辑保留，LLM的BitsAndBytesConfig是生效的）
    logger.info(f'加载LLM模型: {args.llm_name}')
    tok = AutoTokenizer.from_pretrained(args.llm_name, padding_side='left')
    
    # LLM模型加载配置
    llm_kwargs = {
        "torch_dtype": torch.float16 if device=='cuda' else torch.float32,
    }
    
    # 如果启用int4量化
    if args.llm_int4:
        logger.info("启用LLM模型int4量化")
        try:
            # LLM使用Transformers的BitsAndBytesConfig是正确的
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            llm_kwargs["quantization_config"] = bnb_config
        except ImportError:
            logger.error("无法导入BitsAndBytesConfig，请安装bitsandbytes库: pip install bitsandbytes")
            logger.info("将继续使用未量化的模型")
    
    llm = AutoModelForCausalLM.from_pretrained(
            args.llm_name, **llm_kwargs
    )
    
    # 量化模型已经自动移动到GPU，无需再次调用.to(device)
    if not args.llm_int4:
        llm = llm.to(device)
    
    # 启用LLM模型编译
    if device == 'cuda' and hasattr(torch, 'compile'):
        try:
            logger.info("启用LLM模型编译优化")
            llm = torch.compile(llm, mode="reduce-overhead")
        except Exception as e:
            logger.warning(f"无法启用LLM模型编译: {e}")
    
    # 1.2 SD for image（修复int4量化逻辑）
    logger.info(f'加载Stable Diffusion基础模型: {args.sd_model_base}')
    
    # SD模型基础配置
    sd_kwargs = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "use_safetensors": True,
        "device_map": "cuda" if device == "cuda" else "balanced",  # 自动分配设备
    }

    # ========== 修复SD量化核心代码 ==========
    if args.sd_int4:
        logger.info("启用Stable Diffusion模型int4量化（Diffusers官方方案）")
        pipeline_quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["transformer", "text_encoder_2"],
        )
        sd_kwargs["quantization_config"] = pipeline_quant_config
        sd_pipe = StableDiffusionPipeline.from_pretrained(args.sd_model_base, **sd_kwargs)
    else:
        # 不量化时的基础加载
        sd_pipe = StableDiffusionPipeline.from_pretrained(args.sd_model_base, **sd_kwargs)
    # ========== SD量化修复结束 ==========
    
    # 启用xformers加速（原逻辑保留）
    try:
        import xformers
        sd_pipe.enable_xformers_memory_efficient_attention()
        logger.info("成功启用xformers优化")
    except ImportError:
        logger.warning("xformers未安装，无法使用内存高效注意力")

    # 1.3 加载LoRA模型（原逻辑保留，注意量化后LoRA加载兼容性）
    if args.sd_model_lora and args.sd_model_lora != 'None':
        lora_path = Path(args.sd_model_lora)
        logger.info(f'加载Stable Diffusion LoRA模型: {lora_path}')
        
        if lora_path.is_file() and lora_path.suffix == '.safetensors':
            sd_pipe.load_lora_weights(lora_path, adapter_name="illustration")
            sd_pipe.set_adapters(["illustration"], adapter_weights=[0.7])
            logger.info("LoRA模型加载成功")
        else:
            raise ValueError(f'不支持的LoRA模型格式: {lora_path}')
    
    return device, tok, llm, sd_pipe

# ========== 2. 工具函数 ==========
def load_seed_concepts():
    """从文件加载种子概念列表"""
    concepts_file = Path(__file__).parent / 'seed_concepts.txt'
    if not concepts_file.exists():
        logger.error(f'种子概念文件不存在: {concepts_file}')
        # 使用默认概念作为备用
        raise FileNotFoundError(f'种子概念文件不存在: {concepts_file}')
    
    concepts = []
    with concepts_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # 跳过空行和注释
                concepts.append(line)
    
    logger.info(f'成功加载 {len(concepts)} 个种子概念')
    return concepts

# ========== 3. LLM 生成提示词（支持逐个生成） ==========
def generate_prompts(tok, llm, n_prompts=20, device='cuda', yield_one_by_one=False):
    """使用LLM生成图像描述提示词
    
    Args:
        tok: 分词器
        llm: 语言模型
        n_prompts: 需要生成的提示词数量
        device: 运行设备
        yield_one_by_one: 是否逐个生成并返回提示词（生成一个返回一个）
        
    Returns:
        如果yield_one_by_one为True，则逐个yield提示词
        否则返回包含所有提示词的列表
    """
    # 从文件加载种子概念
    seed_concepts = load_seed_concepts()
    
    system = ("You are a visual artist. Write a 10-30 word English image description for the given concept, "
              "including subject, environment, lighting, tone, material, mood. Do not repeat words. "
              "Output only the description.")
    
    prompts = []
    logger.info(f'开始生成 {n_prompts} 个提示词...')
    
    for i in tqdm.tqdm(range(n_prompts), desc='生成提示词'):
        # 选择一个种子概念
        concept = np.random.choice(seed_concepts)
        message = system + '\nConcept: ' + concept
        
        try:
            inputs = tok([message], return_tensors='pt', padding=True).to(device)
            with torch.no_grad():
                # 优化生成参数
                out = llm.generate(
                    **inputs, 
                    max_new_tokens=60, 
                    do_sample=True, 
                    temperature=0.9,
                    top_k=50,  # 限制采样范围
                    top_p=0.95,  # 核采样
                    num_return_sequences=1,
                    use_cache=True  # 使用KV缓存加速
                )
            text = tok.batch_decode(out, skip_special_tokens=True)[0]
            
            # 提取描述部分
            prompt = text.split('Output only the description.')[-1].strip()
            
            if yield_one_by_one:
                logger.debug(f'生成提示词 {i+1}/{n_prompts}: {prompt}')
                yield prompt
            else:
                prompts.append(prompt)
                logger.debug(f'生成提示词 {i+1}/{n_prompts}: {prompt}')
                
        except Exception as e:
            logger.error(f'生成提示词时出错: {e}')
            # 生成备用提示词
            prompt = f'A beautiful image of {concept}, highly detailed, cinematic lighting'
            if yield_one_by_one:
                logger.debug(f'使用备用提示词 {i+1}/{n_prompts}: {prompt}')
                yield prompt
            else:
                prompts.append(prompt)
                logger.debug(f'使用备用提示词 {i+1}/{n_prompts}: {prompt}')
    
    if not yield_one_by_one:
        return prompts[:n_prompts]

# ========== 4. 图像→字符画 ==========
def generate_ascii_from_converter(img: Image.Image, long_edge: int, color: bool, mode: str) -> str:
    """使用外部converter工具生成ASCII字符画"""
    # 直接使用converter模块的函数处理PIL图像，避免创建临时文件
    return converter_generate_ascii(
        img,
        width_chars=long_edge,
        color=color,
        mode=mode
    )

# ========== 5. 主循环 ==========
def main():
    # 解析参数
    args = parse_args()
    
    # 设置输出目录
    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / 'orig').mkdir(exist_ok=True)
    
    # 定义字符画参数
    LONG_EDGE_CHARS = [-1, 16, 32, 64, 96]
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
    
    # 开始生成数据
    meta_list = []
    
    # 生成一个提示词就立即处理一个图像
    prompts_generator = generate_prompts(tok, llm, n_prompts=args.n_prompts, device=device, yield_one_by_one=True)
    
    # 确定起始索引
    if existing_meta is not None and not existing_meta.empty:
        # 获取现有最大索引
        max_idx = max(int(path.stem.split('_')[0]) for path in existing_meta['orig_img'].map(lambda x: Path(x)))
        count_img = max_idx + 1
        logger.info(f'从索引 {count_img} 开始增量生成')
    else:
        count_img = 0
    
    # 生成一个提示词处理一个图像
    for prompt in tqdm.tqdm(prompts_generator, total=args.n_prompts, desc='处理提示词'):
        logger.info(f'处理提示词: {prompt[:50]}...')
        
        # 逐个生成图像，而不是批量生成
        for img_idx in range(args.n_images):
            logger.info(f'生成图像 {count_img:06d}/{len(prompt)*args.n_images} (提示词 {prompt.index(prompt)+1}/{len(prompt)}, 图像 {img_idx+1}/{args.n_images})')
            
            # 生成单个图像
            try:
                img = sd_pipe("illustration style, " + prompt, num_inference_steps=args.inference_steps).images[0]
            except Exception as e:
                logger.error(f'生成图像时出错: {e}')
                continue
            
            # 计算总组合数和当前图像的总进度
            total_combinations = len(COLOR_OPTIONS) * len(MODES) * len(LONG_EDGE_CHARS)
            combination_count = 0
            
            # 保存原图（使用JPEG格式以提高速度）
            orig_path = OUT_DIR / 'orig' / f'{count_img:06d}.jpg'
            img.save(orig_path, format='JPEG', quality=90)
            logger.debug(f'保存原图: {orig_path}')
            
            # 遍历所有字符画参数组合（颜色/黑白 * 3种模式 * 5种字符长度 = 15种组合）
            for color in COLOR_OPTIONS:
                for mode in MODES:
                    for long in LONG_EDGE_CHARS:
                        combination_count += 1
                        logger.info(f'处理图像 {count_img:06d}/{len(prompt)*args.n_images}, 组合 {combination_count}/{total_combinations} (颜色: {"彩色" if color else "黑白"}, 模式: {mode}, 长边字符数: {long})')
                        try:
                            # 使用converter.py生成ASCII字符画
                            ascii_art = generate_ascii_from_converter(img, long, color, mode)
                            if len(ascii_art) < long * long * 0.5:
                                logger.error(f'生成的ASCII字符画为空 (color={color}, mode={mode}, long={long})')
                                logger.error(f'生成的ASCII字符画: {ascii_art}')
                                continue
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
            
            # 优化：定期保存元数据，避免内存占用过大
            if count_img % 50 == 0 and meta_list:
                new_meta = pd.DataFrame(meta_list)
                if existing_meta is not None and not existing_meta.empty:
                    combined_meta = pd.concat([existing_meta, new_meta], ignore_index=True)
                else:
                    combined_meta = new_meta
                combined_meta.to_parquet(meta_file, index=False)
                logger.info(f'增量保存元数据，当前共 {len(combined_meta)} 条记录')
                # 清空已保存的元数据列表
                meta_list.clear()
            
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