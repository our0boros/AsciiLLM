#!/usr/bin/env python3
# pipeline_v2.py
"""
ASCII Art Dataset Generator - 精简版
特性：断点续传、实时保存、零RAM占用
"""

import os
import json
import sqlite3
import argparse
import logging
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Iterator, Optional

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from diffusers import StableDiffusionPipeline

from ascii_art_converter.generator import AsciiArtGenerator, AsciiArtConfig
from ascii_art_converter.constants import RenderMode

# ==================== 配置 ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODES = ['normal', 'complex', 'braille']
LONG_EDGES = [-1, 16, 32, 64, 96]


@dataclass
class Config:
    """全局配置"""
    n_prompts: int = 8000
    n_images: int = 4
    output_dir: str = 'data/ascii_art_dataset'
    llm_name: str = 'models/Qwen3-1.7B'
    sd_model_base: str = 'models/dreamlike-diffusion-1.0'
    sd_model_lora: str = 'models/插画风格lora模型扁平插画_V2.0.safetensors'
    inference_steps: int = 25
    llm_int4: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================== 数据库管理 ====================

class Database:
    """SQLite数据库管理器 - 实时保存，零RAM占用"""
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt TEXT NOT NULL,
        orig_img TEXT NOT NULL,
        w_char INTEGER,
        h_char INTEGER,
        ascii_text TEXT,
        colors TEXT,
        color TEXT,
        mode TEXT,
        long_edge INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS progress (
        key TEXT PRIMARY KEY,
        value INTEGER
    );
    
    CREATE INDEX IF NOT EXISTS idx_orig_img ON samples(orig_img);
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(self.SCHEMA)
    
    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def get_progress(self, key: str, default: int = 0) -> int:
        """获取进度"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM progress WHERE key = ?", (key,)
            ).fetchone()
            return row['value'] if row else default
    
    def set_progress(self, key: str, value: int):
        """设置进度"""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO progress (key, value) VALUES (?, ?)",
                (key, value)
            )
    
    def insert_sample(self, sample: dict):
        """插入单条样本 - 立即写入磁盘"""
        # 序列化 colors（如果是列表/数组）
        if 'colors' in sample and sample['colors'] is not None:
            sample = sample.copy()
            sample['colors'] = json.dumps(sample['colors'])
        
        with self._connect() as conn:
            columns = ', '.join(sample.keys())
            placeholders = ', '.join(['?'] * len(sample))
            conn.execute(
                f"INSERT INTO samples ({columns}) VALUES ({placeholders})",
                list(sample.values())
            )
    
    def image_exists(self, orig_img: str) -> bool:
        """检查图像是否已处理"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM samples WHERE orig_img = ? LIMIT 1", (orig_img,)
            ).fetchone()
            return row is not None
    
    def count_samples(self) -> int:
        """统计样本数"""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    
    def export_to_parquet(self, output_path: Path):
        """导出为 parquet 格式"""
        import pandas as pd
        
        with self._connect() as conn:
            df = pd.read_sql_query("SELECT * FROM samples", conn)
        
        # 反序列化 colors
        if 'colors' in df.columns:
            df['colors'] = df['colors'].apply(
                lambda x: json.loads(x) if x else None
            )
        
        df.to_parquet(output_path, index=False)
        logger.info(f"导出 {len(df)} 条记录到 {output_path}")


# ==================== 模型管理 ====================

class ModelManager:
    """模型加载与管理"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self._llm = None
        self._tokenizer = None
        self._sd_pipe = None
        self._ascii_generator = AsciiArtGenerator()
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_llm()
        return self._tokenizer
    
    @property
    def llm(self):
        if self._llm is None:
            self._load_llm()
        return self._llm
    
    @property
    def sd_pipe(self):
        if self._sd_pipe is None:
            self._load_sd()
        return self._sd_pipe
    
    def _load_llm(self):
        """延迟加载 LLM"""
        logger.info(f"加载 LLM: {self.config.llm_name}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_name, padding_side='left'
        )
        
        kwargs = {"torch_dtype": torch.float16}
        
        if self.config.llm_int4:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        self._llm = AutoModelForCausalLM.from_pretrained(
            self.config.llm_name, **kwargs
        )
        
        if not self.config.llm_int4:
            self._llm = self._llm.to(self.device)
    
    def _load_sd(self):
        """延迟加载 Stable Diffusion"""
        logger.info(f"加载 SD: {self.config.sd_model_base}")
        
        self._sd_pipe = StableDiffusionPipeline.from_pretrained(
            self.config.sd_model_base,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(self.device)
        
        # 加载 LoRA
        if self.config.sd_model_lora:
            lora_path = Path(self.config.sd_model_lora)
            if lora_path.exists():
                self._sd_pipe.load_lora_weights(lora_path, adapter_name="style")
                self._sd_pipe.set_adapters(["style"], adapter_weights=[0.7])
                logger.info("LoRA 加载成功")
        
        # 启用优化
        try:
            self._sd_pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    
    def generate_prompt(self, concept: str) -> str:
        """生成单个提示词"""
        system = (
            "You are a visual artist. Write a 10-30 word English image description "
            "for the given concept, including subject, environment, lighting, tone, "
            "material, mood. Do not repeat words. Output only the description."
        )
        message = f"{system}\nConcept: {concept}"
        
        inputs = self.tokenizer([message], return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            out = self.llm.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                use_cache=True
            )
        
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text.split('Output only the description.')[-1].strip()
    
    def generate_image(self, prompt: str) -> Image.Image:
        """生成单个图像"""
        result = self.sd_pipe(
            f"illustration style, {prompt}",
            num_inference_steps=self.config.inference_steps
        )
        return result.images[0]
    
    def generate_ascii(
        self, image: Image.Image, long_edge: int, mode: str
    ) -> tuple[str, list]:
        """生成 ASCII 字符画"""
        render_mode = RenderMode.BRAILLE if mode == 'braille' else RenderMode.DENSITY
        
        config = AsciiArtConfig(
            width=long_edge if long_edge > 0 else None,
            mode=render_mode,
            colorize=True,
        )
        
        result = self._ascii_generator.convert(image, config)
        return result.text, result.colors


# ==================== 主生成器 ====================

class DatasetGenerator:
    """数据集生成器 - 支持断点续传"""
    
    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'orig').mkdir(exist_ok=True)
        
        self.db = Database(self.output_dir / 'dataset.db')
        self.models = ModelManager(config)
        self.concepts = self._load_concepts()
    
    def _load_concepts(self) -> list[str]:
        """加载种子概念"""
        concepts_file = Path(__file__).parent / 'seed_concepts.txt'
        if not concepts_file.exists():
            raise FileNotFoundError(f"找不到: {concepts_file}")
        
        concepts = []
        with open(concepts_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    concepts.append(line)
        
        logger.info(f"加载 {len(concepts)} 个种子概念")
        return concepts
    
    def _get_image_path(self, idx: int) -> Path:
        return self.output_dir / 'orig' / f'{idx:06d}.jpg'
    
    def run(self):
        """主运行循环"""
        # 获取断点
        start_prompt_idx = self.db.get_progress('prompt_idx', 0)
        start_image_idx = self.db.get_progress('image_idx', 0)
        
        logger.info(f"断点续传: prompt_idx={start_prompt_idx}, image_idx={start_image_idx}")
        logger.info(f"已有样本: {self.db.count_samples()}")
        
        total_images = self.config.n_prompts * self.config.n_images
        global_img_idx = start_prompt_idx * self.config.n_images + start_image_idx
        
        pbar = tqdm(
            total=total_images,
            initial=global_img_idx,
            desc="生成数据集"
        )
        
        try:
            for prompt_idx in range(start_prompt_idx, self.config.n_prompts):
                # 生成提示词
                concept = np.random.choice(self.concepts)
                try:
                    prompt = self.models.generate_prompt(concept)
                except Exception as e:
                    logger.warning(f"提示词生成失败: {e}")
                    prompt = f"A beautiful image of {concept}, highly detailed"
                
                # 确定图像起始索引
                img_start = start_image_idx if prompt_idx == start_prompt_idx else 0
                
                for img_idx in range(img_start, self.config.n_images):
                    global_img_idx = prompt_idx * self.config.n_images + img_idx
                    img_path = self._get_image_path(global_img_idx)
                    rel_path = f'orig/{global_img_idx:06d}.jpg'
                    
                    # 跳过已存在的
                    if img_path.exists() and self.db.image_exists(rel_path):
                        pbar.update(1)
                        continue
                    
                    # 生成图像
                    try:
                        image = self.models.generate_image(prompt)
                        image.save(img_path, format='JPEG', quality=90)
                    except Exception as e:
                        logger.error(f"图像生成失败 [{global_img_idx}]: {e}")
                        pbar.update(1)
                        continue
                    
                    # 生成所有 ASCII 变体
                    for mode in MODES:
                        for long_edge in LONG_EDGES:
                            try:
                                ascii_text, colors = self.models.generate_ascii(
                                    image, long_edge, mode
                                )
                                
                                if len(ascii_text) < 50:
                                    continue
                                
                                lines = ascii_text.splitlines()
                                
                                # 实时保存到数据库
                                self.db.insert_sample({
                                    'prompt': prompt,
                                    'orig_img': rel_path,
                                    'w_char': max(len(l) for l in lines),
                                    'h_char': len(lines),
                                    'ascii_text': ascii_text,
                                    'colors': colors,
                                    'color': 'rgb',
                                    'mode': mode,
                                    'long_edge': long_edge,
                                })
                                
                            except Exception as e:
                                logger.debug(f"ASCII生成失败 [{mode}/{long_edge}]: {e}")
                    
                    # 更新进度
                    self.db.set_progress('prompt_idx', prompt_idx)
                    self.db.set_progress('image_idx', img_idx + 1)
                    pbar.update(1)
                
                # 重置图像索引
                start_image_idx = 0
                self.db.set_progress('image_idx', 0)
        
        except KeyboardInterrupt:
            logger.info("用户中断，进度已保存")
        
        finally:
            pbar.close()
            self._finalize()
    
    def _finalize(self):
        """完成处理"""
        total = self.db.count_samples()
        logger.info(f"生成完成！共 {total} 条样本")
        
        # 可选：导出为 parquet
        parquet_path = self.output_dir / 'meta.parquet'
        self.db.export_to_parquet(parquet_path)


# ==================== 入口 ====================

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description='ASCII Art 数据集生成器')
    parser.add_argument('--n-prompts', type=int, default=8000)
    parser.add_argument('--n-images', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='data/ascii_art_dataset')
    parser.add_argument('--llm-name', type=str, default='models/Qwen3-1.7B')
    parser.add_argument('--sd-model-base', type=str, default='models/dreamlike-diffusion-1.0')
    parser.add_argument('--sd-model-lora', type=str, default='models/插画风格lora模型扁平插画_V2.0.safetensors')
    parser.add_argument('--inference-steps', type=int, default=25)
    parser.add_argument('--no-llm-int4', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    return Config(
        n_prompts=args.n_prompts,
        n_images=args.n_images,
        output_dir=args.output_dir,
        llm_name=args.llm_name,
        sd_model_base=args.sd_model_base,
        sd_model_lora=args.sd_model_lora,
        inference_steps=args.inference_steps,
        llm_int4=not args.no_llm_int4,
        device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu'),
    )


def main():
    config = parse_args()
    generator = DatasetGenerator(config)
    generator.run()


if __name__ == '__main__':
    main()
