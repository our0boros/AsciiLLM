#!/usr/bin/env python3
"""
渐进式SFT训练脚本 - 训练Qwen3模型生成ASCII艺术
按照16->32->64->96->-1的顺序渐进式训练
"""

import os
import json
import sqlite3
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
import peft
from peft import LoraConfig, get_peft_model, TaskType

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 渐进式训练顺序
PROGRESSIVE_SIZES = [16, 32, 64, 96, -1]
MODES = ['normal', 'braille']

@dataclass
class SFTConfig:
    """SFT训练配置"""
    base_model: str = 'models/Qwen3-1.7B'
    dataset_path: str = 'data/ascii_art_dataset/dataset.db'
    output_dir: str = 'models/ascii_art_qwen3'
    use_int4: bool = True
    max_seq_length: int = 4096
    
    # 训练参数
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    
    # LoRA参数
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # 其他
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class DatasetLoader:
    """数据集加载器"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def load_data(self, long_edge: int, mode: str) -> List[Dict]:
        """加载指定尺寸和模式的数据"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT prompt, ascii_text, w_char, h_char, colors, mode, long_edge
                FROM samples 
                WHERE long_edge = ? AND mode = ? AND length(ascii_text) > 50
                ORDER BY RANDOM()
                """,
                (long_edge, mode)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def format_for_sft(self, data: List[Dict]) -> List[Dict]:
        """格式化数据为SFT格式"""
        formatted = []
        for item in data:
            # 构建系统提示
            system_prompt = (
                "你是一个专业的ASCII艺术家。根据用户的描述，生成对应的ASCII艺术作品。"
                f"请使用<{item['mode']}>模式生成ASCII艺术，"
                f"宽度约为{item['long_edge'] if item['long_edge'] > 0 else '自适应'}字符。"
            )
            
            # 构建用户提示
            user_prompt = f"请为以下描述生成ASCII艺术：<ascii-art>{item['prompt']}</ascii-art>"
            
            # 构建助手回答
            assistant_response = f"<ascii-art>\n{item['ascii_text']}\n</ascii-art>"
            
            # 构建对话格式
            conversation = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ]
            }
            formatted.append(conversation)
        
        return formatted


class ProgressiveSFTTrainer:
    """渐进式SFT训练器"""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载tokenizer
        logger.info(f"加载tokenizer: {config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            padding_side='right',
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 设置默认的LoRA目标模块
        if config.lora_target_modules is None:
            config.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
    
    def load_model(self, stage_dir: Optional[str] = None) -> peft.PeftModel:
        """加载模型，支持从上一个阶段继续训练"""
        model_path = stage_dir if stage_dir else self.config.base_model
        
        logger.info(f"加载模型: {model_path}")
        
        # 配置量化
        kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
        if self.config.use_int4:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **kwargs
        )
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none"
        )
        
        # 应用LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def tokenize_function(self, examples):
        """数据tokenization函数"""
        # 将对话格式转换为文本
        texts = []
        for messages in examples["messages"]:
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            texts.append(text)
        
        # Tokenization
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=False,
            return_tensors=None
        )
        
        # 设置labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def prepare_dataset(self, data: List[Dict]) -> Dataset:
        """准备训练数据集"""
        # 转换为HuggingFace Dataset格式
        dataset = Dataset.from_list(data)
        
        # Tokenization
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train_stage(self, stage: int, long_edge: int, mode: str, model_path: Optional[str] = None):
        """训练单个阶段"""
        stage_name = f"stage_{stage}_{mode}_{long_edge if long_edge > 0 else 'auto'}"
        stage_dir = self.output_dir / stage_name
        
        logger.info(f"开始训练阶段 {stage_name}")
        
        # 加载数据
        loader = DatasetLoader(self.config.dataset_path)
        raw_data = loader.load_data(long_edge, mode)
        
        if not raw_data:
            logger.warning(f"没有找到尺寸为{long_edge}，模式为{mode}的数据，跳过此阶段")
            return None
        
        logger.info(f"加载了 {len(raw_data)} 条训练样本")
        
        # 格式化数据
        formatted_data = loader.format_for_sft(raw_data)
        
        # 准备数据集
        dataset = self.prepare_dataset(formatted_data)
        
        # 分割训练集和验证集
        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
        
        # 加载模型
        model = self.load_model(model_path)
        
        # 配置训练参数
        training_args = TrainingArguments(
            output_dir=str(stage_dir),
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="none",
        )
        
        # 配置数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # 开始训练
        trainer.train()
        
        # 保存最终模型
        final_model_dir = stage_dir / "final"
        trainer.save_model(str(final_model_dir))
        
        logger.info(f"阶段 {stage_name} 训练完成，模型保存至 {final_model_dir}")
        
        return str(final_model_dir)
    
    def run_progressive_training(self):
        """运行渐进式训练"""
        logger.info("开始渐进式SFT训练")
        
        current_model_path = None
        
        # 按照渐进顺序训练
        for size_idx, long_edge in enumerate(PROGRESSIVE_SIZES):
            for mode in MODES:
                stage = size_idx * len(MODES) + MODES.index(mode) + 1
                current_model_path = self.train_stage(
                    stage, long_edge, mode, current_model_path
                )
                
                if current_model_path is None:
                    continue
        
        # 保存最终模型
        final_dir = self.output_dir / "final"
        if current_model_path:
            import shutil
            shutil.copytree(current_model_path, final_dir, dirs_exist_ok=True)
            logger.info(f"渐进式训练完成，最终模型保存至 {final_dir}")


def parse_args() -> SFTConfig:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='渐进式SFT训练')
    parser.add_argument('--base-model', type=str, default='models/Qwen3-1.7B')
    parser.add_argument('--dataset-path', type=str, default='data/ascii_art_dataset/dataset.db')
    parser.add_argument('--output-dir', type=str, default='models/ascii_art_qwen3')
    parser.add_argument('--no-int4', action='store_true', help='不使用INT4量化')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max-seq-length', type=int, default=4096)
    
    args = parser.parse_args()
    
    return SFTConfig(
        base_model=args.base_model,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        use_int4=not args.no_int4,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
    )


def main():
    config = parse_args()
    trainer = ProgressiveSFTTrainer(config)
    trainer.run_progressive_training()


if __name__ == '__main__':
    main()