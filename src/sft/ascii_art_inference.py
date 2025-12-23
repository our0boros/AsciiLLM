#!/usr/bin/env python3
"""
ASCII艺术生成推理脚本 - 使用SFT训练后的Qwen3模型
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def load_model(base_model_path, adapter_path=None, use_int4=True):
    """加载模型和分词器"""
    print(f"加载基础模型: {base_model_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        padding_side='right',
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 配置量化
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    if use_int4:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        **kwargs
    )
    
    # 加载LoRA适配器
    if adapter_path:
        print(f"加载适配器: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def generate_ascii_art(model, tokenizer, prompt, mode="normal", long_edge=64, max_new_tokens=1024):
    """生成ASCII艺术"""
    # 构建系统提示
    system_prompt = (
        f"你是一个专业的ASCII艺术家。根据用户的描述，生成对应的ASCII艺术作品。"
        f"请使用<{mode}>模式生成ASCII艺术，"
        f"宽度约为{long_edge if long_edge > 0 else '自适应'}字符。"
    )
    
    # 构建用户提示
    user_prompt = f"请为以下描述生成ASCII艺术：<ascii-art>{prompt}</ascii-art>"
    
    # 构建对话格式
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenization
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取ASCII艺术部分
    if "<ascii-art>" in response:
        ascii_part = response.split("<ascii-art>")[1].split("</ascii-art>")[0].strip()
        return ascii_part
    else:
        return response[len(text):].strip()

def main():
    parser = argparse.ArgumentParser(description='ASCII艺术生成推理')
    parser.add_argument('--base-model', type=str, default='models/Qwen3-1.7B')
    parser.add_argument('--adapter-path', type=str, default=None)
    parser.add_argument('--no-int4', action='store_true')
    parser.add_argument('--prompt', type=str, default='一只可爱的小猫')
    parser.add_argument('--mode', type=str, choices=['normal', 'braille'], default='normal')
    parser.add_argument('--long-edge', type=int, default=64)
    parser.add_argument('--max-tokens', type=int, default=1024)
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model(
        args.base_model,
        args.adapter_path,
        use_int4=not args.no_int4
    )
    
    # 生成ASCII艺术
    print(f"提示词: {args.prompt}")
    print(f"模式: {args.mode}, 长度: {args.long_edge}")
    print("=" * 50)
    
    ascii_art = generate_ascii_art(
        model, tokenizer,
        args.prompt,
        args.mode,
        args.long_edge,
        args.max_tokens
    )
    
    print(ascii_art)

if __name__ == '__main__':
    main()