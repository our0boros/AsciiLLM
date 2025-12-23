#!/usr/bin/env python3
"""
简单的 Qwen3 1.7B 文本生成脚本
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model(model_path, use_int4=True):
    """加载 Qwen3 模型和分词器"""
    print(f"加载模型: {model_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    
    # 配置量化
    kwargs = {"torch_dtype": torch.float16}
    if use_int4:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    
    if not use_int4:
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.9, top_p=0.95):
    """生成文本"""
    inputs = tokenizer([prompt], return_tensors='pt', padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_path = "models/Qwen3-1.7B"
    
    # 加载模型
    model, tokenizer = load_model(model_path)
    
    # 测试提示词
    test_prompts = [
        "请描述一只可爱的小猫:",
        "Write a short story about a robot:",
        "ASCII art representation of a cat:",
        "请生成一个简单的ASCII艺术画，表示一座房子:",
    ]
    
    for prompt in test_prompts:
        print(f"\n提示词: {prompt}")
        print("-" * 50)
        result = generate_text(model, tokenizer, prompt)
        print(result)
        print("=" * 80)

if __name__ == "__main__":
    main()