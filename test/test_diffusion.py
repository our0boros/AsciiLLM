# peft_lora.py
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device=="cuda" else torch.float32

# ① 本地基础模型
model_id = r"models/dreamlike-diffusion-1.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)

# ② 挂 LoRA（PEFT 方式，支持多适配器）
lora_path = Path("models/插画风格lora模型扁平插画_V2.0.safetensors") 
# 插画风格lora模型扁平插画_V2.0.safetensors simple_linear_sd1.5/简易线条小插图_v1.0.safetensors
pipe.load_lora_weights(lora_path, adapter_name="illustration")   # 名字随意

# ③ 触发词 + 权重
prompt = "illustration style, 1girl, upper body" # "flat illustration style, 1girl, upper body"
# 1girl riding a scooter, Black Long Hair, simple lineart, bold illustration, cute style
for w in [0.7]:
    pipe.set_adapters(["illustration"], adapter_weights=[w])
    img = pipe(prompt, num_inference_steps=20).images[0]
    img.save(f"data/weight_{w}.png")