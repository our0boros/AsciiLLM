from diffusers import StableDiffusionPipeline
import torch
import ascii_magic

# 检查CUDA是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

model_id = "sd-legacy/stable-diffusion-v1-5"
# 如果使用CPU，将dtype改为float32，因为float16在CPU上可能不支持
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  

ascii_art = ascii_magic.from_pillow_image(image)
print(ascii_art.to_ascii(columns=64, enhance_image=True))
# image.save("astronaut_rides_horse.png")
