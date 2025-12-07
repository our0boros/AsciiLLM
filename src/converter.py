# converter.py
import subprocess
from pathlib import Path
from typing import List, Dict
import hashlib
import functools

EXE = Path("bin/ascii-image-converter.exe")   # 如已加入PATH 直接写 "ascii-image-converter"

# 检查并缓存EXE是否存在
exists_cache = {}

def check_exe_exists(exe_path: Path) -> bool:
    """检查可执行文件是否存在并缓存结果"""
    if exe_path not in exists_cache:
        exists_cache[exe_path] = exe_path.exists()
    return exists_cache[exe_path]

# 创建一个简单的缓存装饰器
def cache_ascii(maxsize=100):
    """缓存ASCII转换结果的装饰器"""
    cache = {}
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(img_path: Path, width_chars: int = 64, color: bool = False, mode: str = "normal"):
            # 创建缓存键
            cache_key = (str(img_path), width_chars, color, mode, img_path.stat().st_mtime)
            
            if cache_key in cache:
                return cache[cache_key]
            
            # 调用原函数
            result = func(img_path, width_chars, color, mode)
            
            # 保持缓存大小
            if len(cache) >= maxsize:
                # 删除最旧的条目
                del cache[next(iter(cache))]
            
            cache[cache_key] = result
            return result
        
        return wrapper
    
    return decorator

@cache_ascii(maxsize=100)
def img_to_ascii(
    img_path: Path,
    width_chars: int = 64,
    color: bool = False,
    mode: str = "normal", # "normal" | "complex" | "braille"
) -> str:
    """
    调用 ascii-image-converter.exe 返回终端彩色/单色字符画
    width_chars : 输出最长边字符数（程序自动保持比例）
    color       : -C  彩色
    mode        : -c  复杂字符 | -b  点阵⣿模式（优先）
    """
    if not check_exe_exists(EXE):
        raise RuntimeError(f"ASCII转换工具不存在: {EXE}")
    
    if not img_path.exists(): 
        raise FileNotFoundError(img_path)
    
    # 构建命令
    cmd = [str(EXE)]
    
    # 优化参数顺序
    cmd.extend([str(img_path)])
    if width_chars > 0:
        cmd.extend(["-W", str(width_chars)])
    
    # 添加模式参数
    if mode == "braille":
        cmd.append("-b")
    elif mode == "complex":
        cmd.append("-c")
    elif mode != "normal":
        raise ValueError(f"invalid mode: {mode}")
    
    # 添加颜色参数
    if color:
        cmd.append("-C")
    
    # 优化subprocess调用
    completed = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        encoding='utf-8', 
        errors='replace',
        creationflags=subprocess.CREATE_NO_WINDOW  # 在Windows上隐藏控制台窗口
    )
    
    if completed.returncode != 0:
        raise RuntimeError(f"converter failed: {completed.stderr}")
    
    return str(completed.stdout)

def generate_ascii_from_pil(img, width_chars: int = 64, color: bool = False, mode: str = "normal") -> str:
    """
    从PIL图像对象生成ASCII字符画（通过临时文件）
    """
    import tempfile
    from PIL import Image
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        # 保存为PNG格式
        img.save(tmp_path, format='PNG')
        # 调用转换函数
        result = img_to_ascii(tmp_path, width_chars, color, mode)
    finally:
        # 确保临时文件被删除
        if tmp_path.exists():
            tmp_path.unlink()
    
    return result

if __name__ == "__main__":
    lists = [img_to_ascii(Path("data/test.png")), img_to_ascii(Path("data/test.png"), color=True)]
    print(lists)