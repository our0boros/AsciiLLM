# converter.py
import subprocess
from pathlib import Path
from typing import List

EXE = Path("bin/ascii-image-converter.exe")   # 如已加入PATH 直接写 "ascii-image-converter"

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
    if not img_path.exists(): raise FileNotFoundError(img_path)
    if width_chars <= 0: cmd = [str(EXE), str(img_path)]
    else: cmd = [str(EXE), str(img_path), "-W", str(width_chars)]
    if color: cmd.append("-C")
    if mode == "braille": cmd.append("-b")
    elif mode == "complex": cmd.append("-c")
    elif mode == "normal": pass
    else: raise ValueError(f"invalid mode: {mode}")
    # 捕获终端输出
    completed = subprocess.run(
        cmd, capture_output=True, text=True, encoding='utf-8', errors='replace'
    )
    if completed.returncode != 0: raise RuntimeError(f"converter failed: {completed.stderr}")
    return str(completed.stdout)

if __name__ == "__main__":
    lists = [img_to_ascii(Path("data/test.png")), img_to_ascii(Path("data/test.png"), color=True)]
    print(lists)