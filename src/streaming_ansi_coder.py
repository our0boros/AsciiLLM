# ansi_color_stream.py
import re
import math
from typing import Iterable, Tuple, List

# ---------- 正则不变 ----------
ANSI_RE = re.compile(
    r'\x1b\['
    r'(?P<params>[0-9;]*?)'
    r'(?P<cmd>[mKHJfABCDsu])'
)

# ---------- 位打包 ----------
BASE = 0x10000          # UTF-8 4 字节起点
MASK_R = 0x0F0000       # 红 4bit
MASK_G = 0x000F00       # 绿 4bit
MASK_B = 0x00000F       # 蓝 4bit
# 24bit → 20bit 压缩（4+4+4=12bit 实际用）
SHIFT_R = 16
SHIFT_G = 8
SHIFT_B = 0

def _pack_rgb(r: int, g: int, b: int) -> int:
    """24bit → 12bit → codepoint"""
    r4 = r >> 4; g4 = g >> 4; b4 = b >> 4
    return BASE + (r4 << SHIFT_R) | (g4 << SHIFT_G) | (b4 << SHIFT_B)

def _unpack_rgb(code: int) -> Tuple[int, int, int]:
    """codepoint → 24bit"""
    code -= BASE
    r4 = (code & MASK_R) >> SHIFT_R
    g4 = (code & MASK_G) >> SHIFT_G
    b4 = (code & MASK_B) >> SHIFT_B
    # 4bit 还原到 8bit
    return (r4 << 4 | r4, g4 << 4 | g4, b4 << 4 | b4)

class ANSIColorStream:
    def __init__(self):
        self.reset()

    def reset(self):
        self._rgb: Tuple[int, int, int] = (255, 255, 255)

    # ---------- 解析 ----------
    def feed(self, chunk: str) -> Iterable[Tuple[Tuple[int, int, int], str]]:
        for match in ANSI_RE.finditer(chunk):
            # 普通字符
            if match.start() > 0:
                for ch in chunk[:match.start()]:
                    yield (self._rgb, ch)
            # SGR
            if match.group('cmd') == 'm':
                params = match.group('params')
                if params == '0':
                    self._rgb = (255, 255, 255)
                elif params.startswith('38;2;'):
                    r, g, b = map(int, params.split(';')[2:5])
                    self._rgb = (r, g, b)
            chunk = chunk[match.end():]
        # 尾部
        for ch in chunk:
            yield (self._rgb, ch)

    # ---------- 编码 ----------
    def encode_rgb(self, rgb: Tuple[int, int, int]) -> str:
        return chr(_pack_rgb(*rgb))

    # ---------- 解码 ----------
    def decode_char(self, ch: str) -> Tuple[int, int, int]:
        return _unpack_rgb(ord(ch))

