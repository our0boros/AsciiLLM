from ascii_magic import AsciiArt
from PIL import Image

img = Image.open('data/test.png')
my_art = AsciiArt.from_pillow_image(img)
my_art.to_terminal()