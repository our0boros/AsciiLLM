from PIL import Image
from ascii_art_converter.generator import AsciiArtGenerator, AsciiArtConfig
from ascii_art_converter.constants import RenderMode

# Load an image
image = Image.open("data/test.png")

# Create configuration
config = AsciiArtConfig(
    width=None, # None 16 32 64 96 128
    mode=RenderMode.BRAILLE, # RenderMode.DENSITY | RenderMode.BRAILLE
    colorize=True,
)

# Generate ASCII art
generator = AsciiArtGenerator()
result = generator.convert(image, config)
print(result.text, result.colors)
# Print the result
# print(result.text)

# # Save as HTML
# from ascii_art_converter.formatters import HtmlFormatter
# html_content = HtmlFormatter.format(result, image, True)
# with open("output.html", "w", encoding="utf-8") as f:
#     f.write(html_content)

# Save as ANSI
from ascii_art_converter.formatters import AnsiColorFormatter
ansi_content = AnsiColorFormatter.format_result(result, color_mode="24bit")
# with open("output.ansi", "w", encoding="utf-8") as f:
#     f.write(ansi_content)
print(ansi_content)