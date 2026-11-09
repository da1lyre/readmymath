from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "webapp" / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = (
        list("0123456789") +
        list("abcdefghijklmnopqrstuvwxyz") +
        list("αβγδεζηθικλμνξοπρστυφχψω")
)

SIZE = 512

FONT_PATH = Path(__file__).resolve().parent.parent / "webapp" / "fonts" / "STIXTwoMath-Regular.ttf"

font = ImageFont.truetype(str(FONT_PATH), 350)

print(f"Generating {len(SYMBOLS)} examples...")

for char in SYMBOLS:
    filename = f"u{ord(char):04x}.png"
    img = Image.new("L", (SIZE, SIZE), 255)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (SIZE - w) / 2 - bbox[0]
    y = (SIZE - h) / 2 - bbox[1]

    draw.text((x, y), char, fill=0, font=font)
    img.save(TEMPLATES_DIR / filename)

print("Done")