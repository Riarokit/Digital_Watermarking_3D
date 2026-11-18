"""Convert the source watermark PNG into the 1-bit BMP used by the scripts."""

from pathlib import Path

from PIL import Image


original_dir = Path("bmp/original")
original_dir.mkdir(parents=True, exist_ok=True)
source_path = original_dir / "watermark8.png"
output_path = original_dir / "watermark8.bmp"

image = Image.open(source_path).convert("1")
image = image.resize((64, 64), Image.NEAREST)
image.save(output_path)
print(f"Saved BMP: {output_path}")
