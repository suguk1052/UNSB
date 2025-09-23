import argparse
import math
import os
import random
from pathlib import Path

from PIL import Image

GRAY = (128, 128, 128)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def pad_to_aspect_ratio(image: Image.Image, target_ratio: float) -> Image.Image:
    width, height = image.size
    if height == 0:
        return image

    current_ratio = width / height
    if math.isclose(current_ratio, target_ratio, rel_tol=1e-6):
        return image

    if current_ratio < target_ratio:
        new_width = max(width, int(math.ceil(height * target_ratio)))
        new_height = height
    else:
        new_width = width
        new_height = max(height, int(math.ceil(width / target_ratio)))

    canvas = Image.new("RGB", (new_width, new_height), GRAY)
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    canvas.paste(image, (left, top))
    return canvas


def resize_with_letterbox(image: Image.Image, output_width: int, output_height: int) -> Image.Image:
    width, height = image.size
    scale = min(output_width / width, output_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = image.resize((new_width, new_height), resample=Image.BICUBIC)
    canvas = Image.new("RGB", (output_width, output_height), GRAY)
    left = (output_width - new_width) // 2
    top = (output_height - new_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def process_image(path: Path, output_path: Path, output_width: int, output_height: int) -> None:
    with Image.open(path) as img:
        image = img.convert("RGB")

    target_ratio = output_width / output_height
    padded = pad_to_aspect_ratio(image, target_ratio)

    angle = random.uniform(-10.0, 10.0)
    rotated = padded.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=GRAY)

    final_image = resize_with_letterbox(rotated, output_width, output_height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_image.save(output_path)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pad, rotate, and resize images with gray background.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save processed images.")
    parser.add_argument("--output_width", type=int, required=True, help="Output image width.")
    parser.add_argument("--output_height", type=int, required=True, help="Output image height.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for rotation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    image_paths = [
        Path(root) / filename
        for root, _, files in os.walk(args.input_dir)
        for filename in files
        if is_image_file(Path(filename))
    ]

    if not image_paths:
        print("No images found to process.")
        return

    for path in image_paths:
        relative = path.relative_to(args.input_dir)
        output_path = args.output_dir / relative
        process_image(path, output_path, args.output_width, args.output_height)
        print(f"Processed {path} -> {output_path}")


if __name__ == "__main__":
    main()
