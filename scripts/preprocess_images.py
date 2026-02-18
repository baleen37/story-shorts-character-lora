#!/usr/bin/env python3
"""Preprocess raw images for LoRA training.

Resizes to 1024x1024, converts to PNG, normalizes filenames.

Usage:
    python scripts/preprocess_images.py datasets/my_character/raw datasets/my_character/processed
"""

import argparse
import sys
from pathlib import Path

from PIL import Image


def preprocess_image(src: Path, dst: Path, size: int = 1024) -> Path:
    """Resize and convert a single image to square PNG."""
    img = Image.open(src)

    if img.mode == "RGBA":
        pass  # preserve transparency
    elif img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((size, size), Image.LANCZOS)
    img.save(dst, "PNG")
    return dst


def preprocess_directory(
    input_dir: Path, output_dir: Path, size: int = 1024, prefix: str = "char"
) -> list[Path]:
    """Process all images in input_dir, save numbered PNGs to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    sources = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in extensions)

    if not sources:
        print(f"No images found in {input_dir}")
        return []

    results = []
    for i, src in enumerate(sources, start=1):
        dst = output_dir / f"{prefix}_{i:03d}.png"
        preprocess_image(src, dst, size)
        results.append(dst)
        print(f"  [{i}/{len(sources)}] {src.name} -> {dst.name}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Preprocess images for LoRA training")
    parser.add_argument("input_dir", type=Path, help="Directory with raw images")
    parser.add_argument("output_dir", type=Path, help="Directory for processed images")
    parser.add_argument("--size", type=int, default=1024, help="Target size (default: 1024)")
    parser.add_argument("--prefix", default="char", help="Filename prefix (default: char)")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory")
        sys.exit(1)

    print(f"Processing {args.input_dir} -> {args.output_dir} ({args.size}x{args.size})")
    results = preprocess_directory(args.input_dir, args.output_dir, args.size, args.prefix)
    print(f"\nDone: {len(results)} images processed")


if __name__ == "__main__":
    main()
