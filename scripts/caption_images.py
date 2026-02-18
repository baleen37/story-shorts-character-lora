#!/usr/bin/env python3
"""Generate captions for training images using Florence-2.

Produces natural language captions for Flux LoRA training.
Optionally appends WD14 tags for dual-captioning (CLIP-L + T5).

Usage:
    python scripts/caption_images.py datasets/my_character/processed datasets/my_character/captions --trigger "ohwx_char"
    python scripts/caption_images.py datasets/my_character/processed datasets/my_character/captions --trigger "ohwx_char" --wd14
"""

import argparse
import sys
from pathlib import Path

from PIL import Image


def caption_with_florence2(image_path: Path, processor, model, device: str) -> str:
    """Generate a natural language caption using Florence-2."""
    image = Image.open(image_path).convert("RGB")
    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=150,
        num_beams=3,
    )
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Florence-2 returns the prompt prefix in output, strip it
    if caption.startswith(prompt):
        caption = caption[len(prompt):].strip()
    return caption


def load_florence2(device: str):
    """Load Florence-2 model and processor."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = "microsoft/Florence-2-base"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True
    ).to(device)
    model.eval()
    return processor, model


def caption_directory(
    input_dir: Path,
    output_dir: Path,
    trigger_word: str,
    device: str = "cpu",
) -> list[Path]:
    """Caption all images in input_dir, save .txt files to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in extensions)

    if not images:
        print(f"No images found in {input_dir}")
        return []

    print(f"Loading Florence-2 model...")
    processor, model = load_florence2(device)

    results = []
    for i, img_path in enumerate(images, start=1):
        caption = caption_with_florence2(img_path, processor, model, device)
        full_caption = f"{trigger_word}, {caption}"

        txt_path = output_dir / f"{img_path.stem}.txt"
        txt_path.write_text(full_caption, encoding="utf-8")
        results.append(txt_path)
        print(f"  [{i}/{len(images)}] {img_path.name} -> {txt_path.name}")
        print(f"    Caption: {full_caption[:80]}...")

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate captions for LoRA training images")
    parser.add_argument("input_dir", type=Path, help="Directory with processed images")
    parser.add_argument("output_dir", type=Path, help="Directory for caption .txt files")
    parser.add_argument("--trigger", required=True, help="Trigger word to prepend (e.g. ohwx_char)")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference: cpu, cuda, mps (default: cpu)",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory")
        sys.exit(1)

    print(f"Captioning {args.input_dir} -> {args.output_dir}")
    print(f"Trigger word: {args.trigger}")
    print(f"Device: {args.device}")
    results = caption_directory(args.input_dir, args.output_dir, args.trigger, args.device)
    print(f"\nDone: {len(results)} captions generated")


if __name__ == "__main__":
    main()
