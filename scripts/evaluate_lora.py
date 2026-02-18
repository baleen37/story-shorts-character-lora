#!/usr/bin/env python3
"""Evaluate a trained LoRA by generating images from test prompts.

Uses MFLUX (MLX-native) for Apple Silicon inference.
Requires: pip install mflux

Usage:
    python scripts/evaluate_lora.py path/to/lora.safetensors --trigger "ohwx_char"
    python scripts/evaluate_lora.py path/to/lora.safetensors --trigger "ohwx_char" --prompts validation/test_prompts.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_mflux(
    prompt: str,
    output_path: Path,
    lora_path: Path,
    lora_scale: float = 0.8,
    steps: int = 20,
    seed: int = 42,
    quantize: int = 8,
) -> bool:
    """Run MFLUX to generate a single image."""
    cmd = [
        "mflux-generate",
        "--prompt", prompt,
        "--model", "dev",
        "--steps", str(steps),
        "--seed", str(seed),
        "-q", str(quantize),
        "--lora-paths", str(lora_path),
        "--lora-scales", str(lora_scale),
        "--output", str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  MFLUX error: {result.stderr[:200]}")
        return False
    return True


def load_prompts(prompts_path: Path, trigger_word: str) -> list[dict]:
    """Load prompts from JSON file and inject trigger word."""
    data = json.loads(prompts_path.read_text())
    prompts = []
    for category in data.get("categories", []):
        for p in category.get("prompts", []):
            text = p["text"].replace("{trigger}", trigger_word)
            prompts.append({
                "category": category["name"],
                "name": p["name"],
                "text": text,
            })
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA with MFLUX")
    parser.add_argument("lora_path", type=Path, help="Path to LoRA .safetensors file")
    parser.add_argument("--trigger", required=True, help="Trigger word for the LoRA")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("validation/test_prompts.json"),
        help="Path to test prompts JSON (default: validation/test_prompts.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation/results"),
        help="Output directory for generated images",
    )
    parser.add_argument("--steps", type=int, default=20, help="Inference steps (default: 20)")
    parser.add_argument("--scale", type=float, default=0.8, help="LoRA scale (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("-q", "--quantize", type=int, default=8, help="Quantization bits (default: 8)")
    args = parser.parse_args()

    if not args.lora_path.exists():
        print(f"Error: LoRA file not found: {args.lora_path}")
        sys.exit(1)

    prompts = load_prompts(args.prompts, args.trigger)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"LoRA: {args.lora_path}")
    print(f"Prompts: {len(prompts)} from {args.prompts}")
    print(f"Output: {args.output_dir}")
    print(f"Settings: steps={args.steps}, scale={args.scale}, seed={args.seed}, q={args.quantize}")
    print()

    succeeded = 0
    failed = 0
    for i, p in enumerate(prompts, start=1):
        output_path = args.output_dir / f"{p['category']}_{p['name']}.png"
        print(f"[{i}/{len(prompts)}] {p['category']}/{p['name']}")
        print(f"  Prompt: {p['text'][:80]}...")

        ok = run_mflux(
            p["text"], output_path, args.lora_path,
            lora_scale=args.scale, steps=args.steps,
            seed=args.seed, quantize=args.quantize,
        )
        if ok:
            succeeded += 1
            print(f"  -> {output_path}")
        else:
            failed += 1

    print(f"\nDone: {succeeded} succeeded, {failed} failed out of {len(prompts)} prompts")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
