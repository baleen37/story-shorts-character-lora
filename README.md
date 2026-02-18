---
license: apache-2.0
base_model: black-forest-labs/FLUX.2-klein-base-4B
tags:
  - lora
  - text-to-image
  - character
  - flat-design
  - illustration
  - storytelling
  - shorts
  - runware
  - flux
  - flux-lora
  - mflux
  - flux2
language:
  - ko
  - en
pipeline_tag: text-to-image
---

# Story Shorts Character LoRA

A collection of LoRA models for generating **consistent flat-style characters** for AI-narrated story short-form videos.

Characters are designed with:
- **Simple, clean linework** — minimal detail, easy to animate or sequence
- **Flat vector-like illustration style** — bold colors, no complex shading
- **High consistency across expressions and poses** — suitable for multi-scene storytelling

## Intended Use

These models are built to support short-form video production where a narrator tells a story over a series of generated images. Each LoRA trains a specific character so that it remains visually coherent across different scenes, emotions, and backgrounds.

**Workflow:**
1. Generate or collect training images
2. Preprocess + caption with Florence-2
3. Train LoRA with mflux on Mac (Apple Silicon)
4. Validate locally with mflux
5. Deploy to HuggingFace + Runware AI

## Models

| Model | Character | Trigger Word | Notes |
|-------|-----------|--------------|-------|
| *(coming soon)* | — | — | — |

## Usage

```python
# Example using diffusers
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("baleenme/story-shorts-character-lora", weight_name="character_name.safetensors")

prompt = "TRIGGER_WORD, flat illustration, simple linework, white background, smiling"
image = pipe(prompt, num_inference_steps=20, guidance_scale=4.0).images[0]
```

## Local Validation (Apple Silicon)

```bash
# Using MFLUX (MLX-native)
mflux-generate \
  --prompt "TRIGGER_WORD, front view, neutral expression, white background" \
  --model flux2-klein-base-4b \
  --lora-paths path/to/lora.safetensors \
  --lora-scales 0.8 \
  --steps 20

# Or use the evaluation script with test prompts
python scripts/evaluate_lora.py path/to/lora.safetensors --trigger "TRIGGER_WORD"
```

## Runware API

```python
from runware import Runware

runware = Runware(api_key="YOUR_KEY")
await runware.connect()

result = await runware.imageInference(
    positivePrompt="TRIGGER_WORD, reading a book in a cafe, flat illustration style",
    model="runware:400@5",
    lora=[{"model": "your-lora-air", "weight": 0.8}],
    height=1024,
    width=1024,
)
```

## Training Details

- **Base model:** [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
- **Training tool:** [mflux](https://github.com/filipstrand/mflux)
- **Captioning:** Florence-2 (natural language)
- **Image style:** Flat vector illustration, minimal linework
- **Image count per character:** 15–25 curated images
- **LoRA rank:** 16
- **Steps:** 100 epochs (≈2500–4000 steps)

## License

Apache 2.0 — free to use for personal and commercial projects.
