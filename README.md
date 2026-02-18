---
license: apache-2.0
tags:
  - lora
  - text-to-image
  - character
  - flat-design
  - illustration
  - storytelling
  - shorts
  - runware
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
1. Generate training images via [Runware AI](https://runware.ai/)
2. Train LoRA on character-specific image set
3. Use LoRA at inference time to keep character consistent across scenes

## Models

| Model | Character | Trigger Word | Notes |
|-------|-----------|--------------|-------|
| *(coming soon)* | — | — | — |

## Usage

```python
# Example using diffusers
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("base-model-id", torch_dtype=torch.float16)
pipe.load_lora_weights("baleenme/story-shorts-character-lora", weight_name="character_name.safetensors")

prompt = "TRIGGER_WORD, flat illustration, simple linework, white background, smiling"
image = pipe(prompt).images[0]
```

## Training Details

- **Image style:** Flat vector illustration, minimal linework
- **Training tool:** Runware AI
- **Image count per character:** ~20–50 curated images
- **Base model:** *(varies per character, listed per model)*

## License

Apache 2.0 — free to use for personal and commercial projects.
