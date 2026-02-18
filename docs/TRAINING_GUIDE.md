# Character LoRA Training Guide

## Overview

```
1. Collect raw images (Runware AI / manual)
2. Preprocess: resize to 1024x1024 PNG
3. Caption: Florence-2 natural language + trigger word
4. Upload to RunPod/Colab
5. Train with ai-toolkit
6. Download .safetensors
7. Validate locally with MFLUX
8. Deploy to HuggingFace + Runware AI
```

## Step 1: Prepare Raw Images

### Requirements

- **15-25 images** per character
- High resolution source (>= 1024x1024 preferred)
- Diverse angles: front (35%), 3/4 (25%), side (20%), varied (20%)
- Diverse expressions: neutral, smile, serious, surprised
- Diverse distances: closeup, waist-up, full body
- **Avoid**: same background repeated, text overlays, watermarks

### Create character directory

```bash
mkdir -p datasets/my_character/{raw,processed,captions}
```

Place raw images in `datasets/my_character/raw/`.

## Step 2: Preprocess

```bash
python scripts/preprocess_images.py \
  datasets/my_character/raw \
  datasets/my_character/processed \
  --size 1024 \
  --prefix char
```

Output: `datasets/my_character/processed/char_001.png`, `char_002.png`, ...

## Step 3: Caption

```bash
python scripts/caption_images.py \
  datasets/my_character/processed \
  datasets/my_character/captions \
  --trigger "ohwx_mychar" \
  --device mps
```

### Trigger word rules

- Use a unique token that doesn't exist in natural language
- Examples: `ohwx_mychar`, `zwx_hero`, `sks_villain`
- **Don't** use common words like "character" or "person"

### Review captions

After auto-captioning, **manually review** each `.txt` file:
- Remove references to character appearance (model learns this from images)
- Keep environment, pose, lighting, camera angle descriptions
- Ensure trigger word is at the start of every caption
- Target length: 40-100 words

## Step 4: Train

### Copy config

```bash
cp config/ai_toolkit_config.yaml config/my_character.yaml
```

Edit `config/my_character.yaml`:
- Set `folder_path` to your processed images directory
- Set `trigger_word` to your chosen trigger
- Update `name` to identify this training run
- Adjust `steps` (start with 4000, increase if underfitting)
- Replace `{trigger}` in sample prompts with your trigger word

### Upload to training machine

```bash
# Upload dataset + config to RunPod
rsync -avz datasets/my_character/ runpod:/workspace/datasets/my_character/
rsync -avz config/my_character.yaml runpod:/workspace/ai-toolkit/config/
```

### Run training

```bash
# On RunPod
cd /workspace/ai-toolkit
python run.py config/my_character.yaml
```

Training will save checkpoints every 250 steps to `output/`.
Monitor sample images to pick the best checkpoint.

## Step 5: Validate locally

Download the best `.safetensors` checkpoint to `models/checkpoints/`.

```bash
python scripts/evaluate_lora.py \
  models/checkpoints/my_character.safetensors \
  --trigger "ohwx_mychar" \
  --steps 20 \
  --scale 0.8
```

Check `validation/results/` for generated images.

### What to look for

- **Face consistency**: same facial features across all angles
- **Line style consistency**: consistent linework weight and style
- **Color palette consistency**: same colors for hair, skin, clothing
- **No artifacts**: no floating elements, distorted anatomy

## Step 6: Deploy

### Upload to HuggingFace

```bash
huggingface-cli upload baleenme/story-shorts-character-lora \
  models/checkpoints/my_character.safetensors \
  my_character.safetensors
```

### Upload to Runware AI

```python
from runware import Runware, IUploadModelLora

runware = Runware(api_key="YOUR_KEY")
await runware.connect()

payload = IUploadModelLora(
    air="yourorg:12345@1",
    name="my-character-lora",
    downloadURL="https://huggingface.co/baleenme/story-shorts-character-lora/resolve/main/my_character.safetensors",
    architecture="flux1d",
    defaultWeight=0.8,
    format="safetensors",
    positiveTriggerWords="ohwx_mychar",
    shortDescription="Character LoRA for story shorts",
)
uploaded = await runware.modelUpload(payload)
```

### Use in production

```python
result = await runware.imageInference(
    positivePrompt="ohwx_mychar, reading a book in a cafe, flat illustration style",
    model="runware:101@1",  # Flux Dev
    lora=[{"model": uploaded.air, "weight": 0.8}],
    height=1024,
    width=1024,
)
```
