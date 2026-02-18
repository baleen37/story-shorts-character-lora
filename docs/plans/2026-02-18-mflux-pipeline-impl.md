# FLUX.2 Klein 4B mflux Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 기존 Flux.1 Dev + ai-toolkit 기반 프로젝트를 FLUX.2 Klein Base 4B + mflux (Mac 네이티브) 기반으로 전환하고, Jupyter Notebook으로 학습→배포 전체 파이프라인을 구현한다.

**Architecture:** mflux-train으로 Mac M4 Pro에서 로컬 LoRA 학습, HuggingFace Hub에 safetensors 업로드, Runware AI에 등록하여 클라우드 추론. 전 과정을 Jupyter Notebook 셀 단위로 실행.

**Tech Stack:** mflux v0.16.5 (MLX), FLUX.2 Klein Base 4B, Florence-2, huggingface_hub, Runware Python SDK, python-dotenv

---

## Context

### 환경
- 로컬: Mac M4 Pro 48GB — 데이터 준비, 학습, 검증 전부 로컬
- 배포: HuggingFace Hub (공개 URL) → Runware AI (추론)

### 기존 재활용 파일
- `scripts/preprocess_images.py` — 이미지 리사이즈 (그대로 사용)
- `scripts/caption_images.py` — Florence-2 캡션 생성 (그대로 사용)
- `scripts/evaluate_lora.py` — 로컬 검증 (모델명 업데이트 필요)
- `validation/test_prompts.json` — 30개 검증 프롬프트 (그대로 사용)

### 새로 만들 파일
- `config/train.json` — mflux 학습 설정
- `notebooks/pipeline.ipynb` — 전체 파이프라인 Notebook
- `.env` — API 키 (gitignored)

### 수정할 파일
- `requirements.txt` — mflux, python-dotenv 추가
- `.gitignore` — .env, train output 추가
- `scripts/evaluate_lora.py` — 모델명 flux2-klein-base-4b로 변경
- `README.md` — FLUX.2 + mflux 반영

---

## Tasks

### Task 1: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Update requirements**

```txt
# === Data Preparation (local Mac) ===
Pillow>=10.0.0
opencv-python>=4.8.0
numpy>=1.24.0

# === Captioning ===
transformers>=4.44.0
onnxruntime>=1.16.0

# === Training (Mac MLX native) ===
mflux>=0.16.5

# === Inference & Deployment ===
huggingface-hub>=0.20.0
safetensors>=0.4.0
runware>=0.5.0
python-dotenv>=1.0.0

# === Utilities ===
rich>=13.0.0
tqdm>=4.65.0
```

Key changes:
- Remove ai-toolkit comments (no longer used)
- Add `mflux>=0.16.5` as direct dependency
- Add `python-dotenv>=1.0.0` for .env loading
- Remove "Install MFLUX separately" comment

**Step 2: Verify syntax**

Run: `pip install --dry-run -r requirements.txt 2>&1 | head -5`
Expected: no syntax errors

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "Update requirements: add mflux and python-dotenv, remove ai-toolkit references"
```

---

### Task 2: Update .gitignore

**Files:**
- Modify: `.gitignore`

**Step 1: Add entries for .env and mflux training output**

Append to `.gitignore`:
```
# Environment variables (API keys)
.env
.env.*

# mflux training output
train_*/
```

**Step 2: Commit**

```bash
git add .gitignore
git commit -m "Add .env and mflux train output to gitignore"
```

---

### Task 3: Create mflux train.json config

**Files:**
- Create: `config/train.json`

**Step 1: Write train.json**

This is a template — `model` and `data` fields will be overridden from the notebook when a specific character is trained.

```json
{
  "model": "flux2-klein-base-4b",
  "data": "datasets/character/mflux/",
  "seed": 42,
  "steps": 40,
  "guidance": 1.0,
  "quantize": null,
  "low_ram": false,
  "max_resolution": 1024,
  "training_loop": {
    "num_epochs": 100,
    "batch_size": 1,
    "timestep_low": 25,
    "timestep_high": 40
  },
  "optimizer": {
    "name": "AdamW",
    "learning_rate": 1e-4
  },
  "checkpoint": {
    "output_path": "train",
    "save_frequency": 25
  },
  "monitoring": {
    "plot_frequency": 1,
    "generate_image_frequency": 20
  },
  "lora_layers": {
    "targets": [
      { "module_path": "transformer_blocks.{block}.attn.to_q", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.attn.to_k", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.attn.to_v", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.attn.to_out", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.attn.add_q_proj", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.attn.add_k_proj", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.attn.add_v_proj", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.attn.to_add_out", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.ff.linear_in", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.ff.linear_out", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.ff_context.linear_in", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "transformer_blocks.{block}.ff_context.linear_out", "blocks": { "start": 0, "end": 5 }, "rank": 16 },
      { "module_path": "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj", "blocks": { "start": 0, "end": 20 }, "rank": 16 },
      { "module_path": "single_transformer_blocks.{block}.attn.to_out", "blocks": { "start": 0, "end": 20 }, "rank": 16 }
    ]
  }
}
```

**Step 2: Verify JSON**

Run: `python -c "import json; json.load(open('config/train.json')); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add config/train.json
git commit -m "Add mflux FLUX.2 Klein 4B training config template"
```

---

### Task 4: Update evaluate_lora.py for FLUX.2

**Files:**
- Modify: `scripts/evaluate_lora.py:29-36`

**Step 1: Change model from "dev" to "flux2-klein-base-4b"**

In `run_mflux()`, change:
```python
        "--model", "dev",
```
to:
```python
        "--model", "flux2-klein-base-4b",
```

Also remove `-q` (quantize) flag since FLUX.2 Klein 4B fits in 48GB without quantization. Change the function signature to default `quantize=None` and only add `-q` flag if quantize is not None.

Updated `run_mflux`:
```python
def run_mflux(
    prompt: str,
    output_path: Path,
    lora_path: Path,
    lora_scale: float = 0.8,
    steps: int = 20,
    seed: int = 42,
    quantize: int | None = None,
) -> bool:
    """Run MFLUX to generate a single image."""
    cmd = [
        "mflux-generate",
        "--prompt", prompt,
        "--model", "flux2-klein-base-4b",
        "--steps", str(steps),
        "--seed", str(seed),
        "--lora-paths", str(lora_path),
        "--lora-scales", str(lora_scale),
        "--output", str(output_path),
    ]
    if quantize is not None:
        cmd.extend(["-q", str(quantize)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  MFLUX error: {result.stderr[:200]}")
        return False
    return True
```

Also update the argparse default for `-q`:
```python
    parser.add_argument("-q", "--quantize", type=int, default=None, help="Quantization bits (default: None)")
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('scripts/evaluate_lora.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add scripts/evaluate_lora.py
git commit -m "Update evaluate_lora.py for FLUX.2 Klein 4B model"
```

---

### Task 5: Create .env.example

**Files:**
- Create: `.env.example`

**Step 1: Write .env.example**

```
# HuggingFace Hub token (write access)
# Get from: https://huggingface.co/settings/tokens
HF_TOKEN=hf_YOUR_TOKEN_HERE

# Runware AI API key
# Get from: https://runware.ai dashboard
RUNWARE_API_KEY=YOUR_RUNWARE_KEY_HERE

# HuggingFace repo for uploading LoRA models
HF_REPO_ID=your-username/story-shorts-character-lora
```

**Step 2: Commit**

```bash
git add .env.example
git commit -m "Add .env.example with required API key placeholders"
```

---

### Task 6: Create pipeline.ipynb

**Files:**
- Create: `notebooks/pipeline.ipynb`
- Keep: `notebooks/test_pipeline.ipynb` (existing tests still valid)

This is the core deliverable. 9 sections as cells.

**Step 1: Write the notebook**

Cell 1 (markdown): Title
```markdown
# FLUX.2 Klein 4B Character LoRA Pipeline

mflux로 Mac에서 학습 → HuggingFace 업로드 → Runware AI 추론

## Prerequisites
- mflux >= 0.16.5 (`pip install mflux`)
- `.env` file with HF_TOKEN, RUNWARE_API_KEY, HF_REPO_ID
```

Cell 2 (code): Setup
```python
import os, sys, json, shutil, zipfile, subprocess
from pathlib import Path
from dotenv import load_dotenv

# Project root
ROOT = Path(os.getcwd()).parent if Path(os.getcwd()).name == "notebooks" else Path(os.getcwd())
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

# Load environment
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
RUNWARE_API_KEY = os.getenv("RUNWARE_API_KEY")
HF_REPO_ID = os.getenv("HF_REPO_ID")

assert HF_TOKEN, "HF_TOKEN not set in .env"
assert RUNWARE_API_KEY, "RUNWARE_API_KEY not set in .env"
assert HF_REPO_ID, "HF_REPO_ID not set in .env"
print(f"Project root: {ROOT}")
print(f"HF repo: {HF_REPO_ID}")
print("API keys loaded OK")
```

Cell 3 (markdown): Configuration
```markdown
## 1. Configuration

Set your character name and trigger word here. All subsequent cells use these values.
```

Cell 4 (code): Config variables
```python
# === EDIT THESE ===
CHARACTER_NAME = "my_character"       # directory name, no spaces
TRIGGER_WORD = "ohwx_mychar"          # unique trigger word
LORA_VERSION = "1"                    # increment for new versions

# === Derived paths (don't edit) ===
RAW_DIR = ROOT / f"datasets/{CHARACTER_NAME}/raw"
PROCESSED_DIR = ROOT / f"datasets/{CHARACTER_NAME}/processed"
CAPTIONS_DIR = ROOT / f"datasets/{CHARACTER_NAME}/captions"
MFLUX_DATA_DIR = ROOT / f"datasets/{CHARACTER_NAME}/mflux"
TRAIN_CONFIG = ROOT / "config/train.json"
CHECKPOINTS_DIR = ROOT / "models/checkpoints"

for d in [RAW_DIR, PROCESSED_DIR, CAPTIONS_DIR, MFLUX_DATA_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Character: {CHARACTER_NAME}")
print(f"Trigger: {TRIGGER_WORD}")
print(f"Raw images go in: {RAW_DIR}")
```

Cell 5 (markdown): Data prep
```markdown
## 2. Data Preparation

1. Place 15-25 raw images in the `raw/` directory shown above
2. Run preprocessing (resize to 1024x1024 PNG)
3. Run captioning (Florence-2 + trigger word)
4. Merge into mflux data directory (image + txt pairs in same folder)
```

Cell 6 (code): Preprocess
```python
from preprocess_images import preprocess_directory

results = preprocess_directory(RAW_DIR, PROCESSED_DIR, size=1024, prefix="char")
print(f"\nProcessed {len(results)} images")
if not results:
    print("⚠ No images found. Place images in:", RAW_DIR)
```

Cell 7 (code): Caption
```python
from caption_images import caption_directory

results = caption_directory(
    PROCESSED_DIR,
    CAPTIONS_DIR,
    trigger_word=TRIGGER_WORD,
    device="mps",  # Apple Silicon
)
print(f"\nGenerated {len(results)} captions")
```

Cell 8 (code): Merge into mflux format
```python
# mflux expects image + txt pairs in the same directory
# Copy processed images and captions into mflux data dir

import shutil

# Clean previous data
for f in MFLUX_DATA_DIR.iterdir():
    f.unlink()

count = 0
for img_path in sorted(PROCESSED_DIR.glob("*.png")):
    txt_path = CAPTIONS_DIR / f"{img_path.stem}.txt"
    if not txt_path.exists():
        print(f"WARNING: no caption for {img_path.name}, skipping")
        continue
    shutil.copy2(img_path, MFLUX_DATA_DIR / img_path.name)
    shutil.copy2(txt_path, MFLUX_DATA_DIR / txt_path.name)
    count += 1

# Add preview prompt for monitoring training progress
preview_path = MFLUX_DATA_DIR / "preview_1.txt"
preview_path.write_text(f"{TRIGGER_WORD}, front view, neutral expression, white background, flat illustration style")

print(f"Prepared {count} image+caption pairs in {MFLUX_DATA_DIR}")
print(f"Preview prompt: {preview_path.read_text()}")
```

Cell 9 (markdown): Training
```markdown
## 3. Training

Runs mflux-train with the config template. Updates the data path to point to our character's mflux directory.
```

Cell 10 (code): Train
```python
# Update train.json with current character's data path
config = json.loads(TRAIN_CONFIG.read_text())
config["data"] = str(MFLUX_DATA_DIR) + "/"
TRAIN_CONFIG.write_text(json.dumps(config, indent=2))

print(f"Training config updated: data = {config['data']}")
print(f"Model: {config['model']}")
print(f"Epochs: {config['training_loop']['num_epochs']}")
print(f"Batch size: {config['training_loop']['batch_size']}")
print(f"\nStarting training... (this will take a while)")

# Run training
result = subprocess.run(
    ["mflux-train", "--config", str(TRAIN_CONFIG)],
    cwd=str(ROOT),
)

if result.returncode == 0:
    print("\nTraining complete!")
else:
    print(f"\nTraining failed with exit code {result.returncode}")
```

Cell 11 (markdown): Extract checkpoint
```markdown
## 4. Extract LoRA from Checkpoint

mflux saves checkpoints as ZIP files. We extract the safetensors adapter from the best checkpoint.
```

Cell 12 (code): Extract
```python
# Find the latest training run
train_dirs = sorted(ROOT.glob("train_*"), key=lambda p: p.name)
if not train_dirs:
    raise FileNotFoundError("No training output found. Run training first.")

latest_run = train_dirs[-1]
checkpoint_dir = latest_run / "checkpoints"
checkpoints = sorted(checkpoint_dir.glob("*_checkpoint.zip"))

print(f"Training run: {latest_run.name}")
print(f"Found {len(checkpoints)} checkpoints:")
for cp in checkpoints:
    print(f"  {cp.name}")

# Use the latest checkpoint (highest step count)
best_checkpoint = checkpoints[-1]
print(f"\nUsing: {best_checkpoint.name}")

# Extract safetensors adapter
output_safetensors = CHECKPOINTS_DIR / f"{CHARACTER_NAME}.safetensors"
with zipfile.ZipFile(best_checkpoint) as zf:
    adapter_files = [f for f in zf.namelist() if f.endswith("_adapter.safetensors")]
    if not adapter_files:
        raise FileNotFoundError(f"No adapter.safetensors found in {best_checkpoint.name}")

    adapter_name = adapter_files[0]
    with zf.open(adapter_name) as src, open(output_safetensors, "wb") as dst:
        dst.write(src.read())

size_mb = output_safetensors.stat().st_size / (1024 * 1024)
print(f"\nExtracted: {output_safetensors}")
print(f"Size: {size_mb:.1f} MB")
```

Cell 13 (markdown): Local test
```markdown
## 5. Local Test

Quick inference with mflux-generate to verify the LoRA works before uploading.
```

Cell 14 (code): Local test
```python
test_prompt = f"{TRIGGER_WORD}, front view, neutral expression, white background, flat illustration style, clean linework"
test_output = ROOT / "validation/results" / f"{CHARACTER_NAME}_test.png"

result = subprocess.run([
    "mflux-generate",
    "--prompt", test_prompt,
    "--model", "flux2-klein-base-4b",
    "--steps", "20",
    "--seed", "42",
    "--lora-paths", str(output_safetensors),
    "--lora-scales", "0.8",
    "--output", str(test_output),
])

if result.returncode == 0:
    from PIL import Image
    img = Image.open(test_output)
    display(img)
    print(f"Saved: {test_output}")
else:
    print("Generation failed. Check mflux installation.")
```

Cell 15 (markdown): Upload
```markdown
## 6. Upload to HuggingFace Hub

Uploads the safetensors file to a public HF repo so Runware can download it.
```

Cell 16 (code): Upload
```python
from huggingface_hub import HfApi

api = HfApi(token=HF_TOKEN)

# Create repo if it doesn't exist
api.create_repo(repo_id=HF_REPO_ID, repo_type="model", private=False, exist_ok=True)

# Upload safetensors
hf_filename = f"{CHARACTER_NAME}.safetensors"
api.upload_file(
    path_or_fileobj=str(output_safetensors),
    path_in_repo=hf_filename,
    repo_id=HF_REPO_ID,
    repo_type="model",
)

download_url = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{hf_filename}"
print(f"Uploaded: {hf_filename}")
print(f"Download URL: {download_url}")
```

Cell 17 (markdown): Register
```markdown
## 7. Register on Runware AI

Registers the LoRA with Runware so it can be used for inference via API.

**Note:** The `architecture` field uses `flux1d` as FLUX.2-specific values are not yet in the SDK. If registration fails, contact Runware support for the correct architecture value.
```

Cell 18 (code): Register
```python
import asyncio
from runware import Runware, IUploadModelLora

async def register_lora():
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    payload = IUploadModelLora(
        air=f"civitai:{CHARACTER_NAME}@{LORA_VERSION}",
        name=f"story-shorts-{CHARACTER_NAME}",
        downloadURL=download_url,
        uniqueIdentifier=f"story-shorts-{CHARACTER_NAME}-v{LORA_VERSION}",
        version=LORA_VERSION,
        architecture="flux1d",
        format="safetensors",
        positiveTriggerWords=TRIGGER_WORD,
        private=True,
        shortDescription=f"Character LoRA for {CHARACTER_NAME} (FLUX.2 Klein 4B)",
    )

    result = await runware.modelUpload(payload)
    return result

upload_result = asyncio.run(register_lora())
print(f"Runware registration result: {upload_result}")
LORA_AIR = f"civitai:{CHARACTER_NAME}@{LORA_VERSION}"
print(f"LoRA AIR ID: {LORA_AIR}")
```

Cell 19 (markdown): Inference
```markdown
## 8. Runware Inference Test

Generate images using the deployed LoRA via Runware API.
```

Cell 20 (code): Inference
```python
from runware import Runware, IImageInference, ILora

async def run_inference(prompt: str, lora_weight: float = 0.8):
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    payload = IImageInference(
        positivePrompt=prompt,
        model="runware:400@5",  # FLUX.2 Klein 4B Base
        lora=[ILora(model=LORA_AIR, weight=lora_weight)],
        width=1024,
        height=1024,
        numberResults=1,
    )

    images = await runware.imageInference(requestImage=payload)
    return images

# Test prompts
test_prompts = [
    f"{TRIGGER_WORD}, front view, neutral expression, white background, flat illustration",
    f"{TRIGGER_WORD}, sitting and reading a book, cozy room, warm lighting, flat illustration",
    f"{TRIGGER_WORD}, walking down a city street, daytime, flat style",
]

from IPython.display import display, Image as IPImage
import urllib.request

for i, prompt in enumerate(test_prompts):
    print(f"\n[{i+1}/{len(test_prompts)}] {prompt[:60]}...")
    images = asyncio.run(run_inference(prompt))
    for img in images:
        print(f"  URL: {img.imageURL}")
        # Download and display
        img_path = ROOT / f"validation/results/{CHARACTER_NAME}_runware_{i+1}.png"
        urllib.request.urlretrieve(img.imageURL, str(img_path))
        display(IPImage(filename=str(img_path)))
```

**Step 2: Verify notebook structure**

Run: `python -c "import json; nb = json.load(open('notebooks/pipeline.ipynb')); print(f'{len(nb[\"cells\"])} cells'); print('OK')"`
Expected: `20 cells` and `OK`

**Step 3: Commit**

```bash
git add notebooks/pipeline.ipynb
git commit -m "Add FLUX.2 Klein 4B training pipeline notebook (mflux + HF + Runware)"
```

---

### Task 7: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Update to reflect FLUX.2 + mflux stack**

Key changes:
- `base_model` frontmatter: `FLUX.1-dev` → `black-forest-labs/FLUX.2-klein-base-4B`
- Remove `ai-toolkit` tag, add `mflux`, `flux2`
- Workflow: mention mflux local training instead of RunPod/Colab
- Usage: update diffusers example to FLUX.2
- Training Details: update to FLUX.2 Klein 4B + mflux
- Local Validation: update model to `flux2-klein-base-4b`

**Step 2: Commit**

```bash
git add README.md
git commit -m "Update README for FLUX.2 Klein 4B + mflux stack"
```

---

### Task 8: Verify all files and final commit

**Step 1: Verify JSON config**

Run: `python -c "import json; json.load(open('config/train.json')); print('OK')"`
Expected: `OK`

**Step 2: Verify evaluate_lora.py**

Run: `python -c "import ast; ast.parse(open('scripts/evaluate_lora.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Verify notebook**

Run: `python -c "import json; nb = json.load(open('notebooks/pipeline.ipynb')); print(f'{len(nb[\"cells\"])} cells OK')"`
Expected: `20 cells OK`

**Step 4: git status check**

Run: `git status`
Expected: clean working tree (all committed)
