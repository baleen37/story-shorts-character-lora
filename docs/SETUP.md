# Environment Setup

## Local (Mac — data prep & validation)

### Prerequisites

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3/M4)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install MFLUX (local inference)

```bash
pip install mflux
# or with uv:
uv tool install mflux
```

Verify:

```bash
mflux-generate --help
```

## Training Machine (RunPod / Colab)

### Option A: RunPod with ai-toolkit template

1. Launch RunPod pod with **ai-toolkit** template (official, by ostris)
2. Dataset is pre-configured, just upload your data

### Option B: Manual setup

```bash
# Clone ai-toolkit
git clone https://github.com/ostris/ai-toolkit
cd ai-toolkit
git submodule update --init --recursive

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (for Flux Dev access)
huggingface-cli login
```

### GPU Requirements

| VRAM | Batch Size | Notes |
|------|-----------|-------|
| 12GB | 1 | Minimum viable, use quantize=true |
| 16GB | 2 | Comfortable training |
| 24GB | 4 | Fast training, recommended |
| 40GB+ | 4-8 | A100, fastest |

## Runware AI (deployment)

1. Create account at [runware.ai](https://runware.ai)
2. Get API key from dashboard
3. Install SDK: `pip install runware`

```python
from runware import Runware
runware = Runware(api_key="YOUR_API_KEY")
await runware.connect()
```
