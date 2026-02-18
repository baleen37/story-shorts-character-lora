# FLUX.2 Klein 4B LoRA Pipeline — mflux + Runware

**Goal:** Mac M4 Pro에서 mflux로 가상 캐릭터 FLUX.2 Klein Base 4B LoRA를 학습하고, HuggingFace Hub → Runware AI로 배포하는 전체 파이프라인을 Jupyter Notebook으로 구현한다.

**Migration:** 기존 Flux.1 Dev + ai-toolkit (CUDA) 기반 → FLUX.2 Klein Base 4B + mflux (Mac MLX) 기반으로 전환.

---

## Architecture

```
[이미지 + 캡션] → mflux-train → LoRA safetensors → HF Hub → Runware 등록 → Runware 추론
```

| 단계 | 도구 | 실행 환경 |
|------|------|----------|
| 데이터 준비 | Pillow + Florence-2 | Mac (기존 스크립트 재활용) |
| 학습 | mflux v0.16.5 (`mflux-train`) | Mac M4 Pro 48GB |
| 업로드 | huggingface_hub | Mac |
| 등록 | Runware Python SDK | Mac |
| 추론 | Runware API | 클라우드 |

## Key Technical Details

### mflux 학습
- 베이스 모델: `flux2-klein-base-4b` (non-distilled, 학습 전용)
- 설정: `train.json` (JSON 형식)
- 데이터셋: `이미지.png` + `이미지.txt` 쌍, 같은 디렉토리
- 출력: `train_YYYYMMDD_HHMMSS/checkpoints/NNNNNNN_checkpoint.zip`
- ZIP 내부: `*_adapter.safetensors` (diffusers 호환 키 네이밍)
- Python API: `TrainingRunner.train(config_path=...)`

### Runware 등록
- 베이스 모델 AIR: `runware:400@5` (FLUX.2 Klein 4B Base)
- architecture: `flux1d` (FLUX.2 전용 값 없음, 이게 가장 근접)
- LoRA 포맷: safetensors (mflux 출력이 diffusers 호환이므로 동작 예상)
- **리스크**: architecture 값이 정확하지 않을 수 있음 → 실패 시 Runware 지원 문의

### 데이터셋 레이아웃 (mflux용)
기존 `datasets/character/processed/` + `datasets/character/captions/` 구조에서
mflux가 요구하는 "이미지와 txt가 같은 디렉토리" 형태로 합쳐야 함.

```
datasets/character/mflux/
  char_001.png
  char_001.txt
  char_002.png
  char_002.txt
  preview_1.txt     # (선택) 학습 중 프리뷰 프롬프트
```

## What Changes vs Existing

| 항목 | 기존 | 변경 |
|------|------|------|
| 베이스 모델 | Flux.1 Dev 12B | FLUX.2 Klein Base 4B |
| 학습 도구 | ai-toolkit (CUDA) | mflux (MLX, Mac 네이티브) |
| 학습 설정 | `config/ai_toolkit_config.yaml` | `config/train.json` |
| 학습 환경 | RunPod/Colab | 로컬 Mac M4 Pro |
| 배포 | 수동 | Notebook에서 자동화 |

## What Stays

- `scripts/preprocess_images.py` — 그대로 사용
- `scripts/caption_images.py` — 그대로 사용 (출력 디렉토리만 mflux용으로 변경)
- `scripts/evaluate_lora.py` — 모델명만 업데이트 필요
- `validation/test_prompts.json` — 그대로 사용

## Notebook Structure (`notebooks/pipeline.ipynb`)

1. **Setup** — 의존성 확인, 환경변수 로드 (.env)
2. **Data Prep** — 이미지 전처리 + 캡션 생성 + mflux 디렉토리로 합치기
3. **Train Config** — train.json 생성/편집
4. **Train** — mflux-train 실행 (subprocess)
5. **Extract** — 체크포인트 ZIP에서 safetensors 추출
6. **Local Test** — mflux-generate로 로컬 추론 테스트
7. **Upload** — HuggingFace Hub 업로드
8. **Register** — Runware Model Upload
9. **Infer** — Runware 추론 테스트

## Project Structure (Updated)

```
story-shorts-character-lora/
├── notebooks/
│   └── pipeline.ipynb          # NEW: 전체 파이프라인 (기존 test_pipeline.ipynb 교체)
├── config/
│   ├── train.json              # NEW: mflux 학습 설정
│   └── ai_toolkit_config.yaml  # KEEP: 참고용
├── scripts/                    # KEEP: 기존 스크립트 유지
├── datasets/                   # KEEP: 기존 구조 유지
├── models/                     # KEEP
├── validation/                 # KEEP
├── docs/                       # KEEP
├── .env                        # NEW: HF_TOKEN, RUNWARE_API_KEY
├── .gitignore                  # UPDATE: .env 추가
├── requirements.txt            # UPDATE: mflux 추가
└── README.md                   # UPDATE: FLUX.2 + mflux 반영
```
