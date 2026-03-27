<div align="center">

# Armenian Video Dubbing AI

**World-class open-source video dubbing system for the Armenian language**

Automatically dub any video into Eastern or Western Armenian with voice cloning, lip-sync, and emotion preservation.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)

[Quick Start](#quick-start) | [Architecture](#architecture) | [Documentation](docs/) | [API Reference](docs/api.md) | [Contributing](CONTRIBUTING.md)

</div>

---

## Overview

Armenian Video Dubbing AI is a complete end-to-end pipeline that transforms English video content into naturally dubbed Armenian audio+video. It combines state-of-the-art models across five domains:

| Stage | Model | Purpose |
|-------|-------|---------|
| **ASR** | Whisper large-v3 + LoRA | Speech recognition with word-level timestamps |
| **Translation** | SeamlessM4T v2 Large | English to Armenian text translation |
| **TTS** | Fish-Speech S2 Pro / edge-tts | Voice synthesis with speaker cloning |
| **Lip-Sync** | MuseTalk v1.5+ | Real-time lip movement synchronization |
| **Post-Processing** | Demucs + pyloudnorm + FFmpeg | Audio separation, normalization, mixing |

### Key Features

- **Dialect Support** — Eastern Armenian (hye) and Western Armenian (hyw)
- **Voice Cloning** — Clone any speaker's voice from a 10-second reference clip
- **Emotion Preservation** — SSML prosody tags (rate, pitch, volume) per detected emotion
- **4-bit Quantization** — Run on consumer GPUs with BitsAndBytes NF4
- **Duration Matching** — Rubberband time-stretching to match original segment timing
- **Background Audio** — Demucs source separation preserves music and SFX
- **Ethical Safeguards** — AI watermark overlay and voice consent logging
- **Production Ready** — Docker, FastAPI, Gradio UI, nginx, Prometheus metrics

---

## Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.4+ (recommended: RTX 4090 / A100)
- FFmpeg, rubberband-cli
- ~16 GB VRAM (with 4-bit quantization) or ~40 GB (full precision)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/armenian-video-dubbing.git
cd armenian-video-dubbing

# Option 1: Conda environment (recommended)
bash scripts/setup_environment.sh

# Option 2: pip install
pip install -e .

# Option 3: Docker
docker compose up -d
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env with your HuggingFace token and other settings
```

### Dub a Video

```bash
# CLI
python -m src.pipeline input.mp4 --output dubbed.mp4 --dialect eastern --emotion neutral

# Or use the Makefile shortcut
make dub VIDEO=input.mp4

# Python API
from src.pipeline import DubbingPipeline

pipeline = DubbingPipeline()
result = pipeline.dub_video(
    video_path="input.mp4",
    reference_speaker_audio="speaker.wav",
    emotion="neutral",
    output_path="dubbed.mp4"
)
```

---

## Architecture

```
Input Video (.mp4)
     │
     ├──► 1. Extract Audio (FFmpeg)
     │
     ├──► 2. ASR — Whisper large-v3 + LoRA
     │         └─► Timestamped segments
     │
     ├──► 3. Translate — SeamlessM4T v2 (eng → hye/hyw)
     │
     ├──► 4. TTS — Fish-Speech S2 Pro (voice clone)
     │         └─► edge-tts fallback with SSML prosody
     │
     ├──► 5. Duration Match — Rubberband time-stretch
     │
     ├──► 6. Audio Post-Processing
     │         ├─► Demucs source separation (vocals vs. SFX)
     │         ├─► Spectral gate denoising
     │         └─► Loudness normalization (–14 LUFS)
     │
     ├──► 7. Lip-Sync — MuseTalk (optional)
     │
     └──► 8. Final Mix — FFmpeg encode + watermark
              └─► Output: dubbed.mp4
```

---

## Project Structure

```
armenian-video-dubbing/
│
├── src/                          # Core source code
│   ├── inference.py              # 5 inference modules (ASR, Translation, TTS, LipSync, PostProc)
│   ├── pipeline.py               # 8-step dubbing orchestrator + CLI
│   ├── training_utils.py         # Training utilities
│   ├── api/
│   │   └── fastapi_server.py     # Production REST API (auth, Prometheus)
│   ├── ui/
│   │   └── gradio_app.py         # Web interface
│   └── utils/                    # Config loader, helpers, logging
│
├── scripts/
│   ├── data_collection/          # YouTube crawl, Common Voice, data prep
│   ├── training/                 # ASR + TTS fine-tuning scripts
│   ├── evaluation/               # Metrics, human eval, regression testing
│   │   ├── metrics/              # WER, MOS, speaker similarity, lip-sync
│   │   ├── human_eval/           # MOS protocol, A/B testing
│   │   └── regression/           # Baseline comparison, weak-spot analysis
│   ├── inference/                # Batch processing
│   └── deployment/               # Cloud deploy (RunPod, AWS, GCP), cost estimation
│
├── configs/
│   ├── config.yaml               # Master configuration
│   ├── crawl_config.yaml         # YouTube crawl settings
│   └── environment.yaml          # Conda environment spec
│
├── tests/                        # Test suite
├── docker/                       # nginx config, Docker requirements
├── data/                         # Dataset directories (gitignored)
├── models/                       # Model checkpoints (gitignored)
├── outputs/                      # Generated outputs (gitignored)
│
├── Dockerfile                    # Multi-stage CUDA build
├── docker-compose.yaml           # Production stack (nginx + Gradio + API)
├── Makefile                      # 20+ automation targets
├── pyproject.toml                # Package config (setuptools)
└── .env.example                  # Environment variable template
```

---

## Usage

### Web Interface (Gradio)

```bash
make web
# Opens at http://localhost:7860
```

### REST API (FastAPI)

```bash
make api
# Available at http://localhost:8000
# Docs: http://localhost:8000/docs
```

See [API Documentation](docs/api.md) for endpoints and examples.

### Docker (Production)

```bash
# Full stack: nginx + Gradio + API
docker compose up -d

# GPU services only
docker compose up -d gradio api

# With Label Studio for annotation
docker compose --profile dev up -d
```

### Batch Processing

```bash
python scripts/inference/batch_process.py --input videos.csv
```

---

## Configuration

All settings are in [`configs/config.yaml`](configs/config.yaml). Key sections:

| Section | Description |
|---------|-------------|
| `project` | Device (cuda/cpu/mps), dtype, seed |
| `asr` | Whisper model, beam size, VAD, quantization |
| `translation` | SeamlessM4T settings, dialect selection |
| `tts` | Fish-Speech / CosyVoice, voice cloning params |
| `lipsync` | MuseTalk settings, face detection |
| `audio` | Demucs, loudness target, sample rate |
| `timing` | Duration matching, stretch/compress ratios |
| `training` | Hyperparameters for ASR and TTS fine-tuning |
| `evaluation` | Quality thresholds (WER, MOS, similarity) |
| `ethics` | Watermark, consent logging |

---

## Quality Targets

| Metric | Target | Description |
|--------|--------|-------------|
| WER | < 8% | Word Error Rate (ASR accuracy) |
| MOS | > 4.6 | Mean Opinion Score (naturalness) |
| Speaker Similarity | > 0.85 | Voice clone fidelity |
| LSE-C | < 1.8 | Lip-sync confidence error |
| LSE-D | < 1.8 | Lip-sync distance error |
| COMET | > 0.85 | Translation quality |

---

## Development

```bash
# Run tests
make test

# Lint
make lint

# Auto-fix lint issues
make lint-fix

# Format code
make format
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture Guide](docs/architecture.md) | System design and data flow |
| [API Reference](docs/api.md) | REST API endpoints and usage |
| [Deployment Guide](docs/deployment.md) | Docker, cloud, and RunPod setup |
| [Training Guide](docs/training.md) | Fine-tuning ASR and TTS models |
| [Evaluation Guide](docs/evaluation.md) | Metrics, human eval, regression |
| [Configuration Reference](docs/configuration.md) | All config options explained |

---

## Makefile Targets

Run `make help` to see all available targets:

```
api                  Launch FastAPI locally
batch                Batch process (make batch MANIFEST=videos.csv)
build                Build Docker image
clean                Remove temp files and caches
deploy-cloud         Deploy to cloud (auto-detect provider)
deploy-runpod        Deploy to RunPod
dub                  Dub a video (make dub VIDEO=input.mp4 EMOTION=neutral)
evaluate             Run evaluation suite
install              Install dependencies locally (conda env)
lint                 Lint code with ruff
lint-fix             Auto-fix lint issues
logs                 Follow logs from all services
run                  Start all services (Gradio + API + nginx)
run-dev              Start all services including Label Studio
stop                 Stop all services
test                 Run tests
train-asr            Train ASR model
train-tts            Train TTS model
web                  Launch Gradio UI locally
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## Acknowledgments

Built with:
- [OpenAI Whisper](https://github.com/openai/whisper) — ASR backbone
- [Meta SeamlessM4T](https://github.com/facebookresearch/seamless_communication) — Translation
- [Fish-Speech](https://github.com/fishaudio/fish-speech) — TTS and voice cloning
- [MuseTalk](https://github.com/TMElyralab/MuseTalk) — Lip synchronization
- [Demucs](https://github.com/facebookresearch/demucs) — Audio source separation
- [edge-tts](https://github.com/rany2/edge-tts) — Fallback TTS with SSML
