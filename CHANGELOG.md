# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2024-12-01

### Added

- **Phase 0**: Environment setup (conda, Docker, external model repos)
- **Phase 1**: Data collection pipeline (YouTube crawl, Common Voice processing, LabelStudio annotation, train/val/test splits)
- **Phase 2**: Model fine-tuning (Whisper large-v3 ASR with LoRA, Fish-Speech TTS, SeamlessM4T translation evaluation)
- **Phase 3**: Complete inference pipeline
  - ASR with 4-bit quantization (BitsAndBytes NF4)
  - SeamlessM4T v2 Large text-to-text translation (eng → hye/hyw)
  - Fish-Speech S2 Pro TTS with voice cloning (edge-tts fallback with SSML prosody)
  - MuseTalk lip-sync with graceful fallback
  - Demucs source separation, spectral gate denoising, loudness normalization
  - 8-step dubbing orchestrator with CLI
- **Phase 4**: Evaluation and QC
  - 6 metric modules (WER, MOS proxy, speaker similarity, lip-sync, translation, performance)
  - Human evaluation protocol (MOS, A/B testing, LabelStudio config)
  - Regression testing with baseline comparison and weak-spot analysis
- **Phase 5**: Production deployment
  - Docker multi-stage build with CUDA 12.4
  - Docker Compose stack (nginx + Gradio + FastAPI + Label Studio)
  - FastAPI with API key auth and Prometheus metrics
  - Gradio web UI with dialect selector
  - Cloud deployment scripts (RunPod, AWS, GCP)
  - CI/CD pipeline (lint, test, Docker build)
  - Cost estimation calculator
- **Features**:
  - Eastern/Western Armenian dialect selection
  - SSML emotion prosody (rate/pitch/volume per emotion)
  - Voice consent logging
  - AI watermark overlay
  - Duration matching via Rubberband time-stretching
  - Background SFX isolation (Demucs) and re-mixing
  - Batch video processing
