# Architecture Guide

## System Overview

The Armenian Video Dubbing AI is an 8-step pipeline that transforms English video into Armenian-dubbed output with voice cloning, lip synchronization, and emotion preservation.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: English Video                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │  1. Extract Audio   │  FFmpeg
                │     from Video      │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  2. ASR (Whisper)   │  Whisper large-v3 + LoRA
                │  Transcribe Speech  │  4-bit quantized (NF4)
                └──────────┬──────────┘
                           │
                  Timestamped segments
                           │
                ┌──────────▼──────────┐
                │  3. Translate       │  SeamlessM4T v2 Large
                │  eng → hye / hyw    │  Eastern or Western Armenian
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  4. TTS + Voice     │  Fish-Speech S2 Pro
                │  Clone Synthesis    │  edge-tts (SSML) fallback
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  5. Duration Match  │  Rubberband time-stretch
                │  Align to Original  │  0.80x – 1.25x range
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  6. Audio Post-Proc │  Demucs source separation
                │  Denoise + Normalize│  Spectral gate + pyloudnorm
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  7. Lip-Sync        │  MuseTalk v1.5+
                │  (Optional)         │  Graceful skip if unavailable
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │  8. Final Mix       │  FFmpeg encode
                │  Audio + Video      │  AI watermark overlay
                └──────────┬──────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  Output: Armenian Dubbed Video               │
└─────────────────────────────────────────────────────────────┘
```

## Model Stack

### ASR — Automatic Speech Recognition

- **Model**: OpenAI Whisper large-v3
- **Fine-tuning**: LoRA adapters (r=32, alpha=64) on Armenian speech data
- **Quantization**: BitsAndBytes NF4 (4-bit) for reduced VRAM
- **Output**: Segment-level transcriptions with word-level timestamps
- **Source**: `src/inference.py` → `ASRInference`

### Translation

- **Model**: Meta SeamlessM4T v2 Large
- **Mode**: Text-to-text (not speech-to-speech)
- **Languages**: English (eng) → Eastern Armenian (hye) or Western Armenian (hyw)
- **Source**: `src/inference.py` → `TranslationInference`

### TTS — Text-to-Speech

- **Primary**: Fish-Speech S2 Pro with speaker voice cloning
- **Fallback chain**: Fish-Speech → edge-tts (SSML prosody) → gTTS
- **Features**: Emotion tags, 10-second reference cloning, chunk-based synthesis
- **Source**: `src/inference.py` → `TTSInference`

### Lip-Sync

- **Model**: MuseTalk v1.5+ (real-time lip movement inpainting)
- **Behavior**: Gracefully skipped if MuseTalk is not installed
- **Post-processing**: Optional CodeFormer face enhancement
- **Source**: `src/inference.py` → `LipSyncInference`

### Audio Post-Processing

- **Source separation**: Demucs (htdemucs_ft) — isolates vocals from background
- **Denoising**: Spectral gate noise removal
- **Normalization**: pyloudnorm to –14 LUFS
- **Mixing**: Crossfade segments, re-combine with background audio/SFX
- **Source**: `src/inference.py` → `AudioPostProcessor`

## Key Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/inference.py` | ~985 | All 5 inference modules |
| `src/pipeline.py` | ~543 | 8-step dubbing orchestrator |
| `src/api/fastapi_server.py` | ~463 | Production REST API |
| `src/ui/gradio_app.py` | ~229 | Gradio web interface |
| `src/utils/helpers.py` | ~250 | Audio/video utilities |
| `src/utils/config.py` | ~90 | YAML config loader |

## GPU Memory Usage

| Configuration | Estimated VRAM |
|---------------|----------------|
| Full precision (fp32) | ~40 GB |
| Half precision (fp16) | ~20 GB |
| 4-bit quantized (NF4) | ~12–16 GB |

Recommended GPU: NVIDIA RTX 4090 (24 GB) with 4-bit quantization.
