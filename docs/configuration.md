# Configuration Reference

## Overview

All pipeline settings are centralized in [`configs/config.yaml`](../configs/config.yaml). This document describes every configuration option.

## Project

```yaml
project:
  name: "armenian-video-dubbing"
  version: "0.1.0"
  root: "."
  device: "cuda"    # cuda, cpu, mps
  dtype: "float16"  # float16, bfloat16, float32
  seed: 42
```

| Key | Type | Description |
|-----|------|-------------|
| `device` | string | Compute device. `cuda` (NVIDIA GPU), `cpu`, or `mps` (Apple Silicon) |
| `dtype` | string | Default tensor precision |
| `seed` | int | Random seed for reproducibility |

## ASR

```yaml
asr:
  backend: "whisper"    # whisper or nemo
  whisper:
    model: "large-v3"
    language: "hy"
    beam_size: 5
    vad_filter: true
    word_timestamps: true
    batch_size: 16
```

| Key | Type | Description |
|-----|------|-------------|
| `backend` | string | ASR engine: `whisper` or `nemo` |
| `model` | string | Whisper model size |
| `language` | string | Target language code |
| `beam_size` | int | Beam search width |
| `vad_filter` | bool | Voice Activity Detection filtering |
| `word_timestamps` | bool | Enable word-level timestamps |
| `batch_size` | int | Inference batch size |

## Translation

```yaml
translation:
  model: "facebook/seamless-m4t-v2-large"
  source_lang: "eng"
  target_lang: "hye"
  dialect: "eastern"    # eastern or western
  max_length: 512
  num_beams: 5
```

| Key | Type | Description |
|-----|------|-------------|
| `dialect` | string | `eastern` (hye) or `western` (hyw) |
| `source_lang` | string | ISO 639-3 source language |
| `target_lang` | string | ISO 639-3 target language |
| `max_length` | int | Maximum sequence length |
| `num_beams` | int | Beam search width |

## TTS

```yaml
tts:
  backend: "fish-speech"    # fish-speech or cosyvoice
  fish_speech:
    sample_rate: 44100
    max_new_tokens: 2048
    top_p: 0.7
    temperature: 0.7
    repetition_penalty: 1.2
    emotion_tags: true
    reference_audio_seconds: 10
    chunk_length: 200
```

| Key | Type | Description |
|-----|------|-------------|
| `backend` | string | TTS engine |
| `emotion_tags` | bool | Enable emotion-aware synthesis |
| `reference_audio_seconds` | int | Duration of voice reference clip |
| `chunk_length` | int | Characters per synthesis chunk |
| `temperature` | float | Sampling temperature (creativity) |
| `top_p` | float | Nucleus sampling threshold |

## Lip-Sync

```yaml
lipsync:
  model: "musetalk"
  fps: 25
  face_det_batch_size: 8
  bbox_shift: 0
  batch_size: 16
  use_float16: true
  face_enhancement: true
```

| Key | Type | Description |
|-----|------|-------------|
| `model` | string | Lip-sync model (`musetalk`) |
| `fps` | int | Video frame rate |
| `bbox_shift` | int | Vertical face bounding box adjustment |
| `face_enhancement` | bool | Enable CodeFormer post-processing |

## Audio Post-Processing

```yaml
audio:
  demucs:
    model: "htdemucs_ft"
    segment: 7.8
    overlap: 0.25
    shifts: 1
  sample_rate: 44100
  loudness_target: -14.0    # LUFS
  crossfade_ms: 50
```

| Key | Type | Description |
|-----|------|-------------|
| `demucs.model` | string | Demucs variant for source separation |
| `loudness_target` | float | Target loudness in LUFS |
| `crossfade_ms` | int | Crossfade duration between segments |

## Duration Matching

```yaml
timing:
  max_stretch_ratio: 1.25
  min_compress_ratio: 0.80
  silence_threshold_db: -40
  min_pause_ms: 150
  method: "rubberband"
```

| Key | Type | Description |
|-----|------|-------------|
| `max_stretch_ratio` | float | Maximum stretching factor |
| `min_compress_ratio` | float | Maximum compression factor |
| `method` | string | Time-stretch algorithm (`rubberband` or `wsola`) |

## Inference

```yaml
inference:
  batch_video_max_concurrent: 2
  gpu_memory_fraction: 0.9
  enable_quantization: true
  quantization_bits: 4
```

| Key | Type | Description |
|-----|------|-------------|
| `enable_quantization` | bool | Enable model quantization |
| `quantization_bits` | int | Quantization precision (4 or 8) |
| `gpu_memory_fraction` | float | Maximum GPU memory usage |

## Ethics

```yaml
ethics:
  add_watermark: true
  watermark_text: "AI-Dubbed"
  watermark_opacity: 0.3
  consent_required: true
  consent_log_path: "logs/voice_consent.json"
```

| Key | Type | Description |
|-----|------|-------------|
| `add_watermark` | bool | Overlay AI watermark on output |
| `consent_required` | bool | Require voice consent logging |
| `consent_log_path` | string | Path to consent log file |

## Environment Variables

Settings can also be controlled via environment variables (see `.env.example`):

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace access token |
| `ARMTTS_API_KEY` | API authentication key |
| `ARMTTS_CONFIG` | Path to config file |
| `CUDA_VISIBLE_DEVICES` | GPU device selection |
| `WANDB_API_KEY` | Weights & Biases tracking |
