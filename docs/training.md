# Training Guide

## Overview

Fine-tune the ASR and TTS models on Armenian language data to improve dubbing quality.

## Data Preparation

### Phase 1: Data Collection

```bash
# Run the full data collection pipeline
bash scripts/data_collection/run_phase1.sh
```

This runs the following steps:

1. **YouTube Crawl** — Download Armenian speech videos
   ```bash
   python scripts/data_collection/youtube_crawl.py
   ```

2. **Common Voice** — Process Mozilla Common Voice Armenian dataset
   ```bash
   python scripts/data_collection/process_common_voice.py
   ```

3. **Bootstrap Transcribe** — Generate initial transcriptions
   ```bash
   python scripts/data_collection/bootstrap_transcribe.py
   ```

4. **Organize Dataset** — Create train/val/test splits
   ```bash
   python scripts/data_collection/organize_dataset.py
   ```

### Data Format

After preparation, data is organized as:

```
data/
├── splits/
│   ├── train.json    # Training manifest
│   ├── val.json      # Validation manifest
│   └── test.json     # Test manifest
├── processed/        # Processed audio files
└── youtube_crawl/    # Raw crawled data
```

Each manifest entry:
```json
{
  "audio_path": "data/processed/sample_001.wav",
  "text": "Transcription text in Armenian",
  "duration": 5.2,
  "language": "hy"
}
```

---

## ASR Fine-Tuning

Fine-tune Whisper large-v3 with LoRA adapters for Armenian speech recognition.

```bash
# Run training
make train-asr
# or
python scripts/training/train_asr.py
```

### Hyperparameters

Configured in `configs/config.yaml` under `training.asr`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 30 | Training epochs |
| `learning_rate` | 1e-4 | Peak learning rate |
| `warmup_steps` | 500 | LR warmup steps |
| `batch_size` | 16 | Per-device batch size |
| `gradient_accumulation` | 4 | Gradient accumulation steps |
| `lora_r` | 32 | LoRA rank |
| `lora_alpha` | 64 | LoRA alpha |
| `lora_dropout` | 0.05 | LoRA dropout |

### Output

Trained adapter saved to `models/asr/whisper-large-v3-armenian/`.

---

## TTS Fine-Tuning

Fine-tune Fish-Speech S2 Pro for Armenian voice synthesis.

```bash
# Run training
make train-tts
# or
python scripts/training/train_tts.py
```

### Hyperparameters

Configured in `configs/config.yaml` under `training.tts`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Training epochs |
| `learning_rate` | 5e-5 | Peak learning rate |
| `warmup_steps` | 1000 | LR warmup steps |
| `batch_size` | 8 | Per-device batch size |
| `gradient_accumulation` | 8 | Gradient accumulation steps |
| `lora_r` | 64 | LoRA rank |
| `lora_alpha` | 128 | LoRA alpha |

### TTS Data Preparation

```bash
python scripts/data_collection/prepare_tts_data.py
```

### Generate Samples

```bash
python scripts/training/generate_tts_samples.py
```

---

## Model Evaluation

### Run All Evaluations

```bash
make evaluate
# or
python scripts/training/evaluate_all_models.py
```

### Translation Evaluation

```bash
python scripts/training/evaluate_translation.py
```

### Model Export

```bash
python scripts/training/export_models.py
```

Exports optimized models for production inference.

---

## Experiment Tracking

Set up Weights & Biases (optional):

```bash
export WANDB_API_KEY=your-key
export WANDB_PROJECT=armenian-video-dubbing
```

Training scripts will automatically log to W&B if configured.
