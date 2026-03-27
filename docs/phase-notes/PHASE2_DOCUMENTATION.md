# Phase 2: Model Fine-Tuning — Complete Documentation

## Overview

**Phase 2** implements efficient fine-tuning of 3 core models on Armenian data:

1. **ASR (Whisper large-v3)** — Speech recognition to text
2. **TTS (Fish-Speech S2 Pro)** — Text to speech with voice cloning
3. **Translation (SeamlessM4T v2)** — Timing/semantic alignment (pre-trained SOTA)

**Key principle**: Use LoRA (Low-Rank Adaptation) to reduce trainable parameters by 99.5% while maintaining quality.

**Training data sources**:
- **Common Voice hy-AM** (100-300 hours, seed data)
- **YouTube crawl** (5,000-8,000+ hours, scaled data)
- **Merged splits**: train (90%) / val (5%) / test (5%)

**Hardware**: Single GPU (RTX 4090, 24GB VRAM)
- 8-bit quantization for base models
- Gradient accumulation for effective larger batches
- Mixed precision (float16/bfloat16)

---

## Pipeline Architecture

```
Phase 2: Model Fine-Tuning
│
├─ Step 1: ASR Fine-Tuning (Common Voice seed)
│   └─ Output: models/asr/whisper-hy-cv/
│
├─ Step 2: ASR Fine-Tuning (Full merged dataset)
│   └─ Output: models/asr/whisper-hy-full/
│
├─ Step 3: TTS Fine-Tuning (Fish-Speech + emotion/prosody)
│   └─ Output: models/tts/fish-speech-hy/
│
├─ Step 4: Translation Evaluation (SeamlessM4T v2)
│   └─ Output: models/translation/metrics.json
│
├─ Step 5: Comprehensive Evaluation
│   └─ Output: outputs/evaluation_results.json
│
├─ Step 6: TTS Sample Generation
│   └─ Output: outputs/tts_samples/*.wav
│
└─ Step 7: Export to ONNX (inference-optimized)
    └─ Output: models/onnx/{whisper,fish-speech}.onnx
```

---

## Step 1-2: ASR Fine-Tuning (Whisper + LoRA)

### What happens

```python
# Load pre-trained Whisper
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

# Freeze base, add LoRA adapter
model = freeze_base_model(model)
lora_config = LoraConfig(r=32, alpha=64, lora_dropout=0.05)
model = get_peft_model(model, lora_config)

# Result: 1.5B params → only 1.6M params trainable (0.1%)
```

### Configuration (from `configs/config.yaml`)

```yaml
asr:
  whisper:
    model: "large-v3"
    language: "hy"
    beam_size: 5
    vad_filter: true

training:
  asr:
    epochs: 30
    learning_rate: 1.0e-4
    warmup_steps: 500
    batch_size: 16
    gradient_accumulation: 4
    lora_r: 32
    lora_alpha: 64
    lora_dropout: 0.05
    eval_steps: 500
    save_steps: 1000
    fp16: true
```

### Training procedure

1. **Data loading**: JSONL manifests → HuggingFace Dataset
2. **Preprocessing**: Audio WAV → feature extraction (MFCC frames)
3. **Tokenization**: Armenian text → token IDs
4. **Collation**: Pad sequences to variable lengths
5. **Training loop**:
   - Forward: audio features → CTC logits
   - Compute WER loss
   - Backward pass + parameter updates (LoRA only)
   - Gradient accumulation every 4 steps
   - Evaluation every 500 steps
6. **Checkpointing**: Save best by WER on validation set

### Results

**Step 1 (Common Voice)**:
- Train set: ~95,000 hours → ~50K samples
- Expected WER: 6.5-7.5% (base ~8%, so -0.5-1.5% improvement)
- Training time: ~12-16 hours on RTX 4090

**Step 2 (Full merged)**:
- Train set: ~5,500+ hours (Common Voice + filtered YouTube)
- Expected WER: 5.5-6.5% (additional -1% from scale)
- Training time: ~40-60 hours on RTX 4090

### Checkpoints saved

```
models/asr/whisper-hy-full/
├── checkpoints/
│   ├── checkpoint-1000/        # Every 1000 steps
│   ├── checkpoint-2000/
│   └── checkpoint_best/        # Best by WER
├── final_model/                # Final converged model
├── training_results.json       # Metrics: loss, WER, CER
└── logs/                       # TensorBoard logs
```

---

## Step 3: TTS Fine-Tuning (Fish-Speech + Emotion/Prosody)

### What happens

Fish-Speech S2 Pro:
- **VQ-VAE** (frozen): Audio → discrete token codes
- **LM** (fine-tuned): Text tokens + audio codes → decoder
- **Speaker encoder**: Speaker embedding for zero-shot cloning
- **Emotion tags**: `<happy>`, `<sad>`, `<angry>`, etc.

```python
# Fish-Speech components
vq_vae = load_pretrained_vq_vae()  # Frozen
lm = LoRA_wrapped_language_model()   # Fine-tuned
speaker_encoder = load_speaker_encoder()  # Frozen or light fine-tune

# Process training sample
text_tokens = tokenizer.encode(f"<happy> {text}")
audio_codes = vq_vae.encode(reference_audio)
speaker_emb = speaker_encoder(reference_audio)

# Train LM to predict audio_codes from (text_tokens, speaker_emb)
```

### Configuration

```yaml
tts:
  fish_speech:
    model_path: "models/tts/fish-speech-s2-armenian"
    base_model: "fishaudio/fish-speech-s2-pro"
    sample_rate: 44100
    emotion_tags: true
    reference_audio_seconds: 10

training:
  tts:
    epochs: 100
    learning_rate: 5.0e-5
    warmup_steps: 1000
    batch_size: 8
    gradient_accumulation: 8
    lora_r: 64
    lora_alpha: 128
    lora_dropout: 0.1
    eval_steps: 200
```

### Data preparation

1. **Audio loading**: Load WAV at 44.1 kHz (Fish-Speech native)
2. **Emotion tagging**:
   - From metadata: `sample["emotion"]` → `<happy>`, `<sad>`, etc.
   - Default: `<neutral>`
3. **Prosody extraction**:
   - Pitch (F0) via YIN autocorrelation
   - Energy contour via frame-level RMS
   - Stored for reference speaker matching
4. **Speaker embedding**: resemblyzer or WavLM → 256-512d vector

### Training procedure

```python
# Simplified training loop
for epoch in range(100):
    for batch in train_loader:
        # Encode audio codes
        audio_codes = vq_vae.encode(batch["audio"])

        # Tokenize text with emotion
        text_tokens = tokenizer.encode(
            f"{EMOTION_TOKENS[batch['emotion']]} {batch['text']}"
        )

        # Forward pass: predict audio codes
        logits = lm(text_tokens, speaker_embeddings=batch["speaker_emb"])

        # Loss: cross-entropy on audio code prediction
        loss = ce_loss(logits, audio_codes)

        # Backward + optimize (LoRA params only)
        loss.backward()
        optimizer.step()

        # Periodically: synthesize sample + estimate MOS
```

### Results

**Expected metrics**:
- **MOS estimate**: 4.0-4.5 (human perception, 1-5 scale)
  - Base model (no fine-tune): ~3.8
  - After fine-tuning: +0.2-0.7
- **Speaker similarity**: 0.82-0.88 cosine (zero-shot cloning)
- **Prosody match**: Pitch/energy correlation >0.75 with reference

**Training time**: ~60-100 hours on RTX 4090

### Checkpoints

```
models/tts/fish-speech-hy/
├── train_data.json            # Prepared training metadata
├── checkpoints/
│   ├── checkpoint-200/        # Every 200 eval steps
│   └── checkpoint_best/       # Best by MOS estimate
├── final_model/
└── training_metrics.json
```

---

## Step 4: Translation Evaluation (SeamlessM4T v2)

### What happens

**SeamlessM4T v2 Large** (released Oct 2024):
- Pre-trained on 100+ languages including Armenian (hye)
- Encoder: Speech/text → shared embedding
- Decoder: Predicts translation + timing tokens
- **No fine-tuning recommended** — already SOTA

```python
# Load SOTA model
model = AutoModel.from_pretrained("facebook/seamless-m4t-v2-large")

# Inference: English audio → Armenian text + alignment
output = model.generate(
    audio,
    src_lang="eng",
    tgt_lang="hye",  # Eastern Armenian
    return_intermediate_token_ids=True,  # For timing
)
```

### Evaluation

- **Dataset**: Common Voice English test set (100-500 samples)
- **Metrics**: COMET v2 score (neural metric)
  - Compute: ref_text (English) → predicted translation (Armenian)
  - Target: >0.85 COMET
- **Timing extraction**: Intermediate tokens → word-level alignments
- **Result**: Use as-is; no training needed

---

## Step 5: Comprehensive Evaluation

### Metrics computed

| Metric | Model | Target | Method |
|--------|-------|--------|--------|
| WER | ASR | <8% | jiwer on test set |
| CER | ASR | <4% | jiwer (character-level) |
| MOS | TTS | >4.6 | Audio quality metrics |
| Speaker Sim | TTS | >0.85 | Cosine similarity (embeddings) |
| COMET | Translation | >0.85 | Neural translation metric |
| LSE-C | Lip-Sync | <1.8 | Phase 3 (full pipeline) |
| LSE-D | Lip-Sync | <1.8 | Phase 3 (full pipeline) |

### Evaluation script

```bash
python scripts/training/evaluate_all_models.py \
    --test-manifest data/splits/test.jsonl \
    --asr-model models/asr/whisper-hy-full \
    --output-dir outputs/evaluation
```

**Output**: `outputs/evaluation_results.json`
```json
{
  "asr": {
    "wer": 0.075,
    "cer": 0.035,
    "improvement_over_base": 0.025
  },
  "tts": {
    "mos_mean": 4.35,
    "speaker_similarity": 0.86
  },
  "translation": {
    "comet": 0.88,
    "zero_shot": true
  },
  "targets_met": {
    "wer_8pct": true,
    "mos_4_6": false,
    "speaker_sim_85": true
  }
}
```

---

## Step 6-7: Deployment Preparation

### TTS Sample Generation

```bash
python scripts/training/generate_tts_samples.py \
    --model models/tts/fish-speech-hy \
    --output-dir outputs/tts_samples
```

**Generates**:
- 5 reference speakers × 3 emotions × 5 test sentences = 75 samples
- Each: synthesized WAV + metadata (speaker, emotion, text)
- Used for: Human MOS evaluation, speaker verification

### Model Export (ONNX)

```bash
python scripts/training/export_models.py \
    --asr-model models/asr/whisper-hy-full \
    --tts-model models/tts/fish-speech-hy \
    --output-dir models/onnx \
    --quantize  # int8 quantized versions
```

**Outputs**:
```
models/onnx/
├── whisper.onnx           # ASR (int32 context, float32 output)
├── fish-speech.onnx       # TTS (float32)
├── quantized/
│   ├── whisper_int8.onnx  # 3-4x faster inference
│   └── fish-speech_int8.onnx
└── benchmark.json         # Latency: RTX 4090, batch=1
```

**Expected latencies** (per 10-min video):
- ASR: ~15-20 min (1.5x realtime)
- TTS: ~8-12 min (0.8x realtime)
- Total: ~25-30 min ✅ (target: ≤5 min, will optimize in Phase 3)

---

## How to Run Phase 2

### Quick validation (Common Voice only, 2-3 hours)
```bash
python scripts/training/train_asr.py \
    --dataset-type common_voice \
    --output-dir models/asr/whisper-hy-cv \
    --max-train-samples 5000
```

### Full training (merged dataset, 48+ hours)
```bash
bash scripts/training/run_phase2.sh
```

### Resume from checkpoint
```bash
python scripts/training/train_asr.py \
    --resume-from-checkpoint models/asr/whisper-hy-full/checkpoint_best
```

### Evaluation only
```bash
python scripts/training/evaluate_all_models.py \
    --asr-model models/asr/whisper-hy-full \
    --tts-model models/tts/fish-speech-hy
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Out of memory (OOM) | Batch size too large | Reduce `batch_size`, increase `gradient_accumulation` |
| Slow training | GPU underutilized | Increase `num_workers`, check data loading |
| WER not improving | Poor learning rate | Try `lr=2e-4` or `lr=5e-5` |
| TTS sounds robotic | Insufficient prosody | Ensure emotion tags in data, check speaker embeddings |
| Training diverges | Learning rate too high | Reduce to `1e-5`, enable gradient clipping |

---

## Success Criteria ✅

- [x] ASR code complete (train_asr.py)
- [x] TTS code complete (train_tts.py)
- [x] Evaluation suite complete (evaluate_all_models.py)
- [x] Training orchestrator (run_phase2.sh)
- [x] Utility library (training_utils.py)
- [ ] ASR ≤ 8% WER (after full training)
- [ ] TTS ≥ 4.2 MOS (after full training)
- [ ] Speaker similarity ≥ 0.85
- [ ] Models exported to ONNX
- [ ] Evaluation results saved

---

## Next: Phase 3

**Phase 3: End-to-End Inference Pipeline**

- Orchestrate ASR → TTS → Lip-sync → Audio mixing
- Gradio web UI + FastAPI server
- Real-time processing (target: ≤5 min for 10-min video)
- Video I/O, speaker selection, emotion control

