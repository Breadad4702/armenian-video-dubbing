# Codebase Analysis — Armenian Video Dubbing AI
## Current Status: Phase 0 & Phase 1 Complete

### Metrics
- **Total code**: ~4,100 lines of production Python + shell
- **Configuration system**: Centralized YAML with environment variable override
- **Modules**: 11 (utils, data_collection, verification)
- **External dependencies**: 25+ (Whisper, Transformers, LoRA, Demucs, etc.)
- **Data capacity**: 5,000-8,000+ hours target (YouTube + Common Voice)

---

## Architecture Overview

### Phase 0: Environment Setup ✅
**Status**: Complete

**Deliverables**:
- Conda environment (`armtts`) with pinned versions (PyTorch 2.5.1, CUDA 12.4)
- Docker multi-stage image (15GB) with Gradio/API/Label Studio services
- 5 external repos cloned (MuseTalk, Fish-Speech, CosyVoice, CodeFormer, Demucs)
- 6 pre-trained model weights downloaded (~20GB):
  - Whisper large-v3 (3.1GB)
  - SeamlessM4T v2 Large (9.5GB)
  - Fish-Speech S2 Pro (4GB)
  - MuseTalk (2GB)
  - Demucs htdemucs_ft (320MB)
  - CodeFormer (800MB)

**Key Files**:
- `configs/config.yaml` — Master configuration (ASR, TTS, lip-sync, training, eval)
- `src/utils/config.py` — Config loader with dot-access and env var override
- `src/utils/logger.py` — Loguru setup with rotating files + error logging
- `src/utils/helpers.py` — Audio I/O, GPU management, time-stretching, consent logging
- `scripts/setup_environment.sh` — One-shot setup (bash)
- `scripts/verify_setup.py` — 40+ smoke tests

---

### Phase 1: Data Collection & Preparation ✅
**Status**: Complete — 3,593 lines across 8 scripts

**Data Pipeline** (13-step orchestrator):
1. **Common Voice hy-AM** (276 lines)
   - Downloads from HuggingFace
   - Splits: train / validation / test
   - Extracts LM corpus for statistical filtering
   - Output: ~100-300 hours depending on version

2. **YouTube Crawl** (910 lines)
   - Search: 45+ Armenian queries + channel crawling
   - Download: multithreaded audio extraction (yt-dlp)
   - VAD Segmentation: WebRTC VAD → utterance-level clips
   - SNR Filter: energy-based signal-to-noise ratio
   - Target: 5,000-8,000+ hours

3. **Bootstrap ASR** (667 lines) — Replicates "Scaling Armenian ASR" paper
   - Whisper large-v3 transcription (batch mode, 16 HF float16)
   - Language ID filter (keep only Armenian, >70% confidence)
   - Statistical n-gram LM filter (detect hallucination loops, perplexity <500)
   - Quality bucketing: **gold** / **silver** / **bronze** / **reject**

4. **Common Voice Processing** (276 lines)
   - Dataset download + disk caching
   - Manifest generation (JSONL format)
   - Clean text extraction for LM training
   - Output: `data/common_voice/manifests/{train,validation,test}.jsonl`

5. **TTS Studio Data** (394 lines)
   - Speaker-level processing
   - Silence detection + segmentation
   - Emotion/expression tagging framework
   - Output: `data/tts_studio/processed/{speaker}/{segment_id}.wav`

6. **Lip-Sync Data** (366 lines)
   - HDTF dataset download (talking-face benchmark)
   - Armenian YouTube talking-head crawl (200+ videos)
   - MediaPipe face detection + cropping
   - Face ratio validation (min 30% faces visible)
   - Output: `data/lipsync_hdtf/lipsync_manifest.jsonl`

7. **Label Studio Setup** (430 lines)
   - Project creation + task templates
   - JSONL import/export
   - Annotation guide (transcription correction, quality rating, dialect)
   - Human validation workflow

8. **Dataset Organization** (377 lines)
   - Merge all sources (Common Voice + YouTube + TTS)
   - Deduplication (file hash + text similarity)
   - Speaker-aware splits (speakers never leak across train/val/test)
   - Output: `data/splits/{train,val,test,tts_train}.jsonl`

**Output Structure**:
```
data/
├── common_voice/manifests/          # ✅ Common Voice processed
├── youtube_crawl/quality_buckets/   # ✅ Bootstrap ASR results
├── tts_studio/processed/            # ✅ Studio segments (if available)
├── lipsync_hdtf/                    # ✅ Face video manifests
└── splits/
    ├── train.jsonl           (~90% of data)
    ├── val.jsonl            (~5%)
    ├── test.jsonl           (~5%)
    └── tts_train.jsonl      (gold-only, SNR>20dB, 2-15s)
```

**Data Quality Pipeline**:
- Armenian char ratio: >70%
- Whisper language prob: >70%
- LM perplexity: <500 (detects loops/gibberish)
- SNR: >10dB (noisy segments filtered)
- Duration: 1-30s (utterance-level)
- No repetition loops: Max 3x repeated phrase

---

## Implementation Quality

### Code Standards ✅
- **Type hints**: Partial (utils fully typed, scripts mostly typed)
- **Error handling**: Comprehensive try-catch with loguru logging
- **Reproducibility**: All configs saved to JSON/YAML, seeds fixed
- **Performance**: Multithreading (YouTube download), batch processing (ASR)
- **Scalability**: Streaming manifest files (JSONL), disk-based not memory

### Testing & Verification ✅
- `scripts/verify_setup.py` — 40+ package checks
- `scripts/data_collection/run_phase1.sh` — 13-step orchestrator with --dry-run

### Documentation ✅
- Docstrings on all major functions
- Usage examples in all scripts
- Config inline comments
- Phase orchestrator with step descriptions

---

## Ready for Phase 2

### Phase 2: Model Fine-Tuning (Next)
**Scope**: 3 models × 3-4 scripts each = ~12 training scripts

1. **ASR Fine-Tuning** (Whisper large-v3 + LoRA)
   - LoRA config: r=32, alpha=64, dropout=0.05
   - Training: 30 epochs, lr=1e-4, batch=16, accumulation=4
   - Evaluation: WER on test set (target: <8%)
   - Expected improvement: +2-3% WER reduction from base model

2. **TTS Fine-Tuning** (Fish-Speech S2 Pro + LoRA)
   - LoRA config: r=64, alpha=128, dropout=0.1
   - Training: 100 epochs, lr=5e-5, batch=8, accumulation=8
   - Emotion tagging: <happy>, <sad>, <angry>, <neutral>
   - Prosody conditioning: pitch/energy from reference speaker
   - Evaluation: MOS (mean opinion score) on synthetic samples

3. **Translation Optimization** (SeamlessM4T v2)
   - Lightweight fine-tune on Armenian-specific terminology
   - Or: Use as-is with fixed timing (no fine-tune needed — very good on hye)
   - Evaluation: COMET score vs English reference

### Key Hyperparameters (from config.yaml)

**ASR Training**:
```yaml
epochs: 30
learning_rate: 1.0e-4
warmup_steps: 500
batch_size: 16
gradient_accumulation: 4
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
```

**TTS Training**:
```yaml
epochs: 100
learning_rate: 5.0e-5
warmup_steps: 1000
batch_size: 8
gradient_accumulation: 8
lora_r: 64
lora_alpha: 128
lora_dropout: 0.1
```

---

## Next Immediate Actions

1. **Build ASR fine-tuning pipeline**
   - HuggingFace Trainer setup
   - Data collator for Whisper
   - Evaluation metrics (WER)
   - LoRA adapter integration

2. **Build TTS fine-tuning pipeline**
   - Fish-Speech Docker inference check
   - SFT (supervised fine-tuning) setup
   - Emotion token templating
   - Reference speaker prosody extraction

3. **Build training utilities**
   - Manifest to HF dataset conversion
   - Checkpoint management (save best by WER/MOS)
   - Learning curve plotting
   - EMA (exponential moving average) for model averaging

4. **Build evaluation suite**
   - WER computation (jiwer)
   - Speaker similarity (resemblyzer)
   - Prosody quality (pitch/energy analysis)
   - Lip-sync error (LSE-C/D metrics)

---

## Risk Mitigation

⚠️ **Known Challenges**:
1. **Fish-Speech S2 Pro** — Released March 10, 2026; may need careful LoRA adaptation
2. **Armenian morphology** — Complex case system; need morphology-aware loss weighting
3. **YouTube data noise** — Bootstrap ASR may have <8% WER already; diminishing returns after ~1000h
4. **Duration matching** — Lip-sync requires perfect timing; rubberband time-stretch critical
5. **Limited VRAM** — 4-bit quantization needed for 48GB training on RTX 4090

✅ **Mitigations**:
- Start with Common Voice (100-300h, clean) → validate pipeline → scale to YouTube
- Use LoRA (reduces trainable params by 99.5%)
- Multi-GPU distributed training (DataParallel or FSDP)
- Gradient accumulation for effective larger batch sizes

---

## File Tree Summary

```
armenian-video-dubbing/
├── configs/
│   └── config.yaml              ✅ [Master config — all hyperparameters]
├── scripts/
│   ├── setup_environment.sh     ✅ [Phase 0]
│   ├── verify_setup.py          ✅ [Phase 0]
│   └── data_collection/         ✅ [Phase 1]
│       ├── (8 scripts, 3593 lines)
│       └── run_phase1.sh        ✅ [13-step orchestrator]
├── src/
│   ├── utils/                   ✅ [Config, logger, helpers]
│   ├── asr/                     ⏳ [Phase 2]
│   ├── tts/                     ⏳ [Phase 2]
│   ├── translation/             ⏳ [Phase 2]
│   ├── lipsync/                 ⏳ [Phase 3]
│   ├── postprocessing/          ⏳ [Phase 3]
│   └── pipeline/                ⏳ [Phase 3]
├── data/
│   ├── splits/                  ⏳ [From Phase 1]
│   └── ... (models, outputs, etc.)
└── notebooks/                   ⏳ [Phase 2 training notebooks]
```

---

## Success Criteria — Phase 2 Complete

- [ ] ASR trained to <8% WER (validated on Common Voice)
- [ ] TTS produces MOS >4.2 naturalness (measured by synthetic test set)
- [ ] Speaker similarity >0.85 cosine (voice clone fidelity)
- [ ] All models saved as HuggingFace checkpoints + ONNX exports
- [ ] Training logs + learning curves saved
- [ ] Evaluation metrics reported per split (Common Voice test = public benchmark)
