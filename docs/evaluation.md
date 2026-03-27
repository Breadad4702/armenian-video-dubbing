# Evaluation Guide

## Overview

The evaluation framework measures dubbing quality across 6 dimensions with automated metrics, human evaluation protocols, and regression testing.

## Quick Start

```bash
# Run full evaluation
python scripts/evaluation/evaluate_full.py

# Run specific metric
python scripts/evaluation/metrics/wer_metrics.py
```

## Metrics

### 1. Word Error Rate (WER)

Measures ASR transcription accuracy.

- **Target**: WER < 8%
- **Module**: `scripts/evaluation/metrics/wer_metrics.py`
- **Method**: Compare transcribed text against ground truth using `jiwer`

### 2. Mean Opinion Score (MOS) Proxy

Automated proxy for perceived speech naturalness.

- **Target**: MOS > 4.6
- **Module**: `scripts/evaluation/metrics/mos_proxy_metrics.py`
- **Method**: Neural MOS predictor on synthesized speech

### 3. Speaker Similarity

Measures voice cloning fidelity between reference and synthesized speech.

- **Target**: > 0.85
- **Module**: `scripts/evaluation/metrics/speaker_similarity.py`
- **Method**: Cosine similarity of speaker embeddings

### 4. Lip-Sync Quality

Measures synchronization between audio and lip movements.

- **Target**: LSE-C < 1.8, LSE-D < 1.8
- **Module**: `scripts/evaluation/metrics/lipsync_metrics.py`
- **Method**: Lip Sync Error (confidence and distance)

### 5. Translation Quality

Measures translation accuracy from English to Armenian.

- **Target**: COMET > 0.85
- **Module**: `scripts/evaluation/metrics/translation_metrics.py`
- **Method**: COMET score and BLEU

### 6. Performance

Measures inference speed and resource usage.

- **Module**: `scripts/evaluation/metrics/performance_metrics.py`
- **Method**: End-to-end latency, GPU memory, throughput

## Human Evaluation

### Protocol

```bash
python scripts/evaluation/human_eval/protocol.py
```

Implements:
- **MOS Testing** — 5-point Likert scale for naturalness
- **A/B Testing** — Preference comparison between two outputs
- **LabelStudio Integration** — Annotation interface setup

### Setup

1. Start Label Studio: `docker compose --profile dev up label-studio`
2. Run protocol setup: `python scripts/evaluation/human_eval/protocol.py --setup`
3. Import evaluation samples
4. Collect annotations
5. Compute inter-annotator agreement

## Regression Testing

```bash
python scripts/evaluation/regression/regression_test.py
```

Features:
- **Baseline comparison** — Compare current run against stored baselines
- **Weak-spot analysis** — Identify specific failure patterns
- **History tracking** — Track metric trends across iterations
- **Alerts** — Flag regressions exceeding thresholds

## Configuration

Evaluation thresholds in `configs/config.yaml`:

```yaml
evaluation:
  wer_threshold: 0.08
  mos_threshold: 4.6
  speaker_similarity_threshold: 0.85
  lse_c_threshold: 1.8
  lse_d_threshold: 1.8
  comet_threshold: 0.85
```

Evaluation-specific settings in `scripts/evaluation/eval_config.yaml`.

## Directory Structure

```
scripts/evaluation/
├── evaluate_full.py          # Orchestrator
├── eval_config.yaml          # Evaluation config
├── metrics/
│   ├── wer_metrics.py        # WER computation
│   ├── mos_proxy_metrics.py  # MOS proxy
│   ├── speaker_similarity.py # Voice similarity
│   ├── lipsync_metrics.py    # Lip-sync error
│   ├── translation_metrics.py# COMET/BLEU
│   └── performance_metrics.py# Speed/memory
├── human_eval/
│   └── protocol.py           # Human eval framework
├── regression/
│   └── regression_test.py    # Regression testing
├── reporting/
└── test_data/
```
