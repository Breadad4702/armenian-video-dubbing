#!/usr/bin/env bash
# ============================================================================
# Phase 1: Full Data Collection Pipeline — Master Orchestrator
# ============================================================================
# Runs all Phase 1 steps in sequence with progress tracking.
#
# Usage:
#   bash scripts/data_collection/run_phase1.sh             # Full pipeline
#   bash scripts/data_collection/run_phase1.sh --step 3    # Start from step 3
#   bash scripts/data_collection/run_phase1.sh --dry-run   # Show steps only
# ============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[STEP $STEP]${NC} $*"; }
ok()    { echo -e "${GREEN}[DONE]${NC}   Step $STEP complete"; }
err()   { echo -e "${RED}[ERROR]${NC} Step $STEP failed"; }

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

START_STEP=${1:-1}
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step) START_STEP="$2"; shift 2;;
        --dry-run) DRY_RUN=true; shift;;
        *) shift;;
    esac
done

echo ""
echo "============================================================"
echo "  Phase 1: Data Collection & Preparation Pipeline"
echo "============================================================"
echo "  Project root: $PROJECT_ROOT"
echo "  Starting from step: $START_STEP"
echo "  Dry run: $DRY_RUN"
echo "============================================================"
echo ""

run_step() {
    STEP=$1
    DESC=$2
    CMD=$3

    if [ "$STEP" -lt "$START_STEP" ]; then
        echo -e "${YELLOW}[SKIP]${NC}  Step $STEP: $DESC"
        return
    fi

    info "$DESC"

    if [ "$DRY_RUN" = true ]; then
        echo "         Would run: $CMD"
        return
    fi

    if eval "$CMD"; then
        ok
    else
        err
        echo -e "${YELLOW}[WARN]${NC}  Continuing despite error..."
    fi
    echo ""
}

# ============================================================================
# Step 1: Download & Process Common Voice hy-AM
# ============================================================================
run_step 1 \
    "Download & process Common Voice hy-AM" \
    "python scripts/data_collection/process_common_voice.py --output-dir data/common_voice"

# ============================================================================
# Step 2: YouTube Search — Find Armenian content
# ============================================================================
run_step 2 \
    "YouTube search — find Armenian content" \
    "python scripts/data_collection/youtube_crawl.py --config configs/crawl_config.yaml --phase search"

# ============================================================================
# Step 3: YouTube Download — Download audio
# ============================================================================
run_step 3 \
    "YouTube download — download audio tracks" \
    "python scripts/data_collection/youtube_crawl.py --config configs/crawl_config.yaml --phase download"

# ============================================================================
# Step 4: VAD Segmentation — Cut into utterances
# ============================================================================
run_step 4 \
    "VAD segmentation — cut audio into utterances" \
    "python scripts/data_collection/youtube_crawl.py --config configs/crawl_config.yaml --phase segment"

# ============================================================================
# Step 5: SNR Filtering — Remove noisy segments
# ============================================================================
run_step 5 \
    "SNR filtering — remove noisy segments" \
    "python scripts/data_collection/youtube_crawl.py --config configs/crawl_config.yaml --phase filter"

# ============================================================================
# Step 6: Bootstrap ASR — Transcribe with Whisper
# ============================================================================
run_step 6 \
    "Bootstrap ASR — transcribe with Whisper large-v3" \
    "python scripts/data_collection/bootstrap_transcribe.py --phase transcribe --input data/youtube_crawl/segments_filtered.jsonl"

# ============================================================================
# Step 7: Language ID Filtering
# ============================================================================
run_step 7 \
    "Language identification filtering" \
    "python scripts/data_collection/bootstrap_transcribe.py --phase langid --input data/youtube_crawl/segments_filtered.jsonl"

# ============================================================================
# Step 8: LM Filtering — Remove hallucinations
# ============================================================================
run_step 8 \
    "Statistical LM filtering — remove hallucinations" \
    "python scripts/data_collection/bootstrap_transcribe.py --phase lm_filter --input data/youtube_crawl/segments_filtered.jsonl --lm-train-data data/common_voice/lm_corpus.txt"

# ============================================================================
# Step 9: Quality Bucketing — gold/silver/bronze
# ============================================================================
run_step 9 \
    "Quality bucketing — assign gold/silver/bronze tiers" \
    "python scripts/data_collection/bootstrap_transcribe.py --phase bucket --input data/youtube_crawl/segments_filtered.jsonl"

# ============================================================================
# Step 10: Label Studio Setup (optional — for human validation)
# ============================================================================
run_step 10 \
    "Label Studio setup + annotation guide" \
    "python scripts/data_collection/labelstudio_setup.py --action guide --output-dir data/youtube_crawl"

# ============================================================================
# Step 11: TTS Studio Data (if available)
# ============================================================================
run_step 11 \
    "TTS studio data processing" \
    "python scripts/data_collection/prepare_tts_data.py --input-dir data/tts_studio --output-dir data/tts_studio/processed"

# ============================================================================
# Step 12: Lip-Sync Data
# ============================================================================
run_step 12 \
    "Lip-sync data preparation (HDTF + Armenian talking heads)" \
    "python scripts/data_collection/prepare_lipsync_data.py --phase all --output-dir data/lipsync_hdtf"

# ============================================================================
# Step 13: Organize & Split Dataset
# ============================================================================
run_step 13 \
    "Organize dataset — merge all sources, create train/val/test splits" \
    "python scripts/data_collection/organize_dataset.py --cv-dir data/common_voice --yt-dir data/youtube_crawl --studio-dir data/tts_studio/processed --output-dir data/splits"

echo ""
echo "============================================================"
echo "  Phase 1: Data Collection Complete!"
echo "============================================================"
echo ""
echo "  Data summary:"
echo "    data/splits/train.jsonl    — ASR training data"
echo "    data/splits/val.jsonl      — Validation data"
echo "    data/splits/test.jsonl     — Held-out test data"
echo "    data/splits/tts_train.jsonl — TTS training subset"
echo "    data/splits/dataset_stats.json — Full statistics"
echo ""
echo "  Next: Phase 2 — Model Fine-Tuning"
echo "============================================================"
