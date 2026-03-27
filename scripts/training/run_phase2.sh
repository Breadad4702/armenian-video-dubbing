#!/usr/bin/env bash
# ============================================================================
# Phase 2: Model Fine-Tuning Pipeline — Master Orchestrator
# ============================================================================
# Runs all Phase 2 training steps in sequence
#
# Prerequisites: Phase 1 complete (data collected & organized)
#
# Usage:
#   bash scripts/training/run_phase2.sh             # Full pipeline
#   bash scripts/training/run_phase2.sh --step 3    # Start from step 3 (TTS training)
#   bash scripts/training/run_phase2.sh --skip-tts  # Skip TTS (slow), do ASR only
# ============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[STEP $STEP]${NC} $*"; }
ok()    { echo -e "${GREEN}[DONE]${NC}   Step $STEP complete"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Arguments
START_STEP=${1:-1}
SKIP_TTS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step) START_STEP="$2"; shift 2;;
        --skip-tts) SKIP_TTS=true; shift;;
        *) shift;;
    esac
done

echo ""
echo "============================================================"
echo "  Phase 2: Model Fine-Tuning & Training"
echo "============================================================"
echo "  Project root: $PROJECT_ROOT"
echo "  Starting from step: $START_STEP"
echo "  Skip TTS: $SKIP_TTS"
echo "============================================================"
echo ""

# Activate conda environment
if ! command -v python &> /dev/null; then
    warn "Python not found; attempting conda activation..."
    eval "$(conda shell.bash hook)"
    conda activate armtts
fi

run_step() {
    STEP=$1
    DESC=$2
    CMD=$3

    if [ "$STEP" -lt "$START_STEP" ]; then
        echo -e "${YELLOW}[SKIP]${NC}  Step $STEP: $DESC"
        return
    fi

    info "$DESC"

    if eval "$CMD"; then
        ok
    else
        warn "Step $STEP failed (continuing)"
    fi
    echo ""
}

# ============================================================================
# Step 1: ASR Fine-Tuning (Whisper + LoRA)
# ============================================================================
run_step 1 \
    "ASR Fine-Tuning: Whisper large-v3 + LoRA on Common Voice" \
    "python scripts/training/train_asr.py --dataset-type common_voice --output-dir models/asr/whisper-hy-cv --max-train-samples 50000"

# ============================================================================
# Step 2: ASR Fine-Tuning (Merged dataset)
# ============================================================================
run_step 2 \
    "ASR Fine-Tuning: Scale to merged dataset (Common Voice + YouTube)" \
    "python scripts/training/train_asr.py --dataset-type merged --output-dir models/asr/whisper-hy-full"

# ============================================================================
# Step 3: TTS Fine-Tuning (Fish-Speech)
# ============================================================================
if [ "$SKIP_TTS" = false ]; then
    run_step 3 \
        "TTS Fine-Tuning: Fish-Speech S2 Pro with emotion/prosody" \
        "python scripts/training/train_tts.py --dataset-type common_voice --output-dir models/tts/fish-speech-hy"
else
    echo -e "${YELLOW}[SKIP]${NC}  Step 3: TTS Fine-Tuning (--skip-tts enabled)"
    echo ""
fi

# ============================================================================
# Step 4: Translation Optimization (SeamlessM4T)
# ============================================================================
run_step 4 \
    "Translation: SeamlessM4T v2 evaluation (no fine-tune needed, SOTA)" \
    "python scripts/training/evaluate_translation.py --output-dir models/translation/seamless-m4t-v2"

# ============================================================================
# Step 5: Evaluation & Metrics
# ============================================================================
run_step 5 \
    "Comprehensive evaluation: WER, MOS estimation, speaker similarity" \
    "python scripts/training/evaluate_all_models.py --asr-model models/asr/whisper-hy-full --tts-model models/tts/fish-speech-hy"

# ============================================================================
# Step 6: Generate synthetic test samples
# ============================================================================
run_step 6 \
    "Generate TTS test samples + artifacts" \
    "python scripts/training/generate_tts_samples.py --model models/tts/fish-speech-hy --output-dir outputs/tts_samples"

# ============================================================================
# Step 7: Export models to ONNX (deployment-ready)
# ============================================================================
run_step 7 \
    "Export models to ONNX format (inference optimization)" \
    "python scripts/training/export_models.py --asr-model models/asr/whisper-hy-full --tts-model models/tts/fish-speech-hy --output-dir models/onnx"

echo ""
echo "============================================================"
echo "  Phase 2: Model Fine-Tuning Complete!"
echo "============================================================"
echo ""
echo "  Trained models:"
echo "    models/asr/whisper-hy-cv/        — ASR on Common Voice only"
echo "    models/asr/whisper-hy-full/      — ASR on full merged dataset"
echo "    models/tts/fish-speech-hy/       — TTS with emotion/prosody"
echo "    models/translation/              — Translation (evaluated)"
echo ""
echo "  Exported models (ONNX):"
echo "    models/onnx/whisper.onnx         — ASR inference"
echo "    models/onnx/fish-speech.onnx     — TTS inference"
echo ""
echo "  Evaluation results:"
echo "    outputs/evaluation_results.json  — All metrics"
echo "    outputs/tts_samples/             — Synthetic speech samples"
echo ""
echo "  Next: Phase 3 — End-to-End Inference Pipeline"
echo "============================================================"
