#!/usr/bin/env bash
# ============================================================================
# Phase 3: End-to-End Inference Pipeline — Master Orchestrator
# ============================================================================
# Complete dubbing pipeline: ASR → TTS → Lip-sync → Audio mix
#
# Prerequisites: Phase 2 complete (trained models available)
#
# Usage:
#   bash scripts/inference/run_phase3.sh [--video input.mp4] [--mode web|cli|api]
#   bash scripts/inference/run_phase3.sh --mode web      # Start Gradio UI
#   bash scripts/inference/run_phase3.sh --mode api      # Start FastAPI
#   bash scripts/inference/run_phase3.sh --video in.mp4  # CLI processing
# ============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Arguments
MODE="cli"  # cli, web, api, batch
VIDEO_FILE=""
EMOTION="neutral"
SKIP_LIPSYNC=false
NO_BACKGROUND=false
BATCH_MANIFEST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)         VIDEO_FILE="$2"; shift 2;;
        --emotion)       EMOTION="$2"; shift 2;;
        --skip-lipsync)  SKIP_LIPSYNC=true; shift;;
        --no-background) NO_BACKGROUND=true; shift;;
        --mode)          MODE="$2"; shift 2;;
        --batch)         BATCH_MANIFEST="$2"; shift 2;;
        *)               shift;;
    esac
done

# Activate conda environment if needed
if ! command -v python &> /dev/null; then
    info "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate armtts
fi

# Check Python
python --version

echo ""
echo "============================================================"
echo "  Phase 3: End-to-End Inference Pipeline"
echo "============================================================"
echo "  Mode: $MODE"
echo "============================================================"
echo ""

# ============================================================================
# Verify Models
# ============================================================================
verify_models() {
    info "Verifying trained models..."

    models=(
        "models/asr/whisper-hy-full/adapter_config.json"
        "models/tts/fish-speech-hy/train_data.json"
    )

    missing=0
    for model in "${models[@]}"; do
        if [ -f "$model" ]; then
            ok "Found: $model"
        else
            warn "Not found: $model (will use base pre-trained)"
            ((missing++))
        fi
    done

    if [ $missing -gt 0 ]; then
        warn "Some fine-tuned models missing — will use pre-trained defaults"
    fi
}

# ============================================================================
# CLI Mode — Process Single Video
# ============================================================================
run_cli() {
    if [ -z "$VIDEO_FILE" ]; then
        err "CLI mode requires --video argument"
    fi

    if [ ! -f "$VIDEO_FILE" ]; then
        err "Video file not found: $VIDEO_FILE"
    fi

    info "Processing video: $VIDEO_FILE"

    cmd="python src/pipeline.py '$VIDEO_FILE' --emotion $EMOTION --output outputs/dubbed_output.mp4"

    if [ "$SKIP_LIPSYNC" = true ]; then
        cmd="$cmd --skip-lipsync"
    fi

    if [ "$NO_BACKGROUND" = true ]; then
        cmd="$cmd --no-background"
    fi

    info "Command: $cmd"
    eval "$cmd"

    ok "Dubbing complete! Output: outputs/dubbed_output.mp4"
}

# ============================================================================
# Web UI Mode — Gradio
# ============================================================================
run_web() {
    info "Starting Gradio web UI on http://localhost:7860"
    info "Press Ctrl+C to stop"
    echo ""

    python src/ui/gradio_app.py --port 7860
}

# ============================================================================
# API Mode — FastAPI
# ============================================================================
run_api() {
    info "Starting FastAPI server on http://localhost:8000"
    info "API docs: http://localhost:8000/docs"
    info "Press Ctrl+C to stop"
    echo ""

    python src/api/fastapi_server.py --port 8000 --host 0.0.0.0

    echo ""
    echo "API Endpoints:"
    echo "  POST   /api/v1/dub                  — Submit dubbing job"
    echo "  GET    /api/v1/jobs/{job_id}       — Get job status"
    echo "  DELETE /api/v1/jobs/{job_id}       — Cancel job"
    echo "  GET    /api/v1/results/{job_id}    — Download result"
    echo "  GET    /api/v1/health               — Health check"
}

# ============================================================================
# Batch Mode — Process Multiple Videos
# ============================================================================
run_batch() {
    if [ -z "$BATCH_MANIFEST" ]; then
        err "Batch mode requires --batch argument"
    fi

    info "Processing batch: $BATCH_MANIFEST"

    python scripts/inference/batch_process.py \
        --input "$BATCH_MANIFEST" \
        --output-dir outputs/batch_dubbed \
        --emotion "$EMOTION"

    ok "Batch processing complete!"
}

# ============================================================================
# Main
# ============================================================================

verify_models

case "$MODE" in
    cli)
        run_cli
        ;;
    web)
        run_web
        ;;
    api)
        run_api
        ;;
    batch)
        run_batch
        ;;
    *)
        err "Unknown mode: $MODE"
        ;;
esac
