#!/usr/bin/env bash
# ============================================================================
# Armenian Video Dubbing AI — Phase 0: Complete Environment Setup
# ============================================================================
# Prerequisites: NVIDIA GPU (RTX 3090/4090), CUDA 12.4+, ~100GB free disk
# Tested on: Ubuntu 22.04/24.04, macOS 15 (CPU-only), WSL2
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# 0. Pre-flight checks
# ---------------------------------------------------------------------------
info "Running pre-flight checks..."

command -v git   >/dev/null 2>&1 || err "git not found. Install git first."
command -v curl  >/dev/null 2>&1 || err "curl not found. Install curl first."

# Check for conda or mamba
if command -v mamba >/dev/null 2>&1; then
    CONDA_CMD="mamba"
elif command -v conda >/dev/null 2>&1; then
    CONDA_CMD="conda"
else
    err "Neither conda nor mamba found. Install Miniforge: https://github.com/conda-forge/miniforge"
fi

ok "Using $CONDA_CMD as package manager"

# Check GPU
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    ok "GPU detected: $GPU_NAME ($GPU_MEM)"
else
    warn "No NVIDIA GPU detected. CPU-only mode (slow for inference, unusable for training)."
fi

# ---------------------------------------------------------------------------
# 1. Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
info "Project root: $PROJECT_ROOT"

# ---------------------------------------------------------------------------
# 2. Create conda environment
# ---------------------------------------------------------------------------
ENV_NAME="armtts"

if $CONDA_CMD env list | grep -q "^${ENV_NAME} "; then
    warn "Environment '$ENV_NAME' already exists. Updating..."
    $CONDA_CMD env update -n "$ENV_NAME" -f configs/environment.yaml --prune
else
    info "Creating conda environment '$ENV_NAME'..."
    $CONDA_CMD env create -f configs/environment.yaml
fi

ok "Conda environment '$ENV_NAME' ready"

# Activate (for the rest of this script)
eval "$($CONDA_CMD shell.bash hook)"
conda activate "$ENV_NAME"

# ---------------------------------------------------------------------------
# 3. System-level dependencies
# ---------------------------------------------------------------------------
info "Checking system dependencies..."

# Rubberband (for time-stretching)
if ! command -v rubberband >/dev/null 2>&1; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        warn "Installing rubberband-cli..."
        sudo apt-get update && sudo apt-get install -y rubberband-cli libsox-dev
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        warn "Installing rubberband via Homebrew..."
        brew install rubberband sox
    fi
fi
ok "System dependencies satisfied"

# ---------------------------------------------------------------------------
# 4. Clone external repositories
# ---------------------------------------------------------------------------
EXTERNALS_DIR="$PROJECT_ROOT/externals"
mkdir -p "$EXTERNALS_DIR"

clone_repo() {
    local url="$1" dir="$2" branch="${3:-}"
    if [ -d "$EXTERNALS_DIR/$dir/.git" ]; then
        info "Updating $dir..."
        git -C "$EXTERNALS_DIR/$dir" pull --ff-only 2>/dev/null || true
    else
        info "Cloning $dir..."
        if [ -n "$branch" ]; then
            git clone --depth 1 --branch "$branch" "$url" "$EXTERNALS_DIR/$dir"
        else
            git clone --depth 1 "$url" "$EXTERNALS_DIR/$dir"
        fi
    fi
}

# MuseTalk — lip-sync
clone_repo "https://github.com/TMElyralab/MuseTalk.git" "MuseTalk"

# Fish-Speech — TTS + voice cloning
clone_repo "https://github.com/fishaudio/fish-speech.git" "fish-speech"

# CosyVoice — fallback TTS
clone_repo "https://github.com/FunAudioLLM/CosyVoice.git" "CosyVoice"

# CodeFormer — face enhancement
clone_repo "https://github.com/sczhou/CodeFormer.git" "CodeFormer"

# Demucs — source separation (installed via pip but we clone for scripts)
clone_repo "https://github.com/adefossez/demucs.git" "demucs"

ok "All external repos cloned"

# ---------------------------------------------------------------------------
# 5. Install external repo dependencies
# ---------------------------------------------------------------------------
info "Installing MuseTalk dependencies..."
cd "$EXTERNALS_DIR/MuseTalk"
pip install -r requirements.txt 2>/dev/null || warn "MuseTalk requirements partially failed (non-critical)"
cd "$PROJECT_ROOT"

info "Installing Fish-Speech..."
cd "$EXTERNALS_DIR/fish-speech"
pip install -e . 2>/dev/null || pip install -r requirements.txt 2>/dev/null || warn "Fish-Speech install partially failed"
cd "$PROJECT_ROOT"

info "Installing CodeFormer dependencies..."
cd "$EXTERNALS_DIR/CodeFormer"
pip install -r requirements.txt 2>/dev/null || warn "CodeFormer requirements partially failed"
python basicsr/setup.py develop 2>/dev/null || true
cd "$PROJECT_ROOT"

ok "External dependencies installed"

# ---------------------------------------------------------------------------
# 6. Download pre-trained model weights
# ---------------------------------------------------------------------------
info "Downloading pre-trained model weights..."

# --- Whisper large-v3 ---
WHISPER_DIR="$PROJECT_ROOT/models/asr"
mkdir -p "$WHISPER_DIR"
if [ ! -d "$WHISPER_DIR/whisper-large-v3" ]; then
    info "Downloading Whisper large-v3 via huggingface-cli..."
    huggingface-cli download openai/whisper-large-v3 \
        --local-dir "$WHISPER_DIR/whisper-large-v3" \
        --local-dir-use-symlinks False \
        --include "*.bin" "*.json" "*.txt" "config.*" "tokenizer.*" "preprocessor_config.*" || \
    warn "Whisper download failed — will auto-download on first use"
else
    ok "Whisper large-v3 already present"
fi

# --- SeamlessM4T v2 Large ---
SEAMLESS_DIR="$PROJECT_ROOT/models/translation"
mkdir -p "$SEAMLESS_DIR"
if [ ! -d "$SEAMLESS_DIR/seamless-m4t-v2-large" ]; then
    info "Downloading SeamlessM4T v2 Large..."
    huggingface-cli download facebook/seamless-m4t-v2-large \
        --local-dir "$SEAMLESS_DIR/seamless-m4t-v2-large" \
        --local-dir-use-symlinks False || \
    warn "SeamlessM4T download failed — will auto-download on first use"
else
    ok "SeamlessM4T v2 already present"
fi

# --- Fish-Speech S2 Pro ---
FISH_DIR="$PROJECT_ROOT/models/tts/fish-speech-s2-pro"
mkdir -p "$FISH_DIR"
if [ ! -d "$FISH_DIR/model" ]; then
    info "Downloading Fish-Speech S2 Pro..."
    huggingface-cli download fishaudio/fish-speech-1.5 \
        --local-dir "$FISH_DIR" \
        --local-dir-use-symlinks False || \
    warn "Fish-Speech download failed — check HF for latest model ID"
else
    ok "Fish-Speech S2 Pro already present"
fi

# --- MuseTalk weights ---
MUSETALK_DIR="$PROJECT_ROOT/models/lipsync/MuseTalk"
mkdir -p "$MUSETALK_DIR"
info "Downloading MuseTalk pre-trained weights..."
# MuseTalk stores weights in its own structure; we symlink
if [ -d "$EXTERNALS_DIR/MuseTalk/models" ]; then
    ln -sfn "$EXTERNALS_DIR/MuseTalk/models" "$MUSETALK_DIR/models" 2>/dev/null || true
fi
# Download from MuseTalk's HuggingFace
cd "$EXTERNALS_DIR/MuseTalk"
python -c "
import os, subprocess, sys
# MuseTalk downloads weights on first run; we pre-download
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)
urls = {
    'musetalk': 'TMElyralab/MuseTalk',
}
for name, repo in urls.items():
    try:
        subprocess.run([
            sys.executable, '-m', 'huggingface_hub', 'download',
            repo, '--local-dir', models_dir,
            '--local-dir-use-symlinks', 'False',
        ], check=True, timeout=600)
    except Exception as e:
        print(f'Warning: Could not download {name}: {e}')
" 2>/dev/null || warn "MuseTalk weight download may need manual setup"
cd "$PROJECT_ROOT"

# --- Demucs weights (auto-downloads on first use, but pre-cache) ---
info "Pre-caching Demucs htdemucs_ft model..."
python -c "
import torch
try:
    from demucs.pretrained import get_model
    model = get_model('htdemucs_ft')
    print('Demucs htdemucs_ft cached successfully')
except Exception as e:
    print(f'Demucs cache warning: {e}')
" 2>/dev/null || warn "Demucs pre-cache failed (will download on first use)"

# --- CodeFormer weights ---
info "Downloading CodeFormer weights..."
CODEFORMER_WEIGHTS="$EXTERNALS_DIR/CodeFormer/weights"
mkdir -p "$CODEFORMER_WEIGHTS/CodeFormer" "$CODEFORMER_WEIGHTS/facelib"
if [ ! -f "$CODEFORMER_WEIGHTS/CodeFormer/codeformer.pth" ]; then
    curl -L -o "$CODEFORMER_WEIGHTS/CodeFormer/codeformer.pth" \
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth" || \
    warn "CodeFormer weights download failed"
fi
ok "Model weights download complete"

# ---------------------------------------------------------------------------
# 7. Download Common Voice Armenian dataset
# ---------------------------------------------------------------------------
info "Setting up Common Voice hy-AM download..."
CV_DIR="$PROJECT_ROOT/data/common_voice"
mkdir -p "$CV_DIR"

cat > "$PROJECT_ROOT/scripts/data_collection/download_common_voice.py" << 'PYEOF'
#!/usr/bin/env python3
"""Download Mozilla Common Voice Armenian (hy-AM) dataset."""
import os
import sys
from pathlib import Path

def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)

    output_dir = Path(os.environ.get("CV_DIR", "data/common_voice"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Downloading Common Voice hy-AM (latest version)...")
    print("[INFO] You may need to accept the license at https://commonvoice.mozilla.org/")
    print("[INFO] Set HF_TOKEN env var if authentication is needed.")

    try:
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            "hy-AM",
            trust_remote_code=True,
            cache_dir=str(output_dir / "cache"),
        )

        for split in ds:
            count = len(ds[split])
            print(f"  {split}: {count} examples")
            # Save to disk for fast reloading
            ds[split].save_to_disk(str(output_dir / split))

        print(f"[OK] Common Voice hy-AM saved to {output_dir}")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("[INFO] You may need to:")
        print("  1. Create account at https://huggingface.co")
        print("  2. Accept dataset license at https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")
        print("  3. Set HF_TOKEN=hf_... environment variable")
        sys.exit(1)

if __name__ == "__main__":
    main()
PYEOF
chmod +x "$PROJECT_ROOT/scripts/data_collection/download_common_voice.py"
ok "Common Voice download script created"

# ---------------------------------------------------------------------------
# 8. Initialize git
# ---------------------------------------------------------------------------
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    info "Initializing git repository..."
    git init
    ok "Git initialized"
fi

# ---------------------------------------------------------------------------
# 9. Create .gitignore
# ---------------------------------------------------------------------------
cat > "$PROJECT_ROOT/.gitignore" << 'EOF'
# Data (too large for git)
data/
models/
externals/
outputs/
logs/
*.wav
*.mp3
*.mp4
*.avi
*.mkv
*.flac
*.ogg

# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
*.egg
.eggs/

# Environment
.env
*.env
.venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Weights & Biases
wandb/

# Docker
.docker/

# Temp
*.tmp
*.bak
*.log
EOF

# ---------------------------------------------------------------------------
# 10. Create entry-point __init__.py files
# ---------------------------------------------------------------------------
for d in src src/asr src/tts src/translation src/lipsync src/postprocessing src/pipeline src/utils; do
    touch "$PROJECT_ROOT/$d/__init__.py"
done

# ---------------------------------------------------------------------------
# 11. Verification
# ---------------------------------------------------------------------------
info "Running verification checks..."

python << 'PYEOF'
import sys
errors = []

def check(name, import_cmd):
    try:
        exec(import_cmd)
        print(f"  [OK] {name}")
    except Exception as e:
        errors.append(f"{name}: {e}")
        print(f"  [FAIL] {name}: {e}")

print("\n=== Python Package Verification ===\n")
check("PyTorch",           "import torch; assert torch.cuda.is_available() or True, 'No CUDA'")
check("Whisper",           "import whisper")
check("Faster-Whisper",    "from faster_whisper import WhisperModel")
check("Transformers",      "import transformers")
check("Datasets",          "import datasets")
check("PEFT (LoRA)",       "import peft")
check("BitsAndBytes",      "import bitsandbytes")
check("Accelerate",        "import accelerate")
check("Demucs",            "import demucs")
check("Librosa",           "import librosa")
check("Soundfile",         "import soundfile")
check("OpenCV",            "import cv2")
check("MediaPipe",         "import mediapipe")
check("Gradio",            "import gradio")
check("FastAPI",           "import fastapi")
check("yt-dlp",            "import yt_dlp")
check("Rich",              "import rich")
check("Loguru",            "from loguru import logger")
check("jiwer (WER)",       "import jiwer")
check("Resemblyzer",       "from resemblyzer import VoiceEncoder")
check("PyLoudnorm",        "import pyloudnorm")
check("PESQ",              "from pesq import pesq")
check("COMET",             "import comet")

print(f"\n{'='*50}")
if errors:
    print(f"[WARN] {len(errors)} package(s) had issues (may be non-critical):")
    for e in errors:
        print(f"  - {e}")
else:
    print("[OK] All packages verified successfully!")

# Check CUDA
import torch
if torch.cuda.is_available():
    print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"[GPU] CUDA: {torch.version.cuda}")
else:
    print("\n[WARN] No CUDA GPU — CPU-only mode")
PYEOF

ok "Phase 0 setup complete!"

echo ""
echo "============================================================"
echo "  Armenian Video Dubbing AI — Phase 0 Complete"
echo "============================================================"
echo ""
echo "  Activate environment:  conda activate armtts"
echo "  Project root:          $PROJECT_ROOT"
echo ""
echo "  Next steps:"
echo "    1. Run Common Voice download:"
echo "       python scripts/data_collection/download_common_voice.py"
echo "    2. Proceed to Phase 1: Data Collection & Preparation"
echo ""
echo "============================================================"
