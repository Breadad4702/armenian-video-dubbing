#!/usr/bin/env python3
"""
Phase 0 Verification — Smoke test for all components.
Run: python scripts/verify_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = 0
FAIL = 0
WARN = 0


def check(name: str, fn, critical: bool = True):
    global PASS, FAIL, WARN
    try:
        result = fn()
        if result:
            print(f"  \033[92m[PASS]\033[0m {name}: {result}")
        else:
            print(f"  \033[92m[PASS]\033[0m {name}")
        PASS += 1
    except Exception as e:
        if critical:
            print(f"  \033[91m[FAIL]\033[0m {name}: {e}")
            FAIL += 1
        else:
            print(f"  \033[93m[WARN]\033[0m {name}: {e}")
            WARN += 1


def main():
    print("\n" + "=" * 60)
    print("  Armenian Video Dubbing AI — Phase 0 Verification")
    print("=" * 60)

    # --- Core Python ---
    print("\n--- Core Python ---")
    check("Python version", lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # --- PyTorch & CUDA ---
    print("\n--- PyTorch & CUDA ---")
    check("PyTorch import", lambda: None)

    def check_torch():
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            return f"CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}"
        return "CPU only (no CUDA)"
    check("PyTorch CUDA", check_torch, critical=False)

    def check_bf16():
        import torch
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            return "bfloat16 supported"
        return "bfloat16 not available (using float16)"
    check("bfloat16 support", check_bf16, critical=False)

    # --- ASR ---
    print("\n--- ASR Components ---")
    check("openai-whisper", lambda: __import__("whisper") and None)
    check("faster-whisper", lambda: __import__("faster_whisper") and None)

    def check_nemo():
        import nemo.collections.asr as nemo_asr
        return "NeMo ASR ready"
    check("NeMo ASR", check_nemo, critical=False)

    # --- Translation ---
    print("\n--- Translation ---")
    check("transformers", lambda: f"v{__import__('transformers').__version__}")
    check("sentencepiece", lambda: __import__("sentencepiece") and None)

    # --- TTS ---
    print("\n--- TTS / Voice Cloning ---")

    def check_fish():
        fish_dir = PROJECT_ROOT / "externals" / "fish-speech"
        if fish_dir.exists():
            return f"cloned at {fish_dir}"
        return None
    check("Fish-Speech repo", check_fish, critical=False)

    # --- Audio Processing ---
    print("\n--- Audio Processing ---")
    check("librosa", lambda: f"v{__import__('librosa').__version__}")
    check("soundfile", lambda: __import__("soundfile") and None)
    check("demucs", lambda: __import__("demucs") and None)
    check("pyloudnorm", lambda: __import__("pyloudnorm") and None)
    check("pydub", lambda: __import__("pydub") and None)
    check("webrtcvad", lambda: __import__("webrtcvad") and None)

    def check_rubberband():
        import subprocess
        r = subprocess.run(["rubberband", "--version"], capture_output=True, text=True)
        return r.stderr.strip().split("\n")[0] if r.returncode == 0 else r.stdout.strip().split("\n")[0]
    check("rubberband", check_rubberband, critical=False)

    # --- Video / Lip-sync ---
    print("\n--- Video / Lip-sync ---")
    check("OpenCV", lambda: f"v{__import__('cv2').__version__}")
    check("mediapipe", lambda: f"v{__import__('mediapipe').__version__}")
    check("decord", lambda: __import__("decord") and None)

    def check_ffmpeg():
        import subprocess
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return r.stdout.split("\n")[0]
    check("FFmpeg", check_ffmpeg)

    def check_musetalk():
        mt_dir = PROJECT_ROOT / "externals" / "MuseTalk"
        if mt_dir.exists():
            return f"cloned at {mt_dir}"
        return None
    check("MuseTalk repo", check_musetalk, critical=False)

    # --- ML Training ---
    print("\n--- ML Training Tools ---")
    check("accelerate", lambda: f"v{__import__('accelerate').__version__}")
    check("peft (LoRA)", lambda: f"v{__import__('peft').__version__}")
    check("bitsandbytes", lambda: __import__("bitsandbytes") and None, critical=False)
    check("datasets", lambda: f"v{__import__('datasets').__version__}")
    check("wandb", lambda: f"v{__import__('wandb').__version__}")

    # --- Evaluation ---
    print("\n--- Evaluation ---")
    check("jiwer (WER)", lambda: __import__("jiwer") and None)
    check("PESQ", lambda: __import__("pesq") and None, critical=False)
    check("resemblyzer", lambda: __import__("resemblyzer") and None)
    check("COMET", lambda: __import__("comet") and None, critical=False)

    # --- Web UI / API ---
    print("\n--- Web UI & API ---")
    check("gradio", lambda: f"v{__import__('gradio').__version__}")
    check("FastAPI", lambda: f"v{__import__('fastapi').__version__}")
    check("uvicorn", lambda: __import__("uvicorn") and None)

    # --- Utilities ---
    print("\n--- Utilities ---")
    check("yt-dlp", lambda: f"v{__import__('yt_dlp').version.__version__}")
    check("rich", lambda: __import__("rich") and None)
    check("loguru", lambda: __import__("loguru") and None)

    # --- Project Config ---
    print("\n--- Project Configuration ---")

    def check_config():
        from src.utils.config import load_config
        cfg = load_config()
        return f"loaded ({len(cfg)} top-level keys)"
    check("Config loader", check_config)

    # --- Directory structure ---
    print("\n--- Directory Structure ---")
    required_dirs = [
        "configs", "data", "models", "scripts", "src",
        "src/asr", "src/tts", "src/translation", "src/lipsync",
        "src/postprocessing", "src/pipeline", "src/utils",
        "ui", "outputs", "logs", "tests",
    ]
    missing = [d for d in required_dirs if not (PROJECT_ROOT / d).exists()]
    if missing:
        check("directories", lambda: (_ for _ in ()).throw(FileNotFoundError(f"Missing: {missing}")))
    else:
        check("directories", lambda: f"all {len(required_dirs)} present")

    # --- Summary ---
    total = PASS + FAIL + WARN
    print("\n" + "=" * 60)
    print(f"  Results: {PASS}/{total} passed, {FAIL} failed, {WARN} warnings")
    print("=" * 60)

    if FAIL > 0:
        print("\n  \033[91mSome critical checks failed. Fix before proceeding.\033[0m\n")
        sys.exit(1)
    elif WARN > 0:
        print("\n  \033[93mAll critical checks passed. Some optional components missing.\033[0m\n")
    else:
        print("\n  \033[92mAll checks passed! Ready for Phase 1.\033[0m\n")


if __name__ == "__main__":
    main()
