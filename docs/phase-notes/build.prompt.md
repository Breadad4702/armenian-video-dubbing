You are an elite senior ML engineering team (10+ years experience in low-resource language AI, video dubbing, TTS, lip-sync, and production deployment). Your goal is to build the WORLD'S BEST open-source Armenian Video Dubbing AI system in 2026 — perfect lip-sync, natural Eastern Armenian prosody & emotion, zero/few-shot voice cloning, and commercial film/tutorial quality.

Project Requirements (must achieve or exceed):
- WER < 8% on Armenian ASR
- MOS naturalness > 4.6/5 (human blind tests)
- Speaker similarity > 0.85 cosine
- Lip-sync error LSE-C/D < 1.8
- Real-time capable on single RTX 4090 (≤5 min for 10-min video)
- Full Eastern Armenian support (with Western fallback)
- Preserve original speaker emotion, pace, and background SFX/music
- Runs locally/offline, Dockerized, Gradio web UI + API

Use ONLY the absolute latest March 2026 SOTA open-source components:
- Lip-Sync: MuseTalk (latest v1.5+ from https://github.com/TMElyralab/MuseTalk) — best real-time latent inpainting
- TTS + Voice Cloning + Prosody: Fish-Speech S2 Pro (https://github.com/fishaudio/fish-speech — released March 10 2026, 50+ languages, emotion tags, best open-source dubbing model) OR Fun-CosyVoice 3.0-0.5B[](https://github.com/FunAudioLLM/CosyVoice) as fallback
- ASR: Whisper-large-v3 fine-tuned OR NVIDIA NeMo FastConformer-CTC from the official "Scaling Armenian ASR" paper (Robert Hakobyan et al., CSIT 2025 — ~5,350 validated hours, WER 8.56%)
- Translation/Timing: facebook/seamless-m4t-v2-large (native hye support)
- Post-processing: Demucs + CodeFormer + FFmpeg

DATA (you must guide full collection):
- Core: Mozilla Common Voice hy-AM (latest version)
- Massive scale: Replicate exactly the "Scaling Armenian ASR" pipeline (YouTube crawl → bootstrap ASR → statistical LM refinement → LabelStudio validation) to reach 5,000–8,000+ hours
- TTS: Same data + 50–100 hrs studio multi-speaker recordings
- Lip-sync: HDTF + Armenian YouTube talking-head videos

Your output MUST follow this exact structure (use markdown, code blocks, and clear steps):

PHASE 0: Environment Setup
- Exact conda/docker commands
- All GitHub clones + weight downloads

PHASE 1: Data Collection & Preparation
- Full scripts for YouTube crawl + bootstrap transcription
- LabelStudio setup for validation
- Dataset organization (train/val/test splits)

PHASE 2: Model Fine-Tuning (LoRA/QLoRA, efficient on 1–4 GPUs)
- Step-by-step notebooks/scripts for:
  • ASR (Whisper or NeMo)
  • Fish-Speech S2 or CosyVoice 3.0 fine-tuning on Armenian (include emotion/prosody conditioning)
  • SeamlessM4T timing alignment
- Hyperparameters, training commands, expected convergence

PHASE 3: Full End-to-End Inference Pipeline
- Complete ready-to-run Python script (ASR → translate → TTS clone + prosody → MuseTalk lip-sync → FFmpeg mix)
- Gradio web UI (upload video → select reference voice → output dubbed video)
- Command-line version + batch processing

PHASE 4: Evaluation & Quality Control
- Automated metrics script (WER, COMET, MOS estimation, LSE-C/D, speaker similarity)
- Human evaluation protocol for native Armenian speakers
- Iteration loop for weak spots

PHASE 5: Production Deployment
- Docker Compose + FastAPI
- RunPod/AWS one-click deployment
- Cost per minute at scale
- CapCut/Adobe plugin ideas (bonus)

Additional Expert Requirements:
- Prioritize Eastern Armenian (hy-AM) but add dialect selector
- Perfect duration matching + slight audio stretch/compress for lip-sync
- Background music/SFX isolation and re-mixing
- Quantized 4-bit inference for speed
- Full error handling, logging, progress bars
- Ethics: voice consent note + watermark option

Start by confirming you understand every requirement. Then output PHASE 0 immediately with all installation commands. After I approve each phase, deliver the next one with complete code. Never give incomplete or placeholder code — everything must be production-ready and copy-paste executable.

Begin now.