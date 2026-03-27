# ============================================================================
# Armenian Video Dubbing AI — Multi-stage Dockerfile
# ============================================================================
# Build:  docker build -t armtts:latest .
# Run:    docker run --gpus all -p 7860:7860 -p 8000:8000 -v ./models:/app/models armtts:latest
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Base with CUDA + Python
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git git-lfs curl wget \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    libsndfile1 libsox-dev sox \
    rubberband-cli librubberband-dev \
    build-essential cmake ninja-build \
    nginx \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Create non-root user
RUN useradd -m -s /bin/bash armtts
WORKDIR /app

# ---------------------------------------------------------------------------
# Stage 2: Python dependencies
# ---------------------------------------------------------------------------
FROM base AS deps

COPY docker/requirements-docker.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /tmp/requirements.txt && \
    python -m pip install edge-tts prometheus-client

# ---------------------------------------------------------------------------
# Stage 3: Clone externals
# ---------------------------------------------------------------------------
FROM deps AS externals

RUN mkdir -p /app/externals && \
    git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /app/externals/MuseTalk && \
    git clone --depth 1 https://github.com/fishaudio/fish-speech.git /app/externals/fish-speech && \
    git clone --depth 1 https://github.com/sczhou/CodeFormer.git /app/externals/CodeFormer

# Install external deps (non-fatal if they fail)
RUN cd /app/externals/MuseTalk && pip install -r requirements.txt 2>/dev/null; true
RUN cd /app/externals/fish-speech && pip install -e . 2>/dev/null; true
RUN cd /app/externals/CodeFormer && pip install -r requirements.txt 2>/dev/null; true

# ---------------------------------------------------------------------------
# Stage 4: Application
# ---------------------------------------------------------------------------
FROM externals AS app

# Copy application code
COPY configs/ /app/configs/
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY pyproject.toml /app/

# Create necessary directories and __init__.py files
RUN mkdir -p /app/data /app/models /app/outputs /app/logs /app/uploads && \
    find /app/src -type d -exec touch {}/__init__.py \; && \
    chown -R armtts:armtts /app

# Volumes for data and models (persist across containers)
VOLUME ["/app/data", "/app/models", "/app/outputs", "/app/logs"]

# Expose ports: nginx (80), Gradio (7860), FastAPI (8000)
EXPOSE 80 7860 8000

USER armtts

# Health check against FastAPI
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default: launch Gradio UI
CMD ["python", "-m", "src.ui.gradio_app"]
