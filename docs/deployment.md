# Deployment Guide

## Docker (Recommended)

### Quick Start

```bash
# Build the image
make build

# Start all services
docker compose up -d

# Check status
docker compose ps
```

### Service Architecture

| Service | Port | Description |
|---------|------|-------------|
| nginx | 80, 443 | Reverse proxy with rate limiting |
| gradio | 7860 | Web UI |
| api | 8000 | REST API |
| label-studio | 8080 | Annotation tool (dev profile) |

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required:
- `HF_TOKEN` — HuggingFace access token for model downloads

Optional:
- `ARMTTS_API_KEY` — Enable API authentication
- `CUDA_VISIBLE_DEVICES` — GPU selection
- `WANDB_API_KEY` — Experiment tracking

### GPU Access

Docker Compose is configured with NVIDIA GPU support. Ensure you have:
- NVIDIA Container Toolkit installed
- Docker configured for GPU access

```bash
# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### SSL/TLS

Place your certificates in `docker/ssl/`:
```
docker/ssl/
├── cert.pem
└── key.pem
```

The nginx configuration will automatically use them.

---

## Cloud Deployment

### RunPod

```bash
# Set your API key
export RUNPOD_API_KEY=your-key

# Deploy
make deploy-runpod
# or
bash scripts/deployment/deploy_runpod.sh
```

### AWS (EC2 + GPU)

```bash
# Configure in .env
AWS_REGION=us-east-1
AWS_INSTANCE_TYPE=g5.xlarge
AWS_KEY_NAME=armtts-key

# Deploy
bash scripts/deployment/deploy_cloud.sh aws
```

### GCP (Compute Engine + GPU)

```bash
# Configure in .env
GCP_ZONE=us-central1-a
GCP_MACHINE_TYPE=n1-standard-8
GCP_GPU_TYPE=nvidia-tesla-t4

# Deploy
bash scripts/deployment/deploy_cloud.sh gcp
```

### Cost Estimation

```bash
python scripts/deployment/cost_estimate.py
```

Estimates per-minute dubbing costs across 6 cloud providers.

---

## Manual Deployment

### System Requirements

- Ubuntu 22.04+ / macOS 13+
- Python 3.11+
- NVIDIA GPU with CUDA 12.4+
- FFmpeg, rubberband-cli, libsndfile

### Install

```bash
# System deps (Ubuntu)
sudo apt-get install -y ffmpeg libsndfile1 rubberband-cli

# Python environment
bash scripts/setup_environment.sh

# Verify
python scripts/verify_setup.py
```

### Run Services

```bash
# Gradio UI
make web

# FastAPI
make api

# Both with nginx (requires nginx installed)
# Configure docker/nginx.conf for your setup
```

---

## Monitoring

### Prometheus Metrics

The API exposes metrics at `/metrics`:

- `armtts_requests_total` — Total API requests
- `armtts_dub_duration_seconds` — Dubbing job duration
- `armtts_active_jobs` — Currently processing jobs
- `armtts_gpu_memory_used_bytes` — GPU memory usage

### Health Checks

```bash
# API health
curl http://localhost:8000/api/v1/health

# Gradio health
curl http://localhost:7860/
```

Docker Compose includes health checks with 30s intervals.
