# PHASE 5: PRODUCTION DEPLOYMENT — COMPLETE

## What Just Got Built

Full production deployment infrastructure for the Armenian Video Dubbing AI:

```
Developer → git push → CI/CD → Docker Build → Registry → Cloud Deploy
                                    ↓
User → nginx (TLS, rate-limit) → Gradio UI / FastAPI → GPU Pipeline → Output
                                    ↓
Ops → Prometheus /metrics → Monitoring Dashboard
```

---

## Code Breakdown

| Component | Lines | Purpose | Status |
|-----------|-------|---------|--------|
| `Dockerfile` | 90+ | Multi-stage CUDA build (base→deps→externals→app) | ✅ Fixed |
| `docker-compose.yaml` | 95+ | Nginx + Gradio + API + Label Studio (dev profile) | ✅ Fixed |
| `docker/nginx.conf` | 115+ | Reverse proxy, rate limiting, WebSocket, security headers | ✅ New |
| `.dockerignore` | 40+ | Exclude data/models/outputs from build context | ✅ New |
| `Makefile` | 100+ | 20+ targets (build, test, lint, deploy, clean) | ✅ New |
| `.github/workflows/ci.yaml` | 75+ | Lint → Test → Docker Build → Push to GHCR | ✅ New |
| `scripts/deployment/deploy_runpod.sh` | 90+ | One-click RunPod GPU deployment | ✅ New |
| `scripts/deployment/deploy_cloud.sh` | 160+ | Multi-cloud deploy (RunPod/AWS/GCP/local) | ✅ New |
| `src/api/fastapi_server.py` | 310+ | API key auth + Prometheus metrics + job queue | ✅ Enhanced |
| `.env.example` | 40+ | Full production env vars template | ✅ Enhanced |
| `pyproject.toml` | 42 | Fixed entry points and package discovery | ✅ Fixed |

---

## What Was Fixed

### Critical Fixes (Blocking Bugs)
1. **Dockerfile CMD** — `python -m ui.gradio_app` → `python -m src.ui.gradio_app`
2. **docker-compose API** — `src.pipeline.api:app` → `src.api.fastapi_server:create_app --factory`
3. **pyproject.toml entry point** — `src.pipeline.cli:main` → `src.pipeline:main`
4. **Missing `__init__.py`** — All 14 package directories now have init files
5. **FastAPI `torch` import** — Moved to module level from `__main__` block

### New Production Features
6. **API Authentication** — Optional API key via `X-API-Key` header (`ARMTTS_API_KEY` env var)
7. **Prometheus Metrics** — `/metrics` endpoint with request counts, job stats, GPU memory
8. **Nginx Reverse Proxy** — Rate limiting (10r/s API, 2r/min uploads), 500MB upload limit, WebSocket for Gradio, security headers
9. **CI/CD Pipeline** — GitHub Actions: lint (ruff) → test (pytest) → build & push Docker image to GHCR
10. **Cloud Deployment** — One-click scripts for RunPod, AWS EC2 (g5.xlarge), GCP, and local Docker

---

## Deployment Options

### 1. Local Docker (Easiest)
```bash
make build
make run
# Access: http://localhost:7860 (UI) / http://localhost:8000 (API)
```

### 2. RunPod (GPU Cloud)
```bash
export RUNPOD_API_KEY="your-key"
make deploy-runpod
# Access: https://{pod-id}-7860.proxy.runpod.net
```

### 3. AWS EC2
```bash
bash scripts/deployment/deploy_cloud.sh --provider aws
# Launches g5.xlarge with NVIDIA GPU AMI
```

### 4. GCP
```bash
bash scripts/deployment/deploy_cloud.sh --provider gcp
# Launches n1-standard-8 with T4 GPU
```

### 5. Auto-Detect
```bash
bash scripts/deployment/deploy_cloud.sh
# Checks: RUNPOD_API_KEY → AWS CLI → gcloud → local Docker
```

---

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │           nginx (port 80/443)        │
                    │  - TLS termination                   │
                    │  - Rate limiting: 10r/s API, 2r/m up │
                    │  - 500MB upload limit                │
                    │  - Security headers                   │
                    │  - WebSocket proxy                   │
                    └──────┬──────────────┬────────────────┘
                           │              │
              ┌────────────▼──┐    ┌──────▼────────────┐
              │  Gradio UI    │    │  FastAPI API       │
              │  port 7860    │    │  port 8000         │
              │               │    │                    │
              │  - Upload     │    │  - API key auth    │
              │  - Settings   │    │  - Job queue       │
              │  - Download   │    │  - /metrics        │
              └───────┬───────┘    │  - Background jobs │
                      │            └──────┬─────────────┘
                      │                   │
              ┌───────▼───────────────────▼──────────┐
              │           DubbingPipeline             │
              │  ASR → Translation → TTS → Lip-sync  │
              │         (GPU accelerated)             │
              └──────────────────────────────────────┘
```

---

## API Authentication

```bash
# Without API key (when ARMTTS_API_KEY is unset):
curl http://localhost:8000/api/v1/health

# With API key:
export ARMTTS_API_KEY="my-secret-key"
curl -H "X-API-Key: my-secret-key" http://localhost:8000/api/v1/dub \
  -F "video=@input.mp4"
```

---

## Monitoring

### Prometheus Metrics
```bash
curl http://localhost:8000/metrics
```

Exposes:
- `armtts_requests_total` — Total HTTP requests
- `armtts_requests_by_status` — Requests by status code
- `armtts_jobs_submitted_total` — Jobs submitted
- `armtts_jobs_completed_total` — Jobs completed
- `armtts_jobs_failed_total` — Jobs failed
- `armtts_active_jobs` — Currently processing
- `armtts_gpu_available` — GPU status
- `armtts_gpu_memory_used_gb` — GPU VRAM usage
- `armtts_gpu_memory_total_gb` — GPU VRAM total

---

## CI/CD Pipeline

```
git push → GitHub Actions:
  1. Lint (ruff check)
  2. Test (pytest on ubuntu + ffmpeg)
  3. Docker Build + Push to ghcr.io (main branch only)
```

---

## Makefile Targets

```bash
make help          # Show all targets
make build         # Build Docker image
make run           # Start all services
make stop          # Stop all services
make test          # Run tests
make lint          # Lint with ruff
make web           # Launch Gradio locally
make api           # Launch FastAPI locally
make dub VIDEO=x   # Dub a video from CLI
make deploy-runpod # Deploy to RunPod
make clean         # Remove temp files
```

---

## File Structure (Phase 5 Additions)

```
├── .dockerignore                         # NEW: Exclude large dirs from builds
├── .github/
│   └── workflows/
│       └── ci.yaml                       # NEW: CI/CD pipeline
├── docker/
│   ├── nginx.conf                        # NEW: Reverse proxy config
│   ├── ssl/                              # NEW: TLS cert directory
│   └── requirements-docker.txt           # Existing
├── scripts/
│   └── deployment/
│       ├── deploy_runpod.sh              # NEW: RunPod one-click
│       └── deploy_cloud.sh              # NEW: Multi-cloud deploy
├── Dockerfile                            # FIXED: Correct module paths
├── docker-compose.yaml                   # FIXED: Correct commands + nginx
├── Makefile                              # NEW: 20+ dev/deploy targets
├── pyproject.toml                        # FIXED: Entry points
└── .env.example                          # ENHANCED: Production vars
```

---

## Overall Project Status

```
PHASE 0: Environment Setup ..................... ✅ COMPLETE
PHASE 1: Data Collection (13 steps) ........... ✅ COMPLETE
PHASE 2: Model Fine-Tuning (ASR+TTS) ......... ✅ COMPLETE
PHASE 3: Inference Pipeline ................... ✅ COMPLETE (fixed)
PHASE 4: Evaluation & QC ...................... ✅ COMPLETE
PHASE 5: Production Deployment ................ ✅ COMPLETE

Total Code: 10,000+ lines (production-quality)
```
