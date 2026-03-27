# API Reference

## Overview

The FastAPI backend provides a REST API for video dubbing, health checks, and monitoring.

**Base URL**: `http://localhost:8000/api/v1`

## Authentication

Set `ARMTTS_API_KEY` in your `.env` file to enable API key authentication.

```bash
# Include in request headers
curl -H "X-API-Key: your-key-here" http://localhost:8000/api/v1/health
```

If `ARMTTS_API_KEY` is empty, authentication is disabled.

## Endpoints

### Health Check

```
GET /api/v1/health
```

Returns system status, GPU info, and model availability.

**Response**:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA RTX 4090",
  "models_loaded": false
}
```

### Dub Video

```
POST /api/v1/dub
```

Submit a video for dubbing. Returns a job ID for tracking.

**Request** (multipart/form-data):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video` | file | Yes | Input video file (.mp4, .mkv, .avi, .webm) |
| `reference_audio` | file | No | Speaker voice reference (10s WAV) |
| `dialect` | string | No | `eastern` (default) or `western` |
| `emotion` | string | No | `neutral`, `happy`, `sad`, `angry` |
| `output_format` | string | No | `mp4` (default) |

**Response**:
```json
{
  "job_id": "abc123",
  "status": "processing",
  "message": "Video dubbing started"
}
```

### Job Status

```
GET /api/v1/status/{job_id}
```

Check the status of a dubbing job.

**Response**:
```json
{
  "job_id": "abc123",
  "status": "completed",
  "progress": 100,
  "output_url": "/api/v1/download/abc123"
}
```

### Download Result

```
GET /api/v1/download/{job_id}
```

Download the dubbed video file.

### Metrics

```
GET /metrics
```

Prometheus-compatible metrics endpoint for monitoring.

## Running the API

```bash
# Local
make api

# Docker
docker compose up -d api

# Custom settings
uvicorn src.api.fastapi_server:create_app --factory --host 0.0.0.0 --port 8000
```

## Interactive Docs

When running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
