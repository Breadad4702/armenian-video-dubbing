# ============================================================================
# Armenian Video Dubbing AI — Makefile
# ============================================================================

.PHONY: help build run stop test lint clean deploy logs

DOCKER_IMAGE = armtts:latest
COMPOSE = docker compose

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

install: ## Install dependencies locally (conda env)
	bash scripts/setup_environment.sh

test: ## Run tests
	python -m pytest tests/ -v --tb=short

lint: ## Lint code with ruff
	python -m ruff check src/ scripts/ tests/

lint-fix: ## Auto-fix lint issues
	python -m ruff check --fix src/ scripts/ tests/

format: ## Format code with ruff
	python -m ruff format src/ scripts/ tests/

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE) .

run: ## Start all services (Gradio + API + nginx)
	$(COMPOSE) up -d

run-gpu: ## Start GPU services only (Gradio + API)
	$(COMPOSE) up -d gradio api

run-gradio: ## Start Gradio UI only
	$(COMPOSE) up -d gradio

run-api: ## Start FastAPI only
	$(COMPOSE) up -d api

run-dev: ## Start all services including Label Studio
	$(COMPOSE) --profile dev up -d

stop: ## Stop all services
	$(COMPOSE) down

restart: ## Restart all services
	$(COMPOSE) restart

logs: ## Follow logs from all services
	$(COMPOSE) logs -f

logs-api: ## Follow API logs
	$(COMPOSE) logs -f api

logs-gradio: ## Follow Gradio logs
	$(COMPOSE) logs -f gradio

status: ## Show service status
	$(COMPOSE) ps

# ---------------------------------------------------------------------------
# Local inference
# ---------------------------------------------------------------------------

dub: ## Dub a video (make dub VIDEO=input.mp4 EMOTION=neutral)
	python -m src.pipeline $(VIDEO) --emotion $(or $(EMOTION),neutral) --output dubbed_output.mp4

web: ## Launch Gradio UI locally
	python -m src.ui.gradio_app

api: ## Launch FastAPI locally
	python -m uvicorn src.api.fastapi_server:create_app --factory --host 0.0.0.0 --port 8000

batch: ## Batch process (make batch MANIFEST=videos.csv)
	python scripts/inference/batch_process.py --input $(MANIFEST)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

train-asr: ## Train ASR model
	python scripts/training/train_asr.py

train-tts: ## Train TTS model
	python scripts/training/train_tts.py

evaluate: ## Run evaluation suite
	python scripts/training/evaluate_all_models.py

# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------

deploy-runpod: ## Deploy to RunPod
	bash scripts/deployment/deploy_runpod.sh

deploy-cloud: ## Deploy to cloud (auto-detect provider)
	bash scripts/deployment/deploy_cloud.sh

push-image: ## Push Docker image to registry
	docker tag $(DOCKER_IMAGE) $(REGISTRY)/$(DOCKER_IMAGE)
	docker push $(REGISTRY)/$(DOCKER_IMAGE)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean: ## Remove temp files and caches
	rm -rf outputs/temp/*
	rm -rf __pycache__ src/__pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true

clean-docker: ## Remove Docker artifacts
	$(COMPOSE) down --rmi local --volumes --remove-orphans

clean-all: clean clean-docker ## Full cleanup
