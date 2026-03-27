#!/usr/bin/env bash
# ============================================================================
# Deploy to RunPod — One-click GPU cloud deployment
# ============================================================================
# Prerequisites:
#   - RunPod account (https://runpod.io)
#   - RUNPOD_API_KEY environment variable set
#   - Docker image pushed to a registry
#
# Usage:
#   export RUNPOD_API_KEY="your-api-key"
#   bash scripts/deployment/deploy_runpod.sh [--gpu RTX4090] [--image ghcr.io/you/armtts:latest]
# ============================================================================

set -euo pipefail

# Defaults
GPU_TYPE="${GPU_TYPE:-NVIDIA RTX 4090}"
DOCKER_IMAGE="${DOCKER_IMAGE:-armtts:latest}"
VOLUME_SIZE="${VOLUME_SIZE:-50}"
TEMPLATE_NAME="armenian-video-dubbing"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)     GPU_TYPE="$2";      shift 2 ;;
        --image)   DOCKER_IMAGE="$2";  shift 2 ;;
        --volume)  VOLUME_SIZE="$2";   shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Validate
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    echo "Error: RUNPOD_API_KEY not set"
    echo "  export RUNPOD_API_KEY='your-runpod-api-key'"
    exit 1
fi

echo "================================================================"
echo "Armenian Video Dubbing AI — RunPod Deployment"
echo "================================================================"
echo "  GPU:    $GPU_TYPE"
echo "  Image:  $DOCKER_IMAGE"
echo "  Volume: ${VOLUME_SIZE}GB"
echo "================================================================"

# Check for runpodctl
if ! command -v runpodctl &>/dev/null; then
    echo "Installing runpodctl..."
    if [[ "$(uname)" == "Darwin" ]]; then
        brew install runpod/runpodctl/runpodctl 2>/dev/null || {
            curl -sL https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-darwin-amd64 -o /usr/local/bin/runpodctl
            chmod +x /usr/local/bin/runpodctl
        }
    else
        curl -sL https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 -o /usr/local/bin/runpodctl
        chmod +x /usr/local/bin/runpodctl
    fi
fi

runpodctl config --apiKey "$RUNPOD_API_KEY"

# Create pod
echo ""
echo "Creating RunPod instance..."

POD_ID=$(runpodctl create pod \
    --name "$TEMPLATE_NAME" \
    --gpuType "$GPU_TYPE" \
    --gpuCount 1 \
    --imageName "$DOCKER_IMAGE" \
    --volumeSize "$VOLUME_SIZE" \
    --ports "7860/http,8000/http" \
    --env "CUDA_VISIBLE_DEVICES=0" \
    --env "ARMTTS_CONFIG=/app/configs/config.yaml" \
    2>&1 | grep -oP 'pod "\K[^"]+' || echo "")

if [[ -z "$POD_ID" ]]; then
    echo "Falling back to RunPod API..."

    POD_ID=$(curl -s "https://api.runpod.io/graphql" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -d '{
            "query": "mutation { podFindAndDeployOnDemand(input: { name: \"'"$TEMPLATE_NAME"'\", imageName: \"'"$DOCKER_IMAGE"'\", gpuTypeId: \"NVIDIA GeForce RTX 4090\", gpuCount: 1, volumeInGb: '"$VOLUME_SIZE"', containerDiskInGb: 20, ports: \"7860/http,8000/http\", env: [{key: \"CUDA_VISIBLE_DEVICES\", value: \"0\"}, {key: \"ARMTTS_CONFIG\", value: \"/app/configs/config.yaml\"}] }) { id } }"
        }' | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['podFindAndDeployOnDemand']['id'])" 2>/dev/null || echo "")
fi

if [[ -z "$POD_ID" ]]; then
    echo "Error: Failed to create pod. Check your API key and GPU availability."
    exit 1
fi

echo ""
echo "================================================================"
echo "Pod created successfully!"
echo "  Pod ID:  $POD_ID"
echo ""
echo "  Access:"
echo "    Gradio UI:  https://${POD_ID}-7860.proxy.runpod.net"
echo "    API:        https://${POD_ID}-8000.proxy.runpod.net"
echo "    API Docs:   https://${POD_ID}-8000.proxy.runpod.net/docs"
echo ""
echo "  Manage:"
echo "    runpodctl get pod $POD_ID"
echo "    runpodctl stop pod $POD_ID"
echo "    runpodctl remove pod $POD_ID"
echo "================================================================"
