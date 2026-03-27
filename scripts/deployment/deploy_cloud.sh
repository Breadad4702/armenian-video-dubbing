#!/usr/bin/env bash
# ============================================================================
# Cloud Deploy — Auto-detect provider and deploy
# ============================================================================
# Supports: RunPod, AWS EC2, GCP, Local Docker
#
# Usage:
#   bash scripts/deployment/deploy_cloud.sh [--provider runpod|aws|gcp|local]
# ============================================================================

set -euo pipefail

PROVIDER="${PROVIDER:-auto}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --provider) PROVIDER="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [--provider runpod|aws|gcp|local]"
            echo ""
            echo "Providers:"
            echo "  runpod  - Deploy to RunPod (GPU cloud)"
            echo "  aws     - Deploy to AWS EC2 with GPU"
            echo "  gcp     - Deploy to Google Cloud with GPU"
            echo "  local   - Deploy locally with Docker Compose"
            echo "  auto    - Auto-detect based on available credentials"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Auto-detect provider
if [[ "$PROVIDER" == "auto" ]]; then
    if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
        PROVIDER="runpod"
    elif command -v aws &>/dev/null && aws sts get-caller-identity &>/dev/null 2>&1; then
        PROVIDER="aws"
    elif command -v gcloud &>/dev/null && gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1 | grep -q .; then
        PROVIDER="gcp"
    else
        PROVIDER="local"
    fi
    echo "Auto-detected provider: $PROVIDER"
fi

echo "================================================================"
echo "Armenian Video Dubbing AI — Cloud Deployment"
echo "Provider: $PROVIDER"
echo "================================================================"

case "$PROVIDER" in

    # ------------------------------------------------------------------
    runpod)
        bash "$SCRIPT_DIR/deploy_runpod.sh" "$@"
        ;;

    # ------------------------------------------------------------------
    aws)
        echo "AWS Deployment"
        echo ""

        INSTANCE_TYPE="${AWS_INSTANCE_TYPE:-g5.xlarge}"
        REGION="${AWS_REGION:-us-east-1}"
        KEY_NAME="${AWS_KEY_NAME:-armtts-key}"

        echo "  Instance: $INSTANCE_TYPE"
        echo "  Region:   $REGION"
        echo ""

        # Find latest Deep Learning AMI with CUDA
        AMI_ID=$(aws ec2 describe-images \
            --region "$REGION" \
            --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
            --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
            --output text 2>/dev/null || echo "")

        if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
            echo "Error: Could not find GPU AMI. Ensure AWS CLI is configured."
            exit 1
        fi

        echo "  AMI: $AMI_ID"

        # Create security group
        SG_ID=$(aws ec2 create-security-group \
            --group-name armtts-sg \
            --description "Armenian Video Dubbing" \
            --region "$REGION" \
            --output text --query 'GroupId' 2>/dev/null || \
            aws ec2 describe-security-groups \
                --group-names armtts-sg \
                --region "$REGION" \
                --output text --query 'SecurityGroups[0].GroupId')

        aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0 2>/dev/null || true
        aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 80 --cidr 0.0.0.0/0 2>/dev/null || true
        aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 7860 --cidr 0.0.0.0/0 2>/dev/null || true
        aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 8000 --cidr 0.0.0.0/0 2>/dev/null || true

        # User data script
        USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
apt-get update && apt-get install -y docker.io docker-compose-plugin nvidia-container-toolkit
systemctl enable docker && systemctl start docker
nvidia-ctk runtime configure --runtime=docker && systemctl restart docker

cd /opt && git clone --depth 1 https://github.com/YOUR_ORG/armenian-video-dubbing.git
cd armenian-video-dubbing
docker compose up -d
USERDATA
)
        INSTANCE_ID=$(aws ec2 run-instances \
            --image-id "$AMI_ID" \
            --instance-type "$INSTANCE_TYPE" \
            --key-name "$KEY_NAME" \
            --security-group-ids "$SG_ID" \
            --user-data "$USER_DATA" \
            --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100}}]' \
            --region "$REGION" \
            --output text --query 'Instances[0].InstanceId')

        echo ""
        echo "Instance launched: $INSTANCE_ID"
        echo ""

        # Wait for public IP
        echo "Waiting for public IP..."
        sleep 10
        PUBLIC_IP=$(aws ec2 describe-instances \
            --instance-ids "$INSTANCE_ID" \
            --region "$REGION" \
            --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

        echo "================================================================"
        echo "AWS deployment started!"
        echo "  Instance: $INSTANCE_ID"
        echo "  IP:       $PUBLIC_IP"
        echo "  Gradio:   http://$PUBLIC_IP:7860"
        echo "  API:      http://$PUBLIC_IP:8000"
        echo "  SSH:      ssh -i $KEY_NAME.pem ubuntu@$PUBLIC_IP"
        echo "================================================================"
        ;;

    # ------------------------------------------------------------------
    gcp)
        echo "GCP Deployment"
        echo ""

        ZONE="${GCP_ZONE:-us-central1-a}"
        MACHINE_TYPE="${GCP_MACHINE_TYPE:-n1-standard-8}"
        GPU_TYPE="${GCP_GPU_TYPE:-nvidia-tesla-t4}"
        INSTANCE_NAME="armtts-$(date +%s)"

        gcloud compute instances create "$INSTANCE_NAME" \
            --zone="$ZONE" \
            --machine-type="$MACHINE_TYPE" \
            --accelerator="type=$GPU_TYPE,count=1" \
            --maintenance-policy=TERMINATE \
            --boot-disk-size=100GB \
            --image-family=common-cu124-ubuntu-2204 \
            --image-project=deeplearning-platform-release \
            --metadata=startup-script='#!/bin/bash
apt-get update && apt-get install -y docker.io docker-compose-plugin nvidia-container-toolkit
systemctl enable docker && systemctl start docker
nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
cd /opt && git clone --depth 1 https://github.com/YOUR_ORG/armenian-video-dubbing.git
cd armenian-video-dubbing && docker compose up -d' \
            --tags=http-server,https-server

        gcloud compute firewall-rules create armtts-allow-web \
            --allow=tcp:7860,tcp:8000,tcp:80 \
            --target-tags=http-server 2>/dev/null || true

        PUBLIC_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
            --zone="$ZONE" \
            --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

        echo "================================================================"
        echo "GCP deployment started!"
        echo "  Instance: $INSTANCE_NAME"
        echo "  Zone:     $ZONE"
        echo "  IP:       $PUBLIC_IP"
        echo "  Gradio:   http://$PUBLIC_IP:7860"
        echo "  API:      http://$PUBLIC_IP:8000"
        echo "  SSH:      gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
        echo "================================================================"
        ;;

    # ------------------------------------------------------------------
    local)
        echo "Local Docker deployment"
        echo ""
        cd "$PROJECT_ROOT"

        echo "Building Docker image..."
        docker build -t armtts:latest .

        echo ""
        echo "Starting services..."
        docker compose up -d

        echo ""
        echo "================================================================"
        echo "Local deployment complete!"
        echo "  Gradio:   http://localhost:7860"
        echo "  API:      http://localhost:8000"
        echo "  API Docs: http://localhost:8000/docs"
        echo "  Status:   docker compose ps"
        echo "  Logs:     docker compose logs -f"
        echo "  Stop:     docker compose down"
        echo "================================================================"
        ;;

    *)
        echo "Unknown provider: $PROVIDER"
        echo "Supported: runpod, aws, gcp, local"
        exit 1
        ;;
esac
