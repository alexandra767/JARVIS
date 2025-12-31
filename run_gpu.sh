#!/bin/bash
# Alexandra AI - Run with GPU support on DGX Spark
# Uses NGC PyTorch container for CUDA 13.0 / sm_121 support

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Use saved image with all packages pre-installed (fast startup)
if docker image inspect alexandra-gpu:working >/dev/null 2>&1; then
    CONTAINER_IMAGE="alexandra-gpu:working"
    echo -e "${GREEN}Using pre-built image (fast startup)${NC}"
elif docker image inspect alexandra-gpu:latest >/dev/null 2>&1; then
    CONTAINER_IMAGE="alexandra-gpu:latest"
    echo -e "${GREEN}Using pre-built image${NC}"
else
    CONTAINER_IMAGE="nvcr.io/nvidia/pytorch:25.09-py3"
    echo -e "${YELLOW}Using base image (will install packages)${NC}"
fi

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Alexandra AI - GPU Mode${NC}"
echo -e "${GREEN}DGX Spark / GB10 / CUDA 13.0${NC}"
echo -e "${GREEN}================================${NC}"

# Check if running interactively or with a specific app
APP=${1:-"interactive"}

case $APP in
    "ui")
        # Your existing UI with GPU support
        CMD="cd /workspace/ai-clone-chat && python alexandra_ui.py"
        PORTS="-p 7860:7860"
        ;;
    "full")
        CMD="cd /workspace/ai-clone-chat && python alexandra_full.py"
        PORTS="-p 7862:7862"
        ;;
    "realtime")
        CMD="cd /workspace/ai-clone-chat && python app_realtime.py"
        PORTS="-p 7861:7861"
        ;;
    "api")
        CMD="cd /workspace/ai-clone-chat && python alexandra_api.py"
        PORTS="-p 8000:8000"
        ;;
    "server")
        # Persistent model server
        CMD="cd /workspace/ai-clone-chat && python model_server.py"
        PORTS="-p 7865:7865"
        ;;
    "interactive"|*)
        CMD="bash"
        PORTS="-p 7860:7860 -p 7861:7861 -p 7862:7862 -p 8000:8000"
        ;;
esac

echo -e "${YELLOW}Starting: $APP${NC}"
echo ""

# Use -it if interactive, otherwise -d for detached
# Remove --rm temporarily to see errors
if [ -t 0 ]; then
    DOCKER_FLAGS="-it"
else
    DOCKER_FLAGS="-d"
fi

docker run $DOCKER_FLAGS \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=16g \
    -v "$HOME:/workspace" \
    -v "$HOME/ai-clone-chat:/workspace/ai-clone-chat" \
    -v "$HOME/MuseTalk:/workspace/MuseTalk" \
    -v "$HOME/SadTalker:/workspace/SadTalker" \
    -v "$HOME/voice_training:/workspace/voice_training" \
    -v "$HOME/ComfyUI:/workspace/ComfyUI" \
    -v "$HOME/models:/workspace/models" \
    -v "$HOME/.cache:/root/.cache" \
    -w /workspace/ai-clone-chat \
    $PORTS \
    -e PYTHONPATH="/workspace/ai-clone-chat:/workspace/MuseTalk:/workspace/voice_training/F5-TTS" \
    $CONTAINER_IMAGE \
    bash -c "
        # Create symlinks so paths work
        ln -sf /workspace /home/alexandratitus767 2>/dev/null || true
        mkdir -p /root/voice_training
        ln -sf /workspace/voice_training/F5-TTS /root/voice_training/F5-TTS 2>/dev/null || true

        # Run the command
        $CMD
    "
