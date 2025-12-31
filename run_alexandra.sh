#!/bin/bash
# Alexandra AI - Launcher Script
# Run different components of Alexandra AI

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$HOME/MuseTalk/venv/bin/activate"

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║              Alexandra AI - Launcher                       ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if venv exists
if [ ! -f "$VENV_PATH" ]; then
    echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
    exit 1
fi

# Activate venv
source "$VENV_PATH"

# Function to show menu
show_menu() {
    echo -e "${GREEN}Available options:${NC}"
    echo ""
    echo "  1) Full App        - Complete Alexandra with all features (port 7862)"
    echo "  2) Simple Chat     - Basic chat with avatar (port 7860)"
    echo "  3) Real-Time       - MuseTalk-based faster avatar (port 7861)"
    echo "  4) API Server      - REST API for integrations (port 8000)"
    echo "  5) Discord Bot     - Run Discord bot"
    echo "  6) Telegram Bot    - Run Telegram bot"
    echo "  7) All Services    - Start API + Full App"
    echo "  8) Test Components - Check if everything is installed"
    echo "  9) Exit"
    echo ""
}

# Function to run app
run_app() {
    case $1 in
        1)
            echo -e "${YELLOW}Starting Full App on port 7862...${NC}"
            cd "$SCRIPT_DIR"
            python alexandra_full.py
            ;;
        2)
            echo -e "${YELLOW}Starting Simple Chat on port 7860...${NC}"
            cd "$SCRIPT_DIR"
            python app_with_enhanced_memory.py
            ;;
        3)
            echo -e "${YELLOW}Starting Real-Time App on port 7861...${NC}"
            cd "$SCRIPT_DIR"
            python app_realtime.py
            ;;
        4)
            echo -e "${YELLOW}Starting API Server on port 8000...${NC}"
            cd "$SCRIPT_DIR"
            python alexandra_api.py
            ;;
        5)
            echo -e "${YELLOW}Starting Discord Bot...${NC}"
            if [ -z "$DISCORD_TOKEN" ]; then
                echo -e "${RED}Error: DISCORD_TOKEN not set!${NC}"
                echo "Set it with: export DISCORD_TOKEN='your_token'"
                return 1
            fi
            cd "$SCRIPT_DIR"
            python discord_bot.py
            ;;
        6)
            echo -e "${YELLOW}Starting Telegram Bot...${NC}"
            if [ -z "$TELEGRAM_TOKEN" ]; then
                echo -e "${RED}Error: TELEGRAM_TOKEN not set!${NC}"
                echo "Set it with: export TELEGRAM_TOKEN='your_token'"
                return 1
            fi
            cd "$SCRIPT_DIR"
            python telegram_bot.py
            ;;
        7)
            echo -e "${YELLOW}Starting API Server (background) + Full App...${NC}"
            cd "$SCRIPT_DIR"
            python alexandra_api.py &
            API_PID=$!
            echo "API Server PID: $API_PID"
            sleep 3
            python alexandra_full.py
            kill $API_PID 2>/dev/null
            ;;
        8)
            echo -e "${YELLOW}Testing components...${NC}"
            cd "$SCRIPT_DIR"
            python -c "
from alexandra_config import *
import os

print('\\n=== Component Check ===\\n')

# Avatar
for name, path in AVATAR_IMAGES.items():
    status = 'FOUND' if os.path.exists(path) else 'MISSING'
    print(f'Avatar ({name}): {status}')

# Voice model
status = 'FOUND' if os.path.exists(F5_TTS_CHECKPOINT) else 'MISSING'
print(f'F5-TTS Voice Model: {status}')

# SadTalker
status = 'FOUND' if os.path.exists(os.path.join(SADTALKER_DIR, 'inference.py')) else 'MISSING'
print(f'SadTalker: {status}')

# MuseTalk
status = 'FOUND' if os.path.exists(os.path.join(MUSETALK_DIR, 'models/musetalkV15/unet.pth')) else 'MISSING'
print(f'MuseTalk 1.5: {status}')

# Whisper
try:
    import whisper
    print(f'Whisper: FOUND')
except:
    print(f'Whisper: MISSING')

# Discord/Telegram
try:
    import discord
    print(f'Discord.py: FOUND')
except:
    print(f'Discord.py: MISSING')

try:
    import telegram
    print(f'Telegram Bot: FOUND')
except:
    print(f'Telegram Bot: MISSING')

print('\\n=== Environment Variables ===\\n')
print(f'DISCORD_TOKEN: {\"Set\" if os.environ.get(\"DISCORD_TOKEN\") else \"Not set\"}')
print(f'TELEGRAM_TOKEN: {\"Set\" if os.environ.get(\"TELEGRAM_TOKEN\") else \"Not set\"}')
"
            ;;
        9)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
}

# Main loop
if [ $# -eq 0 ]; then
    # Interactive mode
    while true; do
        show_menu
        read -p "Select option (1-9): " choice
        run_app $choice
        echo ""
        read -p "Press Enter to continue..."
        clear
    done
else
    # Direct mode
    run_app $1
fi
