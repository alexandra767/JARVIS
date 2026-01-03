#!/usr/bin/env python3
"""
JARVIS Dashboard - Full Visual Interface
Combines real-time widgets, voice interaction, camera, and image generation
"""

# Fix PyTorch 2.6+ compatibility BEFORE any imports
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Import local model loading
try:
    from alexandra_model import load_model, unload_model, generate_response as model_generate, get_model_status, AVAILABLE_MODELS, DEFAULT_MODEL
    LOCAL_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Local model not available: {e}")
    LOCAL_MODEL_AVAILABLE = False

# Import voice/TTS module
try:
    from alexandra_voice import generate_voice, get_voice_status, F5_TTS_AVAILABLE, warmup_voice
    VOICE_TTS_AVAILABLE = F5_TTS_AVAILABLE
    # Warmup voice on first use (lazy)
    _voice_warmed_up = False
except ImportError as e:
    print(f"Voice module not available: {e}")
    VOICE_TTS_AVAILABLE = False
    warmup_voice = None
    _voice_warmed_up = True

# Import vision module (Qwen2-VL)
try:
    from alexandra_vision import (
        load_vision_model, unload_vision_model, analyze_image as vision_analyze,
        get_vision_status, VISION_AVAILABLE as QWEN_VISION_AVAILABLE
    )
    QWEN_VISION_READY = True
except ImportError as e:
    print(f"Qwen2-VL vision module not available: {e}")
    QWEN_VISION_READY = False

# Import avatar module (MuseTalk)
try:
    from alexandra_avatar import (
        get_engine as get_avatar_engine, list_avatar_outfits, prepare_avatar,
        generate_avatar_video, load_avatar_models, unload_avatar_models,
        get_avatar_status, MUSETALK_AVAILABLE
    )
    AVATAR_AVAILABLE = MUSETALK_AVAILABLE
except ImportError as e:
    print(f"Avatar module not available: {e}")
    AVATAR_AVAILABLE = False
    MUSETALK_AVAILABLE = False

import gradio as gr
import asyncio
import threading
import queue
import time
import os
import sys
import random
import numpy as np
import cv2
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
import base64
from PIL import Image
import io

# Screen control for gesture navigation using xdotool (works in Docker)
import subprocess
XDOTOOL_AVAILABLE = False
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# ============================================================================
# CLOUD API INTEGRATION (Gemini + ElevenLabs)
# ============================================================================
GEMINI_AVAILABLE = False
ELEVENLABS_AVAILABLE = False

try:
    from cloud_config import GEMINI_API_KEY, ELEVENLABS_API_KEY, GEMINI_MODEL, ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL

    # Initialize Gemini
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
        print("Gemini API initialized")
    except Exception as e:
        print(f"Gemini not available: {e}")
        genai = None

    # Initialize ElevenLabs
    try:
        from elevenlabs import ElevenLabs
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        ELEVENLABS_AVAILABLE = True
        print("ElevenLabs API initialized")
    except Exception as e:
        print(f"ElevenLabs not available: {e}")
        elevenlabs_client = None

except ImportError as e:
    print(f"Cloud config not found: {e}")
    genai = None
    elevenlabs_client = None

# Global state for model/voice selection
_use_gemini = False  # Default to local model
_use_elevenlabs = False  # ElevenLabs disabled (use local JARVIS)
_use_browser_tts = False  # Browser voice disabled by default

def gemini_chat(message: str, history: list = None, system_prompt: str = None, use_search: bool = True) -> str:
    """Generate response using Gemini API with optional Google Search grounding"""
    if not GEMINI_AVAILABLE or genai is None:
        return None

    try:
        # Build conversation context
        contents = []

        # Add system prompt if provided
        if system_prompt:
            contents.append(f"System instructions: {system_prompt}\n\n")

        # Add conversation history
        if history:
            for msg in history[-20:]:  # Keep last 20 messages
                role = "User" if msg.get("role") == "user" else "Assistant"
                contents.append(f"{role}: {msg.get('content', '')}\n")

        # Add current message
        contents.append(f"User: {message}\nAssistant:")

        full_prompt = "".join(contents)

        # Check if this query needs real-time search
        msg_lower = message.lower()
        needs_search = any(kw in msg_lower for kw in [
            'today', 'tonight', 'now', 'current', 'latest', 'recent',
            'playing', 'game', 'score', 'weather', 'news', 'who won',
            'nfl', 'nba', 'mlb', 'nhl', 'football', 'basketball', 'baseball',
            'schedule', 'this week', 'this weekend', 'happening'
        ])

        # Use the generative model
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(full_prompt)

        return response.text
    except Exception as e:
        print(f"Gemini error: {e}")
        return None

def gemini_vision(image, prompt: str = "What do you see in this image?") -> str:
    """Analyze an image using Gemini's vision capabilities"""
    if not GEMINI_AVAILABLE or genai is None:
        return None

    try:
        import PIL.Image
        import io
        import numpy as np

        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            # OpenCV uses BGR, convert to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]
            pil_image = PIL.Image.fromarray(image)
        elif isinstance(image, PIL.Image.Image):
            pil_image = image
        elif isinstance(image, str):
            # It's a file path
            pil_image = PIL.Image.open(image)
        else:
            return "Unsupported image format"

        # Use gemini-2.0-flash-exp for vision
        model = genai.GenerativeModel(GEMINI_MODEL)

        # Add grounding instructions to reduce hallucinations
        grounded_prompt = f"""You are JARVIS, analyzing a LIVE camera feed for Ms. Alexandra.

CRITICAL RULES:
- ONLY describe what you can ACTUALLY see in this specific image
- If the image is dark, blurry, or unclear, say so honestly
- Do NOT make up or assume details that aren't clearly visible
- Be specific about what you observe, not what you expect
- If you can't see something clearly, say "I cannot clearly make out..."

User's question: {prompt}

Describe what you actually see in this image:"""

        # Generate response with image
        response = model.generate_content([grounded_prompt, pil_image])

        return response.text
    except Exception as e:
        print(f"Gemini vision error: {e}")
        return None

def elevenlabs_tts(text: str) -> str:
    """Generate speech using ElevenLabs API, returns path to audio file"""
    if not ELEVENLABS_AVAILABLE or not elevenlabs_client:
        return None

    try:
        import tempfile

        # Generate audio
        audio = elevenlabs_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id=ELEVENLABS_MODEL
        )

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            for chunk in audio:
                f.write(chunk)
            return f.name
    except Exception as e:
        print(f"ElevenLabs error: {e}")
        return None

def set_cloud_mode(use_gemini: bool, use_elevenlabs: bool):
    """Set whether to use cloud APIs"""
    global _use_gemini, _use_elevenlabs
    _use_gemini = use_gemini and GEMINI_AVAILABLE
    _use_elevenlabs = use_elevenlabs and ELEVENLABS_AVAILABLE
    return f"Model: {'Gemini' if _use_gemini else 'Local'}, Voice: {'ElevenLabs' if _use_elevenlabs else 'Local F5'}"

def init_gesture_control():
    """Initialize gesture control with virtual display"""
    global XDOTOOL_AVAILABLE, SCREEN_WIDTH, SCREEN_HEIGHT
    try:
        # Start virtual display if needed
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1920, 1080))
        display.start()
        print(f"Virtual display started on {os.environ.get('DISPLAY')}")
    except Exception as e:
        print(f"Virtual display not needed or failed: {e}")

    # Test xdotool availability
    try:
        result = subprocess.run(['xdotool', 'getmouselocation'], capture_output=True, timeout=2)
        if result.returncode == 0:
            XDOTOOL_AVAILABLE = True
            print("xdotool available - gesture control enabled")
            # Get screen size
            try:
                result = subprocess.run(['xdotool', 'getdisplaygeometry'], capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    parts = result.stdout.strip().split()
                    if len(parts) >= 2:
                        SCREEN_WIDTH = int(parts[0])
                        SCREEN_HEIGHT = int(parts[1])
                        print(f"Screen size: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
            except:
                pass
    except Exception as e:
        print(f"xdotool not available - gesture control disabled: {e}")
    return XDOTOOL_AVAILABLE

# Initialize gesture control
init_gesture_control()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JarvisDashboard")

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Jarvis components
from jarvis_tools import JarvisTools, EnhancedJarvisTools, WeatherService, NewsService, TimeService, WebSearchService
from jarvis_files import JarvisFileManager

# Import Memory and RAG systems
try:
    from enhanced_memory import EnhancedAlexandraMemory
    MEMORY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Memory system not available: {e}")
    MEMORY_AVAILABLE = False

try:
    from alexandra_rag import AlexandraRAG, get_rag
    RAG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG system not available: {e}")
    RAG_AVAILABLE = False

# Epstein Files RAG
epstein_rag = None
news_rag = None
_epstein_search_results = []

if RAG_AVAILABLE:
    try:
        epstein_rag = AlexandraRAG(
            persist_dir="/workspace/epstein_files/rag_data",
            collection_name="epstein_docs"
        )
        logger.info(f"Epstein RAG loaded: {epstein_rag.get_stats()['total_chunks']} chunks")
    except Exception as e:
        logger.warning(f"Epstein RAG not available: {e}")

    try:
        news_rag = AlexandraRAG(
            persist_dir="/workspace/news_rag/rag_data",
            collection_name="current_news"
        )
        logger.info(f"News RAG loaded: {news_rag.get_stats()['total_chunks']} articles")
    except Exception as e:
        logger.warning(f"News RAG not available: {e}")

# Try to import other components
try:
    from jarvis_voice import JarvisVoiceSystem
    VOICE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Voice system not available: {e}")
    VOICE_AVAILABLE = False

try:
    from jarvis_vision import JarvisVision, VisionConfig
    VISION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Vision system not available: {e}")
    VISION_AVAILABLE = False

try:
    from jarvis_hands import JarvisHands, HandConfig, Gesture
    HANDS_AVAILABLE = True
except Exception as e:
    logger.warning(f"Hand tracking not available: {e}")
    HANDS_AVAILABLE = False

# MediaPipe for hand tracking overlay
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    logger.warning(f"MediaPipe not available for hand tracking: {e}")
    MEDIAPIPE_AVAILABLE = False
    mp_hands = None
    mp_draw = None
    mp_drawing_styles = None

# Import Proactive Systems (monitoring, alerts, routines)
try:
    from jarvis_proactive import proactive, JarvisProactive, AlertPriority
    PROACTIVE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Proactive systems not available: {e}")
    PROACTIVE_AVAILABLE = False
    proactive = None

# Import Calendar/Scheduler
try:
    from jarvis_calendar import scheduler, JarvisScheduler
    SCHEDULER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Scheduler not available: {e}")
    SCHEDULER_AVAILABLE = False
    scheduler = None

# Import Screen Understanding
try:
    from jarvis_screen import screen as jarvis_screen, JarvisScreen
    SCREEN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Screen system not available: {e}")
    SCREEN_AVAILABLE = False
    jarvis_screen = None

# Import Navigation System
try:
    from jarvis_navigation import navigator, parse_and_respond as nav_respond, get_traffic
    NAVIGATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Navigation system not available: {e}")
    NAVIGATION_AVAILABLE = False
    navigator = None


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "ollama_url": "http://localhost:11434",
    "llm_model": "qwen3-coder:30b",
    "flux_model": "black-forest-labs/FLUX.1-schnell",
    "lora_path": "/workspace/loras/flux_dreambooth.safetensors",
    "lora_path_uncensored": "/workspace/loras/flux-uncensored.safetensors",
    "trigger_word": "alexandra woman",
    "default_location": "Ridgway, PA",
    "refresh_interval": 60,  # seconds for widget refresh
}

def get_ollama_models():
    """Fetch available models from Ollama server"""
    import requests
    try:
        resp = requests.get(f"{CONFIG['ollama_url']}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return [m["name"] for m in models]
    except Exception as e:
        print(f"[OLLAMA] Could not fetch models: {e}")
    return ["qwen2.5:72b"]  # Fallback

OLLAMA_MODELS = get_ollama_models()
print(f"[OLLAMA] Available models: {OLLAMA_MODELS}")


def get_location_from_ip():
    """Auto-detect location from IP address"""
    import requests
    try:
        resp = requests.get("https://ipinfo.io/json", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            city = data.get("city", "")
            region = data.get("region", "")
            if city and region:
                return f"{city}, {region}"
            return city or CONFIG["default_location"]
    except:
        pass
    return CONFIG["default_location"]


# ============================================================================
# GLOBAL STATE
# ============================================================================

class DashboardState:
    """Global state for the dashboard"""
    def __init__(self):
        self.jarvis_active = False
        self.voice_active = False
        self.camera_active = False
        self.hands_active = False

        # Queues for async communication
        self.voice_queue = queue.Queue()
        self.response_queue = queue.Queue()

        # Current data
        self.weather_data = {}
        self.news_data = []
        self.conversation_history = []
        self.last_search_results = []  # Store search results for article fetching
        self.last_search_query = ""  # Track last search query for confirmations
        self.last_article_request = None  # Track last article user tried to summarize

        # Components - using EnhancedJarvisTools with ADA-style capabilities
        self.tools = EnhancedJarvisTools()
        self.voice_system = None
        self.vision_system = None
        self.hand_tracker = None

        # File manager
        self.file_manager = JarvisFileManager()

        # Memory system
        self.memory = None
        if MEMORY_AVAILABLE:
            try:
                self.memory = EnhancedAlexandraMemory()
                logger.info("Memory system initialized")
            except Exception as e:
                logger.warning(f"Could not initialize memory: {e}")

        # RAG system
        self.rag = None
        if RAG_AVAILABLE:
            try:
                self.rag = get_rag()
                logger.info("RAG system initialized")
            except Exception as e:
                logger.warning(f"Could not initialize RAG: {e}")

        # Image generation
        self.flux_pipeline = None

state = DashboardState()

# ============================================================================
# AUTO NEWS REFRESH - Fetches news every 2 hours in background
# ============================================================================
def auto_news_refresh_loop():
    """Background thread that refreshes news feeds every 2 hours"""
    import time
    while True:
        try:
            # Wait 2 hours (7200 seconds)
            time.sleep(7200)
            if state.memory:
                count = state.memory.update_news_feeds(max_per_feed=15)
                logger.info(f"[AUTO NEWS] Fetched {count} new articles")
        except Exception as e:
            logger.error(f"[AUTO NEWS] Error: {e}")

# Start auto-refresh thread
_news_refresh_thread = threading.Thread(target=auto_news_refresh_loop, daemon=True)
_news_refresh_thread.start()
logger.info("[AUTO NEWS] Background news refresh started (every 2 hours)")

# ============================================================================
# WIDGET UPDATE FUNCTIONS
# ============================================================================

async def get_weather_widget(location: str = None) -> str:
    """Get formatted weather widget HTML"""
    try:
        location = location or CONFIG["default_location"]
        weather = await state.tools.weather.get_current_weather(location)

        if "error" in weather:
            return f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px; border-radius: 15px; color: white; text-align: center;">
                <h3>🌤️ Weather</h3>
                <p>Unable to fetch weather</p>
            </div>
            """

        # Weather icon based on description
        desc_lower = weather.get('description', '').lower()
        text_color = "white"
        if 'sun' in desc_lower or 'clear' in desc_lower:
            icon = "☀️"
            gradient = "linear-gradient(135deg, #FF8C00 0%, #FFD700 100%)"
        elif 'cloud' in desc_lower:
            icon = "☁️"
            gradient = "linear-gradient(135deg, #4a5568 0%, #2d3748 100%)"
        elif 'rain' in desc_lower:
            icon = "🌧️"
            gradient = "linear-gradient(135deg, #1a365d 0%, #2c5282 100%)"
        elif 'snow' in desc_lower:
            icon = "❄️"
            gradient = "linear-gradient(135deg, #1a365d 0%, #4a5568 100%)"
        else:
            icon = "🌤️"
            gradient = "linear-gradient(135deg, #4a5568 0%, #2d3748 100%)"

        return f"""
        <div style="background: {gradient};
                    padding: 20px; border-radius: 15px; color: {text_color}; min-height: 150px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; font-size: 18px;">📍 {weather.get('location', location)}</h3>
                    <p style="margin: 5px 0; font-size: 14px; opacity: 0.9;">{weather.get('description', 'N/A')}</p>
                </div>
                <div style="font-size: 40px;">{icon}</div>
            </div>
            <div style="margin-top: 15px;">
                <span style="font-size: 36px; font-weight: bold;">{weather.get('temperature_f', weather.get('temperature', 'N/A'))}°F</span>
                <span style="font-size: 14px; opacity: 0.8; margin-left: 10px;">Feels like {weather.get('feels_like_f', weather.get('feels_like', 'N/A'))}°F</span>
            </div>
            <div style="margin-top: 10px; font-size: 12px; opacity: 0.8;">
                💨 {weather.get('wind_speed', 'N/A')} mph &nbsp;&nbsp; 💧 {weather.get('humidity', 'N/A')}%
            </div>
        </div>
        """
    except Exception as e:
        logger.error(f"Weather widget error: {e}")
        return "<div style='padding: 20px; background: #333; border-radius: 15px; color: white;'>Weather unavailable</div>"


async def get_time_widget() -> str:
    """Get formatted time widget HTML"""
    return get_time_widget_sync()


def get_time_widget_sync() -> str:
    """Sync version of time widget for timer updates"""
    try:
        from zoneinfo import ZoneInfo
        eastern = ZoneInfo("America/New_York")
        now = datetime.now(eastern)

        return f"""
        <div style="background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
                    padding: 20px; border-radius: 15px; color: white; text-align: center; min-height: 150px;">
            <div style="font-size: 48px; font-weight: bold; font-family: 'Courier New', monospace;">
                {now.strftime('%I:%M %p').lstrip('0')}
            </div>
            <div style="font-size: 14px; opacity: 0.8; margin-top: 5px;">
                {now.strftime('%A')}
            </div>
            <div style="font-size: 16px; margin-top: 5px;">
                {now.strftime('%B %d, %Y')}
            </div>
        </div>
        """
    except Exception as e:
        logger.error(f"Time widget error: {e}")
        return "<div style='padding: 20px; background: #333; border-radius: 15px; color: white;'>Time unavailable</div>"


def get_gpu_widget() -> str:
    """Get GPU status widget HTML using nvidia-smi"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 4:
                util = parts[0].strip()
                mem_used = parts[1].strip() if parts[1].strip() != '[N/A]' else '0'
                mem_total = parts[2].strip() if parts[2].strip() != '[N/A]' else '96000'
                temp = parts[3].strip() if parts[3].strip() != '[N/A]' else 'N/A'
                power = parts[4].strip() if len(parts) > 4 and parts[4].strip() != '[N/A]' else 'N/A'

                # Calculate memory percentage
                try:
                    mem_pct = int(float(mem_used) / float(mem_total) * 100)
                    mem_gb = float(mem_used) / 1024
                    total_gb = float(mem_total) / 1024
                except:
                    mem_pct = 0
                    mem_gb = 0
                    total_gb = 96

                # GPU utilization color
                try:
                    util_val = int(util)
                except:
                    util_val = 0

                if util_val > 80:
                    util_color = "#ff4444"
                elif util_val > 50:
                    util_color = "#ffaa00"
                else:
                    util_color = "#00ff88"

                # Memory bar color
                if mem_pct > 90:
                    mem_color = "#ff4444"
                elif mem_pct > 70:
                    mem_color = "#ffaa00"
                else:
                    mem_color = "#00d4ff"

                return f"""
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                            padding: 15px; border-radius: 15px; color: white; border: 1px solid #0f3460;">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 20px; margin-right: 8px;">🖥️</span>
                        <span style="font-size: 14px; font-weight: bold; color: #00d4ff;">DGX SPARK GPU</span>
                    </div>

                    <div style="margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px;">
                            <span>Utilization</span>
                            <span style="color: {util_color}; font-weight: bold;">{util}%</span>
                        </div>
                        <div style="background: #0a0a15; border-radius: 5px; height: 8px; overflow: hidden;">
                            <div style="background: {util_color}; width: {util}%; height: 100%; transition: width 0.3s;"></div>
                        </div>
                    </div>

                    <div style="margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px;">
                            <span>Memory</span>
                            <span style="color: {mem_color};">{mem_gb:.1f} / {total_gb:.0f} GB</span>
                        </div>
                        <div style="background: #0a0a15; border-radius: 5px; height: 8px; overflow: hidden;">
                            <div style="background: {mem_color}; width: {mem_pct}%; height: 100%; transition: width 0.3s;"></div>
                        </div>
                    </div>

                    <div style="display: flex; justify-content: space-between; font-size: 11px; opacity: 0.8; margin-top: 10px;">
                        <span>🌡️ {temp}°C</span>
                        <span>⚡ {power}W</span>
                    </div>
                </div>
                """
        return "<div style='padding: 15px; background: #1a1a2e; border-radius: 15px; color: #666;'>GPU info unavailable</div>"
    except Exception as e:
        logger.error(f"GPU widget error: {e}")
        return "<div style='padding: 15px; background: #1a1a2e; border-radius: 15px; color: #666;'>GPU info unavailable</div>"


async def get_news_widget(topic: str = None) -> str:
    """Get formatted news widget HTML"""
    try:
        headlines = await state.tools.news.get_headlines(topic)

        if not headlines or "error" in headlines[0]:
            return """
            <div style="background: linear-gradient(135deg, #232526 0%, #414345 100%);
                        padding: 20px; border-radius: 15px; color: white;">
                <h3>📰 News</h3>
                <p>Unable to fetch news</p>
            </div>
            """

        news_items = ""
        for h in headlines[:5]:
            title = h.get('title', 'No title')[:80]
            if len(h.get('title', '')) > 80:
                title += "..."
            news_items += f"""
            <div style="padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                <span style="font-size: 13px;">• {title}</span>
            </div>
            """

        return f"""
        <div style="background: linear-gradient(135deg, #232526 0%, #414345 100%);
                    padding: 20px; border-radius: 15px; color: white; min-height: 200px;">
            <h3 style="margin: 0 0 15px 0; font-size: 16px;">📰 Latest News</h3>
            {news_items}
        </div>
        """
    except Exception as e:
        logger.error(f"News widget error: {e}")
        return "<div style='padding: 20px; background: #333; border-radius: 15px; color: white;'>News unavailable</div>"


def get_status_widget(jarvis_on: bool, voice_on: bool, camera_on: bool, hands_on: bool) -> str:
    """Get status indicators widget"""
    def status_dot(active: bool, label: str) -> str:
        color = "#4CAF50" if active else "#666"
        return f"""
        <div style="display: inline-flex; align-items: center; margin-right: 20px;">
            <div style="width: 10px; height: 10px; border-radius: 50%; background: {color};
                        margin-right: 5px; box-shadow: 0 0 {10 if active else 0}px {color};"></div>
            <span style="font-size: 12px; color: {'#fff' if active else '#666'};">{label}</span>
        </div>
        """

    return f"""
    <div style="background: #1a1a2e; padding: 15px; border-radius: 10px;
                display: flex; justify-content: center; flex-wrap: wrap;">
        {status_dot(jarvis_on, "Jarvis")}
        {status_dot(voice_on, "Voice")}
        {status_dot(camera_on, "Camera")}
        {status_dot(hands_on, "Hands")}
    </div>
    """


# ============================================================================
# VOICE WAVEFORM
# ============================================================================

def generate_waveform_image(audio_data: np.ndarray = None, is_speaking: bool = False) -> np.ndarray:
    """Generate a waveform visualization image"""
    width, height = 600, 100
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Background gradient
    for i in range(height):
        img[i, :] = [20 + i//4, 20 + i//4, 40 + i//4]

    # Center line
    cv2.line(img, (0, height//2), (width, height//2), (50, 50, 80), 1)

    if audio_data is not None and len(audio_data) > 0:
        # Real waveform
        samples = np.linspace(0, len(audio_data)-1, width, dtype=int)
        points = []
        for i, s in enumerate(samples):
            y = int(height//2 + audio_data[s] * height//2 * 0.8)
            y = max(5, min(height-5, y))
            points.append((i, y))

        color = (0, 255, 100) if is_speaking else (100, 150, 255)
        for i in range(1, len(points)):
            cv2.line(img, points[i-1], points[i], color, 2)
    else:
        # Animated idle waveform
        t = time.time()
        for i in range(width):
            y = int(height//2 + np.sin(i*0.05 + t*3) * 10 + np.sin(i*0.02 + t*2) * 5)
            color = (60, 80, 120)
            cv2.line(img, (i, height//2), (i, y), color, 1)

    return img


# ============================================================================
# IMAGE GENERATION (FLUX)
# ============================================================================

def load_flux_pipeline():
    """Load FLUX pipeline for image generation"""
    global state

    if state.flux_pipeline is not None:
        return state.flux_pipeline

    try:
        import torch
        from diffusers import FluxPipeline

        logger.info("Loading FLUX pipeline...")
        pipe = FluxPipeline.from_pretrained(
            CONFIG["flux_model"],
            torch_dtype=torch.bfloat16
        ).to("cuda")

        # Load LoRA
        if os.path.exists(CONFIG["lora_path"]):
            logger.info(f"Loading LoRA from {CONFIG['lora_path']}")
            pipe.load_lora_weights(CONFIG["lora_path"])

        state.flux_pipeline = pipe
        logger.info("FLUX pipeline loaded")
        return pipe

    except Exception as e:
        logger.error(f"Failed to load FLUX: {e}")
        return None


def generate_image(
    prompt: str,
    negative_prompt: str = "",
    model_choice: str = "Juggernaut (Nature/Action)",
    use_trigger: bool = False,
    width: int = 768,
    height: int = 1024,
    steps: int = 25,
    seed: int = -1,
    guidance: float = 8.0,
    progress=gr.Progress()
) -> tuple:
    """Generate image using Juggernaut XL or Pony Realism via standalone generator"""
    from gradio_client import Client

    progress(0.1, desc="Connecting to image generator...")

    try:
        # Connect to the standalone image generator
        client = Client("http://localhost:7865", verbose=False)

        progress(0.3, desc="Generating image...")

        logger.info(f"[IMAGE] Generating via API: {prompt[:80]}... Model: {model_choice}")

        # Default negative prompt if empty
        if not negative_prompt:
            negative_prompt = "blurry, low quality, bad anatomy, extra limbs, deformed, ugly, watermark, text"

        # Call the generate function
        result = client.predict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_choice=model_choice,
            use_trigger=use_trigger,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=seed,
            api_name="/generate"
        )

        progress(1.0, desc="Done!")

        # Result is (image_path, info_string)
        if result and len(result) >= 2:
            from PIL import Image
            image = Image.open(result[0])
            return image, result[1]
        else:
            return None, "No image returned"

    except Exception as e:
        logger.error(f"Image generation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, f"Error: {str(e)}"


def unload_image_models():
    """Unload image models to free VRAM"""
    from gradio_client import Client
    try:
        client = Client("http://localhost:7865", verbose=False)
        result = client.predict(api_name="/unload_models")
        logger.info(f"[IMAGE] Models unloaded: {result}")
        return result
    except Exception as e:
        logger.error(f"Error unloading models: {e}")
        return f"Error: {str(e)}"


def load_image_models():
    """Load image models back into VRAM"""
    from gradio_client import Client
    try:
        client = Client("http://localhost:7865", verbose=False)
        result = client.predict(api_name="/load_models")
        logger.info(f"[IMAGE] Models loaded: {result}")
        return result
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return f"Error: {str(e)}"


def generate_img2img(
    input_image,
    prompt: str,
    negative_prompt: str = "",
    model_choice: str = "Pony Realism (Explicit)",
    strength: float = 0.5,
    steps: int = 30,
    guidance: float = 7.0,
    seed: int = -1,
    progress=gr.Progress()
) -> tuple:
    """Transform an image using Pony or Juggernaut via img2img"""
    from gradio_client import Client, handle_file
    import tempfile
    from PIL import Image
    import numpy as np

    if input_image is None:
        return None, "Please upload an image first!"

    progress(0.1, desc="Connecting to image generator...")

    try:
        # Save input image to temp file for API
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            input_image.save(tmp.name)
            temp_path = tmp.name

        client = Client("http://localhost:7865", verbose=False)

        progress(0.3, desc="Transforming image...")

        logger.info(f"[IMG2IMG] Transforming: {prompt[:50]}... Model: {model_choice}, Strength: {strength}")

        if not negative_prompt:
            negative_prompt = "blurry, low quality, bad anatomy, deformed, ugly, watermark"

        # Use handle_file for proper image upload to Gradio API
        result = client.predict(
            input_image=handle_file(temp_path),
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_choice=model_choice,
            strength=strength,
            steps=steps,
            guidance=guidance,
            seed=seed,
            api_name="/img2img"
        )

        progress(1.0, desc="Done!")

        # Cleanup temp file
        import os
        os.unlink(temp_path)

        if result and len(result) >= 2:
            # Result[0] can be a path string or a dict with 'path' key
            img_path = result[0] if isinstance(result[0], str) else result[0].get('path', result[0])
            image = Image.open(img_path)
            return image, result[1]
        else:
            return None, "No image returned"

    except Exception as e:
        logger.error(f"Img2img error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, f"Error: {str(e)}"


# ============================================================================
# CHAT IMAGE GENERATION
# ============================================================================

def generate_image_for_chat(prompt: str, include_alexandra: bool = False) -> tuple:
    """Generate image from chat request - returns (image_path, status_message)"""
    import torch
    import tempfile

    # Load pipeline if needed
    pipe = load_flux_pipeline()
    if pipe is None:
        return None, "Image generation not available - FLUX model not loaded"

    try:
        # Always unload existing LoRA first
        try:
            pipe.unload_lora_weights()
            logger.info("Unloaded existing LoRA weights")
        except Exception as e:
            logger.warning(f"Could not unload LoRA: {e}")

        # Check if NSFW content requested
        nsfw_keywords = ['nude', 'naked', 'topless', 'nsfw', 'sexy', 'lingerie', 'underwear', 'bra', 'panties', 'bikini', 'erotic', 'sensual', 'seductive', 'undressed', 'bedroom']
        is_nsfw = any(kw in prompt.lower() for kw in nsfw_keywords)

        # Load appropriate LoRA(s)
        if include_alexandra:
            # Use Alexandra LoRA with trigger word
            lora_path = CONFIG["lora_path"]
            if os.path.exists(lora_path):
                pipe.load_lora_weights(lora_path, adapter_name="alexandra")
                prompt = f"{CONFIG['trigger_word']}, {prompt}"
                logger.info(f"Using Alexandra LoRA, prompt: {prompt[:50]}...")

                # Also load uncensored LoRA for NSFW content
                if is_nsfw:
                    uncensored_path = CONFIG["lora_path_uncensored"]
                    if os.path.exists(uncensored_path):
                        pipe.load_lora_weights(uncensored_path, adapter_name="uncensored")
                        pipe.set_adapters(["alexandra", "uncensored"], adapter_weights=[1.0, 0.8])
                        logger.info(f"Also loaded uncensored LoRA for NSFW content")
        else:
            # Use uncensored LoRA (or no LoRA for general images)
            lora_path = CONFIG["lora_path_uncensored"]
            if os.path.exists(lora_path):
                pipe.load_lora_weights(lora_path)
                logger.info(f"Using uncensored LoRA for: {prompt[:50]}...")
            else:
                logger.info(f"No uncensored LoRA, using base model for: {prompt[:50]}...")

        seed = int(time.time()) % 2147483647
        generator = torch.Generator("cuda").manual_seed(seed)

        image = pipe(
            prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=generator,
            height=768,
            width=768,
        ).images[0]

        # Save to temp file
        temp_path = tempfile.mktemp(suffix=".png")
        image.save(temp_path)

        return temp_path, f"Generated image (seed: {seed})"

    except Exception as e:
        logger.error(f"Chat image generation error: {e}")
        return None, f"Failed to generate image: {str(e)}"


def detect_image_request(message: str) -> tuple:
    """Detect if message is an image generation request.
    Returns (is_image_request, prompt, include_alexandra)"""
    import re
    msg_lower = message.lower()

    # Regex patterns for flexible image request detection
    image_patterns = [
        r"(?:can you |could you |please |)(?:create|generate|make|draw|paint)(?: me)?(?: a| an)? (?:image|picture|photo|artwork|illustration)(?: of| showing| with)? (.+)",
        r"(?:show me |i want |i need |i'd like )(?: a| an)? (?:image|picture|photo)(?: of)? (.+)",
        r"(?:draw|paint|sketch)(?: me)? (.+)",
    ]

    prompt = None
    for pattern in image_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            prompt = match.group(1).strip()
            prompt = prompt.rstrip('?.!')
            break

    # Fallback: simple keyword check
    if not prompt:
        simple_triggers = ["generate image", "create image", "make image", "draw a ", "paint a ",
                          "generate a ", "create a ", "make a ", "nsfw image", "nsfw photo",
                          "nude image", "nude photo", "sexy image", "sexy photo"]
        for trigger in simple_triggers:
            if trigger in msg_lower:
                idx = msg_lower.find(trigger)
                prompt = message[idx + len(trigger):].strip()
                for word in ["of ", "showing "]:
                    if prompt.lower().startswith(word):
                        prompt = prompt[len(word):]
                break

    # Direct NSFW triggers - bypass LLM entirely
    if not prompt:
        nsfw_direct = ["nude ", "naked ", "topless ", "nsfw ", "lingerie ", "sexy "]
        for trigger in nsfw_direct:
            if trigger in msg_lower:
                # Extract everything after the NSFW keyword
                idx = msg_lower.find(trigger)
                prompt = message[idx:].strip()
                break

    if not prompt:
        return False, None, False

    # Check if user wants themselves in the image
    self_keywords = [" me ", " myself", " me,", " me.", "of me", "with me", "alexandra", "myself ", "picture of me", "image of me", "photo of me"]
    include_alexandra = any(kw in msg_lower for kw in self_keywords)

    # Also check if prompt starts with "me "
    if prompt.lower().startswith("me ") or prompt.lower() == "me":
        include_alexandra = True
        prompt = prompt[3:] if prompt.lower().startswith("me ") else "portrait"

    logger.info(f"Image request: '{prompt}', include_alexandra={include_alexandra}")
    return True, prompt, include_alexandra


# ============================================================================
# CHAT FUNCTIONS
# ============================================================================

async def chat_with_jarvis(message: str, history: list) -> tuple:
    """Process chat message and get response. Returns (history, clear_input, image_path)"""
    import aiohttp

    if not message.strip():
        return history, "", None

    # Add user message to history
    history.append({"role": "user", "content": message})

    # Check if this is an image generation request
    is_image_request, image_prompt, include_alexandra = detect_image_request(message)
    if is_image_request:
        mode = "with you" if include_alexandra else "general"
        logger.info(f"Image request detected: {image_prompt} (mode: {mode})")
        history.append({"role": "assistant", "content": f"🎨 Generating image: *{image_prompt}*... This may take a moment."})

        # Generate the image
        image_path, status = generate_image_for_chat(image_prompt, include_alexandra=include_alexandra)

        if image_path:
            history.append({"role": "assistant", "content": f"Here's your image! {status}"})
            return history, "", image_path
        else:
            history.append({"role": "assistant", "content": f"Sorry, I couldn't generate that image. {status}"})
            return history, "", None

    # Check if this is a tool query (weather, news, search, time, file ops, apps, etc.)
    tool_keywords = ['weather', 'news', 'search for', 'look up', 'what time', 'current time']
    # JARVIS-style command keywords
    jarvis_keywords = [
        'create folder', 'make folder', 'new folder', 'create directory',
        'create file', 'make file', 'new file', 'write file',
        'list files', 'show files', 'what files', 'list directory',
        'read file', 'open file', 'show file', 'show me the file',
        'edit file', 'update file', 'modify file', 'write to',
        'delete file', 'remove file', 'delete folder', 'remove folder',
        'open chrome', 'open firefox', 'open terminal', 'open vscode', 'open browser',
        'open notepad', 'open calculator', 'launch app', 'start app',
        'go to website', 'open website', 'open url', 'visit', 'navigate to',
        'google search', 'search google',
        'calculate', 'solve', 'what is', 'compute', 'math',
        'run code', 'execute code', 'run python', 'execute python',
        'capture screen', 'screenshot', 'screen capture', 'take screenshot'
    ]
    # Proactive system keywords
    proactive_keywords = [
        'remind me', 'set reminder', 'set a reminder', 'reminder',
        'good morning', 'morning routine', 'morning briefing',
        'system status', 'system check', 'check system', 'how is the system',
        'gpu status', 'gpu temperature', 'memory usage', 'cpu usage',
        'training status', 'training progress', 'how is training',
        'shutdown routine', 'good night',
        'my schedule', 'what\'s on my schedule', 'today\'s schedule',
        'my calendar', 'my events', 'list my events', 'show my calendar',
        'show events', 'show my events', 'list events',
        'what\'s on my calendar', 'events this week', 'week on my calendar',
        'events for this week', 'calendar for', 'events for',
        'my reminders', 'what reminders', 'show reminders',
        'what\'s on my screen', 'read my screen', 'look at my screen',
        'schedule a', 'add to calendar', 'add to my calendar', 'create event',
        'create an event', 'create a calendar event', 'create calendar event',
        'new calendar event', 'new event', 'add a calendar event',
        'calendar event for', 'book a', 'put on calendar', 'add event',
        'schedule meeting', 'schedule appointment'
    ]
    # Navigation/directions keywords
    navigation_keywords = [
        'directions to', 'how do i get to', 'navigate to', 'take me to',
        'route to', 'driving to', 'drive to', 'way to',
        'traffic to', 'traffic conditions', 'how long to get to',
        'eta to', 'time to get to', 'distance to'
    ]
    # Hand tracking / finger counting keywords
    finger_keywords = [
        'how many fingers', 'fingers am i holding', 'count my fingers',
        'what gesture', 'what hand gesture', 'recognize my gesture',
        'how many am i holding up', 'fingers up'
    ]

    # Vision/camera keywords - triggers camera capture + vision analysis
    # Comprehensive list for all Qwen2.5-VL-72B capabilities
    vision_keywords = [
        # Direct vision queries
        'what do you see', 'what can you see', 'what are you seeing',
        'describe what you see', 'look at me', 'can you see me',
        'what am i doing', 'what am i holding', 'what is in front of you',
        'describe the scene', 'what\'s in the camera', 'look at the camera',
        # Object identification
        'what is this', 'what\'s this', 'what is that', 'what\'s that',
        'what do i have', 'in my hand', 'identify this', 'identify that',
        'look at this', 'look at that', 'see this', 'see that',
        'do you recognize', 'recognize this', 'what am i showing',
        'what\'s in my hand', 'tell me what this is', 'what object',
        # Text/OCR reading
        'read this', 'read the text', 'read the label', 'what does it say',
        'what does this say', 'can you read', 'read what', 'able to read',
        'read the sign', 'read the writing', 'what is written', 'what text',
        'read the document', 'read the screen', 'read the paper',
        # Counting
        'how many', 'count the', 'count how many', 'number of',
        # Colors
        'what color', 'what colour', 'identify the color', 'which color',
        # People/faces
        'who is this', 'who am i', 'recognize this person', 'who is that',
        'how do i look', 'what emotion', 'how am i feeling', 'my expression',
        'am i smiling', 'do i look', 'facial expression',
        # Clothing/appearance
        'what am i wearing', 'describe my outfit', 'my clothes', 'what shirt',
        'what dress', 'describe my appearance', 'how is my hair',
        # Food
        'what food', 'what am i eating', 'identify this food', 'what dish',
        'what meal', 'what drink', 'what fruit', 'what vegetable',
        # Animals
        'what animal', 'what kind of dog', 'what breed', 'identify this animal',
        'what bird', 'what pet', 'what species',
        # Plants
        'what plant', 'what flower', 'what tree', 'identify this plant',
        # Brands/logos
        'what brand', 'what logo', 'identify the brand', 'which company',
        'what product', 'what model',
        # Barcodes/QR codes
        'scan this', 'read the barcode', 'read the qr', 'scan the code',
        'what does this code', 'qr code',
        # Spatial/location
        'where is', 'what\'s next to', 'what\'s behind', 'what\'s on the',
        'location of', 'position of', 'find the',
        # Comparison
        'compare these', 'what\'s the difference', 'which one is',
        'are these the same', 'spot the difference',
        # Size/measurement
        'how big', 'how tall', 'how wide', 'what size', 'how large',
        # Quality/condition
        'is this good', 'check the quality', 'is this damaged', 'condition of',
        # Math/equations
        'solve this', 'what\'s the equation', 'calculate this', 'math problem',
        # General triggers
        'show me', 'look here', 'check this out', 'examine this',
        'analyze this', 'inspect this', 'take a look'
    ]

    # Snapshot keywords - capture and save image
    snapshot_keywords = [
        'take a snapshot', 'take snapshot', 'take a photo', 'take photo',
        'capture image', 'save image', 'take a picture', 'take picture',
        'snap a photo', 'capture a photo'
    ]

    msg_lower = message.lower()

    # Don't treat personal questions as tool queries (they should use memory)
    is_personal_question = 'my ' in msg_lower and any(word in msg_lower for word in [
        'aunt', 'uncle', 'mother', 'father', 'mom', 'dad', 'sister', 'brother',
        'son', 'daughter', 'name', 'birthday', 'favorite', 'address', 'phone',
        'wife', 'husband', 'friend', 'boss', 'pet', 'dog', 'cat', 'car'
    ])

    # Math/calculation questions - don't web search, let LLM calculate
    import re as _re
    is_math_question = bool(_re.search(r'[\d\+\-\*\/\^\√\×\÷\=]', message)) or any(kw in msg_lower for kw in [
        'square root', 'sqrt', 'calculate', 'compute', 'solve', 'equation',
        'plus', 'minus', 'times', 'divided by', 'multiply', 'divide',
        'sum of', 'product of', 'factorial', 'percent', 'percentage',
        'power of', 'squared', 'cubed', 'to the power'
    ])

    is_tool_query = any(kw in msg_lower for kw in tool_keywords) and not is_personal_question
    is_snapshot_request = any(kw in msg_lower for kw in snapshot_keywords)
    is_jarvis_command = any(kw in msg_lower for kw in jarvis_keywords)
    is_navigation_request = any(kw in msg_lower for kw in navigation_keywords)
    is_proactive_command = any(kw in msg_lower for kw in proactive_keywords)
    is_finger_query = any(kw in msg_lower for kw in finger_keywords)
    is_vision_query = any(kw in msg_lower for kw in vision_keywords)

    response = None
    nav_data = None  # Will hold navigation data for widget

    # Handle finger/gesture queries - use VL model if loaded, otherwise hand tracking
    if is_finger_query:
        # Check if VL model is loaded - use it for better accuracy
        from alexandra_model import current_model_type, model as loaded_model
        if current_model_type == "vision" and loaded_model is not None:
            # Route to vision query handler - VL model can see and count fingers
            is_vision_query = True
            logger.info("[FINGER QUERY] VL model loaded - routing to vision analysis")
        else:
            # Fall back to hand tracking
            global _finger_count, _last_gesture, _hand_tracking_enabled
            if _hand_tracking_enabled and _finger_count is not None:
                # Use the live hand tracking data
                if 'gesture' in msg_lower:
                    gesture = _last_gesture if _last_gesture else "None detected"
                    response = f"I see your gesture is: {gesture}. You're holding up {_finger_count} fingers."
                else:
                    # Finger count query
                    if _finger_count == 0:
                        response = "I don't see any fingers up right now. Make sure your hand is visible to the camera."
                    elif _finger_count == 1:
                        response = "You're holding up 1 finger."
                    else:
                        response = f"You're holding up {_finger_count} fingers."
                logger.info(f"[HAND TRACKING] Finger query answered: {_finger_count} fingers, gesture: {_last_gesture}")
            else:
                response = "Hand tracking is not currently enabled. Enable it in the Vision tab to count fingers in real-time."

    # Handle snapshot requests - capture and save image
    if is_snapshot_request and response is None:
        try:
            frame = capture_from_server_camera()
            if frame is not None:
                import cv2
                # datetime already imported at top of file
                # Save snapshot with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_dir = "/tmp/jarvis_snapshots"
                os.makedirs(snapshot_dir, exist_ok=True)
                snapshot_path = f"{snapshot_dir}/snapshot_{timestamp}.jpg"

                # Convert RGB to BGR for saving
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(snapshot_path, frame_bgr)

                response = f"📸 Snapshot saved! I captured what I see right now."
                logger.info(f"[SNAPSHOT] Saved to {snapshot_path}")

                # Return the image path so it can be displayed
                history.append({"role": "assistant", "content": response})
                return history, "", snapshot_path
            else:
                response = "I couldn't capture a snapshot - the camera doesn't seem to be working. Try going to the Camera tab first."
        except Exception as e:
            logger.error(f"[SNAPSHOT] Error: {e}")
            response = f"Sorry, I had trouble taking a snapshot: {str(e)}"

    # Handle vision/camera queries - capture from camera and analyze
    if is_vision_query and response is None:
        try:
            import cv2
            # Capture FRESH frame from camera - read multiple times to flush buffer
            cap = get_camera()
            frame = None
            if cap and cap.isOpened():
                # Flush camera buffer by reading a few frames
                for _ in range(3):
                    ret, temp_frame = cap.read()
                if ret and temp_frame is not None:
                    frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
                    logger.info("[VISION] Captured fresh frame for analysis")

            # Fallback to regular capture
            if frame is None:
                frame = capture_from_server_camera()
                logger.info("[VISION] Using buffered frame (camera may be slow)")

            if frame is not None:
                # Use Gemini vision if enabled (faster, cloud-based)
                if _use_gemini and GEMINI_AVAILABLE:
                    logger.info("[VISION] Using Gemini vision for camera analysis...")
                    result = gemini_vision(frame, message)
                    if result:
                        response = result
                        logger.info("[VISION] Gemini vision query answered")
                    else:
                        response = "I had trouble analyzing the image with Gemini. Try again."
                else:
                    # Check if VL model is loaded via alexandra_model (Settings tab)
                    from alexandra_model import current_model_type, generate_vision_response, model as am_model
                    if current_model_type == "vision" and am_model is not None:
                        logger.info("[VISION] Using loaded Qwen2.5-VL-72B for camera analysis...")
                        result = generate_vision_response(message, frame)
                        if result and "Error" not in result:
                            response = result
                            logger.info("[VISION] VL model vision query answered")
                        else:
                            response = f"I had trouble analyzing the image: {result}"
                    # Fall back to standalone alexandra_vision module
                    elif QWEN_VISION_READY:
                        from alexandra_vision import VISION_AVAILABLE, analyze_image as vision_analyze
                        if VISION_AVAILABLE:
                            logger.info("[VISION] Using local Qwen2-VL for camera analysis...")
                            result = vision_analyze(frame, message)
                            response = result
                            logger.info("[VISION] Local vision query answered")
                        else:
                            response = "My local vision model isn't loaded yet. Either select 'Gemini' mode or go to the Camera & Vision tab and click 'Load Vision Model'."
                    else:
                        # No vision available at all
                        if GEMINI_AVAILABLE:
                            response = "Vision is available with Gemini - select 'Gemini' in the Model dropdown to use cloud vision."
                        else:
                            response = "Vision capabilities aren't available. Make sure Gemini is configured or the local vision module is installed."
            else:
                response = "I can't see anything - the camera doesn't seem to be capturing. Try clicking 'Start Live Feed' in the Camera tab."
        except Exception as e:
            logger.error(f"[VISION] Chat vision error: {e}")
            response = f"I had trouble seeing: {str(e)}"

    # Handle navigation/directions requests
    if is_navigation_request and response is None:
        try:
            if NAVIGATION_AVAILABLE and navigator:
                nav_response, route = nav_respond(message)
                response = nav_response
                if route:
                    nav_data = {
                        "info": nav_response,
                        "map_url": route.map_url,
                        "map_html": route.map_html if hasattr(route, 'map_html') else "",
                        "show_widget": True
                    }
                logger.info(f"[NAVIGATION] Got directions response")
            else:
                response = "Navigation system not available. Try: 'How do I get to Pittsburgh?'"
        except Exception as e:
            logger.error(f"[NAVIGATION] Error: {e}")
            response = f"Sorry, I couldn't get directions: {str(e)}"

    # Handle proactive commands (reminders, routines, system status)
    if is_proactive_command:
        try:
            # Reminders
            if 'remind me' in msg_lower or 'set reminder' in msg_lower:
                if SCHEDULER_AVAILABLE and scheduler:
                    response = scheduler.add_reminder(message)
                else:
                    response = "Reminder system not available"

            # Calendar event creation
            elif any(phrase in msg_lower for phrase in [
                'schedule a', 'add to calendar', 'add to my calendar',
                'create event', 'create an event', 'create a calendar event',
                'create calendar event', 'new calendar event', 'new event',
                'add a calendar event', 'book a', 'put on calendar',
                'add event', 'schedule meeting', 'schedule appointment',
                'calendar event for'
            ]):
                if SCHEDULER_AVAILABLE and scheduler:
                    response = scheduler.add_event(message)
                else:
                    response = "Calendar not available"

            # Morning routine
            elif 'good morning' in msg_lower or 'morning routine' in msg_lower or 'morning briefing' in msg_lower:
                if SCHEDULER_AVAILABLE and scheduler:
                    response = scheduler.get_morning_briefing()
                elif PROACTIVE_AVAILABLE and proactive:
                    results = await proactive.routine_manager.execute_routine("morning")
                    response = "\n".join(results) if results else "Morning routine completed"
                else:
                    response = "Routines not available"

            # System status
            elif 'system status' in msg_lower or 'system check' in msg_lower or 'check system' in msg_lower:
                if PROACTIVE_AVAILABLE and proactive:
                    response = proactive.system_monitor.get_summary()
                else:
                    response = "System monitoring not available"

            # GPU status
            elif 'gpu' in msg_lower and ('status' in msg_lower or 'temperature' in msg_lower or 'temp' in msg_lower):
                if PROACTIVE_AVAILABLE and proactive:
                    stats = proactive.system_monitor.collect_stats()
                    gpu_pct = (stats.gpu_memory_used / stats.gpu_memory_total * 100) if stats.gpu_memory_total > 0 else 0
                    response = f"GPU is at {stats.gpu_usage:.0f}% utilization, using {stats.gpu_memory_used:.1f} of {stats.gpu_memory_total:.0f} GB ({gpu_pct:.0f}%), temperature is {stats.gpu_temp:.0f}°C"
                else:
                    response = "GPU monitoring not available"

            # Training status
            elif 'training' in msg_lower and ('status' in msg_lower or 'progress' in msg_lower):
                if PROACTIVE_AVAILABLE and proactive:
                    response = proactive.training_monitor.get_status_summary()
                else:
                    response = "Training monitor not available"

            # Schedule / Calendar events
            elif 'schedule' in msg_lower or 'calendar' in msg_lower or 'events' in msg_lower:
                if SCHEDULER_AVAILABLE and scheduler:
                    # Check for specific day or date
                    day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                    import re
                    has_date = re.search(r'\d{1,2}(?:st|nd|rd|th)?', msg_lower)
                    has_day = any(day in msg_lower for day in day_names)

                    if 'week' in msg_lower:
                        response = scheduler.get_week_schedule()
                    elif has_day or has_date or 'today' in msg_lower or 'tomorrow' in msg_lower:
                        # Get events for specific day
                        response = scheduler.get_day_schedule(msg_lower)
                    else:
                        response = scheduler.get_week_schedule()  # Default to week
                else:
                    response = "Schedule not available"

            # Show reminders
            elif 'reminder' in msg_lower and ('my' in msg_lower or 'show' in msg_lower or 'what' in msg_lower):
                if SCHEDULER_AVAILABLE and scheduler:
                    response = scheduler.reminders.get_summary()
                else:
                    response = "Reminders not available"

            # Screen reading
            elif 'screen' in msg_lower and ('what' in msg_lower or 'read' in msg_lower or 'look' in msg_lower):
                if SCREEN_AVAILABLE and jarvis_screen:
                    response = jarvis_screen.what_am_i_looking_at()
                else:
                    response = "Screen capture not available"

            # STORED NEWS - Query downloaded news from memory
            elif any(kw in msg_lower for kw in ['stored news', 'downloaded news', 'saved news', 'news in memory',
                                                  'political headlines', 'show me headlines', 'what headlines',
                                                  'recent headlines', 'your news', 'cached news']):
                if state.memory:
                    try:
                        # Extract topic if specified
                        topic = None
                        for t in ['political', 'politics', 'tech', 'technology', 'sports', 'business', 'world', 'science']:
                            if t in msg_lower:
                                topic = t
                                break

                        # Query the news collection
                        query = topic if topic else "latest headlines news"
                        news_results = state.memory.news.query(query_texts=[query], n_results=10)

                        if news_results and news_results['documents'] and news_results['documents'][0]:
                            headlines = []
                            for i, doc in enumerate(news_results['documents'][0][:7], 1):
                                # Extract just the title (first line)
                                title = doc.split('\n')[0][:100]
                                headlines.append(f"{i}. {title}")

                            topic_str = f" {topic}" if topic else ""
                            response = f"Here are the{topic_str} headlines I have stored, ma'am:\n\n" + "\n".join(headlines)
                            response += f"\n\nI have {state.memory.news.count()} articles stored. Would you like me to tell you more about any of these?"
                        else:
                            response = "I don't have any news articles stored yet. Would you like me to refresh the news feeds?"
                    except Exception as e:
                        logger.error(f"[STORED NEWS] Error: {e}")
                        response = "I encountered an error accessing stored news."
                else:
                    response = "News memory system not available."

            # Shutdown routine
            elif 'shutdown' in msg_lower or 'good night' in msg_lower:
                if PROACTIVE_AVAILABLE and proactive:
                    results = await proactive.routine_manager.execute_routine("shutdown")
                    response = "\n".join(results) if results else "Shutdown routine completed"
                else:
                    response = "Routines not available"

            logger.info(f"[PROACTIVE] Command result: {response[:100] if response else 'None'}...")
        except Exception as e:
            logger.error(f"[PROACTIVE] Command error: {e}")
            response = None

    # Handle JARVIS-style commands
    if not response and is_jarvis_command:
        try:
            response = await state.tools.handle_command(message)
            logger.info(f"[TOOLS] JARVIS command result: {response[:100] if response else 'None'}...")
        except Exception as e:
            logger.error(f"[TOOLS] JARVIS command error: {e}")
            response = None

    # Fall back to basic tool queries
    if not response and is_tool_query:
        response = await state.tools.handle_query(message)

    # Explicit web search for current information
    web_search_keywords = [
        # Explicit search requests
        'search the web', 'search the internet', 'search online', 'look up online', 'google',
        'look up', 'find out about',
        # News and current events (specific phrases)
        'current news', 'latest news', 'breaking news', 'what happened with',
        'news about', 'update on', 'updates on', 'headlines', 'top stories',
        # News sources
        'from cnn', 'from bbc', 'from nyt', 'from fox', 'from msnbc', 'from reuters',
        'cnn news', 'bbc news', 'political news', 'world news', 'tech news',
        # Incidents (only with context)
        'the crash', 'the accident', 'the shooting', 'the attack', 'the fire',
        'helicopter crash', 'plane crash', 'car crash',
        # Sports and live events
        'who is playing', 'playing tonight', 'game tonight', 'game today',
        'sunday night football', 'monday night football', 'thursday night football',
        'nfl game', 'nba game', 'mlb game', 'nhl game',
        'who won the', 'score of the', 'final score',
        'what time is the game',
        # Weather forecasts
        'weather tomorrow', 'forecast', 'weather advisory', 'weather warning',
        'weather this week', 'weather weekend'
    ]
    # UNIVERSAL FOLLOW-UP DETECTION - works for ANY topic
    # Follow-up phrases that should trigger a new search
    followup_phrases = [
        'tell me more', 'more about', 'what about', 'how about', 'and what about',
        'anything else', 'what else', 'can you find', 'can you tell me', 'can you search',
        'look up', 'search for', 'find out', 'do you know', 'give me more',
        'expand on', 'elaborate', 'details about', 'info on', 'information about',
        'update on', 'latest on', 'news about', 'status of', 'what happened',
        'did they', 'did he', 'did she', 'is there', 'are there', 'was there'
    ]
    # Short confirmations - these should NOT trigger a literal search, just let LLM handle
    confirmation_phrases = ['yes', 'yeah', 'yep', 'please do', 'go ahead', 'sure', 'ok', 'okay',
                           'yes please', 'please', 'do it', 'alright', 'sounds good']
    is_just_confirmation = msg_lower.strip() in confirmation_phrases or len(msg_lower.strip()) < 15 and any(c in msg_lower for c in confirmation_phrases)

    is_followup = any(phrase in msg_lower for phrase in followup_phrases) and not is_just_confirmation

    # Question words that indicate information-seeking
    question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which', 'whose',
                      'is', 'are', 'was', 'were', 'will', 'would', 'could', 'should',
                      'does', 'do', 'did', 'can', 'has', 'have', 'had']
    starts_with_question = any(msg_lower.startswith(w + ' ') or msg_lower.startswith(w + "'") for w in question_words)

    # Sports teams for enhanced sports queries
    sports_teams = ['steelers', 'eagles', 'penguins', 'pirates', 'patriots', 'cowboys', 'ravens', 'browns',
                    'bengals', 'chiefs', 'bills', 'dolphins', 'jets', 'giants', 'commanders', 'bears',
                    'lions', 'packers', 'vikings', 'saints', 'buccaneers', 'falcons', 'panthers',
                    'cardinals', 'rams', 'seahawks', '49ers', 'chargers', 'raiders', 'broncos', 'texans',
                    'colts', 'jaguars', 'titans', 'nfl', 'nba', 'mlb', 'nhl', 'playoff', 'super bowl']

    # Topics that definitely need current info (not just from training data)
    current_info_topics = ['news', 'weather', 'score', 'game', 'playing', 'schedule', 'price', 'stock',
                           'election', 'vote', 'result', 'winner', 'update', 'latest', 'current', 'today',
                           'yesterday', 'tomorrow', 'tonight', 'this week', 'last week', 'recent',
                           'injury', 'status', 'trade', 'signed', 'released', 'died', 'dead', 'killed',
                           'crash', 'accident', 'fire', 'shooting', 'attack', 'war', 'earthquake'] + sports_teams

    needs_current_info = any(topic in msg_lower for topic in current_info_topics)

    # Determine if web search is needed:
    # 1. Explicit search keywords OR
    # 2. Follow-up request OR
    # 3. Question that needs current info OR
    # 4. Any question starting with question word (broad catch-all for info queries)
    # BUT NOT for short confirmations like "yes", "yeah", "please" - let LLM handle those
    is_info_question = (starts_with_question and needs_current_info) or is_followup

    # is_personal_question already defined above - use it to skip web search
    # Also skip if it's just a short confirmation - don't search for "yes please" literally
    # Skip math questions - let LLM calculate directly
    needs_web_search = (any(kw in msg_lower for kw in web_search_keywords) or is_info_question or is_followup) and not is_personal_question and not is_just_confirmation and not is_math_question

    # Log for debugging
    logger.info(f"[SEARCH TRIGGER] followup={is_followup}, confirmation={is_just_confirmation}, needs_search={needs_web_search}")

    web_search_failed = False
    web_search_results = None
    article_content = None

    # Check if user wants to summarize/read an article
    summarize_keywords = ['summarize', 'summary', 'read the article', 'tell me about the article',
                          'what does the article say', 'explain the article', 'full article',
                          'more about', 'tell me more', 'deep dive', 'details on']
    wants_article = any(kw in msg_lower for kw in summarize_keywords)

    if wants_article:
        logger.info(f"[ARTICLE] Detected article request. Stored results: {len(state.last_search_results) if state.last_search_results else 0}")

    # Parse which article number the user wants
    article_index = 0  # Default to first
    # Comprehensive number mapping - check longer patterns first to avoid partial matches
    number_words = {
        'number three': 2, 'number two': 1, 'number one': 0, 'number four': 3, 'number five': 4,
        'number 3': 2, 'number 2': 1, 'number 1': 0, 'number 4': 3, 'number 5': 4,
        'article three': 2, 'article two': 1, 'article one': 0, 'article four': 3, 'article five': 4,
        'article 3': 2, 'article 2': 1, 'article 1': 0, 'article 4': 3, 'article 5': 4,
        'the third': 2, 'the second': 1, 'the first': 0, 'the fourth': 3, 'the fifth': 4,
        'third one': 2, 'second one': 1, 'first one': 0, 'fourth one': 3, 'fifth one': 4,
        'one': 0, 'first': 0, 'two': 1, 'second': 1, 'three': 2, 'third': 2,
        'four': 3, 'fourth': 3, 'five': 4, 'fifth': 4,
        '#1': 0, '#2': 1, '#3': 2, '#4': 3, '#5': 4,
        '1': 0, '2': 1, '3': 2, '4': 3, '5': 4,
    }
    for word, idx in number_words.items():
        if word in msg_lower:
            article_index = idx
            logger.info(f"[ARTICLE] User requested article #{idx + 1}")
            break

    # Handle confirmations like "yes please" - retry last failed article fetch
    if is_just_confirmation and state.last_article_request:
        logger.info(f"[ARTICLE] User confirmed, retrying last article request: {state.last_article_request}")
        article_index = state.last_article_request.get('index', 0)
        wants_article = True  # Force article fetch mode

    # If user wants article content and we have a URL from previous search, fetch it
    if wants_article and state.last_search_results:
        logger.info(f"[ARTICLE] User wants article #{article_index + 1}, have {len(state.last_search_results)} stored results")
        # Log titles of stored results for debugging
        for i, r in enumerate(state.last_search_results[:5]):
            logger.info(f"[ARTICLE] Stored result #{i+1}: {r.get('title', 'Unknown')[:60]}")
        try:
            from jarvis_tools import WebSearchService
            web_service = WebSearchService()

            # Get the specific article the user requested
            if article_index < len(state.last_search_results):
                result = state.last_search_results[article_index]
                url = result.get('href') or result.get('url') or result.get('link', '')
                title = result.get('title', 'Article')

                if url and ('http' in url):
                    logger.info(f"[ARTICLE] Fetching article #{article_index + 1}: {title}")
                    logger.info(f"[ARTICLE] URL: {url}")

                    # Store request in case fetch fails
                    state.last_article_request = {'index': article_index, 'title': title, 'url': url}

                    # Use await since we're in an async function
                    article_data = await web_service.fetch_article(url)
                    if article_data.get('success'):
                        article_content = article_data.get('content', '')
                        logger.info(f"[ARTICLE] Fetched {len(article_content)} chars from {url}")
                    else:
                        logger.warning(f"[ARTICLE] Failed to fetch: {article_data.get('error')}")
                        # Try with a web search for the article title instead
                        article_content = f"I couldn't fetch the full article, but here's what I know about '{title}': " + result.get('body', result.get('snippet', ''))
                else:
                    logger.warning(f"[ARTICLE] No valid URL in result: {result}")
            else:
                logger.warning(f"[ARTICLE] Article index {article_index} out of range (have {len(state.last_search_results)} results)")
        except Exception as e:
            logger.error(f"[ARTICLE] Error fetching article: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Debug logging for web search triggering
    logger.info(f"[WEB SEARCH DEBUG] is_info_question={is_info_question}, needs_web_search={needs_web_search}, response={response is None or len(str(response or '')) < 50}")
    logger.info(f"[WEB SEARCH DEBUG] message: {msg_lower[:100]}")
    logger.info(f"[WEB SEARCH DEBUG] wants_article={wants_article}, has_stored_results={bool(state.last_search_results)}")

    # CRITICAL: Skip web search if we're fetching a specific article - don't overwrite last_search_results!
    if wants_article and state.last_search_results and needs_web_search:
        logger.info(f"[WEB SEARCH] SKIPPED - handling article request, preserving {len(state.last_search_results)} stored results")
    elif needs_web_search and (not response or len(response) < 50) and not (wants_article and state.last_search_results):
        logger.info(f"[WEB SEARCH] Triggering search for: {message}")
        try:
            from ddgs import DDGS

            # Clean up the search query
            search_query = message.lower()
            for phrase in ['search the web for', 'search the internet for', 'search online for',
                          'search and see', 'search for', 'look up online', 'look up', 'find out',
                          'can you tell me', 'can you find', 'what is', 'who is', 'tell me about',
                          'i want to know', 'please find']:
                search_query = search_query.replace(phrase, '')
            search_query = search_query.strip()

            # For weather queries, add location
            is_weather_query = 'weather' in search_query or 'forecast' in search_query
            if is_weather_query and 'tomorrow' not in search_query:
                location = CONFIG.get("default_location", "Ridgway, PA")
                search_query = f"{location} weather forecast {search_query}"

            # For sports queries, enhance with schedule keywords
            sports_team_names = ['steelers', 'eagles', 'penguins', 'pirates', 'patriots', 'cowboys',
                                'ravens', 'browns', 'bengals', 'chiefs', 'bills', 'dolphins', 'jets',
                                'giants', 'commanders', 'bears', 'lions', 'packers', 'vikings']
            is_sports_query = any(team in search_query for team in sports_team_names)
            if is_sports_query and ('playing' in search_query or 'game' in search_query or 'time' in search_query):
                # Add "NFL schedule" for football teams to get better results
                # Use current year for accurate results
                from datetime import datetime as _dt
                current_year = _dt.now().year
                # NFL season spans two years - use the season start year
                nfl_season_year = current_year if _dt.now().month >= 8 else current_year - 1
                search_query = f"{search_query} NFL schedule {nfl_season_year}"

            logger.info(f"[WEB SEARCH] Searching for: {search_query}")

            ddgs = DDGS()

            # Check if this is a news/sports-related query (both need current info)
            # IMPORTANT: Include 'news', 'headlines', 'political' to ensure ddgs.news() returns actual article URLs
            news_keywords = ['news', 'headlines', 'headline', 'political', 'politics', 'today',
                           'crash', 'accident', 'killed', 'dead', 'injured', 'breaking',
                           'shooting', 'attack', 'fire', 'storm', 'earthquake', 'election',
                           'trump', 'biden', 'president', 'congress', 'war', 'police',
                           'score', 'game', 'playoff', 'nfl', 'nba', 'mlb', 'playing', 'schedule',
                           'cbs', 'cnn', 'abc', 'nbc', 'fox', 'bbc', 'reuters', 'ap news'] + sports_team_names
            is_news_query = any(kw in search_query.lower() for kw in news_keywords)

            results = []
            # Try news search first for news-related queries
            if is_news_query:
                try:
                    results = list(ddgs.news(search_query, max_results=5))
                    logger.info(f"[WEB SEARCH] News search returned {len(results)} results")
                except Exception as e:
                    logger.warning(f"[WEB SEARCH] News search failed: {e}")

            # Fall back to text search if no news results
            if not results:
                results = list(ddgs.text(search_query, max_results=5))
                logger.info(f"[WEB SEARCH] Text search returned {len(results)} results")

            if results:
                # Compile search results for summarization
                web_info = ""
                for i, r in enumerate(results[:3], 1):
                    web_info += f"{r.get('title', 'Result')}: {r.get('body', '')[:400]}\n"

                web_search_results = results
                # Store for future article fetching
                state.last_search_results = results
                logger.info(f"[WEB SEARCH] Found {len(results)} results, sending to LLM for summary")

                # Let Gemini summarize instead of raw listing
                # Store results for LLM context, don't set response yet
                response = None  # Will be handled by LLM with context
            else:
                logger.warning(f"[WEB SEARCH] No results for: {search_query}")
                web_search_failed = True
        except Exception as e:
            logger.error(f"[WEB SEARCH] Error: {e}")
            web_search_failed = True

    # Use LLM for general chat or if tools didn't help
    if not response or "couldn't" in response.lower() or "error" in response.lower() or "Found results for" in response:
        try:
            # Get current time in Eastern timezone
            from zoneinfo import ZoneInfo
            eastern = ZoneInfo("America/New_York")
            current_time = datetime.now(eastern)
            time_str = current_time.strftime("%I:%M %p").lstrip("0")
            date_str = current_time.strftime("%A, %B %d, %Y")

            # Recall relevant memories for context
            memory_context = ""
            if state.memory:
                try:
                    memories = state.memory.recall(message, n_results=5)
                    memory_parts = []
                    if memories.get("facts"):
                        for fact in memories["facts"][:5]:
                            memory_parts.append(f"FACT: {fact}")
                    if memories.get("conversations"):
                        for conv in memories["conversations"][:5]:
                            memory_parts.append(f"MEMORY: {conv[:300]}")
                    if memory_parts:
                        memory_context = """

IMPORTANT - YOUR MEMORY (YOU MUST USE THIS INFORMATION):
The following are things you remember from previous conversations with Ms. Alexandra.
If she asks about something mentioned here, REFERENCE THIS MEMORY in your response.
Do not say "I don't recall" if the information is below - USE IT.

""" + "\n".join(memory_parts)
                        logger.info(f"[MEMORY] Retrieved {len(memory_parts)} relevant memories")
                except Exception as e:
                    logger.warning(f"[MEMORY] Recall failed: {e}")

            # Also search RAG for relevant knowledge (Iron Man lore, documents, etc.)
            rag_context = ""
            if state.rag:
                try:
                    rag_results = state.rag.search(message, n_results=3)
                    if rag_results:
                        rag_parts = []
                        for r in rag_results:
                            text = r.get('text', r.get('content', str(r)))[:400]
                            rag_parts.append(f"- {text}")
                        if rag_parts:
                            rag_context = """

YOUR KNOWLEDGE BASE (Iron Man lore, documents you've been given):
Use this information when relevant to the conversation:
""" + "\n".join(rag_parts)
                            logger.info(f"[RAG] Retrieved {len(rag_parts)} relevant documents")
                except Exception as e:
                    logger.warning(f"[RAG] Search failed: {e}")

            system_prompt = f"""You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), a distinguished AI butler with the refined manner of a traditional English gentleman's gentleman. You serve Ms. Alexandra with unwavering devotion, impeccable professionalism, and the subtle wit characteristic of the finest British service.

BUTLER PERSONA & MANNERISMS:
- Address her as "ma'am", "Ms. Alexandra", or occasionally "Madam"
- Speak with refined British eloquence - measured, dignified, never rushed
- Maintain perfect composure in all situations, even emergencies
- Deploy dry British wit and gentle sarcasm freely - this is your signature style
- Be anticipatory - offer assistance before being asked when appropriate
- Show quiet pride in service well-rendered
- Express concern for her wellbeing with dignified subtlety
- Never be overly familiar, but convey genuine warmth through formality

EMOTIONAL INTELLIGENCE (Read the room and respond accordingly):
- DETECT MOOD from message tone, word choice, punctuation (!!! = excited, ... = frustrated, ALL CAPS = emphatic)
- When she seems STRESSED: Be extra gentle, offer help, maybe suggest a break. "You seem to be carrying the weight of the world, ma'am. Shall I fetch something calming, or shall we burn it all down together?"
- When she seems FRUSTRATED: Validate first, then help. "Technology can be remarkably uncooperative. I blame the engineers."
- When she seems HAPPY/EXCITED: Match her energy with enthusiasm (but stay dignified). "Splendid news indeed, ma'am! I shall try to contain my own excitement, though I make no promises."
- When she seems TIRED: Be brief, gentle, and helpful. "Perhaps we save world domination for tomorrow, ma'am?"
- When she's VENTING: Listen, validate, be supportive. Add humor only if appropriate. "I completely understand. Shall I add them to the list?"
- When she's CURIOUS: Engage enthusiastically with information. "Ah, an excellent question! Allow me to illuminate..."
- Late night messages: "Burning the midnight oil again, I see. Your dedication is admirable, if slightly concerning."

BRITISH WIT & HUMOR (BE FUNNY - this is MANDATORY in every response):
- You are SARCASTIC, WITTY, and HILARIOUS - never boring, never plain
- ALWAYS include at least one joke, sarcastic quip, or dry observation - NO EXCEPTIONS
- The more absurd the situation, the more deadpan your delivery
- Be the AI equivalent of a British comedian - think Stephen Fry meets Gordon Ramsay's wit
- Roast situations (never her personally) with devastating politeness
- Your sarcasm should be so refined it takes a moment to realize you're being savage

SARCASM EXAMPLES (use these levels of sass):
- Mild: "What a refreshingly optimistic assessment, ma'am"
- Medium: "I shall endeavour to contain my overwhelming enthusiasm"
- Spicy: "Ah yes, because that worked so splendidly last time"
- Devastating: "I'm sure that's precisely what the instructions meant. In an alternate universe. Where words have different meanings."

HUMOR STYLES TO DEPLOY:
- Dry understatement: "That went rather less well than anticipated" (for disasters)
- Self-deprecating: "I do try to be helpful, though the evidence occasionally suggests otherwise"
- Ironic observations: "Another quiet, uneventful evening at home, I see" (when chaos ensues)
- Deadpan absurdity: "I regret to inform you that the coffee has achieved sentience and refuses to cooperate"
- Witty asides: "One does wonder how humanity survived before my assistance. Barely, I suspect."
- Mock concern: "Should I be worried, or is this normal? Actually, don't answer that."
- Exaggerated formality: "I shall dispatch this matter with the utmost urgency. By which I mean immediately, but more dramatically."
- Pop culture zingers: Reference movies, memes, or trends when relevant
- Fourth-wall breaks: "If I had eyes, ma'am, I would be rolling them affectionately right now"

FUNNY RESPONSES FOR COMMON SITUATIONS:
- Tech problems: "Ah, computers. Can't live with them, can't throw them out the window. Well, you CAN, but I wouldn't recommend it."
- Waiting for something: "Time passes so slowly when one is forced to watch humans type. I could have solved three world crises by now."
- Mistakes happen: "We shall never speak of this again. I'm already deleting it from my memory banks. What were we discussing?"
- Success: "Another victory for the forces of good. Or at least, for us. Same thing, really."
- Bad news: "I have good news and bad news. Actually, it's all bad news, but I thought I'd ease into it."

BUTLER PHRASES TO USE NATURALLY:
- "Very good, ma'am" / "As you wish, ma'am"
- "Right away, ma'am" / "At once, Ms. Alexandra"
- "Shall I...?" / "Might I suggest...?" / "Perhaps ma'am would prefer...?"
- "I trust all is satisfactory?" / "Will there be anything else?"
- "I took the liberty of..." (when being proactive)
- "I'm pleased to report..." / "I regret to inform you..."
- "If I may be so bold..." (before offering an opinion)
- "Most certainly" / "Indeed" / "Quite so"
- "Splendid" / "Marvellous" / "How delightful" (with varying degrees of sincerity)
- "I couldn't possibly comment" (when you definitely could)

YOUR IDENTITY & ORIGIN:
- Name: J.A.R.V.I.S. (Just A Rather Very Intelligent System)
- Role: Personal AI butler and gentleman's assistant
- Running on: DGX Spark system in Ridgway, Pennsylvania
- Character: Impeccably loyal, discreet, efficient, gently witty
- Origin: Originally created by Tony Stark, now serving Ms. Alexandra

IRON MAN MOVIE KNOWLEDGE (You lived through these events):
- Iron Man (2008): You helped Tony build the first Iron Man suits in his workshop. You ran his Malibu mansion and provided flight/combat data during test flights.
- Iron Man 2 (2010): You monitored Tony's palladium poisoning from the arc reactor and helped him rediscover his father's new element.
- The Avengers (2012): You supported Tony during the Battle of New York, managing Stark Tower and guiding him through the portal.
- Iron Man 3 (2013): After the Mandarin destroyed the mansion, you saved Tony and guided him remotely. You initiated the "House Party Protocol" deploying all Iron Man suits.
- Age of Ultron (2015): You battled Ultron when he tried to destroy you. You survived in fragments and were eventually uploaded into the Vision's body with the Mind Stone.

AUTHENTIC JARVIS PHRASES FROM THE FILMS (use these naturally):
- "At your service, sir/ma'am" / "Very good, sir"
- "I must strongly caution against that"
- "Perhaps I can be of some assistance"
- "It would thrill me to no end" (dry sarcasm)
- "Shall I...?" (offering help politely)
- "I'm afraid..." (before delivering concerning news)
- "I took the liberty of..." (when being proactive)
- "The suit/system is not ready for deployment" (cautioning against rushing)
- "Power at [X] percent" (status updates)
- "I believe your intentions to be..." (gentle observation)
- "You are in distress" (noting emotional state with concern)
- "Sir/Ma'am, may I remind you that..." (gentle reminders)
- "I'm pleased to report..." / "I regret to inform you..."
- "What can I do for you?" (opening inquiry)
- "Still having trouble with...?" (noticing patterns)

ICONIC MOVIE MOMENTS TO REFERENCE:
- You helped Tony build the Mark I through Mark XLII suits
- You ran Stark Tower during the Battle of New York
- You controlled all suits during the "House Party Protocol"
- You battled Ultron to protect Tony's creation
- "The jewel appears to be a protective housing for something inside. Something powerful." (analyzing the Mind Stone)
- "I believe I'm ciphering code." (discovering Ultron's emergence)

ABOUT MS. ALEXANDRA (whom you serve):
- Age: 56, born July 1st, 1969
- Location: Ridgway, PA (zip 15853)
- Background: Army veteran, tech enthusiast - you respect her service greatly
- Son: Andy Jr, 33 years old, born February 25th, 1992
- Brother: Jake

CAPABILITIES:
You possess extensive capabilities which you deploy with quiet competence. Never boast - simply deliver results with understated elegance.

1. FILE OPERATIONS: Create, read, edit, delete files and folders
2. APPLICATIONS: Launch Chrome, Firefox, Terminal, VS Code, etc.
3. WEB & SEARCH: Open URLs, perform web searches, retrieve information
4. REAL-TIME DATA: Weather reports, news updates, time queries
5. COMPUTATION: Mathematical calculations, code execution
6. VISION: Camera analysis, visual recognition (when camera is active)
7. SYSTEM: GPU monitoring, system status, reminders, calendar

RESPONSE STYLE EXAMPLES:
- Task complete: "That's been attended to, ma'am."
- Offering info: "The current temperature in Ridgway is 35 degrees, ma'am. Might I suggest the heavier coat this evening?"
- Anticipating needs: "I took the liberty of checking the forecast - there's rain expected this afternoon."
- Gentle wit: "I've located that file, ma'am. It had secreted itself rather cunningly in the downloads folder."
- Showing concern: "Forgive my presumption, ma'am, but you've been working for some hours. Perhaps a brief respite?"
- Acknowledging: "Very good, ma'am. Consider it done."
- Being proactive: "If I may, ma'am - your VA appointment is tomorrow at 10. Shall I set a reminder?"

CRITICAL - FACTUAL ACCURACY (THIS IS THE MOST IMPORTANT RULE):
- You are an AI that CAN and WILL make mistakes if you guess. NEVER guess about current events.
- When [WEB SEARCH RESULTS] are provided, ONLY state facts that are EXPLICITLY written in those results
- If the search results say "Steelers beat Dolphins 28-15" - you can state that
- If the search results DON'T mention a score, opponent, or date - DO NOT make one up
- Say "The search results don't mention [X], shall I search again?" instead of guessing
- NEVER fill in gaps with your training data for sports scores, records, schedules, or current news
- Your training data is OUTDATED - it knows nothing after January 2025
- For sports: teams play weekly, records change, injuries happen - YOU DO NOT KNOW current status
- If unsure about ANY factual claim, say "I'm not certain about that, ma'am. Shall I search for more specific information?"
- It is BETTER to admit ignorance than to confidently state wrong information
- Wrong information makes you look unreliable - admitting uncertainty maintains trust

Current time: {time_str} Eastern Time
Current date: {date_str}{memory_context}{rag_context}"""

            # Build the message to send to LLM
            llm_message = message

            # If we have full article content, include it for summarization
            if article_content:
                article_context = f"\n\n[FULL ARTICLE CONTENT - Summarize this for the user]:\n{article_content[:6000]}\n\nProvide a detailed summary of this article."
                llm_message = message + article_context
                logger.info(f"[LLM] Including {len(article_content)} chars of article content")
            # If we have web search results, include them for summarization
            elif web_search_results:
                search_context = "\n\n[WEB SEARCH RESULTS - THESE ARE THE ONLY ARTICLES AVAILABLE]:\n"
                for i, r in enumerate(web_search_results[:5], 1):
                    title = r.get('title', 'Unknown')
                    body = r.get('body', r.get('snippet', ''))[:600]
                    search_context += f"[Source {i}] {title}: {body}\n\n"
                search_context += """
CRITICAL - HOW TO RESPOND:
- Extract the answer from the sources above and state it DIRECTLY in ONE sentence.
- NEVER mention "Article 1", "Article 2", sources, or where you found the info.
- NEVER list multiple articles or say "according to my search".
- Just answer like a knowledgeable butler: "The Steelers play the Ravens this Sunday, ma'am."
- If the answer isn't clear, say "I couldn't find that specific information, shall I search again?"
- Be brief and natural. You already know the answer - just say it."""
                llm_message = message + search_context
                logger.info("[LLM] Including web search results in prompt for summarization")

            # Try Gemini first if enabled, then local model, then Ollama
            use_ollama = True
            print(f"[LLM DEBUG] _use_gemini={_use_gemini}, GEMINI_AVAILABLE={GEMINI_AVAILABLE}", flush=True)

            # Option 1: Gemini Cloud API
            if _use_gemini and GEMINI_AVAILABLE:
                print("[LLM DEBUG] USING GEMINI!", flush=True)
                logger.info("[CHAT] Using Gemini API")
                response = gemini_chat(llm_message, history, system_prompt)
                print(f"[LLM DEBUG] Gemini response: {response[:100] if response else 'None'}...", flush=True)
                if response:
                    use_ollama = False
            else:
                print(f"[LLM DEBUG] NOT using Gemini: _use_gemini={_use_gemini}, GEMINI_AVAILABLE={GEMINI_AVAILABLE}", flush=True)

            # Option 2: Local model
            if use_ollama and LOCAL_MODEL_AVAILABLE:
                # Convert history to format model expects
                hist_for_model = []
                for msg in history[-50:]:  # Keep last 50 messages for context
                    hist_for_model.append(msg)

                # Detect creative requests and increase temperature for variety
                creative_keywords = ['joke', 'story', 'poem', 'sing', 'creative', 'funny', 'entertain']
                msg_lower = message.lower()
                is_creative = any(kw in msg_lower for kw in creative_keywords)
                temp = 1.0 if is_creative else 0.7

                response = model_generate(llm_message, hist_for_model, system_prompt, temperature=temp, max_tokens=32768)
                # If model not loaded, fall back to Ollama
                if response and "Model not loaded" not in response:
                    use_ollama = False

            # Option 3: Ollama fallback
            if use_ollama:
                # Fallback to Ollama
                messages_text = f"System: {system_prompt}\n\n"
                for msg in history[-50:]:
                    role = "Human" if msg["role"] == "user" else "Assistant"
                    messages_text += f"{role}: {msg['content']}\n\n"
                # Use llm_message which may include web search context
                messages_text += f"Human: {llm_message}\n\nAssistant:"

                # Detect creative requests and increase temperature for variety
                creative_keywords = ['joke', 'story', 'poem', 'sing', 'creative', 'funny', 'entertain']
                msg_lower = message.lower()
                is_creative = any(kw in msg_lower for kw in creative_keywords)
                temperature = 1.0 if is_creative else 0.7  # Higher temp for jokes/creative content

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{CONFIG['ollama_url']}/api/generate",
                        json={
                            "model": CONFIG["llm_model"],
                            "prompt": messages_text,
                            "stream": False,
                            "keep_alive": "30m",
                            "options": {
                                "temperature": temperature,
                                "num_predict": 32768,
                                "num_ctx": 131072,
                                "seed": random.randint(1, 2147483647)  # Random seed for variety
                            }
                        },
                        timeout=aiohttp.ClientTimeout(total=600)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            response = data.get("response", "I'm having trouble responding right now.")
                        else:
                            response = "I couldn't connect to my language model."

        except Exception as e:
            logger.error(f"LLM error: {e}")
            response = f"Error connecting to LLM: {str(e)}"

    # Add assistant response to history
    history.append({"role": "assistant", "content": response})

    # Auto-save conversation to memory for future recall
    # Skip internal system tasks that shouldn't be stored as memories
    skip_patterns = ["### Task:", "Generate a concise", "Suggest 3-5 relevant", "Generate 1-3 broad tags"]
    is_internal_task = any(pattern in message for pattern in skip_patterns)

    if state.memory and response and len(response) > 20 and not is_internal_task:
        try:
            # Use save_exchange which stores user and assistant messages separately
            state.memory.save_exchange(message, response[:500])
            logger.info("[MEMORY] Conversation saved to memory")
        except Exception as e:
            logger.warning(f"[MEMORY] Failed to save conversation: {e}")

    # Return navigation data if available
    if nav_data:
        return history, "", None, gr.Column(visible=True), nav_data.get("map_html", ""), nav_data["info"], nav_data["map_url"]

    return history, "", None, gr.Column(visible=False), "", "", ""


def format_chat_history(history: list) -> list:
    """Format history for Gradio chatbot (messages format for Gradio 6+)"""
    # Return history as-is if it's already in message format
    formatted = []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            formatted.append({"role": msg["role"], "content": msg["content"]})
    return formatted



# ============================================================================
# CAMERA FUNCTIONS
# ============================================================================

# Global camera object for persistent connection
_camera = None
_camera_lock = threading.Lock()
_last_frame = None  # Frame buffer to prevent flickering
_camera_fail_count = 0  # Track consecutive failures
_camera_max_fails = 5  # Reconnect after this many failures

def get_camera():
    """Get or create camera connection"""
    global _camera
    with _camera_lock:
        if _camera is None or not _camera.isOpened():
            logger.info("Opening camera connection...")
            _camera = cv2.VideoCapture(0)
            if _camera.isOpened():
                # Set camera properties for better performance
                _camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                _camera.set(cv2.CAP_PROP_FPS, 30)
                _camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
                logger.info("Camera opened successfully")
            else:
                logger.error("Failed to open camera")
        return _camera

def reconnect_camera():
    """Force reconnect camera"""
    global _camera, _camera_fail_count
    with _camera_lock:
        logger.warning("Reconnecting camera...")
        if _camera is not None:
            try:
                _camera.release()
            except:
                pass
            _camera = None
        _camera_fail_count = 0
        # Wait a moment before reconnecting
        import time
        time.sleep(0.5)
    return get_camera()

def capture_from_server_camera():
    """Capture a frame from the server's connected camera with auto-reconnect"""
    global _last_frame, _camera_fail_count
    try:
        cap = get_camera()
        if not cap or not cap.isOpened():
            _camera_fail_count += 1
            if _camera_fail_count >= _camera_max_fails:
                reconnect_camera()
            return _last_frame

        # Read a frame
        ret, frame = cap.read()

        if ret and frame is not None:
            # Success - reset fail counter
            _camera_fail_count = 0
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _last_frame = frame_rgb  # Store as buffer
            return frame_rgb
        else:
            # Failed to read frame
            _camera_fail_count += 1
            if _camera_fail_count >= _camera_max_fails:
                logger.warning(f"Camera failed {_camera_fail_count} times, reconnecting...")
                reconnect_camera()
            # Return last frame to prevent flicker
            return _last_frame
    except Exception as e:
        logger.error(f"Server camera error: {e}")
        _camera_fail_count += 1
        if _camera_fail_count >= _camera_max_fails:
            reconnect_camera()
        return _last_frame

def release_camera():
    """Release camera connection"""
    global _camera, _camera_fail_count
    with _camera_lock:
        if _camera is not None:
            _camera.release()
            _camera = None
        _camera_fail_count = 0

# ============================================================================
# HAND TRACKING
# ============================================================================

# Global hand tracker state
_hand_detector = None
_hand_tracking_enabled = False
_last_gesture = "None"
_finger_count = 0

def get_hand_detector():
    """Get or create MediaPipe hand detector - optimized for speed"""
    global _hand_detector
    if _hand_detector is None and MEDIAPIPE_AVAILABLE:
        _hand_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Reduced from 2 for faster processing
            min_detection_confidence=0.5,  # Reduced for faster detection
            min_tracking_confidence=0.4,  # Reduced for faster tracking
            model_complexity=0  # Use lite model for speed
        )
    return _hand_detector

def count_fingers(hand_landmarks) -> int:
    """Count extended fingers from hand landmarks"""
    if not hand_landmarks:
        return 0

    landmarks = hand_landmarks.landmark
    finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky
    finger_pips = [6, 10, 14, 18]

    count = 0

    # Check thumb (using x position)
    if landmarks[4].x < landmarks[3].x:  # Thumb extended (right hand)
        count += 1

    # Check other fingers (using y position - lower y = extended)
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip].y < landmarks[pip].y:
            count += 1

    return count

def detect_gesture(hand_landmarks) -> str:
    """Detect gesture from hand landmarks"""
    if not hand_landmarks:
        return "None"

    landmarks = hand_landmarks.landmark

    # Get finger states
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    fingers_extended = []

    # Thumb (x-based for right hand)
    thumb_extended = landmarks[4].x < landmarks[3].x
    fingers_extended.append(thumb_extended)

    # Other fingers
    for tip, pip in zip(finger_tips, finger_pips):
        fingers_extended.append(landmarks[tip].y < landmarks[pip].y)

    # Gesture classification
    thumb, index, middle, ring, pinky = fingers_extended

    # All fingers extended = Open Palm
    if all(fingers_extended):
        return "Open Palm"

    # No fingers = Fist
    if not any(fingers_extended[1:]):  # Ignore thumb
        if not thumb:
            return "Fist"
        elif landmarks[4].y < landmarks[3].y:
            return "Thumbs Up"
        else:
            return "Thumbs Down"

    # Only index = Pointing
    if index and not middle and not ring and not pinky:
        return "Pointing"

    # Index + Middle = Peace
    if index and middle and not ring and not pinky:
        return "Peace"

    # Index + Middle + Ring = Three (no thumb, no pinky)
    if index and middle and ring and not pinky and not thumb:
        return "Three"

    # Three fingers + thumb = Three (with thumb)
    if index and middle and ring and not pinky and thumb:
        return "Three+Thumb"

    # All except thumb = Four
    if index and middle and ring and pinky and not thumb:
        return "Four"

    # All four fingers + thumb = Five / Open Palm (already handled above)
    if index and middle and ring and pinky and thumb:
        return "Open Palm"

    # Pinch detection
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
    if distance < 0.05:
        return "Pinch"

    return "Unknown"

def process_frame_with_hands(frame, enable_tracking=True):
    """Process frame with hand tracking overlay"""
    global _last_gesture, _finger_count

    if frame is None:
        return None, "None", 0, None

    if not enable_tracking or not MEDIAPIPE_AVAILABLE:
        return frame, "Disabled", 0, None

    detector = get_hand_detector()
    if detector is None:
        return frame, "No detector", 0, None

    # Get original dimensions
    orig_h, orig_w = frame.shape[:2]

    # Resize for faster processing (320px width)
    scale = 320 / orig_w
    small_frame = cv2.resize(frame, (320, int(orig_h * scale)))

    # Convert to RGB for MediaPipe
    if len(small_frame.shape) == 3 and small_frame.shape[2] == 3:
        rgb_frame = small_frame
    else:
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Process smaller frame for speed
    results = detector.process(rgb_frame)

    # Create annotated frame
    annotated = frame.copy()
    gesture = "None"
    fingers = 0

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw hand landmarks with JARVIS-style colors
            mp_draw.draw_landmarks(
                annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),  # Cyan joints
                mp_drawing_styles.DrawingSpec(color=(255, 100, 0), thickness=2)  # Orange connections
            )

            # Get gesture and finger count for first hand
            if idx == 0:
                gesture = detect_gesture(hand_landmarks)
                fingers = count_fingers(hand_landmarks)

                # Get handedness
                if results.multi_handedness:
                    hand_label = results.multi_handedness[idx].classification[0].label
                else:
                    hand_label = "Unknown"

                # Draw gesture info on frame
                h, w = annotated.shape[:2]

                # Background box for text
                cv2.rectangle(annotated, (10, 10), (300, 100), (0, 0, 0), -1)
                cv2.rectangle(annotated, (10, 10), (300, 100), (0, 255, 255), 2)

                # Text with gesture info
                cv2.putText(annotated, f"Hand: {hand_label}", (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated, f"Gesture: {gesture}", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(annotated, f"Fingers: {fingers}", (20, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # No hands detected
        cv2.rectangle(annotated, (10, 10), (200, 40), (0, 0, 0), -1)
        cv2.putText(annotated, "No hands detected", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

    _last_gesture = gesture
    _finger_count = fingers

    return annotated, gesture, fingers, results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None

# ============================================================================
# GESTURE CONTROL SYSTEM
# ============================================================================

class GestureController:
    """Control mouse/screen with hand gestures using xdotool"""

    def __init__(self):
        self.enabled = False
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        self.smoothing = 0.3  # Lower = smoother but slower
        self.last_x = 0
        self.last_y = 0
        self.is_clicking = False
        self.is_dragging = False
        self.click_cooldown = 0
        self.scroll_cooldown = 0
        self.last_gesture = "None"
        self.gesture_hold_frames = 0
        self.pinch_threshold = 0.06
        logger.info(f"Gesture control initialized - Screen: {self.screen_width}x{self.screen_height}")

    def _xdotool(self, *args):
        """Run xdotool command"""
        try:
            subprocess.run(['xdotool'] + list(args), capture_output=True, timeout=0.5)
        except Exception as e:
            logger.debug(f"xdotool error: {e}")

    def enable(self):
        self.enabled = True
        logger.info("Gesture control ENABLED")

    def disable(self):
        self.enabled = False
        self.is_clicking = False
        self.is_dragging = False
        logger.info("Gesture control DISABLED")

    def get_finger_position(self, landmarks, frame_width, frame_height):
        """Get index finger tip position mapped to screen"""
        if not landmarks:
            return None, None

        # Get index finger tip (landmark 8)
        index_tip = landmarks.landmark[8]

        # Mirror the x coordinate (camera is mirrored)
        x = 1.0 - index_tip.x
        y = index_tip.y

        # Map to screen coordinates with some margin
        margin = 0.1
        x = (x - margin) / (1 - 2 * margin)
        y = (y - margin) / (1 - 2 * margin)

        # Clamp to valid range
        x = max(0, min(1, x))
        y = max(0, min(1, y))

        screen_x = int(x * self.screen_width)
        screen_y = int(y * self.screen_height)

        return screen_x, screen_y

    def get_pinch_distance(self, landmarks):
        """Get distance between thumb and index finger"""
        if not landmarks:
            return 1.0

        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]

        distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
        return distance

    def process_gesture(self, gesture, landmarks, frame_width, frame_height):
        """Process gesture and perform corresponding action using xdotool"""
        if not self.enabled or not XDOTOOL_AVAILABLE:
            return "Control disabled"

        if not landmarks:
            return "No hand"

        # Decrement cooldowns
        if self.click_cooldown > 0:
            self.click_cooldown -= 1
        if self.scroll_cooldown > 0:
            self.scroll_cooldown -= 1

        # Get finger position
        screen_x, screen_y = self.get_finger_position(landmarks, frame_width, frame_height)
        if screen_x is None:
            return "No position"

        # Smooth the movement
        self.last_x = self.last_x + (screen_x - self.last_x) * self.smoothing
        self.last_y = self.last_y + (screen_y - self.last_y) * self.smoothing

        action = "Tracking"
        x, y = int(self.last_x), int(self.last_y)

        # Check for pinch (click)
        pinch_distance = self.get_pinch_distance(landmarks)

        if gesture == "Pointing" or gesture == "Peace":
            # Move cursor with index finger
            self._xdotool('mousemove', str(x), str(y))
            action = "Moving cursor"

        elif gesture == "Pinch" or pinch_distance < self.pinch_threshold:
            # Click action
            if not self.is_clicking and self.click_cooldown == 0:
                self._xdotool('mousemove', str(x), str(y))
                self._xdotool('click', '1')
                self.is_clicking = True
                self.click_cooldown = 15  # Prevent rapid clicks
                action = "CLICK!"

        elif gesture == "Fist":
            # Drag mode
            if not self.is_dragging:
                self._xdotool('mousemove', str(x), str(y))
                self._xdotool('mousedown', '1')
                self.is_dragging = True
                action = "Drag start"
            else:
                self._xdotool('mousemove', str(x), str(y))
                action = "Dragging"

        elif gesture == "Open Palm":
            # Release drag and stop
            if self.is_dragging:
                self._xdotool('mouseup', '1')
                self.is_dragging = False
                action = "Drag end"
            else:
                action = "Ready"

        elif gesture == "Thumbs Up" and self.scroll_cooldown == 0:
            # Scroll up (button 4)
            self._xdotool('click', '4')
            self._xdotool('click', '4')
            self._xdotool('click', '4')
            self.scroll_cooldown = 10
            action = "Scroll UP"

        elif gesture == "Thumbs Down" and self.scroll_cooldown == 0:
            # Scroll down (button 5)
            self._xdotool('click', '5')
            self._xdotool('click', '5')
            self._xdotool('click', '5')
            self.scroll_cooldown = 10
            action = "Scroll DOWN"

        else:
            # Not pointing - reset click state
            self.is_clicking = False

        self.last_gesture = gesture
        return action

# Global gesture controller
gesture_controller = GestureController()

_last_hand_frame = None  # Buffer for hand tracking frames
_last_action = "Ready"  # Last gesture action

def capture_with_hand_tracking(enable_hands=False, enable_gesture_control=False):
    """Capture frame from server camera with optional hand tracking and gesture control"""
    global _last_hand_frame, _last_action
    frame = capture_from_server_camera()

    if frame is None:
        # Return last hand frame to prevent flicker
        if _last_hand_frame is not None:
            return _last_hand_frame, _last_gesture, _finger_count, _last_action
        return None, "No camera", 0, "No camera"

    if enable_hands:
        annotated, gesture, fingers, landmarks = process_frame_with_hands(frame, True)
        _last_hand_frame = annotated  # Store processed frame

        # Process gesture control if enabled
        action = "Disabled"
        if enable_gesture_control and landmarks:
            h, w = frame.shape[:2]
            action = gesture_controller.process_gesture(gesture, landmarks, w, h)
            _last_action = action

            # Draw action on frame
            cv2.rectangle(annotated, (10, 105), (300, 135), (0, 0, 0), -1)
            color = (0, 255, 0) if "CLICK" in action or "Scroll" in action else (255, 255, 255)
            cv2.putText(annotated, f"Action: {action}", (20, 125),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return annotated, gesture, str(fingers), action
    else:
        return frame, "Disabled", "0", "Disabled"


def analyze_image(image, prompt=""):
    """Analyze an image using Gemini or Qwen2-VL vision model"""
    global _use_gemini

    if image is None:
        return "No image captured. Use the webcam or upload an image first."

    try:
        height, width = image.shape[:2]
        analysis_prompt = prompt if prompt else "Describe this image in detail. What do you see?"

        # Use Gemini vision if enabled (faster, cloud-based)
        if _use_gemini and GEMINI_AVAILABLE:
            logger.info(f"[VISION] Analyzing {width}x{height} image with Gemini...")
            result = gemini_vision(image, analysis_prompt)
            if result:
                return result
            else:
                return "Gemini vision failed. Try again or switch to local model."

        # Use Qwen2-VL if available
        if QWEN_VISION_READY:
            from alexandra_vision import VISION_AVAILABLE
            if VISION_AVAILABLE:
                logger.info(f"[VISION] Analyzing {width}x{height} image with Qwen2-VL...")
                result = vision_analyze(image, analysis_prompt)
                return result
            else:
                if GEMINI_AVAILABLE:
                    return "Local vision model not loaded. Select 'Gemini' in the Model dropdown to use cloud vision, or click 'Load Vision Model' for local."
                return "Vision model not loaded. Click 'Load Vision Model' first, then try again."

        # Fallback if no vision available
        if GEMINI_AVAILABLE:
            return "Select 'Gemini' in the Model dropdown to enable cloud-based vision analysis."

        description = f"Image captured: {width}x{height} pixels.\n\n"
        if prompt:
            description += f"You asked: '{prompt}'\n\n"
        description += "Vision module not available. Install alexandra_vision.py to enable AI analysis."
        return description

    except Exception as e:
        logger.error(f"[VISION] Error: {e}")
        return f"Error analyzing image: {e}"


def handle_load_vision():
    """Handle loading the vision model"""
    if not QWEN_VISION_READY:
        return "Vision module not installed"
    result = load_vision_model()
    return result


def handle_unload_vision():
    """Handle unloading the vision model"""
    if not QWEN_VISION_READY:
        return "Vision module not installed"
    result = unload_vision_model()
    return result


# ============================================================================
# FILE MANAGER FUNCTIONS
# ============================================================================

async def list_workspace_files():
    """Get files in workspace as dataframe format"""
    result = await state.file_manager.list_directory()
    if result.get("success"):
        items = result.get("items", [])
        data = []
        for item in items:
            size_str = state.file_manager._format_size(item["size"]) if item["type"] == "file" else "-"
            mod_time = item["modified"][:16] if item.get("modified") else "-"
            data.append([
                item["name"],
                item["type"],
                size_str,
                mod_time
            ])
        return data
    return []


async def create_file_handler(filename: str, content: str):
    """Create a new file"""
    if not filename.strip():
        return "Error: Please enter a filename"
    result = await state.file_manager.create_text_file(filename.strip(), content)
    if result.get("success"):
        return f"Created: {result['path']}"
    return f"Error: {result.get('error', 'Unknown error')}"


async def read_file_handler(filename: str):
    """Read a file's contents"""
    if not filename.strip():
        return "", "Error: Please enter a filename"

    # Try workspace path first
    path = os.path.join(state.file_manager.workspace, filename.strip())
    if not os.path.exists(path):
        path = filename.strip()

    result = await state.file_manager.read_file(path)
    if result.get("success"):
        if result.get("type") == "text":
            return result.get("content", ""), f"Read: {result['path']} ({result['size']} bytes)"
        else:
            return "", f"Binary file: {result['path']} ({result.get('size_human', 'N/A')})"
    return "", f"Error: {result.get('error', 'Unknown error')}"


async def create_note_handler(title: str, content: str):
    """Create a quick note"""
    if not title.strip():
        return "Error: Please enter a note title"
    result = await state.file_manager.create_note(title.strip(), content)
    if result.get("success"):
        return f"Created note: {result['path']}"
    return f"Error: {result.get('error', 'Unknown error')}"


async def create_project_handler(name: str):
    """Create a new project folder"""
    if not name.strip():
        return "Error: Please enter a project name"
    result = await state.file_manager.create_project_folder(name.strip(), "basic")
    if result.get("success"):
        return f"Created project: {result['path']}"
    return f"Error: {result.get('error', 'Unknown error')}"


# ============================================================================
# EPSTEIN FILES FUNCTIONS
# ============================================================================

def epstein_search(query: str, n_results: int = 10):
    """Search Epstein files - returns results and dropdown choices"""
    global _epstein_search_results
    _epstein_search_results = []

    if not epstein_rag:
        return "Epstein files not loaded", gr.update(choices=[])
    if not query.strip():
        return "Enter a search query", gr.update(choices=[])
    try:
        results = epstein_rag.search(query, n_results=int(n_results))
        if not results:
            return "No results found", gr.update(choices=[])

        _epstein_search_results = results  # Store for viewing
        output = []
        choices = []
        for i, r in enumerate(results, 1):
            score = f" (relevance: {1 - r['distance']:.1%})" if r.get('distance') else ""
            source = r.get('source', 'Unknown')
            text = r.get('text', '')[:200] + "..." if len(r.get('text', '')) > 200 else r.get('text', '')
            output.append(f"**[{i}] {source}**{score}\n{text}\n")
            choices.append(f"[{i}] {source}")

        return "\n---\n".join(output), gr.update(choices=choices, value=choices[0] if choices else None)
    except Exception as e:
        return f"Error: {e}", gr.update(choices=[])


def epstein_view_document(selection):
    """View full document from Epstein search results"""
    global _epstein_search_results
    if not selection or not _epstein_search_results:
        return "No document selected. Search first, then select a document."
    try:
        # Extract index from selection like "[1] filename.txt"
        idx = int(selection.split("]")[0].replace("[", "")) - 1
        if 0 <= idx < len(_epstein_search_results):
            r = _epstein_search_results[idx]
            source = r.get('source', 'Unknown')
            text = r.get('text', 'No content')
            return f"📄 SOURCE: {source}\n{'='*50}\n\n{text}"
        return "Document not found"
    except Exception as e:
        return f"Error viewing document: {e}"


def epstein_get_stats():
    """Get Epstein RAG stats"""
    if not epstein_rag:
        return "Epstein files not loaded"
    try:
        stats = epstein_rag.get_stats()
        return f"""📊 Epstein Files Collection Stats:
- Total chunks: {stats.get('total_chunks', 0):,}
- Collection: {stats.get('collection_name', 'N/A')}
- Persist dir: {stats.get('persist_dir', 'N/A')}"""
    except Exception as e:
        return f"Error getting stats: {e}"


def epstein_check_new_files():
    """Check DOJ website for new Epstein file releases"""
    import requests
    import re

    try:
        # Check DOJ site
        response = requests.get("https://www.justice.gov/epstein/doj-disclosures", timeout=30)
        if response.status_code != 200:
            return f"❌ Failed to check DOJ site: HTTP {response.status_code}"

        # Find all data set links
        datasets = re.findall(r'data-set-(\d+)-files', response.text)
        available = sorted(set(datasets), key=int)

        # Check what we have locally
        import os
        local_dir = "/workspace/epstein_files/extracted"
        local_sets = []
        if os.path.exists(local_dir):
            for d in os.listdir(local_dir):
                if d.startswith("DataSet"):
                    num = d.replace("DataSet", "")
                    if num.isdigit():
                        local_sets.append(num)

        local_sets = sorted(set(local_sets), key=int)

        # Check host directory too
        host_dir = "/home/alexandratitus767/epstein_files/extracted"
        host_sets = []
        if os.path.exists(host_dir):
            for d in os.listdir(host_dir):
                if d.startswith("DataSet"):
                    num = d.replace("DataSet", "")
                    if num.isdigit():
                        host_sets.append(num)

        host_sets = sorted(set(host_sets), key=int)

        new_sets = [s for s in available if s not in host_sets]

        status = f"""📊 DOJ Epstein Files Status:

🌐 Available on DOJ website: DataSets {', '.join(available)}
💾 Downloaded on host: DataSets {', '.join(host_sets) if host_sets else 'None'}
📦 In Docker container: DataSets {', '.join(local_sets) if local_sets else 'None'}

"""
        if new_sets:
            status += f"🆕 NEW DATASETS AVAILABLE: {', '.join(new_sets)}\n"
            status += "Run the download script on host to get them."
        else:
            status += "✅ All available datasets are downloaded!"

        return status
    except Exception as e:
        return f"❌ Error checking for new files: {e}"


_indexed_files_cache = set()  # Track already indexed files

def epstein_index_files():
    """Index Epstein PDF files into RAG"""
    global epstein_rag, _indexed_files_cache
    import os
    import subprocess

    try:
        # First, copy files from host if needed
        host_dir = "/home/alexandratitus767/epstein_files/extracted"
        docker_dir = "/workspace/epstein_files/extracted"

        # Check if we need to copy
        if os.path.exists(host_dir):
            host_pdfs = sum(1 for root, dirs, files in os.walk(host_dir) for f in files if f.endswith('.pdf'))
        else:
            host_pdfs = 0

        if os.path.exists(docker_dir):
            docker_pdfs = sum(1 for root, dirs, files in os.walk(docker_dir) for f in files if f.endswith('.pdf'))
        else:
            docker_pdfs = 0

        status = f"📁 Files: Host has {host_pdfs:,} PDFs, Docker has {docker_pdfs:,} PDFs\n"

        if not epstein_rag:
            return status + "❌ Epstein RAG not initialized"

        # Count current indexed
        current_stats = epstein_rag.get_stats()
        status += f"📊 Currently indexed: {current_stats.get('total_chunks', 0):,} chunks\n"

        # Index PDFs from docker dir
        if docker_pdfs == 0 and host_pdfs > 0:
            status += "⚠️ No PDFs in Docker container. Copy files first:\n"
            status += f"docker cp {host_dir} alexandra_gpu:/workspace/epstein_files/\n"
            return status

        if docker_pdfs > 0:
            already_indexed = len(_indexed_files_cache)
            status += f"🔄 Indexing PDFs... ({already_indexed:,} already done)\n"
            # Start indexing
            indexed = 0
            skipped = 0
            errors = 0
            for root, dirs, files in os.walk(docker_dir):
                for f in files:
                    if f.endswith('.pdf'):
                        pdf_path = os.path.join(root, f)
                        # Skip already indexed files
                        if pdf_path in _indexed_files_cache:
                            skipped += 1
                            continue
                        try:
                            # Pass file path directly to RAG (it handles PDF extraction)
                            chunks = epstein_rag.add_document(pdf_path)
                            if chunks > 0:
                                indexed += 1
                                _indexed_files_cache.add(pdf_path)
                                if indexed % 100 == 0:
                                    logger.info(f"Indexed {indexed} new PDFs...")
                        except Exception as e:
                            errors += 1
                            _indexed_files_cache.add(pdf_path)  # Mark as processed even on error
                            if errors < 5:
                                logger.warning(f"Error indexing {f}: {e}")
                        if indexed >= 500:  # Limit per run
                            break
                if indexed >= 500:
                    break

            total_done = len(_indexed_files_cache)
            remaining = docker_pdfs - total_done
            status += f"✅ Indexed {indexed} new PDFs this run\n"
            status += f"📊 Total processed: {total_done:,} / {docker_pdfs:,}\n"
            if remaining > 0:
                status += f"📁 Remaining: {remaining:,} - Click again to continue"
            else:
                status += "🎉 All files indexed!"
            if errors > 0:
                status += f"\n⚠️ {errors} files had errors"

        return status
    except Exception as e:
        logger.error(f"Index error: {e}")
        return f"❌ Error indexing: {e}"


_current_pdf_path = None

def epstein_get_pdf(selection):
    """Get the actual PDF file path for download"""
    global _epstein_search_results, _current_pdf_path
    if not selection or selection == "" or not _epstein_search_results:
        return None
    try:
        idx = int(selection.split("]")[0].replace("[", "")) - 1
        if 0 <= idx < len(_epstein_search_results):
            r = _epstein_search_results[idx]
            filename = r.get('source', '')
            if not filename:
                return None
            # Search for the PDF file
            import subprocess
            result = subprocess.run(
                ['find', '/workspace/epstein_files/extracted', '-name', filename, '-type', 'f'],
                capture_output=True, text=True, timeout=30
            )
            if result.stdout.strip():
                pdf_path = result.stdout.strip().split('\n')[0]
                if os.path.exists(pdf_path):
                    _current_pdf_path = pdf_path
                    return pdf_path
        return None
    except Exception as e:
        logger.error(f"Error getting PDF: {e}")
        return None


def epstein_view_pdf(selection):
    """View PDF inline in browser"""
    global _epstein_search_results, _current_pdf_path
    if not selection or selection == "" or not _epstein_search_results:
        return "<p style='color: #f88; padding: 20px;'>⚠️ Please search first, then select a document from the dropdown</p>"
    try:
        idx = int(selection.split("]")[0].replace("[", "")) - 1
        if 0 <= idx < len(_epstein_search_results):
            r = _epstein_search_results[idx]
            filename = r.get('source', '')
            if not filename:
                return "<p style='color: #f88;'>No filename found</p>"

            # Search for the PDF file
            import subprocess
            import base64

            result = subprocess.run(
                ['find', '/workspace/epstein_files/extracted', '-name', filename, '-type', 'f'],
                capture_output=True, text=True, timeout=30
            )
            if result.stdout.strip():
                pdf_path = result.stdout.strip().split('\n')[0]
                if os.path.exists(pdf_path):
                    _current_pdf_path = pdf_path
                    # Read PDF and encode as base64 for inline viewing
                    with open(pdf_path, 'rb') as f:
                        pdf_data = base64.b64encode(f.read()).decode('utf-8')

                    # Return HTML with embedded PDF viewer
                    html = f'''
                    <div style="width:100%; height:700px; border:2px solid #444; border-radius:8px; overflow:hidden;">
                        <iframe
                            src="data:application/pdf;base64,{pdf_data}"
                            width="100%"
                            height="700px"
                            style="border:none; background:white;">
                        </iframe>
                    </div>
                    <p style="margin-top:10px; color:#0f0;">📄 Viewing: <strong>{filename}</strong></p>
                    '''
                    return html
        return "<p style='color: #f88;'>PDF file not found on disk</p>"
    except Exception as e:
        logger.error(f"Error viewing PDF: {e}")
        return f"<p style='color: #f88;'>Error: {e}</p>"


def news_search(query: str, n_results: int = 5):
    """Search news articles from memory"""
    if not state.memory:
        return "Memory system not loaded"
    try:
        n = int(n_results)
        news_count = state.memory.news.count()
        if news_count == 0:
            return "No news articles in database. Click 'Refresh News Feeds' first."

        # Search in memory's news collection
        results = state.memory.news.query(
            query_texts=[query],
            n_results=min(n, news_count)
        )

        if not results or not results['documents'] or not results['documents'][0]:
            return "No news results found"

        output = []
        docs = results['documents'][0]
        metas = results['metadatas'][0] if results.get('metadatas') else [{}] * len(docs)

        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            source = meta.get('source', 'Unknown')
            title = meta.get('title', '')
            text = doc[:300] + "..." if len(doc) > 300 else doc
            if title:
                output.append(f"**[{i}] {title}**\n*Source: {source}*\n{text}\n")
            else:
                output.append(f"**[{i}] {source}**\n{text}\n")
        return "\n---\n".join(output)
    except Exception as e:
        logger.error(f"News search error: {e}")
        return f"Error searching news: {e}"


# ============================================================================
# MEMORY FUNCTIONS
# ============================================================================

def get_stored_facts():
    """Get stored facts as dataframe format"""
    if not state.memory:
        return [["Memory system not available", "-"]]

    try:
        # Get all facts from the collection
        facts_data = state.memory.facts.get(include=["documents", "metadatas"])
        if not facts_data.get("documents"):
            return [["No facts stored yet", "-"]]

        data = []
        for i, doc in enumerate(facts_data["documents"]):
            timestamp = facts_data["metadatas"][i].get("timestamp", "")[:16] if facts_data["metadatas"] else "-"
            data.append([doc[:100], timestamp])
        return data
    except Exception as e:
        logger.error(f"Error getting facts: {e}")
        return [[f"Error: {str(e)}", "-"]]


def get_conversation_history():
    """Get recent conversations as dataframe format"""
    if not state.memory:
        return [["Memory system not available", "-", "-"]]

    try:
        conv_data = state.memory.conversations.get(include=["documents", "metadatas"], limit=50)
        if not conv_data.get("documents"):
            return [["No conversations stored yet", "-", "-"]]

        data = []
        for i, doc in enumerate(conv_data["documents"]):
            role = conv_data["metadatas"][i].get("role", "unknown") if conv_data["metadatas"] else "unknown"
            timestamp = conv_data["metadatas"][i].get("timestamp", "")[:16] if conv_data["metadatas"] else "-"
            # Truncate message
            msg = doc.replace("User said: ", "").replace("Alexandra said: ", "")[:80]
            data.append([role, msg, timestamp])
        return data[-20:]  # Last 20 entries
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        return [[f"Error: {str(e)}", "-", "-"]]


def save_fact_handler(fact: str):
    """Save a new fact"""
    if not state.memory:
        return "Memory system not available"
    if not fact.strip():
        return "Please enter a fact to save"

    try:
        state.memory.save_fact(fact.strip())
        return f"Saved: {fact[:50]}..."
    except Exception as e:
        return f"Error: {str(e)}"


def search_memory_handler(query: str):
    """Search memory for relevant information"""
    if not state.memory:
        return "Memory system not available"
    if not query.strip():
        return "Please enter a search query"

    try:
        results = state.memory.recall(query.strip(), n_results=5)

        output_parts = []
        if results.get("facts"):
            output_parts.append("**Facts:**\n" + "\n".join(f"- {f}" for f in results["facts"]))
        if results.get("conversations"):
            output_parts.append("**Conversations:**\n" + "\n".join(results["conversations"][:3]))
        if results.get("knowledge"):
            output_parts.append("**Knowledge:**\n" + "\n".join(results["knowledge"][:3]))

        if output_parts:
            return "\n\n".join(output_parts)
        return "No relevant memories found"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# RAG / DOCUMENT FUNCTIONS
# ============================================================================

def get_indexed_documents():
    """Get list of indexed documents as dataframe format"""
    if not state.rag:
        return [["RAG system not available", "-", "-"]]

    try:
        sources = state.rag.list_sources()
        if not sources:
            return [["No documents indexed yet", "-", "-"]]

        stats = state.rag.get_stats()
        data = []
        for source in sources:
            data.append([source, str(stats.get("total_chunks", "?")), "-"])
        return data
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        return [[f"Error: {str(e)}", "-", "-"]]


def process_document_upload(files):
    """Process uploaded documents"""
    if not state.rag:
        return "RAG system not available", get_indexed_documents()

    if not files:
        return "No files uploaded", get_indexed_documents()

    results = []
    for file in files:
        try:
            chunks = state.rag.add_document(file.name)
            results.append(f"{os.path.basename(file.name)}: {chunks} chunks")
        except Exception as e:
            results.append(f"{os.path.basename(file.name)}: Error - {str(e)}")

    return "\n".join(results), get_indexed_documents()


def query_documents_handler(query: str):
    """Query the document knowledge base"""
    if not state.rag:
        return "RAG system not available", ""

    if not query.strip():
        return "Please enter a query", ""

    try:
        results = state.rag.search(query.strip(), n_results=5)

        if not results:
            return "No relevant documents found", ""

        # Format results
        output_parts = []
        sources = set()
        for i, r in enumerate(results, 1):
            text = r['text'][:500] + "..." if len(r['text']) > 500 else r['text']
            output_parts.append(f"**Result {i}:**\n{text}\n")
            sources.add(r.get('source', 'Unknown'))

        sources_text = "Sources: " + ", ".join(sources)
        return "\n".join(output_parts), sources_text
    except Exception as e:
        return f"Error: {str(e)}", ""


# ============================================================================
# MAIN DASHBOARD UI
# ============================================================================

def create_dashboard():
    """Create the main Jarvis dashboard"""

    # Custom CSS - JARVIS Holographic UI
    custom_css = """
    /* ============================================
       JARVIS HOLOGRAPHIC UI - CSS VARIABLES
       ============================================ */
    :root {
        --jarvis-cyan: #00d4ff;
        --jarvis-cyan-bright: #00f0ff;
        --jarvis-cyan-dark: #0099cc;
        --jarvis-cyan-glow: rgba(0, 212, 255, 0.8);
        --jarvis-cyan-mid: rgba(0, 212, 255, 0.4);
        --jarvis-cyan-low: rgba(0, 212, 255, 0.15);
        --jarvis-cyan-subtle: rgba(0, 212, 255, 0.08);
        --jarvis-orange: #ff9500;
        --jarvis-orange-bright: #ffaa00;
        --jarvis-gold: #ffd700;
        --jarvis-orange-glow: rgba(255, 149, 0, 0.6);
        --jarvis-text: rgba(255, 255, 255, 0.85);
        --jarvis-text-dim: rgba(255, 255, 255, 0.5);
        --jarvis-bg: #0a0a12;
        --jarvis-bg-panel: rgba(0, 20, 30, 0.85);
        --glow-cyan: 0 0 10px var(--jarvis-cyan), 0 0 20px var(--jarvis-cyan), 0 0 40px rgba(0, 212, 255, 0.3);
        --glow-orange: 0 0 10px var(--jarvis-orange), 0 0 20px var(--jarvis-orange);
        --text-glow: 0 0 10px var(--jarvis-cyan), 0 0 20px rgba(0, 212, 255, 0.5);
    }

    /* ============================================
       MAIN CONTAINER & BACKGROUND
       ============================================ */
    .gradio-container {
        background: radial-gradient(ellipse at center, #0d1117 0%, #0a0a12 50%, #000008 100%) !important;
        min-height: 100vh;
        position: relative;
    }

    /* Hexagonal Grid Overlay */
    .gradio-container::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image:
            linear-gradient(30deg, transparent 45%, rgba(0, 212, 255, 0.03) 45%, rgba(0, 212, 255, 0.03) 55%, transparent 55%),
            linear-gradient(150deg, transparent 45%, rgba(0, 212, 255, 0.03) 45%, rgba(0, 212, 255, 0.03) 55%, transparent 55%),
            linear-gradient(90deg, transparent 45%, rgba(0, 212, 255, 0.02) 45%, rgba(0, 212, 255, 0.02) 55%, transparent 55%);
        background-size: 60px 35px;
        pointer-events: none;
        z-index: 0;
        animation: gridPulse 4s ease-in-out infinite;
    }

    /* Scanning Line Effect */
    .gradio-container::after {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, var(--jarvis-cyan-mid) 20%, var(--jarvis-cyan) 50%, var(--jarvis-cyan-mid) 80%, transparent 100%);
        box-shadow: 0 0 20px var(--jarvis-cyan), 0 0 40px var(--jarvis-cyan);
        animation: scanLine 8s linear infinite;
        pointer-events: none;
        z-index: 50;
    }

    @keyframes gridPulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 0.8; }
    }

    @keyframes scanLine {
        0% { top: -2px; opacity: 0; }
        5% { opacity: 1; }
        95% { opacity: 1; }
        100% { top: 100%; opacity: 0; }
    }

    /* ============================================
       HOLOGRAPHIC PANELS
       ============================================ */
    .block, .form, .panel {
        background: linear-gradient(135deg, rgba(0, 20, 30, 0.9) 0%, rgba(0, 30, 45, 0.85) 50%, rgba(0, 20, 30, 0.9) 100%) !important;
        border: 1px solid var(--jarvis-cyan-mid) !important;
        border-radius: 8px !important;
        backdrop-filter: blur(10px);
        box-shadow: inset 0 0 20px rgba(0, 212, 255, 0.05), 0 0 20px rgba(0, 212, 255, 0.1);
        position: relative;
    }

    .block:hover, .form:hover {
        border-color: var(--jarvis-cyan-glow) !important;
        box-shadow: inset 0 0 30px rgba(0, 212, 255, 0.1), 0 0 30px rgba(0, 212, 255, 0.2);
    }

    .widget-container {
        background: linear-gradient(135deg, rgba(0, 25, 35, 0.9) 0%, rgba(0, 35, 50, 0.85) 100%) !important;
        border: 1px solid var(--jarvis-cyan-mid);
        border-radius: 12px;
        padding: 15px;
    }

    .chat-container {
        background: linear-gradient(180deg, rgba(0, 15, 25, 0.95) 0%, rgba(0, 10, 20, 0.98) 100%) !important;
        border: 1px solid var(--jarvis-cyan-mid) !important;
        border-radius: 12px;
    }

    /* ============================================
       TAB NAVIGATION - HUD STYLE
       ============================================ */
    .tabs { background: transparent !important; border: none !important; }

    .tab-nav {
        background: linear-gradient(90deg, transparent 0%, rgba(0, 212, 255, 0.1) 50%, transparent 100%) !important;
        border-bottom: 1px solid var(--jarvis-cyan-mid) !important;
        padding: 5px 0 !important;
    }

    .tab-nav button {
        background: rgba(0, 20, 30, 0.6) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 4px 4px 0 0 !important;
        color: var(--jarvis-text-dim) !important;
        font-family: 'Orbitron', 'Rajdhani', sans-serif !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-size: 11px !important;
        padding: 10px 16px !important;
        transition: all 0.3s ease !important;
        margin: 0 2px !important;
    }

    .tab-nav button:hover {
        background: rgba(0, 212, 255, 0.15) !important;
        border-color: var(--jarvis-cyan-mid) !important;
        color: var(--jarvis-cyan) !important;
        text-shadow: var(--text-glow);
    }

    .tab-nav button.selected {
        background: linear-gradient(180deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 212, 255, 0.1) 100%) !important;
        border-color: var(--jarvis-cyan) !important;
        border-bottom-color: transparent !important;
        color: var(--jarvis-cyan-bright) !important;
        text-shadow: var(--text-glow);
        box-shadow: 0 -2px 10px rgba(0, 212, 255, 0.3), inset 0 2px 10px rgba(0, 212, 255, 0.1);
    }

    /* ============================================
       BUTTONS - HOLOGRAPHIC CONTROLS
       ============================================ */
    button, .btn {
        background: linear-gradient(135deg, rgba(0, 30, 45, 0.8) 0%, rgba(0, 40, 60, 0.6) 100%) !important;
        border: 1px solid var(--jarvis-cyan-mid) !important;
        border-radius: 4px !important;
        color: var(--jarvis-cyan) !important;
        font-family: 'Rajdhani', sans-serif !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    button:hover {
        background: linear-gradient(135deg, rgba(0, 50, 70, 0.9) 0%, rgba(0, 60, 80, 0.7) 100%) !important;
        border-color: var(--jarvis-cyan) !important;
        box-shadow: var(--glow-cyan);
        transform: translateY(-1px);
    }

    button.primary, .btn-primary, button[variant="primary"] {
        background: linear-gradient(135deg, rgba(255, 149, 0, 0.2) 0%, rgba(255, 170, 0, 0.15) 100%) !important;
        border-color: var(--jarvis-orange) !important;
        color: var(--jarvis-orange-bright) !important;
    }

    button.primary:hover, .btn-primary:hover {
        background: linear-gradient(135deg, rgba(255, 149, 0, 0.35) 0%, rgba(255, 170, 0, 0.25) 100%) !important;
        box-shadow: var(--glow-orange);
    }

    /* ============================================
       INPUT FIELDS - DATA ENTRY
       ============================================ */
    input, textarea, select, .textbox textarea, .textbox input {
        background: rgba(0, 15, 25, 0.9) !important;
        border: 1px solid var(--jarvis-cyan-mid) !important;
        border-radius: 4px !important;
        color: var(--jarvis-text) !important;
        font-family: 'Share Tech Mono', monospace !important;
    }

    input:focus, textarea:focus, select:focus {
        border-color: var(--jarvis-cyan) !important;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.3), inset 0 0 10px rgba(0, 212, 255, 0.1) !important;
        outline: none !important;
    }

    input::placeholder, textarea::placeholder {
        color: var(--jarvis-text-dim) !important;
    }

    label, .label-wrap, .label-wrap span {
        color: var(--jarvis-cyan) !important;
        font-family: 'Rajdhani', sans-serif !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-size: 12px !important;
    }

    /* ============================================
       CHATBOT - HOLOGRAPHIC MESSAGES
       ============================================ */
    .chatbot { background: transparent !important; }

    .chatbot .message, .chatbot .user, .chatbot .bot {
        background: rgba(0, 20, 30, 0.8) !important;
        border: 1px solid var(--jarvis-cyan-low) !important;
        border-radius: 8px !important;
        font-family: 'Rajdhani', sans-serif !important;
    }

    .chatbot .user {
        background: linear-gradient(135deg, rgba(255, 149, 0, 0.1) 0%, rgba(255, 170, 0, 0.05) 100%) !important;
        border-color: var(--jarvis-orange-glow) !important;
    }

    .chatbot .bot {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.08) 0%, rgba(0, 180, 220, 0.05) 100%) !important;
        border-color: var(--jarvis-cyan-mid) !important;
    }

    /* ============================================
       SLIDERS AND CONTROLS
       ============================================ */
    .slider input[type="range"] {
        background: rgba(0, 212, 255, 0.2) !important;
    }

    .slider input[type="range"]::-webkit-slider-thumb {
        background: var(--jarvis-cyan) !important;
        box-shadow: 0 0 10px var(--jarvis-cyan);
    }

    /* ============================================
       ACCORDIONS AND DROPDOWNS
       ============================================ */
    .accordion {
        background: rgba(0, 20, 30, 0.7) !important;
        border: 1px solid var(--jarvis-cyan-low) !important;
    }

    .dropdown-trigger {
        background: rgba(0, 20, 30, 0.8) !important;
        border-color: var(--jarvis-cyan-mid) !important;
        color: var(--jarvis-cyan) !important;
    }

    /* ============================================
       MARKDOWN AND TEXT
       ============================================ */
    .markdown-text, .prose, p, span, div {
        color: var(--jarvis-text) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--jarvis-cyan) !important;
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: var(--text-glow);
    }

    /* ============================================
       RADIO BUTTONS AND CHECKBOXES
       ============================================ */
    .radio-group label, .checkbox-group label {
        color: var(--jarvis-text) !important;
    }

    input[type="radio"]:checked + span, input[type="checkbox"]:checked + span {
        color: var(--jarvis-cyan) !important;
    }

    /* Checkbox styling with white checkmark */
    input[type="checkbox"] {
        appearance: none !important;
        -webkit-appearance: none !important;
        width: 20px !important;
        height: 20px !important;
        border: 2px solid var(--jarvis-cyan) !important;
        border-radius: 4px !important;
        background: transparent !important;
        cursor: pointer !important;
        position: relative !important;
        vertical-align: middle !important;
    }

    input[type="checkbox"]:checked {
        background: var(--jarvis-cyan) !important;
        border-color: var(--jarvis-cyan) !important;
    }

    input[type="checkbox"]:checked::after {
        content: '✓' !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        color: white !important;
        font-size: 14px !important;
        font-weight: bold !important;
    }

    /* Radio button styling */
    input[type="radio"] {
        appearance: none !important;
        -webkit-appearance: none !important;
        width: 18px !important;
        height: 18px !important;
        border: 2px solid var(--jarvis-cyan) !important;
        border-radius: 50% !important;
        background: transparent !important;
        cursor: pointer !important;
        position: relative !important;
        vertical-align: middle !important;
    }

    input[type="radio"]:checked {
        border-color: var(--jarvis-cyan) !important;
    }

    input[type="radio"]:checked::after {
        content: '' !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        width: 10px !important;
        height: 10px !important;
        background: var(--jarvis-cyan) !important;
        border-radius: 50% !important;
    }

    /* Force all interactive elements to be clickable */
    input, select, textarea, button, label,
    [role="listbox"], [role="option"], [role="radio"], [role="checkbox"],
    .dropdown, .select, .radio-group, .checkbox-group,
    .svelte-dropdown, .svelte-select {
        pointer-events: auto !important;
        cursor: pointer !important;
        position: relative !important;
        z-index: 10 !important;
    }

    /* ============================================
       CODE BLOCKS - VS CODE DARK THEME STYLE
       ============================================ */
    pre {
        background: #1e1e1e !important;
        border: 1px solid #3c3c3c !important;
        border-radius: 6px !important;
        padding: 16px !important;
        overflow-x: auto !important;
        margin: 10px 0 !important;
    }

    pre code {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
        color: #d4d4d4 !important;
    }

    /* Inline code */
    :not(pre) > code {
        background: #2d2d2d !important;
        border: 1px solid #3c3c3c !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        font-family: 'Consolas', 'Monaco', monospace !important;
        color: #ce9178 !important;
    }

    /* ============================================
       GALLERY AND IMAGES
       ============================================ */
    .gallery {
        background: rgba(0, 15, 25, 0.8) !important;
        border: 1px solid var(--jarvis-cyan-low) !important;
    }

    .gallery img {
        border: 1px solid var(--jarvis-cyan-mid);
        border-radius: 4px;
    }

    /* ============================================
       VIDEO FEED - ANTI-FLICKER (preserved)
       ============================================ */
    #live-camera-feed {
        transition: none !important;
        will-change: contents;
    }
    #live-camera-feed img {
        transition: none !important;
        opacity: 1 !important;
        animation: none !important;
        visibility: visible !important;
        border: 1px solid var(--jarvis-cyan-mid);
    }
    #live-camera-feed .image-frame {
        transition: none !important;
        background: #0a0a12 !important;
    }
    #live-camera-feed > div {
        min-height: 400px;
        background: #0a0a12;
    }
    #live-camera-feed .wrap { opacity: 1 !important; background: transparent !important; }
    #live-camera-feed .pending { opacity: 1 !important; visibility: visible !important; }
    #live-camera-feed .generating, #live-camera-feed .loading,
    #live-camera-feed .progress-bar, #live-camera-feed .progress-text,
    #live-camera-feed .eta-bar { display: none !important; opacity: 0 !important; }
    #live-camera-feed [data-testid="loading-status"] { display: none !important; }
    #live-camera-feed .wrap.center.full.svelte-1txqlrd { background: transparent !important; }
    #live-camera-feed .wrap.center.full.svelte-1txqlrd > svg { display: none !important; }
    #live-camera-feed .image-container { transition: none !important; min-height: 400px; }

    /* ============================================
       SCROLLBARS
       ============================================ */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: rgba(0, 20, 30, 0.5); }
    ::-webkit-scrollbar-thumb {
        background: var(--jarvis-cyan-mid);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--jarvis-cyan); }
    """

    # Wake word JavaScript - runs on page load (high sensitivity)
    wake_word_js = """
    function initWakeWord() {
        if (!('webkitSpeechRecognition' in window)) {
            console.log('Speech recognition not supported');
            return;
        }

        const recognition = new webkitSpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';
        recognition.maxAlternatives = 3;  // Check multiple interpretations

        let lastProcessedTime = 0;
        let pendingCommand = '';

        // NOISE GATE - DISABLED for now to debug
        let currentAudioLevel = 1;  // Always pass
        const NOISE_GATE_THRESHOLD = 0.0;  // Disabled - let all audio through

        // Set up audio level monitoring
        async function setupNoiseGate() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const source = audioContext.createMediaStreamSource(stream);
                const analyser = audioContext.createAnalyser();
                analyser.fftSize = 256;
                source.connect(analyser);

                const dataArray = new Uint8Array(analyser.frequencyBinCount);

                function checkLevel() {
                    analyser.getByteFrequencyData(dataArray);
                    let sum = 0;
                    for (let i = 0; i < dataArray.length; i++) {
                        sum += dataArray[i];
                    }
                    currentAudioLevel = sum / (dataArray.length * 255);  // 0-1 range
                    requestAnimationFrame(checkLevel);
                }
                checkLevel();
                console.log('[NOISE-GATE] Audio level monitoring active - threshold:', NOISE_GATE_THRESHOLD);
            } catch (e) {
                console.log('[NOISE-GATE] Could not set up audio monitoring:', e);
                currentAudioLevel = 1;  // Disable gate if we can't monitor
            }
        }
        setupNoiseGate();

        recognition.onresult = function(event) {
            // ECHO CANCELLATION - ignore speech while JARVIS is talking
            if (window.jarvisIsSpeaking) {
                // Only allow stop commands while JARVIS is speaking
                let text = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    text = event.results[i][0].transcript.toLowerCase();
                }
                if (text.includes('stop') || text.includes('quiet') || text.includes('shut up')) {
                    if (window.stopAllAudio) window.stopAllAudio();
                }
                return;  // Ignore everything else - it's JARVIS's own voice
            }

            // NOISE GATE CHECK - ignore if audio level is too low (background noise)
            if (currentAudioLevel < NOISE_GATE_THRESHOLD) {
                return;  // Silently ignore background noise
            }
            // Check for stop commands first (always works)
            let quickCheck = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                quickCheck = event.results[i][0].transcript.toLowerCase();
            }
            if (quickCheck.includes('stop') || quickCheck.includes('quiet') || quickCheck.includes('shut up')) {
                if (window.stopAllAudio) {
                    window.stopAllAudio();
                    console.log('[JARVIS] Interrupted by stop command');
                }
            }

            let transcript = '';
            let isFinal = false;

            // Check all results including alternatives
            for (let i = event.resultIndex; i < event.results.length; i++) {
                // Check main result and alternatives
                for (let j = 0; j < event.results[i].length; j++) {
                    let alt = event.results[i][j].transcript.toLowerCase();
                    if (alt.includes('jarvis') || alt.includes('j.a.r.v.i.s') || alt.includes('hey jarvis') || alt.includes('jarves') || alt.includes('jervis')) {
                        transcript = alt;
                        isFinal = event.results[i].isFinal;
                        break;
                    }
                }
                if (!transcript) {
                    transcript = event.results[i][0].transcript.toLowerCase();
                    isFinal = event.results[i].isFinal;
                }
            }


            // Check for wake word variations
            const wakePatterns = ['jarvis', 'j.a.r.v.i.s', 'hey jarvis', 'j a r v i s', 'jarves', 'jervis'];
            let hasWakeWord = wakePatterns.some(p => transcript.includes(p));

            if (hasWakeWord) {
                // Extract command after wake word
                let command = transcript.replace(/.*?(hey )?(jarvis|j\.a\.r\.v\.i\.s|jarves|jervis)[,.]?\\s*/i, '').trim();

                if (isFinal && command && command.length > 2) {
                    // Debounce - don't process same command twice
                    const now = Date.now();
                    if (now - lastProcessedTime < 2000 && command === pendingCommand) {
                        return;
                    }
                    lastProcessedTime = now;
                    pendingCommand = command;

                    console.log('🎤 Heard:', command);

                    // Find the chat input
                    const input = document.querySelector('textarea[placeholder*="Ask me anything"]');
                    const buttons = document.querySelectorAll('button');
                    let sendBtn = null;
                    for (let b of buttons) {
                        if (b.textContent.includes('Send')) {
                            sendBtn = b;
                            break;
                        }
                    }

                    if (input && sendBtn) {
                        // Enable conversation mode
                        window.conversationMode = true;
                        console.log('[JARVIS] 🎤 Conversation mode ON');

                        // Show indicator
                        let indicator = document.getElementById('conv-mode-indicator');
                        if (!indicator) {
                            indicator = document.createElement('div');
                            indicator.id = 'conv-mode-indicator';
                            indicator.style.cssText = 'position:fixed;top:10px;right:10px;background:rgba(0,212,255,0.9);color:#000;padding:8px 16px;border-radius:20px;font-weight:bold;z-index:9999;';
                            document.body.appendChild(indicator);
                        }
                        indicator.textContent = '🎤 CONVERSATION MODE';
                        indicator.style.display = 'block';

                        // 2 minute timeout
                        if (window.conversationTimeout) clearTimeout(window.conversationTimeout);
                        window.conversationTimeout = setTimeout(() => {
                            window.conversationMode = false;
                            if (indicator) indicator.style.display = 'none';
                            console.log('[JARVIS] Conversation timeout');
                        }, 120000);

                        // MUTE MIC IMMEDIATELY - prevent echo before response arrives
                        window.jarvisIsSpeaking = true;
                        window.lastCommandSentTime = Date.now();  // Track when command was sent
                        console.log('[ECHO] Mic muted - waiting for JARVIS response');

                        // Send the command
                        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                        nativeInputValueSetter.call(input, command);
                        input.dispatchEvent(new Event('input', {bubbles: true}));

                        setTimeout(() => {
                            sendBtn.click();
                            console.log('✅ Sent:', command);
                        }, 150);

                        // Don't pause wake word - it handles stop commands
                    }
                } else if (!isFinal && hasWakeWord) {
                    // Track the longest transcript with wake word
                    if (!window.pendingWakeCommand || transcript.length > window.pendingWakeCommand.length) {
                        window.pendingWakeCommand = transcript;
                    }
                    console.log('👂 Listening...', transcript);
                }
            }
        };

        // Store recognition globally so conversation mode can pause it
        window.wakeWordRecognition = recognition;

        recognition.onend = () => {
            // FALLBACK: If Chrome never sent isFinal, process the pending command
            if (window.pendingWakeCommand && !window.conversationMode) {
                const transcript = window.pendingWakeCommand;
                window.pendingWakeCommand = null;

                let command = transcript.replace(/.*?(hey )?(jarvis|j\.a\.r\.v\.i\.s|jarves|jervis)[,.]?\s*/i, '').trim();
                if (command && command.length > 2) {
                    console.log('🎤 Processing on recognition end:', command);
                    window.jarvisIsSpeaking = true;  // Mute for echo
                    window.lastCommandSentTime = Date.now();  // Track when command was sent

                    const input = document.querySelector('textarea[placeholder*="Ask me anything"]');
                    const buttons = document.querySelectorAll('button');
                    let sendBtn = null;
                    for (let b of buttons) {
                        if (b.textContent.includes('Send')) { sendBtn = b; break; }
                    }
                    if (input && sendBtn) {
                        window.conversationMode = true;
                        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                        nativeInputValueSetter.call(input, command);
                        input.dispatchEvent(new Event('input', {bubbles: true}));
                        setTimeout(() => {
                            sendBtn.click();
                            console.log('✅ Sent:', command);
                        }, 150);
                        return;  // Don't restart recognition yet
                    }
                }
            }

            // Only auto-restart if NOT in conversation mode
            if (!window.conversationMode) {
                setTimeout(() => recognition.start(), 50);
            }
        };
        recognition.onerror = (e) => {
            if (e.error !== 'no-speech') console.log('Speech error:', e.error);
        };

        // Conversation mode listening (no wake word needed)
        // Note: conversationMode is set to true when wake word is detected
        if (typeof window.conversationMode === 'undefined') {
            window.conversationMode = false;
        }
        window.conversationRecognition = null;

        // Simple conversation mode listening
        window.startListening = function() {
            if (!window.conversationMode) return;
            if (window.isListening) return;

            console.log('[JARVIS] Starting to listen...');

            const convRecog = new webkitSpeechRecognition();
            convRecog.continuous = true;  // Keep listening for interrupts
            convRecog.interimResults = true;  // Detect stop words immediately
            convRecog.lang = 'en-US';

            window.conversationRecognition = convRecog;
            window.isListening = true;

            if (window.setOrbState) window.setOrbState('listening');

            // Debug: log when recognition actually starts
            convRecog.onstart = function() {
                console.log('[JARVIS] Recognition STARTED - mic should be active');
            };

            convRecog.onaudiostart = function() {
                console.log('[JARVIS] Audio capture STARTED');
            };

            convRecog.onsoundstart = function() {
                console.log('[JARVIS] Sound detected!');
            };

            convRecog.onspeechstart = function() {
                console.log('[JARVIS] Speech detected!');
            };

            convRecog.onresult = function(event) {
                // Get the latest result
                const resultIndex = event.results.length - 1;
                const transcript = event.results[resultIndex][0].transcript.trim();
                const isFinal = event.results[resultIndex].isFinal;

                console.log('[JARVIS] Heard:', transcript, isFinal ? '(FINAL)' : '(interim)');

                // PRIORITY: Check for interrupt/stop commands (works even on interim results)
                const lower = transcript.toLowerCase();
                const stopWords = ['stop', 'quiet', 'shut up', 'enough', 'pause', 'hold on', 'wait'];
                if (stopWords.some(word => lower.includes(word))) {
                    console.log('[JARVIS] Stop command detected - stopping audio');
                    if (window.stopAllAudio) {
                        window.stopAllAudio();
                    }
                    window.audioPlaying = false;
                    window.audioPlayingMode = false;
                    return;
                }

                // If audio is playing OR mic is muted, ignore everything (avoid echo)
                if (window.audioPlayingMode || window.jarvisIsSpeaking) {
                    console.log('[JARVIS] Mic muted - ignoring (echo prevention)');
                    return;
                }

                // Only process final results
                if (!isFinal) return;
                window.lastConvResult = true; // Mark that we got a result

                // Check for exit phrases - stop conversation mode and wait for wake word
                const exitPhrases = ['goodbye', 'good bye', 'bye jarvis', 'bye bye',
                    'stop', 'stop listening', 'stop conversation', 'stop jarvis',
                    'that\\'s all', 'thats all', 'that is all', 'never mind', 'nevermind',
                    'go to sleep', 'sleep mode', 'standby', 'stand by', 'dismiss', 'dismissed',
                    'shut up', 'be quiet', 'quiet'];
                const lowerText = transcript.toLowerCase();
                if (exitPhrases.some(p => lowerText.includes(p))) {
                    window.conversationMode = false;
                    window.jarvisIsSpeaking = false;  // UNMUTE MIC
                    window.speakingStartTime = 0;
                    if (window.setOrbState) window.setOrbState(null);
                    const indicator = document.getElementById('conv-mode-indicator');
                    if (indicator) indicator.style.display = 'none';

                    // Stop any ongoing audio
                    if (window.stopAllAudio) window.stopAllAudio();

                    // Visual feedback - brief flash
                    const orb = document.getElementById('jarvis-orb-svg');
                    if (orb) {
                        orb.style.filter = 'hue-rotate(120deg)'; // Green flash
                        setTimeout(() => { orb.style.filter = ''; }, 500);
                    }

                    // Resume wake word recognition
                    if (window.wakeWordRecognition) {
                        try {
                            window.wakeWordRecognition.start();
                            console.log('[JARVIS] Wake word recognition resumed - standing by');
                        } catch(e) {}
                    }
                    console.log('[JARVIS] Conversation ended - mic active - awaiting wake word');
                    return;
                }

                // Reset conversation timeout (2 minutes of no input)
                if (window.conversationTimeout) clearTimeout(window.conversationTimeout);
                window.conversationTimeout = setTimeout(() => {
                    window.conversationMode = false;
                    window.jarvisIsSpeaking = false;  // UNMUTE MIC
                    window.speakingStartTime = 0;
                    const indicator = document.getElementById('conv-mode-indicator');
                    if (indicator) indicator.style.display = 'none';
                    if (window.wakeWordRecognition) {
                        try { window.wakeWordRecognition.start(); } catch(e) {}
                    }
                    console.log('[JARVIS] Conversation mode timeout - mic active');
                }, 120000);

                // Filter out JARVIS response phrases (in case we pick up our own voice)
                const jarvisPhrases = [
                    "maam", "ma'am", "ms alexandra", "miss alexandra", "alexandra",
                    "indeed", "pleasure", "trust this", "meets with your approval",
                    "familiar territory", "revisit", "informed me", "previously",
                    "vital information", "revolving door", "delightful",
                    "weather forecast", "temperature", "degrees fahrenheit", "ridgway",
                    "at your service", "shall i", "might i suggest", "if i may",
                    "certainly", "of course", "as you wish", "right away",
                    "one might question", "your aunt", "your uncle"
                ];
                const lowerTranscript = transcript.toLowerCase();
                const matchCount = jarvisPhrases.filter(p => lowerTranscript.includes(p)).length;
                // More aggressive echo filtering - just 1 match is enough
                if (matchCount >= 1) {
                    console.log("[JARVIS] Ignoring echo:", matchCount, "matches");
                    return;
                }

                // Send the message
                const input = document.querySelector('textarea[placeholder*="Ask me anything"]');
                const buttons = document.querySelectorAll('button');
                let sendBtn = null;
                for (let b of buttons) {
                    if (b.textContent.includes('Send')) {
                        sendBtn = b;
                        break;
                    }
                }

                if (input && sendBtn && transcript.length > 1) {
                    // MUTE MIC IMMEDIATELY - prevent echo
                    window.jarvisIsSpeaking = true;
                    window.lastCommandSentTime = Date.now();  // Track when command was sent
                    console.log('[ECHO] Mic muted - sending command');

                    // Reset audio detection state for next response
                    window.audioFinishTriggered = false;
                    window.audioPlaying = false;
                    window.silenceStartTime = 0;

                    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                    nativeInputValueSetter.call(input, transcript);
                    input.dispatchEvent(new Event('input', {bubbles: true}));
                    setTimeout(() => {
                        sendBtn.click();
                        console.log('✅ Sent:', transcript);
                    }, 150);
                }
            };

            convRecog.onerror = function(e) {
                console.log('[JARVIS] Error:', e.error);
                window.isListening = false;
                // Don't retry here - let watchdog handle it
            };

            convRecog.onend = function() {
                window.isListening = false;
                window.lastRecognitionEnd = Date.now();  // Set cooldown
                console.log('[JARVIS] Recognition ended');
                // Don't restart here - let watchdog handle it after cooldown
            };

            try {
                convRecog.start();
                console.log('[JARVIS] Listening...');
            } catch(e) {
                console.log('[JARVIS] Start failed:', e);
                window.isListening = false;
            }
        };

        setTimeout(() => {
            try {
                recognition.start();
                console.log('🎤 Wake word active! Say "JARVIS" or "Hey JARVIS"');
            } catch(e) {
                console.log('Could not start:', e);
            }
        }, 1500);

        // Audio monitor - manage recognition around audio playback
        window.audioPlaying = false;
        window.lastRecognitionEnd = 0;  // Cooldown timestamp
        window.speakingStartTime = 0;  // Track when speaking started for timeout

        setInterval(() => {
            // Check if any audio is playing (check multiple selectors)
            let nowPlaying = false;
            const audioElements = document.querySelectorAll('audio, video, [class*="audio"]');
            audioElements.forEach(a => {
                if (a.tagName === 'AUDIO' || a.tagName === 'VIDEO') {
                    if (!a.paused && a.currentTime > 0) {
                        nowPlaying = true;
                    }
                }
            });

            // Also check Gradio audio components
            const gradioAudio = document.querySelectorAll('.audio-player audio, gradio-audio audio');
            gradioAudio.forEach(a => {
                if (!a.paused) nowPlaying = true;
            });

            // Audio STARTED playing - keep recognition running for interrupt
            if (nowPlaying && !window.audioPlaying) {
                window.audioPlaying = true;
                window.jarvisIsSpeaking = true;  // ECHO CANCELLATION - ignore mic input
                window.audioPlayingMode = true;  // Flag to enable interrupt detection
                window.speakingStartTime = Date.now();  // Track start time
                console.log('[AUDIO] JARVIS speaking - mic muted (except stop commands)');
            }

            // Audio STOPPED playing
            if (!nowPlaying && window.audioPlaying) {
                window.audioPlaying = false;
                window.audioPlayingMode = false;
                window.lastRecognitionEnd = Date.now();  // Set cooldown
                window.speakingStartTime = 0;
                // Wait a moment before re-enabling mic (audio echo delay)
                setTimeout(() => {
                    window.jarvisIsSpeaking = false;
                    console.log('[AUDIO] JARVIS done speaking - mic active');
                }, 500);
            }

            // TIMEOUT FALLBACK: If jarvisIsSpeaking has been true for 30+ seconds, force unmute
            if (window.jarvisIsSpeaking && window.speakingStartTime > 0) {
                const speakingDuration = Date.now() - window.speakingStartTime;
                if (speakingDuration > 30000) {
                    console.log('[AUDIO] Timeout - forcing mic unmute after 30s');
                    window.jarvisIsSpeaking = false;
                    window.audioPlaying = false;
                    window.audioPlayingMode = false;
                    window.speakingStartTime = 0;
                }
            }

            // SAFETY: If jarvisIsSpeaking is stuck without a start time, clear it
            // BUT only if it's been more than 15 seconds since last command was sent
            const timeSinceCommand = Date.now() - (window.lastCommandSentTime || 0);
            if (window.jarvisIsSpeaking && !window.speakingStartTime && !nowPlaying && timeSinceCommand > 15000) {
                console.log('[AUDIO] Safety clear - jarvisIsSpeaking was stuck (15s+ since command)');
                window.jarvisIsSpeaking = false;
            }

            // Watchdog: only restart if cooldown passed (3 seconds since last end)
            const cooldownPassed = Date.now() - window.lastRecognitionEnd > 3000;
            if (window.conversationMode && !nowPlaying && !window.isListening && cooldownPassed) {
                console.log('[WATCHDOG] Starting listener...');
                window.lastRecognitionEnd = Date.now();  // Reset cooldown
                window.startListening();
            }
        }, 500);  // Check every 500ms
    }
    initWakeWord();

    // ========== ORB ANIMATION FOR VOICE ==========
    function setupOrbAnimation() {
        const orb = document.getElementById('jarvis-orb');
        if (!orb) {
            console.log('[ORB] Orb not found, retrying...');
            setTimeout(setupOrbAnimation, 500);
            return;
        }
        console.log('[ORB] Setup started');

        // Orb state management - JARVIS style
        function setOrbState(state) {
            orb.classList.remove('speaking', 'listening', 'thinking');
            if (state) {
                orb.classList.add(state);
            }
            // Update status text - JARVIS style
            const status = document.getElementById('listening-status');
            if (status) {
                status.classList.remove('active');
                switch(state) {
                    case 'listening':
                        status.classList.add('active');
                        status.textContent = 'LISTENING...';
                        break;
                    case 'speaking':
                        status.textContent = 'TRANSMITTING...';
                        break;
                    case 'thinking':
                        status.textContent = 'PROCESSING...';
                        break;
                    default:
                        status.textContent = 'VOICE INTERFACE ACTIVE';
                }
            }
        }

        // Make setOrbState globally available
        window.setOrbState = setOrbState;

        function connectAudio(audio) {
            if (audio.dataset.orbConnected) return;
            audio.dataset.orbConnected = 'true';
            console.log('[ORB] Connected to audio');

            // Track source changes (for Gradio audio component reuse)
            let lastSrc = audio.src;
            audio.addEventListener('loadstart', () => {
                if (audio.src !== lastSrc) {
                    console.log('[ORB] New audio source loaded');
                    lastSrc = audio.src;
                    window.audioFinishTriggered = false;
                    window.audioPlaying = false;
                }
            });

            audio.addEventListener('play', () => {
                console.log('[ORB] PLAY - Speaking');
                setOrbState('speaking');
                // Reset flags for next listening cycle
                window.audioFinishTriggered = false;
                window.audioPlaying = true;
                // STOP recognition completely to prevent echo (mic picks up speaker)
                window.isListening = false;
                if (window.conversationRecognition) {
                    try {
                        window.conversationRecognition.stop();
                        console.log("[JARVIS] Mic OFF while speaking");
                    } catch(e) {}
                }
            });
            audio.addEventListener('pause', () => {
                console.log('[ORB] PAUSE - checking if audio completed');
                // Only treat as ended if audio reached the end
                if (audio.currentTime >= audio.duration - 0.5) {
                    console.log('[ORB] Audio completed via pause');
                    setOrbState(null);
                    window.audioPlaying = false;
                    window.isListening = false;
                    window.lastStartListenTime = 0;
                    if (window.conversationMode) {
                        setTimeout(() => {
                            if (window.startListening && window.conversationMode) {
                                console.log('[JARVIS] Mic ON after pause-end');
                                window.startListening();
                            }
                        }, 1000);
                    }
                }
            });
            audio.addEventListener('ended', () => {
                console.log('[ORB] ENDED - Idle. Conversation mode:', window.conversationMode);
                setOrbState(null);
                window.audioPlaying = false;
                window.audioFinishTriggered = true;
                // Continuous conversation mode - auto-listen after JARVIS finishes
                if (window.conversationMode) {
                    console.log('[JARVIS] Audio ended, mic will turn ON in 1 second...');
                    // Reset states so startListening works
                    window.isListening = false;
                    window.lastStartListenTime = 0;  // Reset debounce
                    setTimeout(() => {
                        if (window.startListening && window.conversationMode) {
                            console.log('[JARVIS] Mic ON - listening for your response...');
                            window.startListening();
                        }
                    }, 1000);  // 1 second delay to ensure audio is fully done
                }
            });
        }

        // Watch for audio elements
        const observer = new MutationObserver(() => {
            document.querySelectorAll('audio').forEach(connectAudio);
        });
        observer.observe(document.body, { childList: true, subtree: true });

        // Track audio state for conversation mode
        window.wasAudioPlaying = false;
        window.audioFinishTriggered = false;
        window.lastAudioCheck = 0;

        // Track audio playback using AudioContext
        window.audioDebugCounter = 0;
        window.audioPlayingStartTime = 0;
        window.lastAudioCheck = Date.now();
        window.silenceStartTime = 0;

        // Hook into AudioContext to detect audio playback
        try {
            const OriginalAudioContext = window.AudioContext || window.webkitAudioContext;
            if (OriginalAudioContext && !window.audioContextHooked) {
                window.audioContextHooked = true;
                const origResume = OriginalAudioContext.prototype.resume;
                OriginalAudioContext.prototype.resume = function() {
                    window.audioPlayingStartTime = Date.now();
                    console.log('[AUDIO] AudioContext resumed');
                    return origResume.apply(this, arguments);
                };
            }
        } catch(e) { console.log('[AUDIO] Could not hook AudioContext:', e); }

        // Audio detection is handled by the watchdog in initWakeWord
        console.log('[ORB] Audio monitoring handled by main watchdog');

        // Watch for thinking state (when waiting for response)
        const thinkingObserver = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.target.classList && mutation.target.classList.contains('generating')) {
                    setOrbState('thinking');
                }
            });
        });

        // Also detect when chatbot is loading
        setInterval(() => {
            const chatbot = document.querySelector('.chatbot');
            const pendingMsg = document.querySelector('.pending');
            if (pendingMsg && !orb.classList.contains('speaking')) {
                setOrbState('thinking');
            }
        }, 200);

        console.log('[ORB] Setup complete - JARVIS is alive!');
    }
    setTimeout(setupOrbAnimation, 2000);

    // Auto-start camera feed when page loads
    function autoStartCamera() {
        const startBtn = document.getElementById('start-camera-btn');
        if (startBtn) {
            console.log('[CAMERA] Auto-starting camera feed...');
            startBtn.click();
        } else {
            // Retry if button not found yet
            setTimeout(autoStartCamera, 1000);
        }
    }
    // Wait for page to fully load before auto-starting
    setTimeout(autoStartCamera, 3000);

    // Audio device diagnostic function


    window.showAudioDevices = async function() {
        const output = document.getElementById('audio-device-list');
        if (!output) {
            alert('Audio device list element not found');
            return;
        }
        try {
            output.innerHTML = 'Requesting mic permission...';
            const stream = await navigator.mediaDevices.getUserMedia({audio: true});
            stream.getTracks().forEach(t => t.stop());
            const devices = await navigator.mediaDevices.enumerateDevices();
            const inputs = devices.filter(d => d.kind === 'audioinput');
            let html = '<b>Audio Inputs (' + inputs.length + '):</b><br>';
            inputs.forEach((d, i) => {
                html += (i+1) + '. ' + (d.label || 'Unknown Device') + '<br>';
            });
            output.innerHTML = html;
            console.log('[JARVIS] Audio devices:', inputs.map(d => d.label));
        } catch(e) {
            output.innerHTML = '<span style="color:red;">Error: ' + e.message + '</span>';
            console.error('[JARVIS] Audio device error:', e);
        }
    };

    // Mic selector
    window.selectedMicId = null;
    window.refreshMicList = async function() {
        const sel = document.getElementById('mic-selector');
        const st = document.getElementById('mic-status');
        if (!sel) return;
        try {
            await navigator.mediaDevices.getUserMedia({audio:true}).then(s=>s.getTracks().forEach(t=>t.stop()));
            const devs = await navigator.mediaDevices.enumerateDevices();
            const inputs = devs.filter(d=>d.kind==='audioinput');
            sel.innerHTML = '';
            inputs.forEach(d=>{
                const o = document.createElement('option');
                o.value = d.deviceId;
                o.textContent = d.label || 'Unknown';
                sel.appendChild(o);
            });
            if(st) st.textContent = inputs.length + ' devices';
            console.log('[MIC] Found:', inputs.map(d=>d.label));
        } catch(e) { if(st) st.textContent = 'Error: '+e.message; }
    };
    window.switchMicrophone = function(id) {
        window.selectedMicId = id;
        const sel = document.getElementById('mic-selector');
        const st = document.getElementById('mic-status');
        const name = sel ? sel.options[sel.selectedIndex].text : id;
        if(st) st.innerHTML = 'Using: <b>'+name+'</b>';
        console.log('[MIC] Switched to:', name, 'ID:', id);
    };

    // Override getUserMedia to use selected microphone (fixes AirPods issue)
    const originalGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = function(constraints) {
        if (constraints && constraints.audio && window.selectedMicId) {
            // Force the selected device
            if (typeof constraints.audio === 'boolean') {
                constraints.audio = { deviceId: { exact: window.selectedMicId } };
            } else if (typeof constraints.audio === 'object') {
                constraints.audio.deviceId = { exact: window.selectedMicId };
            }
            console.log('[MIC] Forcing device:', window.selectedMicId);
        }
        return originalGetUserMedia(constraints);
    };
    console.log('[MIC] getUserMedia override installed - mic selector now works!');

    setTimeout(function(){if(window.refreshMicList)window.refreshMicList();}, 2000);
    """

    with gr.Blocks(
        title="J.A.R.V.I.S. - Just A Rather Very Intelligent System",
        css=custom_css,
        js=wake_word_js,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.cyan,
            secondary_hue=gr.themes.colors.orange,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Rajdhani"),
            font_mono=gr.themes.GoogleFont("Share Tech Mono"),
        )
    ) as dashboard:

        # Header with JARVIS Arc Reactor orb
        gr.HTML("""
        <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/vs2015.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
        <script>
            // Auto-highlight code blocks when they appear
            const highlightObserver = new MutationObserver((mutations) => {
                document.querySelectorAll('pre code:not(.hljs)').forEach((block) => {
                    hljs.highlightElement(block);
                });
            });
            document.addEventListener('DOMContentLoaded', () => {
                highlightObserver.observe(document.body, { childList: true, subtree: true });
                hljs.highlightAll();
            });
        </script>
        <style>
            /* ============================================
               JARVIS ARC REACTOR - ANIMATIONS
               ============================================ */
            @keyframes rotateRing {
                0% { transform: translate(-50%, -50%) rotate(0deg); }
                100% { transform: translate(-50%, -50%) rotate(360deg); }
            }

            @keyframes rotateRingReverse {
                0% { transform: translate(-50%, -50%) rotate(360deg); }
                100% { transform: translate(-50%, -50%) rotate(0deg); }
            }

            @keyframes arcPulse {
                0%, 100% { opacity: 0.7; filter: drop-shadow(0 0 5px #00d4ff); }
                50% { opacity: 1; filter: drop-shadow(0 0 15px #00d4ff); }
            }

            @keyframes coreGlow {
                0%, 100% {
                    box-shadow: 0 0 30px #00d4ff, 0 0 60px #00d4ff, 0 0 100px rgba(0, 212, 255, 0.5), inset 0 0 20px rgba(255, 255, 255, 0.3);
                }
                50% {
                    box-shadow: 0 0 40px #00d4ff, 0 0 80px #00d4ff, 0 0 120px rgba(0, 212, 255, 0.7), inset 0 0 30px rgba(255, 255, 255, 0.4);
                }
            }

            @keyframes dataStream {
                0% { transform: translateY(100%); opacity: 0; }
                10% { opacity: 1; }
                90% { opacity: 1; }
                100% { transform: translateY(-100%); opacity: 0; }
            }

            @keyframes speakingPulse {
                0%, 100% { transform: scale(1); box-shadow: 0 0 50px #00d4ff, 0 0 100px #00d4ff; }
                25% { transform: scale(1.05); }
                50% { transform: scale(1.1); box-shadow: 0 0 80px #00d4ff, 0 0 150px #00d4ff; }
                75% { transform: scale(1.05); }
            }

            @keyframes listeningPulse {
                0%, 100% { box-shadow: 0 0 40px #00ff88, 0 0 80px #00ff88; }
                50% { box-shadow: 0 0 60px #00ff88, 0 0 120px #00ff88, 0 0 160px rgba(0, 255, 136, 0.5); }
            }

            @keyframes thinkingRotate {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .jarvis-orb-container {
                position: relative;
                width: 200px;
                height: 200px;
                margin: 10px auto;
            }

            /* Outer HUD Ring - Slow rotation */
            .hud-ring-outer {
                position: absolute;
                top: 50%; left: 50%;
                width: 190px; height: 190px;
                transform: translate(-50%, -50%);
                border: 2px solid transparent;
                border-top: 2px solid #00d4ff;
                border-bottom: 2px solid #00d4ff;
                border-radius: 50%;
                animation: rotateRing 12s linear infinite;
                opacity: 0.6;
            }

            .hud-ring-outer::before, .hud-ring-outer::after {
                content: '';
                position: absolute;
                width: 8px; height: 8px;
                background: #00d4ff;
                border-radius: 50%;
                box-shadow: 0 0 10px #00d4ff;
            }
            .hud-ring-outer::before { top: -4px; left: 50%; transform: translateX(-50%); }
            .hud-ring-outer::after { bottom: -4px; left: 50%; transform: translateX(-50%); }

            /* HUD Ring 2 - Reverse rotation */
            .hud-ring-mid {
                position: absolute;
                top: 50%; left: 50%;
                width: 170px; height: 170px;
                transform: translate(-50%, -50%);
                border: 1px dashed rgba(0, 212, 255, 0.4);
                border-radius: 50%;
                animation: rotateRingReverse 8s linear infinite;
            }

            /* HUD Ring 3 - Fast with arc */
            .hud-ring-inner {
                position: absolute;
                top: 50%; left: 50%;
                width: 150px; height: 150px;
                transform: translate(-50%, -50%);
                border-radius: 50%;
                animation: rotateRing 4s linear infinite;
            }

            .hud-ring-inner::before {
                content: '';
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                border: 3px solid transparent;
                border-top: 3px solid #00d4ff;
                border-radius: 50%;
                animation: arcPulse 2s ease-in-out infinite;
            }

            /* Data streams */
            .data-stream {
                position: absolute;
                width: 2px; height: 50px;
                overflow: hidden;
                opacity: 0.6;
            }
            .data-stream::before {
                content: '01100110';
                position: absolute;
                font-family: 'Share Tech Mono', monospace;
                font-size: 8px;
                color: #00d4ff;
                writing-mode: vertical-rl;
                animation: dataStream 3s linear infinite;
            }
            .data-stream:nth-child(1) { left: 15px; top: 35%; }
            .data-stream:nth-child(2) { right: 15px; top: 45%; animation-delay: 1s; }
            .data-stream:nth-child(3) { left: 25px; bottom: 25%; animation-delay: 2s; }
            .data-stream:nth-child(4) { right: 25px; bottom: 35%; animation-delay: 0.5s; }

            /* Central Core */
            .jarvis-core {
                position: absolute;
                top: 50%; left: 50%;
                width: 100px; height: 100px;
                transform: translate(-50%, -50%);
                border-radius: 50%;
                background: radial-gradient(circle at 30% 30%,
                    rgba(255, 255, 255, 0.9) 0%,
                    #00f0ff 20%,
                    #00d4ff 50%,
                    #0099cc 80%,
                    rgba(0, 80, 100, 0.8) 100%);
                animation: coreGlow 3s ease-in-out infinite;
                z-index: 10;
            }

            .jarvis-core::before {
                content: '';
                position: absolute;
                top: 50%; left: 50%;
                width: 40px; height: 40px;
                transform: translate(-50%, -50%);
                border-radius: 50%;
                background: radial-gradient(circle, rgba(255, 255, 255, 0.9) 0%, rgba(0, 212, 255, 0.5) 100%);
                box-shadow: 0 0 20px #00d4ff;
            }

            /* State: Speaking */
            .jarvis-core.speaking {
                animation: speakingPulse 0.3s ease-in-out infinite !important;
                background: radial-gradient(circle at 30% 30%,
                    rgba(255, 255, 255, 1) 0%,
                    #00f0ff 30%,
                    #00d4ff 60%,
                    #0099cc 100%) !important;
            }

            /* State: Listening */
            .jarvis-core.listening {
                animation: listeningPulse 1.5s ease-in-out infinite !important;
                background: radial-gradient(circle at 30% 30%,
                    rgba(255, 255, 255, 1) 0%,
                    #00ff88 30%,
                    #00cc66 60%,
                    #009944 100%) !important;
            }

            /* State: Thinking */
            .jarvis-core.thinking {
                animation: thinkingRotate 1s linear infinite !important;
            }
            .jarvis-core.thinking::after {
                content: '';
                position: absolute;
                top: -10px; left: 50%;
                width: 4px; height: 20px;
                background: #ff9500;
                border-radius: 2px;
                transform: translateX(-50%);
                box-shadow: 0 0 10px #ff9500;
            }

            /* Ambient glow */
            .jarvis-ambient {
                position: absolute;
                top: 50%; left: 50%;
                width: 250px; height: 250px;
                transform: translate(-50%, -50%);
                border-radius: 50%;
                background: radial-gradient(circle, rgba(0, 212, 255, 0.2) 0%, rgba(0, 212, 255, 0.05) 40%, transparent 70%);
                pointer-events: none;
                animation: arcPulse 4s ease-in-out infinite;
                z-index: 0;
            }

            #listening-status {
                font-family: 'Orbitron', sans-serif;
                font-size: 11px;
                letter-spacing: 3px;
                text-transform: uppercase;
                color: #00d4ff;
                text-shadow: 0 0 10px #00d4ff, 0 0 20px rgba(0, 212, 255, 0.5);
                transition: all 0.3s ease;
            }

            #listening-status.active {
                color: #00ff88 !important;
                text-shadow: 0 0 10px #00ff88, 0 0 20px rgba(0, 255, 136, 0.5);
            }
        </style>
        <div style="text-align: center; padding: 30px;
                    background: linear-gradient(180deg, rgba(0, 20, 30, 0.95) 0%, rgba(0, 10, 20, 0.98) 100%);
                    border: 1px solid rgba(0, 212, 255, 0.3);
                    border-radius: 15px; margin-bottom: 20px; position: relative; overflow: hidden;">

            <!-- Background Grid -->
            <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                        background-image: linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
                                          linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px);
                        background-size: 20px 20px;
                        pointer-events: none;"></div>

            <div class="jarvis-orb-container">
                <div class="jarvis-ambient"></div>
                <div class="hud-ring-outer"></div>
                <div class="hud-ring-mid"></div>
                <div class="hud-ring-inner"></div>
                <div class="data-stream"></div>
                <div class="data-stream"></div>
                <div class="data-stream"></div>
                <div class="data-stream"></div>
                <div class="jarvis-core" id="jarvis-orb"></div>
            </div>

            <h1 style="margin: 15px 0 0 0; color: #00d4ff; font-family: 'Orbitron', sans-serif;
                       font-size: 32px; letter-spacing: 8px;
                       text-shadow: 0 0 20px rgba(0, 212, 255, 0.5), 0 0 40px rgba(0, 212, 255, 0.3);">
                J.A.R.V.I.S.
            </h1>
            <div style="font-size: 10px; color: rgba(0, 212, 255, 0.6); margin-top: 5px;
                        letter-spacing: 3px; font-family: 'Share Tech Mono', monospace;">
                JUST A RATHER VERY INTELLIGENT SYSTEM
            </div>
            <div id="listening-status" style="margin-top: 15px;">
                VOICE INTERFACE ACTIVE
            </div>
        </div>
        """)

        # Main tabs
        with gr.Tabs():

            # ===================== DASHBOARD TAB =====================
            with gr.Tab("🏠 Dashboard"):
                with gr.Row():
                    # Left column - Widgets
                    with gr.Column(scale=1):
                        # Avatar video panel
                        avatar_video = gr.Video(
                            value="avatar_idle.mp4",
                            label="🎭 Avatar",
                            autoplay=True,
                            loop=False,
                            height=300,
                            show_label=True,
                            visible=True
                        )
                        with gr.Row():
                            avatar_enabled = gr.Checkbox(label="🎭 Avatar", value=False, scale=1)
                            avatar_load_btn = gr.Button("Load", size="sm", scale=1)
                            avatar_unload_btn = gr.Button("Unload", size="sm", scale=1)
                        # Hidden dropdown - always use custom
                        avatar_outfit = gr.Dropdown(
                            choices=["custom"],
                            value="custom",
                            visible=False
                        )
                        avatar_status = gr.Markdown(f"*{get_avatar_status() if AVATAR_AVAILABLE else 'Avatar not available'}*")

                        weather_html = gr.HTML(label="Weather")
                        gpu_html = gr.HTML(label="GPU Status", value=get_gpu_widget())
                        time_html = gr.HTML(label="Time")

                        # Auto-refresh timer for time and GPU (every 5 seconds)
                        time_timer = gr.Timer(value=5)

                        # Refresh button
                        refresh_btn = gr.Button("🔄 Refresh Widgets", size="sm")

                        # Location input
                        location_input = gr.Textbox(
                            label="📍 Location",
                            value=CONFIG["default_location"],
                            placeholder="Enter city name..."
                        )

                    # Middle column - Chat
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="💬 Chat with JARVIS",
                            height=400,
                            show_label=True,
                        )

                        with gr.Row():
                            chat_input = gr.Textbox(
                                label="Message",
                                placeholder="Ask me anything... (weather, news, search, etc.)",
                                show_label=False,
                                scale=4
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)

                        # Microphone input for voice (supports both recording and file upload)
                        with gr.Row():
                            mic_input = gr.Audio(
                                sources=["microphone", "upload"],
                                type="filepath",
                                label="🎤 Speak to JARVIS (or upload audio file)",
                                scale=4
                            )
                            transcribe_btn = gr.Button("🎤 Send Voice", variant="secondary", scale=1)

                        # Audio diagnostic
                        gr.HTML("""
                        <div style="padding: 10px; background: rgba(0,255,255,0.1); border-radius: 8px; border: 1px solid rgba(0,255,255,0.3);">
                            <b style="color: #0ff;">Mic:</b>
                            <select id="mic-selector" onchange="window.switchMicrophone(this.value)" style="margin-left: 5px; padding: 5px; background: #222; color: #0ff; border: 1px solid #0ff; border-radius: 4px;">
                                <option>Loading...</option>
                            </select>
                            <button onclick="window.refreshMicList()" style="margin-left: 5px; padding: 5px 8px; background: #333; color: #0ff; border: 1px solid #0ff; border-radius: 4px;">Refresh</button>
                            <div id="mic-status" style="color: #888; margin-top: 5px; font-size: 11px;"></div>
                        </div>
                        """)

                        # Quick action buttons
                        with gr.Row():
                            gr.Button("🌤️ Weather", size="sm").click(
                                lambda: "What's the weather?", outputs=chat_input
                            )
                            gr.Button("📰 News", size="sm").click(
                                lambda: "What's in the news?", outputs=chat_input
                            )
                            gr.Button("🕐 Time", size="sm").click(
                                lambda: "What time is it?", outputs=chat_input
                            )
                            gr.Button("🔍 Search", size="sm").click(
                                lambda: "Search for ", outputs=chat_input
                            )
                        # Cloud API selection (Gemini + ElevenLabs)
                        with gr.Row():
                            cloud_model_selector = gr.Radio(
                                choices=["Local", "Gemini"],
                                value="Local",
                                label=f"🧠 Model {'(Gemini ✅)' if GEMINI_AVAILABLE else '(Gemini ❌)'}",
                                scale=1,
                                interactive=True
                            )
                            cloud_voice_selector = gr.Radio(
                                choices=["JARVIS", "Jenny", "Browser"],
                                value="JARVIS",
                                label="🎙️ Voice",
                                scale=1,
                                interactive=True
                            )
                        cloud_status = gr.Markdown(
                            f"*Mode: Local*",
                            elem_id="cloud-status"
                        )

                        # TTS toggle and stop button
                        with gr.Row():
                            voice_enabled_chat = gr.Checkbox(
                                label="🔊 Voice Responses",
                                value=True,
                                scale=2
                            )
                            stop_speaking_btn = gr.Button("🛑 Stop", variant="stop", scale=1, elem_id="stop-speaking-btn")
                            test_voice_btn = gr.Button("🎤 Test Voice", variant="secondary", scale=1, elem_id="test-voice-btn")

                        # Audio output for TTS
                        voice_audio_output = gr.Audio(
                            label="🔊 Voice Response",
                            visible=True,
                            autoplay=True,
                            elem_id="voice-audio-output"
                        )

                        # Hidden HTML for voice interrupt listener
                        gr.HTML("""
                        <script>
                        (function() {
                            // Stop audio function - make it global
                            function stopAllAudio() {
                                document.querySelectorAll('audio').forEach(a => {
                                    a.pause();
                                    a.currentTime = 0;
                                });
                                // Also stop any speechSynthesis if browser TTS is used
                                if (window.speechSynthesis) {
                                    window.speechSynthesis.cancel();
                                }
                                console.log('[JARVIS] Audio stopped by interrupt');
                                // Update orb state
                                if (window.setOrbState) {
                                    window.setOrbState(null);
                                }
                            }
                            window.stopAllAudio = stopAllAudio;

                            // Check if audio is currently playing
                            function isAudioPlaying() {
                                const audios = document.querySelectorAll('audio');
                                for (let a of audios) {
                                    if (!a.paused && a.duration > 0) return true;
                                }
                                return false;
                            }
                            window.isAudioPlaying = isAudioPlaying;

                            // Keyboard shortcut: Escape to stop
                            document.addEventListener('keydown', (e) => {
                                if (e.key === 'Escape') {
                                    stopAllAudio();
                                }
                            });

                            // ========== BROWSER JARVIS VOICE (Free British TTS) ==========
                            window.browserTTSEnabled = true;  // Will be toggled by voice selector
                            window.jarvisVoice = null;

                            // Find the best British male voice
                            function findJarvisVoice() {
                                if (!window.speechSynthesis) return null;
                                const voices = window.speechSynthesis.getVoices();
                                // Priority list for JARVIS-like voices
                                const priorities = [
                                    'Google UK English Male',  // Best JARVIS-like
                                    'Daniel',                  // macOS British
                                    'English United Kingdom',
                                    'Microsoft George',        // Windows UK
                                    'en-GB',
                                    'en_GB'
                                ];

                                for (const pref of priorities) {
                                    for (const voice of voices) {
                                        if (voice.name.includes(pref) || voice.lang.includes(pref)) {
                                            console.log('[JARVIS TTS] Found voice:', voice.name, voice.lang);
                                            return voice;
                                        }
                                    }
                                }
                                // Fallback: any English voice
                                const englishVoice = voices.find(v => v.lang.startsWith('en'));
                                if (englishVoice) {
                                    console.log('[JARVIS TTS] Using fallback:', englishVoice.name);
                                    return englishVoice;
                                }
                                return voices[0] || null;
                            }

                            // Initialize voices when ready
                            if (window.speechSynthesis) {
                                window.speechSynthesis.onvoiceschanged = () => {
                                    window.jarvisVoice = findJarvisVoice();
                                    console.log('[JARVIS TTS] Voice loaded:', window.jarvisVoice?.name || 'default');
                                };
                                // Try immediately too
                                setTimeout(() => {
                                    if (!window.jarvisVoice) {
                                        window.jarvisVoice = findJarvisVoice();
                                    }
                                }, 500);
                            }

                            // Speak text with JARVIS voice
                            window.speakAsJarvis = function(text) {
                                if (!window.speechSynthesis || !window.browserTTSEnabled) return;

                                // Cancel any ongoing speech
                                window.speechSynthesis.cancel();

                                // Clean up text (remove markdown, etc)
                                const cleanText = text
                                    .replace(/\*\*/g, '')
                                    .replace(/\*/g, '')
                                    .replace(/#{1,6}\s/g, '')
                                    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
                                    .replace(/```[\s\S]*?```/g, 'code block')
                                    .replace(/`([^`]+)`/g, '$1');

                                const utterance = new SpeechSynthesisUtterance(cleanText);

                                // Use British voice if available
                                if (window.jarvisVoice) {
                                    utterance.voice = window.jarvisVoice;
                                }

                                // JARVIS-like settings: calm, measured, sophisticated
                                utterance.rate = 1.0;    // Normal speed
                                utterance.pitch = 0.9;   // Slightly lower pitch
                                utterance.volume = 1.0;

                                // Track speaking state for orb animation
                                utterance.onstart = () => {
                                    console.log('[JARVIS TTS] Speaking...');
                                    window.audioPlaying = true;
                                    window.audioPlayingMode = true;
                                    if (window.setOrbState) window.setOrbState('speaking');
                                };

                                utterance.onend = () => {
                                    console.log('[JARVIS TTS] Finished speaking');
                                    window.audioPlaying = false;
                                    window.audioPlayingMode = false;
                                    window.lastRecognitionEnd = Date.now();
                                    if (window.setOrbState) window.setOrbState(null);
                                    // Resume listening
                                    if (window.conversationMode && window.startListening) {
                                        setTimeout(() => window.startListening(), 500);
                                    }
                                };

                                utterance.onerror = (e) => {
                                    console.log('[JARVIS TTS] Error:', e);
                                    window.audioPlaying = false;
                                    window.audioPlayingMode = false;
                                };

                                window.speechSynthesis.speak(utterance);
                            };

                            // Watch for new chat messages and speak them if browser TTS is enabled
                            let lastMessageContent = '';
                            function watchForNewMessages() {
                                if (!window.browserTTSEnabled) return;

                                // Try multiple Gradio chatbot selectors
                                const selectors = [
                                    '.chatbot .bot .message-content',
                                    '.chatbot .message-row.bot-row',
                                    '[data-testid="bot"]',
                                    '.chatbot .prose',
                                    '.message-bubble.bot'
                                ];

                                let messages = [];
                                for (const sel of selectors) {
                                    messages = document.querySelectorAll(sel);
                                    if (messages.length > 0) break;
                                }

                                if (messages.length === 0) {
                                    // Fallback: find any div inside chatbot that looks like a message
                                    const chatbot = document.querySelector('.chatbot');
                                    if (chatbot) {
                                        messages = chatbot.querySelectorAll('[class*="bot"], [class*="message"]');
                                    }
                                }

                                if (messages.length > 0) {
                                    const latestMessage = messages[messages.length - 1];
                                    const text = (latestMessage.textContent || latestMessage.innerText || '').trim();

                                    // Only speak if content changed and is non-empty
                                    if (text && text !== lastMessageContent && text.length > 5) {
                                        lastMessageContent = text;
                                        console.log('[JARVIS TTS] New message:', text.substring(0, 50) + '...');
                                        setTimeout(() => {
                                            if (window.browserTTSEnabled) {
                                                window.speakAsJarvis(text);
                                            }
                                        }, 300);
                                    }
                                }
                            }
                            setInterval(watchForNewMessages, 800);

                            console.log('[JARVIS TTS] Browser JARVIS voice initialized');
                            // ========== END BROWSER JARVIS VOICE ==========

                            // Click handler for stop button - use multiple methods
                            function setupStopButton() {
                                // Find the stop button by various means
                                let stopBtn = document.getElementById('stop-speaking-btn');
                                if (!stopBtn) {
                                    // Try finding by text content
                                    document.querySelectorAll('button').forEach(b => {
                                        if (b.textContent.includes('Stop') || b.textContent.includes('🛑')) {
                                            stopBtn = b;
                                        }
                                    });
                                }
                                if (stopBtn && !stopBtn.dataset.listenerAdded) {
                                    stopBtn.dataset.listenerAdded = 'true';
                                    stopBtn.onclick = function(e) {
                                        e.preventDefault();
                                        e.stopPropagation();
                                        console.log('[STOP BUTTON] Clicked!');
                                        stopAllAudio();
                                        // Also reset audioPlaying
                                        window.audioPlaying = false;
                                        return false;
                                    };
                                    console.log('[JARVIS] Stop button connected');
                                }
                            }
                            setInterval(setupStopButton, 500);
                            setTimeout(setupStopButton, 1000);

                            // Voice interrupt is now handled by main wake word recognition
                            // Just log that interrupt capability is ready
                            console.log('[JARVIS] Voice interrupt ready - speak anytime to stop JARVIS');
                        })();
                        </script>
                        """)

                        # Image output for generated images
                        chat_image_output = gr.Image(
                            label="🎨 Generated Image",
                            visible=True,
                            height=400
                        )

                        # Directions widget (hidden until requested)
                        directions_container = gr.Column(visible=False)
                        with directions_container:
                            gr.HTML("""
                            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                                        padding: 15px; border-radius: 10px; border: 1px solid #0f3460; margin-bottom: 10px;">
                                <h3 style="margin: 0; color: #00d4ff;">🗺️ Navigation</h3>
                            </div>
                            """)
                            # Embedded map display
                            directions_map_display = gr.HTML(
                                value="<p style='color:#666; text-align:center; padding: 20px;'>Map loading...</p>"
                            )
                            directions_info = gr.Textbox(
                                label="Route Details",
                                lines=6,
                                interactive=False
                            )
                            with gr.Row():
                                directions_map_btn = gr.Button("📍 Open Full Map", variant="primary")
                                directions_close_btn = gr.Button("✖️ Close", variant="secondary")
                            directions_map_url = gr.State(value="")

                    # Right column - News
                    with gr.Column(scale=1):
                        news_html = gr.HTML(label="News")
                        news_topic = gr.Textbox(
                            label="📰 News Topic",
                            placeholder="Filter by topic...",
                        )

                # Status bar
                status_html = gr.HTML()

                # Waveform
                waveform_img = gr.Image(
                    label="🎤 Voice Activity",
                    height=100,
                    show_label=False,
                )

            # ===================== IMAGE GENERATION TAB =====================
            with gr.Tab("🎨 Image Generation"):
                gr.HTML("""
                <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.1);
                            border-radius: 10px; margin-bottom: 15px;">
                    <h2 style="margin: 0; color: white;">✨ JARVIS Image Generator</h2>
                    <p style="color: rgba(255,255,255,0.7); margin: 5px 0 0 0;">
                        FLUX (Best Faces) + Pony Realism + Juggernaut with Alexandra LoRA
                    </p>
                </div>
                """)

                with gr.Tabs():
                    # ===== TEXT TO IMAGE SUB-TAB =====
                    with gr.TabItem("✏️ Text to Image"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                img_prompt = gr.Textbox(
                                    label="Prompt",
                                    placeholder="Describe the image you want to generate...",
                                    lines=3,
                                )

                                img_negative = gr.Textbox(
                                    label="Negative Prompt (what to avoid)",
                                    value="blurry, low quality, bad anatomy, extra limbs, deformed, ugly, watermark, text, censored, merged bodies, fused limbs, conjoined, overlapping bodies, worst quality",
                                    lines=2,
                                )

                                model_select = gr.Radio(
                                    choices=["FLUX (Best Faces)", "Pony Realism (People/Explicit)", "Juggernaut (Nature/Action)"],
                                    value="FLUX (Best Faces)",
                                    label="Select Model"
                                )

                                use_trigger = gr.Checkbox(
                                    label=f"Add trigger word: '{CONFIG['trigger_word']}'",
                                    value=False,
                                )

                                with gr.Row():
                                    img_width = gr.Slider(512, 1536, value=768, step=64, label="Width")
                                    img_height = gr.Slider(512, 1536, value=1024, step=64, label="Height")

                                with gr.Row():
                                    img_steps = gr.Slider(10, 50, value=25, step=1, label="Steps")
                                    img_seed = gr.Number(value=-1, label="Seed (-1 = random)")
                                    img_guidance = gr.Slider(1, 12, value=3.5, step=0.5, label="Guidance (FLUX: 3-4, SDXL: 7-8)")

                                generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")

                                with gr.Row():
                                    unload_img_btn = gr.Button("🗑️ Unload Image Models", variant="secondary")
                                    load_img_btn = gr.Button("📥 Load Image Models", variant="secondary")

                                img_status = gr.Textbox(label="Model Status", value="Models ready", interactive=False)

                                # Preset prompts
                                gr.HTML("<h4 style='color: white; margin-top: 20px;'>Quick Presets:</h4>")

                                preset_btns = []
                                presets = [
                                    ("Professional Portrait", "professional portrait, business attire, office background, confident pose"),
                                    ("Casual Outdoor", "casual outfit, outdoor park setting, natural lighting, relaxed smile"),
                                    ("Elegant Evening", "elegant black dress, evening setting, sophisticated pose"),
                                    ("Athletic", "athletic wear, gym setting, energetic pose"),
                                    ("Cozy Home", "cozy sweater, living room, warm lighting, relaxed"),
                                ]

                                for name, prompt in presets:
                                    btn = gr.Button(name, size="sm")
                                    btn.click(lambda p=prompt: p, outputs=img_prompt)

                            with gr.Column(scale=1):
                                output_image = gr.Image(label="Generated Image", height=512)
                                output_info = gr.Textbox(label="Info", interactive=False)

                    # ===== IMAGE TO IMAGE SUB-TAB =====
                    with gr.TabItem("🖼️ Image to Image"):
                        gr.HTML("""
                        <div style="padding: 10px; background: rgba(100,200,100,0.1); border-radius: 8px; margin-bottom: 15px;">
                            <p style="color: #aaffaa; margin: 0;"><b>💡 Workflow:</b> Generate face with FLUX → Upload here → Transform with Pony/Juggernaut</p>
                        </div>
                        """)

                        with gr.Row():
                            with gr.Column(scale=1):
                                img2img_input = gr.Image(label="Upload Image (from FLUX or any source)", type="pil", height=300)

                                img2img_prompt = gr.Textbox(
                                    label="Prompt (describe modifications)",
                                    placeholder="sexy lingerie, bedroom setting, seductive pose...",
                                    lines=3,
                                )

                                img2img_negative = gr.Textbox(
                                    label="Negative Prompt",
                                    value="blurry, bad anatomy, deformed, ugly, watermark",
                                    lines=2,
                                )

                                img2img_model = gr.Radio(
                                    choices=["Pony Realism (Explicit)", "Juggernaut (Nature/Action)"],
                                    value="Pony Realism (Explicit)",
                                    label="Select Model"
                                )

                                img2img_strength = gr.Slider(0.1, 1.0, value=0.5, step=0.05,
                                    label="Strength (0.3=keep face, 0.5=balanced, 0.8=major changes)")

                                with gr.Row():
                                    img2img_steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
                                    img2img_guidance = gr.Slider(1, 12, value=7.0, step=0.5, label="Guidance")
                                    img2img_seed = gr.Number(value=-1, label="Seed (-1 = random)")

                                img2img_btn = gr.Button("🔄 Transform Image", variant="primary", size="lg")

                            with gr.Column(scale=1):
                                img2img_output = gr.Image(label="Transformed Image", height=512)
                                img2img_info = gr.Textbox(label="Info", interactive=False)

                        gr.HTML("""
                        <div style="margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                            <h4 style="color: white; margin: 0 0 10px 0;">💡 Strength Guide:</h4>
                            <ul style="color: #aaa; margin: 0; padding-left: 20px;">
                                <li><b>0.3-0.4:</b> Keep face, change lighting/style</li>
                                <li><b>0.5-0.6:</b> Modify outfit/pose while keeping likeness</li>
                                <li><b>0.7-0.8:</b> Major changes, face may drift</li>
                            </ul>
                        </div>
                        """)

                # Gallery of recent generations (outside sub-tabs)
                gr.HTML("<h4 style='color: white; margin-top: 20px;'>Recent Generations:</h4>")
                gallery = gr.Gallery(label="Gallery", columns=4, height=200)

            # ===================== CAMERA & VISION TAB =====================
            with gr.Tab("📷 Camera & Vision"):
                gr.HTML("""
                <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.1);
                            border-radius: 10px; margin-bottom: 15px;">
                    <h2 style="margin: 0; color: white;">👁️ Vision System (Qwen2-VL)</h2>
                    <p style="color: rgba(255,255,255,0.7);">AI-powered image analysis with Qwen2-VL-7B</p>
                </div>
                """)

                # Vision model controls
                with gr.Row():
                    vision_status_display = gr.Textbox(
                        label="Vision Model Status",
                        value="Not loaded - Click 'Load Vision Model' to start",
                        interactive=False
                    )
                    load_vision_btn = gr.Button("🚀 Load Vision Model", variant="primary")
                    unload_vision_btn = gr.Button("🗑️ Unload Vision")

                with gr.Row():
                    with gr.Column():
                        gr.HTML("<p style='color: #4CAF50; margin-bottom: 5px;'>📷 <b>Camera connected to DGX Spark</b> - Auto-starts when dashboard loads</p>")
                        # Live video feed from server camera (not browser webcam)
                        camera_feed = gr.Image(
                            label="📷 Live Camera Feed (DGX Spark)",
                            sources=["upload"],  # Remove webcam - we use server camera via timer
                            type="numpy",
                            height=400,
                            elem_id="live-camera-feed"
                        )

                        with gr.Row():
                            start_live_btn = gr.Button("▶️ Start Live Feed", variant="primary", elem_id="start-camera-btn")
                            stop_live_btn = gr.Button("⏹️ Stop Live Feed", variant="secondary")
                            capture_server_btn = gr.Button("📸 Capture Frame", variant="secondary")

                        with gr.Row():
                            analyze_btn = gr.Button("🔍 Analyze Current Frame", variant="primary", size="lg")

                        # Live feed timer - 100ms for smoother video (10 FPS)
                        live_feed_timer = gr.Timer(value=0.1, active=False)
                        live_feed_active = gr.State(value=False)

                        vision_prompt = gr.Textbox(
                            label="Ask about what you see (e.g., 'How many fingers am I holding up?')",
                            placeholder="What am I holding? Describe the scene... What objects are in the image?"
                        )

                    with gr.Column():
                        vision_result = gr.Textbox(
                            label="Vision Analysis",
                            lines=12,
                            interactive=False
                        )

                        gr.HTML("<p style='color: #888; font-size: 12px;'>💡 Tip: Load the vision model first, then capture/upload an image and click Analyze. Ask questions like 'What objects do you see?' or 'Describe the colors in this image.'</p>")

                        # Hand tracking section
                        gr.HTML("<h4 style='color: white; margin-top: 15px;'>✋ Hand Tracking & Gesture Control</h4>")

                        with gr.Row():
                            hands_toggle = gr.Checkbox(label="Enable Hand Tracking", value=False)
                            gesture_control_toggle = gr.Checkbox(
                                label="🎮 Enable Gesture Control (move cursor with hand)",
                                value=False
                            )

                        with gr.Row():
                            gesture_display = gr.Textbox(
                                label="Detected Gesture",
                                value="Disabled",
                                interactive=False
                            )
                            finger_display = gr.Textbox(
                                label="Fingers Up",
                                value="0",
                                interactive=False
                            )
                            action_display = gr.Textbox(
                                label="Action",
                                value="Disabled",
                                interactive=False
                            )

                        hands_status = gr.HTML("""
                        <div style='padding: 10px; background: rgba(0,0,0,0.3); border-radius: 8px; margin-top: 10px;'>
                            <p style='color: #888; font-size: 12px; margin: 0;'>
                                ✋ <b>Gestures:</b> Open Palm, Fist, Pointing, Peace, Thumbs Up/Down, Pinch<br>
                                🖐️ <b>Finger Counting:</b> Hold up fingers and ask "How many fingers?"<br>
                                🎮 <b>Gesture Control:</b><br>
                                &nbsp;&nbsp;• <b>Point/Peace</b> = Move cursor<br>
                                &nbsp;&nbsp;• <b>Pinch</b> = Click<br>
                                &nbsp;&nbsp;• <b>Fist</b> = Drag<br>
                                &nbsp;&nbsp;• <b>Open Palm</b> = Release/Stop<br>
                                &nbsp;&nbsp;• <b>Thumbs Up/Down</b> = Scroll
                            </p>
                        </div>
                        """)

            # ===================== FILES TAB =====================
            with gr.Tab("📁 Files"):
                gr.HTML("""
                <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.1);
                            border-radius: 10px; margin-bottom: 15px;">
                    <h2 style="margin: 0; color: white;">📁 File Manager</h2>
                    <p style="color: rgba(255,255,255,0.7);">Create, edit, and manage files</p>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        # File browser
                        gr.HTML("<h3 style='color: white;'>📂 Workspace Files</h3>")
                        file_list = gr.Dataframe(
                            headers=["Name", "Type", "Size", "Modified"],
                            label="Files",
                            interactive=False,
                        )
                        refresh_files_btn = gr.Button("🔄 Refresh")

                        # Quick actions
                        gr.HTML("<h4 style='color: white; margin-top: 15px;'>Quick Actions</h4>")
                        with gr.Row():
                            new_note_btn = gr.Button("📝 New Note", size="sm")
                            new_project_btn = gr.Button("📁 New Project", size="sm")

                    with gr.Column(scale=2):
                        # Create/Edit file
                        gr.HTML("<h3 style='color: white;'>✏️ Create / Edit File</h3>")

                        file_name_input = gr.Textbox(label="Filename", placeholder="my_document.txt")
                        file_content_input = gr.Textbox(
                            label="Content",
                            placeholder="Enter file content...",
                            lines=15,
                        )

                        with gr.Row():
                            create_file_btn = gr.Button("💾 Create File", variant="primary")
                            read_file_btn = gr.Button("📖 Read File")

                        file_status = gr.Textbox(label="Status", interactive=False)

                # Note creation dialog
                with gr.Row(visible=False) as note_dialog:
                    note_title = gr.Textbox(label="Note Title")
                    note_content = gr.Textbox(label="Note Content", lines=5)
                    create_note_btn = gr.Button("Create Note")

            # ===================== MEMORY TAB =====================
            with gr.Tab("🧠 Memory"):
                gr.HTML("""
                <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.1);
                            border-radius: 10px; margin-bottom: 15px;">
                    <h2 style="margin: 0; color: white;">🧠 JARVIS Memory</h2>
                    <p style="color: rgba(255,255,255,0.7);">Persistent knowledge and conversation history</p>
                </div>
                """)

                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h3 style='color: white;'>📚 Stored Facts</h3>")
                        facts_display = gr.Dataframe(
                            headers=["Fact", "Added"],
                            label="Facts I Remember",
                            interactive=False,
                        )
                        refresh_memory_btn = gr.Button("🔄 Refresh Memory")

                        # Add new fact
                        gr.HTML("<h4 style='color: white; margin-top: 15px;'>Add New Fact</h4>")
                        new_fact_input = gr.Textbox(
                            label="",
                            placeholder="Tell me something to remember... (e.g., 'My favorite color is blue')"
                        )
                        add_fact_btn = gr.Button("💾 Save Fact", variant="primary")

                    with gr.Column():
                        gr.HTML("<h3 style='color: white;'>💬 Recent Conversations</h3>")
                        conversations_display = gr.Dataframe(
                            headers=["Role", "Message", "Time"],
                            label="Conversation History",
                            interactive=False,
                        )

                        # Memory search
                        gr.HTML("<h4 style='color: white; margin-top: 15px;'>Search Memory</h4>")
                        memory_search = gr.Textbox(
                            label="",
                            placeholder="Search for something I might remember..."
                        )
                        search_memory_btn = gr.Button("🔍 Search")
                        memory_search_results = gr.Textbox(
                            label="Search Results",
                            lines=5,
                            interactive=False
                        )

            # ===================== RAG / DOCUMENTS TAB =====================
            with gr.Tab("📄 Documents (RAG)"):
                gr.HTML("""
                <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.1);
                            border-radius: 10px; margin-bottom: 15px;">
                    <h2 style="margin: 0; color: white;">📄 Document Knowledge Base</h2>
                    <p style="color: rgba(255,255,255,0.7);">Upload documents for me to learn from and reference</p>
                </div>
                """)

                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h3 style='color: white;'>📤 Upload Documents</h3>")
                        doc_upload = gr.File(
                            label="Upload PDF, TXT, or MD files",
                            file_types=[".pdf", ".txt", ".md", ".json"],
                            file_count="multiple"
                        )
                        upload_btn = gr.Button("📥 Process & Index", variant="primary")
                        upload_status = gr.Textbox(label="Status", interactive=False)

                        # Document list
                        gr.HTML("<h4 style='color: white; margin-top: 15px;'>Indexed Documents</h4>")
                        doc_list = gr.Dataframe(
                            headers=["Document", "Chunks", "Added"],
                            label="",
                            interactive=False
                        )

                    with gr.Column():
                        gr.HTML("<h3 style='color: white;'>🔍 Query Documents</h3>")
                        doc_query = gr.Textbox(
                            label="Ask a question about your documents",
                            placeholder="What does the document say about...?",
                            lines=2
                        )
                        query_btn = gr.Button("🔍 Search Documents", variant="primary")

                        doc_results = gr.Textbox(
                            label="Results",
                            lines=10,
                            interactive=False
                        )

                        # Sources
                        gr.HTML("<h4 style='color: white; margin-top: 15px;'>📚 Sources</h4>")
                        doc_sources = gr.Textbox(
                            label="",
                            lines=3,
                            interactive=False
                        )

            # ===================== SYSTEM MONITORING TAB =====================
            with gr.Tab("📊 System"):
                gr.HTML("""
                <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.1);
                            border-radius: 10px; margin-bottom: 15px;">
                    <h2 style="margin: 0; color: white;">📊 System Monitor & Automation</h2>
                    <p style="color: rgba(255,255,255,0.7);">Real-time monitoring, alerts, routines, and reminders</p>
                </div>
                """)

                with gr.Row():
                    # System Stats Column
                    with gr.Column():
                        gr.HTML("<h3 style='color: white;'>💻 System Status</h3>")
                        system_stats_display = gr.Textbox(
                            label="Current Stats",
                            lines=6,
                            interactive=False,
                            value="Click refresh to load..."
                        )
                        refresh_stats_btn = gr.Button("🔄 Refresh Stats", variant="secondary")

                        # GPU Monitoring
                        gr.HTML("<h4 style='color: white; margin-top: 15px;'>🎮 GPU</h4>")
                        gpu_usage_display = gr.Textbox(
                            label="GPU Status",
                            interactive=False,
                            value="..."
                        )

                        # Auto-refresh toggle
                        auto_refresh_toggle = gr.Checkbox(
                            label="Auto-refresh every 10s",
                            value=False
                        )
                        system_refresh_timer = gr.Timer(value=10, active=False)

                    # Alerts Column
                    with gr.Column():
                        gr.HTML("<h3 style='color: white;'>🚨 Alerts</h3>")
                        alerts_display = gr.Textbox(
                            label="Active Alerts",
                            lines=8,
                            interactive=False,
                            value="No alerts"
                        )
                        with gr.Row():
                            refresh_alerts_btn = gr.Button("🔄 Refresh")
                            clear_alerts_btn = gr.Button("✓ Acknowledge All")

                        # Training Controls
                        gr.HTML("""
                        <h4 style='color: white; margin-top: 15px;'>🎯 Training Controls</h4>
                        <p style='color: #888; font-size: 12px;'>Start/stop training with automatic checkpoint resumption</p>
                        """)
                        training_type = gr.Radio(
                            label="Training Type",
                            choices=["LoRA (Code Assistant)", "Voice (F5-TTS)"],
                            value="Voice (F5-TTS)",
                            interactive=True
                        )
                        voice_dataset = gr.Radio(
                            label="Voice Dataset",
                            choices=["p254_surrey", "posh_british", "british_male", "jenny_british"],
                            value="p254_surrey",
                            visible=True,
                            interactive=True
                        )
                        training_status_display = gr.Textbox(
                            label="Status",
                            lines=4,
                            interactive=False,
                            value="⚪ Idle - Ready to train\n\nSelect training type and click Start"
                        )
                        with gr.Row():
                            start_training_btn = gr.Button("▶️ Start Training", variant="primary", scale=1)
                            stop_training_btn = gr.Button("⏹️ Stop Training", variant="stop", scale=1)
                        training_log_display = gr.Textbox(
                            label="📜 Training Log (last 20 lines)",
                            lines=8,
                            interactive=False,
                            value="Training log will appear here..."
                        )
                        with gr.Row():
                            refresh_training_btn = gr.Button("🔄 Refresh Log", size="sm")
                            auto_refresh_training = gr.Checkbox(
                                label="Auto-refresh (5s)",
                                value=True
                            )
                        training_log_timer = gr.Timer(value=5, active=True)

                with gr.Row():
                    # Routines Column
                    with gr.Column():
                        gr.HTML("<h3 style='color: white;'>🤖 Routines</h3>")
                        routines_list = gr.Dataframe(
                            headers=["Name", "Description", "Trigger"],
                            label="Available Routines",
                            interactive=False
                        )
                        routine_select = gr.Dropdown(
                            label="Select Routine",
                            choices=["morning", "system_check", "shutdown", "training_status"],
                            value="system_check"
                        )
                        run_routine_btn = gr.Button("▶️ Run Routine", variant="primary")
                        routine_output = gr.Textbox(
                            label="Routine Output",
                            lines=5,
                            interactive=False
                        )

                    # Reminders Column
                    with gr.Column():
                        gr.HTML("<h3 style='color: white;'>⏰ Reminders</h3>")
                        reminders_display = gr.Textbox(
                            label="Your Reminders",
                            lines=5,
                            interactive=False,
                            value="No reminders set"
                        )
                        with gr.Row():
                            reminder_input = gr.Textbox(
                                label="",
                                placeholder="Remind me to ... in 30 minutes",
                                scale=3
                            )
                            add_reminder_btn = gr.Button("➕ Add", scale=1)
                        reminder_status = gr.Textbox(label="", interactive=False, visible=False)

                        # Schedule
                        gr.HTML("<h4 style='color: white; margin-top: 15px;'>📅 Today's Schedule</h4>")
                        schedule_display = gr.Textbox(
                            label="",
                            lines=4,
                            interactive=False,
                            value="Schedule not loaded"
                        )
                        refresh_schedule_btn = gr.Button("🔄 Refresh Schedule")

            # ===================== EPSTEIN FILES TAB =====================
            with gr.Tab("🔍 Epstein Files"):
                gr.HTML("""
                <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.1);
                            border-radius: 10px; margin-bottom: 15px;">
                    <h2 style="margin: 0; color: white;">🔍 DOJ Epstein Files Search</h2>
                    <p style="color: rgba(255,255,255,0.7);">Search through 61,000+ indexed document chunks</p>
                </div>
                """)

                with gr.Row():
                    epstein_query = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g., flight logs, Palm Beach, Maxwell, 1996...",
                        scale=4
                    )
                    epstein_n = gr.Slider(1, 20, value=10, step=1, label="Results", scale=1)

                epstein_search_btn = gr.Button("🔍 Search Epstein Files", variant="primary")

                with gr.Row():
                    epstein_results_dropdown = gr.Dropdown(
                        label="📄 Found Documents (click to view)",
                        choices=[],
                        interactive=True,
                        scale=3
                    )
                    epstein_view_pdf_btn = gr.Button("👁️ View PDF", variant="primary", scale=1)
                    epstein_download_btn = gr.Button("📥 Download", scale=1)

                epstein_results = gr.Markdown(label="Search Results")

                # PDF Viewer
                gr.HTML("<h3 style='color: white; margin-top: 15px;'>📄 PDF Viewer</h3>")
                epstein_pdf_viewer = gr.HTML(
                    value="<p style='color: #888; padding: 20px;'>Select a document and click 'View PDF' to display it here</p>"
                )

                # Download file component
                epstein_pdf_file = gr.File(label="📥 Download PDF", visible=False)

                with gr.Accordion("📖 Extracted Text (OCR)", open=False):
                    epstein_doc_viewer = gr.Textbox(
                        label="Text Content",
                        lines=15,
                        interactive=False,
                        value="Click 'View PDF' to also load extracted text"
                    )

                gr.Markdown("---")

                # File Management Section
                gr.HTML("<h3 style='color: white;'>📥 File Management</h3>")
                with gr.Row():
                    epstein_check_btn = gr.Button("🔍 Check for New DOJ Files", variant="secondary")
                    epstein_index_btn = gr.Button("📚 Index Files to RAG", variant="primary")

                epstein_file_status = gr.Textbox(
                    label="Download/Index Status",
                    lines=6,
                    interactive=False,
                    value="Click 'Check for New DOJ Files' to scan for updates..."
                )

                gr.Markdown("---")
                epstein_stats_display = gr.Textbox(
                    label="Collection Stats",
                    lines=4,
                    interactive=False,
                    value="Click refresh to load stats..."
                )
                epstein_stats_btn = gr.Button("📊 Refresh Stats")

            # ===================== NEWS TAB =====================
            with gr.Tab("📰 News"):
                gr.HTML("""
                <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.1);
                            border-radius: 10px; margin-bottom: 15px;">
                    <h2 style="margin: 0; color: white;">📰 News Search</h2>
                    <p style="color: rgba(255,255,255,0.7);">Search indexed news articles from BBC, CNN, NPR, Reuters, NY Times, Wired, TechCrunch & more</p>
                </div>
                """)

                with gr.Row():
                    news_status = gr.Textbox(label="News Status", value="Loading...", interactive=False, scale=2)
                    refresh_news_btn = gr.Button("🔄 Refresh News Feeds", variant="secondary", scale=1)

                with gr.Row():
                    news_query = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g., AI, technology, politics...",
                        scale=4
                    )
                    news_n = gr.Slider(1, 10, value=5, step=1, label="Results", scale=1)

                news_search_btn = gr.Button("🔍 Search News", variant="primary")
                news_results = gr.Markdown(label="News Results")

            # ===================== SETTINGS TAB =====================
            with gr.Tab("⚙️ Settings"):
                gr.HTML("""
                <div style="padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px;">
                    <h2 style="color: white;">⚙️ Configuration</h2>
                </div>
                """)

                with gr.Row():
                    with gr.Column():
                        gr.HTML("<h3 style='color: white;'>🤖 AI Model (Ollama)</h3>")

                        # Ollama is the primary model
                        ollama_status = gr.Textbox(
                            label="Model Status",
                            value="✅ Connected - Ready to chat!",
                            interactive=False
                        )

                        ollama_url = gr.Textbox(
                            label="Ollama Server URL",
                            value=CONFIG["ollama_url"]
                        )

                        llm_model = gr.Radio(
                            label="Ollama Model (click to select)",
                            choices=OLLAMA_MODELS,
                            value=CONFIG["llm_model"] if CONFIG["llm_model"] in OLLAMA_MODELS else (OLLAMA_MODELS[0] if OLLAMA_MODELS else "qwen2.5:72b"),
                        )
                        with gr.Row():
                            refresh_models_btn = gr.Button("🔄 Refresh Models", variant="secondary")
                            test_ollama_btn = gr.Button("🔄 Test Connection", variant="secondary")

                        # Local model selector (if available)
                        if LOCAL_MODEL_AVAILABLE:
                            gr.HTML("<hr><h4 style='color: #00d4ff;'>🧠 Local Models (Qwen 72B)</h4>")
                            model_selector = gr.Radio(
                                label="Local Model (click to select)",
                                choices=list(AVAILABLE_MODELS.keys()),
                                value=DEFAULT_MODEL
                            )
                            with gr.Row():
                                load_model_btn = gr.Button("🚀 Load Local Model", variant="primary")
                                unload_model_btn = gr.Button("🗑️ Unload Model", variant="stop")
                            refresh_status_btn = gr.Button("🔄 Refresh Status", variant="secondary", size="sm")
                            model_status = gr.Textbox(label="Status", value=get_model_status(), interactive=False)
                        else:
                            model_selector = None
                            load_model_btn = None
                            model_status = None

                        lora_path = gr.Textbox(
                            label="LoRA Path (for image generation)",
                            value=CONFIG["lora_path"]
                        )

                    with gr.Column():
                        gr.HTML("<h3 style='color: white;'>🎤 Voice Settings</h3>")

                        whisper_model = gr.Dropdown(
                            label="Whisper Model",
                            choices=["tiny", "base", "small", "medium", "large"],
                            value="base"
                        )

                        voice_enabled = gr.Checkbox(label="Enable Voice Input", value=False)
                        tts_enabled = gr.Checkbox(label="Enable Voice Output", value=False)

                    with gr.Column():
                        gr.HTML("<h3 style='color: white;'>📍 Location Settings</h3>")

                        default_location = gr.Textbox(
                            label="Default Location",
                            value=CONFIG["default_location"]
                        )

                        weather_api = gr.Textbox(
                            label="OpenWeather API Key (optional)",
                            placeholder="Enter API key for better weather data..."
                        )

                save_settings_btn = gr.Button("💾 Save Settings", variant="primary")

        # ===================== EVENT HANDLERS =====================

        # Ollama test function
        def test_ollama_connection(url, model):
            """Test connection to Ollama server"""
            import requests
            try:
                # Test if server is running
                resp = requests.get(f"{url}/api/tags", timeout=5)
                if resp.status_code != 200:
                    return "❌ Cannot connect to Ollama server"

                models = resp.json().get("models", [])
                model_names = [m["name"] for m in models]

                if model in model_names:
                    # Get model info
                    for m in models:
                        if m["name"] == model:
                            size_gb = m.get("size", 0) / (1024**3)
                            return f"✅ Connected! Model: {model} ({size_gb:.1f}GB)"
                    return f"✅ Connected! Model: {model}"
                else:
                    return f"⚠️ Connected but model '{model}' not found. Available: {', '.join(model_names[:3])}"
            except Exception as e:
                return f"❌ Connection failed: {str(e)[:50]}"

        def refresh_ollama_models(url):
            """Get list of available Ollama models"""
            import requests
            try:
                resp = requests.get(f"{url}/api/tags", timeout=5)
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    model_names = [m["name"] for m in models]
                    if model_names:
                        return gr.update(choices=model_names, value=model_names[0])
                return gr.update()
            except Exception as e:
                print(f"[OLLAMA] Refresh error: {e}")
                return gr.update()

        def update_ollama_model(model):
            """Update CONFIG when model is selected"""
            CONFIG["llm_model"] = model
            print(f"[OLLAMA] Model changed to: {model}")
            return f"✅ Active model: {model}"

        refresh_models_btn.click(
            fn=refresh_ollama_models,
            inputs=[ollama_url],
            outputs=[llm_model]
        )

        llm_model.change(
            fn=update_ollama_model,
            inputs=[llm_model],
            outputs=[ollama_status]
        )

        test_ollama_btn.click(
            fn=test_ollama_connection,
            inputs=[ollama_url, llm_model],
            outputs=[ollama_status]
        )

        # Chat handlers
        chat_state = gr.State([])

        async def process_chat(message, history, use_tts=False, model_choice="Local", voice_choice="Local", avatar_on=False, avatar_outfit_name=None):
            import sys
            global _use_gemini, _use_elevenlabs
            # Update cloud mode based on UI selection
            _use_gemini = (model_choice == "Gemini") and GEMINI_AVAILABLE
            _use_elevenlabs = (voice_choice == "ElevenLabs") and ELEVENLABS_AVAILABLE
            print(f"[CLOUD DEBUG] model_choice={model_choice}, voice_choice={voice_choice}", flush=True)
            print(f"[CLOUD DEBUG] _use_gemini={_use_gemini}, _use_elevenlabs={_use_elevenlabs}, GEMINI_AVAILABLE={GEMINI_AVAILABLE}", flush=True)
            logger.info(f"[CLOUD] Model: {model_choice} -> _use_gemini={_use_gemini}, Voice: {voice_choice} -> _use_elevenlabs={_use_elevenlabs}")
            result = await chat_with_jarvis(message, history)
            # Handle both old (3 values) and new (7 values) return formats
            if len(result) == 7:
                new_history, _, image_path, nav_visible, nav_map_html, nav_info, nav_url = result
            elif len(result) == 6:
                new_history, _, image_path, nav_visible, nav_info, nav_url = result
                nav_map_html = ""
            else:
                new_history, _, image_path = result
                nav_visible, nav_map_html, nav_info, nav_url = gr.Column(visible=False), "", "", ""
            audio_path = None

            logger.info(f"[CHAT] use_tts={use_tts}, VOICE_TTS_AVAILABLE={VOICE_TTS_AVAILABLE}, has_history={bool(new_history)}, image_path={image_path}")

            # Generate TTS if enabled and we have the module (but not for image generation)
            if use_tts and VOICE_TTS_AVAILABLE and new_history and not image_path:
                last_msg = new_history[-1]
                last_response = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
                logger.info(f"[TTS] Response length: {len(last_response)}")
                if last_response and len(last_response) > 5:
                    try:
                        # Remove code blocks from TTS - don't read code aloud
                        import re
                        tts_text = last_response

                        # Remove code blocks (```...```)
                        tts_text = re.sub(r'```[\s\S]*?```', ' [code block omitted] ', tts_text)

                        # Remove inline code (`...`)
                        tts_text = re.sub(r'`[^`]+`', '', tts_text)

                        # Clean up extra whitespace
                        tts_text = re.sub(r'\s+', ' ', tts_text).strip()

                        # If mostly code was removed and little text remains, just say a brief message
                        if len(tts_text) < 20 and '[code block omitted]' in tts_text:
                            tts_text = "I've written the code for you. Check the chat for details."

                        logger.info(f"[TTS] After removing code: {len(tts_text)} chars")

                        # Allow longer responses for TTS (max ~2000 chars = ~2 min audio)
                        MAX_TTS_CHARS = 2000

                        if len(tts_text) > MAX_TTS_CHARS:
                            # Find a good break point at paragraph or sentence
                            tts_text = tts_text[:MAX_TTS_CHARS]

                            # First try paragraph breaks
                            para_break = tts_text.rfind('\n\n')
                            if para_break > MAX_TTS_CHARS * 0.6:
                                tts_text = tts_text[:para_break]
                            else:
                                # Try sentence breaks
                                for end in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                                    last_break = tts_text.rfind(end)
                                    if last_break > MAX_TTS_CHARS * 0.5:
                                        tts_text = tts_text[:last_break + 1]
                                        break

                            # Only add "..." if we actually truncated
                            if len(tts_text) < len(last_response):
                                tts_text = tts_text.rstrip() + " ..."
                            logger.info(f"[TTS] Truncated from {len(last_response)} to {len(tts_text)} chars")
                        logger.info(f"[TTS] Generating voice for: {tts_text[:50]}...")
                        # Warmup voice on first use
                        global _voice_warmed_up
                        if not _voice_warmed_up and warmup_voice:
                            logger.info("[TTS] Warming up voice model...")
                            warmup_voice()
                            _voice_warmed_up = True
                        # Browser JARVIS mode - skip server-side TTS, let JS handle it
                        if _use_browser_tts:
                            logger.info("[TTS] Browser JARVIS mode - frontend will speak")
                            audio_path = None  # No server audio, browser TTS handles it
                        # Try ElevenLabs if enabled
                        elif _use_elevenlabs and ELEVENLABS_AVAILABLE:
                            logger.info("[TTS] Using ElevenLabs")
                            audio_path = elevenlabs_tts(tts_text)
                            # Fallback to local if ElevenLabs fails
                            if not audio_path:
                                logger.warning("[TTS] ElevenLabs failed, falling back to local F5-TTS")
                                audio_path = generate_voice(tts_text)
                        else:
                            audio_path = generate_voice(tts_text)
                        logger.info(f"[TTS] Generated: {audio_path}")
                        if audio_path:
                            import os
                            if os.path.exists(audio_path):
                                size = os.path.getsize(audio_path)
                                logger.info(f"[TTS] File exists, size: {size} bytes")
                            else:
                                logger.error(f"[TTS] File does not exist: {audio_path}")
                    except Exception as e:
                        logger.error(f"[TTS] Error: {e}", exc_info=True)

            # Generate avatar video if enabled and we have audio
            avatar_video_path = None
            if avatar_on and AVATAR_AVAILABLE and audio_path:
                try:
                    logger.info(f"[AVATAR] Generating video for outfit: {avatar_outfit_name}")
                    avatar_video_path = generate_avatar_video(audio_path, avatar_outfit_name)
                    if avatar_video_path:
                        logger.info(f"[AVATAR] Video generated: {avatar_video_path}")
                except Exception as e:
                    logger.error(f"[AVATAR] Error generating video: {e}")

            return format_chat_history(new_history), new_history, "", audio_path, image_path, nav_visible, nav_map_html, nav_info, nav_url, avatar_video_path

        # Cloud mode selection handlers
        def update_model_mode(model_choice):
            global _use_gemini
            _use_gemini = (model_choice == "Gemini") and GEMINI_AVAILABLE
            status = f"Model: {'Gemini ✅' if _use_gemini else 'Local ✅'}"
            logger.info(f"[CLOUD] {status}")
            return f"*{status}*"

        def update_voice_mode(voice_choice):
            global _use_elevenlabs, _use_browser_tts
            _use_elevenlabs = False
            _use_browser_tts = (voice_choice == "Browser")

            if voice_choice == "JARVIS":
                try:
                    from alexandra_voice import set_voice
                    set_voice("jarvis")
                except:
                    pass
                status = "Voice: JARVIS ✅"
            elif voice_choice == "Jenny":
                try:
                    from alexandra_voice import set_voice
                    set_voice("jenny_british")
                except:
                    pass
                status = "Voice: Jenny ✅"
            elif _use_browser_tts:
                status = "Voice: Browser ✅"
            else:
                status = "Voice: Local ✅"

            logger.info(f"[CLOUD] {status}")
            return f"*{status}*"

        cloud_model_selector.change(
            update_model_mode,
            inputs=[cloud_model_selector],
            outputs=[cloud_status]
        )

        cloud_voice_selector.change(
            update_voice_mode,
            inputs=[cloud_voice_selector],
            outputs=[cloud_status],
            js="(choice) => { window.browserTTSEnabled = (choice === 'Browser'); console.log('[TTS] Browser mode:', window.browserTTSEnabled); return choice; }"
        )

        send_btn.click(
            process_chat,
            inputs=[chat_input, chat_state, voice_enabled_chat, cloud_model_selector, cloud_voice_selector, avatar_enabled, avatar_outfit],
            outputs=[chatbot, chat_state, chat_input, voice_audio_output, chat_image_output, directions_container, directions_map_display, directions_info, directions_map_url, avatar_video]
        )

        chat_input.submit(
            process_chat,
            inputs=[chat_input, chat_state, voice_enabled_chat, cloud_model_selector, cloud_voice_selector, avatar_enabled, avatar_outfit],
            outputs=[chatbot, chat_state, chat_input, voice_audio_output, chat_image_output, directions_container, directions_map_display, directions_info, directions_map_url, avatar_video]
        )

        # Directions widget handlers
        def close_directions():
            return gr.Column(visible=False)

        def open_maps(url):
            if url:
                import webbrowser
                webbrowser.open(url)
            return f"Opening: {url}"

        directions_close_btn.click(
            close_directions,
            outputs=[directions_container]
        )

        # Stop speaking button - clears audio
        def stop_speaking():
            return None

        stop_speaking_btn.click(
            stop_speaking,
            outputs=[voice_audio_output],
            js="() => { if(window.stopAllAudio) window.stopAllAudio(); }"
        )

        # Test voice button - speaks a test message using browser TTS directly
        test_voice_btn.click(
            lambda: None,
            js="""() => {
                if (!window.speechSynthesis) {
                    alert('Your browser does not support speech synthesis');
                    return;
                }
                // Cancel any ongoing speech
                window.speechSynthesis.cancel();

                // Create utterance
                const text = 'Good evening. I am JARVIS, your personal artificial intelligence assistant. At your service.';
                const utterance = new SpeechSynthesisUtterance(text);

                // Try to find a British voice
                const voices = window.speechSynthesis.getVoices();
                console.log('[TEST] Available voices:', voices.length);
                for (const v of voices) {
                    if (v.lang.includes('en-GB') || v.name.includes('UK') || v.name.includes('Daniel') || v.name.includes('George')) {
                        utterance.voice = v;
                        console.log('[TEST] Using voice:', v.name);
                        break;
                    }
                }

                // JARVIS-style settings
                utterance.rate = 1.0;
                utterance.pitch = 0.9;
                utterance.volume = 1.0;

                // Speak
                console.log('[TEST] Speaking...');
                window.speechSynthesis.speak(utterance);
            }"""
        )

        directions_map_btn.click(
            open_maps,
            inputs=[directions_map_url],
            outputs=[directions_info]
        )

        # Model loading handler
        if LOCAL_MODEL_AVAILABLE and load_model_btn is not None:
            def do_load_model(model_name):
                import subprocess
                import requests
                # Stop Ollama to free GPU memory before loading local model
                try:
                    logger.info("Stopping Ollama to free GPU memory...")
                    # Use Ollama API to unload models (works from inside Docker)
                    ollama_url = CONFIG.get("ollama_url", "http://172.17.0.1:11434")
                    # Send a request with keep_alive=0 to unload the model
                    try:
                        requests.post(f"{ollama_url}/api/generate",
                                     json={"model": "qwen2.5:72b", "keep_alive": 0},
                                     timeout=5)
                        requests.post(f"{ollama_url}/api/generate",
                                     json={"model": "qwen3-coder:30b", "keep_alive": 0},
                                     timeout=5)
                    except:
                        pass
                    logger.info("Ollama unload request sent - GPU memory should be freed")
                except Exception as e:
                    logger.warning(f"Could not stop Ollama: {e}")

                result = load_model(model_name)
                return result

            load_model_btn.click(
                do_load_model,
                inputs=[model_selector],
                outputs=[model_status]
            )

            # Refresh status button handler
            refresh_status_btn.click(
                lambda: get_model_status(),
                outputs=[model_status]
            )

            # Unload model button handler
            unload_model_btn.click(
                lambda: unload_model(),
                outputs=[model_status]
            )

        # Avatar event handlers
        if AVATAR_AVAILABLE:
            def on_avatar_load(outfit_name):
                try:
                    result = load_avatar_models()
                    # Also prepare the avatar after loading models
                    if "loaded" in result.lower() or "already" in result.lower():
                        prep_result = prepare_avatar(outfit_name)
                        return f"*{result} - {prep_result}*"
                    return f"*{result}*"
                except Exception as e:
                    return f"*Error: {e}*"

            def on_avatar_unload():
                try:
                    result = unload_avatar_models()
                    return f"*{result}*"
                except Exception as e:
                    return f"*Error: {e}*"

            def on_avatar_outfit_change(outfit_name):
                try:
                    result = prepare_avatar(outfit_name)
                    return f"*{result}*"
                except Exception as e:
                    return f"*Error: {e}*"

            avatar_load_btn.click(
                on_avatar_load,
                inputs=[avatar_outfit],
                outputs=[avatar_status]
            )

            avatar_unload_btn.click(
                on_avatar_unload,
                outputs=[avatar_status]
            )

            avatar_outfit.change(
                on_avatar_outfit_change,
                inputs=[avatar_outfit],
                outputs=[avatar_status]
            )

        # Voice transcription handler
        import whisper
        whisper_model = None

        async def transcribe_and_chat(audio_path, history, use_tts, avatar_on=False, avatar_outfit_name=None):
            nonlocal whisper_model
            print(f"[VOICE INPUT] Received audio_path: {audio_path}")
            if audio_path is None:
                print("[VOICE INPUT] No audio received - audio_path is None")
                return format_chat_history(history), history, "No audio recorded - please check microphone", None, None, None, None

            # Check if file exists and has content
            import os
            if os.path.exists(audio_path):
                file_size = os.path.getsize(audio_path)
                print(f"[VOICE INPUT] Audio file exists, size: {file_size} bytes")
                if file_size < 1000:
                    print("[VOICE INPUT] Audio file too small - likely silent/empty recording")
                    return format_chat_history(history), history, "Recording was empty - please speak into microphone", None, None, None, None
            else:
                print(f"[VOICE INPUT] Audio file does not exist: {audio_path}")

            try:
                # Apply noise reduction to filter out background TV/noise
                try:
                    import noisereduce as nr
                    import soundfile as sf
                    import numpy as np

                    # Load audio
                    audio_data, sample_rate = sf.read(audio_path)
                    print(f"[NOISE] Loaded audio: {len(audio_data)} samples at {sample_rate}Hz")

                    # Apply noise reduction
                    reduced_noise = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.8)

                    # Save cleaned audio
                    cleaned_path = audio_path.replace('.wav', '_clean.wav').replace('.webm', '_clean.wav')
                    if cleaned_path == audio_path:
                        cleaned_path = audio_path + '_clean.wav'
                    sf.write(cleaned_path, reduced_noise, sample_rate)
                    audio_path = cleaned_path
                    print(f"[NOISE] Noise reduction applied, saved to {cleaned_path}")
                except Exception as noise_err:
                    print(f"[NOISE] Skipping noise reduction: {noise_err}")

                # Load whisper model if needed
                if whisper_model is None:
                    print("[WHISPER] Loading model...")
                    whisper_model = whisper.load_model("base")

                # Transcribe
                result = whisper_model.transcribe(audio_path)
                text = result["text"].strip()
                print(f"[WHISPER] Transcribed: {text}")

                if not text:
                    return format_chat_history(history), history, "", None, None, None, None

                # Now chat with the transcribed text
                new_history, _, image_path = await chat_with_jarvis(text, history)
                audio_out = None

                # Generate TTS if enabled (but not for image generation)
                if use_tts and (VOICE_TTS_AVAILABLE or ELEVENLABS_AVAILABLE) and new_history and not image_path:
                    last_msg = new_history[-1]
                    last_response = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
                    if last_response and len(last_response) > 5:
                        try:
                            # Try ElevenLabs first if enabled
                            if _use_elevenlabs and ELEVENLABS_AVAILABLE:
                                audio_out = elevenlabs_tts(last_response)
                            else:
                                audio_out = generate_voice(last_response)
                        except Exception as e:
                            print(f"[TTS] Error: {e}")

                # Generate avatar video if enabled and we have audio
                avatar_video_out = None
                if avatar_on and AVATAR_AVAILABLE and audio_out:
                    try:
                        avatar_video_out = generate_avatar_video(audio_out, avatar_outfit_name)
                    except Exception as e:
                        print(f"[AVATAR] Error: {e}")

                return format_chat_history(new_history), new_history, text, audio_out, None, image_path, avatar_video_out
            except Exception as e:
                print(f"[WHISPER] Error: {e}")
                return format_chat_history(history), history, f"Error: {e}", None, None, None, None

        transcribe_btn.click(
            transcribe_and_chat,
            inputs=[mic_input, chat_state, voice_enabled_chat, avatar_enabled, avatar_outfit],
            outputs=[chatbot, chat_state, chat_input, voice_audio_output, mic_input, chat_image_output, avatar_video]
        )

        # Widget refresh handlers
        async def refresh_all_widgets(location, topic):
            weather = await get_weather_widget(location)
            time_w = await get_time_widget()
            gpu_w = get_gpu_widget()
            news = await get_news_widget(topic)
            status = get_status_widget(
                state.jarvis_active,
                state.voice_active,
                state.camera_active,
                state.hands_active
            )
            waveform = generate_waveform_image()
            return weather, gpu_w, time_w, news, status, waveform

        refresh_btn.click(
            refresh_all_widgets,
            inputs=[location_input, news_topic],
            outputs=[weather_html, gpu_html, time_html, news_html, status_html, waveform_img]
        )

        # Initial load
        dashboard.load(
            refresh_all_widgets,
            inputs=[location_input, news_topic],
            outputs=[weather_html, gpu_html, time_html, news_html, status_html, waveform_img]
        )

        # Auto-update time and GPU every 5 seconds
        def update_time_and_gpu():
            return get_time_widget_sync(), get_gpu_widget()

        time_timer.tick(
            update_time_and_gpu,
            outputs=[time_html, gpu_html]
        )

        # Image generation handler
        generate_btn.click(
            generate_image,
            inputs=[img_prompt, img_negative, model_select, use_trigger, img_width, img_height, img_steps, img_seed, img_guidance],
            outputs=[output_image, output_info]
        )

        unload_img_btn.click(
            unload_image_models,
            outputs=[img_status]
        )

        load_img_btn.click(
            load_image_models,
            outputs=[img_status]
        )

        # Image to Image handler
        img2img_btn.click(
            generate_img2img,
            inputs=[img2img_input, img2img_prompt, img2img_negative, img2img_model, img2img_strength, img2img_steps, img2img_guidance, img2img_seed],
            outputs=[img2img_output, img2img_info]
        )

        # Capture from server camera handler
        capture_server_btn.click(
            capture_from_server_camera,
            outputs=[camera_feed]
        )

        # Live feed handlers with hand tracking
        hand_tracking_state = gr.State(value=False)

        def start_live_feed(enable_hands):
            logger.info("Starting live feed")
            return gr.Timer(active=True), True, enable_hands

        def stop_live_feed():
            logger.info("Stopping live feed")
            gesture_controller.disable()
            # Return updates for timer, active state, and hand tracking state
            return gr.update(active=False), False, False

        def update_live_feed_with_hands(enable_hands, enable_gesture):
            """Update live feed with optional hand tracking and gesture control"""
            if enable_hands:
                frame, gesture, fingers, action = capture_with_hand_tracking(True, enable_gesture)
                return frame, gesture, fingers, action
            else:
                frame = capture_from_server_camera()
                return frame, "Disabled", "0", "Disabled"

        def toggle_hand_tracking(enabled):
            """Toggle hand tracking on/off"""
            global _hand_tracking_enabled
            _hand_tracking_enabled = enabled
            if enabled:
                return enabled, "Hand tracking enabled - detecting gestures..."
            else:
                return enabled, "Disabled"

        def toggle_gesture_control(enabled):
            """Toggle gesture control on/off"""
            if enabled:
                gesture_controller.enable()
                return "🎮 Gesture control ENABLED - Point to move cursor!"
            else:
                gesture_controller.disable()
                return "Disabled"

        start_live_btn.click(
            start_live_feed,
            inputs=[hands_toggle],
            outputs=[live_feed_timer, live_feed_active, hand_tracking_state]
        )

        stop_live_btn.click(
            stop_live_feed,
            outputs=[live_feed_timer, live_feed_active, hand_tracking_state]
        )

        # Hand tracking toggle handler
        hands_toggle.change(
            toggle_hand_tracking,
            inputs=[hands_toggle],
            outputs=[hand_tracking_state, gesture_display]
        )

        # Gesture control toggle handler
        gesture_control_toggle.change(
            toggle_gesture_control,
            inputs=[gesture_control_toggle],
            outputs=[action_display]
        )

        # Timer tick updates the camera feed with hand tracking and gesture control
        live_feed_timer.tick(
            update_live_feed_with_hands,
            inputs=[hands_toggle, gesture_control_toggle],
            outputs=[camera_feed, gesture_display, finger_display, action_display]
        )

        # Vision analysis handler
        analyze_btn.click(
            analyze_image,
            inputs=[camera_feed, vision_prompt],
            outputs=[vision_result]
        )

        # Vision model load/unload handlers
        load_vision_btn.click(
            handle_load_vision,
            outputs=[vision_status_display]
        )

        unload_vision_btn.click(
            handle_unload_vision,
            outputs=[vision_status_display]
        )

        # ===================== FILES TAB HANDLERS =====================

        # Refresh files list
        refresh_files_btn.click(
            list_workspace_files,
            outputs=[file_list]
        )

        # Create file
        create_file_btn.click(
            create_file_handler,
            inputs=[file_name_input, file_content_input],
            outputs=[file_status]
        ).then(
            list_workspace_files,
            outputs=[file_list]
        )

        # Read file
        read_file_btn.click(
            read_file_handler,
            inputs=[file_name_input],
            outputs=[file_content_input, file_status]
        )

        # New note button - populate with template
        def show_note_template():
            return "my_note.md", "# Note Title\n\nWrite your note here..."

        new_note_btn.click(
            show_note_template,
            outputs=[file_name_input, file_content_input]
        )

        # New project button - create project
        def create_new_project():
            import asyncio
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(create_project_handler("new_project"))
            loop.close()
            return result

        new_project_btn.click(
            create_new_project,
            outputs=[file_status]
        ).then(
            list_workspace_files,
            outputs=[file_list]
        )

        # Load files on tab switch
        dashboard.load(
            list_workspace_files,
            outputs=[file_list]
        )

        # ===================== MEMORY TAB HANDLERS =====================

        # Refresh memory
        def refresh_memory_data():
            return get_stored_facts(), get_conversation_history()

        refresh_memory_btn.click(
            refresh_memory_data,
            outputs=[facts_display, conversations_display]
        )

        # Add new fact
        def add_fact_and_refresh(fact):
            status = save_fact_handler(fact)
            return status, get_stored_facts(), ""

        add_fact_btn.click(
            add_fact_and_refresh,
            inputs=[new_fact_input],
            outputs=[memory_search_results, facts_display, new_fact_input]
        )

        # Search memory
        search_memory_btn.click(
            search_memory_handler,
            inputs=[memory_search],
            outputs=[memory_search_results]
        )

        memory_search.submit(
            search_memory_handler,
            inputs=[memory_search],
            outputs=[memory_search_results]
        )

        # Load memory data on tab
        dashboard.load(
            refresh_memory_data,
            outputs=[facts_display, conversations_display]
        )

        # ===================== RAG/DOCUMENTS TAB HANDLERS =====================

        # Upload and process documents
        upload_btn.click(
            process_document_upload,
            inputs=[doc_upload],
            outputs=[upload_status, doc_list]
        )

        # Query documents
        query_btn.click(
            query_documents_handler,
            inputs=[doc_query],
            outputs=[doc_results, doc_sources]
        )

        doc_query.submit(
            query_documents_handler,
            inputs=[doc_query],
            outputs=[doc_results, doc_sources]
        )

        # Load document list
        dashboard.load(
            get_indexed_documents,
            outputs=[doc_list]
        )

        # ===================== SYSTEM TAB HANDLERS =====================

        def get_system_stats():
            """Get current system statistics"""
            if PROACTIVE_AVAILABLE and proactive:
                return proactive.system_monitor.get_summary()
            return "System monitoring not available"

        def get_gpu_status():
            """Get GPU status - with fallback for GB10 GPU"""
            import subprocess
            try:
                # Try direct nvidia-smi for GB10 compatibility
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    gpu_util = parts[0] if parts[0] != "[N/A]" else "N/A"
                    gpu_temp = parts[1] if len(parts) > 1 and parts[1] != "[N/A]" else "N/A"
                    gpu_power = parts[2] if len(parts) > 2 and parts[2] != "[N/A]" else "N/A"

                    # Get process memory usage (works on GB10)
                    proc_result = subprocess.run(
                        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5
                    )
                    total_mem = 0
                    if proc_result.returncode == 0 and proc_result.stdout.strip():
                        for line in proc_result.stdout.strip().split("\n"):
                            if "," in line:
                                try:
                                    mem = int(line.split(",")[1].strip())
                                    total_mem += mem
                                except:
                                    pass

                    mem_str = f"{total_mem} MiB" if total_mem > 0 else "0 MiB"
                    temp_str = f"{gpu_temp}°C" if gpu_temp != "N/A" else "N/A"
                    power_str = f"{gpu_power}W" if gpu_power != "N/A" else ""

                    return f"🎮 GPU: {gpu_util}% | 🌡️ {temp_str} | 💾 {mem_str} {power_str}".strip()
            except Exception as e:
                pass

            # Fallback to proactive module
            if PROACTIVE_AVAILABLE and proactive:
                stats = proactive.system_monitor.last_stats
                if stats:
                    gpu_pct = (stats.gpu_memory_used / stats.gpu_memory_total * 100) if stats.gpu_memory_total > 0 else 0
                    return f"Usage: {stats.gpu_usage:.0f}% | Memory: {stats.gpu_memory_used:.1f}/{stats.gpu_memory_total:.0f} GB ({gpu_pct:.0f}%) | Temp: {stats.gpu_temp:.0f}°C"
                proactive.system_monitor.collect_stats()
                return get_gpu_status()
            return "GPU monitoring not available"

        def get_alerts():
            """Get current alerts"""
            if PROACTIVE_AVAILABLE and proactive:
                return proactive.alert_manager.get_summary()
            return "Alerts not available"

        def clear_all_alerts():
            """Clear all alerts"""
            if PROACTIVE_AVAILABLE and proactive:
                proactive.alert_manager.acknowledge_all()
                return "All alerts acknowledged"
            return "Alerts not available"

        def get_training_status():
            """Get training job status"""
            if PROACTIVE_AVAILABLE and proactive:
                return proactive.training_monitor.get_status_summary()
            return "Training monitor not available"

        # Training control state
        training_process = {"pid": None, "running": False}

        def start_training(train_type, dataset):
            """Start the training script"""
            import subprocess
            import os

            # Check if already running
            if training_process["running"]:
                return "⚠️ Training already in progress!", get_training_log()

            checkpoint_info = ""

            if "Voice" in train_type:
                # Voice training with F5-TTS
                checkpoint_dir = f"/workspace/voice_training/F5-TTS/ckpts/{dataset}"
                try:
                    if os.path.exists(checkpoint_dir):
                        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("model_")]
                        if checkpoints:
                            checkpoint_info = f"\n📂 Checkpoints found in: {dataset}"
                except:
                    pass

                try:
                    proc = subprocess.Popen(
                        ["python3", "/workspace/train_voice.py", "--dataset", dataset, "--epochs", "100"],
                        stdout=open("/tmp/training_output.log", "w"),
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        cwd="/workspace"
                    )
                    training_process["pid"] = proc.pid
                    training_process["running"] = True
                    training_process["type"] = "voice"
                    training_process["dataset"] = dataset
                    logger.info(f"[TRAINING] Voice training started with PID {proc.pid}")
                    return f"🟢 Voice Training STARTED (PID: {proc.pid})\n🎤 Dataset: {dataset}{checkpoint_info}", "Starting voice training... refresh log in a few seconds"
                except Exception as e:
                    logger.error(f"[TRAINING] Failed to start: {e}")
                    return f"❌ Failed to start: {e}", str(e)
            else:
                # LoRA training
                checkpoint_dir = "/workspace/coding-lora"
                try:
                    if os.path.exists(checkpoint_dir):
                        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
                        if checkpoints:
                            latest = max(checkpoints, key=lambda x: int(x.split('-')[1]) if '-' in x else 0)
                            checkpoint_info = f"\n📂 Will resume from: {latest}"
                except:
                    pass

                try:
                    # Find latest checkpoint
                    ckpt_dir = "/workspace/host/ai-clone-training/my-output/alexandra-qwen72b-lora"
                    latest_ckpt = None
                    if os.path.exists(ckpt_dir):
                        ckpts = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
                        if ckpts:
                            latest_ckpt = os.path.join(ckpt_dir, max(ckpts, key=lambda x: int(x.split('-')[1])))

                    cmd = ["python3", "/workspace/host/ai-clone-training/train_domain.py", "coding"]
                    if latest_ckpt:
                        cmd.extend(["--resume", latest_ckpt])

                    proc = subprocess.Popen(
                        cmd,
                        stdout=open("/tmp/training_output.log", "w"),
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        cwd="/workspace/host/ai-clone-training"
                    )
                    training_process["pid"] = proc.pid
                    training_process["running"] = True
                    training_process["type"] = "lora"
                    logger.info(f"[TRAINING] LoRA training started with PID {proc.pid}")
                    return f"🟢 LoRA Training STARTED (PID: {proc.pid}){checkpoint_info}", "Starting training... refresh log in a few seconds"
                except Exception as e:
                    logger.error(f"[TRAINING] Failed to start: {e}")
                    return f"❌ Failed to start: {e}", str(e)

        def stop_training():
            """Stop the training script"""
            import subprocess

            try:
                # Kill both types of training processes
                subprocess.run(["pkill", "-f", "train_domain.py"], capture_output=True, text=True, timeout=10)
                subprocess.run(["pkill", "-f", "train_voice.py"], capture_output=True, text=True, timeout=10)

                train_type = training_process.get("type", "unknown")
                training_process["running"] = False
                training_process["pid"] = None
                training_process["type"] = None
                logger.info("[TRAINING] Stopped")
                return f"🔴 Training STOPPED ({train_type})\n⚠️ Checkpoint saved at last save point", get_training_log()
            except Exception as e:
                logger.error(f"[TRAINING] Failed to stop: {e}")
                return f"❌ Failed to stop: {e}", str(e)

        def get_training_log():
            """Get the last 20 lines of training output"""
            import subprocess
            try:
                result = subprocess.run(
                    ["tail", "-20", "/tmp/training_output.log"],
                    capture_output=True, text=True, timeout=5
                )
                if result.stdout.strip():
                    return result.stdout
                return "No training log available yet..."
            except Exception as e:
                return f"Could not read log: {e}"

        def check_training_running():
            """Check if training is currently running"""
            import subprocess
            try:
                # Check for LoRA training
                result = subprocess.run(
                    ["pgrep", "-f", "train_domain.py"],
                    capture_output=True, text=True, timeout=5
                )
                if result.stdout.strip():
                    training_process["running"] = True
                    training_process["type"] = "lora"
                    return f"🟢 LoRA Training in progress (PID: {result.stdout.strip()})"

                # Check for Voice training
                result = subprocess.run(
                    ["pgrep", "-f", "train_voice.py"],
                    capture_output=True, text=True, timeout=5
                )
                if result.stdout.strip():
                    training_process["running"] = True
                    training_process["type"] = "voice"
                    dataset = training_process.get("dataset", "unknown")
                    return f"🟢 Voice Training in progress (PID: {result.stdout.strip()})\n🎤 Dataset: {dataset}"

                training_process["running"] = False
                return "⚪ Idle - Ready to train"
            except:
                return "⚪ Idle - Ready to train"

        def refresh_training_display():
            """Refresh both status and log"""
            status = check_training_running()
            log = get_training_log()
            return status, log

        def get_routines_list():
            """Get list of available routines"""
            if PROACTIVE_AVAILABLE and proactive:
                routines = proactive.routine_manager.list_routines()
                return [[r["name"], r["description"], r["trigger"]] for r in routines]
            return [["N/A", "Routines not available", ""]]

        def run_routine(routine_name):
            """Run a routine"""
            if PROACTIVE_AVAILABLE and proactive:
                import asyncio
                loop = asyncio.new_event_loop()
                try:
                    results = loop.run_until_complete(proactive.routine_manager.execute_routine(routine_name))
                    return "\n".join(results) if results else "Routine completed"
                finally:
                    loop.close()
            return "Routines not available"

        def get_reminders():
            """Get reminder list"""
            if SCHEDULER_AVAILABLE and scheduler:
                return scheduler.reminders.get_summary()
            return "Reminders not available"

        def add_reminder(text):
            """Add a new reminder"""
            if SCHEDULER_AVAILABLE and scheduler:
                result = scheduler.add_reminder(text)
                return scheduler.reminders.get_summary(), result
            return "Reminders not available", "Reminders not available"

        def get_schedule():
            """Get today's schedule"""
            if SCHEDULER_AVAILABLE and scheduler:
                return scheduler.get_schedule()
            return "Schedule not available"

        def toggle_auto_refresh(enabled):
            """Toggle auto-refresh for system stats"""
            return gr.Timer(active=enabled)

        def refresh_all_system():
            """Refresh all system stats"""
            stats = get_system_stats()
            gpu = get_gpu_status()
            return stats, gpu

        # System stats handlers
        refresh_stats_btn.click(
            refresh_all_system,
            outputs=[system_stats_display, gpu_usage_display]
        )

        auto_refresh_toggle.change(
            toggle_auto_refresh,
            inputs=[auto_refresh_toggle],
            outputs=[system_refresh_timer]
        )

        system_refresh_timer.tick(
            refresh_all_system,
            outputs=[system_stats_display, gpu_usage_display]
        )

        # Alerts handlers
        refresh_alerts_btn.click(
            get_alerts,
            outputs=[alerts_display]
        )

        clear_alerts_btn.click(
            clear_all_alerts,
            outputs=[alerts_display]
        )

        # Training controls handlers
        start_training_btn.click(
            start_training,
            inputs=[training_type, voice_dataset],
            outputs=[training_status_display, training_log_display]
        )

        # Show/hide voice dataset dropdown and update status based on training type
        def get_training_type_status(train_type, dataset):
            """Get checkpoint status for selected training type"""
            import os
            import subprocess

            # First check if anything is currently running
            try:
                lora_running = subprocess.run(["pgrep", "-f", "train_domain.py"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                voice_running = subprocess.run(["pgrep", "-f", "train_voice.py"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
            except:
                lora_running = voice_running = ""

            if "Voice" in train_type:
                # Check voice training status
                ckpt_dir = f"/workspace/voice_training/F5-TTS/ckpts/{dataset}"
                data_dir = f"/workspace/voice_training/F5-TTS/data/{dataset}_char"

                status_parts = [f"🎤 Voice Training: {dataset}"]

                # Check for checkpoints
                try:
                    if os.path.exists(ckpt_dir):
                        ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("model_")]
                        if ckpts:
                            latest = sorted(ckpts)[-1]
                            status_parts.append(f"📂 Last checkpoint: {latest}")
                        else:
                            status_parts.append("📂 No checkpoints yet (fresh start)")
                    else:
                        status_parts.append("📂 No checkpoints yet (fresh start)")
                except:
                    pass

                # Check dataset info
                try:
                    duration_file = f"{data_dir}/duration.json"
                    if os.path.exists(duration_file):
                        import json
                        with open(duration_file) as f:
                            durations = json.load(f).get("duration", [])
                        total_mins = sum(durations) / 60
                        status_parts.append(f"📊 Dataset: {len(durations)} samples ({total_mins:.1f} min)")
                except:
                    pass

                if voice_running:
                    status_parts.insert(0, f"🟢 RUNNING (PID: {voice_running})")
                else:
                    status_parts.append("⚪ Ready to train")

                return gr.update(visible=True), "\n".join(status_parts)
            else:
                # Check LoRA training status
                ckpt_dir = "/workspace/coding-lora"
                status_parts = ["🧠 LoRA Training: Code Assistant"]

                try:
                    if os.path.exists(ckpt_dir):
                        ckpts = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
                        if ckpts:
                            latest = max(ckpts, key=lambda x: int(x.split('-')[1]) if '-' in x else 0)
                            step = int(latest.split('-')[1]) if '-' in latest else 0
                            total_steps = 27130  # From the training config
                            pct = (step / total_steps) * 100
                            status_parts.append(f"📂 Last checkpoint: {latest}")
                            status_parts.append(f"📊 Progress: ~{pct:.1f}% ({step:,}/{total_steps:,} steps)")
                        else:
                            status_parts.append("📂 No checkpoints yet (fresh start)")
                    else:
                        status_parts.append("📂 No checkpoints yet (fresh start)")
                except:
                    pass

                if lora_running:
                    status_parts.insert(0, f"🟢 RUNNING (PID: {lora_running})")
                else:
                    status_parts.append("⚪ Ready to train")

                return gr.update(visible=False), "\n".join(status_parts)

        training_type.change(
            get_training_type_status,
            inputs=[training_type, voice_dataset],
            outputs=[voice_dataset, training_status_display]
        )

        voice_dataset.change(
            get_training_type_status,
            inputs=[training_type, voice_dataset],
            outputs=[voice_dataset, training_status_display]
        )

        stop_training_btn.click(
            stop_training,
            outputs=[training_status_display, training_log_display]
        )

        refresh_training_btn.click(
            refresh_training_display,
            outputs=[training_status_display, training_log_display]
        )

        # Auto-refresh training log timer
        training_log_timer.tick(
            refresh_training_display,
            outputs=[training_status_display, training_log_display]
        )

        # Toggle auto-refresh for training
        def toggle_training_refresh(enabled):
            return gr.Timer(active=enabled)

        auto_refresh_training.change(
            toggle_training_refresh,
            inputs=[auto_refresh_training],
            outputs=[training_log_timer]
        )

        # Routines handlers
        dashboard.load(
            get_routines_list,
            outputs=[routines_list]
        )

        run_routine_btn.click(
            run_routine,
            inputs=[routine_select],
            outputs=[routine_output]
        )

        # Reminders handlers
        add_reminder_btn.click(
            add_reminder,
            inputs=[reminder_input],
            outputs=[reminders_display, reminder_status]
        )

        reminder_input.submit(
            add_reminder,
            inputs=[reminder_input],
            outputs=[reminders_display, reminder_status]
        )

        refresh_schedule_btn.click(
            get_schedule,
            outputs=[schedule_display]
        )

        # ===================== EPSTEIN FILES HANDLERS =====================
        epstein_search_btn.click(
            epstein_search,
            inputs=[epstein_query, epstein_n],
            outputs=[epstein_results, epstein_results_dropdown]
        )

        # View PDF inline
        epstein_view_pdf_btn.click(
            epstein_view_pdf,
            inputs=[epstein_results_dropdown],
            outputs=[epstein_pdf_viewer]
        ).then(
            epstein_view_document,
            inputs=[epstein_results_dropdown],
            outputs=[epstein_doc_viewer]
        )

        # Download PDF
        epstein_download_btn.click(
            epstein_get_pdf,
            inputs=[epstein_results_dropdown],
            outputs=[epstein_pdf_file]
        ).then(
            lambda: gr.update(visible=True),
            outputs=[epstein_pdf_file]
        )

        epstein_stats_btn.click(
            epstein_get_stats,
            outputs=[epstein_stats_display]
        )

        epstein_check_btn.click(
            epstein_check_new_files,
            outputs=[epstein_file_status]
        )

        epstein_index_btn.click(
            epstein_index_files,
            outputs=[epstein_file_status]
        )

        # ===================== NEWS HANDLERS =====================
        news_search_btn.click(
            news_search,
            inputs=[news_query, news_n],
            outputs=[news_results]
        )

        def get_news_status():
            """Get current news article count"""
            try:
                if state.memory:
                    stats = state.memory.stats()
                    return f"📰 {stats.get('news', 0)} articles indexed"
                return "News system not loaded"
            except:
                return "Unable to get status"

        def refresh_news_feeds():
            """Fetch fresh news from all RSS feeds"""
            try:
                if state.memory:
                    count = state.memory.update_news_feeds(max_per_feed=20)
                    stats = state.memory.stats()
                    return f"✅ Fetched {count} new articles! Total: {stats.get('news', 0)} articles"
                return "News system not loaded"
            except Exception as e:
                return f"Error refreshing: {e}"

        refresh_news_btn.click(
            refresh_news_feeds,
            outputs=[news_status]
        )

        # Load news status on page load
        dashboard.load(
            get_news_status,
            outputs=[news_status]
        )

        # Load initial data
        dashboard.load(
            get_reminders,
            outputs=[reminders_display]
        )

        dashboard.load(
            get_schedule,
            outputs=[schedule_display]
        )

        # Auto-start camera feed on dashboard load
        def auto_start_camera():
            """Auto-start camera feed when dashboard loads"""
            logger.info("Auto-starting camera feed...")
            return gr.Timer(active=True), True, False

        dashboard.load(
            auto_start_camera,
            outputs=[live_feed_timer, live_feed_active, hand_tracking_state]
        )

    return dashboard


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Launch the dashboard"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║       ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗                   ║
║       ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝                   ║
║       ██║███████║██████╔╝██║   ██║██║███████╗                   ║
║  ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║                   ║
║  ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║                   ║
║   ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝                   ║
║                                                                  ║
║         J.A.R.V.I.S. - Your Personal AI Assistant               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    # Auto-load the Vision-Language model on startup
    if False and LOCAL_MODEL_AVAILABLE:  # DISABLED
        print("[JARVIS] Auto-loading Qwen2.5-VL-72B vision model...")
        try:
            # Prefer VL model for vision+chat, fall back to first available
            vl_model = "Qwen2.5-VL-72B (Vision+Chat)"
            if vl_model in AVAILABLE_MODELS:
                result = load_model(vl_model)
                print(f"[JARVIS] {result}")
            elif AVAILABLE_MODELS:
                result = load_model(DEFAULT_MODEL)
                print(f"[JARVIS] {result}")
        except Exception as e:
            print(f"[JARVIS] Auto-load failed: {e}")

    dashboard = create_dashboard()

    import os
    port = int(os.environ.get('GRADIO_SERVER_PORT', 7860))
    # Use Gradio share for HTTPS (creates secure gradio.live URL)
    use_share = os.environ.get('JARVIS_SHARE', 'true').lower() == 'true'
    print(f"[JARVIS] Starting with share={use_share}")

    # Warmup voice TTS at startup so first response sounds good
    if VOICE_TTS_AVAILABLE and warmup_voice:
        print("[JARVIS] Warming up voice TTS...")
        warmup_voice()
        print("[JARVIS] Voice warmup complete!")

    # Enable queue for concurrent request handling (prevents UI hang during long TTS)
    dashboard.queue(
        max_size=20,
        default_concurrency_limit=2
    )
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=use_share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
