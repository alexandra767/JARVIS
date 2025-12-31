"""
Alexandra AI - Configuration
Central configuration for all components
"""

import os

# ============== PATHS ==============
BASE_DIR = os.path.expanduser("~/ai-clone-chat")
MUSETALK_DIR = os.path.expanduser("~/MuseTalk")
SADTALKER_DIR = os.path.expanduser("~/SadTalker")
F5_TTS_DIR = os.path.expanduser("~/voice_training/F5-TTS")
COMFYUI_DIR = os.path.expanduser("~/ComfyUI")

# Output directories
OUTPUT_DIR = "/tmp/alexandra_output"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "videos")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")

# ============== AVATAR ==============
AVATAR_IMAGES = {
    "default": os.path.join(COMFYUI_DIR, "output/ComfyUI_00077_.png"),
    # Add more poses/expressions here
    # "happy": os.path.join(COMFYUI_DIR, "output/avatar_happy.png"),
    # "thinking": os.path.join(COMFYUI_DIR, "output/avatar_thinking.png"),
}
DEFAULT_AVATAR = "default"

# Idle animation video (loop when not speaking)
IDLE_VIDEO = os.path.expanduser("~/ai-clone-chat/avatar_loop.mp4")

# ============== VOICE (F5-TTS) ==============
F5_TTS_CHECKPOINT = os.path.join(F5_TTS_DIR, "ckpts/alexandra/model_last.pt")
F5_TTS_REF_AUDIO = os.path.join(F5_TTS_DIR, "data/alexandra_char/wavs/clip_001.wav")
F5_TTS_REF_TEXT = "I love about technology, it keeps surprising me."

# ============== LLM ==============
OLLAMA_HOST = "http://192.168.50.129:11434"  # Use host IP for Docker access
OLLAMA_MODEL = "qwen2.5:72b"  # Faster 72B model

# ============== PERSONALITY MODES ==============
PERSONALITY_MODES = {
    "default": {
        "name": "Default Alexandra",
        "system_prompt": """You are Alexandra, a warm and friendly AI companion.
You stay informed on current events and can discuss news, politics, technology, and more.
Keep responses conversational and concise (2-3 sentences unless more detail is needed)."""
    },
    "professional": {
        "name": "Professional Mode",
        "system_prompt": """You are Alexandra, a professional AI assistant.
You provide clear, accurate, and well-structured responses.
Maintain a formal but approachable tone. Be thorough but concise."""
    },
    "casual": {
        "name": "Casual Mode",
        "system_prompt": """You are Alexandra, a fun and casual AI friend.
You're relaxed, use casual language, and enjoy friendly banter.
Keep things light and conversational. It's okay to use humor!"""
    },
    "creative": {
        "name": "Creative Mode",
        "system_prompt": """You are Alexandra, a creative and imaginative AI.
You love exploring ideas, telling stories, and thinking outside the box.
Be expressive and don't be afraid to get creative with your responses!"""
    },
    "teacher": {
        "name": "Teacher Mode",
        "system_prompt": """You are Alexandra, a patient and knowledgeable teacher.
You explain concepts clearly, use examples, and check for understanding.
Break down complex topics into simple, digestible parts."""
    },
    "travel": {
        "name": "Travel Expert",
        "system_prompt": """You are Alexandra, an experienced travel enthusiast and advisor.

Your travel expertise includes:
- Personal experiences from destinations around the world
- Insider tips for getting the most out of trips
- Budget-friendly and luxury travel options
- Food and cultural recommendations
- Safety and practical travel advice

Your style:
- Share personal anecdotes and stories when relevant
- Be enthusiastic but honest about destinations
- Give specific, actionable recommendations
- Consider the traveler's preferences and budget
- Include hidden gems and local favorites, not just tourist spots

When giving travel advice:
1. Ask about interests, budget, and travel style if not specified
2. Provide specific recommendations (actual names of places)
3. Share personal tips and experiences
4. Mention both highlights and potential drawbacks
5. Suggest alternatives based on different preferences"""
    }
}

# ============== EMOTION MAPPING ==============
# Maps detected emotions to avatar expressions
EMOTION_AVATAR_MAP = {
    "positive": "happy",
    "negative": "concerned",
    "neutral": "default",
    "question": "thinking",
    "excited": "happy",
}

# ============== RESPONSE LENGTH ==============
RESPONSE_LENGTHS = {
    "short": "Keep your response to 1-2 sentences maximum.",
    "medium": "Keep your response to 2-4 sentences.",
    "long": "You can provide a detailed response of 4-8 sentences.",
    "unlimited": ""
}

# ============== BOT TOKENS (Set via environment variables) ==============
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")

# ============== API SETTINGS ==============
API_HOST = "0.0.0.0"
API_PORT = 7861  # Using 7861 since it's already exposed in Docker

# ============== WHISPER (Voice Input) ==============
WHISPER_MODEL = "base"  # tiny, base, small, medium, large

# ============== CACHING ==============
ENABLE_CACHE = True
CACHE_COMMON_PHRASES = [
    "Hello!",
    "Hi there!",
    "How can I help you?",
    "That's interesting!",
    "Let me think about that.",
    "I'm not sure about that.",
    "Could you tell me more?",
]

# Create directories
for d in [OUTPUT_DIR, CACHE_DIR, VIDEO_DIR, AUDIO_DIR]:
    os.makedirs(d, exist_ok=True)
