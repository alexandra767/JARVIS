"""
Alexandra AI - Voice Module
Shared voice/TTS functionality using F5-TTS
"""

import os
import re
import subprocess
import tempfile

# F5-TTS Configuration
F5_TTS_DIR = "/workspace/host/voice_training/F5-TTS"
# Use jenny_british model - well-trained, use p254 reference for male voice cloning
F5_TTS_CHECKPOINT = os.path.join(F5_TTS_DIR, "ckpts/jenny_british/model_last.pt")

# Voice configurations
VOICE_CONFIGS = {
    "jarvis": {
        # Use jenny_british model with p254 Surrey male reference audio for voice cloning
        # This gives British male voice without vocab mismatch issues
        "ref_audio": os.path.join(F5_TTS_DIR, "data/p254_surrey/wavs/p254_003.wav"),
        "ref_text": "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
        "vocab_file": os.path.join(F5_TTS_DIR, "data/jenny_british_char/vocab.txt"),
        "checkpoint": os.path.join(F5_TTS_DIR, "ckpts/jenny_british/model_last.pt"),
        "pitch_shift": 1.0,  # Natural male voice
        "speed": 1.0,
    },
    "jarvis_surrey": {
        # Same config - British male via voice cloning
        "ref_audio": os.path.join(F5_TTS_DIR, "data/p254_surrey/wavs/p254_003.wav"),
        "ref_text": "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
        "vocab_file": os.path.join(F5_TTS_DIR, "data/jenny_british_char/vocab.txt"),
        "checkpoint": os.path.join(F5_TTS_DIR, "ckpts/jenny_british/model_last.pt"),
        "pitch_shift": 1.0,
        "speed": 1.0,
    },
    "jarvis_london": {
        # Original London accent (p243) - backup option
        "ref_audio": os.path.join(F5_TTS_DIR, "data/british_male/wavs/p243_003_mic1.wav"),
        "ref_text": "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
        "vocab_file": os.path.join(F5_TTS_DIR, "data/british_male_char/vocab.txt"),
        "pitch_shift": 1.0,
        "speed": 1.0,
    },
    "jenny_british": {
        # Female British voice (fallback)
        "ref_audio": os.path.join(F5_TTS_DIR, "data/jenny_british/wavs/clip_0000.wav"),
        "ref_text": "It was a bright cold day in April, and the clocks were striking thirteen.",
        "vocab_file": os.path.join(F5_TTS_DIR, "data/jenny_british_char/vocab.txt"),
        "checkpoint": os.path.join(F5_TTS_DIR, "ckpts/jenny_british/model_last.pt"),
        "pitch_shift": 1.0,
    },
    "alexandra": {
        "ref_audio": os.path.join(F5_TTS_DIR, "data/alexandra/wavs/clip_0000.wav"),
        "ref_text": "Hello, I'm Alexandra. How can I help you today?",
        "pitch_shift": 1.15,  # Feminine pitch
    },
}

# Current voice selection (can be changed at runtime)
current_voice = "jarvis"

# Legacy compatibility
F5_TTS_REF_AUDIO = VOICE_CONFIGS["jarvis"]["ref_audio"]
F5_TTS_REF_TEXT = VOICE_CONFIGS["jarvis"]["ref_text"]

# Check if F5-TTS is available
F5_TTS_AVAILABLE = os.path.exists(F5_TTS_CHECKPOINT)


def clean_text_for_speech(text):
    """Clean text speak and prepare for natural TTS"""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)

    # British pronunciation corrections (for more authentic JARVIS sound)
    british_pronunciations = {
        r"\bma'am\b": "mum",
        r"\bMa'am\b": "Mum",
        r"\bmadam\b": "madum",
        r"\bMadam\b": "Madum",
        r"\bMs\.": "Miz",
        r"\bMrs\.": "Missus",
        r"\bMr\.": "Mister",
        r"\bcan't\b": "cahnt",
        r"\bCan't\b": "Cahnt",
        r"\bshan't\b": "shahnt",
        r"\brather\b": "rahther",
        r"\bRather\b": "Rahther",
    }

    for pattern, replacement in british_pronunciations.items():
        text = re.sub(pattern, replacement, text)

    # Replace text speak with natural pauses or remove
    replacements = {
        r"\blol\b": "",
        r"\bLOL\b": "",
        r"\blmao\b": "",
        r"\bLMAO\b": "",
        r"\brofl\b": "",
        r"\bhaha\b": "ha ha",
        r"\bhahaha\b": "ha ha ha",
        r"\bHaha\b": "ha ha",
        r"\bHahaha\b": "ha ha ha",
        r"\bhehe\b": "he he",
        r"\bomg\b": "oh my god",
        r"\bOMG\b": "oh my god",
        r"\bbtw\b": "by the way",
        r"\bBTW\b": "by the way",
        r"\bidk\b": "I don't know",
        r"\bIDK\b": "I don't know",
        r"\bimo\b": "in my opinion",
        r"\bIMO\b": "in my opinion",
        r"\btbh\b": "to be honest",
        r"\bTBH\b": "to be honest",
        r"\brn\b": "right now",
        r"\bwbu\b": "what about you",
        r"\bwyd\b": "what are you doing",
        r"\btysm\b": "thank you so much",
        r"\bnp\b": "no problem",
        r"\bthx\b": "thanks",
        r"\bpls\b": "please",
        r"\bu\b": "you",
        r"\br\b": "are",
        r"\bur\b": "your",
        r"\b:[\)]+": "",
        r"\b:\(": "",
        r"\b<3\b": "",
        r"\.\.\.+": "...",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Clean up extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove non-speech characters
    text = re.sub(r"[^\w\s.,!?'-]", "", text).strip()

    return text


def set_voice(voice_name):
    """Set the current voice for TTS.

    Args:
        voice_name: Name of voice ('jarvis', 'jenny_british', 'alexandra')

    Returns:
        True if voice set successfully, False otherwise
    """
    global current_voice, F5_TTS_REF_AUDIO, F5_TTS_REF_TEXT

    if voice_name in VOICE_CONFIGS:
        current_voice = voice_name
        F5_TTS_REF_AUDIO = VOICE_CONFIGS[voice_name]["ref_audio"]
        F5_TTS_REF_TEXT = VOICE_CONFIGS[voice_name]["ref_text"]
        print(f"[VOICE] Switched to {voice_name} voice")
        return True
    else:
        print(f"[VOICE] Unknown voice: {voice_name}")
        return False


def get_available_voices():
    """Get list of available voice names"""
    return list(VOICE_CONFIGS.keys())


def generate_voice(text, output_path=None, pitch_shift=None, voice=None):
    """Generate voice using F5-TTS with optional pitch shifting.

    Args:
        text: Text to synthesize
        output_path: Output audio file path (optional, uses temp if None)
        pitch_shift: Pitch multiplier (None=use voice default, 1.0=normal, 1.15=feminine)
        voice: Voice to use (None=use current_voice)

    Returns:
        Path to generated audio file, or None if failed
    """
    global current_voice

    if not F5_TTS_AVAILABLE:
        print("[VOICE] F5-TTS not available")
        return None

    # Get voice config
    voice_name = voice or current_voice
    if voice_name not in VOICE_CONFIGS:
        voice_name = "jarvis"  # Default to JARVIS

    voice_config = VOICE_CONFIGS[voice_name]
    ref_audio = voice_config["ref_audio"]
    ref_text = voice_config["ref_text"]

    # Use voice default pitch if not specified
    if pitch_shift is None:
        pitch_shift = voice_config.get("pitch_shift", 1.0)

    clean_text = clean_text_for_speech(text)

    if len(clean_text) < 5:
        return None

    # Use system Python
    f5_python = "/usr/bin/python3"
    if not os.path.exists(f5_python):
        f5_python = os.path.expanduser("~/MuseTalk/venv/bin/python")

    # Set up output paths - use unique filename to avoid Gradio caching issues
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav", prefix="alexandra_voice_")
        output_dir = os.path.dirname(output_path)
    else:
        output_dir = os.path.dirname(output_path)
        if not output_dir:
            output_dir = "."

    os.makedirs(output_dir, exist_ok=True)

    # Generate to temp file if pitch shifting
    if pitch_shift != 1.0:
        temp_output = os.path.join(output_dir, "temp_voice.wav")
        output_file = "temp_voice.wav"
    else:
        temp_output = output_path
        output_file = os.path.basename(output_path)

    # Get speed from voice config or default
    speed = str(voice_config.get("speed", 1.0))
    vocab_file = voice_config.get("vocab_file", "")
    # Use voice-specific checkpoint if available, otherwise use default
    checkpoint = voice_config.get("checkpoint", F5_TTS_CHECKPOINT)

    voice_cmd = [
        f5_python, "/workspace/ai-clone-chat/simple_f5_infer.py",
        "--ref_audio", ref_audio,
        "--ref_text", ref_text,
        "--gen_text", clean_text[:2000],
        "--output_dir", output_dir,
        "--output_file", output_file,
        "--ckpt_file", checkpoint,
        "--speed", speed
    ]

    # Add vocab file if specified (required for custom-trained models)
    if vocab_file:
        voice_cmd.extend(["--vocab_file", vocab_file])

    print(f"[VOICE] Using checkpoint: {checkpoint}")

    print(f"[VOICE] Using {voice_name} voice (speed={speed})")
    
    # Dynamic timeout based on text length
    voice_timeout = max(300, len(clean_text) // 3)
    est_audio_secs = len(clean_text) / 15
    
    print(f"[VOICE] Generating {len(clean_text)} chars (~{est_audio_secs:.0f}s audio)")
    
    import time as _time
    _voice_start = _time.time()
    
    try:
        result = subprocess.run(
            voice_cmd, 
            capture_output=True, 
            text=True, 
            cwd=F5_TTS_DIR, 
            timeout=voice_timeout
        )
        _voice_elapsed = _time.time() - _voice_start
        
        if result.returncode != 0:
            print(f"[VOICE] Failed after {_voice_elapsed:.1f}s: {result.stderr[:300]}")
            return None
        else:
            print(f"[VOICE] Complete! Took {_voice_elapsed:.1f}s")
    except subprocess.TimeoutExpired:
        print(f"[VOICE] Timeout after {voice_timeout}s")
        return None
    except Exception as e:
        print(f"[VOICE] Error: {e}")
        return None

    if not os.path.exists(temp_output):
        print(f"[VOICE] Output file not created: {temp_output}")
        print(f"[VOICE] Subprocess stderr: {result.stderr[:500] if result and result.stderr else 'none'}")
        return None

    print(f"[VOICE] File created: {temp_output}")
    
    # Apply pitch shifting for feminization
    if pitch_shift != 1.0:
        tempo_compensation = 1.0 / pitch_shift
        pitch_cmd = [
            "ffmpeg", "-y", "-i", temp_output,
            "-af", f"asetrate=44100*{pitch_shift},atempo={tempo_compensation},aresample=44100",
            "-ar", "44100",
            output_path
        ]
        try:
            pitch_result = subprocess.run(pitch_cmd, capture_output=True, text=True, timeout=60)
            if pitch_result.returncode != 0:
                print(f"[VOICE] Pitch shift failed, using original")
                os.rename(temp_output, output_path)
            else:
                print(f"[VOICE] Pitch shift applied: {pitch_shift}x")
                os.remove(temp_output)
        except Exception as e:
            print(f"[VOICE] Pitch shift error: {e}")
            os.rename(temp_output, output_path)
    
    return output_path if os.path.exists(output_path) else None


def get_voice_status():
    """Get voice system status"""
    if F5_TTS_AVAILABLE:
        return "F5-TTS Ready"
    return "Voice not available"


# Track if warmup has been done
_warmup_done = False

def warmup_voice():
    """Run a quick warmup generation to pre-load models"""
    global _warmup_done
    if _warmup_done:
        return "Already warmed up"

    if not F5_TTS_AVAILABLE:
        return "Voice not available"

    # Set to jenny_british (the voice used by JARVIS) before warmup
    set_voice("jenny_british")

    print("[VOICE] Warming up TTS model with jenny_british voice...")
    # Generate a short phrase to warm up the model
    result = generate_voice("Hello, I am ready to assist you.", output_path="/tmp/warmup_voice.wav", voice="jenny_british")
    if result:
        _warmup_done = True
        print("[VOICE] Warmup complete!")
        return "Warmup complete"
    return "Warmup failed"
