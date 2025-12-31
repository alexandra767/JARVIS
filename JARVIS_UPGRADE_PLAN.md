# Alexandra → Jarvis Upgrade Plan

## Overview
Transform Alexandra into a full Jarvis-like assistant with real-time voice, vision, hand tracking, and browser control.

---

## Phase 1: Real-Time Voice Loop (Priority: Critical)

### Current State
- Voice input: Whisper (works but not streaming)
- LLM: Qwen 72B via Ollama (works)
- TTS: F5-TTS (works but not streaming)

### What to Build
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Whisper   │ ──► │  Qwen 72B   │ ──► │   F5-TTS    │
│  (Stream)   │     │  (Stream)   │     │  (Stream)   │
└─────────────┘     └─────────────┘     └─────────────┘
       ▲                                       │
       │            INTERRUPTION               │
       └───────────────────────────────────────┘
```

### Files to Create/Modify
1. `jarvis_voice_loop.py` - Main real-time voice handler
2. Modify `voice_input.py` - Add streaming Whisper
3. Modify `simple_f5_infer.py` - Add streaming TTS

### Key Features
- Continuous listening (always on)
- Wake word detection ("Hey Alexandra" - already exists)
- Interruption support (stop speaking when user talks)
- Streaming output (start speaking before full response)

### Code Structure
```python
# jarvis_voice_loop.py
import asyncio
import whisper
import torch
from queue import Queue

class JarvisVoiceLoop:
    def __init__(self):
        self.listening = True
        self.speaking = False
        self.interrupt_flag = False

    async def listen_loop(self):
        """Continuously listen for voice input"""
        while self.listening:
            audio = await self.capture_audio()
            if self.detect_speech(audio):
                if self.speaking:
                    self.interrupt()  # Stop current speech
                text = await self.transcribe(audio)
                await self.process_and_respond(text)

    async def process_and_respond(self, text):
        """Stream LLM response to TTS"""
        self.speaking = True
        async for chunk in self.stream_llm(text):
            if self.interrupt_flag:
                break
            await self.stream_tts(chunk)
        self.speaking = False

    def interrupt(self):
        """Handle interruption"""
        self.interrupt_flag = True
        self.stop_audio_playback()
```

### Estimated Latency
- Whisper transcription: 0.5-1s (with faster-whisper)
- Qwen 72B first token: 0.5-1s
- F5-TTS first audio: 0.5s
- **Total to first sound: ~1.5-2.5s**

---

## Phase 2: Live Camera Vision (Priority: High)

### Current State
- BLIP for image captioning (CPU, slow)
- No live camera feed

### What to Build
- Real-time webcam capture
- Fast vision model (Qwen2-VL or LLaVA)
- Object detection and scene understanding

### Files to Create
1. `jarvis_vision.py` - Camera and vision processing
2. Modify `alexandra_ui.py` - Add camera tab

### Code Structure
```python
# jarvis_vision.py
import cv2
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

class JarvisVision:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    def capture_frame(self):
        """Capture current camera frame"""
        ret, frame = self.camera.read()
        return frame

    def analyze(self, prompt="What do you see?"):
        """Analyze current camera view"""
        frame = self.capture_frame()
        inputs = self.processor(
            text=prompt,
            images=frame,
            return_tensors="pt"
        ).to("cuda")

        output = self.model.generate(**inputs, max_new_tokens=256)
        return self.processor.decode(output[0], skip_special_tokens=True)

    def watch_for_objects(self, target_objects):
        """Continuous monitoring for specific objects"""
        pass
```

### VRAM Requirements
- Qwen2-VL-7B: ~16GB
- You have 122GB, plenty of room alongside Qwen 72B

---

## Phase 3: Hand Tracking & Gesture Control (Priority: Medium)

### What to Build
- MediaPipe hand detection
- Gesture recognition (pinch, wave, point, etc.)
- Mouse cursor control
- Click/drag actions

### Files to Create
1. `jarvis_hands.py` - Hand tracking engine
2. `jarvis_gestures.py` - Gesture definitions

### Code Structure
```python
# jarvis_hands.py
import cv2
import mediapipe as mp
import pyautogui

class JarvisHands:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        self.screen_w, self.screen_h = pyautogui.size()

    def process_frame(self, frame):
        """Detect hands and extract landmarks"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results.multi_hand_landmarks

    def get_gesture(self, landmarks):
        """Recognize gesture from hand landmarks"""
        # Pinch detection (thumb + index close)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = self.calculate_distance(thumb_tip, index_tip)

        if distance < 0.05:
            return "pinch"
        elif self.is_pointing(landmarks):
            return "point"
        elif self.is_open_palm(landmarks):
            return "open"
        return None

    def move_cursor(self, landmarks):
        """Move mouse cursor based on index finger position"""
        index_tip = landmarks[8]
        x = int(index_tip.x * self.screen_w)
        y = int(index_tip.y * self.screen_h)
        pyautogui.moveTo(x, y, duration=0.1)

# Gesture mappings
GESTURES = {
    "pinch": "click",
    "pinch_hold": "drag",
    "two_finger_pinch": "right_click",
    "wave": "dismiss",
    "thumbs_up": "confirm",
    "open_palm": "stop"
}
```

---

## Phase 4: Web Browser Automation (Priority: Medium)

### What to Build
- Playwright browser control
- Screenshot → LLM → Action loop
- Form filling, clicking, navigation

### Files to Create
1. `jarvis_browser.py` - Browser automation agent
2. Integrate with tool calling system

### Code Structure
```python
# jarvis_browser.py
import asyncio
from playwright.async_api import async_playwright
import base64

class JarvisBrowser:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.browser = None
        self.page = None

    async def start(self):
        """Launch browser"""
        pw = await async_playwright().start()
        self.browser = await pw.chromium.launch(headless=False)
        self.page = await self.browser.new_page()

    async def screenshot(self):
        """Capture current page"""
        return await self.page.screenshot(type="png")

    async def execute_task(self, task: str):
        """Execute a web task using LLM guidance"""
        while True:
            # Get screenshot
            screenshot = await self.screenshot()

            # Ask LLM what to do
            action = await self.llm.analyze_and_act(
                screenshot=screenshot,
                task=task,
                available_actions=["click", "type", "scroll", "navigate", "done"]
            )

            if action["type"] == "done":
                return action["result"]
            elif action["type"] == "click":
                await self.page.click(action["selector"])
            elif action["type"] == "type":
                await self.page.fill(action["selector"], action["text"])
            elif action["type"] == "scroll":
                await self.page.evaluate(f"window.scrollBy(0, {action['amount']})")
            elif action["type"] == "navigate":
                await self.page.goto(action["url"])

# Tool definition for Alexandra
BROWSER_TOOL = {
    "name": "web_browser",
    "description": "Control web browser to perform tasks like shopping, research, booking",
    "parameters": {
        "task": "Description of what to do on the web"
    }
}
```

---

## Phase 5: Latency Optimization (Priority: High)

### Optimizations to Implement

1. **Use faster-whisper instead of OpenAI Whisper**
   ```bash
   pip install faster-whisper
   ```
   - 4x faster transcription
   - Same accuracy

2. **Stream LLM output**
   ```python
   # Already supported by Ollama
   async for chunk in ollama.chat(model, messages, stream=True):
       yield chunk['message']['content']
   ```

3. **Sentence-level TTS streaming**
   ```python
   # Generate TTS per sentence, not full response
   for sentence in split_sentences(response):
       audio = generate_tts(sentence)
       play_audio_async(audio)
   ```

4. **Preload models in GPU memory**
   - Keep Whisper, LLM, TTS all loaded
   - No cold start delays

5. **Use speculative decoding** (if supported)
   - Draft model generates candidates
   - Main model verifies

### Target Latency After Optimization
| Component | Before | After |
|-----------|--------|-------|
| Whisper | 1-2s | 0.3-0.5s |
| LLM first token | 1-2s | 0.5-1s |
| TTS first audio | 1s | 0.3s |
| **Total** | **4-9s** | **1.5-2.5s** |

---

## Phase 6: Unified Jarvis Interface (Priority: Medium)

### What to Build
- Full-screen Jarvis mode
- Floating avatar with lip sync
- Voice waveform visualization
- Status indicators
- Hand tracking overlay

### UI Layout
```
┌──────────────────────────────────────────────────────────┐
│                    JARVIS MODE                            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│     ┌─────────────┐           ┌─────────────────────┐   │
│     │             │           │                     │   │
│     │   AVATAR    │           │   CAMERA FEED       │   │
│     │  (Talking)  │           │  + Hand Tracking    │   │
│     │             │           │                     │   │
│     └─────────────┘           └─────────────────────┘   │
│                                                          │
│     ┌──────────────────────────────────────────────┐    │
│     │  ░░░░░░░░░░░░░░░░  Voice Waveform            │    │
│     └──────────────────────────────────────────────┘    │
│                                                          │
│     Status: Listening... | Mode: Normal | Vision: ON    │
│                                                          │
│     [Tools Active: Smart Home, Browser, GitHub]          │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Implementation Order

| Phase | Feature | Time Estimate | Dependencies |
|-------|---------|---------------|--------------|
| 1 | Real-time voice loop | 2-3 days | faster-whisper, streaming TTS |
| 2 | Live camera vision | 1-2 days | Qwen2-VL model download |
| 3 | Hand tracking | 2-3 days | MediaPipe, PyAutoGUI |
| 4 | Browser automation | 2-3 days | Playwright |
| 5 | Latency optimization | 1-2 days | All above working |
| 6 | Unified interface | 2-3 days | All above working |

**Total estimated time: 10-16 days**

---

## Required Installations

```bash
# Voice optimization
pip install faster-whisper

# Vision
pip install qwen-vl-utils transformers accelerate

# Hand tracking
pip install mediapipe pyautogui

# Browser automation
pip install playwright
playwright install chromium

# Audio streaming
pip install sounddevice numpy
```

---

## VRAM Budget

| Component | VRAM |
|-----------|------|
| Qwen 72B (4-bit) | ~40GB |
| Qwen2-VL-7B | ~16GB |
| Whisper Large | ~3GB |
| F5-TTS | ~4GB |
| MediaPipe | ~1GB |
| **Total** | **~64GB** |
| **Available** | **122GB** |
| **Headroom** | **58GB** |

You have plenty of room!

---

## Next Steps

1. Start with Phase 1 (real-time voice) - this is the core
2. Test latency and optimize
3. Add vision capabilities
4. Layer in hand tracking and browser control
5. Build unified interface

Ready to start building?
