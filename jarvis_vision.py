#!/usr/bin/env python3
"""
JARVIS Vision System - Real-time Camera and Image Analysis
Uses Qwen2-VL or LLaVA for visual understanding
"""

import asyncio
import cv2
import numpy as np
import base64
import io
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import threading
import queue

logger = logging.getLogger("JarvisVision")


@dataclass
class VisionConfig:
    camera_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    model_name: str = "qwen2-vl"  # or "llava", "blip"


class CameraCapture:
    """
    Async camera capture with frame buffering.
    Runs capture in separate thread for non-blocking operation.
    """

    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.capture_thread = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()

    def start(self):
        """Start camera capture"""
        if self.running:
            return

        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            raise RuntimeError("Camera not available")

        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info(f"Camera {self.camera_id} started")

    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Camera stopped")

    def _capture_loop(self):
        """Continuous capture loop in separate thread"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame

                # Also put in queue for consumers
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Remove old frame and add new
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except:
                        pass
            else:
                time.sleep(0.01)

    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame (non-blocking)"""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

    async def get_frame_async(self) -> Optional[np.ndarray]:
        """Get frame asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_frame)


class VisionModel:
    """
    Vision model wrapper supporting multiple backends.
    Optimized for real-time analysis.
    """

    def __init__(self, model_name: str = "qwen2-vl", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self):
        """Load vision model"""
        if self._loaded:
            return

        logger.info(f"Loading vision model: {self.model_name}")

        try:
            if "qwen" in self.model_name.lower():
                self._load_qwen_vl()
            elif "llava" in self.model_name.lower():
                self._load_llava()
            elif "blip" in self.model_name.lower():
                self._load_blip()
            else:
                # Default to BLIP (lighter weight)
                self._load_blip()

            self._loaded = True
            logger.info(f"Vision model loaded: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            # Fallback to Ollama with vision
            self._setup_ollama_vision()

    def _load_qwen_vl(self):
        """Load Qwen2-VL model"""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch

        model_id = "Qwen/Qwen2-VL-7B-Instruct"

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def _load_llava(self):
        """Load LLaVA model"""
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        import torch

        model_id = "llava-hf/llava-1.5-7b-hf"

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def _load_blip(self):
        """Load BLIP model (lighter weight option)"""
        from transformers import BlipProcessor, BlipForConditionalGeneration

        model_id = "Salesforce/blip-image-captioning-large"

        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto"
        )

    def _setup_ollama_vision(self):
        """Setup Ollama with LLaVA as fallback"""
        logger.info("Setting up Ollama vision fallback")
        self.model_name = "ollama_llava"
        self._loaded = True

    async def analyze(self, image: np.ndarray, prompt: str = "Describe what you see.") -> str:
        """Analyze image with prompt"""
        if not self._loaded:
            self.load()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._analyze_sync,
                image,
                prompt
            )
            return result
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return f"I had trouble analyzing the image: {str(e)}"

    def _analyze_sync(self, image: np.ndarray, prompt: str) -> str:
        """Synchronous image analysis"""
        try:
            if self.model_name == "ollama_llava":
                return self._analyze_ollama(image, prompt)

            if "qwen" in self.model_name.lower():
                return self._analyze_qwen(image, prompt)
            elif "llava" in self.model_name.lower():
                return self._analyze_llava(image, prompt)
            else:
                return self._analyze_blip(image, prompt)

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return f"Error: {str(e)}"

    def _analyze_qwen(self, image: np.ndarray, prompt: str) -> str:
        """Analyze with Qwen2-VL"""
        from PIL import Image
        import torch

        # Convert numpy to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Build conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        # Decode
        output = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        return output.strip()

    def _analyze_llava(self, image: np.ndarray, prompt: str) -> str:
        """Analyze with LLaVA"""
        from PIL import Image
        import torch

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        inputs = self.processor(
            text=full_prompt,
            images=pil_image,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        decoded = self.processor.decode(output[0], skip_special_tokens=True)

        # Extract assistant response
        if "ASSISTANT:" in decoded:
            return decoded.split("ASSISTANT:")[-1].strip()
        return decoded.strip()

    def _analyze_blip(self, image: np.ndarray, prompt: str) -> str:
        """Analyze with BLIP"""
        from PIL import Image
        import torch

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # BLIP works best with simple prompts or captioning
        if "describe" in prompt.lower() or "what" in prompt.lower():
            # Captioning mode
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=100)
            return self.processor.decode(output[0], skip_special_tokens=True)
        else:
            # VQA mode
            inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=100)
            return self.processor.decode(output[0], skip_special_tokens=True)

    def _analyze_ollama(self, image: np.ndarray, prompt: str) -> str:
        """Analyze with Ollama LLaVA"""
        import requests
        from PIL import Image
        import io

        # Convert to base64
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Call Ollama
        response = requests.post(
            "http://192.168.50.129:11434/api/generate",
            json={
                "model": "llava",
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            },
            timeout=30
        )

        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Ollama error: {response.status_code}"


class JarvisVision:
    """
    Complete vision system for Jarvis.
    Combines camera capture with vision model analysis.
    """

    def __init__(self, config: VisionConfig = None):
        self.config = config or VisionConfig()
        self.camera = CameraCapture(
            camera_id=self.config.camera_id,
            width=self.config.width,
            height=self.config.height
        )
        self.model = VisionModel(
            model_name=self.config.model_name,
            device="cuda"
        )
        self.running = False

    async def start(self):
        """Start vision system"""
        self.camera.start()
        self.model.load()
        self.running = True
        logger.info("Vision system started")

    async def stop(self):
        """Stop vision system"""
        self.running = False
        self.camera.stop()
        logger.info("Vision system stopped")

    async def capture_frame(self) -> Optional[np.ndarray]:
        """Capture current frame"""
        return await self.camera.get_frame_async()

    async def analyze(self, frame: Optional[np.ndarray] = None, prompt: str = "What do you see?") -> str:
        """
        Analyze current view or provided frame.
        """
        if frame is None:
            frame = await self.capture_frame()

        if frame is None:
            return "I can't see anything - the camera might not be available."

        return await self.model.analyze(frame, prompt)

    async def describe_scene(self) -> str:
        """Get general description of current scene"""
        return await self.analyze(prompt="Describe what you see in detail.")

    async def identify_object(self) -> str:
        """Identify main object in view"""
        return await self.analyze(prompt="What is the main object in this image?")

    async def answer_question(self, question: str) -> str:
        """Answer a question about the current view"""
        return await self.analyze(prompt=question)

    async def detect_person(self) -> Dict[str, Any]:
        """Detect if a person is in frame"""
        frame = await self.capture_frame()
        if frame is None:
            return {"detected": False, "error": "No camera feed"}

        # Use OpenCV for quick person detection
        # (Faster than running through vision model for simple detection)
        try:
            # HOG person detector
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

            return {
                "detected": len(boxes) > 0,
                "count": len(boxes),
                "confidence": float(max(weights)) if len(weights) > 0 else 0
            }
        except Exception as e:
            logger.error(f"Person detection error: {e}")
            return {"detected": False, "error": str(e)}

    def get_frame_base64(self, frame: np.ndarray = None) -> Optional[str]:
        """Get frame as base64 for sending to APIs"""
        if frame is None:
            frame = self.camera.get_frame()
        if frame is None:
            return None

        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')


# Test functions
async def test_camera():
    """Test camera capture"""
    print("Testing camera...")
    camera = CameraCapture()
    camera.start()

    for i in range(5):
        frame = await camera.get_frame_async()
        if frame is not None:
            print(f"Frame {i}: {frame.shape}")
            cv2.imwrite(f"/tmp/test_frame_{i}.jpg", frame)
        await asyncio.sleep(1)

    camera.stop()
    print("Camera test complete")


async def test_vision():
    """Test vision analysis"""
    print("Testing vision system...")
    vision = JarvisVision(VisionConfig(model_name="blip"))
    await vision.start()

    print("Capturing and analyzing...")
    result = await vision.describe_scene()
    print(f"Description: {result}")

    await vision.stop()
    print("Vision test complete")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "camera":
            asyncio.run(test_camera())
        elif sys.argv[1] == "vision":
            asyncio.run(test_vision())
    else:
        asyncio.run(test_vision())
