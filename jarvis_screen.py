#!/usr/bin/env python3
"""
JARVIS Screen Understanding
- Screen capture
- OCR text extraction
- Window detection
- Screen analysis with vision model
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger("JarvisScreen")

# ============================================================================
# SCREEN CAPTURE
# ============================================================================

class ScreenCapture:
    """Capture screen contents"""

    def __init__(self):
        self.last_capture = None
        self.last_capture_time = None

    def capture_screen(self, monitor: int = 0) -> Optional[np.ndarray]:
        """Capture the entire screen"""
        try:
            # Try PIL ImageGrab first (works on some systems)
            from PIL import ImageGrab
            screenshot = ImageGrab.grab()
            img_array = np.array(screenshot)
            self.last_capture = img_array
            self.last_capture_time = datetime.now()
            return img_array
        except Exception:
            pass

        try:
            # Try mss (multi-platform screen capture)
            import mss
            with mss.mss() as sct:
                monitor_info = sct.monitors[monitor + 1]  # 0 is all monitors
                screenshot = sct.grab(monitor_info)
                img_array = np.array(screenshot)
                # Convert BGRA to RGB
                img_array = img_array[:, :, :3][:, :, ::-1]
                self.last_capture = img_array
                self.last_capture_time = datetime.now()
                return img_array
        except ImportError:
            pass

        try:
            # Fallback to scrot on Linux
            import tempfile
            from PIL import Image

            temp_path = tempfile.mktemp(suffix=".png")
            subprocess.run(["scrot", temp_path], check=True, capture_output=True)

            img = Image.open(temp_path)
            img_array = np.array(img)
            os.unlink(temp_path)

            self.last_capture = img_array
            self.last_capture_time = datetime.now()
            return img_array
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")

        return None

    def capture_region(self, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
        """Capture a specific region of the screen"""
        try:
            from PIL import ImageGrab
            screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
            return np.array(screenshot)
        except:
            # Capture full screen and crop
            full = self.capture_screen()
            if full is not None:
                return full[y:y+height, x:x+width]
        return None

    def capture_active_window(self) -> Optional[Tuple[np.ndarray, str]]:
        """Capture the active window and return (image, window_title)"""
        try:
            # Get active window info on Linux
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=5
            )
            window_title = result.stdout.strip() if result.returncode == 0 else "Unknown"

            # Get window geometry
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowgeometry", "--shell"],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                # Parse geometry
                geo = {}
                for line in result.stdout.strip().split('\n'):
                    if '=' in line:
                        key, val = line.split('=')
                        geo[key] = int(val)

                x, y = geo.get('X', 0), geo.get('Y', 0)
                w, h = geo.get('WIDTH', 800), geo.get('HEIGHT', 600)

                region = self.capture_region(x, y, w, h)
                return region, window_title

        except Exception as e:
            logger.debug(f"Active window capture failed: {e}")

        # Fallback to full screen
        return self.capture_screen(), "Full Screen"

    def save_screenshot(self, path: str = None) -> str:
        """Save screenshot to file"""
        if path is None:
            path = f"/tmp/jarvis_screen_{int(time.time())}.png"

        img = self.capture_screen()
        if img is not None:
            from PIL import Image
            Image.fromarray(img).save(path)
            return path

        return ""


# ============================================================================
# OCR TEXT EXTRACTION
# ============================================================================

class ScreenOCR:
    """Extract text from screen using OCR"""

    def __init__(self):
        self.ocr_engine = None
        self._init_ocr()

    def _init_ocr(self):
        """Initialize OCR engine"""
        try:
            import pytesseract
            self.ocr_engine = "tesseract"
            logger.info("OCR initialized with Tesseract")
        except ImportError:
            try:
                import easyocr
                self.reader = easyocr.Reader(['en'])
                self.ocr_engine = "easyocr"
                logger.info("OCR initialized with EasyOCR")
            except ImportError:
                logger.warning("No OCR engine available. Install pytesseract or easyocr")

    def extract_text(self, image: np.ndarray) -> str:
        """Extract text from image"""
        if image is None:
            return ""

        if self.ocr_engine == "tesseract":
            try:
                import pytesseract
                from PIL import Image
                pil_img = Image.fromarray(image)
                text = pytesseract.image_to_string(pil_img)
                return text.strip()
            except Exception as e:
                logger.error(f"Tesseract OCR failed: {e}")

        elif self.ocr_engine == "easyocr":
            try:
                results = self.reader.readtext(image)
                text = " ".join([r[1] for r in results])
                return text.strip()
            except Exception as e:
                logger.error(f"EasyOCR failed: {e}")

        return ""

    def extract_text_with_positions(self, image: np.ndarray) -> List[Dict]:
        """Extract text with bounding boxes"""
        if image is None:
            return []

        if self.ocr_engine == "tesseract":
            try:
                import pytesseract
                from PIL import Image
                pil_img = Image.fromarray(image)
                data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)

                results = []
                for i in range(len(data['text'])):
                    if data['text'][i].strip():
                        results.append({
                            'text': data['text'][i],
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i],
                            'confidence': data['conf'][i]
                        })
                return results
            except Exception as e:
                logger.error(f"Tesseract OCR failed: {e}")

        elif self.ocr_engine == "easyocr":
            try:
                results = self.reader.readtext(image)
                return [
                    {
                        'text': r[1],
                        'bbox': r[0],
                        'confidence': r[2]
                    }
                    for r in results
                ]
            except Exception as e:
                logger.error(f"EasyOCR failed: {e}")

        return []


# ============================================================================
# SCREEN ANALYZER
# ============================================================================

class ScreenAnalyzer:
    """Analyze screen content with vision model"""

    def __init__(self):
        self.capture = ScreenCapture()
        self.ocr = ScreenOCR()
        self.vision_analyze = None  # Set this to vision model function

    def analyze_screen(self, prompt: str = None) -> Dict[str, Any]:
        """Capture and analyze current screen"""
        # Capture screen
        image = self.capture.capture_screen()

        if image is None:
            return {"error": "Could not capture screen"}

        result = {
            "timestamp": datetime.now().isoformat(),
            "size": image.shape[:2],
            "text": "",
            "analysis": ""
        }

        # Extract text with OCR
        if self.ocr.ocr_engine:
            result["text"] = self.ocr.extract_text(image)

        # Analyze with vision model if available
        if self.vision_analyze:
            default_prompt = prompt or "Describe what's on this screen. What application is open? What is the user doing?"
            result["analysis"] = self.vision_analyze(image, default_prompt)
        else:
            # Provide basic info without vision model
            result["analysis"] = f"Screen captured ({image.shape[1]}x{image.shape[0]}). "
            if result["text"]:
                result["analysis"] += f"Text detected on screen."

        return result

    def analyze_active_window(self, prompt: str = None) -> Dict[str, Any]:
        """Analyze the active window"""
        image, title = self.capture.capture_active_window()

        if image is None:
            return {"error": "Could not capture active window"}

        result = {
            "timestamp": datetime.now().isoformat(),
            "window_title": title,
            "size": image.shape[:2] if len(image.shape) >= 2 else (0, 0),
            "text": "",
            "analysis": ""
        }

        # Extract text
        if self.ocr.ocr_engine:
            result["text"] = self.ocr.extract_text(image)

        # Vision analysis
        if self.vision_analyze:
            default_prompt = prompt or f"This is a screenshot of '{title}'. Describe what you see and what the user might be working on."
            result["analysis"] = self.vision_analyze(image, default_prompt)
        else:
            result["analysis"] = f"Active window: {title}"

        return result

    def find_on_screen(self, target: str) -> List[Dict]:
        """Find specific text/element on screen"""
        image = self.capture.capture_screen()
        if image is None:
            return []

        # Use OCR to find text
        text_items = self.ocr.extract_text_with_positions(image)

        matches = []
        target_lower = target.lower()

        for item in text_items:
            if target_lower in item.get('text', '').lower():
                matches.append(item)

        return matches

    def describe_for_voice(self, prompt: str = None) -> str:
        """Get voice-friendly description of screen"""
        result = self.analyze_screen(prompt)

        if "error" in result:
            return f"Sorry, I couldn't capture the screen: {result['error']}"

        if result.get("analysis"):
            return result["analysis"]

        if result.get("text"):
            text_preview = result["text"][:200]
            return f"I can see text on your screen: {text_preview}"

        return "I captured your screen but couldn't analyze its contents."

    def help_with_screen(self, question: str) -> str:
        """Answer a question about what's on screen"""
        result = self.analyze_screen(question)

        if "error" in result:
            return f"I couldn't see your screen: {result['error']}"

        return result.get("analysis", "I'm not sure what I'm looking at.")


# ============================================================================
# WINDOW MANAGER
# ============================================================================

class WindowManager:
    """Manage and interact with windows"""

    def get_open_windows(self) -> List[Dict]:
        """Get list of open windows"""
        try:
            result = subprocess.run(
                ["wmctrl", "-l"],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode != 0:
                return []

            windows = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split(None, 3)
                if len(parts) >= 4:
                    windows.append({
                        "id": parts[0],
                        "desktop": parts[1],
                        "hostname": parts[2],
                        "title": parts[3]
                    })

            return windows
        except Exception as e:
            logger.debug(f"Could not get windows: {e}")
            return []

    def get_active_window(self) -> Optional[Dict]:
        """Get active window info"""
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                return {"title": result.stdout.strip()}
        except:
            pass

        return None

    def focus_window(self, title_contains: str) -> bool:
        """Focus a window by title"""
        try:
            result = subprocess.run(
                ["wmctrl", "-a", title_contains],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except:
            return False

    def close_window(self, title_contains: str) -> bool:
        """Close a window by title"""
        try:
            result = subprocess.run(
                ["wmctrl", "-c", title_contains],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except:
            return False

    def list_windows_summary(self) -> str:
        """Get summary of open windows for voice"""
        windows = self.get_open_windows()

        if not windows:
            return "I couldn't detect any open windows."

        unique_titles = list(set(w["title"] for w in windows))

        if len(unique_titles) == 1:
            return f"You have one window open: {unique_titles[0]}"

        return f"You have {len(unique_titles)} windows open: " + ", ".join(unique_titles[:5])


# ============================================================================
# UNIFIED SCREEN SYSTEM
# ============================================================================

class JarvisScreen:
    """
    Unified screen understanding system.
    """

    def __init__(self):
        self.capture = ScreenCapture()
        self.ocr = ScreenOCR()
        self.analyzer = ScreenAnalyzer()
        self.windows = WindowManager()

    def set_vision_model(self, analyze_func):
        """Set the vision model analysis function"""
        self.analyzer.vision_analyze = analyze_func

    def what_am_i_looking_at(self) -> str:
        """Describe what's currently on screen"""
        return self.analyzer.describe_for_voice()

    def help_with_this(self, question: str = None) -> str:
        """Help with current screen content"""
        if question:
            return self.analyzer.help_with_screen(question)
        return self.analyzer.describe_for_voice("What is the user looking at and how can I help them?")

    def read_screen(self) -> str:
        """Read text from screen"""
        image = self.capture.capture_screen()
        if image is None:
            return "Could not capture screen"

        text = self.ocr.extract_text(image)
        if text:
            return text[:1000]  # Limit for voice
        return "No readable text found on screen"

    def take_screenshot(self) -> str:
        """Take and save a screenshot"""
        path = self.capture.save_screenshot()
        if path:
            return f"Screenshot saved to {path}"
        return "Failed to take screenshot"

    def whats_open(self) -> str:
        """List open windows"""
        return self.windows.list_windows_summary()

    def focus(self, window_name: str) -> str:
        """Focus a specific window"""
        if self.windows.focus_window(window_name):
            return f"Focused {window_name}"
        return f"Could not find window: {window_name}"


# Create global instance
screen = JarvisScreen()


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing Screen Understanding...")

    # Test screen capture
    print("\n=== Screen Capture ===")
    img = screen.capture.capture_screen()
    if img is not None:
        print(f"Captured: {img.shape}")
    else:
        print("Screen capture not available (may need display)")

    # Test window listing
    print("\n=== Open Windows ===")
    print(screen.whats_open())

    # Test OCR
    print("\n=== OCR Status ===")
    print(f"OCR Engine: {screen.ocr.ocr_engine or 'None'}")

    if img is not None and screen.ocr.ocr_engine:
        print("\n=== Screen Text (first 200 chars) ===")
        text = screen.ocr.extract_text(img)
        print(text[:200] if text else "No text found")
