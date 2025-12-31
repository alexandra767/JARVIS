#!/usr/bin/env python3
"""
JARVIS Hand Tracking - Gesture Control System
Uses MediaPipe for hand detection and gesture recognition
Uses xdotool for cursor control (works in Docker)
"""

import asyncio
import cv2
import numpy as np
import mediapipe as mp
import subprocess
import os
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import time
import math

logger = logging.getLogger("JarvisHands")

# Use xdotool instead of pyautogui (works in Docker with virtual display)
XDOTOOL_AVAILABLE = False
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

try:
    result = subprocess.run(['xdotool', 'getmouselocation'], capture_output=True, timeout=2)
    if result.returncode == 0:
        XDOTOOL_AVAILABLE = True
        # Get screen size
        result = subprocess.run(['xdotool', 'getdisplaygeometry'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if len(parts) >= 2:
                SCREEN_WIDTH = int(parts[0])
                SCREEN_HEIGHT = int(parts[1])
except Exception:
    pass

def _xdotool(*args):
    """Run xdotool command"""
    try:
        subprocess.run(['xdotool'] + list(args), capture_output=True, timeout=0.5)
    except Exception:
        pass


class Gesture(Enum):
    NONE = "none"
    PINCH = "pinch"
    PINCH_HOLD = "pinch_hold"
    OPEN_PALM = "open_palm"
    CLOSED_FIST = "closed_fist"
    POINTING = "pointing"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PEACE = "peace"
    WAVE = "wave"
    GRAB = "grab"


@dataclass
class HandState:
    """Current state of a detected hand"""
    landmarks: List[Tuple[float, float, float]]
    gesture: Gesture
    is_left: bool
    confidence: float
    palm_center: Tuple[float, float]
    index_tip: Tuple[float, float]
    thumb_tip: Tuple[float, float]


@dataclass
class HandConfig:
    """Configuration for hand tracking"""
    max_hands: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    cursor_sensitivity: float = 1.5
    smoothing_factor: float = 0.3
    pinch_threshold: float = 0.05
    click_hold_time: float = 0.3


class GestureRecognizer:
    """
    Recognizes gestures from hand landmarks.
    """

    def __init__(self, config: HandConfig = None):
        self.config = config or HandConfig()

        # Gesture state tracking
        self.pinch_start_time = None
        self.last_gesture = Gesture.NONE
        self.gesture_history = []

    def recognize(self, landmarks: List) -> Gesture:
        """
        Recognize gesture from MediaPipe hand landmarks.
        Landmarks are 21 points (0-20) representing hand joints.
        """
        if not landmarks:
            return Gesture.NONE

        # Extract key landmark positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        thumb_ip = landmarks[3]
        index_pip = landmarks[6]
        middle_pip = landmarks[10]
        ring_pip = landmarks[14]
        pinky_pip = landmarks[18]

        wrist = landmarks[0]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]

        # Calculate finger states (extended or not)
        fingers_extended = self._get_fingers_extended(landmarks)

        # Check for specific gestures
        gesture = self._classify_gesture(landmarks, fingers_extended)

        # Track gesture for pinch hold detection
        self._track_gesture(gesture)

        return gesture

    def _get_fingers_extended(self, landmarks: List) -> List[bool]:
        """
        Determine which fingers are extended.
        Returns [thumb, index, middle, ring, pinky]
        """
        extended = []

        # Thumb (compare x position since thumb moves horizontally)
        thumb_extended = landmarks[4].x < landmarks[3].x  # Assumes right hand
        extended.append(thumb_extended)

        # Other fingers (compare y position - lower y means extended)
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        for tip, pip in zip(finger_tips, finger_pips):
            extended.append(landmarks[tip].y < landmarks[pip].y)

        return extended

    def _classify_gesture(self, landmarks: List, fingers_extended: List[bool]) -> Gesture:
        """Classify the gesture based on finger states and positions"""

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]

        # Pinch detection (thumb and index close together)
        pinch_distance = self._distance(thumb_tip, index_tip)
        if pinch_distance < self.config.pinch_threshold:
            if self.pinch_start_time:
                if time.time() - self.pinch_start_time > self.config.click_hold_time:
                    return Gesture.PINCH_HOLD
            else:
                self.pinch_start_time = time.time()
            return Gesture.PINCH
        else:
            self.pinch_start_time = None

        # All fingers extended = open palm
        if all(fingers_extended):
            return Gesture.OPEN_PALM

        # No fingers extended = closed fist
        if not any(fingers_extended[1:]):  # Ignore thumb
            return Gesture.CLOSED_FIST

        # Only index extended = pointing
        if fingers_extended[1] and not any(fingers_extended[2:]):
            return Gesture.POINTING

        # Thumb up (thumb extended, others closed)
        if fingers_extended[0] and not any(fingers_extended[1:]):
            # Check thumb is pointing up
            if landmarks[4].y < landmarks[3].y:
                return Gesture.THUMBS_UP
            else:
                return Gesture.THUMBS_DOWN

        # Peace sign (index and middle extended)
        if fingers_extended[1] and fingers_extended[2] and not fingers_extended[3] and not fingers_extended[4]:
            return Gesture.PEACE

        # Grab gesture (fingers curved)
        if self._is_grabbing(landmarks):
            return Gesture.GRAB

        return Gesture.NONE

    def _is_grabbing(self, landmarks: List) -> bool:
        """Detect grabbing gesture (curved fingers)"""
        # Check if fingertips are lower than knuckles but higher than palm
        tips = [landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        mcps = [landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
        pips = [landmarks[6], landmarks[10], landmarks[14], landmarks[18]]

        curved = 0
        for tip, mcp, pip in zip(tips, mcps, pips):
            if pip.y < tip.y < mcp.y:  # Finger is curved
                curved += 1

        return curved >= 3

    def _distance(self, p1, p2) -> float:
        """Calculate 3D distance between two landmarks"""
        return math.sqrt(
            (p1.x - p2.x) ** 2 +
            (p1.y - p2.y) ** 2 +
            (p1.z - p2.z) ** 2
        )

    def _track_gesture(self, gesture: Gesture):
        """Track gesture history for complex gesture detection"""
        self.gesture_history.append((gesture, time.time()))

        # Keep only last 2 seconds of history
        cutoff = time.time() - 2.0
        self.gesture_history = [
            (g, t) for g, t in self.gesture_history if t > cutoff
        ]

        # Detect wave (rapid left-right motion)
        # Could be expanded for more complex gesture sequences

        self.last_gesture = gesture


class CursorController:
    """
    Controls mouse cursor based on hand position.
    """

    def __init__(self, config: HandConfig = None):
        self.config = config or HandConfig()
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT

        # Smoothing state
        self.last_x = self.screen_width // 2
        self.last_y = self.screen_height // 2

        # Click state
        self.is_dragging = False
        self.last_click_time = 0

    def move_cursor(self, hand_x: float, hand_y: float):
        """
        Move cursor based on hand position.
        hand_x, hand_y are normalized [0, 1]
        """
        # Apply sensitivity
        target_x = int(hand_x * self.screen_width * self.config.cursor_sensitivity)
        target_y = int(hand_y * self.screen_height * self.config.cursor_sensitivity)

        # Clamp to screen bounds
        target_x = max(0, min(self.screen_width - 1, target_x))
        target_y = max(0, min(self.screen_height - 1, target_y))

        # Apply smoothing
        smooth_x = int(self.last_x + (target_x - self.last_x) * self.config.smoothing_factor)
        smooth_y = int(self.last_y + (target_y - self.last_y) * self.config.smoothing_factor)

        # Move cursor using xdotool
        _xdotool('mousemove', str(smooth_x), str(smooth_y))

        self.last_x = smooth_x
        self.last_y = smooth_y

    def click(self):
        """Perform mouse click"""
        now = time.time()
        if now - self.last_click_time > 0.3:  # Debounce
            _xdotool('click', '1')
            self.last_click_time = now
            logger.debug("Click")

    def double_click(self):
        """Perform double click"""
        _xdotool('click', '1')
        _xdotool('click', '1')
        logger.debug("Double click")

    def right_click(self):
        """Perform right click"""
        _xdotool('click', '3')
        logger.debug("Right click")

    def start_drag(self):
        """Start drag operation"""
        if not self.is_dragging:
            _xdotool('mousedown', '1')
            self.is_dragging = True
            logger.debug("Drag start")

    def end_drag(self):
        """End drag operation"""
        if self.is_dragging:
            _xdotool('mouseup', '1')
            self.is_dragging = False
            logger.debug("Drag end")

    def scroll(self, amount: int):
        """Scroll by amount using xdotool"""
        if amount > 0:
            for _ in range(abs(amount)):
                _xdotool('click', '4')  # Scroll up
        else:
            for _ in range(abs(amount)):
                _xdotool('click', '5')  # Scroll down


class JarvisHands:
    """
    Complete hand tracking system for Jarvis.
    Integrates MediaPipe detection with gesture recognition and cursor control.
    """

    def __init__(self, config: HandConfig = None):
        self.config = config or HandConfig()

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Components
        self.gesture_recognizer = GestureRecognizer(self.config)
        self.cursor_controller = CursorController(self.config)

        # State
        self.running = False
        self.camera = None
        self.current_hands: List[HandState] = []

        # Callbacks
        self.on_gesture: Optional[Callable[[Gesture, HandState], None]] = None
        self.cursor_enabled = True

    async def start(self, camera=None):
        """Start hand tracking"""
        self.running = True

        if camera:
            self.camera = camera
        else:
            # Create our own camera
            self.camera = cv2.VideoCapture(0)

        logger.info("Hand tracking started")

    async def stop(self):
        """Stop hand tracking"""
        self.running = False
        if isinstance(self.camera, cv2.VideoCapture):
            self.camera.release()
        logger.info("Hand tracking stopped")

    async def detect(self) -> Optional[Dict]:
        """
        Detect hands and gestures in current frame.
        Returns dict with gesture info if detected.
        """
        if not self.running:
            return None

        # Get frame
        if isinstance(self.camera, cv2.VideoCapture):
            ret, frame = self.camera.read()
            if not ret:
                return None
        else:
            frame = await self.camera.get_frame_async()
            if frame is None:
                return None

        # Process frame
        result = await self._process_frame(frame)

        return result

    async def _process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """Process frame for hand detection"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self.hands.process,
            rgb_frame
        )

        if not results.multi_hand_landmarks:
            self.current_hands = []
            return None

        # Process each detected hand
        detected_gestures = []

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get handedness
            handedness = results.multi_handedness[idx]
            is_left = handedness.classification[0].label == "Left"
            confidence = handedness.classification[0].score

            # Recognize gesture
            gesture = self.gesture_recognizer.recognize(hand_landmarks.landmark)

            # Get key positions
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            palm_center = hand_landmarks.landmark[9]  # Middle MCP

            hand_state = HandState(
                landmarks=hand_landmarks.landmark,
                gesture=gesture,
                is_left=is_left,
                confidence=confidence,
                palm_center=(palm_center.x, palm_center.y),
                index_tip=(index_tip.x, index_tip.y),
                thumb_tip=(thumb_tip.x, thumb_tip.y)
            )

            self.current_hands.append(hand_state)
            detected_gestures.append(hand_state)

            # Handle gesture actions
            await self._handle_gesture(hand_state)

            # Move cursor with index finger (right hand only)
            if self.cursor_enabled and not is_left:
                self.cursor_controller.move_cursor(1 - index_tip.x, index_tip.y)

        if detected_gestures:
            return {
                "type": detected_gestures[0].gesture.value,
                "hands": detected_gestures,
                "primary_hand": detected_gestures[0]
            }

        return None

    async def _handle_gesture(self, hand_state: HandState):
        """Handle detected gesture"""
        gesture = hand_state.gesture

        # Callback
        if self.on_gesture:
            self.on_gesture(gesture, hand_state)

        # Cursor control actions
        if self.cursor_enabled:
            if gesture == Gesture.PINCH:
                self.cursor_controller.click()

            elif gesture == Gesture.PINCH_HOLD:
                self.cursor_controller.start_drag()

            elif gesture in [Gesture.OPEN_PALM, Gesture.NONE]:
                self.cursor_controller.end_drag()

            elif gesture == Gesture.PEACE:
                self.cursor_controller.right_click()

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw hand landmarks on frame (for debugging/display)"""
        if not self.current_hands:
            return frame

        annotated = frame.copy()

        for hand_state in self.current_hands:
            # Create landmark list for drawing
            # This is simplified - full implementation would recreate the landmark proto
            pass

        return annotated


class HandGestureActions:
    """
    Maps gestures to Jarvis actions.
    """

    def __init__(self, jarvis_core=None):
        self.jarvis = jarvis_core
        self.actions = {
            Gesture.THUMBS_UP: self._confirm,
            Gesture.THUMBS_DOWN: self._cancel,
            Gesture.OPEN_PALM: self._stop,
            Gesture.WAVE: self._dismiss,
            Gesture.POINTING: self._select,
            Gesture.PEACE: self._screenshot,
        }

    async def handle(self, gesture: Gesture, hand_state: HandState):
        """Handle gesture action"""
        action = self.actions.get(gesture)
        if action:
            await action(hand_state)

    async def _confirm(self, hand_state: HandState):
        """Confirm current action"""
        logger.info("Gesture: Confirm (thumbs up)")
        if self.jarvis:
            await self.jarvis.event_queue.put({
                "type": "gesture_confirm",
                "data": hand_state
            })

    async def _cancel(self, hand_state: HandState):
        """Cancel current action"""
        logger.info("Gesture: Cancel (thumbs down)")
        if self.jarvis:
            await self.jarvis.event_queue.put({
                "type": "gesture_cancel",
                "data": hand_state
            })

    async def _stop(self, hand_state: HandState):
        """Stop/interrupt current action"""
        logger.info("Gesture: Stop (open palm)")
        if self.jarvis:
            await self.jarvis.event_queue.put({
                "type": "interrupt",
                "data": hand_state
            })

    async def _dismiss(self, hand_state: HandState):
        """Dismiss notification/dialog"""
        logger.info("Gesture: Dismiss (wave)")

    async def _select(self, hand_state: HandState):
        """Select pointed item"""
        logger.info("Gesture: Select (pointing)")

    async def _screenshot(self, hand_state: HandState):
        """Take screenshot using scrot or import"""
        logger.info("Gesture: Screenshot (peace)")
        try:
            subprocess.run(['scrot', '/tmp/jarvis_screenshot.png'], capture_output=True, timeout=5)
        except Exception:
            try:
                subprocess.run(['import', '-window', 'root', '/tmp/jarvis_screenshot.png'], capture_output=True, timeout=5)
            except Exception as e:
                logger.warning(f"Screenshot failed: {e}")


# Test function
async def test_hand_tracking():
    """Test hand tracking"""
    print("Testing hand tracking...")
    print("Controls:")
    print("  - Move index finger to control cursor")
    print("  - Pinch (thumb + index) to click")
    print("  - Open palm to stop")
    print("  - Thumbs up to confirm")
    print("Press Ctrl+C to exit")

    hands = JarvisHands()
    await hands.start()

    try:
        while True:
            result = await hands.detect()
            if result:
                print(f"Gesture: {result['type']}")
            await asyncio.sleep(0.033)  # ~30 FPS

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await hands.stop()


if __name__ == "__main__":
    asyncio.run(test_hand_tracking())
