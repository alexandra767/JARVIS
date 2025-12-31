#!/usr/bin/env python3
"""
JARVIS Core - Async Event Loop Coordinator
The brain that coordinates all Jarvis subsystems
"""

import asyncio
import signal
import sys
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("JarvisCore")


class JarvisState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    EXECUTING_TOOL = "executing_tool"


@dataclass
class JarvisEvent:
    """Event for inter-component communication"""
    type: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"


class JarvisCore:
    """
    Central coordinator for all Jarvis subsystems.
    Uses async event-driven architecture for real-time responsiveness.
    """

    def __init__(self):
        self.state = JarvisState.IDLE
        self.running = False

        # Event queues for inter-component communication
        self.event_queue: asyncio.Queue = None
        self.voice_queue: asyncio.Queue = None
        self.vision_queue: asyncio.Queue = None
        self.action_queue: asyncio.Queue = None

        # Subsystem references (set during initialization)
        self.voice_system = None
        self.vision_system = None
        self.hand_tracker = None
        self.browser_agent = None
        self.llm_client = None
        self.tts_engine = None

        # State flags
        self.is_speaking = False
        self.interrupt_requested = False
        self.wake_word_detected = False

        # Callbacks
        self.on_state_change: Optional[Callable] = None
        self.on_response: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

        # Configuration
        self.config = {
            "wake_words": ["hey alexandra", "alexandra", "hey jarvis", "jarvis"],
            "always_listening": True,
            "vision_enabled": True,
            "hand_tracking_enabled": True,
            "browser_enabled": True,
            "interrupt_sensitivity": 0.5,
            "response_speed": "balanced",  # fast, balanced, quality
        }

        # Conversation context
        self.conversation_history = []
        self.current_context = {}

    async def initialize(self):
        """Initialize all async components"""
        logger.info("Initializing Jarvis Core...")

        # Create event queues
        self.event_queue = asyncio.Queue()
        self.voice_queue = asyncio.Queue()
        self.vision_queue = asyncio.Queue()
        self.action_queue = asyncio.Queue()

        logger.info("Jarvis Core initialized")

    async def start(self):
        """Start the main event loop"""
        self.running = True
        await self.initialize()

        logger.info("Starting Jarvis subsystems...")
        self._set_state(JarvisState.LISTENING)

        try:
            # Run all subsystems concurrently
            tasks = [
                asyncio.create_task(self._event_loop(), name="event_loop"),
                asyncio.create_task(self._voice_loop(), name="voice_loop"),
            ]

            # Add optional subsystems
            if self.config["vision_enabled"] and self.vision_system:
                tasks.append(asyncio.create_task(self._vision_loop(), name="vision_loop"))

            if self.config["hand_tracking_enabled"] and self.hand_tracker:
                tasks.append(asyncio.create_task(self._hand_tracking_loop(), name="hand_loop"))

            # Add action executor
            tasks.append(asyncio.create_task(self._action_executor(), name="action_executor"))

            logger.info(f"Running {len(tasks)} concurrent tasks")

            # Wait for all tasks
            await asyncio.gather(*tasks)

        except asyncio.CancelledError:
            logger.info("Jarvis shutting down...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            if self.on_error:
                self.on_error(e)
        finally:
            self.running = False

    async def stop(self):
        """Gracefully stop all subsystems"""
        logger.info("Stopping Jarvis...")
        self.running = False
        self._set_state(JarvisState.IDLE)

    def _set_state(self, new_state: JarvisState):
        """Update state and notify listeners"""
        old_state = self.state
        self.state = new_state
        logger.debug(f"State: {old_state.value} -> {new_state.value}")
        if self.on_state_change:
            self.on_state_change(old_state, new_state)

    async def _event_loop(self):
        """Main event processing loop"""
        logger.info("Event loop started")

        while self.running:
            try:
                # Wait for events with timeout
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(),
                        timeout=0.1
                    )
                    await self._handle_event(event)
                except asyncio.TimeoutError:
                    continue

            except Exception as e:
                logger.error(f"Error in event loop: {e}")

    async def _handle_event(self, event: JarvisEvent):
        """Process incoming events"""
        logger.debug(f"Event: {event.type} from {event.source}")

        if event.type == "voice_input":
            await self._process_voice_input(event.data)

        elif event.type == "interrupt":
            await self._handle_interrupt()

        elif event.type == "vision_result":
            await self._process_vision_result(event.data)

        elif event.type == "gesture_detected":
            await self._handle_gesture(event.data)

        elif event.type == "tool_result":
            await self._process_tool_result(event.data)

    async def _voice_loop(self):
        """Handle voice input/output"""
        logger.info("Voice loop started")

        while self.running:
            try:
                if not self.voice_system:
                    await asyncio.sleep(0.1)
                    continue

                # Check for voice input
                if self.state in [JarvisState.IDLE, JarvisState.LISTENING]:
                    self._set_state(JarvisState.LISTENING)

                    # Get transcribed text from voice system
                    text = await self.voice_system.listen()

                    if text:
                        # Check for wake word if not always listening
                        if not self.config["always_listening"]:
                            if not self._check_wake_word(text):
                                continue
                            text = self._remove_wake_word(text)

                        # Process the input
                        await self.event_queue.put(JarvisEvent(
                            type="voice_input",
                            data=text,
                            source="voice_system"
                        ))

                await asyncio.sleep(0.01)  # Small yield

            except Exception as e:
                logger.error(f"Error in voice loop: {e}")
                await asyncio.sleep(0.5)

    async def _vision_loop(self):
        """Handle camera and vision processing"""
        logger.info("Vision loop started")

        while self.running:
            try:
                if not self.vision_system:
                    await asyncio.sleep(0.1)
                    continue

                # Capture and process frames
                frame = await self.vision_system.capture_frame()

                # Check for vision requests in queue
                try:
                    request = self.vision_queue.get_nowait()
                    result = await self.vision_system.analyze(frame, request["prompt"])
                    await self.event_queue.put(JarvisEvent(
                        type="vision_result",
                        data={"request": request, "result": result},
                        source="vision_system"
                    ))
                except asyncio.QueueEmpty:
                    pass

                await asyncio.sleep(0.033)  # ~30 FPS

            except Exception as e:
                logger.error(f"Error in vision loop: {e}")
                await asyncio.sleep(0.5)

    async def _hand_tracking_loop(self):
        """Handle hand tracking and gestures"""
        logger.info("Hand tracking loop started")

        while self.running:
            try:
                if not self.hand_tracker:
                    await asyncio.sleep(0.1)
                    continue

                # Process hand tracking
                gesture = await self.hand_tracker.detect()

                if gesture:
                    await self.event_queue.put(JarvisEvent(
                        type="gesture_detected",
                        data=gesture,
                        source="hand_tracker"
                    ))

                await asyncio.sleep(0.033)  # ~30 FPS

            except Exception as e:
                logger.error(f"Error in hand tracking loop: {e}")
                await asyncio.sleep(0.5)

    async def _action_executor(self):
        """Execute queued actions/tools"""
        logger.info("Action executor started")

        while self.running:
            try:
                # Wait for actions
                try:
                    action = await asyncio.wait_for(
                        self.action_queue.get(),
                        timeout=0.1
                    )

                    self._set_state(JarvisState.EXECUTING_TOOL)
                    result = await self._execute_action(action)

                    await self.event_queue.put(JarvisEvent(
                        type="tool_result",
                        data={"action": action, "result": result},
                        source="action_executor"
                    ))

                except asyncio.TimeoutError:
                    continue

            except Exception as e:
                logger.error(f"Error in action executor: {e}")

    async def _process_voice_input(self, text: str):
        """Process transcribed voice input"""
        logger.info(f"Processing: {text}")
        self._set_state(JarvisState.THINKING)

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": text,
            "timestamp": datetime.now().isoformat()
        })

        # Check if vision context is needed
        vision_context = None
        if self._needs_vision(text):
            vision_context = await self._get_vision_context(text)

        # Generate response
        response = await self._generate_response(text, vision_context)

        # Speak the response
        await self._speak_response(response)

    async def _generate_response(self, text: str, vision_context: Optional[str] = None) -> str:
        """Generate LLM response with streaming"""
        if not self.llm_client:
            return "I'm sorry, my language model is not connected."

        # Build context
        messages = self._build_messages(text, vision_context)

        # Stream response
        response_text = ""
        self._set_state(JarvisState.THINKING)

        async for chunk in self.llm_client.stream(messages):
            if self.interrupt_requested:
                self.interrupt_requested = False
                break
            response_text += chunk

        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        })

        return response_text

    async def _speak_response(self, text: str):
        """Speak response with streaming TTS"""
        if not self.tts_engine:
            logger.warning("TTS engine not available")
            if self.on_response:
                self.on_response(text)
            return

        self._set_state(JarvisState.SPEAKING)
        self.is_speaking = True

        try:
            # Stream TTS sentence by sentence
            sentences = self._split_sentences(text)

            for sentence in sentences:
                if self.interrupt_requested:
                    logger.info("Speech interrupted")
                    self.interrupt_requested = False
                    break

                await self.tts_engine.speak(sentence)

        finally:
            self.is_speaking = False
            self._set_state(JarvisState.LISTENING)

        if self.on_response:
            self.on_response(text)

    async def _handle_interrupt(self):
        """Handle voice interruption"""
        logger.info("Interrupt requested")
        self.interrupt_requested = True

        if self.tts_engine:
            await self.tts_engine.stop()

        self._set_state(JarvisState.INTERRUPTED)
        await asyncio.sleep(0.1)
        self._set_state(JarvisState.LISTENING)

    async def _process_vision_result(self, data: Dict):
        """Process vision analysis result"""
        # Integrate vision result into current context
        self.current_context["vision"] = data["result"]

    async def _handle_gesture(self, gesture: Dict):
        """Handle detected gesture"""
        logger.info(f"Gesture: {gesture['type']}")

        gesture_actions = {
            "wave": self._handle_interrupt,
            "thumbs_up": lambda: self._confirm_action(),
            "thumbs_down": lambda: self._cancel_action(),
            "pinch": lambda: self._execute_click(),
            "open_palm": lambda: self._handle_interrupt(),
        }

        action = gesture_actions.get(gesture["type"])
        if action:
            await action()

    async def _execute_action(self, action: Dict) -> Any:
        """Execute a tool/action"""
        action_type = action.get("type")

        if action_type == "browser":
            if self.browser_agent:
                return await self.browser_agent.execute(action["task"])

        elif action_type == "smart_home":
            # Use existing Alexandra smart home integration
            pass

        elif action_type == "code_execution":
            # Use existing Alexandra code sandbox
            pass

        return None

    def _check_wake_word(self, text: str) -> bool:
        """Check if text contains wake word"""
        text_lower = text.lower()
        return any(wake in text_lower for wake in self.config["wake_words"])

    def _remove_wake_word(self, text: str) -> str:
        """Remove wake word from text"""
        text_lower = text.lower()
        for wake in self.config["wake_words"]:
            if wake in text_lower:
                idx = text_lower.find(wake)
                text = text[:idx] + text[idx + len(wake):]
        return text.strip()

    def _needs_vision(self, text: str) -> bool:
        """Check if query needs vision context"""
        vision_keywords = [
            "see", "look", "show", "what is this", "what am i",
            "holding", "wearing", "in front", "looking at",
            "this", "that", "here", "there"
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in vision_keywords)

    async def _get_vision_context(self, text: str) -> str:
        """Get vision context for query"""
        if not self.vision_system:
            return None

        # Request vision analysis
        await self.vision_queue.put({"prompt": text})

        # Wait for result with timeout
        try:
            result = await asyncio.wait_for(
                self._wait_for_vision_result(),
                timeout=5.0
            )
            return result
        except asyncio.TimeoutError:
            return None

    async def _wait_for_vision_result(self) -> str:
        """Wait for vision result from event queue"""
        while True:
            # This is simplified - in practice we'd use a specific response queue
            await asyncio.sleep(0.1)
            if "vision" in self.current_context:
                result = self.current_context.pop("vision")
                return result

    def _build_messages(self, text: str, vision_context: Optional[str] = None) -> list:
        """Build messages for LLM"""
        messages = []

        # System prompt
        messages.append({
            "role": "system",
            "content": self._get_system_prompt()
        })

        # Recent conversation history (last 10 exchanges)
        for msg in self.conversation_history[-20:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Current message with vision context if available
        content = text
        if vision_context:
            content = f"[Vision context: {vision_context}]\n\nUser: {text}"

        messages.append({
            "role": "user",
            "content": content
        })

        return messages

    def _get_system_prompt(self) -> str:
        """Get system prompt for Jarvis mode"""
        return """You are Alexandra, an advanced AI assistant similar to JARVIS from Iron Man.

You are helpful, witty, and capable. You can:
- Control smart home devices
- Search the web
- Execute code
- Manage files and projects
- See through the camera when asked
- Help with any task

Keep responses concise for voice - typically 1-3 sentences unless more detail is requested.
Be conversational and natural. You can see what the user is looking at when they ask about visual things.

Current capabilities: Smart home, code execution, web search, vision, memory, file management."""

    def _split_sentences(self, text: str) -> list:
        """Split text into sentences for streaming TTS"""
        import re
        # Split on sentence endings but keep the punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


# Signal handlers for graceful shutdown
def setup_signal_handlers(jarvis: JarvisCore):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(sig, frame):
        print("\nShutdown requested...")
        asyncio.create_task(jarvis.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point"""
    jarvis = JarvisCore()
    setup_signal_handlers(jarvis)

    print("=" * 50)
    print("  JARVIS MODE - Alexandra AI")
    print("=" * 50)
    print("Starting...")

    await jarvis.start()


if __name__ == "__main__":
    asyncio.run(main())
