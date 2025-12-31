#!/usr/bin/env python3
"""
JARVIS MODE - Unified Real-Time AI Assistant
Combines voice, vision, hands, and browser into one seamless experience
Integrates with existing Alexandra AI infrastructure
"""

import asyncio
import os
import sys
import signal
from typing import Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("JarvisMode")

# Import Jarvis subsystems
from jarvis_core import JarvisCore, JarvisState, JarvisEvent
from jarvis_voice import JarvisVoiceSystem
from jarvis_vision import JarvisVision, VisionConfig
from jarvis_hands import JarvisHands, HandConfig, Gesture
from jarvis_browser import JarvisBrowser, BrowserConfig

# Import existing Alexandra components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from alexandra_router import AlexandraRouter
    from alexandra_features import AlexandraFeatures
    from enhanced_memory import EnhancedMemory
    ALEXANDRA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Alexandra components not fully available: {e}")
    ALEXANDRA_AVAILABLE = False


@dataclass
class JarvisConfig:
    """Configuration for Jarvis mode"""
    # Voice settings
    whisper_model: str = "base"  # tiny, base, small, medium, large
    voice_device: str = "cuda"

    # Vision settings
    vision_enabled: bool = True
    vision_model: str = "blip"  # blip, llava, qwen2-vl
    camera_id: int = 0

    # Hand tracking
    hand_tracking_enabled: bool = True
    cursor_control: bool = True

    # Browser
    browser_enabled: bool = True
    browser_headless: bool = False

    # LLM settings (Ollama)
    ollama_url: str = "http://192.168.50.129:11434"
    llm_model: str = "qwen2.5:72b"  # Your local model

    # Behavior
    always_listening: bool = True
    wake_words: tuple = ("hey alexandra", "alexandra", "hey jarvis", "jarvis")
    response_style: str = "concise"  # concise, detailed, conversational


class OllamaLLMClient:
    """
    Async client for Ollama LLM with streaming support.
    Integrates with existing Alexandra router for domain-specific responses.
    """

    def __init__(self, config: JarvisConfig):
        self.config = config
        self.router = None

        # Try to use Alexandra router
        if ALEXANDRA_AVAILABLE:
            try:
                self.router = AlexandraRouter()
                logger.info("Using Alexandra router for LLM")
            except Exception as e:
                logger.warning(f"Could not initialize Alexandra router: {e}")

    async def stream(self, messages: list) -> AsyncIterator[str]:
        """Stream LLM response"""
        import aiohttp

        # Build prompt from messages
        prompt = self._build_prompt(messages)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.ollama_url}/api/generate",
                    json={
                        "model": self.config.llm_model,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 500,  # Keep responses concise for voice
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    async for line in response.content:
                        if line:
                            try:
                                import json
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield f"I'm having trouble connecting to my brain. Error: {str(e)}"

    async def generate(self, messages: list) -> str:
        """Generate complete response (non-streaming)"""
        response = ""
        async for chunk in self.stream(messages):
            response += chunk
        return response

    def _build_prompt(self, messages: list) -> str:
        """Build prompt from messages"""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"Human: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        prompt += "Assistant:"
        return prompt


class JarvisMode:
    """
    Complete Jarvis-like AI assistant.
    Coordinates all subsystems for seamless interaction.
    """

    def __init__(self, config: JarvisConfig = None):
        self.config = config or JarvisConfig()

        # Core
        self.core = JarvisCore()

        # Subsystems
        self.voice: Optional[JarvisVoiceSystem] = None
        self.vision: Optional[JarvisVision] = None
        self.hands: Optional[JarvisHands] = None
        self.browser: Optional[JarvisBrowser] = None
        self.llm: Optional[OllamaLLMClient] = None

        # Alexandra integration
        self.features: Optional[AlexandraFeatures] = None
        self.memory: Optional[EnhancedMemory] = None

        # State
        self.running = False
        self.paused = False

        # Callbacks for UI
        self.on_listening = None
        self.on_thinking = None
        self.on_speaking = None
        self.on_response = None
        self.on_state_change = None

    async def initialize(self):
        """Initialize all subsystems"""
        logger.info("=" * 60)
        logger.info("  JARVIS MODE - Initializing")
        logger.info("=" * 60)

        # Initialize LLM client
        logger.info("Initializing LLM client...")
        self.llm = OllamaLLMClient(self.config)

        # Initialize voice system
        logger.info("Initializing voice system...")
        self.voice = JarvisVoiceSystem(
            whisper_model=self.config.whisper_model,
            device=self.config.voice_device
        )

        # Initialize vision (optional)
        if self.config.vision_enabled:
            logger.info("Initializing vision system...")
            self.vision = JarvisVision(VisionConfig(
                camera_id=self.config.camera_id,
                model_name=self.config.vision_model
            ))

        # Initialize hand tracking (optional)
        if self.config.hand_tracking_enabled:
            logger.info("Initializing hand tracking...")
            self.hands = JarvisHands(HandConfig(
                cursor_sensitivity=1.5
            ))
            self.hands.cursor_enabled = self.config.cursor_control

        # Initialize browser (optional)
        if self.config.browser_enabled:
            logger.info("Initializing browser agent...")
            self.browser = JarvisBrowser(BrowserConfig(
                headless=self.config.browser_headless,
                ollama_url=self.config.ollama_url
            ))

        # Initialize Alexandra features
        if ALEXANDRA_AVAILABLE:
            logger.info("Initializing Alexandra features...")
            try:
                self.features = AlexandraFeatures()
                self.memory = EnhancedMemory()
            except Exception as e:
                logger.warning(f"Could not initialize Alexandra features: {e}")

        # Link subsystems to core
        self.core.voice_system = self.voice
        self.core.vision_system = self.vision
        self.core.hand_tracker = self.hands
        self.core.browser_agent = self.browser
        self.core.llm_client = self.llm
        self.core.tts_engine = self.voice.output if self.voice else None

        # Set core callbacks
        self.core.on_state_change = self._on_state_change
        self.core.on_response = self._on_response

        logger.info("Initialization complete!")

    async def start(self):
        """Start Jarvis mode"""
        self.running = True

        await self.initialize()

        logger.info("=" * 60)
        logger.info("  JARVIS MODE - Active")
        logger.info("=" * 60)
        logger.info("Say 'Hey Alexandra' or just start talking...")
        logger.info("Press Ctrl+C to exit")
        logger.info("")

        try:
            # Start subsystems
            await self.voice.start()

            if self.vision:
                await self.vision.start()

            if self.hands:
                await self.hands.start()

            # Main loop
            await self._main_loop()

        except asyncio.CancelledError:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            raise
        finally:
            await self.stop()

    async def stop(self):
        """Stop all subsystems"""
        self.running = False

        if self.voice:
            await self.voice.stop()

        if self.vision:
            await self.vision.stop()

        if self.hands:
            await self.hands.stop()

        if self.browser:
            await self.browser.stop()

        logger.info("Jarvis mode stopped")

    async def _main_loop(self):
        """Main interaction loop"""
        while self.running:
            try:
                if self.paused:
                    await asyncio.sleep(0.1)
                    continue

                # Listen for voice input
                if self.on_listening:
                    self.on_listening()

                text = await self.voice.listen()

                if text:
                    # Check wake word if not always listening
                    if not self.config.always_listening:
                        if not self._check_wake_word(text):
                            continue
                        text = self._remove_wake_word(text)

                    if text.strip():
                        # Process the input
                        await self._process_input(text)

                # Also process hand gestures
                if self.hands:
                    gesture_result = await self.hands.detect()
                    if gesture_result:
                        await self._handle_gesture(gesture_result)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(0.5)

    async def _process_input(self, text: str):
        """Process voice input and generate response"""
        logger.info(f"Processing: {text}")

        if self.on_thinking:
            self.on_thinking()

        # Check if vision context is needed
        vision_context = None
        if self._needs_vision(text) and self.vision:
            logger.info("Getting vision context...")
            vision_context = await self.vision.analyze(
                prompt=f"The user asked: {text}. Describe what you see that's relevant."
            )

        # Check for tool use
        tool_result = await self._check_tools(text)

        # Build messages
        messages = self._build_messages(text, vision_context, tool_result)

        # Generate and speak response
        if self.on_speaking:
            self.on_speaking()

        response_text = ""

        # Stream response to TTS
        async def response_stream():
            nonlocal response_text
            async for chunk in self.llm.stream(messages):
                response_text += chunk
                yield chunk

        await self.voice.speak_streaming(response_stream())

        # Store in memory
        if self.memory:
            try:
                self.memory.add_conversation(
                    user_message=text,
                    assistant_message=response_text
                )
            except Exception as e:
                logger.warning(f"Could not save to memory: {e}")

        if self.on_response:
            self.on_response(response_text)

    async def _check_tools(self, text: str) -> Optional[str]:
        """Check if any tools should be used"""
        text_lower = text.lower()

        # Smart home
        if any(kw in text_lower for kw in ["turn on", "turn off", "lights", "thermostat"]):
            if self.features:
                try:
                    result = await self._execute_smart_home(text)
                    return f"Smart home result: {result}"
                except Exception as e:
                    logger.warning(f"Smart home error: {e}")

        # Web browsing
        if any(kw in text_lower for kw in ["go to", "browse", "search for", "add to cart", "buy"]):
            if self.browser:
                try:
                    result = await self.browser.execute(text)
                    return f"Browser result: {result.get('result', 'completed')}"
                except Exception as e:
                    logger.warning(f"Browser error: {e}")

        # Code execution
        if "run code" in text_lower or "execute" in text_lower:
            if self.features:
                try:
                    result = self.features.execute_code_safely(text)
                    return f"Code result: {result}"
                except Exception as e:
                    logger.warning(f"Code execution error: {e}")

        return None

    async def _execute_smart_home(self, text: str) -> str:
        """Execute smart home command"""
        if not self.features:
            return "Smart home not available"

        # Use existing Alexandra smart home
        result = self.features.parse_smart_home_command(text)
        return str(result)

    async def _handle_gesture(self, gesture_result: Dict):
        """Handle detected gesture"""
        gesture_type = gesture_result.get("type", "none")

        if gesture_type == "open_palm":
            # Stop current speech
            if self.voice:
                await self.voice.output.stop()

        elif gesture_type == "thumbs_up":
            # Confirm action
            logger.info("Gesture: Thumbs up - Confirming")

        elif gesture_type == "thumbs_down":
            # Cancel action
            logger.info("Gesture: Thumbs down - Canceling")

    def _check_wake_word(self, text: str) -> bool:
        """Check if text contains wake word"""
        text_lower = text.lower()
        return any(wake in text_lower for wake in self.config.wake_words)

    def _remove_wake_word(self, text: str) -> str:
        """Remove wake word from text"""
        text_lower = text.lower()
        for wake in self.config.wake_words:
            if wake in text_lower:
                idx = text_lower.find(wake)
                text = text[:idx] + text[idx + len(wake):]
        return text.strip()

    def _needs_vision(self, text: str) -> bool:
        """Check if query needs vision"""
        vision_keywords = [
            "see", "look", "show", "what is this", "what am i",
            "holding", "wearing", "in front", "looking at",
            "this", "that", "here", "can you see"
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in vision_keywords)

    def _build_messages(
        self,
        text: str,
        vision_context: Optional[str] = None,
        tool_result: Optional[str] = None
    ) -> list:
        """Build messages for LLM"""
        messages = []

        # System prompt
        system_prompt = """You are Alexandra, an advanced AI assistant similar to JARVIS from Iron Man.

You are helpful, witty, and capable. Keep responses concise for voice - typically 1-3 sentences unless more detail is needed.

You can:
- See through the camera when relevant
- Control smart home devices
- Browse the web and complete tasks
- Execute code
- Remember conversations
- Help with any task

Be conversational and natural."""

        messages.append({"role": "system", "content": system_prompt})

        # Add context if available
        context_parts = []
        if vision_context:
            context_parts.append(f"[What I see: {vision_context}]")
        if tool_result:
            context_parts.append(f"[Action result: {tool_result}]")

        # User message
        user_content = text
        if context_parts:
            user_content = "\n".join(context_parts) + f"\n\nUser: {text}"

        messages.append({"role": "user", "content": user_content})

        return messages

    def _on_state_change(self, old_state: JarvisState, new_state: JarvisState):
        """Handle state changes"""
        if self.on_state_change:
            self.on_state_change(old_state.value, new_state.value)

    def _on_response(self, response: str):
        """Handle response complete"""
        logger.info(f"Response: {response[:100]}...")


def setup_signal_handlers(jarvis: JarvisMode):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(sig, frame):
        print("\nShutdown requested...")
        jarvis.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point"""
    print("=" * 60)
    print("  JARVIS MODE - Alexandra AI")
    print("=" * 60)
    print("")

    # Configuration
    config = JarvisConfig(
        whisper_model="base",
        vision_enabled=True,
        hand_tracking_enabled=True,
        browser_enabled=True,
        always_listening=True,
    )

    jarvis = JarvisMode(config)
    setup_signal_handlers(jarvis)

    await jarvis.start()


if __name__ == "__main__":
    asyncio.run(main())
