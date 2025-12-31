#!/usr/bin/env python3
"""
JARVIS Voice System - Real-time Speech-to-Text and Text-to-Speech
Uses faster-whisper for optimized transcription and streaming TTS
"""

import asyncio
import numpy as np
import sounddevice as sd
import queue
import threading
import time
import os
import sys
from typing import Optional, AsyncIterator, Callable
from dataclasses import dataclass
import logging
import wave
import tempfile

logger = logging.getLogger("JarvisVoice")

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.float32
CHUNK_DURATION = 0.5  # seconds
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.0  # seconds of silence to end utterance
MAX_RECORDING_DURATION = 30  # seconds


@dataclass
class AudioConfig:
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    chunk_duration: float = CHUNK_DURATION
    silence_threshold: float = SILENCE_THRESHOLD
    silence_duration: float = SILENCE_DURATION


class VoiceActivityDetector:
    """Simple voice activity detection using energy threshold"""

    def __init__(self, threshold: float = SILENCE_THRESHOLD):
        self.threshold = threshold
        self.speaking = False
        self.silence_start = None

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if audio chunk contains speech"""
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        return energy > self.threshold

    def process(self, audio_chunk: np.ndarray) -> str:
        """
        Process audio chunk and return state.
        Returns: 'speech', 'silence', or 'end_of_utterance'
        """
        is_speech = self.is_speech(audio_chunk)

        if is_speech:
            self.speaking = True
            self.silence_start = None
            return "speech"
        else:
            if self.speaking:
                if self.silence_start is None:
                    self.silence_start = time.time()
                elif time.time() - self.silence_start > SILENCE_DURATION:
                    self.speaking = False
                    self.silence_start = None
                    return "end_of_utterance"
            return "silence"


class JarvisVoiceInput:
    """
    Real-time voice input with faster-whisper transcription.
    Supports continuous listening and interruption detection.
    """

    def __init__(self, model_size: str = "base", device: str = "cuda"):
        self.model_size = model_size
        self.device = device
        self.model = None
        self.audio_queue = queue.Queue()
        self.recording = False
        self.stream = None
        self.vad = VoiceActivityDetector()

        # For interrupt detection
        self.interrupt_callback: Optional[Callable] = None
        self.is_speaking = False  # Set by TTS when speaking

        self._load_model()

    def _load_model(self):
        """Load faster-whisper model"""
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading faster-whisper model: {self.model_size}")

            compute_type = "float16" if self.device == "cuda" else "int8"
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type
            )
            logger.info("Whisper model loaded")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            logger.warning(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())

    async def start_listening(self):
        """Start continuous audio capture"""
        if self.recording:
            return

        self.recording = True
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._audio_callback,
            blocksize=int(SAMPLE_RATE * CHUNK_DURATION)
        )
        self.stream.start()
        logger.info("Voice input started")

    async def stop_listening(self):
        """Stop audio capture"""
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        logger.info("Voice input stopped")

    async def listen(self) -> Optional[str]:
        """
        Listen for speech and return transcribed text.
        Returns None if no speech detected or interrupted.
        """
        if not self.recording:
            await self.start_listening()

        audio_buffer = []
        utterance_started = False

        try:
            while self.recording:
                try:
                    # Get audio chunk with timeout
                    chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                # Check for voice activity
                state = self.vad.process(chunk)

                if state == "speech":
                    utterance_started = True
                    audio_buffer.append(chunk)

                    # Check for interruption (user speaking while TTS is playing)
                    if self.is_speaking and self.interrupt_callback:
                        logger.info("Interrupt detected!")
                        await self.interrupt_callback()
                        self.is_speaking = False

                elif state == "silence" and utterance_started:
                    audio_buffer.append(chunk)

                elif state == "end_of_utterance" and audio_buffer:
                    # Transcribe the complete utterance
                    audio_data = np.concatenate(audio_buffer)
                    text = await self._transcribe(audio_data)
                    audio_buffer = []
                    utterance_started = False

                    if text and text.strip():
                        return text.strip()

                # Prevent buffer overflow
                if len(audio_buffer) > int(MAX_RECORDING_DURATION / CHUNK_DURATION):
                    logger.warning("Max recording duration reached")
                    audio_data = np.concatenate(audio_buffer)
                    text = await self._transcribe(audio_data)
                    audio_buffer = []
                    utterance_started = False
                    if text and text.strip():
                        return text.strip()

                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Error in listen: {e}")
            return None

        return None

    async def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe audio using faster-whisper"""
        if self.model is None:
            logger.error("Whisper model not loaded")
            return None

        try:
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                audio
            )
            return result
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    def _transcribe_sync(self, audio: np.ndarray) -> Optional[str]:
        """Synchronous transcription"""
        try:
            segments, info = self.model.transcribe(
                audio,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            text = " ".join(segment.text for segment in segments)
            logger.debug(f"Transcribed: {text}")
            return text

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None


class JarvisVoiceOutput:
    """
    Text-to-Speech using F5-TTS with streaming support.
    Integrates with existing Alexandra TTS setup.
    """

    def __init__(self):
        self.speaking = False
        self.stop_requested = False
        self.audio_queue = queue.Queue()
        self.playback_thread = None

        # F5-TTS configuration (from existing setup)
        self.model_path = os.path.expanduser(
            "~/voice_training/F5-TTS/ckpts/alexandra/model_last.pt"
        )
        self.ref_audio = os.path.expanduser(
            "~/voice_training/F5-TTS/data/alexandra_char/wavs/clip_001.wav"
        )
        self.ref_text = "This is Alexandra speaking, your personal AI assistant."

        self.tts_model = None
        self._load_model()

    def _load_model(self):
        """Load F5-TTS model"""
        try:
            # Import F5-TTS
            sys.path.insert(0, os.path.expanduser("~/voice_training/F5-TTS"))

            # Check if model exists
            if not os.path.exists(self.model_path):
                logger.warning(f"F5-TTS model not found at {self.model_path}")
                logger.info("Will use fallback TTS")
                return

            logger.info("F5-TTS model path verified")
            # Actual model loading happens on first use to save memory

        except Exception as e:
            logger.error(f"Failed to setup F5-TTS: {e}")

    async def speak(self, text: str):
        """
        Speak text using TTS.
        Supports interruption via stop_requested flag.
        """
        if not text or not text.strip():
            return

        self.speaking = True
        self.stop_requested = False

        try:
            # Generate audio
            audio = await self._generate_audio(text)

            if audio is not None and not self.stop_requested:
                await self._play_audio(audio)

        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            self.speaking = False

    async def speak_streaming(self, text_iterator: AsyncIterator[str]):
        """
        Stream TTS from text iterator (for real-time LLM output).
        Generates and plays audio sentence by sentence.
        """
        self.speaking = True
        self.stop_requested = False

        sentence_buffer = ""
        sentence_endings = ".!?;"

        try:
            async for chunk in text_iterator:
                if self.stop_requested:
                    break

                sentence_buffer += chunk

                # Check for complete sentences
                for ending in sentence_endings:
                    if ending in sentence_buffer:
                        parts = sentence_buffer.split(ending, 1)
                        complete_sentence = parts[0] + ending

                        # Speak the complete sentence
                        audio = await self._generate_audio(complete_sentence)
                        if audio is not None and not self.stop_requested:
                            await self._play_audio(audio)

                        sentence_buffer = parts[1] if len(parts) > 1 else ""

            # Speak any remaining text
            if sentence_buffer.strip() and not self.stop_requested:
                audio = await self._generate_audio(sentence_buffer)
                if audio is not None:
                    await self._play_audio(audio)

        except Exception as e:
            logger.error(f"Streaming TTS error: {e}")
        finally:
            self.speaking = False

    async def stop(self):
        """Stop current speech"""
        self.stop_requested = True
        sd.stop()
        self.speaking = False
        logger.info("TTS stopped")

    async def _generate_audio(self, text: str) -> Optional[np.ndarray]:
        """Generate audio from text using F5-TTS"""
        try:
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(
                None,
                self._generate_audio_sync,
                text
            )
            return audio
        except Exception as e:
            logger.error(f"Audio generation error: {e}")
            return None

    def _generate_audio_sync(self, text: str) -> Optional[np.ndarray]:
        """Synchronous audio generation"""
        try:
            # Try F5-TTS first
            if os.path.exists(self.model_path):
                return self._generate_f5tts(text)

            # Fallback to pyttsx3 or espeak
            return self._generate_fallback(text)

        except Exception as e:
            logger.error(f"Audio generation error: {e}")
            return None

    def _generate_f5tts(self, text: str) -> Optional[np.ndarray]:
        """Generate audio using F5-TTS"""
        try:
            # This integrates with your existing F5-TTS setup
            # Using the simple_f5_infer.py approach

            import torch
            import torchaudio

            # Import F5-TTS modules
            sys.path.insert(0, os.path.expanduser("~/voice_training/F5-TTS"))

            from f5_tts.api import F5TTS

            # Initialize model if not done
            if self.tts_model is None:
                self.tts_model = F5TTS(
                    model_type="F5-TTS",
                    ckpt_file=self.model_path,
                    vocab_file="",
                    device="cuda"
                )

            # Generate audio
            audio, sr = self.tts_model.infer(
                ref_file=self.ref_audio,
                ref_text=self.ref_text,
                gen_text=text,
                target_rms=0.1
            )

            return audio

        except Exception as e:
            logger.error(f"F5-TTS generation error: {e}")
            return self._generate_fallback(text)

    def _generate_fallback(self, text: str) -> Optional[np.ndarray]:
        """Fallback TTS using pyttsx3 or espeak"""
        try:
            import pyttsx3
            engine = pyttsx3.init()

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            engine.save_to_file(text, temp_path)
            engine.runAndWait()

            # Load and return audio
            import scipy.io.wavfile as wav
            sr, audio = wav.read(temp_path)
            os.unlink(temp_path)

            # Convert to float32
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0

            return audio

        except Exception as e:
            logger.warning(f"Fallback TTS failed: {e}")
            return None

    async def _play_audio(self, audio: np.ndarray):
        """Play audio through speakers"""
        try:
            if self.stop_requested:
                return

            # Determine sample rate
            sr = 24000  # F5-TTS default

            # Play audio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: sd.play(audio, sr)
            )

            # Wait for playback to complete (with interruption check)
            while sd.get_stream().active:
                if self.stop_requested:
                    sd.stop()
                    break
                await asyncio.sleep(0.05)

        except Exception as e:
            logger.error(f"Audio playback error: {e}")


class JarvisVoiceSystem:
    """
    Complete voice system combining input and output.
    Handles the full voice interaction loop.
    """

    def __init__(self, whisper_model: str = "base", device: str = "cuda"):
        self.input = JarvisVoiceInput(model_size=whisper_model, device=device)
        self.output = JarvisVoiceOutput()

        # Link interrupt detection
        self.input.interrupt_callback = self.handle_interrupt

    async def start(self):
        """Start voice system"""
        await self.input.start_listening()
        logger.info("Voice system started")

    async def stop(self):
        """Stop voice system"""
        await self.input.stop_listening()
        await self.output.stop()
        logger.info("Voice system stopped")

    async def listen(self) -> Optional[str]:
        """Listen for voice input"""
        # Update speaking state for interrupt detection
        self.input.is_speaking = self.output.speaking
        return await self.input.listen()

    async def speak(self, text: str):
        """Speak text"""
        await self.output.speak(text)

    async def speak_streaming(self, text_iterator: AsyncIterator[str]):
        """Stream speech from text iterator"""
        await self.output.speak_streaming(text_iterator)

    async def handle_interrupt(self):
        """Handle voice interruption"""
        logger.info("Handling interrupt")
        await self.output.stop()


# Test functions
async def test_voice_input():
    """Test voice input"""
    print("Testing voice input...")
    voice = JarvisVoiceInput(model_size="base", device="cuda")
    await voice.start_listening()

    print("Speak something...")
    text = await voice.listen()
    print(f"You said: {text}")

    await voice.stop_listening()


async def test_voice_output():
    """Test voice output"""
    print("Testing voice output...")
    voice = JarvisVoiceOutput()
    await voice.speak("Hello, I am Alexandra, your personal AI assistant. How can I help you today?")


async def test_full_system():
    """Test full voice system"""
    print("Testing full voice system...")
    voice = JarvisVoiceSystem(whisper_model="base", device="cuda")
    await voice.start()

    print("Say something...")
    text = await voice.listen()
    print(f"You said: {text}")

    print("Speaking response...")
    await voice.speak(f"You said: {text}")

    await voice.stop()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test = sys.argv[1]
        if test == "input":
            asyncio.run(test_voice_input())
        elif test == "output":
            asyncio.run(test_voice_output())
        elif test == "full":
            asyncio.run(test_full_system())
    else:
        asyncio.run(test_full_system())
