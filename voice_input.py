"""
Alexandra AI - Voice Input
Speech-to-text using OpenAI Whisper
"""

import os
import tempfile
import numpy as np

class VoiceInput:
    """Handle voice input transcription"""

    def __init__(self, model_name="base"):
        """
        Initialize Whisper model

        Args:
            model_name: tiny, base, small, medium, large
        """
        self.model = None
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Load Whisper model"""
        try:
            import whisper
            print(f"[VoiceInput] Loading Whisper {self.model_name} model...")
            self.model = whisper.load_model(self.model_name)
            print(f"[VoiceInput] Model loaded successfully")
        except ImportError:
            print("[VoiceInput] Whisper not installed. Run: pip install openai-whisper")
        except Exception as e:
            print(f"[VoiceInput] Error loading model: {e}")

    def transcribe(self, audio_path):
        """
        Transcribe audio file to text

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)

        Returns:
            dict with 'text', 'language', 'segments'
        """
        if not self.model:
            return {"text": "", "error": "Model not loaded"}

        if not os.path.exists(audio_path):
            return {"text": "", "error": f"Audio file not found: {audio_path}"}

        try:
            result = self.model.transcribe(audio_path)
            return {
                "text": result["text"].strip(),
                "language": result.get("language", "en"),
                "segments": result.get("segments", [])
            }
        except Exception as e:
            return {"text": "", "error": str(e)}

    def transcribe_array(self, audio_array, sample_rate=16000):
        """
        Transcribe audio from numpy array

        Args:
            audio_array: numpy array of audio data
            sample_rate: audio sample rate

        Returns:
            dict with transcription results
        """
        if not self.model:
            return {"text": "", "error": "Model not loaded"}

        try:
            # Ensure audio is float32 and normalized
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)

            if audio_array.max() > 1.0:
                audio_array = audio_array / 32768.0

            result = self.model.transcribe(audio_array)
            return {
                "text": result["text"].strip(),
                "language": result.get("language", "en"),
                "segments": result.get("segments", [])
            }
        except Exception as e:
            return {"text": "", "error": str(e)}


class RealtimeVoiceInput:
    """Real-time voice input with silence detection"""

    def __init__(self, model_name="base"):
        self.voice_input = VoiceInput(model_name)
        self.is_listening = False

    def listen_and_transcribe(self, timeout=10, silence_threshold=0.01, silence_duration=1.5):
        """
        Listen to microphone and transcribe when speech ends

        Args:
            timeout: Maximum listening time in seconds
            silence_threshold: Audio level below which is considered silence
            silence_duration: How long silence must last to stop recording

        Returns:
            Transcribed text
        """
        try:
            import sounddevice as sd
            import scipy.io.wavfile as wav
        except ImportError:
            return {"text": "", "error": "sounddevice not installed"}

        sample_rate = 16000
        audio_chunks = []
        silence_counter = 0
        max_silence_chunks = int(silence_duration * sample_rate / 1024)

        print("[VoiceInput] Listening...")
        self.is_listening = True

        def audio_callback(indata, frames, time, status):
            nonlocal silence_counter, audio_chunks

            if not self.is_listening:
                raise sd.CallbackStop()

            audio_chunks.append(indata.copy())

            # Check for silence
            volume = np.abs(indata).mean()
            if volume < silence_threshold:
                silence_counter += 1
                if silence_counter >= max_silence_chunks:
                    self.is_listening = False
                    raise sd.CallbackStop()
            else:
                silence_counter = 0

        try:
            with sd.InputStream(samplerate=sample_rate, channels=1,
                              dtype=np.float32, callback=audio_callback,
                              blocksize=1024):
                sd.sleep(int(timeout * 1000))
        except sd.CallbackStop:
            pass

        self.is_listening = False

        if not audio_chunks:
            return {"text": "", "error": "No audio captured"}

        # Combine chunks and transcribe
        audio_data = np.concatenate(audio_chunks, axis=0).flatten()

        # Save to temp file (Whisper works better with files)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, sample_rate, audio_data)
            result = self.voice_input.transcribe(f.name)
            os.unlink(f.name)

        return result

    def stop_listening(self):
        """Stop current listening session"""
        self.is_listening = False


# Singleton instance
_voice_input = None

def get_voice_input(model_name="base"):
    """Get or create voice input singleton"""
    global _voice_input
    if _voice_input is None:
        _voice_input = VoiceInput(model_name)
    return _voice_input

def transcribe_audio(audio_path):
    """Convenience function to transcribe audio"""
    return get_voice_input().transcribe(audio_path)


if __name__ == "__main__":
    # Test
    vi = VoiceInput("base")

    # Test with a file if provided
    import sys
    if len(sys.argv) > 1:
        result = vi.transcribe(sys.argv[1])
        print(f"Transcription: {result['text']}")
        if 'error' in result:
            print(f"Error: {result['error']}")
