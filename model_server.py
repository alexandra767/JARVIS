"""
Alexandra AI - Persistent Model Server
Keeps all models loaded in GPU memory for fast responses
Run inside NGC container: ./run_gpu.sh server
"""

import os
import sys
import time
import torch
import threading
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

# Add paths
sys.path.insert(0, "/workspace/MuseTalk")
sys.path.insert(0, "/workspace/voice_training/F5-TTS")
sys.path.insert(0, "/workspace/ai-clone-chat")

# Configuration
@dataclass
class ModelConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Paths
    musetalk_dir: str = "/workspace/MuseTalk"
    f5_tts_dir: str = "/workspace/voice_training/F5-TTS"
    f5_checkpoint: str = "/workspace/voice_training/F5-TTS/ckpts/alexandra/model_last.pt"
    f5_ref_audio: str = "/workspace/voice_training/F5-TTS/data/alexandra_char/wavs/clip_001.wav"
    f5_ref_text: str = "I love about technology, it keeps surprising me."
    avatar_image: str = "/workspace/ComfyUI/output/ComfyUI_00077_.png"

config = ModelConfig()


class PersistentModels:
    """Singleton class that keeps all models loaded in memory"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = config.device
        self.dtype = config.dtype

        # Model references (loaded lazily or on demand)
        self._musetalk_models = None
        self._f5_tts_model = None
        self._whisper_model = None

        # Status
        self.models_loaded = {
            "musetalk": False,
            "f5_tts": False,
            "whisper": False
        }

        self._initialized = True
        print(f"[ModelServer] Initialized on device: {self.device}")

    def load_all(self):
        """Load all models into GPU memory"""
        print("[ModelServer] Loading all models...")
        start = time.time()

        self.load_musetalk()
        self.load_f5_tts()
        self.load_whisper()

        elapsed = time.time() - start
        print(f"[ModelServer] All models loaded in {elapsed:.1f}s")
        self._print_memory_usage()

    def _print_memory_usage(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[ModelServer] GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    def load_musetalk(self):
        """Load MuseTalk models"""
        if self.models_loaded["musetalk"]:
            return self._musetalk_models

        print("[ModelServer] Loading MuseTalk...")
        start = time.time()

        try:
            os.chdir(config.musetalk_dir)
            from musetalk.utils.utils import load_all_model
            from musetalk.utils.preprocessing import get_landmark_and_bbox
            from musetalk.utils.blending import get_image
            from musetalk.utils.face_parsing import FaceParsing

            # Load models
            vae, unet, pe = load_all_model(device=self.device)

            # Move to GPU
            if hasattr(vae, 'to'):
                vae = vae.to(self.device)
            if hasattr(unet, 'to'):
                unet = unet.to(self.device)
            if hasattr(pe, 'to'):
                pe = pe.to(self.device)

            # Load face parsing
            fp = FaceParsing()

            self._musetalk_models = {
                "vae": vae,
                "unet": unet,
                "pe": pe,
                "face_parsing": fp,
                "timesteps": torch.tensor([0], device=self.device)
            }

            self.models_loaded["musetalk"] = True
            print(f"[ModelServer] MuseTalk loaded in {time.time() - start:.1f}s")

        except Exception as e:
            print(f"[ModelServer] Failed to load MuseTalk: {e}")
            self._musetalk_models = None

        return self._musetalk_models

    def load_f5_tts(self):
        """Load F5-TTS model"""
        if self.models_loaded["f5_tts"]:
            return self._f5_tts_model

        print("[ModelServer] Loading F5-TTS...")
        start = time.time()

        try:
            sys.path.insert(0, os.path.join(config.f5_tts_dir, "src"))
            from f5_tts.api import F5TTS

            # Initialize F5-TTS with correct parameters
            vocoder_path = os.path.join(config.f5_tts_dir, "ckpts")

            self._f5_tts_model = F5TTS(
                model="F5TTS_v1_Base",
                ckpt_file=config.f5_checkpoint,
                vocab_file="",
                ode_method="euler",
                use_ema=True,
                vocoder_local_path=vocoder_path,
                device=self.device
            )

            self.models_loaded["f5_tts"] = True
            print(f"[ModelServer] F5-TTS loaded in {time.time() - start:.1f}s")

        except Exception as e:
            print(f"[ModelServer] Failed to load F5-TTS: {e}")
            import traceback
            traceback.print_exc()
            self._f5_tts_model = None

        return self._f5_tts_model

    def load_whisper(self):
        """Load Whisper for audio processing"""
        if self.models_loaded["whisper"]:
            return self._whisper_model

        print("[ModelServer] Loading Whisper...")
        start = time.time()

        try:
            os.chdir(config.musetalk_dir)
            from musetalk.whisper.whisper import load_model as load_whisper

            whisper_path = os.path.join(config.musetalk_dir, "models/whisper")
            self._whisper_model = load_whisper("tiny", whisper_path, self.device)

            self.models_loaded["whisper"] = True
            print(f"[ModelServer] Whisper loaded in {time.time() - start:.1f}s")

        except Exception as e:
            print(f"[ModelServer] Failed to load Whisper: {e}")
            self._whisper_model = None

        return self._whisper_model

    @property
    def musetalk(self):
        return self._musetalk_models or self.load_musetalk()

    @property
    def f5_tts(self):
        return self._f5_tts_model or self.load_f5_tts()

    @property
    def whisper(self):
        return self._whisper_model or self.load_whisper()

    def unload_all(self):
        """Unload all models to free GPU memory for training/other tasks"""
        print("[ModelServer] Unloading all models...")

        if self._musetalk_models:
            del self._musetalk_models
            self._musetalk_models = None
            self.models_loaded["musetalk"] = False

        if self._f5_tts_model:
            del self._f5_tts_model
            self._f5_tts_model = None
            self.models_loaded["f5_tts"] = False

        if self._whisper_model:
            del self._whisper_model
            self._whisper_model = None
            self.models_loaded["whisper"] = False

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._print_memory_usage()
        print("[ModelServer] All models unloaded - memory freed for training")

    def get_memory_status(self) -> dict:
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9
            }
        return {"error": "CUDA not available"}

    def generate_speech(self, text: str, output_path: str) -> Optional[str]:
        """Generate speech using F5-TTS"""
        if not self.f5_tts:
            print("[ModelServer] F5-TTS not loaded")
            return None

        try:
            # Clean text for TTS
            import re
            clean_text = text.replace(chr(8212), "-").replace(chr(8211), "-")
            clean_text = re.sub(r"[()\\[\\]{}]", "", clean_text)
            clean_text = re.sub(r"[^\w\s.,!?'-]", "", clean_text)
            clean_text = clean_text.strip()

            if len(clean_text) < 5:
                print("[ModelServer] Text too short for TTS")
                return None

            # Generate audio
            wav, sr, _ = self._f5_tts_model.infer(
                ref_file=config.f5_ref_audio,
                ref_text=config.f5_ref_text,
                gen_text=clean_text,
                file_wave=output_path,
                seed=None  # Random seed
            )

            if os.path.exists(output_path):
                return output_path

        except Exception as e:
            print(f"[ModelServer] TTS error: {e}")
            import traceback
            traceback.print_exc()

        return None

    def generate_lipsync(self, audio_path: str, image_path: str, output_dir: str) -> Optional[str]:
        """Generate lip-synced video using MuseTalk"""
        if not self.musetalk:
            print("[ModelServer] MuseTalk not loaded")
            return None

        import cv2
        import subprocess

        try:
            os.chdir(config.musetalk_dir)
            from musetalk.utils.utils import datagen
            from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder
            from musetalk.utils.blending import get_image
            from musetalk.whisper.audio2feature import Audio2Feature

            models = self._musetalk_models
            vae = models["vae"]
            unet = models["unet"]
            pe = models["pe"]
            fp = models["face_parsing"]
            timesteps = models["timesteps"]

            # Read image
            frame = cv2.imread(image_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get face coordinates
            coord_list, frame_list = get_landmark_and_bbox([image_path], 0)
            bbox = coord_list[0]

            if bbox == coord_placeholder:
                print("[ModelServer] No face detected in avatar image")
                return None

            # Process audio
            audio_processor = Audio2Feature(model_path=os.path.join(config.musetalk_dir, "models/whisper/tiny.pt"))
            whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
            whisper_chunks = audio_processor.get_whisper_chunk(
                whisper_input_features, self.device, self.dtype,
                self.whisper, librosa_length, fps=25
            )

            # Prepare face
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(crop_frame)

            # Generate frames
            os.makedirs(output_dir, exist_ok=True)
            batch_size = 8
            video_num = len(whisper_chunks)
            gen = datagen(whisper_chunks, [latents]*video_num, batch_size, 0, self.device)

            res_frames = []
            for whisper_batch, latent_batch in gen:
                audio_feature = pe(whisper_batch)
                latent_batch = latent_batch.to(dtype=self.dtype)
                pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature).sample
                recon = vae.decode_latents(pred_latents)
                res_frames.extend(recon)

            # Write frames
            for i, res_frame in enumerate(res_frames):
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                combine_frame = get_image(frame.copy(), res_frame, [x1, y1, x2, y2], mode='jaw', fp=fp)
                cv2.imwrite(f'{output_dir}/{i:08d}.png', cv2.cvtColor(combine_frame, cv2.COLOR_RGB2BGR))

            # Create video with ffmpeg
            output_video = os.path.join(output_dir, "output.mp4")
            subprocess.run([
                'ffmpeg', '-y', '-framerate', '25',
                '-i', f'{output_dir}/%08d.png',
                '-i', audio_path,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-shortest',
                output_video
            ], capture_output=True)

            if os.path.exists(output_video):
                return output_video

        except Exception as e:
            print(f"[ModelServer] Lipsync error: {e}")
            import traceback
            traceback.print_exc()

        return None


# Global instance
models = PersistentModels()


def start_server():
    """Start the model server with Gradio UI"""
    import gradio as gr
    import uuid
    import ollama

    OUTPUT_DIR = "/tmp/alexandra_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pre-load models
    print("="*60)
    print("Alexandra AI - Persistent Model Server")
    print("="*60)
    models.load_all()
    print("="*60)

    def respond(user_input: str, progress=gr.Progress()):
        """Generate avatar response"""
        if not user_input.strip():
            return "", None, "Please enter a message"

        session_id = uuid.uuid4().hex[:8]
        session_dir = os.path.join(OUTPUT_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        status_log = []
        start_time = time.time()

        try:
            # Step 1: Get LLM response
            progress(0.1, desc="Thinking...")
            llm_start = time.time()

            try:
                response = ollama.chat(
                    model="qwen2.5:72b",
                    messages=[{"role": "user", "content": user_input}]
                )
                response_text = response['message']['content']
            except Exception as e:
                response_text = f"Hello! You asked: {user_input}. (LLM unavailable: {str(e)[:30]})"

            status_log.append(f"[LLM] {time.time() - llm_start:.1f}s")

            # Step 2: Generate voice
            progress(0.3, desc="Generating voice...")
            voice_start = time.time()
            voice_path = os.path.join(session_dir, "voice.wav")
            voice_result = models.generate_speech(response_text, voice_path)

            if not voice_result:
                status_log.append("[VOICE] Failed")
                return response_text, None, "\n".join(status_log)

            status_log.append(f"[VOICE] {time.time() - voice_start:.1f}s")

            # Step 3: Generate lip-synced video
            progress(0.5, desc="Animating avatar...")
            lipsync_start = time.time()
            video_path = models.generate_lipsync(voice_path, config.avatar_image, session_dir)

            if not video_path:
                status_log.append("[VIDEO] Failed")
                return response_text, None, "\n".join(status_log)

            status_log.append(f"[VIDEO] {time.time() - lipsync_start:.1f}s")

            progress(1.0, desc="Done!")
            total_time = time.time() - start_time
            status_log.append(f"[TOTAL] {total_time:.1f}s")

            return response_text, video_path, "\n".join(status_log)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return "", None, f"Error: {str(e)}"

    def test_components():
        """Test all components"""
        tests = []
        tests.append(f"CUDA: {'Available' if torch.cuda.is_available() else 'NOT AVAILABLE'}")
        if torch.cuda.is_available():
            tests.append(f"GPU: {torch.cuda.get_device_name(0)}")
            mem = models.get_memory_status()
            tests.append(f"Memory: {mem['allocated_gb']:.1f}GB used / {mem['total_gb']:.0f}GB total")
            tests.append(f"Free: {mem['free_gb']:.1f}GB available")
        tests.append(f"MuseTalk: {'Loaded' if models.models_loaded['musetalk'] else 'Not loaded'}")
        tests.append(f"F5-TTS: {'Loaded' if models.models_loaded['f5_tts'] else 'Not loaded'}")
        tests.append(f"Whisper: {'Loaded' if models.models_loaded['whisper'] else 'Not loaded'}")
        tests.append(f"Avatar: {'Found' if os.path.exists(config.avatar_image) else 'MISSING'}")
        return "\n".join(tests)

    def unload_for_training():
        """Unload avatar models to free memory for training"""
        models.unload_all()
        return "Models unloaded! Memory freed for training/image generation."

    def reload_models():
        """Reload all avatar models"""
        models.load_all()
        return test_components()

    # Gradio UI
    with gr.Blocks(title="Alexandra AI - GPU Server", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Alexandra AI - Persistent GPU Server")
        gr.Markdown("*Models stay loaded for fast responses*")

        with gr.Row():
            with gr.Column(scale=2):
                video = gr.Video(label="Alexandra", autoplay=True, height=512)
            with gr.Column(scale=1):
                response_text = gr.Textbox(label="Response", lines=4)
                status = gr.Textbox(label="Status", lines=8)

        user_input = gr.Textbox(
            label="You:",
            placeholder="Type your message...",
            lines=2
        )

        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
            test_btn = gr.Button("Refresh Status")

        with gr.Accordion("Memory Management", open=False):
            gr.Markdown("**Free GPU memory for training or image generation:**")
            with gr.Row():
                unload_btn = gr.Button("Unload Models (Free Memory)", variant="stop")
                reload_btn = gr.Button("Reload Models", variant="secondary")
            unload_btn.click(unload_for_training, outputs=[status])
            reload_btn.click(reload_models, outputs=[status])

        submit_btn.click(
            respond,
            inputs=[user_input],
            outputs=[response_text, video, status]
        )
        user_input.submit(
            respond,
            inputs=[user_input],
            outputs=[response_text, video, status]
        )
        test_btn.click(test_components, outputs=[status])

        # Show initial status
        app.load(test_components, outputs=[status])

    print("\nStarting server on http://0.0.0.0:7865")
    app.launch(server_name="0.0.0.0", server_port=7865)


if __name__ == "__main__":
    start_server()
