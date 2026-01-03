"""
MuseTalk Avatar Integration for JARVIS
Generates real-time lip-synced avatar videos from F5-TTS audio
"""

import os
import sys
import torch

# Fix PyTorch 2.6+ checkpoint loading - use weights_only=False for trusted models
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Fix PyTorch 2.6+ meta tensor issue - use assign=True for load_state_dict
_original_load_state_dict = torch.nn.Module.load_state_dict
def _patched_load_state_dict(self, state_dict, strict=True, assign=False):
    try:
        return _original_load_state_dict(self, state_dict, strict=strict, assign=True)
    except Exception:
        return _original_load_state_dict(self, state_dict, strict=strict, assign=assign)
torch.nn.Module.load_state_dict = _patched_load_state_dict

# Disable accelerate's device_map which causes meta tensor issues
os.environ["ACCELERATE_DISABLE_RICH"] = "1"

# Patch diffusers to avoid meta tensors
try:
    import diffusers.models.modeling_utils
    _orig_load_model = diffusers.models.modeling_utils.load_model_dict_into_meta
    def _patched_load_model(*args, **kwargs):
        # Force non-meta loading
        kwargs['keep_in_fp32_modules'] = []
        return _orig_load_model(*args, **kwargs)
    diffusers.models.modeling_utils.load_model_dict_into_meta = _patched_load_model
except Exception:
    pass

import cv2
import numpy as np
import tempfile
import subprocess
from typing import Optional, Dict, Any

# Add MuseTalk to path - check both Docker and host paths
MUSETALK_DIR_OPTIONS = [
    "/workspace/host/MuseTalk",  # Docker path
    "/home/alexandratitus767/MuseTalk",  # Host path
]
MUSETALK_DIR = None
for path in MUSETALK_DIR_OPTIONS:
    if os.path.exists(path):
        MUSETALK_DIR = path
        sys.path.insert(0, MUSETALK_DIR)
        break

MUSETALK_AVAILABLE = MUSETALK_DIR is not None

# Avatar image directory - check both Docker and host paths
AVATAR_DIR_OPTIONS = [
    "/workspace/host/ai-clone-training/alexandra_avatar_sets",  # Docker path
    "/home/alexandratitus767/ai-clone-training/alexandra_avatar_sets",  # Host path
]
AVATAR_DIR = None
for path in AVATAR_DIR_OPTIONS:
    if os.path.exists(path):
        AVATAR_DIR = path
        break

if AVATAR_DIR is None:
    AVATAR_DIR = AVATAR_DIR_OPTIONS[0]  # Default

# Global engine instance
_engine = None


class MuseTalkEngine:
    """Singleton engine for MuseTalk avatar generation"""

    def __init__(self):
        self.loaded = False
        self.vae = None
        self.unet = None
        self.pe = None
        self.whisper = None
        self.audio_processor = None
        self.fp = None
        self.device = "cuda"
        self.weight_dtype = torch.float32  # Use float32 to match UNet
        self.timesteps = torch.tensor([0], device="cuda")
        self.avatar_cache: Dict[str, Dict[str, Any]] = {}
        self.current_avatar_id: Optional[str] = None

    def load_models(self) -> str:
        """Load MuseTalk models (lazy loading)"""
        if self.loaded:
            return "Models already loaded"

        if not MUSETALK_AVAILABLE:
            return "MuseTalk not available"

        # Save current directory and change to MuseTalk dir (required for relative paths)
        original_cwd = os.getcwd()

        try:
            os.chdir(MUSETALK_DIR)
            print(f"[AVATAR] Loading MuseTalk models from {MUSETALK_DIR}...", flush=True)

            from musetalk.utils.audio_processor import AudioProcessor
            from musetalk.models.unet import UNet, PositionalEncoding
            from diffusers import AutoencoderKL
            from transformers import WhisperModel
            import torchvision.transforms as transforms

            # Load VAE manually to avoid meta tensor issues
            print("[AVATAR] Loading VAE...", flush=True)
            vae_path = f"{MUSETALK_DIR}/models/sd-vae"

            # Force non-meta loading by using low_cpu_mem_usage=False
            vae_model = AutoencoderKL.from_pretrained(
                vae_path,
                low_cpu_mem_usage=False,
                device_map=None,
                torch_dtype=torch.float32
            )
            vae_model = vae_model.to(self.device)

            # Create a simple wrapper class for VAE
            class VAEWrapper:
                def __init__(self, vae, device):
                    self.vae = vae
                    self.device = device
                    self.scaling_factor = vae.config.scaling_factor
                    self._resized_img = 256
                    self.transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    self._mask_tensor = self._get_mask_tensor()

                def _get_mask_tensor(self):
                    # Upper half = 1, lower half = 0 (mask the mouth region)
                    mask = torch.zeros((256, 256))
                    mask[:128, :] = 1
                    return mask.to(self.device)

                def _preprocess_img(self, img, half_mask=False):
                    """Preprocess image for VAE encoding"""
                    if isinstance(img, np.ndarray):
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img.astype(np.float32) / 255.0
                        img = np.transpose(img, (2, 0, 1))
                        img = torch.from_numpy(img)

                    if half_mask:
                        # Apply mask - zero out lower half (mouth region)
                        img = img * self._mask_tensor.cpu()

                    img = self.transform(img)
                    img = img.unsqueeze(0).to(self.device, dtype=self.vae.dtype)
                    return img

                def _encode_latents(self, image):
                    """Encode image to latents"""
                    with torch.no_grad():
                        latent_dist = self.vae.encode(image).latent_dist
                        latents = self.scaling_factor * latent_dist.sample()
                    return latents

                def get_latents_for_unet(self, img):
                    """Process image and get 8-channel latents for UNet
                    Returns concatenation of masked_latents + ref_latents (4+4=8 channels)
                    """
                    # Get masked latents (lower half masked out)
                    masked_img = self._preprocess_img(img, half_mask=True)
                    masked_latents = self._encode_latents(masked_img)  # [1, 4, 32, 32]

                    # Get reference latents (full image)
                    ref_img = self._preprocess_img(img, half_mask=False)
                    ref_latents = self._encode_latents(ref_img)  # [1, 4, 32, 32]

                    # Concatenate: [1, 8, 32, 32]
                    latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)
                    return latent_model_input

                def decode_latents(self, latents):
                    """Decode latents to images"""
                    latents = latents / self.scaling_factor
                    with torch.no_grad():
                        imgs = self.vae.decode(latents).sample
                    imgs = (imgs / 2 + 0.5).clamp(0, 1)
                    imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
                    imgs = (imgs * 255).astype(np.uint8)
                    return [img for img in imgs]

            self.vae = VAEWrapper(vae_model, self.device)
            print("[AVATAR] VAE loaded!", flush=True)

            # Load UNet
            print("[AVATAR] Loading UNet...", flush=True)
            self.unet = UNet(
                unet_config=f"{MUSETALK_DIR}/models/musetalkV15/musetalk.json",
                model_path=f"{MUSETALK_DIR}/models/musetalkV15/unet.pth",
                device=self.device
            )
            print("[AVATAR] UNet loaded!", flush=True)

            # Load Positional Encoding
            self.pe = PositionalEncoding(d_model=384)
            print("[AVATAR] PE loaded!", flush=True)

            # Audio processor for Whisper features
            print("[AVATAR] Loading AudioProcessor...", flush=True)
            self.audio_processor = AudioProcessor(f"{MUSETALK_DIR}/models/whisper")
            print("[AVATAR] AudioProcessor loaded!", flush=True)

            # Whisper model for audio encoding
            print("[AVATAR] Loading Whisper...", flush=True)
            self.whisper = WhisperModel.from_pretrained(
                f"{MUSETALK_DIR}/models/whisper"
            ).to(self.device)
            print("[AVATAR] Whisper loaded!", flush=True)

            # Skip FaceParsing entirely - causes meta tensor issues on PyTorch 2.6+
            # Will use simple blending instead
            self.fp = None
            print("[AVATAR] Using simple blending (FaceParsing skipped for PyTorch 2.6 compatibility)", flush=True)

            self.loaded = True
            print("[AVATAR] MuseTalk models loaded!", flush=True)
            return "Models loaded successfully"

        except Exception as e:
            print(f"[AVATAR] Error loading models: {e}", flush=True)
            return f"Error loading models: {e}"

        finally:
            # Restore original directory
            os.chdir(original_cwd)

    def unload_models(self) -> str:
        """Unload models to free GPU memory"""
        if not self.loaded:
            return "Models not loaded"

        try:
            del self.vae
            del self.unet
            del self.pe
            del self.whisper
            del self.audio_processor
            del self.fp

            self.vae = None
            self.unet = None
            self.pe = None
            self.whisper = None
            self.audio_processor = None
            self.fp = None
            self.loaded = False
            self.avatar_cache.clear()

            torch.cuda.empty_cache()
            print("[AVATAR] Models unloaded", flush=True)
            return "Models unloaded"

        except Exception as e:
            return f"Error unloading: {e}"

    def _detect_face_opencv(self, frame: np.ndarray) -> Optional[tuple]:
        """Detect face using OpenCV (avoids mmpose/dwpose PyTorch 2.6 issues)"""
        # Try DNN face detector first (more accurate)
        try:
            # Check for DNN model files
            dnn_proto = cv2.data.haarcascades.replace("haarcascades", "").rstrip("/") + "/../dnn/deploy.prototxt"
            dnn_model = cv2.data.haarcascades.replace("haarcascades", "").rstrip("/") + "/../dnn/res10_300x300_ssd_iter_140000.caffemodel"

            if os.path.exists(dnn_proto) and os.path.exists(dnn_model):
                net = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model)
                h, w = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype(int)
                        # Add padding
                        pad = int((x2 - x1) * 0.3)
                        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                        return (x1, y1, x2, y2)
        except Exception:
            pass

        # Fall back to Haar cascade
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            if len(faces) > 0:
                # Get largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                # Add padding for better results
                pad = int(w * 0.4)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)
                return (x1, y1, x2, y2)
        except Exception as e:
            print(f"[AVATAR] Haar cascade failed: {e}", flush=True)

        # Last resort: use center crop (for controlled avatar images)
        h, w = frame.shape[:2]
        # Assume face is in center-upper portion
        size = min(w, h) * 0.7
        cx, cy = w // 2, int(h * 0.4)
        x1 = int(cx - size // 2)
        y1 = int(cy - size // 2)
        x2 = int(cx + size // 2)
        y2 = int(cy + size // 2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        print(f"[AVATAR] Using center crop fallback: ({x1}, {y1}, {x2}, {y2})", flush=True)
        return (x1, y1, x2, y2)

    def prepare_avatar(self, image_path: str, avatar_id: str) -> Dict[str, Any]:
        """Pre-process avatar image for fast inference"""
        if avatar_id in self.avatar_cache:
            print(f"[AVATAR] Using cached avatar: {avatar_id}", flush=True)
            return self.avatar_cache[avatar_id]

        if not self.loaded:
            result = self.load_models()
            if not self.loaded:
                raise ValueError(f"Failed to load models: {result}")

        if not os.path.exists(image_path):
            raise ValueError(f"Avatar image not found: {image_path}")

        print(f"[AVATAR] Preparing avatar: {avatar_id}", flush=True)

        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Use OpenCV face detection (avoids mmpose/dwpose PyTorch 2.6 issues)
        bbox = self._detect_face_opencv(frame)
        if bbox is None:
            raise ValueError("No face detected in avatar image")

        x1, y1, x2, y2 = bbox
        print(f"[AVATAR] Face detected at: ({x1}, {y1}, {x2}, {y2})", flush=True)

        # Crop and resize face
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        # Get VAE latents
        latents = self.vae.get_latents_for_unet(crop_frame)

        # Cache avatar data
        self.avatar_cache[avatar_id] = {
            "frame": frame,
            "bbox": bbox,
            "latents": latents,
            "crop_frame": crop_frame,
            "image_path": image_path
        }

        self.current_avatar_id = avatar_id
        print(f"[AVATAR] Avatar prepared: {avatar_id}", flush=True)
        return self.avatar_cache[avatar_id]

    @torch.no_grad()
    def generate_video(self, audio_path: str, avatar_id: Optional[str] = None) -> str:
        """Generate lip-synced video from audio"""
        if not self.loaded:
            self.load_models()

        # Use current avatar if not specified
        if avatar_id is None:
            avatar_id = self.current_avatar_id

        if avatar_id is None or avatar_id not in self.avatar_cache:
            # Auto-prepare avatar if not cached
            print(f"[AVATAR] Auto-preparing avatar: {avatar_id}", flush=True)
            image_path = get_avatar_image_path(avatar_id)
            if not image_path:
                raise ValueError(f"No image found for avatar: {avatar_id}")
            self.prepare_avatar(image_path, avatar_id)

        avatar_data = self.avatar_cache[avatar_id]
        print(f"[AVATAR] Generating video for: {os.path.basename(audio_path)}", flush=True)

        from musetalk.utils.utils import datagen

        # Extract audio features
        whisper_features, librosa_len = self.audio_processor.get_audio_feature(audio_path)
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_len,
            fps=25,
            audio_padding_length_left=2,
            audio_padding_length_right=2
        )

        video_num = len(whisper_chunks)
        print(f"[AVATAR] Audio frames: {video_num}", flush=True)

        # Prepare latents (repeat for all frames)
        input_latent_list = [avatar_data["latents"]] * video_num

        # Generate frames in batches
        batch_size = 8
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list,
            batch_size=batch_size,
            delay_frame=0,
            device=self.device
        )

        res_frames = []
        for whisper_batch, latent_batch in gen:
            audio_feature = self.pe(whisper_batch)
            # Ensure consistent dtypes
            audio_feature = audio_feature.to(dtype=self.weight_dtype)
            latent_batch = latent_batch.to(dtype=self.weight_dtype)

            pred_latents = self.unet.model(
                latent_batch,
                self.timesteps,
                encoder_hidden_states=audio_feature
            ).sample

            recon = self.vae.decode_latents(pred_latents)
            res_frames.extend(recon)

        print(f"[AVATAR] Generated {len(res_frames)} frames", flush=True)

        # Blend frames back to full image
        bbox = avatar_data["bbox"]
        ori_frame = avatar_data["frame"]  # This is BGR from cv2.imread
        x1, y1, x2, y2 = bbox

        # Simple direct blending
        face_h, face_w = y2 - y1, x2 - x1
        output_frames = []

        for res_frame in res_frames:
            frame = ori_frame.copy()
            res_frame_bgr = cv2.cvtColor(res_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            res_frame_resized = cv2.resize(
                res_frame_bgr,
                (face_w, face_h),
                interpolation=cv2.INTER_LANCZOS4
            )
            # Direct paste
            frame[y1:y2, x1:x2] = res_frame_resized
            output_frames.append(frame)

        # Write video with audio
        output_path = tempfile.mktemp(suffix=".mp4")
        temp_video = tempfile.mktemp(suffix="_noaudio.mp4")

        # Write frames to video (frames are already BGR, write directly)
        height, width = output_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, 25, (width, height))

        for frame in output_frames:
            out.write(frame)  # Already BGR, no conversion needed
        out.release()

        # Add audio with ffmpeg (high quality encoding)
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", temp_video,
                "-i", audio_path,
                "-c:v", "libx264",
                "-crf", "18",  # High quality (lower = better, 18 is visually lossless)
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_path
            ], check=True, capture_output=True)

            os.unlink(temp_video)
            print(f"[AVATAR] Video saved: {output_path}", flush=True)
            return output_path

        except subprocess.CalledProcessError as e:
            print(f"[AVATAR] FFmpeg error: {e}", flush=True)
            # Return video without audio as fallback
            os.rename(temp_video, output_path)
            return output_path

    @torch.no_grad()
    def generate_video_from_animation(self, audio_path: str, video_path: str) -> str:
        """Generate lip-synced video from animated source video"""
        if not self.loaded:
            self.load_models()

        print(f"[AVATAR] Processing animated video with lip-sync...", flush=True)

        from musetalk.utils.utils import datagen

        # Read source video frames
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        source_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            source_frames.append(frame)
        cap.release()
        print(f"[AVATAR] Source video: {len(source_frames)} frames at {fps} fps", flush=True)

        if len(source_frames) == 0:
            raise ValueError("No frames in source video")

        # Extract audio features
        whisper_features, librosa_len = self.audio_processor.get_audio_feature(audio_path)
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_len,
            fps=int(fps),
            audio_padding_length_left=2,
            audio_padding_length_right=2
        )

        video_num = len(whisper_chunks)
        print(f"[AVATAR] Audio frames needed: {video_num}", flush=True)

        # Detect face in first frame for consistent bbox
        first_frame = source_frames[0]
        bbox = self._detect_face_opencv(first_frame)
        if bbox is None:
            raise ValueError("No face detected in video")

        x1, y1, x2, y2 = bbox
        print(f"[AVATAR] Face bbox: ({x1},{y1},{x2},{y2})", flush=True)

        # Upscale source frames 4x for better quality input to MuseTalk
        print(f"[AVATAR] Upscaling source frames 4x for better quality...", flush=True)
        upscale_factor = 4
        upscaled_frames = []
        for frame in source_frames:
            h, w = frame.shape[:2]
            upscaled = cv2.resize(frame, (w * upscale_factor, h * upscale_factor),
                                  interpolation=cv2.INTER_LANCZOS4)
            upscaled_frames.append(upscaled)

        # Adjust bbox for upscaled frames
        ux1, uy1, ux2, uy2 = x1 * upscale_factor, y1 * upscale_factor, x2 * upscale_factor, y2 * upscale_factor

        # Process each frame - get latents for each source frame
        # Loop source frames to match audio length
        frame_latents = []
        for i in range(video_num):
            src_idx = i % len(upscaled_frames)
            frame = upscaled_frames[src_idx]
            crop = frame[uy1:uy2, ux1:ux2]
            # Downscale to 256x256 but from 4x higher res source (preserves more detail)
            crop_resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
            latent = self.vae.get_latents_for_unet(crop_resized)
            frame_latents.append(latent)

        print(f"[AVATAR] Encoded {len(frame_latents)} frame latents", flush=True)

        # Generate lip-synced frames
        batch_size = 8
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=frame_latents,
            batch_size=batch_size,
            delay_frame=0,
            device=self.device
        )

        res_frames = []
        for whisper_batch, latent_batch in gen:
            audio_feature = self.pe(whisper_batch)
            audio_feature = audio_feature.to(dtype=self.weight_dtype)
            latent_batch = latent_batch.to(dtype=self.weight_dtype)

            pred_latents = self.unet.model(
                latent_batch,
                self.timesteps,
                encoder_hidden_states=audio_feature
            ).sample

            recon = self.vae.decode_latents(pred_latents)
            res_frames.extend(recon)

        print(f"[AVATAR] Generated {len(res_frames)} lip-synced frames", flush=True)

        # Simple direct blending
        face_h, face_w = y2 - y1, x2 - x1

        output_frames = []
        for i, res_frame in enumerate(res_frames):
            src_idx = i % len(source_frames)
            frame = source_frames[src_idx].copy()

            # Convert and resize generated face
            res_frame_bgr = cv2.cvtColor(res_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            res_frame_resized = cv2.resize(
                res_frame_bgr,
                (face_w, face_h),
                interpolation=cv2.INTER_LANCZOS4
            )

            # Apply unsharp mask for sharpening (as recommended)
            gaussian = cv2.GaussianBlur(res_frame_resized, (0, 0), 3)
            res_frame_sharp = cv2.addWeighted(res_frame_resized, 1.5, gaussian, -0.5, 0)
            res_frame_sharp = np.clip(res_frame_sharp, 0, 255).astype(np.uint8)

            frame[y1:y2, x1:x2] = res_frame_sharp
            output_frames.append(frame)

        # Write video
        output_path = tempfile.mktemp(suffix=".mp4")
        temp_video = tempfile.mktemp(suffix="_noaudio.mp4")

        height, width = output_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, int(fps), (width, height))

        for frame in output_frames:
            out.write(frame)
        out.release()

        # Add audio
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", temp_video,
                "-i", audio_path,
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_path
            ], check=True, capture_output=True)

            os.unlink(temp_video)
            print(f"[AVATAR] Animated lip-sync video saved: {output_path}", flush=True)
            return output_path

        except subprocess.CalledProcessError as e:
            print(f"[AVATAR] FFmpeg error: {e}", flush=True)
            os.rename(temp_video, output_path)
            return output_path


def get_engine() -> MuseTalkEngine:
    """Get or create the global MuseTalk engine"""
    global _engine
    if _engine is None:
        _engine = MuseTalkEngine()
    return _engine


def list_avatar_outfits() -> list:
    """List available avatar outfit directories"""
    if not os.path.exists(AVATAR_DIR):
        return []
    return sorted([
        d for d in os.listdir(AVATAR_DIR)
        if os.path.isdir(os.path.join(AVATAR_DIR, d))
    ])


def get_avatar_image_path(outfit: str, pose: str = "front_standing_friendly") -> str:
    """Get path to avatar image for given outfit and pose"""
    # Try different file extensions
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(AVATAR_DIR, outfit, f"{pose}{ext}")
        if os.path.exists(path):
            return path

    # Fall back to first image in directory
    outfit_dir = os.path.join(AVATAR_DIR, outfit)
    if os.path.exists(outfit_dir):
        for f in os.listdir(outfit_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                return os.path.join(outfit_dir, f)

    return ""


def load_avatar_models() -> str:
    """Load MuseTalk models"""
    return get_engine().load_models()


def unload_avatar_models() -> str:
    """Unload MuseTalk models"""
    return get_engine().unload_models()


def prepare_avatar(outfit: str, pose: str = "front_standing_friendly") -> str:
    """Prepare avatar for given outfit"""
    image_path = get_avatar_image_path(outfit, pose)
    if not image_path:
        return f"No image found for outfit: {outfit}"

    try:
        get_engine().prepare_avatar(image_path, outfit)
        return f"Avatar ready: {outfit}"
    except Exception as e:
        return f"Error: {e}"


def generate_wav2lip_video(image_path: str, audio_path: str) -> Optional[str]:
    """Generate lip-synced video using Wav2Lip (512x512 output, no blur)"""
    WAV2LIP_DIR = "/workspace/host/Wav2Lip"

    if not os.path.exists(WAV2LIP_DIR):
        print("[AVATAR] Wav2Lip not available", flush=True)
        return None

    checkpoint_path = f"{WAV2LIP_DIR}/checkpoints/wav2lip_gan.pth"
    if not os.path.exists(checkpoint_path):
        print("[AVATAR] Wav2Lip checkpoint not found", flush=True)
        return None

    output_path = tempfile.mktemp(suffix=".mp4")

    try:
        print(f"[AVATAR] Running Wav2Lip...", flush=True)
        result = subprocess.run([
            "python3", f"{WAV2LIP_DIR}/inference.py",
            "--checkpoint_path", checkpoint_path,
            "--face", image_path,
            "--audio", audio_path,
            "--outfile", output_path,
        ], capture_output=True, text=True, timeout=300, cwd=WAV2LIP_DIR)

        if result.returncode != 0:
            print(f"[AVATAR] Wav2Lip error: {result.stderr[:500]}", flush=True)
            return None

        if os.path.exists(output_path):
            print(f"[AVATAR] Wav2Lip video: {output_path}", flush=True)
            return output_path

        print("[AVATAR] Wav2Lip: No video generated", flush=True)
        return None

    except subprocess.TimeoutExpired:
        print("[AVATAR] Wav2Lip timeout", flush=True)
        return None
    except Exception as e:
        print(f"[AVATAR] Wav2Lip error: {e}", flush=True)
        return None


def generate_avatar_video(audio_path: str, outfit: Optional[str] = None, use_lipsync: bool = True) -> Optional[str]:
    """Generate avatar video from audio path

    use_lipsync=True (default): Use Wav2Lip for lip-sync (512x512, sharp)
    use_lipsync=False: Loop original animated video with audio (no lip-sync)
    """
    idle_video = "/workspace/ai-clone-chat/avatar_idle.mp4"
    frame_image = "/workspace/ai-clone-chat/avatar_frame.png"

    if use_lipsync:
        # Try Wav2Lip (512x512 output, no blur)
        if os.path.exists(frame_image):
            result = generate_wav2lip_video(frame_image, audio_path)
            if result:
                return result
            print("[AVATAR] Wav2Lip failed, falling back to animation", flush=True)

    # Fallback: Just loop the animated video with audio overlay (no lip-sync)
    try:
        if not os.path.exists(idle_video):
            print(f"[AVATAR] Idle video not found: {idle_video}", flush=True)
            return None

        output_path = tempfile.mktemp(suffix=".mp4")

        # Get audio duration
        probe_result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True
        )
        audio_duration = float(probe_result.stdout.strip())

        # Loop video to match audio length and overlay audio
        subprocess.run([
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", idle_video,
            "-i", audio_path,
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "192k",
            "-t", str(audio_duration),
            "-shortest",
            output_path
        ], check=True, capture_output=True)

        print(f"[AVATAR] Animated video saved (no lip-sync): {output_path}", flush=True)
        return output_path

    except Exception as e:
        print(f"[AVATAR] Error creating video: {e}", flush=True)
        return None


def get_avatar_status() -> str:
    """Get current avatar status"""
    engine = get_engine()
    if not MUSETALK_AVAILABLE:
        return "MuseTalk not installed"

    status = []
    status.append(f"Models: {'Loaded' if engine.loaded else 'Not loaded'}")
    status.append(f"Cached avatars: {len(engine.avatar_cache)}")
    if engine.current_avatar_id:
        status.append(f"Current: {engine.current_avatar_id}")

    return " | ".join(status)
