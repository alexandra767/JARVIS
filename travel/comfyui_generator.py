"""
Alexandra AI - ComfyUI Integration
Generate Midjourney-quality images with FLUX and AI videos with Wan
"""

import os
import sys
import json
import time
import uuid
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ============== CONFIGURATION ==============

class ComfyUIConfig:
    """ComfyUI configuration"""

    HOME = Path("/home/alexandratitus767")
    COMFYUI_DIR = HOME / "ComfyUI"

    # API settings
    SERVER_ADDRESS = "127.0.0.1:8188"

    # Model paths (relative to ComfyUI models folder)
    FLUX_MODEL = "flux1-dev.safetensors"
    REALISTIC_MODEL = "realisticVisionV51.safetensors"

    # Wan video models
    WAN_T2V_MODEL = "Wan2_1-T2V-14B_fp8_e4m3fn.safetensors"
    WAN_I2V_MODEL = "Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors"

    # Output
    OUTPUT_DIR = HOME / "ai-clone-chat" / "travel" / "generated"

    @classmethod
    def ensure_dirs(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============== COMFYUI API CLIENT ==============

class ComfyUIClient:
    """Client for ComfyUI API"""

    def __init__(self, server_address: str = None):
        self.server_address = server_address or ComfyUIConfig.SERVER_ADDRESS
        self.client_id = str(uuid.uuid4())

    def is_running(self) -> bool:
        """Check if ComfyUI server is running"""
        try:
            urllib.request.urlopen(f"http://{self.server_address}/system_stats", timeout=2)
            return True
        except:
            return False

    def queue_prompt(self, prompt: dict) -> str:
        """Queue a prompt for execution, returns prompt_id"""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')

        req = urllib.request.Request(
            f"http://{self.server_address}/prompt",
            data=data,
            headers={'Content-Type': 'application/json'}
        )

        response = urllib.request.urlopen(req)
        return json.loads(response.read())['prompt_id']

    def get_history(self, prompt_id: str) -> dict:
        """Get execution history for a prompt"""
        response = urllib.request.urlopen(
            f"http://{self.server_address}/history/{prompt_id}"
        )
        return json.loads(response.read())

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Get generated image from ComfyUI"""
        params = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        })

        response = urllib.request.urlopen(
            f"http://{self.server_address}/view?{params}"
        )
        return response.read()

    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> dict:
        """Wait for prompt to complete and return outputs"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)

            if prompt_id in history:
                return history[prompt_id]

            time.sleep(1)

        raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")

    def generate_and_save(self, prompt: dict, output_path: str) -> bool:
        """Generate image and save to path"""
        try:
            prompt_id = self.queue_prompt(prompt)
            print(f"  Queued prompt: {prompt_id[:8]}...")

            result = self.wait_for_completion(prompt_id)

            # Find output images
            outputs = result.get('outputs', {})
            for node_id, node_output in outputs.items():
                if 'images' in node_output:
                    for img_info in node_output['images']:
                        img_data = self.get_image(
                            img_info['filename'],
                            img_info.get('subfolder', ''),
                            img_info.get('type', 'output')
                        )

                        with open(output_path, 'wb') as f:
                            f.write(img_data)

                        print(f"  Saved: {output_path}")
                        return True

            return False

        except Exception as e:
            print(f"  Error: {e}")
            return False


# ============== FLUX IMAGE GENERATOR ==============

class FLUXGenerator:
    """Generate high-quality images with FLUX"""

    def __init__(self):
        ComfyUIConfig.ensure_dirs()
        self.client = ComfyUIClient()

    def _build_flux_workflow(self, prompt: str, negative_prompt: str = "",
                             width: int = 1080, height: int = 1920,
                             steps: int = 20, cfg: float = 3.5,
                             seed: int = None) -> dict:
        """Build FLUX workflow for ComfyUI"""

        if seed is None:
            seed = int(time.time() * 1000) % (2**32)

        # FLUX Dev workflow
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg,
                    "denoise": 1,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "seed": seed,
                    "steps": steps
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": ComfyUIConfig.FLUX_MODEL
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "batch_size": 1,
                    "height": height,
                    "width": width
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": prompt
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": negative_prompt
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                }
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "flux_travel",
                    "images": ["8", 0]
                }
            }
        }

        return workflow

    def generate_travel_image(self, destination: str, scene_type: str = "landscape",
                             output_path: str = None, style: str = "photo") -> Optional[str]:
        """Generate a travel image for a destination"""

        # Build detailed prompt based on scene type
        scene_prompts = {
            "landscape": f"stunning panoramic landscape photograph of {destination}, golden hour lighting, dramatic sky, professional travel photography, 8k, ultra detailed, vibrant colors",
            "city": f"beautiful cityscape of {destination}, urban photography, architectural details, street life, golden hour, professional photo, 8k quality",
            "food": f"delicious local cuisine from {destination}, food photography, appetizing, professional lighting, restaurant setting, 8k, mouth-watering",
            "culture": f"cultural scene in {destination}, local traditions, authentic atmosphere, documentary style photography, vibrant, 8k detailed",
            "beach": f"pristine beach in {destination}, crystal clear turquoise water, white sand, tropical paradise, travel photography, 8k, stunning",
            "nature": f"breathtaking natural scenery of {destination}, lush vegetation, dramatic lighting, national geographic style, 8k ultra detailed",
            "nightlife": f"vibrant nightlife scene in {destination}, city lights, energetic atmosphere, neon colors, urban night photography, 8k",
            "historic": f"historic landmark in {destination}, ancient architecture, dramatic lighting, travel documentary style, 8k detailed photography",
        }

        style_modifiers = {
            "photo": ", photorealistic, professional DSLR photography, sharp focus",
            "cinematic": ", cinematic composition, film grain, movie still, anamorphic",
            "artistic": ", artistic interpretation, painterly style, vibrant artistic",
            "minimal": ", minimalist composition, clean aesthetic, modern",
        }

        prompt = scene_prompts.get(scene_type, scene_prompts["landscape"])
        prompt += style_modifiers.get(style, style_modifiers["photo"])

        negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, watermark, text, logo, oversaturated, cartoon, anime, illustration, painting, drawing, art, sketch"

        # Set output path
        if not output_path:
            timestamp = int(time.time())
            output_path = str(ComfyUIConfig.OUTPUT_DIR / f"travel_{destination.replace(' ', '_')}_{scene_type}_{timestamp}.png")

        print(f"[FLUX] Generating {scene_type} image of {destination}...")

        # Check if ComfyUI is running
        if not self.client.is_running():
            print("[FLUX] ComfyUI not running! Start it with: python main.py --listen")
            return None

        # Build and execute workflow
        workflow = self._build_flux_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=1080,
            height=1920,  # Vertical for shorts/TikTok
            steps=20,
            cfg=3.5
        )

        if self.client.generate_and_save(workflow, output_path):
            return output_path

        return None

    def generate_batch(self, destination: str, count: int = 5,
                      scene_types: List[str] = None) -> List[str]:
        """Generate multiple images for a destination"""

        if scene_types is None:
            scene_types = ["landscape", "city", "food", "culture", "nature"]

        generated = []
        for i, scene_type in enumerate(scene_types[:count]):
            print(f"\n[FLUX] Image {i+1}/{count}: {scene_type}")
            path = self.generate_travel_image(destination, scene_type)
            if path:
                generated.append(path)

        return generated

    def generate_avatar(self, preset_name: str, output_path: str = None,
                       seed: int = None) -> Optional[str]:
        """Generate an avatar image using a preset"""

        preset = AvatarPresets.get_preset(preset_name)
        if not preset:
            print(f"[FLUX] Unknown preset: {preset_name}")
            print(f"[FLUX] Available: {', '.join(AvatarPresets.list_presets())}")
            return None

        if not output_path:
            timestamp = int(time.time())
            output_path = str(ComfyUIConfig.OUTPUT_DIR / f"avatar_{preset_name}_{timestamp}.png")

        print(f"[FLUX] Generating avatar: {preset_name}")
        print(f"[FLUX] {preset['description']}")

        if not self.client.is_running():
            print("[FLUX] ComfyUI not running! Start it with: python main.py --listen")
            return None

        workflow = self._build_flux_workflow(
            prompt=preset["prompt"],
            negative_prompt=preset["negative"],
            width=1080,
            height=1440,  # Portrait orientation for avatar
            steps=25,  # More steps for quality
            cfg=3.5,
            seed=seed
        )

        if self.client.generate_and_save(workflow, output_path):
            return output_path

        return None

    def generate_avatar_batch(self, preset_name: str, count: int = 4) -> List[str]:
        """Generate multiple variations of an avatar"""

        generated = []
        for i in range(count):
            print(f"\n[FLUX] Avatar variation {i+1}/{count}")
            path = self.generate_avatar(preset_name)
            if path:
                generated.append(path)

        return generated

    @staticmethod
    def list_avatar_presets():
        """List available avatar presets"""
        print("\nAvailable Avatar Presets:")
        print("=" * 50)
        for name, desc in AvatarPresets.describe_presets().items():
            print(f"  {name}")
            print(f"    → {desc}\n")


# ============== WAN VIDEO GENERATOR ==============

class WanVideoGenerator:
    """Generate AI videos with Wan 2.1/2.2"""

    def __init__(self):
        ComfyUIConfig.ensure_dirs()
        self.client = ComfyUIClient()

    def _build_wan_t2v_workflow(self, prompt: str, negative_prompt: str = "",
                                 width: int = 720, height: int = 1280,
                                 frames: int = 81, steps: int = 20,
                                 seed: int = None) -> dict:
        """Build Wan Text-to-Video workflow"""

        if seed is None:
            seed = int(time.time() * 1000) % (2**32)

        # Simplified Wan T2V workflow
        workflow = {
            "1": {
                "class_type": "WanVideoModelLoader",
                "inputs": {
                    "model_name": ComfyUIConfig.WAN_T2V_MODEL
                }
            },
            "2": {
                "class_type": "WanVideoTextEncode",
                "inputs": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "model": ["1", 0]
                }
            },
            "3": {
                "class_type": "WanVideoSampler",
                "inputs": {
                    "model": ["1", 0],
                    "conditioning": ["2", 0],
                    "width": width,
                    "height": height,
                    "frames": frames,
                    "steps": steps,
                    "seed": seed,
                    "cfg": 7.0
                }
            },
            "4": {
                "class_type": "WanVideoDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "model": ["1", 0]
                }
            },
            "5": {
                "class_type": "SaveAnimatedWEBP",
                "inputs": {
                    "filename_prefix": "wan_video",
                    "images": ["4", 0],
                    "fps": 16,
                    "lossless": False,
                    "quality": 85
                }
            }
        }

        return workflow

    def _build_wan_i2v_workflow(self, image_path: str, prompt: str,
                                 frames: int = 81, steps: int = 20,
                                 seed: int = None) -> dict:
        """Build Wan Image-to-Video workflow"""

        if seed is None:
            seed = int(time.time() * 1000) % (2**32)

        workflow = {
            "1": {
                "class_type": "WanVideoModelLoader",
                "inputs": {
                    "model_name": ComfyUIConfig.WAN_I2V_MODEL
                }
            },
            "2": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": image_path
                }
            },
            "3": {
                "class_type": "WanVideoI2VEncode",
                "inputs": {
                    "image": ["2", 0],
                    "prompt": prompt,
                    "model": ["1", 0]
                }
            },
            "4": {
                "class_type": "WanVideoSampler",
                "inputs": {
                    "model": ["1", 0],
                    "conditioning": ["3", 0],
                    "frames": frames,
                    "steps": steps,
                    "seed": seed,
                    "cfg": 7.0
                }
            },
            "5": {
                "class_type": "WanVideoDecode",
                "inputs": {
                    "samples": ["4", 0],
                    "model": ["1", 0]
                }
            },
            "6": {
                "class_type": "SaveAnimatedWEBP",
                "inputs": {
                    "filename_prefix": "wan_i2v",
                    "images": ["5", 0],
                    "fps": 16
                }
            }
        }

        return workflow

    def generate_video_from_text(self, prompt: str, output_path: str = None,
                                  duration_seconds: float = 5) -> Optional[str]:
        """Generate video from text prompt"""

        frames = int(duration_seconds * 16)  # 16 fps

        if not output_path:
            timestamp = int(time.time())
            output_path = str(ComfyUIConfig.OUTPUT_DIR / f"wan_t2v_{timestamp}.webp")

        print(f"[Wan T2V] Generating {duration_seconds}s video...")

        if not self.client.is_running():
            print("[Wan] ComfyUI not running!")
            return None

        workflow = self._build_wan_t2v_workflow(
            prompt=prompt,
            frames=frames,
            steps=20
        )

        if self.client.generate_and_save(workflow, output_path):
            return output_path

        return None

    def animate_image(self, image_path: str, prompt: str = "",
                     output_path: str = None, duration_seconds: float = 5) -> Optional[str]:
        """Animate a still image to video"""

        frames = int(duration_seconds * 16)

        if not output_path:
            timestamp = int(time.time())
            output_path = str(ComfyUIConfig.OUTPUT_DIR / f"wan_i2v_{timestamp}.webp")

        print(f"[Wan I2V] Animating image to {duration_seconds}s video...")

        if not self.client.is_running():
            print("[Wan] ComfyUI not running!")
            return None

        workflow = self._build_wan_i2v_workflow(
            image_path=image_path,
            prompt=prompt or "smooth camera motion, cinematic movement",
            frames=frames,
            steps=20
        )

        if self.client.generate_and_save(workflow, output_path):
            return output_path

        return None


# ============== AVATAR PRESETS ==============

class AvatarPresets:
    """Pre-built prompts for consistent avatar/host images"""

    PRESETS = {
        "blonde_office_casual": {
            "prompt": """Friendly woman with blonde styled voluminous waves sitting behind a modern desk,
looking directly at camera with warm approachable smile,
natural makeup with soft smokey eyeshadow, subtle winged eyeliner,
nude pink lipstick, defined eyebrows,
flawless natural skin with realistic texture,
wearing casual chic blouse or cozy sweater,
modern home office background, plants, bookshelves, soft decor,
warm natural window lighting, golden hour tones,
shallow depth of field, subtle bokeh,
upper body portrait, centered composition, eye contact with viewer,
lifestyle photography style, authentic and relatable,
ultra-high detail, realistic skin texture, photorealistic,
shot on Canon EOS R5 with 85mm f/1.4 lens,
8k resolution, cinematic soft lighting""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing",
            "description": "Casual blonde woman at desk - friendly content creator vibe"
        },

        "blonde_office_professional": {
            "prompt": """Professional woman with blonde Hollywood waves sitting behind executive desk in luxurious office,
looking directly at camera with confident welcoming expression,
professional glamour makeup, smokey eyeshadow, winged eyeliner, classic red lipstick, defined eyebrows,
flawless natural skin with realistic texture,
wearing elegant professional blazer,
modern office background with large windows, city view, soft natural lighting,
warm golden-hour tones from window, shallow depth of field, subtle bokeh,
upper body portrait framing, centered composition,
high-fashion editorial photography style, ultra-high detail,
realistic skin texture, photorealistic,
shot on Canon EOS R5 with 85mm f/1.4 lens,
cinematic lighting, 8k resolution""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing",
            "description": "Professional blonde executive - corporate/business vibe"
        },

        "blonde_creator_studio": {
            "prompt": """Friendly blonde woman content creator sitting at modern desk with ring light,
looking directly at camera with bright engaging smile,
natural glowing makeup, soft eyeshadow, glossy lips,
flawless skin with realistic texture, blonde voluminous styled hair,
wearing casual white t-shirt or trendy top,
YouTube studio setup background, minimalist aesthetic, professional microphone visible,
ring light reflection in eyes, soft diffused lighting,
upper body portrait, centered, direct eye contact,
lifestyle influencer photography, authentic and relatable,
ultra-high detail, photorealistic, 8k resolution,
shot on Sony A7IV with 50mm f/1.4 lens""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing",
            "description": "Content creator studio setup - YouTube/TikTok vibe"
        },

        "blonde_travel_host": {
            "prompt": """Friendly blonde woman travel host sitting casually at desk,
looking at camera with warm adventurous smile,
natural sun-kissed makeup, light bronzer, nude pink lips,
blonde beach waves hair, flawless natural skin texture,
wearing light flowy blouse or casual linen shirt,
travel-themed background with world map, travel photos, plants, passport visible,
warm natural lighting, golden hour glow, wanderlust aesthetic,
upper body portrait, welcoming composition, eye contact,
travel lifestyle photography, authentic explorer vibe,
ultra-high detail, photorealistic, 8k resolution,
shot on Canon EOS R5 with 85mm f/1.4 lens""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing",
            "description": "Travel content host - wanderlust/explorer vibe"
        },

        "blonde_cozy_casual": {
            "prompt": """Warm friendly blonde woman sitting at cozy home desk,
looking at camera with genuine relaxed smile,
minimal natural makeup, soft rosy cheeks, nude lips,
blonde soft waves, flawless natural skin with realistic texture,
wearing cozy oversized sweater or comfortable casual top,
cozy home office with warm lighting, coffee cup on desk, plants, fairy lights,
soft warm window light, hygge aesthetic, shallow depth of field,
upper body portrait, intimate welcoming composition,
lifestyle photography, authentic and approachable,
ultra-high detail, photorealistic, 8k resolution""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing",
            "description": "Cozy casual at home - relaxed friendly vibe"
        },

        # ============== BRUNETTE PRESETS ==============

        "brunette_office_casual": {
            "prompt": """Friendly woman with brunette styled voluminous waves sitting behind a modern desk,
looking directly at camera with warm approachable smile,
natural makeup with soft smokey eyeshadow, subtle winged eyeliner,
nude pink lipstick, defined eyebrows,
flawless natural skin with realistic texture,
wearing casual chic blouse or cozy sweater,
modern home office background, plants, bookshelves, soft decor,
warm natural window lighting, golden hour tones,
shallow depth of field, subtle bokeh,
upper body portrait, centered composition, eye contact with viewer,
lifestyle photography style, authentic and relatable,
ultra-high detail, realistic skin texture, photorealistic,
shot on Canon EOS R5 with 85mm f/1.4 lens,
8k resolution, cinematic soft lighting""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing, blonde hair",
            "description": "Casual brunette woman at desk - friendly content creator vibe"
        },

        "brunette_travel_host": {
            "prompt": """Friendly brunette woman travel host sitting casually at desk,
looking at camera with warm adventurous smile,
natural sun-kissed makeup, light bronzer, nude pink lips,
brunette beach waves hair, flawless natural skin texture,
wearing light flowy blouse or casual linen shirt,
travel-themed background with world map, travel photos, plants, passport visible,
warm natural lighting, golden hour glow, wanderlust aesthetic,
upper body portrait, welcoming composition, eye contact,
travel lifestyle photography, authentic explorer vibe,
ultra-high detail, photorealistic, 8k resolution,
shot on Canon EOS R5 with 85mm f/1.4 lens""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing, blonde hair",
            "description": "Brunette travel content host - wanderlust/explorer vibe"
        },

        "brunette_professional": {
            "prompt": """Professional woman with brunette Hollywood waves sitting behind executive desk in luxurious office,
looking directly at camera with confident welcoming expression,
professional glamour makeup, smokey eyeshadow, winged eyeliner, classic red lipstick, defined eyebrows,
flawless natural skin with realistic texture,
wearing elegant professional blazer,
modern office background with large windows, city view, soft natural lighting,
warm golden-hour tones from window, shallow depth of field, subtle bokeh,
upper body portrait framing, centered composition,
high-fashion editorial photography style, ultra-high detail,
realistic skin texture, photorealistic,
shot on Canon EOS R5 with 85mm f/1.4 lens,
cinematic lighting, 8k resolution""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing, blonde hair",
            "description": "Professional brunette executive - corporate/business vibe"
        },

        # ============== REDHEAD PRESETS ==============

        "redhead_office_casual": {
            "prompt": """Friendly woman with red auburn styled voluminous waves sitting behind a modern desk,
looking directly at camera with warm approachable smile,
natural makeup with soft earthy eyeshadow, subtle winged eyeliner,
nude coral lipstick, defined eyebrows,
flawless natural skin with light freckles, realistic texture,
wearing casual chic blouse or cozy sweater,
modern home office background, plants, bookshelves, soft decor,
warm natural window lighting, golden hour tones,
shallow depth of field, subtle bokeh,
upper body portrait, centered composition, eye contact with viewer,
lifestyle photography style, authentic and relatable,
ultra-high detail, realistic skin texture, photorealistic,
shot on Canon EOS R5 with 85mm f/1.4 lens,
8k resolution, cinematic soft lighting""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing, blonde hair, brown hair",
            "description": "Casual redhead woman at desk - friendly content creator vibe"
        },

        "redhead_travel_host": {
            "prompt": """Friendly redhead woman travel host sitting casually at desk,
looking at camera with warm adventurous smile,
natural sun-kissed makeup, light bronzer, coral lips,
red auburn waves hair, flawless natural skin with light freckles,
wearing light flowy blouse or casual linen shirt,
travel-themed background with world map, travel photos, plants, passport visible,
warm natural lighting, golden hour glow, wanderlust aesthetic,
upper body portrait, welcoming composition, eye contact,
travel lifestyle photography, authentic explorer vibe,
ultra-high detail, photorealistic, 8k resolution,
shot on Canon EOS R5 with 85mm f/1.4 lens""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing, blonde hair, brown hair",
            "description": "Redhead travel content host - wanderlust/explorer vibe"
        },

        # ============== BLACK HAIR PRESETS ==============

        "blackhair_office_casual": {
            "prompt": """Friendly woman with black styled voluminous waves sitting behind a modern desk,
looking directly at camera with warm approachable smile,
natural makeup with soft smokey eyeshadow, subtle winged eyeliner,
nude mauve lipstick, defined eyebrows,
flawless natural skin with realistic texture,
wearing casual chic blouse or cozy sweater,
modern home office background, plants, bookshelves, soft decor,
warm natural window lighting, golden hour tones,
shallow depth of field, subtle bokeh,
upper body portrait, centered composition, eye contact with viewer,
lifestyle photography style, authentic and relatable,
ultra-high detail, realistic skin texture, photorealistic,
shot on Canon EOS R5 with 85mm f/1.4 lens,
8k resolution, cinematic soft lighting""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing, blonde hair, red hair",
            "description": "Casual black-haired woman at desk - friendly content creator vibe"
        },

        "blackhair_professional": {
            "prompt": """Professional woman with sleek black hair sitting behind executive desk in luxurious office,
looking directly at camera with confident welcoming expression,
professional glamour makeup, smokey eyeshadow, winged eyeliner, deep berry lipstick, defined eyebrows,
flawless natural skin with realistic texture,
wearing elegant professional blazer,
modern office background with large windows, city view, soft natural lighting,
warm golden-hour tones from window, shallow depth of field, subtle bokeh,
upper body portrait framing, centered composition,
high-fashion editorial photography style, ultra-high detail,
realistic skin texture, photorealistic,
shot on Canon EOS R5 with 85mm f/1.4 lens,
cinematic lighting, 8k resolution""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing, blonde hair, red hair",
            "description": "Professional black-haired executive - corporate/business vibe"
        },

        # ============== DIFFERENT SETTINGS ==============

        "beach_presenter": {
            "prompt": """Friendly blonde woman sitting at outdoor beach cafe table,
looking at camera with bright relaxed smile,
natural beachy makeup, sun-kissed glow, coral lips,
blonde beach waves hair, flawless tanned skin,
wearing light summer dress or flowy beach cover-up,
tropical beach background, palm trees, ocean view, blue sky,
bright natural sunlight, vacation paradise aesthetic,
upper body portrait, relaxed welcoming composition,
travel lifestyle photography, summer vibes,
ultra-high detail, photorealistic, 8k resolution""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing",
            "description": "Beach/vacation presenter - tropical paradise vibe"
        },

        "cafe_casual": {
            "prompt": """Friendly blonde woman sitting at cozy coffee shop table,
looking at camera with warm genuine smile,
natural minimal makeup, soft rosy cheeks, nude lips,
blonde styled waves, flawless natural skin,
wearing casual sweater or trendy casual top,
coffee shop background, exposed brick, warm lighting, coffee cup on table,
soft warm ambient lighting, cozy cafe aesthetic,
upper body portrait, intimate welcoming composition,
lifestyle photography, authentic and approachable,
ultra-high detail, photorealistic, 8k resolution""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing",
            "description": "Coffee shop casual - cozy cafe vibe"
        },

        "outdoor_adventure": {
            "prompt": """Adventurous blonde woman outdoors in nature setting,
looking at camera with excited confident smile,
natural minimal makeup, healthy glow,
blonde hair in practical ponytail or loose waves,
wearing outdoor hiking gear or casual athletic wear,
beautiful mountain or forest background, natural scenery,
bright natural daylight, adventure lifestyle aesthetic,
upper body portrait, energetic composition,
outdoor adventure photography, explorer vibes,
ultra-high detail, photorealistic, 8k resolution""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing",
            "description": "Outdoor adventure - hiking/nature explorer vibe"
        },

        "kitchen_cooking": {
            "prompt": """Friendly blonde woman in modern kitchen setting,
looking at camera with warm welcoming smile,
natural fresh makeup, healthy glow, soft pink lips,
blonde hair styled back or in loose waves,
wearing casual apron over nice top or casual cooking attire,
modern kitchen background, cooking ingredients visible, warm lighting,
bright natural kitchen lighting, home cooking aesthetic,
upper body portrait, inviting composition,
food lifestyle photography, home chef vibes,
ultra-high detail, photorealistic, 8k resolution""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing",
            "description": "Kitchen/cooking presenter - home chef vibe"
        },

        # ============== DIFFERENT STYLES ==============

        "glamour_evening": {
            "prompt": """Elegant blonde woman in sophisticated evening setting,
looking at camera with confident glamorous expression,
full glamour makeup, dramatic smokey eyes, bold red lips, contoured cheekbones,
blonde hair in elegant updo or Hollywood waves,
wearing elegant evening dress or sophisticated outfit,
luxurious background, soft dramatic lighting, bokeh lights,
golden hour or evening ambient lighting, high fashion aesthetic,
upper body portrait, glamorous composition,
high-fashion editorial photography, elegant luxury vibes,
ultra-high detail, photorealistic, 8k resolution""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing",
            "description": "Glamorous evening look - sophisticated luxury vibe"
        },

        "sporty_active": {
            "prompt": """Athletic blonde woman in fitness or sports setting,
looking at camera with energetic confident smile,
minimal natural makeup, healthy athletic glow,
blonde hair in high ponytail or athletic style,
wearing stylish athletic wear or fitness outfit,
modern gym or outdoor fitness background,
bright energetic lighting, active lifestyle aesthetic,
upper body portrait, dynamic energetic composition,
fitness lifestyle photography, healthy active vibes,
ultra-high detail, photorealistic, 8k resolution""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing",
            "description": "Sporty athletic look - fitness/active lifestyle vibe"
        },

        "bohemian_creative": {
            "prompt": """Creative blonde woman with artistic bohemian style,
looking at camera with thoughtful creative expression,
natural artistic makeup, earthy tones, soft berry lips,
blonde hair in loose natural waves with subtle braids or accessories,
wearing bohemian style clothing, flowy fabrics, artistic jewelry,
creative studio or artistic background, plants, artwork visible,
soft natural lighting with warm tones, artistic aesthetic,
upper body portrait, creative artistic composition,
lifestyle editorial photography, bohemian creative vibes,
ultra-high detail, photorealistic, 8k resolution""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing",
            "description": "Bohemian artistic look - creative/artistic vibe"
        },

        # ============== DIFFERENT SHOTS ==============

        "closeup_face": {
            "prompt": """Beautiful blonde woman extreme closeup portrait,
looking directly at camera with warm engaging expression,
perfect natural makeup, soft smokey eyeshadow, nude pink lips,
flawless skin with realistic texture and pores,
blonde styled hair framing face,
soft diffused studio lighting, clean simple background,
shallow depth of field, sharp focus on eyes,
closeup face portrait, beauty photography,
ultra-high detail, photorealistic, 8k resolution,
shot on Canon EOS R5 with 85mm f/1.4 lens""",
            "negative": "ugly, deformed, bad anatomy, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing",
            "description": "Face closeup - perfect for thumbnails"
        },

        "full_body_standing": {
            "prompt": """Beautiful blonde woman standing confidently in modern office,
full body view, weight on one leg, relaxed confident pose,
looking at camera with warm professional smile,
natural glamour makeup, nude pink lips,
blonde styled voluminous waves,
wearing professional casual outfit, blazer and pants or elegant dress,
modern office background with clean aesthetic,
soft natural lighting, professional photography,
full body portrait, centered composition,
lifestyle editorial photography, professional vibes,
ultra-high detail, photorealistic, 8k resolution""",
            "negative": "ugly, deformed, bad anatomy, bad hands, missing fingers, extra fingers, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing",
            "description": "Full body standing pose - professional/lifestyle"
        },

        "side_profile": {
            "prompt": """Beautiful blonde woman elegant side profile portrait,
looking slightly away from camera with thoughtful expression,
perfect natural makeup, defined features, soft pink lips,
blonde styled waves falling elegantly,
flawless skin with realistic texture,
soft studio lighting, clean gradient background,
shallow depth of field, artistic composition,
side profile portrait, beauty editorial photography,
ultra-high detail, photorealistic, 8k resolution,
shot on Canon EOS R5 with 85mm f/1.4 lens""",
            "negative": "ugly, deformed, bad anatomy, blurry, low quality, watermark, text, logo, cartoon, anime, illustration, painting, drawing",
            "description": "Side profile shot - artistic/editorial"
        },
    }

    @classmethod
    def get_preset(cls, name: str) -> Optional[Dict]:
        """Get a preset by name"""
        return cls.PRESETS.get(name)

    @classmethod
    def list_presets(cls) -> List[str]:
        """List all available presets"""
        return list(cls.PRESETS.keys())

    @classmethod
    def describe_presets(cls) -> Dict[str, str]:
        """Get descriptions of all presets"""
        return {name: data["description"] for name, data in cls.PRESETS.items()}


# ============== TRAVEL IMAGE PRESETS ==============

class TravelImagePresets:
    """Pre-built prompts for travel content"""

    @staticmethod
    def get_scene_prompts(destination: str) -> Dict[str, str]:
        """Get various scene prompts for a destination"""
        return {
            "hero": f"breathtaking aerial view of {destination}, stunning landscape, golden hour, professional travel photography, 8k, cinematic composition, dramatic clouds",
            "street": f"charming street scene in {destination}, local life, authentic atmosphere, morning light, travel documentary photography, 8k detailed",
            "landmark": f"iconic landmark of {destination}, dramatic angle, blue hour photography, professional architecture photo, 8k ultra detailed",
            "food": f"traditional local dish from {destination}, food photography, appetizing presentation, restaurant ambiance, professional lighting, 8k",
            "market": f"vibrant local market in {destination}, colorful produce, busy atmosphere, authentic culture, travel photography, 8k",
            "sunset": f"magnificent sunset over {destination}, golden hour, silhouettes, dramatic sky colors, professional landscape photography, 8k",
            "nature": f"pristine natural landscape near {destination}, untouched beauty, dramatic lighting, national geographic style, 8k",
            "hotel": f"luxury hotel room with view of {destination}, elegant interior, morning light, travel lifestyle photography, 8k",
            "transport": f"local transportation in {destination}, authentic travel experience, street photography style, cultural immersion, 8k",
            "people": f"friendly locals in {destination}, genuine smiles, cultural portrait, travel documentary, warm lighting, 8k",
        }

    @staticmethod
    def get_video_prompts(destination: str) -> Dict[str, str]:
        """Get prompts for video generation"""
        return {
            "aerial_pan": f"smooth aerial drone shot flying over {destination}, revealing beautiful landscape, cinematic camera movement",
            "street_walk": f"first person walking through streets of {destination}, steady cam, immersive travel experience",
            "timelapse": f"timelapse of {destination} cityscape, clouds moving, day to night transition, cinematic",
            "water": f"gentle waves on beach in {destination}, crystal clear water, relaxing ocean movement, 4k",
            "crowd": f"busy street scene in {destination}, people walking, urban life, smooth motion",
        }


# ============== CLI ==============

def main():
    import argparse

    parser = argparse.ArgumentParser(description="ComfyUI Image/Video Generator")
    parser.add_argument("destination", nargs="?", help="Travel destination or avatar preset name")
    parser.add_argument("--type", "-t", choices=["image", "video", "batch", "animate", "avatar", "avatar-batch"],
                       default="image", help="Generation type")
    parser.add_argument("--scene", "-s", default="landscape",
                       help="Scene type (landscape, city, food, etc.)")
    parser.add_argument("--preset", "-p", help="Avatar preset name")
    parser.add_argument("--count", "-n", type=int, default=5,
                       help="Number of images for batch")
    parser.add_argument("--image", "-i", help="Image path for animation")
    parser.add_argument("--output", "-o", help="Output path")
    parser.add_argument("--list-presets", action="store_true", help="List avatar presets")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    args = parser.parse_args()

    # List presets
    if args.list_presets:
        FLUXGenerator.list_avatar_presets()
        return

    # Avatar generation
    if args.type == "avatar":
        preset = args.preset or args.destination
        if not preset:
            print("Specify preset name: --preset blonde_office_casual")
            FLUXGenerator.list_avatar_presets()
            return
        gen = FLUXGenerator()
        path = gen.generate_avatar(preset, args.output, args.seed)
        if path:
            print(f"\nGenerated: {path}")

    elif args.type == "avatar-batch":
        preset = args.preset or args.destination
        if not preset:
            print("Specify preset name: --preset blonde_office_casual")
            FLUXGenerator.list_avatar_presets()
            return
        gen = FLUXGenerator()
        paths = gen.generate_avatar_batch(preset, args.count)
        print(f"\nGenerated {len(paths)} avatar variations")
        for p in paths:
            print(f"  - {p}")

    elif args.type == "image":
        if not args.destination:
            print("Specify destination: python comfyui_generator.py Barcelona --type image")
            return
        gen = FLUXGenerator()
        path = gen.generate_travel_image(args.destination, args.scene, args.output)
        if path:
            print(f"\nGenerated: {path}")

    elif args.type == "batch":
        if not args.destination:
            print("Specify destination: python comfyui_generator.py Barcelona --type batch")
            return
        gen = FLUXGenerator()
        paths = gen.generate_batch(args.destination, args.count)
        print(f"\nGenerated {len(paths)} images")
        for p in paths:
            print(f"  - {p}")

    elif args.type == "video":
        if not args.destination:
            print("Specify destination for video")
            return
        gen = WanVideoGenerator()
        prompts = TravelImagePresets.get_video_prompts(args.destination)
        prompt = prompts.get("aerial_pan", f"beautiful scene of {args.destination}")
        path = gen.generate_video_from_text(prompt, args.output)
        if path:
            print(f"\nGenerated: {path}")

    elif args.type == "animate":
        if not args.image:
            print("--image required for animation")
            return
        gen = WanVideoGenerator()
        path = gen.animate_image(args.image, output_path=args.output)
        if path:
            print(f"\nGenerated: {path}")


if __name__ == "__main__":
    main()
