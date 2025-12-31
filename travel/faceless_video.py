"""
Alexandra AI - Faceless Video Generator
Create high-quality videos using voice + images/stock footage (no avatar)
Supports AI image generation with FLUX via ComfyUI
"""

import os
import sys
import json
import subprocess
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from travel_config import CONTENT_DIR

# Try to import ComfyUI generator
try:
    from comfyui_generator import FLUXGenerator, WanVideoGenerator, ComfyUIClient
    HAS_COMFYUI = True
except ImportError:
    HAS_COMFYUI = False
    print("[Faceless] ComfyUI integration not available")

# ============== CONFIGURATION ==============

class FacelessConfig:
    """Configuration for faceless video generation"""

    HOME = Path("/home/alexandratitus767")

    # Output directories
    OUTPUT_DIR = HOME / "ai-clone-chat" / "travel" / "output"
    TEMP_DIR = HOME / "ai-clone-chat" / "travel" / "temp"
    ASSETS_DIR = HOME / "ai-clone-chat" / "travel" / "assets"

    # Stock footage sources
    STOCK_DIR = ASSETS_DIR / "stock"
    IMAGES_DIR = ASSETS_DIR / "images"
    MUSIC_DIR = ASSETS_DIR / "music"

    # ComfyUI for AI image generation
    COMFYUI_DIR = HOME / "ComfyUI"
    COMFYUI_OUTPUT = COMFYUI_DIR / "output"

    # Video settings
    DEFAULT_RESOLUTION = "1080x1920"  # Vertical for shorts
    DEFAULT_FPS = 30
    DEFAULT_IMAGE_DURATION = 5  # seconds per image

    @classmethod
    def ensure_dirs(cls):
        """Create all necessary directories"""
        for d in [cls.OUTPUT_DIR, cls.TEMP_DIR, cls.ASSETS_DIR,
                  cls.STOCK_DIR, cls.IMAGES_DIR, cls.MUSIC_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# ============== STOCK FOOTAGE MANAGER ==============

class StockManager:
    """Manage stock footage and images"""

    # Free stock footage sources (for reference)
    STOCK_SOURCES = {
        "pexels": "https://www.pexels.com/search/videos/",
        "pixabay": "https://pixabay.com/videos/search/",
        "coverr": "https://coverr.co/search?q=",
    }

    def __init__(self):
        FacelessConfig.ensure_dirs()
        self.stock_dir = FacelessConfig.STOCK_DIR
        self.images_dir = FacelessConfig.IMAGES_DIR

    def list_stock(self, media_type: str = "all") -> List[Path]:
        """List available stock media"""
        files = []

        if media_type in ["all", "video"]:
            for ext in ["*.mp4", "*.mov", "*.webm"]:
                files.extend(self.stock_dir.glob(ext))

        if media_type in ["all", "image"]:
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                files.extend(self.images_dir.glob(ext))
                files.extend(self.stock_dir.glob(ext))

        return sorted(files)

    def organize_by_topic(self, topic: str) -> Dict[str, List[Path]]:
        """Find stock media related to a topic"""
        topic_lower = topic.lower().replace(" ", "_")
        topic_dir = self.stock_dir / topic_lower

        result = {"videos": [], "images": []}

        if topic_dir.exists():
            for f in topic_dir.iterdir():
                if f.suffix.lower() in [".mp4", ".mov", ".webm"]:
                    result["videos"].append(f)
                elif f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    result["images"].append(f)

        return result

    def get_placeholder_images(self, count: int, color: str = "travel") -> List[str]:
        """Generate placeholder images with travel-themed gradients"""
        placeholders = []
        colors = {
            "travel": ["#1a5276", "#2ecc71"],  # Blue to green
            "food": ["#e74c3c", "#f39c12"],    # Red to orange
            "city": ["#34495e", "#9b59b6"],    # Gray to purple
            "beach": ["#3498db", "#1abc9c"],   # Blue to teal
            "nature": ["#27ae60", "#2ecc71"],  # Greens
        }

        gradient = colors.get(color, colors["travel"])

        for i in range(count):
            placeholder = FacelessConfig.TEMP_DIR / f"placeholder_{i:03d}.png"

            # Create gradient image with FFmpeg
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"color=c={gradient[0]}:s=1080x1920:d=1",
                "-vf", f"drawtext=text='Slide {i+1}':fontsize=72:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
                "-frames:v", "1",
                str(placeholder)
            ]

            try:
                subprocess.run(cmd, capture_output=True, check=True)
                placeholders.append(str(placeholder))
            except:
                pass

        return placeholders


# ============== AI IMAGE GENERATOR ==============

class AIImageGenerator:
    """Generate travel images using FLUX via ComfyUI"""

    def __init__(self):
        FacelessConfig.ensure_dirs()
        self.flux_gen = FLUXGenerator() if HAS_COMFYUI else None
        self.wan_gen = WanVideoGenerator() if HAS_COMFYUI else None

    def is_available(self) -> bool:
        """Check if AI image generation is available"""
        if not HAS_COMFYUI:
            return False
        client = ComfyUIClient()
        return client.is_running()

    def generate_travel_images(self, destination: str, count: int = 5,
                               scenes: List[str] = None) -> List[str]:
        """Generate AI travel images for a destination"""

        if not self.is_available():
            print("[AI Images] ComfyUI not available, using placeholders")
            return []

        if scenes is None:
            # Default scenes for travel video
            scenes = ["hero", "street", "food", "landmark", "sunset"]

        generated = []
        for i, scene in enumerate(scenes[:count]):
            print(f"[AI Images] Generating {i+1}/{count}: {scene}")
            path = self.flux_gen.generate_travel_image(destination, scene)
            if path:
                generated.append(path)

        return generated

    def animate_images(self, images: List[str], destination: str = "") -> List[str]:
        """Convert static images to short video clips using Wan I2V"""

        if not self.is_available() or not self.wan_gen:
            print("[AI Video] Wan not available")
            return images  # Return original images

        animated = []
        for i, img_path in enumerate(images):
            print(f"[AI Video] Animating {i+1}/{len(images)}...")
            video_path = self.wan_gen.animate_image(
                img_path,
                prompt=f"gentle camera movement, cinematic, {destination}",
                duration_seconds=3
            )
            if video_path:
                animated.append(video_path)
            else:
                animated.append(img_path)  # Keep original if animation fails

        return animated


# ============== FACELESS VIDEO GENERATOR ==============

class FacelessVideoGenerator:
    """Generate faceless videos from audio + images"""

    def __init__(self):
        FacelessConfig.ensure_dirs()
        self.stock = StockManager()
        self.ai_gen = AIImageGenerator() if HAS_COMFYUI else None

    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds"""
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries",
            "format=duration", "-of", "csv=p=0", audio_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except:
            return 30.0  # Default fallback

    def create_ken_burns(self, image_path: str, output_path: str,
                        duration: float, direction: str = "zoom_in") -> bool:
        """
        Create Ken Burns effect (zoom/pan) on a single image

        Directions: zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down
        """

        effects = {
            "zoom_in": "scale=8000:-1,zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1080x1920:fps={fps}",
            "zoom_out": "scale=8000:-1,zoompan=z='if(lte(zoom,1.0),1.5,max(1.001,zoom-0.0015))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1080x1920:fps={fps}",
            "pan_left": "scale=8000:-1,zoompan=z='1.2':x='if(lte(on,1),0,min(iw/zoom,x+2))':y='ih/2-(ih/zoom/2)':d={frames}:s=1080x1920:fps={fps}",
            "pan_right": "scale=8000:-1,zoompan=z='1.2':x='if(lte(on,1),iw/zoom,max(0,x-2))':y='ih/2-(ih/zoom/2)':d={frames}:s=1080x1920:fps={fps}",
        }

        fps = FacelessConfig.DEFAULT_FPS
        frames = int(duration * fps)

        filter_str = effects.get(direction, effects["zoom_in"]).format(frames=frames, fps=fps)

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", image_path,
            "-vf", filter_str,
            "-t", str(duration),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
            return os.path.exists(output_path)
        except Exception as e:
            print(f"[Ken Burns] Error: {e}")
            return False

    def create_simple_slideshow(self, images: List[str], output_path: str,
                                duration_per_image: float = 5.0) -> bool:
        """Create a simple slideshow from images"""

        # Create concat file
        concat_file = FacelessConfig.TEMP_DIR / "slideshow_concat.txt"

        with open(concat_file, 'w') as f:
            for img in images:
                f.write(f"file '{img}'\n")
                f.write(f"duration {duration_per_image}\n")
            # Add last image again (FFmpeg concat quirk)
            f.write(f"file '{images[-1]}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(FacelessConfig.DEFAULT_FPS),
            output_path
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return os.path.exists(output_path)
        except Exception as e:
            print(f"[Slideshow] Error: {e}")
            return False

    def create_animated_slideshow(self, images: List[str], output_path: str,
                                  total_duration: float) -> bool:
        """Create slideshow with Ken Burns effects"""

        duration_per_image = total_duration / len(images)
        directions = ["zoom_in", "zoom_out", "pan_left", "pan_right"]

        # Generate individual clips with Ken Burns
        clips = []
        for i, img in enumerate(images):
            clip_path = str(FacelessConfig.TEMP_DIR / f"kb_clip_{i:03d}.mp4")
            direction = directions[i % len(directions)]

            print(f"  [Clip {i+1}/{len(images)}] {direction}...")

            if self.create_ken_burns(img, clip_path, duration_per_image, direction):
                clips.append(clip_path)
            else:
                # Fallback to static image
                clips.append(img)

        # Concatenate clips
        concat_file = FacelessConfig.TEMP_DIR / "kb_concat.txt"
        with open(concat_file, 'w') as f:
            for clip in clips:
                f.write(f"file '{clip}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return os.path.exists(output_path)
        except Exception as e:
            print(f"[Animated Slideshow] Error: {e}")
            return False

    def add_audio_to_video(self, video_path: str, audio_path: str,
                          output_path: str, bg_music_path: str = None,
                          music_volume: float = 0.1) -> bool:
        """Combine video with voice audio and optional background music"""

        if bg_music_path and os.path.exists(bg_music_path):
            # Video + Voice + Background music
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-i", bg_music_path,
                "-filter_complex",
                f"[2:a]volume={music_volume}[bg];[1:a][bg]amix=inputs=2:duration=first[aout]",
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path
            ]
        else:
            # Video + Voice only
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path
            ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return os.path.exists(output_path)
        except Exception as e:
            print(f"[Add Audio] Error: {e}")
            return False

    def add_subtitles(self, video_path: str, script: str, output_path: str,
                     style: str = "default") -> bool:
        """Add subtitles to video"""

        # Create SRT file from script
        srt_path = FacelessConfig.TEMP_DIR / "subtitles.srt"

        # Simple sentence-based subtitles
        sentences = script.replace('\n', ' ').split('. ')
        duration_per_sentence = 3.0  # Rough estimate

        with open(srt_path, 'w') as f:
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue

                start_time = i * duration_per_sentence
                end_time = start_time + duration_per_sentence

                start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d},000"
                end_str = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d},000"

                f.write(f"{i+1}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{sentence.strip()}.\n\n")

        # Subtitle styles
        styles = {
            "default": "FontSize=24,FontName=Arial,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2",
            "bold": "FontSize=28,FontName=Arial Black,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=3",
            "minimal": "FontSize=20,FontName=Helvetica,PrimaryColour=&HFFFFFF,Outline=1",
            "tiktok": "FontSize=32,FontName=Arial Black,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=3,MarginV=100",
        }

        style_str = styles.get(style, styles["default"])

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"subtitles={srt_path}:force_style='{style_str}'",
            "-c:a", "copy",
            output_path
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return os.path.exists(output_path)
        except Exception as e:
            print(f"[Subtitles] Error: {e}")
            return False

    def generate_faceless_video(self, audio_path: str, output_path: str,
                                images: List[str] = None,
                                script: str = None,
                                topic: str = "travel",
                                add_subtitles: bool = True,
                                bg_music_path: str = None,
                                ken_burns: bool = True,
                                use_ai_images: bool = False,
                                animate_images: bool = False) -> bool:
        """
        Generate a complete faceless video

        Args:
            audio_path: Path to voice audio file
            output_path: Path for final video
            images: List of image paths (optional, will use placeholders if None)
            script: Script text for subtitles (optional)
            topic: Topic for placeholder colors
            add_subtitles: Whether to add subtitles
            bg_music_path: Optional background music
            ken_burns: Use Ken Burns effect (zoom/pan)
            use_ai_images: Generate images with FLUX (requires ComfyUI)
            animate_images: Animate images with Wan I2V (requires ComfyUI)
        """

        print(f"[Faceless] Generating video...")

        # Get audio duration
        duration = self.get_audio_duration(audio_path)
        print(f"[Faceless] Audio duration: {duration:.1f}s")

        # Prepare images
        if not images or len(images) == 0:
            num_images = max(3, int(duration / 5))  # ~5 seconds per image

            # Try AI image generation first
            if use_ai_images and self.ai_gen and self.ai_gen.is_available():
                print(f"[Faceless] Generating {num_images} AI images with FLUX...")
                images = self.ai_gen.generate_travel_images(topic, num_images)

            # Fallback to placeholders if AI generation failed or not requested
            if not images or len(images) == 0:
                print(f"[Faceless] Using {num_images} placeholder images")
                images = self.stock.get_placeholder_images(num_images, topic)

        # Optionally animate images with Wan I2V
        if animate_images and self.ai_gen and self.ai_gen.is_available():
            print(f"[Faceless] Animating images with Wan I2V...")
            images = self.ai_gen.animate_images(images, topic)

        # Calculate images needed
        num_images = len(images)
        duration_per_image = duration / num_images
        print(f"[Faceless] {num_images} images, {duration_per_image:.1f}s each")

        # Step 1: Create slideshow
        slideshow_path = str(FacelessConfig.TEMP_DIR / "slideshow.mp4")

        if ken_burns:
            print("[Faceless] Creating animated slideshow with Ken Burns...")
            success = self.create_animated_slideshow(images, slideshow_path, duration)
        else:
            print("[Faceless] Creating simple slideshow...")
            success = self.create_simple_slideshow(images, slideshow_path, duration_per_image)

        if not success:
            print("[Faceless] Slideshow creation failed!")
            return False

        # Step 2: Add audio
        with_audio_path = str(FacelessConfig.TEMP_DIR / "with_audio.mp4")
        print("[Faceless] Adding audio...")

        success = self.add_audio_to_video(
            slideshow_path, audio_path, with_audio_path,
            bg_music_path
        )

        if not success:
            print("[Faceless] Audio addition failed!")
            return False

        # Step 3: Add subtitles (optional)
        if add_subtitles and script:
            print("[Faceless] Adding subtitles...")
            success = self.add_subtitles(with_audio_path, script, output_path, "tiktok")
            if not success:
                # Fall back to video without subtitles
                shutil.copy(with_audio_path, output_path)
        else:
            shutil.copy(with_audio_path, output_path)

        if os.path.exists(output_path):
            print(f"[Faceless] Video created: {output_path}")
            return True

        return False


# ============== CLI ==============

def main():
    """Command line interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Faceless Video Generator")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("-o", "--output", default="faceless_video.mp4", help="Output path")
    parser.add_argument("-i", "--images", nargs="+", help="Image files to use")
    parser.add_argument("-s", "--script", help="Script text for subtitles")
    parser.add_argument("--no-subtitles", action="store_true", help="Disable subtitles")
    parser.add_argument("--no-ken-burns", action="store_true", help="Disable Ken Burns effect")
    parser.add_argument("-m", "--music", help="Background music file")

    args = parser.parse_args()

    generator = FacelessVideoGenerator()

    success = generator.generate_faceless_video(
        audio_path=args.audio,
        output_path=args.output,
        images=args.images,
        script=args.script,
        add_subtitles=not args.no_subtitles,
        bg_music_path=args.music,
        ken_burns=not args.no_ken_burns
    )

    if success:
        print(f"\nVideo created: {args.output}")
    else:
        print("\nVideo creation failed!")


if __name__ == "__main__":
    main()
