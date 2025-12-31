"""
Alexandra AI - YouTube Video Pipeline
End-to-end automation: Topic -> Research -> Script -> Voice -> Avatar -> Final Video
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from travel_config import CONTENT_DIR
from travel_research import TravelResearcher
from youtube_creator import YouTubeContentCreator, TravelVideoTemplates
from social_media import TikTokTemplates, MultiPlatformExport, BatchCreator, PLATFORMS
from faceless_video import FacelessVideoGenerator, FacelessConfig

# ============== CONFIGURATION ==============

class PipelineConfig:
    """Central configuration for all pipeline paths"""

    # Base directories
    HOME = Path("/home/alexandratitus767")

    # Voice (F5-TTS)
    F5_TTS_DIR = HOME / "voice_training" / "F5-TTS"
    VOICE_DATA_DIR = HOME / "voice_training" / "data" / "alexandra"
    REFERENCE_AUDIO = VOICE_DATA_DIR / "wavs" / "clip_000.wav"
    REFERENCE_TEXT = "Hello, it's nice to meet you I've been looking forward to this conversation all day"

    # Avatar (MuseTalk)
    MUSETALK_DIR = HOME / "MuseTalk"
    AVATAR_VIDEO = HOME / "ai-clone-chat" / "avatar_silent.mp4"

    # Output
    OUTPUT_DIR = HOME / "ai-clone-chat" / "travel" / "output"
    TEMP_DIR = HOME / "ai-clone-chat" / "travel" / "temp"

    # LLM
    OLLAMA_MODEL = "qwen3-coder"  # or your fine-tuned model

    @classmethod
    def ensure_dirs(cls):
        """Create output directories if they don't exist"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============== SCRIPT GENERATOR ==============

class ScriptGenerator:
    """Generate video scripts using LLM"""

    def __init__(self, model: str = None):
        self.model = model or PipelineConfig.OLLAMA_MODEL
        self.researcher = TravelResearcher()

    def research_topic(self, topic: str, topic_type: str = "destination") -> Dict:
        """Research a topic using web search"""
        print(f"[Research] Searching for: {topic}")

        if topic_type == "destination":
            research = self.researcher.research_destination(topic)
        else:
            research = self.researcher.search_destination(topic, "general")

        return research

    def generate_script(self, topic: str, video_type: str = "guide",
                       research: Dict = None, duration_minutes: int = 5) -> str:
        """Generate a video script using Ollama"""

        # Build context from research
        context = ""
        if research and "sections" in research:
            for section, results in research.get("sections", {}).items():
                if results:
                    context += f"\n{section.upper()}:\n"
                    for r in results[:2]:
                        if isinstance(r, dict):
                            context += f"- {r.get('title', '')}: {r.get('body', '')[:200]}\n"

        # Build prompt based on video type
        prompts = {
            "guide": f"""Write a {duration_minutes}-minute YouTube travel guide script about {topic}.

Research context:
{context}

Requirements:
- Start with an engaging hook (first 5 seconds are crucial)
- Include personal-feeling anecdotes and tips
- Cover: best time to visit, top attractions, food recommendations, where to stay, budget tips
- End with a call to action (subscribe, comment)
- Use conversational, enthusiastic tone
- Write ONLY the spoken words, no stage directions
- Break into natural paragraphs (each ~30 seconds of speaking)

Write the complete script:""",

            "tips": f"""Write a {duration_minutes}-minute YouTube tips video script about {topic}.

Research context:
{context}

Requirements:
- Hook viewers in first 5 seconds
- Number each tip clearly
- Give specific, actionable advice
- Use personal experiences and examples
- End with call to action
- Conversational tone
- Write ONLY spoken words

Write the complete script:""",

            "short": f"""Write a 45-second YouTube Short script about {topic}.

Research context:
{context}

Requirements:
- Immediate hook in first 2 seconds
- 3 quick tips or facts
- Fast-paced, punchy delivery
- End with "Follow for more"
- Under 150 words total

Write the complete script:"""
        }

        prompt = prompts.get(video_type, prompts["guide"])

        # Call Ollama
        print(f"[Script] Generating {video_type} script with {self.model}...")

        try:
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=120
            )
            script = result.stdout.strip()

            # Clean up any thinking tags if present
            if "<think>" in script:
                import re
                script = re.sub(r'<think>.*?</think>', '', script, flags=re.DOTALL)

            return script.strip()

        except subprocess.TimeoutExpired:
            print("[Script] Timeout - using template")
            return self._get_template_script(topic, video_type)
        except Exception as e:
            print(f"[Script] Error: {e}")
            return self._get_template_script(topic, video_type)

    def _get_template_script(self, topic: str, video_type: str) -> str:
        """Fallback template script"""
        templates = TravelVideoTemplates()
        if video_type == "guide":
            sections = templates.destination_guide(topic)
        elif video_type == "short":
            sections = templates.youtube_short(topic, f"You need to visit {topic}!")
        else:
            sections = templates.top_tips(topic, 5)

        return "\n\n".join([s["text"] for s in sections])


# ============== VOICE GENERATOR (F5-TTS) ==============

class VoiceGenerator:
    """Generate speech using F5-TTS"""

    def __init__(self):
        self.f5_dir = PipelineConfig.F5_TTS_DIR
        self.ref_audio = PipelineConfig.REFERENCE_AUDIO
        self.ref_text = PipelineConfig.REFERENCE_TEXT

    def _create_config(self, text: str, output_dir: str, output_file: str) -> str:
        """Create a TOML config file for F5-TTS"""
        config_content = f'''# F5-TTS Generation Config
model = "F5TTS_v1_Base"
ref_audio = "{self.ref_audio}"
ref_text = "{self.ref_text}"
gen_text = """{text}"""
gen_file = ""
remove_silence = true
output_dir = "{output_dir}"
output_file = "{output_file}"
'''
        config_path = PipelineConfig.TEMP_DIR / "f5tts_config.toml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        return str(config_path)

    def generate(self, text: str, output_path: str) -> bool:
        """Generate speech from text using F5-TTS"""
        print(f"[Voice] Generating audio ({len(text)} chars)...")

        output_dir = str(Path(output_path).parent)
        output_file = Path(output_path).name

        # Create config file
        config_path = self._create_config(text, output_dir, output_file)

        # F5-TTS inference command
        cmd = [
            "python3", "-m", "f5_tts.infer.infer_cli",
            "--config", config_path
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.f5_dir / "src"),
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0 and os.path.exists(output_path):
                print(f"[Voice] Generated: {output_path}")
                return True
            else:
                print(f"[Voice] F5-TTS output: {result.stdout[:300]}")
                print(f"[Voice] Error: {result.stderr[:500]}")
                return False

        except Exception as e:
            print(f"[Voice] Exception: {e}")
            return False

    def generate_chunked(self, text: str, output_path: str,
                        chunk_size: int = 500) -> bool:
        """Generate speech in chunks for longer texts"""

        # Split text into chunks at sentence boundaries
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())

        print(f"[Voice] Processing {len(chunks)} chunks...")

        # Generate each chunk
        temp_files = []
        for i, chunk in enumerate(chunks):
            temp_path = str(PipelineConfig.TEMP_DIR / f"chunk_{i:03d}.wav")
            if self.generate(chunk, temp_path):
                temp_files.append(temp_path)
            else:
                print(f"[Voice] Failed on chunk {i}")
                return False

        # Concatenate with FFmpeg
        if temp_files:
            return self._concatenate_audio(temp_files, output_path)

        return False

    def _concatenate_audio(self, files: List[str], output: str) -> bool:
        """Concatenate audio files using FFmpeg"""
        list_file = str(PipelineConfig.TEMP_DIR / "audio_list.txt")

        with open(list_file, 'w') as f:
            for file in files:
                f.write(f"file '{file}'\n")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file, "-c", "copy", output
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"[Voice] Concatenated to: {output}")
            return True
        except Exception as e:
            print(f"[Voice] Concat error: {e}")
            return False


# ============== AVATAR GENERATOR (MuseTalk) ==============

class AvatarGenerator:
    """Generate talking head video using MuseTalk"""

    def __init__(self):
        self.musetalk_dir = PipelineConfig.MUSETALK_DIR
        self.avatar_video = PipelineConfig.AVATAR_VIDEO

    def generate(self, audio_path: str, output_path: str,
                video_path: str = None) -> bool:
        """Generate talking head video from audio"""

        video = video_path or str(self.avatar_video)
        print(f"[Avatar] Generating video from: {audio_path}")

        # Create temp config for this job
        config = {
            "task_0": {
                "video_path": video,
                "audio_path": audio_path
            }
        }

        config_path = PipelineConfig.TEMP_DIR / "musetalk_config.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Run MuseTalk inference
        cmd = [
            "python3", "-m", "scripts.inference",
            "--inference_config", str(config_path),
            "--result_dir", str(Path(output_path).parent),
            "--unet_model_path", "models/musetalk/pytorch_model.bin",
            "--unet_config", "models/musetalk/musetalk.json",
            "--version", "v1"
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.musetalk_dir),
                capture_output=True,
                text=True,
                timeout=600,
                env={**os.environ, "PYTHONPATH": str(self.musetalk_dir)}
            )

            # Find the output file
            result_dir = Path(output_path).parent
            generated = list(result_dir.glob("*.mp4"))

            if generated:
                # Rename to desired output
                shutil.move(str(generated[0]), output_path)
                print(f"[Avatar] Generated: {output_path}")
                return True
            else:
                print(f"[Avatar] No output found. stderr: {result.stderr[:500]}")
                return False

        except Exception as e:
            print(f"[Avatar] Exception: {e}")
            return False


# ============== VIDEO ASSEMBLER ==============

class VideoAssembler:
    """Assemble final video with FFmpeg"""

    def add_background_music(self, video_path: str, music_path: str,
                            output_path: str, music_volume: float = 0.1) -> bool:
        """Add background music to video"""
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", music_path,
            "-filter_complex",
            f"[1:a]volume={music_volume}[bg];[0:a][bg]amix=inputs=2:duration=first",
            "-c:v", "copy",
            output_path
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except Exception as e:
            print(f"[Assemble] Music error: {e}")
            return False

    def add_intro_outro(self, main_video: str, intro: str, outro: str,
                       output_path: str) -> bool:
        """Add intro and outro to video"""

        # Create concat list
        list_file = PipelineConfig.TEMP_DIR / "video_list.txt"
        videos = [v for v in [intro, main_video, outro] if v and os.path.exists(v)]

        with open(list_file, 'w') as f:
            for v in videos:
                f.write(f"file '{v}'\n")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            output_path
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except Exception as e:
            print(f"[Assemble] Concat error: {e}")
            return False

    def create_final_video(self, video_path: str, output_path: str,
                          add_watermark: bool = False) -> bool:
        """Create final video with optional processing"""

        filters = []

        if add_watermark:
            # Add "Alexandra AI" watermark
            filters.append(
                "drawtext=text='Alexandra AI':fontsize=24:fontcolor=white@0.7:"
                "x=w-tw-20:y=h-th-20"
            )

        if filters:
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", ",".join(filters),
                "-c:a", "copy",
                output_path
            ]
        else:
            shutil.copy(video_path, output_path)
            return True

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except Exception as e:
            print(f"[Assemble] Final error: {e}")
            return False


# ============== MAIN PIPELINE ==============

class VideoPipeline:
    """Main pipeline orchestrating all components"""

    def __init__(self):
        PipelineConfig.ensure_dirs()

        self.script_gen = ScriptGenerator()
        self.voice_gen = VoiceGenerator()
        self.avatar_gen = AvatarGenerator()
        self.faceless_gen = FacelessVideoGenerator()
        self.assembler = VideoAssembler()
        self.youtube = YouTubeContentCreator()

    def create_video(self, topic: str, video_type: str = "guide",
                    auto_approve: bool = False) -> Optional[str]:
        """
        Full pipeline: Topic -> Final Video

        Args:
            topic: The video topic (e.g., "Barcelona travel")
            video_type: "guide", "tips", or "short"
            auto_approve: Skip script approval step

        Returns:
            Path to final video or None if failed
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_id = f"{video_type}_{timestamp}"
        project_dir = PipelineConfig.OUTPUT_DIR / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Creating {video_type} video: {topic}")
        print(f"Project: {project_id}")
        print(f"{'='*50}\n")

        # Step 1: Research
        print("[1/5] Researching topic...")
        research = self.script_gen.research_topic(topic)

        # Save research
        with open(project_dir / "research.json", 'w') as f:
            json.dump(research, f, indent=2, default=str)

        # Step 2: Generate Script
        print("\n[2/5] Generating script...")
        script = self.script_gen.generate_script(
            topic, video_type, research,
            duration_minutes=1 if video_type == "short" else 5
        )

        # Save script
        script_path = project_dir / "script.txt"
        with open(script_path, 'w') as f:
            f.write(script)

        print(f"\nScript preview:\n{'-'*40}")
        print(script[:500] + "..." if len(script) > 500 else script)
        print(f"{'-'*40}\n")

        # Step 3: Approval
        if not auto_approve:
            print(f"Script saved to: {script_path}")
            response = input("\nApprove script? [Y/n/edit]: ").strip().lower()

            if response == 'n':
                print("Cancelled.")
                return None
            elif response == 'edit':
                print(f"Edit the script at: {script_path}")
                input("Press Enter when done editing...")
                with open(script_path, 'r') as f:
                    script = f.read()

        # Step 4: Generate Voice
        print("\n[3/5] Generating voice...")
        audio_path = str(project_dir / "voice.wav")

        if len(script) > 500:
            success = self.voice_gen.generate_chunked(script, audio_path)
        else:
            success = self.voice_gen.generate(script, audio_path)

        if not success:
            print("Voice generation failed!")
            return None

        # Step 5: Generate Avatar Video
        print("\n[4/5] Generating avatar video...")
        avatar_path = str(project_dir / "avatar.mp4")

        success = self.avatar_gen.generate(audio_path, avatar_path)

        if not success:
            print("Avatar generation failed!")
            # Fallback: just use audio with static image
            print("Creating audio-only version...")
            avatar_path = audio_path  # Use audio for now

        # Step 6: Final Assembly
        print("\n[5/5] Assembling final video...")
        final_path = str(project_dir / f"final_{topic.replace(' ', '_')}.mp4")

        if avatar_path.endswith('.mp4'):
            success = self.assembler.create_final_video(
                avatar_path, final_path, add_watermark=True
            )
        else:
            # Audio only - create video with static image
            final_path = avatar_path  # Just the audio for now
            success = True

        if success:
            print(f"\n{'='*50}")
            print(f"VIDEO CREATED SUCCESSFULLY!")
            print(f"Location: {final_path}")
            print(f"{'='*50}\n")

            # Save project metadata
            metadata = {
                "topic": topic,
                "type": video_type,
                "created": timestamp,
                "script_path": str(script_path),
                "audio_path": audio_path,
                "video_path": final_path,
                "script_length": len(script)
            }
            with open(project_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            return final_path

        return None

    def quick_video(self, topic: str) -> Optional[str]:
        """Create a video with minimal interaction"""
        return self.create_video(topic, "short", auto_approve=True)

    def list_projects(self) -> List[Dict]:
        """List all created video projects"""
        projects = []
        for p in PipelineConfig.OUTPUT_DIR.iterdir():
            if p.is_dir():
                meta_path = p / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        projects.append(json.load(f))
        return projects

    # ============== SOCIAL MEDIA FEATURES ==============

    def create_tiktok(self, topic: str, hook_style: str = "curiosity",
                     auto_approve: bool = False) -> Optional[str]:
        """Create a TikTok-optimized video with trending hooks"""

        print(f"\n{'='*50}")
        print(f"Creating TikTok: {topic}")
        print(f"Hook style: {hook_style}")
        print(f"{'='*50}\n")

        # Get TikTok-style hook
        hook = TikTokTemplates.get_hook(topic, hook_style)
        print(f"Hook: {hook}\n")

        # Generate with TikTok template
        template = TikTokTemplates.quick_tips(topic, hook_style)

        # Create the video
        return self.create_video(topic, "short", auto_approve)

    def create_batch(self, destination: str, count: int = 5,
                    auto_approve: bool = False) -> List[str]:
        """Create multiple videos from one destination"""

        print(f"\n{'='*50}")
        print(f"Batch Creating {count} Videos: {destination}")
        print(f"{'='*50}\n")

        batch = BatchCreator()

        # Generate ideas and scripts
        scripts = batch.generate_batch_scripts(destination, count)

        # Save batch for review
        batch_dir = batch.save_batch(destination, scripts)
        print(f"\nScripts saved to: {batch_dir}")

        if not auto_approve:
            print("\nReview the scripts in the batch folder.")
            response = input("Proceed with video creation? [y/N]: ").strip().lower()
            if response != 'y':
                print("Batch cancelled. Scripts saved for later use.")
                return []

        # Create each video
        created_videos = []
        for i, script in enumerate(scripts):
            print(f"\n[{i+1}/{count}] Creating: {script['title']}")
            video_path = self.create_video(
                f"{destination} - {script['angle']}",
                "short",
                auto_approve=True
            )
            if video_path:
                created_videos.append(video_path)

        print(f"\n{'='*50}")
        print(f"Batch Complete: {len(created_videos)}/{count} videos created")
        print(f"{'='*50}\n")

        return created_videos

    def export_to_platforms(self, video_path: str, destination: str,
                           description: str = "") -> Dict:
        """Export a video to all social platforms with optimized metadata"""

        print(f"\nExporting to all platforms...")

        exporter = MultiPlatformExport()
        desc = description or f"Amazing {destination} travel tips you need to know!"

        results = exporter.export_all_platforms(
            video_path, destination, desc, "tips"
        )

        print("\nExport complete:")
        for platform, data in results.items():
            if "error" not in data:
                print(f"  {platform}: {data.get('video_path', 'N/A')}")

        return results

    def show_hooks(self, destination: str):
        """Display all available TikTok hooks for a destination"""

        print(f"\n{'='*50}")
        print(f"TikTok Hooks for: {destination}")
        print(f"{'='*50}")

        hooks = TikTokTemplates.get_all_hooks(destination)
        for style, hook_list in hooks.items():
            print(f"\n{style.upper()}:")
            for hook in hook_list:
                print(f"  - {hook}")

    # ============== FACELESS VIDEO ==============

    def create_faceless(self, topic: str, video_type: str = "short",
                       images: List[str] = None,
                       auto_approve: bool = False,
                       add_subtitles: bool = True,
                       ken_burns: bool = True,
                       use_ai_images: bool = False,
                       animate_images: bool = False) -> Optional[str]:
        """
        Create a faceless video (voice + images, no avatar)

        Args:
            topic: Video topic
            video_type: "short" or "guide"
            images: Optional list of image paths
            auto_approve: Skip script approval
            add_subtitles: Add subtitles to video
            ken_burns: Use Ken Burns effect (zoom/pan on images)
            use_ai_images: Generate images with FLUX (requires ComfyUI running)
            animate_images: Animate images with Wan I2V (requires ComfyUI)
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_id = f"faceless_{timestamp}"
        project_dir = PipelineConfig.OUTPUT_DIR / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Creating FACELESS Video: {topic}")
        print(f"Type: {video_type} | Subtitles: {add_subtitles}")
        print(f"Project: {project_id}")
        print(f"{'='*50}\n")

        # Step 1: Research
        print("[1/4] Researching topic...")
        research = self.script_gen.research_topic(topic)

        # Step 2: Generate Script
        print("\n[2/4] Generating script...")
        script = self.script_gen.generate_script(
            topic, video_type, research,
            duration_minutes=1 if video_type == "short" else 5
        )

        # Save script
        script_path = project_dir / "script.txt"
        with open(script_path, 'w') as f:
            f.write(script)

        print(f"\nScript preview:\n{'-'*40}")
        print(script[:500] + "..." if len(script) > 500 else script)
        print(f"{'-'*40}\n")

        # Step 3: Approval
        if not auto_approve:
            print(f"Script saved to: {script_path}")
            response = input("\nApprove script? [Y/n/edit]: ").strip().lower()

            if response == 'n':
                print("Cancelled.")
                return None
            elif response == 'edit':
                print(f"Edit the script at: {script_path}")
                input("Press Enter when done editing...")
                with open(script_path, 'r') as f:
                    script = f.read()

        # Step 4: Generate Voice
        print("\n[3/4] Generating voice...")
        audio_path = str(project_dir / "voice.wav")

        if len(script) > 500:
            success = self.voice_gen.generate_chunked(script, audio_path)
        else:
            success = self.voice_gen.generate(script, audio_path)

        if not success:
            print("Voice generation failed!")
            return None

        # Step 5: Generate Faceless Video
        print("\n[4/4] Creating faceless video...")
        final_path = str(project_dir / f"faceless_{topic.replace(' ', '_')}.mp4")

        success = self.faceless_gen.generate_faceless_video(
            audio_path=audio_path,
            output_path=final_path,
            images=images,
            script=script if add_subtitles else None,
            topic=topic.split()[0].lower() if topic else "travel",
            add_subtitles=add_subtitles,
            ken_burns=ken_burns,
            use_ai_images=use_ai_images,
            animate_images=animate_images
        )

        if success:
            print(f"\n{'='*50}")
            print(f"FACELESS VIDEO CREATED!")
            print(f"Location: {final_path}")
            print(f"{'='*50}\n")

            # Save metadata
            metadata = {
                "topic": topic,
                "type": f"faceless_{video_type}",
                "created": timestamp,
                "script_path": str(script_path),
                "audio_path": audio_path,
                "video_path": final_path,
                "script_length": len(script),
                "subtitles": add_subtitles,
                "ken_burns": ken_burns,
                "ai_images": use_ai_images,
                "animated": animate_images
            }
            with open(project_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            return final_path

        return None


# ============== CLI INTERFACE ==============

def main():
    """Command line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Alexandra AI - YouTube Video Pipeline"
    )
    parser.add_argument("topic", nargs="?", help="Video topic")
    parser.add_argument("--type", "-t", choices=["guide", "tips", "short", "tiktok", "faceless"],
                       default="guide", help="Video type")
    parser.add_argument("--auto", "-a", action="store_true",
                       help="Auto-approve script")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List existing projects")
    parser.add_argument("--batch", "-b", type=int, metavar="N",
                       help="Create N videos for topic")
    parser.add_argument("--hooks", action="store_true",
                       help="Show TikTok hooks for topic")
    parser.add_argument("--export", "-e", metavar="VIDEO",
                       help="Export video to all platforms")
    parser.add_argument("--hook-style", choices=["controversial", "curiosity", "listicle", "story", "challenge"],
                       default="curiosity", help="TikTok hook style")
    parser.add_argument("--faceless", "-f", action="store_true",
                       help="Create faceless video (voice + images)")
    parser.add_argument("--images", "-i", nargs="+",
                       help="Image files for faceless video")
    parser.add_argument("--no-subtitles", action="store_true",
                       help="Disable subtitles in faceless video")
    parser.add_argument("--no-ken-burns", action="store_true",
                       help="Disable Ken Burns effect")
    parser.add_argument("--ai-images", action="store_true",
                       help="Generate images with FLUX (requires ComfyUI)")
    parser.add_argument("--animate", action="store_true",
                       help="Animate images with Wan I2V (requires ComfyUI)")

    args = parser.parse_args()

    pipeline = VideoPipeline()

    # List projects
    if args.list:
        projects = pipeline.list_projects()
        if projects:
            print("\nExisting projects:")
            for p in projects:
                print(f"  - {p.get('topic')} ({p.get('type')}) - {p.get('created')}")
        else:
            print("No projects found.")
        return

    # Show hooks
    if args.hooks and args.topic:
        pipeline.show_hooks(args.topic)
        return

    # Export to platforms
    if args.export and args.topic:
        pipeline.export_to_platforms(args.export, args.topic)
        return

    # Batch creation
    if args.batch and args.topic:
        pipeline.create_batch(args.topic, args.batch, args.auto)
        return

    # Faceless video (command line flag)
    if args.faceless and args.topic:
        pipeline.create_faceless(
            args.topic,
            video_type="short",
            images=args.images,
            auto_approve=args.auto,
            add_subtitles=not args.no_subtitles,
            ken_burns=not args.no_ken_burns,
            use_ai_images=args.ai_images,
            animate_images=args.animate
        )
        return

    if not args.topic:
        # Interactive mode
        print("\n" + "="*50)
        print("  Alexandra AI - Video Creator")
        print("  YouTube | TikTok | Instagram")
        print("="*50 + "\n")

        topic = input("Enter video topic: ").strip()
        if not topic:
            print("No topic provided.")
            return

        print("\nVideo types:")
        print("  1. guide    - Full destination guide (5+ min)")
        print("  2. tips     - Tips list video (3-5 min)")
        print("  3. short    - YouTube Short (< 60 sec)")
        print("  4. tiktok   - TikTok optimized (trending hooks)")
        print("  5. faceless - Voice + images (highest quality)")
        print("  6. batch    - Create 5 videos at once")

        choice = input("\nSelect type [1-6]: ").strip()

        if choice == "6":
            count = input("How many videos? [5]: ").strip()
            count = int(count) if count.isdigit() else 5
            pipeline.create_batch(topic, count)
        elif choice == "5":
            print("\nFaceless video options:")
            subs = input("Add subtitles? [Y/n]: ").strip().lower() != 'n'
            kb = input("Use Ken Burns effect (zoom/pan)? [Y/n]: ").strip().lower() != 'n'
            ai_img = input("Generate AI images with FLUX? [y/N]: ").strip().lower() == 'y'
            animate = False
            if ai_img:
                animate = input("Animate images with Wan I2V? [y/N]: ").strip().lower() == 'y'
            pipeline.create_faceless(topic, "short", None, False, subs, kb, ai_img, animate)
        elif choice == "4":
            print("\nHook styles:")
            print("  1. controversial - Bold claims that grab attention")
            print("  2. curiosity     - Makes viewers want to know more")
            print("  3. listicle      - Numbered tips format")
            print("  4. story         - Personal story style")
            print("  5. challenge     - Test or challenge format")
            style_choice = input("\nSelect hook style [1-5]: ").strip()
            styles = {"1": "controversial", "2": "curiosity", "3": "listicle", "4": "story", "5": "challenge"}
            hook_style = styles.get(style_choice, "curiosity")
            pipeline.create_tiktok(topic, hook_style)
        else:
            video_type = {"1": "guide", "2": "tips", "3": "short"}.get(choice, "guide")
            pipeline.create_video(topic, video_type)
    else:
        # Command line mode
        if args.type == "faceless":
            pipeline.create_faceless(
                args.topic,
                video_type="short",
                images=args.images,
                auto_approve=args.auto,
                add_subtitles=not args.no_subtitles,
                ken_burns=not args.no_ken_burns,
                use_ai_images=args.ai_images,
                animate_images=args.animate
            )
        elif args.type == "tiktok":
            pipeline.create_tiktok(args.topic, args.hook_style, args.auto)
        else:
            pipeline.create_video(args.topic, args.type, args.auto)


if __name__ == "__main__":
    main()
