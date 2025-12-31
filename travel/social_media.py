"""
Alexandra AI - Social Media Features
TikTok templates, multi-platform export, and batch creation
"""

import os
import json
import subprocess
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from travel_config import CONTENT_DIR

# ============== PLATFORM SETTINGS ==============

PLATFORMS = {
    "tiktok": {
        "name": "TikTok",
        "resolution": "1080x1920",
        "aspect": "9:16",
        "max_duration": 180,  # 3 minutes
        "optimal_duration": 30,  # 15-45 seconds performs best
        "hashtag_limit": 5,
        "caption_limit": 2200,
    },
    "youtube_shorts": {
        "name": "YouTube Shorts",
        "resolution": "1080x1920",
        "aspect": "9:16",
        "max_duration": 60,
        "optimal_duration": 45,
        "hashtag_limit": 15,
        "caption_limit": 100,  # Title limit
    },
    "instagram_reels": {
        "name": "Instagram Reels",
        "resolution": "1080x1920",
        "aspect": "9:16",
        "max_duration": 90,
        "optimal_duration": 30,
        "hashtag_limit": 30,
        "caption_limit": 2200,
    },
    "instagram_story": {
        "name": "Instagram Story",
        "resolution": "1080x1920",
        "aspect": "9:16",
        "max_duration": 60,
        "optimal_duration": 15,
        "hashtag_limit": 10,
        "caption_limit": 500,
    }
}


# ============== TIKTOK TEMPLATES ==============

class TikTokTemplates:
    """TikTok-optimized script templates with trending hooks"""

    # Trending hook styles that perform well on TikTok
    HOOKS = {
        "controversial": [
            "Stop going to {destination} without knowing this...",
            "I can't believe tourists still make this mistake in {destination}...",
            "POV: You're about to save $500 on your {destination} trip",
            "The {destination} travel hack no one talks about...",
            "Why is no one talking about this place in {destination}?",
        ],
        "curiosity": [
            "I found the most underrated spot in {destination}...",
            "This changed everything about my {destination} trip...",
            "What I wish I knew before visiting {destination}...",
            "The secret to {destination} that locals don't share...",
            "You're doing {destination} wrong if you're not doing this...",
        ],
        "listicle": [
            "3 things you NEED to know about {destination}",
            "5 places in {destination} that will blow your mind",
            "The only 4 places worth visiting in {destination}",
            "3 mistakes to avoid in {destination}",
            "Top 3 foods you MUST try in {destination}",
        ],
        "story": [
            "So I just got back from {destination} and...",
            "Story time: What happened in {destination}...",
            "I almost didn't go to {destination}, but...",
            "Let me tell you about {destination}...",
            "Okay so {destination} completely surprised me...",
        ],
        "challenge": [
            "Can you visit {destination} for under $50 a day?",
            "I tried to see all of {destination} in 24 hours...",
            "Rating {destination}'s top tourist spots honestly...",
            "Testing viral {destination} travel hacks...",
            "I asked locals for their {destination} secrets...",
        ]
    }

    @classmethod
    def get_hook(cls, destination: str, style: str = "curiosity") -> str:
        """Get a random hook for the destination"""
        import random
        hooks = cls.HOOKS.get(style, cls.HOOKS["curiosity"])
        hook = random.choice(hooks)
        return hook.format(destination=destination)

    @classmethod
    def get_all_hooks(cls, destination: str) -> Dict[str, List[str]]:
        """Get all hooks for a destination organized by style"""
        result = {}
        for style, hooks in cls.HOOKS.items():
            result[style] = [h.format(destination=destination) for h in hooks]
        return result

    @staticmethod
    def quick_tips(destination: str, hook_style: str = "listicle") -> Dict:
        """Quick tips format - performs well on TikTok"""
        hook = TikTokTemplates.get_hook(destination, hook_style)
        return {
            "type": "quick_tips",
            "duration": 30,
            "sections": [
                {"time": "0-3s", "type": "hook", "text": hook},
                {"time": "3-10s", "type": "tip1", "text": f"[Tip 1 about {destination}]"},
                {"time": "10-17s", "type": "tip2", "text": f"[Tip 2 about {destination}]"},
                {"time": "17-24s", "type": "tip3", "text": f"[Tip 3 about {destination}]"},
                {"time": "24-30s", "type": "cta", "text": "Follow for more travel tips!"},
            ]
        }

    @staticmethod
    def storytime(destination: str) -> Dict:
        """Story format - high engagement on TikTok"""
        hook = TikTokTemplates.get_hook(destination, "story")
        return {
            "type": "storytime",
            "duration": 45,
            "sections": [
                {"time": "0-5s", "type": "hook", "text": hook},
                {"time": "5-15s", "type": "setup", "text": f"[Set the scene in {destination}]"},
                {"time": "15-30s", "type": "story", "text": "[The main story/discovery]"},
                {"time": "30-40s", "type": "payoff", "text": "[The reveal/lesson learned]"},
                {"time": "40-45s", "type": "cta", "text": "Have you been? Comment below!"},
            ]
        }

    @staticmethod
    def before_after(destination: str) -> Dict:
        """Before/After expectations - viral format"""
        return {
            "type": "before_after",
            "duration": 20,
            "sections": [
                {"time": "0-2s", "type": "hook", "text": f"{destination}: Expectations vs Reality"},
                {"time": "2-8s", "type": "expectation", "text": f"[What tourists expect from {destination}]"},
                {"time": "8-10s", "type": "transition", "text": "But actually..."},
                {"time": "10-18s", "type": "reality", "text": f"[The real {destination} experience]"},
                {"time": "18-20s", "type": "cta", "text": "Which side did you experience?"},
            ]
        }

    @staticmethod
    def pov_style(destination: str) -> Dict:
        """POV format - immersive and popular"""
        return {
            "type": "pov",
            "duration": 25,
            "sections": [
                {"time": "0-3s", "type": "hook", "text": f"POV: Your first day in {destination}"},
                {"time": "3-8s", "type": "scene1", "text": "[Morning activity]"},
                {"time": "8-13s", "type": "scene2", "text": "[Afternoon discovery]"},
                {"time": "13-18s", "type": "scene3", "text": "[Evening experience]"},
                {"time": "18-22s", "type": "reaction", "text": "[Your reaction/feeling]"},
                {"time": "22-25s", "type": "cta", "text": "Save this for your trip!"},
            ]
        }

    @staticmethod
    def ranking(destination: str, topic: str = "places") -> Dict:
        """Ranking format - drives comments"""
        return {
            "type": "ranking",
            "duration": 35,
            "sections": [
                {"time": "0-3s", "type": "hook", "text": f"Ranking {topic} in {destination}..."},
                {"time": "3-10s", "type": "rank3", "text": f"[#3 {topic}] - Good but..."},
                {"time": "10-17s", "type": "rank2", "text": f"[#2 {topic}] - Really great..."},
                {"time": "17-27s", "type": "rank1", "text": f"[#1 {topic}] - The best because..."},
                {"time": "27-32s", "type": "hot_take", "text": "[Controversial opinion]"},
                {"time": "32-35s", "type": "cta", "text": "What's your ranking? Comment!"},
            ]
        }


# ============== MULTI-PLATFORM EXPORT ==============

class MultiPlatformExport:
    """Export videos with platform-specific metadata"""

    def __init__(self):
        self.export_dir = Path(CONTENT_DIR) / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def generate_hashtags(self, destination: str, topic_type: str,
                         platform: str) -> List[str]:
        """Generate platform-optimized hashtags"""

        # Base hashtags
        base = [
            "travel", "traveltiktok", "traveltips",
            destination.lower().replace(" ", ""),
            f"{destination.lower().replace(' ', '')}travel"
        ]

        # Topic-specific hashtags
        topic_tags = {
            "food": ["foodie", "foodtiktok", "travelfood", "wheretoeat", "foodtravel"],
            "tips": ["travelhacks", "traveltips", "traveladvice", "travelsmart", "travelguide"],
            "hidden_gems": ["hiddengems", "offthebeatenpath", "secretspots", "localtips", "explore"],
            "budget": ["budgettravel", "cheaptravel", "travelonabudget", "savemoney", "backpacker"],
            "luxury": ["luxurytravel", "luxuryhotel", "fivestar", "treatyourself", "splurge"],
        }

        # Platform-specific trending tags
        platform_tags = {
            "tiktok": ["fyp", "foryou", "foryoupage", "viral", "tiktoktravel"],
            "youtube_shorts": ["shorts", "youtubeshorts", "subscribe", "viral"],
            "instagram_reels": ["reels", "reelsinstagram", "instareels", "explorepage", "instatravel"],
        }

        # Combine and limit
        tags = base + topic_tags.get(topic_type, []) + platform_tags.get(platform, [])
        limit = PLATFORMS[platform]["hashtag_limit"]

        return list(dict.fromkeys(tags))[:limit]  # Remove duplicates, limit count

    def generate_caption(self, destination: str, description: str,
                        platform: str, hashtags: List[str]) -> str:
        """Generate platform-optimized caption"""

        limit = PLATFORMS[platform]["caption_limit"]
        hashtag_str = " ".join([f"#{tag}" for tag in hashtags])

        if platform == "tiktok":
            caption = f"{description}\n\n{hashtag_str}"
        elif platform == "youtube_shorts":
            # YouTube Shorts title is short
            caption = description[:limit]
        elif platform == "instagram_reels":
            caption = f"{description}\n\n.\n.\n.\n{hashtag_str}"
        else:
            caption = f"{description}\n\n{hashtag_str}"

        return caption[:limit]

    def export_for_platform(self, video_path: str, destination: str,
                           description: str, topic_type: str,
                           platform: str) -> Dict:
        """Export video with platform-specific metadata"""

        platform_config = PLATFORMS.get(platform)
        if not platform_config:
            return {"error": f"Unknown platform: {platform}"}

        # Create platform export folder
        platform_dir = self.export_dir / platform / datetime.now().strftime("%Y%m%d")
        platform_dir.mkdir(parents=True, exist_ok=True)

        # Generate metadata
        hashtags = self.generate_hashtags(destination, topic_type, platform)
        caption = self.generate_caption(destination, description, platform, hashtags)

        # Copy/convert video if needed
        video_name = f"{destination.replace(' ', '_')}_{platform}.mp4"
        output_path = platform_dir / video_name

        # For now, just copy (could add resolution conversion later)
        if os.path.exists(video_path):
            import shutil
            shutil.copy(video_path, output_path)

        # Save metadata
        metadata = {
            "platform": platform,
            "platform_name": platform_config["name"],
            "destination": destination,
            "video_path": str(output_path),
            "caption": caption,
            "hashtags": hashtags,
            "description": description,
            "exported": datetime.now().isoformat(),
            "specs": {
                "resolution": platform_config["resolution"],
                "max_duration": platform_config["max_duration"],
            }
        }

        meta_path = platform_dir / f"{destination.replace(' ', '_')}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def export_all_platforms(self, video_path: str, destination: str,
                            description: str, topic_type: str = "tips") -> Dict[str, Dict]:
        """Export to all supported platforms"""

        results = {}
        for platform in ["tiktok", "youtube_shorts", "instagram_reels"]:
            results[platform] = self.export_for_platform(
                video_path, destination, description, topic_type, platform
            )

        return results


# ============== BATCH CREATION ==============

class BatchCreator:
    """Create multiple videos from one topic"""

    def __init__(self):
        self.templates = TikTokTemplates()

    def generate_batch_ideas(self, destination: str, count: int = 5) -> List[Dict]:
        """Generate multiple video ideas for a destination"""

        ideas = [
            {
                "title": f"3 Things You NEED to Know About {destination}",
                "template": "quick_tips",
                "hook_style": "listicle",
                "topic_type": "tips",
                "angle": "general tips"
            },
            {
                "title": f"The {destination} Mistake Everyone Makes",
                "template": "quick_tips",
                "hook_style": "controversial",
                "topic_type": "tips",
                "angle": "mistakes to avoid"
            },
            {
                "title": f"Best Food in {destination}",
                "template": "ranking",
                "hook_style": "listicle",
                "topic_type": "food",
                "angle": "food recommendations"
            },
            {
                "title": f"Hidden Gems in {destination}",
                "template": "quick_tips",
                "hook_style": "curiosity",
                "topic_type": "hidden_gems",
                "angle": "off the beaten path"
            },
            {
                "title": f"{destination} on a Budget",
                "template": "quick_tips",
                "hook_style": "controversial",
                "topic_type": "budget",
                "angle": "budget tips"
            },
            {
                "title": f"First Time in {destination}? Watch This",
                "template": "pov_style",
                "hook_style": "story",
                "topic_type": "tips",
                "angle": "first timer guide"
            },
            {
                "title": f"{destination}: Expectations vs Reality",
                "template": "before_after",
                "hook_style": "controversial",
                "topic_type": "tips",
                "angle": "honest review"
            },
            {
                "title": f"Is {destination} Worth It?",
                "template": "storytime",
                "hook_style": "story",
                "topic_type": "tips",
                "angle": "honest opinion"
            },
            {
                "title": f"What Locals Don't Tell You About {destination}",
                "template": "quick_tips",
                "hook_style": "curiosity",
                "topic_type": "hidden_gems",
                "angle": "insider secrets"
            },
            {
                "title": f"Day in {destination}: What to Do",
                "template": "pov_style",
                "hook_style": "story",
                "topic_type": "tips",
                "angle": "itinerary"
            },
        ]

        return ideas[:count]

    def generate_batch_scripts(self, destination: str, count: int = 5,
                               llm_callback=None) -> List[Dict]:
        """Generate scripts for batch of videos"""

        ideas = self.generate_batch_ideas(destination, count)
        scripts = []

        for i, idea in enumerate(ideas):
            print(f"[Batch] Generating script {i+1}/{count}: {idea['title']}")

            # Get template structure
            template_func = getattr(self.templates, idea["template"], None)
            if template_func:
                if idea["template"] == "ranking":
                    template = template_func(destination, "places")
                else:
                    template = template_func(destination)
            else:
                template = self.templates.quick_tips(destination, idea["hook_style"])

            # Build script prompt
            hook = TikTokTemplates.get_hook(destination, idea["hook_style"])

            script_data = {
                "id": i + 1,
                "title": idea["title"],
                "hook": hook,
                "template": template,
                "topic_type": idea["topic_type"],
                "angle": idea["angle"],
                "script": None  # Will be filled by LLM when available
            }

            # If LLM callback provided, generate full script
            if llm_callback:
                prompt = f"""Write a {template['duration']}-second TikTok script about {destination}.

Topic: {idea['title']}
Angle: {idea['angle']}
Hook (use this exactly): "{hook}"

Structure:
{json.dumps(template['sections'], indent=2)}

Requirements:
- Fast paced, punchy delivery
- Each section should flow naturally
- Conversational, energetic tone
- End with engagement CTA
- Write ONLY the spoken words

Write the script:"""
                script_data["script"] = llm_callback(prompt)

            scripts.append(script_data)

        return scripts

    def save_batch(self, destination: str, scripts: List[Dict]) -> str:
        """Save batch scripts to file"""

        batch_dir = Path(CONTENT_DIR) / "batches" / datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir.mkdir(parents=True, exist_ok=True)

        batch_data = {
            "destination": destination,
            "created": datetime.now().isoformat(),
            "count": len(scripts),
            "scripts": scripts
        }

        # Save full batch
        batch_file = batch_dir / "batch.json"
        with open(batch_file, 'w') as f:
            json.dump(batch_data, f, indent=2)

        # Save individual scripts
        for script in scripts:
            script_file = batch_dir / f"script_{script['id']:02d}.txt"
            content = f"""TITLE: {script['title']}
HOOK: {script['hook']}
ANGLE: {script['angle']}
DURATION: {script['template']['duration']} seconds

TEMPLATE:
{json.dumps(script['template']['sections'], indent=2)}

SCRIPT:
{script.get('script', '[To be generated]')}
"""
            with open(script_file, 'w') as f:
                f.write(content)

        print(f"[Batch] Saved {len(scripts)} scripts to: {batch_dir}")
        return str(batch_dir)


# ============== CLI INTERFACE ==============

def main():
    """Command line interface for social media features"""
    import argparse

    parser = argparse.ArgumentParser(description="Social Media Video Tools")
    subparsers = parser.add_subparsers(dest="command")

    # Hooks command
    hooks_parser = subparsers.add_parser("hooks", help="Generate TikTok hooks")
    hooks_parser.add_argument("destination", help="Destination name")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Generate batch of video ideas")
    batch_parser.add_argument("destination", help="Destination name")
    batch_parser.add_argument("--count", "-n", type=int, default=5, help="Number of videos")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export video for platforms")
    export_parser.add_argument("video", help="Path to video file")
    export_parser.add_argument("destination", help="Destination name")
    export_parser.add_argument("--description", "-d", default="", help="Video description")

    args = parser.parse_args()

    if args.command == "hooks":
        print(f"\nTikTok Hooks for: {args.destination}\n" + "="*50)
        hooks = TikTokTemplates.get_all_hooks(args.destination)
        for style, hook_list in hooks.items():
            print(f"\n{style.upper()}:")
            for hook in hook_list:
                print(f"  - {hook}")

    elif args.command == "batch":
        print(f"\nGenerating {args.count} video ideas for: {args.destination}\n")
        batch = BatchCreator()
        scripts = batch.generate_batch_scripts(args.destination, args.count)
        batch_dir = batch.save_batch(args.destination, scripts)
        print(f"\nScripts saved to: {batch_dir}")

    elif args.command == "export":
        print(f"\nExporting to all platforms...")
        exporter = MultiPlatformExport()
        results = exporter.export_all_platforms(
            args.video, args.destination,
            args.description or f"Amazing {args.destination} travel tips!"
        )
        for platform, data in results.items():
            print(f"\n{platform}:")
            print(f"  Caption: {data.get('caption', '')[:100]}...")
            print(f"  Hashtags: {', '.join(data.get('hashtags', []))}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
