"""
Alexandra AI - YouTube Travel Content Creator
Create publish-ready travel videos for YouTube
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from travel_config import CONTENT_DIR, SCRIPTS_DIR

# YouTube video settings
YOUTUBE_SETTINGS = {
    "shorts": {
        "resolution": "1080x1920",  # 9:16 vertical
        "max_duration": 60,
        "description": "YouTube Shorts (< 60 seconds)"
    },
    "standard": {
        "resolution": "1920x1080",  # 16:9 horizontal
        "max_duration": None,
        "description": "Standard YouTube video"
    },
    "square": {
        "resolution": "1080x1080",  # 1:1 square
        "max_duration": None,
        "description": "Square format (Instagram/Facebook)"
    }
}

class YouTubeContentCreator:
    """Create YouTube-ready travel content"""

    def __init__(self):
        self.output_dir = os.path.join(CONTENT_DIR, "youtube")
        self.projects_dir = os.path.join(self.output_dir, "projects")
        os.makedirs(self.projects_dir, exist_ok=True)

    def create_project(self, name: str, video_type: str = "standard") -> Dict:
        """Create a new YouTube video project"""
        project_id = f"yt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project_dir = os.path.join(self.projects_dir, project_id)
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(os.path.join(project_dir, "clips"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "audio"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "final"), exist_ok=True)

        project = {
            "id": project_id,
            "name": name,
            "type": video_type,
            "settings": YOUTUBE_SETTINGS.get(video_type, YOUTUBE_SETTINGS["standard"]),
            "created": datetime.now().isoformat(),
            "status": "draft",
            "clips": [],
            "script": [],
            "metadata": {
                "title": "",
                "description": "",
                "tags": [],
                "thumbnail": None
            }
        }

        # Save project file
        with open(os.path.join(project_dir, "project.json"), 'w') as f:
            json.dump(project, f, indent=2)

        return project

    def add_script_section(self, project_id: str, text: str,
                          section_type: str = "talking_head",
                          notes: str = "") -> bool:
        """
        Add a script section to the project

        section_types:
        - talking_head: Alexandra speaking to camera
        - voiceover: Voice over b-roll (future feature)
        - intro: Opening hook
        - outro: Call to action/closing
        """
        project_dir = os.path.join(self.projects_dir, project_id)
        project_file = os.path.join(project_dir, "project.json")

        if not os.path.exists(project_file):
            return False

        with open(project_file, 'r') as f:
            project = json.load(f)

        project["script"].append({
            "id": len(project["script"]) + 1,
            "type": section_type,
            "text": text,
            "notes": notes,
            "clip_generated": False,
            "clip_path": None
        })

        with open(project_file, 'w') as f:
            json.dump(project, f, indent=2)

        return True

    def set_metadata(self, project_id: str, title: str = None,
                    description: str = None, tags: List[str] = None) -> bool:
        """Set YouTube metadata for the project"""
        project_dir = os.path.join(self.projects_dir, project_id)
        project_file = os.path.join(project_dir, "project.json")

        if not os.path.exists(project_file):
            return False

        with open(project_file, 'r') as f:
            project = json.load(f)

        if title:
            project["metadata"]["title"] = title
        if description:
            project["metadata"]["description"] = description
        if tags:
            project["metadata"]["tags"] = tags

        with open(project_file, 'w') as f:
            json.dump(project, f, indent=2)

        return True

    def get_project(self, project_id: str) -> Optional[Dict]:
        """Get project details"""
        project_file = os.path.join(self.projects_dir, project_id, "project.json")

        if not os.path.exists(project_file):
            return None

        with open(project_file, 'r') as f:
            return json.load(f)

    def list_projects(self) -> List[Dict]:
        """List all projects"""
        projects = []
        for name in os.listdir(self.projects_dir):
            project_file = os.path.join(self.projects_dir, name, "project.json")
            if os.path.exists(project_file):
                with open(project_file, 'r') as f:
                    project = json.load(f)
                    projects.append({
                        "id": project["id"],
                        "name": project["name"],
                        "type": project["type"],
                        "status": project["status"],
                        "clips": len(project.get("clips", [])),
                        "script_sections": len(project.get("script", []))
                    })
        return projects

    def export_script_for_generation(self, project_id: str) -> str:
        """Export script as text file for batch video generation"""
        project = self.get_project(project_id)
        if not project:
            return None

        script_file = os.path.join(self.projects_dir, project_id, "script_for_generation.txt")

        with open(script_file, 'w') as f:
            for section in project.get("script", []):
                f.write(section["text"] + "\n")

        return script_file

    def generate_description(self, project_id: str, destination: str,
                            include_timestamps: bool = True) -> str:
        """Generate YouTube description template"""
        project = self.get_project(project_id)
        if not project:
            return ""

        description = f"""🌍 {project['metadata'].get('title', 'Travel Video')}

In this video, I'm sharing everything you need to know about {destination}!

📍 What's covered:
"""
        # Add timestamps based on script sections
        if include_timestamps and project.get("script"):
            timestamp = 0
            for i, section in enumerate(project["script"]):
                minutes = timestamp // 60
                seconds = timestamp % 60
                description += f"{minutes}:{seconds:02d} - Section {i+1}\n"
                # Estimate ~30 seconds per section
                timestamp += 30

        description += f"""
💡 Travel Tips:
• [Add your tips here]

📱 Follow me:
• [Your social links]

🎒 My Travel Gear:
• [Affiliate links if applicable]

#travel #{destination.lower().replace(' ', '')} #traveltips #travelguide

---
🤖 Created with Alexandra AI
"""
        return description

    def generate_tags(self, destination: str, video_type: str = "guide") -> List[str]:
        """Generate YouTube tags for travel video"""
        base_tags = [
            "travel",
            "travel guide",
            "travel tips",
            "travel vlog",
            destination.lower(),
            f"{destination.lower()} travel",
            f"{destination.lower()} guide",
            f"things to do in {destination.lower()}",
            f"visit {destination.lower()}",
            f"{destination.lower()} travel tips",
        ]

        type_tags = {
            "guide": ["travel guide", "destination guide", "where to go"],
            "tips": ["travel tips", "travel hacks", "travel advice"],
            "food": ["food tour", "where to eat", "best restaurants", "food guide"],
            "budget": ["budget travel", "cheap travel", "travel on a budget"],
            "itinerary": ["travel itinerary", "trip planning", "day by day"]
        }

        tags = base_tags + type_tags.get(video_type, [])

        # Remove duplicates and limit to 500 chars total (YouTube limit)
        unique_tags = list(dict.fromkeys(tags))
        return unique_tags[:30]  # YouTube allows up to 500 chars, ~30 tags is safe


class TravelVideoTemplates:
    """Pre-built templates for travel videos"""

    @staticmethod
    def destination_guide(destination: str) -> List[Dict]:
        """Full destination guide template"""
        return [
            {"type": "intro", "text": f"Have you ever dreamed of visiting {destination}? In this video, I'm going to show you everything you need to know to plan your perfect trip."},
            {"type": "talking_head", "text": f"First, let's talk about the best time to visit {destination}. This is super important for your trip planning."},
            {"type": "talking_head", "text": f"Now let's cover the top attractions and must-see places in {destination}. These are the spots you absolutely cannot miss."},
            {"type": "talking_head", "text": f"One of my favorite parts of any trip is the food, and {destination} does not disappoint. Here are my restaurant recommendations."},
            {"type": "talking_head", "text": f"Let's talk about where to stay in {destination}. I'll break down the best neighborhoods and accommodation options."},
            {"type": "talking_head", "text": f"Budget time! Here's what you can expect to spend per day in {destination}, and some money-saving tips."},
            {"type": "talking_head", "text": f"Before I wrap up, here are some insider tips that most tourists don't know about {destination}."},
            {"type": "outro", "text": f"That's my complete guide to {destination}! If you found this helpful, make sure to subscribe for more travel content. Drop a comment below if you have any questions, and I'll see you in the next video!"},
        ]

    @staticmethod
    def top_tips(topic: str, count: int = 10) -> List[Dict]:
        """Tips video template"""
        sections = [
            {"type": "intro", "text": f"I'm sharing my top {count} {topic} tips that will totally change how you travel. Let's get into it!"}
        ]

        for i in range(1, count + 1):
            sections.append({
                "type": "talking_head",
                "text": f"[TIP {i}: Add your tip here]"
            })

        sections.append({
            "type": "outro",
            "text": f"Those are my top {count} {topic} tips! Which one was your favorite? Let me know in the comments!"
        })

        return sections

    @staticmethod
    def comparison(dest1: str, dest2: str) -> List[Dict]:
        """Destination comparison template"""
        return [
            {"type": "intro", "text": f"{dest1} or {dest2}? If you're trying to decide between these two amazing destinations, this video will help you choose."},
            {"type": "talking_head", "text": f"Let's start with a quick overview of both {dest1} and {dest2}, so you know what each destination offers."},
            {"type": "talking_head", "text": f"When it comes to beaches and nature, here's how {dest1} and {dest2} compare."},
            {"type": "talking_head", "text": f"Food lovers, this one's for you. Let's talk about the culinary scenes in both destinations."},
            {"type": "talking_head", "text": f"Budget is always a factor. Here's the cost comparison between {dest1} and {dest2}."},
            {"type": "talking_head", "text": f"What about nightlife and entertainment? Here's the breakdown."},
            {"type": "talking_head", "text": f"So which should YOU choose? Here's my recommendation based on different travel styles."},
            {"type": "outro", "text": f"Have you been to {dest1} or {dest2}? I'd love to hear about your experience in the comments!"},
        ]

    @staticmethod
    def youtube_short(destination: str, hook: str) -> List[Dict]:
        """YouTube Short template (under 60 seconds)"""
        return [
            {"type": "intro", "text": hook},  # 5-10 seconds
            {"type": "talking_head", "text": f"[Quick tip 1 about {destination}]"},  # 10-15 seconds
            {"type": "talking_head", "text": f"[Quick tip 2 about {destination}]"},  # 10-15 seconds
            {"type": "talking_head", "text": f"[Quick tip 3 about {destination}]"},  # 10-15 seconds
            {"type": "outro", "text": "Follow for more travel tips!"},  # 5 seconds
        ]


def create_youtube_project_from_template(name: str, template_type: str, **kwargs) -> Dict:
    """Quick helper to create a project with template"""
    creator = YouTubeContentCreator()
    templates = TravelVideoTemplates()

    # Determine video type
    video_type = "shorts" if template_type == "short" else "standard"

    # Create project
    project = creator.create_project(name, video_type)

    # Get template
    if template_type == "guide":
        sections = templates.destination_guide(kwargs.get("destination", "Unknown"))
    elif template_type == "tips":
        sections = templates.top_tips(kwargs.get("topic", "travel"), kwargs.get("count", 10))
    elif template_type == "comparison":
        sections = templates.comparison(kwargs.get("dest1", "A"), kwargs.get("dest2", "B"))
    elif template_type == "short":
        sections = templates.youtube_short(
            kwargs.get("destination", "travel"),
            kwargs.get("hook", "You won't believe this travel hack!")
        )
    else:
        sections = []

    # Add sections to project
    for section in sections:
        creator.add_script_section(
            project["id"],
            section["text"],
            section["type"]
        )

    # Set metadata
    destination = kwargs.get("destination", kwargs.get("topic", "Travel"))
    creator.set_metadata(
        project["id"],
        title=name,
        tags=creator.generate_tags(destination, template_type)
    )

    return creator.get_project(project["id"])


if __name__ == "__main__":
    # Demo
    print("Creating YouTube travel project...")

    project = create_youtube_project_from_template(
        name="Complete Barcelona Travel Guide 2025",
        template_type="guide",
        destination="Barcelona"
    )

    print(f"\nProject created: {project['id']}")
    print(f"Name: {project['name']}")
    print(f"Sections: {len(project['script'])}")
    print(f"\nScript preview:")
    for i, section in enumerate(project['script'][:3]):
        print(f"  {i+1}. [{section['type']}] {section['text'][:50]}...")

    creator = YouTubeContentCreator()
    print(f"\nGenerated tags: {project['metadata']['tags'][:10]}...")
    print(f"\nDescription template:")
    print(creator.generate_description(project['id'], "Barcelona")[:500])
