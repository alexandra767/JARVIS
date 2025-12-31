"""
Alexandra AI - Travel Content Creator
Generate travel videos and content with your avatar
"""

import os
import sys
import json
from typing import List, Dict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from travel_config import CONTENT_DIR, SCRIPTS_DIR, CONTENT_TEMPLATES

class TravelContentCreator:
    """Create travel content scripts and videos"""

    def __init__(self):
        self.content_dir = CONTENT_DIR
        self.scripts_dir = SCRIPTS_DIR

    def generate_script(self, template_type: str, **kwargs) -> Dict:
        """
        Generate a content script from template

        template_types:
        - destination_review
        - travel_tips
        - comparison
        - itinerary
        - packing_list
        """
        template = CONTENT_TEMPLATES.get(template_type)
        if not template:
            return {"error": f"Unknown template: {template_type}"}

        script = {
            "type": template_type,
            "created": datetime.now().isoformat(),
            "title": template["title"].format(**kwargs),
            "sections": []
        }

        if template_type == "destination_review":
            destination = kwargs.get("destination", "Unknown")
            script["sections"] = self._generate_review_sections(destination, template["sections"])

        elif template_type == "travel_tips":
            topic = kwargs.get("topic", "General")
            count = kwargs.get("count", 10)
            script["sections"] = self._generate_tips_sections(topic, count)

        elif template_type == "comparison":
            dest1 = kwargs.get("destination1", "Place A")
            dest2 = kwargs.get("destination2", "Place B")
            script["sections"] = self._generate_comparison_sections(dest1, dest2, template["sections"])

        elif template_type == "itinerary":
            destination = kwargs.get("destination", "Unknown")
            duration = kwargs.get("duration", "3 days")
            script["sections"] = self._generate_itinerary_sections(destination, duration)

        elif template_type == "packing_list":
            destination = kwargs.get("destination", "")
            trip_type = kwargs.get("trip_type", "general")
            script["sections"] = self._generate_packing_sections(destination, trip_type, template["categories"])

        return script

    def _generate_review_sections(self, destination: str, section_names: List[str]) -> List[Dict]:
        """Generate review script sections"""
        sections = []
        for name in section_names:
            sections.append({
                "name": name,
                "prompt": f"Write about {name} for {destination}",
                "script": f"[WRITE: {name} for {destination}]",
                "duration_hint": "30-60 seconds"
            })
        return sections

    def _generate_tips_sections(self, topic: str, count: int) -> List[Dict]:
        """Generate tips list sections"""
        sections = [{
            "name": "intro",
            "script": f"[WRITE: Introduction to {topic} tips]",
            "duration_hint": "15-20 seconds"
        }]

        for i in range(1, count + 1):
            sections.append({
                "name": f"tip_{i}",
                "script": f"[WRITE: Tip #{i} about {topic}]",
                "duration_hint": "15-30 seconds"
            })

        sections.append({
            "name": "outro",
            "script": f"[WRITE: Wrap up and call to action]",
            "duration_hint": "15-20 seconds"
        })

        return sections

    def _generate_comparison_sections(self, dest1: str, dest2: str, section_names: List[str]) -> List[Dict]:
        """Generate comparison script sections"""
        sections = []
        for name in section_names:
            sections.append({
                "name": name,
                "prompt": f"Compare {dest1} and {dest2}: {name}",
                "script": f"[WRITE: {name} comparing {dest1} vs {dest2}]",
                "duration_hint": "45-60 seconds"
            })
        return sections

    def _generate_itinerary_sections(self, destination: str, duration: str) -> List[Dict]:
        """Generate itinerary script sections"""
        sections = [{
            "name": "intro",
            "script": f"[WRITE: Introduction to {duration} in {destination}]",
            "duration_hint": "20-30 seconds"
        }]

        # Parse duration (e.g., "3 days" -> 3)
        try:
            days = int(duration.split()[0])
        except:
            days = 3

        for day in range(1, days + 1):
            sections.append({
                "name": f"day_{day}",
                "script": f"[WRITE: Day {day} itinerary for {destination}]",
                "duration_hint": "60-90 seconds"
            })

        sections.append({
            "name": "tips",
            "script": f"[WRITE: General tips for this itinerary]",
            "duration_hint": "30-45 seconds"
        })

        sections.append({
            "name": "outro",
            "script": f"[WRITE: Closing thoughts on {destination}]",
            "duration_hint": "15-20 seconds"
        })

        return sections

    def _generate_packing_sections(self, destination: str, trip_type: str, categories: List[str]) -> List[Dict]:
        """Generate packing list script sections"""
        sections = [{
            "name": "intro",
            "script": f"[WRITE: Introduction to packing for {trip_type} trip to {destination}]",
            "duration_hint": "15-20 seconds"
        }]

        for category in categories:
            sections.append({
                "name": category.lower().replace(" ", "_"),
                "script": f"[WRITE: {category} items for {trip_type} trip]",
                "duration_hint": "20-30 seconds"
            })

        return sections

    def save_script(self, script: Dict, filename: str = None) -> str:
        """Save script to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{script['type']}_{timestamp}.json"

        filepath = os.path.join(self.scripts_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(script, f, indent=2)

        return filepath

    def load_script(self, filename: str) -> Dict:
        """Load script from file"""
        filepath = os.path.join(self.scripts_dir, filename)

        with open(filepath, 'r') as f:
            return json.load(f)

    def script_to_video_lines(self, script: Dict) -> List[str]:
        """Convert script to lines for video generation"""
        lines = []

        for section in script.get("sections", []):
            text = section.get("script", "")
            if text and not text.startswith("[WRITE:"):
                lines.append(text)

        return lines

    def generate_video_batch_file(self, script: Dict) -> str:
        """Generate a batch file for video creation"""
        lines = self.script_to_video_lines(script)

        if not lines:
            # Return placeholder lines if script not filled in
            lines = [section.get("script", "") for section in script.get("sections", [])]

        batch_file = os.path.join(self.scripts_dir, f"{script['type']}_batch.txt")

        with open(batch_file, 'w') as f:
            for line in lines:
                f.write(line + "\n")

        return batch_file


class TravelVideoProject:
    """Manage a travel video project"""

    def __init__(self, name: str):
        self.name = name
        self.project_dir = os.path.join(CONTENT_DIR, name)
        os.makedirs(self.project_dir, exist_ok=True)

        self.config_file = os.path.join(self.project_dir, "project.json")
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load project configuration"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {
            "name": self.name,
            "created": datetime.now().isoformat(),
            "scripts": [],
            "videos": [],
            "status": "draft"
        }

    def _save_config(self):
        """Save project configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def add_script(self, script: Dict):
        """Add a script to the project"""
        script_file = os.path.join(self.project_dir, f"script_{len(self.config['scripts']) + 1}.json")

        with open(script_file, 'w') as f:
            json.dump(script, f, indent=2)

        self.config["scripts"].append(script_file)
        self._save_config()

    def get_all_lines(self) -> List[str]:
        """Get all script lines for the project"""
        all_lines = []

        for script_file in self.config["scripts"]:
            if os.path.exists(script_file):
                with open(script_file, 'r') as f:
                    script = json.load(f)
                    for section in script.get("sections", []):
                        text = section.get("script", "")
                        if text and not text.startswith("[WRITE:"):
                            all_lines.append(text)

        return all_lines


# Quick content generators

def quick_destination_intro(destination: str, country: str) -> str:
    """Generate a quick destination intro script"""
    return f"""
Hey everyone! Today I want to talk about one of my favorite destinations: {destination}, {country}.

If you're thinking about visiting {destination}, you're going to love this video.
I'll share my personal experiences, favorite spots, and tips that will make your trip amazing.

Let's dive in!
    """.strip()

def quick_tip_video(topic: str, tips: List[str]) -> str:
    """Generate a quick tips video script"""
    script = f"Here are my top {len(tips)} {topic} tips:\n\n"

    for i, tip in enumerate(tips, 1):
        script += f"Number {i}: {tip}\n\n"

    script += "Those are my tips! Let me know in the comments if you have any questions."

    return script

def quick_outro() -> str:
    """Generate a standard outro"""
    return """
Thanks so much for watching! If you found this helpful, give it a thumbs up and subscribe for more travel content.

Drop a comment below and let me know where you want me to cover next.

Until next time, happy travels!
    """.strip()


if __name__ == "__main__":
    # Demo
    creator = TravelContentCreator()

    # Generate a destination review script
    script = creator.generate_script(
        "destination_review",
        destination="Barcelona"
    )

    filepath = creator.save_script(script)
    print(f"Created script: {filepath}")
    print(f"\nScript has {len(script['sections'])} sections:")
    for section in script["sections"]:
        print(f"  - {section['name']}: {section['duration_hint']}")

    # Generate a tips video script
    tips_script = creator.generate_script(
        "travel_tips",
        topic="Packing Light",
        count=5
    )
    creator.save_script(tips_script)

    print(f"\n\nFiles saved to: {SCRIPTS_DIR}")
    print("\nEdit the scripts to add your content, then use batch video generation!")
