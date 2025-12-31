"""
Alexandra AI - Travel Knowledge Base
Store and retrieve your personal travel knowledge
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional

from travel_config import KNOWLEDGE_DIR, TRAVEL_CATEGORIES

class TravelKnowledge:
    """Manage personal travel knowledge base"""

    def __init__(self):
        self.destinations_file = os.path.join(KNOWLEDGE_DIR, "destinations.json")
        self.experiences_file = os.path.join(KNOWLEDGE_DIR, "experiences.json")
        self.tips_file = os.path.join(KNOWLEDGE_DIR, "tips.json")
        self.favorites_file = os.path.join(KNOWLEDGE_DIR, "favorites.json")

        # Load existing data
        self.destinations = self._load_json(self.destinations_file, {})
        self.experiences = self._load_json(self.experiences_file, [])
        self.tips = self._load_json(self.tips_file, {})
        self.favorites = self._load_json(self.favorites_file, {})

    def _load_json(self, filepath, default):
        """Load JSON file or return default"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default

    def _save_json(self, filepath, data):
        """Save data to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    # ============== Destinations ==============

    def add_destination(self, name: str, country: str, data: Dict):
        """
        Add a destination you've visited

        data should include:
        - visited_date: when you went
        - duration: how long you stayed
        - rating: 1-10
        - highlights: list of best things
        - lowlights: list of negatives
        - tips: your personal tips
        - would_return: bool
        - best_for: list (beaches, nightlife, culture, etc.)
        - budget_per_day: approximate daily cost
        - accommodation: where you stayed
        - restaurants: favorites
        - activities: what you did
        """
        key = f"{name.lower()}_{country.lower()}"
        self.destinations[key] = {
            "name": name,
            "country": country,
            "added": datetime.now().isoformat(),
            **data
        }
        self._save_json(self.destinations_file, self.destinations)
        return f"Added {name}, {country} to your travel knowledge"

    def get_destination(self, name: str) -> Optional[Dict]:
        """Get info about a destination"""
        name_lower = name.lower()
        for key, dest in self.destinations.items():
            if name_lower in key or name_lower in dest.get('name', '').lower():
                return dest
        return None

    def search_destinations(self, criteria: Dict) -> List[Dict]:
        """
        Search destinations by criteria

        criteria can include:
        - country: str
        - min_rating: int
        - best_for: str (activity type)
        - max_budget: float
        - would_return: bool
        """
        results = []
        for dest in self.destinations.values():
            match = True

            if 'country' in criteria:
                if criteria['country'].lower() not in dest.get('country', '').lower():
                    match = False

            if 'min_rating' in criteria:
                if dest.get('rating', 0) < criteria['min_rating']:
                    match = False

            if 'best_for' in criteria:
                if criteria['best_for'] not in dest.get('best_for', []):
                    match = False

            if 'max_budget' in criteria:
                if dest.get('budget_per_day', float('inf')) > criteria['max_budget']:
                    match = False

            if 'would_return' in criteria:
                if dest.get('would_return') != criteria['would_return']:
                    match = False

            if match:
                results.append(dest)

        return sorted(results, key=lambda x: x.get('rating', 0), reverse=True)

    # ============== Experiences ==============

    def add_experience(self, destination: str, title: str, story: str,
                      category: str = "general", rating: int = None):
        """Add a travel experience/story"""
        experience = {
            "id": len(self.experiences) + 1,
            "destination": destination,
            "title": title,
            "story": story,
            "category": category,
            "rating": rating,
            "added": datetime.now().isoformat()
        }
        self.experiences.append(experience)
        self._save_json(self.experiences_file, self.experiences)
        return f"Added experience: {title}"

    def get_experiences(self, destination: str = None, category: str = None) -> List[Dict]:
        """Get experiences, optionally filtered"""
        results = self.experiences

        if destination:
            results = [e for e in results if destination.lower() in e.get('destination', '').lower()]

        if category:
            results = [e for e in results if category.lower() == e.get('category', '').lower()]

        return results

    # ============== Tips ==============

    def add_tip(self, category: str, tip: str, destination: str = None):
        """Add a travel tip"""
        if category not in self.tips:
            self.tips[category] = []

        tip_entry = {
            "tip": tip,
            "destination": destination,
            "added": datetime.now().isoformat()
        }
        self.tips[category].append(tip_entry)
        self._save_json(self.tips_file, self.tips)
        return f"Added tip to {category}"

    def get_tips(self, category: str = None, destination: str = None) -> List[Dict]:
        """Get tips, optionally filtered"""
        if category and category in self.tips:
            tips = self.tips[category]
        else:
            tips = [tip for cat_tips in self.tips.values() for tip in cat_tips]

        if destination:
            tips = [t for t in tips if destination.lower() in (t.get('destination') or '').lower()]

        return tips

    # ============== Favorites ==============

    def add_favorite(self, category: str, name: str, location: str, notes: str = ""):
        """Add a favorite place (restaurant, hotel, activity)"""
        if category not in self.favorites:
            self.favorites[category] = []

        favorite = {
            "name": name,
            "location": location,
            "notes": notes,
            "added": datetime.now().isoformat()
        }
        self.favorites[category].append(favorite)
        self._save_json(self.favorites_file, self.favorites)
        return f"Added {name} to {category} favorites"

    def get_favorites(self, category: str = None, location: str = None) -> Dict:
        """Get favorites, optionally filtered"""
        if category:
            favs = {category: self.favorites.get(category, [])}
        else:
            favs = self.favorites

        if location:
            filtered = {}
            for cat, items in favs.items():
                filtered_items = [i for i in items if location.lower() in i.get('location', '').lower()]
                if filtered_items:
                    filtered[cat] = filtered_items
            favs = filtered

        return favs

    # ============== RAG Integration ==============

    def build_context(self, query: str, max_items: int = 5) -> str:
        """Build context string for RAG based on query"""
        context_parts = []
        query_lower = query.lower()

        # Search destinations
        for dest in self.destinations.values():
            if any(term in query_lower for term in [
                dest.get('name', '').lower(),
                dest.get('country', '').lower()
            ]):
                context_parts.append(f"Destination: {dest.get('name')}, {dest.get('country')}")
                context_parts.append(f"  Rating: {dest.get('rating', 'N/A')}/10")
                if dest.get('highlights'):
                    context_parts.append(f"  Highlights: {', '.join(dest.get('highlights', []))}")
                if dest.get('tips'):
                    context_parts.append(f"  Tips: {dest.get('tips')}")

        # Search experiences
        relevant_exp = [e for e in self.experiences
                       if query_lower in e.get('destination', '').lower()
                       or query_lower in e.get('story', '').lower()][:max_items]

        for exp in relevant_exp:
            context_parts.append(f"Experience at {exp.get('destination')}: {exp.get('title')}")
            context_parts.append(f"  {exp.get('story')[:200]}...")

        # Search tips
        for category, tips in self.tips.items():
            if category.lower() in query_lower or query_lower in category.lower():
                context_parts.append(f"Tips for {category}:")
                for tip in tips[:3]:
                    context_parts.append(f"  - {tip.get('tip')}")

        return "\n".join(context_parts) if context_parts else ""

    # ============== Stats ==============

    def stats(self) -> Dict:
        """Get knowledge base statistics"""
        countries = set(d.get('country') for d in self.destinations.values())
        return {
            "destinations": len(self.destinations),
            "countries": len(countries),
            "experiences": len(self.experiences),
            "tips": sum(len(t) for t in self.tips.values()),
            "favorites": sum(len(f) for f in self.favorites.values())
        }


# Singleton instance
_knowledge = None

def get_travel_knowledge():
    global _knowledge
    if _knowledge is None:
        _knowledge = TravelKnowledge()
    return _knowledge


if __name__ == "__main__":
    # Test/Demo
    tk = TravelKnowledge()

    # Add sample data
    tk.add_destination(
        name="Barcelona",
        country="Spain",
        data={
            "visited_date": "2023-06",
            "duration": "5 days",
            "rating": 9,
            "highlights": ["La Sagrada Familia", "Gothic Quarter", "Beach", "Food"],
            "lowlights": ["Crowded in summer", "Pickpockets"],
            "tips": "Book Sagrada Familia tickets weeks in advance. Visit the beach early morning.",
            "would_return": True,
            "best_for": ["culture", "food", "nightlife", "beaches"],
            "budget_per_day": 150,
            "accommodation": "Hotel in Eixample",
            "restaurants": ["Can Paixano", "Bar del Pla", "La Boqueria market"],
            "activities": ["Walking tour", "Beach day", "Gaudi architecture", "Tapas crawl"]
        }
    )

    tk.add_experience(
        destination="Barcelona",
        title="Getting Lost in the Gothic Quarter",
        story="Wandering through the narrow medieval streets of the Gothic Quarter was magical. We stumbled upon a tiny plaza with live flamenco...",
        category="culture"
    )

    tk.add_tip("packing", "Always bring a portable charger for long days of sightseeing")
    tk.add_tip("food", "Eat dinner late in Spain (9-10pm) to experience the local culture", "Spain")

    tk.add_favorite("restaurants", "Can Paixano", "Barcelona", "Best cava and sandwiches. Always packed!")

    print("Travel Knowledge Stats:", tk.stats())
    print("\nBarcelona info:", tk.get_destination("Barcelona"))
