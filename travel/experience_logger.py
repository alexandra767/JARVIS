"""
Alexandra AI - Travel Experience Logger
Comprehensive logging of personal travel experiences for training data
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from travel_config import KNOWLEDGE_DIR

class TravelExperienceLogger:
    """Log detailed travel experiences for training Alexandra"""

    def __init__(self):
        self.experiences_file = os.path.join(KNOWLEDGE_DIR, "detailed_experiences.json")
        self.experiences = self._load_experiences()

    def _load_experiences(self) -> Dict:
        """Load saved experiences"""
        if os.path.exists(self.experiences_file):
            with open(self.experiences_file, 'r') as f:
                return json.load(f)
        return {
            "trips": [],
            "places": {},
            "tips": [],
            "stories": [],
            "ratings": {}
        }

    def _save_experiences(self):
        """Save experiences to file"""
        os.makedirs(os.path.dirname(self.experiences_file), exist_ok=True)
        with open(self.experiences_file, 'w') as f:
            json.dump(self.experiences, f, indent=2)

    # ============== TRIP LOGGING ==============

    def log_trip(self,
                 destination: str,
                 country: str,
                 dates: str,
                 duration_days: int,
                 trip_type: str = "leisure",
                 companions: str = "solo",
                 budget_total: float = 0,
                 highlights: List[str] = None,
                 lowlights: List[str] = None,
                 would_return: bool = True,
                 overall_rating: int = 8,
                 notes: str = "") -> str:
        """
        Log a complete trip experience

        trip_type: leisure, business, adventure, cultural, beach, city_break, road_trip
        companions: solo, couple, family, friends, group
        """
        trip_id = f"trip_{len(self.experiences['trips']) + 1}"

        trip = {
            "id": trip_id,
            "destination": destination,
            "country": country,
            "dates": dates,
            "duration_days": duration_days,
            "trip_type": trip_type,
            "companions": companions,
            "budget": {
                "total": budget_total,
                "per_day": budget_total / duration_days if duration_days > 0 else 0,
                "breakdown": {}
            },
            "highlights": highlights or [],
            "lowlights": lowlights or [],
            "would_return": would_return,
            "overall_rating": overall_rating,
            "notes": notes,
            "places_visited": [],
            "logged": datetime.now().isoformat()
        }

        self.experiences["trips"].append(trip)
        self._save_experiences()
        return trip_id

    def add_budget_breakdown(self, trip_id: str, category: str, amount: float):
        """Add budget breakdown to a trip"""
        for trip in self.experiences["trips"]:
            if trip["id"] == trip_id:
                trip["budget"]["breakdown"][category] = amount
                self._save_experiences()
                return True
        return False

    # ============== PLACE LOGGING ==============

    def log_place(self,
                  name: str,
                  destination: str,
                  place_type: str,
                  rating: int = 8,
                  price_level: str = "$$",
                  visited_date: str = "",
                  what_i_loved: str = "",
                  what_i_didnt_like: str = "",
                  must_try: List[str] = None,
                  tips: List[str] = None,
                  would_recommend: bool = True,
                  best_for: List[str] = None,
                  address: str = "",
                  notes: str = "") -> str:
        """
        Log a specific place (restaurant, hotel, attraction, etc.)

        place_type: restaurant, hotel, attraction, bar, cafe, museum,
                   beach, park, viewpoint, market, shop, neighborhood
        price_level: $, $$, $$$, $$$$
        best_for: couples, families, solo, groups, instagram, foodies, etc.
        """
        place_id = f"place_{len(self.experiences['places']) + 1}"

        place = {
            "id": place_id,
            "name": name,
            "destination": destination,
            "type": place_type,
            "rating": rating,
            "price_level": price_level,
            "visited_date": visited_date,
            "what_i_loved": what_i_loved,
            "what_i_didnt_like": what_i_didnt_like,
            "must_try": must_try or [],
            "tips": tips or [],
            "would_recommend": would_recommend,
            "best_for": best_for or [],
            "address": address,
            "notes": notes,
            "logged": datetime.now().isoformat()
        }

        # Organize by destination
        if destination not in self.experiences["places"]:
            self.experiences["places"][destination] = []

        self.experiences["places"][destination].append(place)
        self._save_experiences()
        return place_id

    # ============== STORY LOGGING ==============

    def log_story(self,
                  title: str,
                  destination: str,
                  story: str,
                  story_type: str = "experience",
                  lesson_learned: str = "",
                  tags: List[str] = None) -> str:
        """
        Log a travel story or anecdote

        story_type: experience, mishap, adventure, cultural, funny,
                   heartwarming, lesson, discovery
        """
        story_id = f"story_{len(self.experiences['stories']) + 1}"

        story_entry = {
            "id": story_id,
            "title": title,
            "destination": destination,
            "story": story,
            "type": story_type,
            "lesson_learned": lesson_learned,
            "tags": tags or [],
            "logged": datetime.now().isoformat()
        }

        self.experiences["stories"].append(story_entry)
        self._save_experiences()
        return story_id

    # ============== TIP LOGGING ==============

    def log_tip(self,
                tip: str,
                category: str,
                destination: str = "",
                importance: str = "helpful",
                source: str = "personal") -> str:
        """
        Log a travel tip

        category: packing, money, safety, transport, food, accommodation,
                 booking, culture, photography, general
        importance: essential, helpful, nice_to_know
        source: personal, local, research
        """
        tip_id = f"tip_{len(self.experiences['tips']) + 1}"

        tip_entry = {
            "id": tip_id,
            "tip": tip,
            "category": category,
            "destination": destination,  # Empty = general tip
            "importance": importance,
            "source": source,
            "logged": datetime.now().isoformat()
        }

        self.experiences["tips"].append(tip_entry)
        self._save_experiences()
        return tip_id

    # ============== RATINGS ==============

    def rate_destination(self,
                         destination: str,
                         overall: int,
                         food: int = None,
                         culture: int = None,
                         nightlife: int = None,
                         nature: int = None,
                         safety: int = None,
                         value: int = None,
                         accessibility: int = None,
                         friendliness: int = None):
        """Rate a destination on multiple factors (1-10 scale)"""
        self.experiences["ratings"][destination] = {
            "overall": overall,
            "food": food,
            "culture": culture,
            "nightlife": nightlife,
            "nature": nature,
            "safety": safety,
            "value": value,
            "accessibility": accessibility,
            "friendliness": friendliness,
            "rated": datetime.now().isoformat()
        }
        self._save_experiences()

    # ============== RETRIEVAL ==============

    def get_trip(self, trip_id: str) -> Optional[Dict]:
        """Get a specific trip"""
        for trip in self.experiences["trips"]:
            if trip["id"] == trip_id:
                return trip
        return None

    def get_trips_by_destination(self, destination: str) -> List[Dict]:
        """Get all trips to a destination"""
        return [t for t in self.experiences["trips"]
                if destination.lower() in t["destination"].lower()]

    def get_places_by_destination(self, destination: str) -> List[Dict]:
        """Get all logged places in a destination"""
        return self.experiences["places"].get(destination, [])

    def get_places_by_type(self, place_type: str, destination: str = None) -> List[Dict]:
        """Get places by type, optionally filtered by destination"""
        places = []
        for dest, dest_places in self.experiences["places"].items():
            if destination and destination.lower() not in dest.lower():
                continue
            for place in dest_places:
                if place["type"] == place_type:
                    places.append(place)
        return places

    def get_tips_by_category(self, category: str) -> List[Dict]:
        """Get tips by category"""
        return [t for t in self.experiences["tips"] if t["category"] == category]

    def get_destination_tips(self, destination: str) -> List[Dict]:
        """Get all tips for a destination"""
        return [t for t in self.experiences["tips"]
                if destination.lower() in t.get("destination", "").lower()]

    def get_stories_by_destination(self, destination: str) -> List[Dict]:
        """Get stories about a destination"""
        return [s for s in self.experiences["stories"]
                if destination.lower() in s["destination"].lower()]

    def get_all_destinations(self) -> List[str]:
        """Get list of all destinations you've logged"""
        destinations = set()
        for trip in self.experiences["trips"]:
            destinations.add(trip["destination"])
        for dest in self.experiences["places"].keys():
            destinations.add(dest)
        return sorted(list(destinations))

    def get_stats(self) -> Dict:
        """Get statistics about logged experiences"""
        return {
            "total_trips": len(self.experiences["trips"]),
            "destinations_visited": len(self.get_all_destinations()),
            "places_logged": sum(len(p) for p in self.experiences["places"].values()),
            "stories_shared": len(self.experiences["stories"]),
            "tips_collected": len(self.experiences["tips"]),
            "destinations_rated": len(self.experiences["ratings"])
        }

    # ============== EXPORT FOR TRAINING ==============

    def export_for_training(self) -> Dict:
        """Export all experiences in a format ready for training data generation"""
        return {
            "trips": self.experiences["trips"],
            "places": self.experiences["places"],
            "stories": self.experiences["stories"],
            "tips": self.experiences["tips"],
            "ratings": self.experiences["ratings"],
            "destinations": self.get_all_destinations(),
            "stats": self.get_stats()
        }


# Singleton
_logger = None

def get_experience_logger():
    global _logger
    if _logger is None:
        _logger = TravelExperienceLogger()
    return _logger


if __name__ == "__main__":
    # Demo
    logger = TravelExperienceLogger()

    print("Travel Experience Logger Demo\n")

    # Log a sample trip
    trip_id = logger.log_trip(
        destination="Barcelona",
        country="Spain",
        dates="March 2024",
        duration_days=5,
        trip_type="city_break",
        companions="couple",
        budget_total=1500,
        highlights=["La Sagrada Familia", "Gothic Quarter", "Beach sunset"],
        lowlights=["Crowded tourist areas", "Pickpocket attempt"],
        would_return=True,
        overall_rating=9,
        notes="Amazing city, perfect mix of culture and beach"
    )
    print(f"Logged trip: {trip_id}")

    # Log a place
    place_id = logger.log_place(
        name="Bar Cañete",
        destination="Barcelona",
        place_type="restaurant",
        rating=9,
        price_level="$$$",
        what_i_loved="Best tapas I've ever had. The ham and seafood were incredible.",
        must_try=["Jamón ibérico", "Gambas al ajillo", "Patatas bravas"],
        tips=["Make a reservation", "Sit at the bar for best experience"],
        best_for=["foodies", "couples", "special occasion"]
    )
    print(f"Logged place: {place_id}")

    # Log a story
    story_id = logger.log_story(
        title="The Pickpocket Near La Rambla",
        destination="Barcelona",
        story="I was walking down La Rambla when I felt someone bump into me. Instinctively checked my pocket - phone was being pulled out! Grabbed it back and the guy just walked away like nothing happened. Lesson learned: use a money belt in crowded tourist areas.",
        story_type="mishap",
        lesson_learned="Always be aware of your belongings in crowded tourist areas",
        tags=["safety", "pickpocket", "tourist-trap"]
    )
    print(f"Logged story: {story_id}")

    # Log tips
    logger.log_tip(
        tip="The metro is the fastest way to get around. Buy a T-Casual card for 10 trips.",
        category="transport",
        destination="Barcelona",
        importance="helpful"
    )

    logger.log_tip(
        tip="Always carry a photocopy of your passport, not the original.",
        category="safety",
        importance="essential"
    )

    # Rate destination
    logger.rate_destination(
        destination="Barcelona",
        overall=9,
        food=9,
        culture=10,
        nightlife=8,
        nature=7,
        safety=6,
        value=7,
        friendliness=8
    )

    print(f"\nStats: {logger.get_stats()}")
    print(f"Destinations: {logger.get_all_destinations()}")
