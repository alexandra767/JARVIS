"""
Alexandra AI - Travel Research
Research destinations and plan trips using web search and LLM
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Optional

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import web search from enhanced memory
try:
    from enhanced_memory import EnhancedAlexandraMemory
    HAS_WEB_SEARCH = True
except ImportError:
    HAS_WEB_SEARCH = False
    print("[Travel Research] Web search not available")

from travel_config import KNOWLEDGE_DIR

class TravelResearcher:
    """Research destinations and plan trips"""

    def __init__(self):
        self.memory = EnhancedAlexandraMemory() if HAS_WEB_SEARCH else None
        self.research_cache = {}
        self.saved_trips_file = os.path.join(KNOWLEDGE_DIR, "planned_trips.json")
        self.saved_trips = self._load_trips()

    def _load_trips(self) -> Dict:
        """Load saved trip plans"""
        if os.path.exists(self.saved_trips_file):
            with open(self.saved_trips_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_trips(self):
        """Save trip plans"""
        with open(self.saved_trips_file, 'w') as f:
            json.dump(self.saved_trips, f, indent=2)

    def search_destination(self, destination: str, query_type: str = "general") -> Dict:
        """
        Search for information about a destination

        query_types:
        - general: overview, best time to visit, highlights
        - food: restaurants, local cuisine
        - hotels: where to stay
        - activities: things to do
        - budget: costs, money tips
        - safety: travel advisories, tips
        - transportation: how to get there/around
        """
        if not HAS_WEB_SEARCH:
            return {"error": "Web search not available", "results": []}

        # Build search query
        query_templates = {
            "general": f"{destination} travel guide best time to visit",
            "food": f"{destination} best restaurants local food where to eat",
            "hotels": f"{destination} best hotels where to stay neighborhoods",
            "activities": f"{destination} things to do attractions activities",
            "budget": f"{destination} travel budget cost daily expenses tips",
            "safety": f"{destination} travel safety tips advisories 2024",
            "transportation": f"{destination} how to get there airport transportation",
            "itinerary": f"{destination} itinerary day by day travel plan",
            "hidden_gems": f"{destination} hidden gems off beaten path local secrets",
            "nightlife": f"{destination} nightlife bars clubs entertainment",
            "shopping": f"{destination} shopping markets souvenirs where to buy",
        }

        query = query_templates.get(query_type, f"{destination} {query_type} travel")

        # Search
        results = self.memory.search_web_duckduckgo(query, max_results=5)

        # Cache results
        cache_key = f"{destination}_{query_type}"
        self.research_cache[cache_key] = {
            "query": query,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

        return {
            "destination": destination,
            "query_type": query_type,
            "results": results
        }

    def research_destination(self, destination: str) -> Dict:
        """Do comprehensive research on a destination"""
        research = {
            "destination": destination,
            "researched": datetime.now().isoformat(),
            "sections": {}
        }

        # Research multiple aspects
        aspects = ["general", "food", "activities", "hotels", "budget", "transportation"]

        for aspect in aspects:
            result = self.search_destination(destination, aspect)
            research["sections"][aspect] = result.get("results", [])

        return research

    def compare_destinations(self, dest1: str, dest2: str, criteria: List[str] = None) -> Dict:
        """Compare two destinations"""
        if criteria is None:
            criteria = ["budget", "activities", "food", "safety"]

        comparison = {
            "destinations": [dest1, dest2],
            "compared": datetime.now().isoformat(),
            "criteria": {}
        }

        for criterion in criteria:
            comparison["criteria"][criterion] = {
                dest1: self.search_destination(dest1, criterion).get("results", [])[:2],
                dest2: self.search_destination(dest2, criterion).get("results", [])[:2]
            }

        return comparison

    # ============== Trip Planning ==============

    def create_trip(self, name: str, destinations: List[str], dates: str = "", notes: str = "") -> str:
        """Create a new trip plan"""
        trip_id = f"trip_{len(self.saved_trips) + 1}"

        self.saved_trips[trip_id] = {
            "name": name,
            "destinations": destinations,
            "dates": dates,
            "notes": notes,
            "created": datetime.now().isoformat(),
            "research": {},
            "itinerary": [],
            "packing_list": [],
            "bookings": [],
            "budget": {
                "estimated": 0,
                "actual": 0,
                "breakdown": {}
            }
        }

        self._save_trips()
        return trip_id

    def add_to_itinerary(self, trip_id: str, day: int, activities: List[str]) -> bool:
        """Add activities to a trip day"""
        if trip_id not in self.saved_trips:
            return False

        trip = self.saved_trips[trip_id]

        # Extend itinerary if needed
        while len(trip["itinerary"]) < day:
            trip["itinerary"].append({"day": len(trip["itinerary"]) + 1, "activities": []})

        trip["itinerary"][day - 1]["activities"].extend(activities)
        self._save_trips()
        return True

    def set_trip_budget(self, trip_id: str, category: str, amount: float, is_actual: bool = False) -> bool:
        """Set budget for a trip category"""
        if trip_id not in self.saved_trips:
            return False

        trip = self.saved_trips[trip_id]
        trip["budget"]["breakdown"][category] = amount

        # Update totals
        total = sum(trip["budget"]["breakdown"].values())
        if is_actual:
            trip["budget"]["actual"] = total
        else:
            trip["budget"]["estimated"] = total

        self._save_trips()
        return True

    def add_booking(self, trip_id: str, booking_type: str, name: str,
                   confirmation: str = "", dates: str = "", cost: float = 0) -> bool:
        """Add a booking to a trip"""
        if trip_id not in self.saved_trips:
            return False

        self.saved_trips[trip_id]["bookings"].append({
            "type": booking_type,
            "name": name,
            "confirmation": confirmation,
            "dates": dates,
            "cost": cost,
            "added": datetime.now().isoformat()
        })

        self._save_trips()
        return True

    def add_packing_item(self, trip_id: str, item: str, category: str = "general", packed: bool = False) -> bool:
        """Add item to packing list"""
        if trip_id not in self.saved_trips:
            return False

        self.saved_trips[trip_id]["packing_list"].append({
            "item": item,
            "category": category,
            "packed": packed
        })

        self._save_trips()
        return True

    def get_trip(self, trip_id: str) -> Optional[Dict]:
        """Get a trip plan"""
        return self.saved_trips.get(trip_id)

    def list_trips(self) -> List[Dict]:
        """List all trip plans"""
        return [
            {"id": tid, "name": t["name"], "destinations": t["destinations"], "dates": t["dates"]}
            for tid, t in self.saved_trips.items()
        ]

    def research_for_trip(self, trip_id: str) -> bool:
        """Do research for all destinations in a trip"""
        if trip_id not in self.saved_trips:
            return False

        trip = self.saved_trips[trip_id]

        for dest in trip["destinations"]:
            research = self.research_destination(dest)
            trip["research"][dest] = research

        self._save_trips()
        return True

    # ============== Smart Suggestions ==============

    def suggest_destinations(self, preferences: Dict) -> List[Dict]:
        """
        Suggest destinations based on preferences

        preferences can include:
        - budget: "budget", "mid-range", "luxury"
        - type: "beach", "city", "nature", "culture"
        - duration: "weekend", "week", "two_weeks"
        - from_location: starting point for flight suggestions
        """
        budget = preferences.get("budget", "mid-range")
        trip_type = preferences.get("type", "")
        duration = preferences.get("duration", "week")

        query = f"best {trip_type} destinations {budget} travel {duration} trip 2024"

        if HAS_WEB_SEARCH:
            results = self.memory.search_web_duckduckgo(query, max_results=8)
            return results

        return []

    def get_packing_suggestions(self, destination: str, duration: str, activities: List[str] = None) -> List[str]:
        """Get packing suggestions for a trip"""
        suggestions = [
            # Essentials
            "Passport/ID",
            "Phone & charger",
            "Wallet & cards",
            "Travel insurance docs",
            "Medications",

            # Clothing basics
            f"Underwear ({duration})",
            f"Socks ({duration})",
            "Comfortable walking shoes",
            "Casual outfit x 3-4",
        ]

        # Add based on activities
        if activities:
            if "beach" in activities or "swimming" in activities:
                suggestions.extend(["Swimsuit", "Sunscreen", "Beach towel", "Flip flops"])
            if "hiking" in activities:
                suggestions.extend(["Hiking boots", "Daypack", "Water bottle", "Rain jacket"])
            if "nightlife" in activities or "dining" in activities:
                suggestions.extend(["Nice outfit", "Dress shoes"])

        return suggestions


# Singleton
_researcher = None

def get_researcher():
    global _researcher
    if _researcher is None:
        _researcher = TravelResearcher()
    return _researcher


if __name__ == "__main__":
    # Test
    researcher = TravelResearcher()

    print("Testing destination search...")
    results = researcher.search_destination("Barcelona", "food")
    print(f"Found {len(results.get('results', []))} results for Barcelona food")

    print("\nCreating test trip...")
    trip_id = researcher.create_trip(
        name="Spain Adventure",
        destinations=["Barcelona", "Madrid"],
        dates="March 2025",
        notes="First time in Spain!"
    )
    print(f"Created trip: {trip_id}")

    print("\nTrips:", researcher.list_trips())
