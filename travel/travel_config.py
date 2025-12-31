"""
Alexandra AI - Travel Module Configuration
Settings for travel knowledge, content creation, and training
"""

import os

# ============== PATHS ==============
TRAVEL_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(TRAVEL_DIR, "knowledge")
TRAINING_DATA_DIR = os.path.join(TRAVEL_DIR, "training_data")
CONTENT_DIR = os.path.join(TRAVEL_DIR, "content")
SCRIPTS_DIR = os.path.join(TRAVEL_DIR, "scripts")

# Create directories
for d in [KNOWLEDGE_DIR, TRAINING_DATA_DIR, CONTENT_DIR, SCRIPTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============== TRAVEL CATEGORIES ==============
TRAVEL_CATEGORIES = {
    "destinations": {
        "beaches": [],
        "cities": [],
        "mountains": [],
        "countryside": [],
        "islands": [],
        "historical": [],
    },
    "activities": {
        "adventure": [],
        "relaxation": [],
        "cultural": [],
        "food_wine": [],
        "nightlife": [],
        "family": [],
    },
    "accommodation": {
        "luxury": [],
        "boutique": [],
        "budget": [],
        "airbnb": [],
        "resorts": [],
        "hostels": [],
    },
    "transportation": {
        "flights": [],
        "trains": [],
        "road_trips": [],
        "cruises": [],
        "local_transport": [],
    },
    "tips": {
        "packing": [],
        "budgeting": [],
        "safety": [],
        "photography": [],
        "local_customs": [],
    }
}

# ============== TRAVEL PERSONALITY ==============
TRAVEL_PERSONALITY = {
    "name": "Travel Expert Alexandra",
    "system_prompt": """You are Alexandra, an experienced travel enthusiast and advisor.

Your travel expertise includes:
- Personal experiences from destinations around the world
- Insider tips for getting the most out of trips
- Budget-friendly and luxury travel options
- Food and cultural recommendations
- Safety and practical travel advice

Your style:
- Share personal anecdotes and stories from your travels
- Be enthusiastic but honest about destinations
- Give specific, actionable recommendations
- Consider the traveler's preferences and budget
- Include hidden gems and local favorites, not just tourist spots

When giving travel advice:
1. Ask about their interests, budget, and travel style if not specified
2. Provide specific recommendations (names of places, not just "a nice restaurant")
3. Share personal tips and experiences
4. Mention both highlights and potential drawbacks
5. Suggest alternatives based on different preferences
"""
}

# ============== CONTENT TEMPLATES ==============
CONTENT_TEMPLATES = {
    "destination_review": {
        "title": "{destination} Travel Guide - My Experience",
        "sections": [
            "Introduction and first impressions",
            "Best time to visit",
            "Top attractions and must-sees",
            "Hidden gems and local favorites",
            "Food and dining recommendations",
            "Where to stay",
            "Getting around",
            "Budget breakdown",
            "Tips and advice",
            "Final thoughts and rating"
        ]
    },
    "travel_tips": {
        "title": "{count} {topic} Tips You Need to Know",
        "format": "numbered_list"
    },
    "comparison": {
        "title": "{destination1} vs {destination2}: Which Should You Visit?",
        "sections": [
            "Overview of both destinations",
            "Best for: beaches/culture/food/nightlife",
            "Budget comparison",
            "Best time to visit each",
            "Who should choose which",
            "My personal preference and why"
        ]
    },
    "itinerary": {
        "title": "{duration} in {destination}: The Perfect Itinerary",
        "format": "day_by_day"
    },
    "packing_list": {
        "title": "What to Pack for {destination}/{trip_type}",
        "categories": [
            "Essentials",
            "Clothing",
            "Electronics",
            "Toiletries",
            "Documents",
            "Nice to have"
        ]
    }
}

# ============== TRAINING DATA FORMAT ==============
TRAINING_DATA_FORMAT = {
    "conversation": {
        "format": "jsonl",
        "structure": {
            "messages": [
                {"role": "system", "content": "system_prompt"},
                {"role": "user", "content": "user_question"},
                {"role": "assistant", "content": "travel_response"}
            ]
        }
    },
    "qa_pairs": {
        "format": "jsonl",
        "structure": {
            "question": "travel question",
            "answer": "detailed travel answer"
        }
    }
}

# ============== DATA SOURCES ==============
DATA_SOURCES = {
    "personal": {
        "description": "Your own travel experiences and preferences",
        "files": [
            "my_destinations.json",
            "my_favorites.json",
            "my_tips.json"
        ]
    },
    "scraped": {
        "description": "Collected from travel sites (for personal use)",
        "sources": [
            "tripadvisor",
            "lonely_planet",
            "travel_blogs"
        ]
    },
    "generated": {
        "description": "AI-generated training examples",
        "types": [
            "qa_pairs",
            "destination_descriptions",
            "travel_conversations"
        ]
    }
}
