"""
Alexandra AI - Travel Dataset Builder
Aggregate travel data from multiple sources for comprehensive training
"""

import os
import sys
import json
import re
from datetime import datetime
from typing import List, Dict, Optional
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from travel_config import KNOWLEDGE_DIR

# Try to import web search
try:
    from enhanced_memory import EnhancedAlexandraMemory
    HAS_WEB_SEARCH = True
except ImportError:
    HAS_WEB_SEARCH = False

class TravelDatasetBuilder:
    """Build comprehensive travel training datasets"""

    def __init__(self):
        self.dataset_dir = os.path.join(KNOWLEDGE_DIR, "datasets")
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.memory = EnhancedAlexandraMemory() if HAS_WEB_SEARCH else None

        # Popular destinations for base knowledge
        self.popular_destinations = [
            # Europe
            "Paris", "London", "Rome", "Barcelona", "Amsterdam", "Prague",
            "Vienna", "Lisbon", "Berlin", "Athens", "Dublin", "Budapest",
            "Copenhagen", "Stockholm", "Edinburgh", "Florence", "Venice",
            # Asia
            "Tokyo", "Bangkok", "Singapore", "Bali", "Hong Kong", "Seoul",
            "Kyoto", "Dubai", "Mumbai", "Hanoi", "Kuala Lumpur", "Manila",
            # Americas
            "New York", "Los Angeles", "Miami", "San Francisco", "Las Vegas",
            "Mexico City", "Cancun", "Rio de Janeiro", "Buenos Aires", "Lima",
            # Other
            "Sydney", "Melbourne", "Auckland", "Cape Town", "Marrakech", "Cairo"
        ]

        # Travel topics for Q&A generation
        self.travel_topics = [
            "best time to visit", "budget tips", "must-see attractions",
            "local food", "where to stay", "getting around", "safety tips",
            "hidden gems", "day trips", "nightlife", "shopping",
            "cultural etiquette", "visa requirements", "packing tips",
            "best neighborhoods", "photography spots", "local experiences"
        ]

    # ============== WEB RESEARCH ==============

    def research_destination(self, destination: str) -> Dict:
        """Research a destination from the web"""
        if not HAS_WEB_SEARCH:
            return {"error": "Web search not available"}

        research = {
            "destination": destination,
            "researched": datetime.now().isoformat(),
            "data": {}
        }

        search_queries = [
            f"{destination} travel guide 2024",
            f"{destination} best things to do",
            f"{destination} where to eat local food",
            f"{destination} budget travel tips",
            f"{destination} hidden gems off beaten path",
            f"{destination} best time to visit weather",
            f"{destination} safety tips tourists",
            f"{destination} neighborhoods where to stay"
        ]

        for query in search_queries:
            topic = query.replace(f"{destination} ", "")
            results = self.memory.search_web_duckduckgo(query, max_results=3)
            research["data"][topic] = results

        return research

    # ============== Q&A PAIR GENERATION ==============

    def generate_qa_pairs_from_research(self, research: Dict) -> List[Dict]:
        """Generate Q&A training pairs from research data"""
        qa_pairs = []
        destination = research.get("destination", "Unknown")

        for topic, results in research.get("data", {}).items():
            if not results:
                continue

            # Combine snippets into knowledge
            combined_info = " ".join([r.get("snippet", "") for r in results if r.get("snippet")])

            if not combined_info:
                continue

            # Generate questions for this topic
            questions = self._generate_questions(destination, topic)

            for question in questions:
                qa_pairs.append({
                    "instruction": question,
                    "input": "",
                    "output": f"Based on my knowledge of {destination}, {combined_info[:500]}",
                    "source": "web_research",
                    "destination": destination,
                    "topic": topic
                })

        return qa_pairs

    def _generate_questions(self, destination: str, topic: str) -> List[str]:
        """Generate natural questions for a topic"""
        question_templates = {
            "travel guide 2024": [
                f"Tell me about {destination}",
                f"What should I know before visiting {destination}?",
                f"Give me an overview of {destination} as a travel destination"
            ],
            "best things to do": [
                f"What are the best things to do in {destination}?",
                f"What should I see in {destination}?",
                f"What are the must-do activities in {destination}?",
                f"I'm going to {destination}, what should I do there?"
            ],
            "where to eat local food": [
                f"Where should I eat in {destination}?",
                f"What's the local food like in {destination}?",
                f"Can you recommend restaurants in {destination}?",
                f"What food should I try in {destination}?"
            ],
            "budget travel tips": [
                f"How can I visit {destination} on a budget?",
                f"Is {destination} expensive?",
                f"What are some budget tips for {destination}?",
                f"How much does a trip to {destination} cost?"
            ],
            "hidden gems off beaten path": [
                f"What are the hidden gems in {destination}?",
                f"What do tourists miss in {destination}?",
                f"Tell me about off-the-beaten-path spots in {destination}",
                f"What's something unique to do in {destination}?"
            ],
            "best time to visit weather": [
                f"When is the best time to visit {destination}?",
                f"What's the weather like in {destination}?",
                f"When should I go to {destination}?",
                f"What's the peak season for {destination}?"
            ],
            "safety tips tourists": [
                f"Is {destination} safe for tourists?",
                f"What safety tips do you have for {destination}?",
                f"What should I watch out for in {destination}?",
                f"Are there any scams to avoid in {destination}?"
            ],
            "neighborhoods where to stay": [
                f"Where should I stay in {destination}?",
                f"What are the best neighborhoods in {destination}?",
                f"Which area is best for tourists in {destination}?",
                f"Can you recommend hotels in {destination}?"
            ]
        }

        return question_templates.get(topic, [f"Tell me about {topic} in {destination}"])

    # ============== SYNTHETIC DATA GENERATION ==============

    def generate_synthetic_qa_pairs(self, count: int = 500) -> List[Dict]:
        """Generate synthetic Q&A pairs using templates"""
        qa_pairs = []

        # General travel questions
        general_questions = self._generate_general_travel_qa()
        qa_pairs.extend(general_questions)

        # Destination-specific placeholders
        destination_qa = self._generate_destination_qa_templates()
        qa_pairs.extend(destination_qa)

        # Tips and advice
        tips_qa = self._generate_tips_qa()
        qa_pairs.extend(tips_qa)

        # Comparison questions
        comparison_qa = self._generate_comparison_qa()
        qa_pairs.extend(comparison_qa)

        # Conversational exchanges
        conversational_qa = self._generate_conversational_qa()
        qa_pairs.extend(conversational_qa)

        # Shuffle and limit
        random.shuffle(qa_pairs)
        return qa_pairs[:count]

    def _generate_general_travel_qa(self) -> List[Dict]:
        """Generate general travel Q&A"""
        pairs = [
            {
                "instruction": "What's your favorite travel destination?",
                "input": "",
                "output": "That's a tough question! I love so many places for different reasons. If I had to pick one, I'd say [DESTINATION] really captured my heart. The combination of [REASON1], [REASON2], and [REASON3] made it unforgettable. What kind of trip are you looking for? I might have a more specific recommendation!",
                "category": "general"
            },
            {
                "instruction": "How do you plan your trips?",
                "input": "",
                "output": "I usually start by deciding on my must-see attractions, then build my itinerary around those. I research neighborhoods to find the best area to stay, book accommodations early for better prices, and always leave room for spontaneous discoveries. I also look up local food spots and check if there are any events during my visit. Want me to help you plan a trip?",
                "category": "general"
            },
            {
                "instruction": "What's the most important travel tip you've learned?",
                "input": "",
                "output": "The best tip I've learned is to always have a backup plan but be flexible enough to abandon it! Some of my best travel experiences happened when plans fell through and I had to improvise. Also, learn a few basic phrases in the local language - people really appreciate the effort and it opens so many doors.",
                "category": "general"
            },
            {
                "instruction": "How do you pack for a trip?",
                "input": "",
                "output": "I'm a big believer in packing light. I lay out everything I think I need, then put half of it back. I always pack versatile clothing that can be mixed and matched, and I roll my clothes instead of folding to save space. My essentials: a good daypack, portable charger, universal adapter, and a small first aid kit. What kind of trip are you packing for?",
                "category": "general"
            },
            {
                "instruction": "What do you do on long flights?",
                "input": "",
                "output": "Long flights are actually great for catching up on movies, reading, or planning the trip ahead. I always bring noise-canceling headphones, download entertainment beforehand, and bring my own snacks. Staying hydrated is key - I drink tons of water and skip the alcohol. I also try to walk around every few hours to avoid stiffness.",
                "category": "general"
            },
            {
                "instruction": "How do you deal with jet lag?",
                "input": "",
                "output": "I try to adjust to the new time zone before I even land - changing my sleep schedule a few days before helps. Once I arrive, I force myself to stay awake until local bedtime, get some sunlight during the day, and avoid napping. It usually takes me about a day per time zone crossed to fully adjust. Caffeine strategically helps too!",
                "category": "general"
            },
            {
                "instruction": "Do you prefer solo travel or group travel?",
                "input": "",
                "output": "Both have their magic! Solo travel gives you complete freedom and pushes you to meet new people. Group travel is great for safety, shared experiences, and splitting costs. I love solo trips for self-discovery and group trips with friends for adventure. It really depends on the destination and what kind of experience I'm looking for.",
                "category": "general"
            },
            {
                "instruction": "How do you stay safe while traveling?",
                "input": "",
                "output": "A few key things: I always research the area beforehand, keep copies of important documents separate from originals, stay aware of my surroundings, and trust my instincts. I don't flash expensive items, I use hotel safes for valuables, and I share my itinerary with someone back home. Most places are safer than they seem, but common sense goes a long way!",
                "category": "general"
            },
            {
                "instruction": "What's the best way to experience local culture?",
                "input": "",
                "output": "Get off the tourist path! Stay in local neighborhoods instead of hotel districts, eat where locals eat, take public transportation, and strike up conversations with people. I love visiting local markets, attending community events, and learning about traditions from residents. Walking tours led by locals are also fantastic for authentic insights.",
                "category": "general"
            },
            {
                "instruction": "How do you save money while traveling?",
                "input": "",
                "output": "My biggest money-savers: travel during shoulder season, be flexible with dates, stay in apartments with kitchens to cook some meals, use public transit instead of taxis, and look for free walking tours and museum days. I also use credit cards with no foreign transaction fees and always pay in local currency. Happy hour and lunch specials are great for dining out!",
                "category": "general"
            }
        ]
        return pairs

    def _generate_destination_qa_templates(self) -> List[Dict]:
        """Generate destination-specific Q&A templates"""
        pairs = []

        templates = [
            {
                "q": "What's {destination} like?",
                "a": "{destination} is an amazing destination! It's known for [MAIN_ATTRACTIONS]. The vibe is [VIBE_DESCRIPTION]. I'd recommend spending [DURATION] there to really experience it. The best time to visit is [BEST_TIME]. Let me know if you want specific recommendations!"
            },
            {
                "q": "Should I visit {destination}?",
                "a": "Absolutely! {destination} is perfect if you're into [INTEREST1] or [INTEREST2]. It's [BUDGET_LEVEL] compared to other destinations. The highlights include [HIGHLIGHT1] and [HIGHLIGHT2]. Just be prepared for [CONSIDERATION]. Want me to tell you more about specific aspects?"
            },
            {
                "q": "How many days do I need in {destination}?",
                "a": "I'd recommend [DURATION] in {destination} for a good experience. That gives you time to see [MAIN_ATTRACTIONS] without rushing. If you're short on time, [MINIMUM] days works for the highlights. With more time, you could add [EXTENSION_IDEAS]. What are your main interests?"
            },
            {
                "q": "What's the food like in {destination}?",
                "a": "The food scene in {destination} is [FOOD_DESCRIPTION]! You have to try [MUST_TRY_DISHES]. For restaurants, I'd recommend [RESTAURANT_TYPES]. Don't miss [FOOD_EXPERIENCE]. The best areas for food are [FOOD_AREAS]. Are you into any particular cuisine?"
            },
            {
                "q": "Where should I stay in {destination}?",
                "a": "It depends on what you're looking for! [NEIGHBORHOOD1] is great for [REASON1]. [NEIGHBORHOOD2] is better if you want [REASON2]. For first-timers, I'd suggest [RECOMMENDATION]. Budget-wise, expect [PRICE_RANGE]. Want specific hotel recommendations?"
            }
        ]

        for dest in self.popular_destinations[:20]:  # Use top 20 destinations
            for template in templates:
                pairs.append({
                    "instruction": template["q"].format(destination=dest),
                    "input": "",
                    "output": template["a"].format(destination=dest),
                    "category": "destination",
                    "destination": dest
                })

        return pairs

    def _generate_tips_qa(self) -> List[Dict]:
        """Generate travel tips Q&A"""
        tips_pairs = [
            {
                "instruction": "What are your top packing tips?",
                "input": "",
                "output": "My top packing tips: 1) Roll your clothes instead of folding - saves space and reduces wrinkles. 2) Pack a small day bag that fits inside your main luggage. 3) Bring a reusable water bottle and tote bag. 4) Keep essentials in your carry-on in case luggage gets lost. 5) Use packing cubes to stay organized. What kind of trip are you packing for?",
                "category": "tips"
            },
            {
                "instruction": "How do I avoid tourist traps?",
                "input": "",
                "output": "Great question! To avoid tourist traps: walk a few blocks away from major attractions before eating, check Google Maps reviews and look for places with local reviewers, ask your accommodation staff where they eat, visit popular spots early morning or late evening, and be wary of anyone aggressively approaching tourists. Research common scams for your destination beforehand too!",
                "category": "tips"
            },
            {
                "instruction": "What should I always have in my travel bag?",
                "input": "",
                "output": "My travel bag essentials: universal power adapter, portable phone charger, reusable water bottle, snacks, hand sanitizer, basic first aid kit, photocopies of important documents, a pen (always need one for customs forms!), earbuds, a light jacket or scarf, and sunscreen. I also keep a small ziplock bag for liquids. What destination are you heading to?",
                "category": "tips"
            },
            {
                "instruction": "How do I find authentic local experiences?",
                "input": "",
                "output": "For authentic experiences: use apps like EatWith or WithLocals to connect with locals, join free walking tours led by residents, visit on weekdays rather than weekends, explore neighborhoods outside the tourist center, check local event calendars and Facebook groups, visit markets early in the morning, and don't be afraid to strike up conversations. Some of my best discoveries came from random recommendations!",
                "category": "tips"
            },
            {
                "instruction": "What are common travel mistakes to avoid?",
                "input": "",
                "output": "Common mistakes I've seen: overpacking, trying to see too much too fast, not having travel insurance, only eating at tourist restaurants, not learning basic local phrases, forgetting to notify your bank about travel, not making copies of important documents, exchanging money at airports (worst rates!), and not researching local customs. The biggest one? Not leaving room for spontaneity!",
                "category": "tips"
            }
        ]
        return tips_pairs

    def _generate_comparison_qa(self) -> List[Dict]:
        """Generate destination comparison Q&A"""
        comparisons = [
            ("Paris", "London"),
            ("Barcelona", "Madrid"),
            ("Tokyo", "Seoul"),
            ("Bali", "Thailand"),
            ("Rome", "Florence"),
            ("New York", "Los Angeles"),
            ("Amsterdam", "Copenhagen"),
            ("Dubai", "Singapore")
        ]

        pairs = []
        for dest1, dest2 in comparisons:
            pairs.append({
                "instruction": f"Should I go to {dest1} or {dest2}?",
                "input": "",
                "output": f"Both are fantastic! {dest1} is better for [REASON1], while {dest2} excels at [REASON2]. Budget-wise, [BUDGET_COMPARISON]. For first-timers, I'd suggest [RECOMMENDATION]. If you're into [INTEREST], go with [CHOICE]. What matters most to you on this trip? I can give you a more specific recommendation!",
                "category": "comparison"
            })
            pairs.append({
                "instruction": f"What's the difference between {dest1} and {dest2}?",
                "input": "",
                "output": f"They're quite different! {dest1} offers [DEST1_STRENGTHS], while {dest2} is known for [DEST2_STRENGTHS]. The vibe in {dest1} is [VIBE1], whereas {dest2} feels more [VIBE2]. Cost-wise, [COST_COMPARISON]. I personally [PREFERENCE_STATEMENT]. What kind of experience are you looking for?",
                "category": "comparison"
            })

        return pairs

    def _generate_conversational_qa(self) -> List[Dict]:
        """Generate conversational travel exchanges"""
        pairs = [
            {
                "instruction": "I've never traveled internationally before. Any advice?",
                "input": "",
                "output": "How exciting that you're planning your first international trip! My advice: start with a destination that's relatively easy to navigate and English-friendly if that's your language. Make sure your passport is valid for at least 6 months beyond your travel dates. Get travel insurance - trust me on this one. Download offline maps and translation apps. And most importantly, embrace the adventure! The first trip is always special. Where are you thinking of going?",
                "category": "conversational"
            },
            {
                "instruction": "I'm scared of flying but want to travel. Help!",
                "input": "",
                "output": "I totally understand - flying anxiety is super common! A few things that help: choose aisle seats for more control, practice deep breathing techniques, download calming apps or playlists, avoid caffeine before flying, and maybe consider talking to a doctor about short-term medication for flights. Remember, turbulence feels scary but it's normal and safe. Start with shorter flights to build confidence. The destinations are worth it, I promise!",
                "category": "conversational"
            },
            {
                "instruction": "How do you afford to travel so much?",
                "input": "",
                "output": "Great question! It's really about priorities and smart planning. I use credit card points and miles, travel during off-peak seasons, stay in apartments or hostels, cook some of my own meals, and prioritize experiences over luxury. I also set a dedicated travel savings fund. You don't need to be rich to travel - you just need to be strategic! Want some specific budget travel tips?",
                "category": "conversational"
            },
            {
                "instruction": "What was your worst travel experience?",
                "input": "",
                "output": "Oh, I have a few stories! The worst was probably [EXPERIENCE]. At the time it was stressful, but looking back, it makes for a great story and taught me valuable lessons. That's the thing about travel mishaps - they usually become your best memories! The key is staying calm, being flexible, and remembering that these challenges are part of the adventure. What's prompting this question - planning to be prepared?",
                "category": "conversational"
            },
            {
                "instruction": "I want to travel but my partner doesn't. What should I do?",
                "input": "",
                "output": "This is more common than you think! A few suggestions: try finding what specific concerns they have (fear of flying, leaving home, money, etc.) and address those. Start with shorter, easier trips. Consider what kind of travel might appeal to them - not everyone loves cities or adventure. Some people just need to experience one great trip to catch the bug. You could also take occasional solo trips while they stay home. Communication is key!",
                "category": "conversational"
            }
        ]
        return pairs

    # ============== EXPORT FUNCTIONS ==============

    def export_dataset(self, qa_pairs: List[Dict], filename: str, format: str = "alpaca") -> str:
        """Export dataset in specified format"""
        filepath = os.path.join(self.dataset_dir, filename)

        if format == "alpaca":
            # Alpaca/Llama format
            data = []
            for pair in qa_pairs:
                data.append({
                    "instruction": pair.get("instruction", ""),
                    "input": pair.get("input", ""),
                    "output": pair.get("output", "")
                })

        elif format == "openai":
            # OpenAI chat format
            data = []
            for pair in qa_pairs:
                data.append({
                    "messages": [
                        {"role": "user", "content": pair.get("instruction", "")},
                        {"role": "assistant", "content": pair.get("output", "")}
                    ]
                })

        elif format == "sharegpt":
            # ShareGPT format
            data = []
            for pair in qa_pairs:
                data.append({
                    "conversations": [
                        {"from": "human", "value": pair.get("instruction", "")},
                        {"from": "gpt", "value": pair.get("output", "")}
                    ]
                })

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath

    def build_comprehensive_dataset(self,
                                     include_synthetic: bool = True,
                                     include_web_research: bool = True,
                                     destinations_to_research: List[str] = None,
                                     synthetic_count: int = 500) -> Dict:
        """Build a comprehensive travel dataset"""
        all_pairs = []
        stats = {
            "synthetic": 0,
            "web_research": 0,
            "destinations_researched": []
        }

        # Add synthetic data
        if include_synthetic:
            synthetic = self.generate_synthetic_qa_pairs(synthetic_count)
            all_pairs.extend(synthetic)
            stats["synthetic"] = len(synthetic)

        # Add web research
        if include_web_research and HAS_WEB_SEARCH:
            destinations = destinations_to_research or self.popular_destinations[:10]
            for dest in destinations:
                print(f"Researching {dest}...")
                research = self.research_destination(dest)
                qa_pairs = self.generate_qa_pairs_from_research(research)
                all_pairs.extend(qa_pairs)
                stats["destinations_researched"].append(dest)
            stats["web_research"] = len(all_pairs) - stats["synthetic"]

        # Export in multiple formats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = {}

        for fmt in ["alpaca", "openai", "sharegpt"]:
            filename = f"travel_dataset_{timestamp}_{fmt}.json"
            filepath = self.export_dataset(all_pairs, filename, fmt)
            files[fmt] = filepath

        return {
            "total_pairs": len(all_pairs),
            "stats": stats,
            "files": files
        }


# Singleton
_builder = None

def get_dataset_builder():
    global _builder
    if _builder is None:
        _builder = TravelDatasetBuilder()
    return _builder


if __name__ == "__main__":
    print("Travel Dataset Builder Demo\n")

    builder = TravelDatasetBuilder()

    # Generate synthetic data only (no web search needed)
    print("Generating synthetic training data...")
    synthetic = builder.generate_synthetic_qa_pairs(100)

    print(f"Generated {len(synthetic)} Q&A pairs")
    print("\nSample pairs:")
    for pair in synthetic[:3]:
        print(f"\nQ: {pair['instruction']}")
        print(f"A: {pair['output'][:200]}...")

    # Export
    filepath = builder.export_dataset(synthetic, "travel_synthetic_demo.json", "alpaca")
    print(f"\nExported to: {filepath}")
