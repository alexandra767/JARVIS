"""
Alexandra AI - Unified Travel Training Data Generator
Combines all sources into a comprehensive training dataset
"""

import os
import sys
import json
import random
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from travel_config import KNOWLEDGE_DIR
from experience_logger import TravelExperienceLogger, get_experience_logger
from dataset_builder import TravelDatasetBuilder, get_dataset_builder
from training_templates import get_all_training_examples, PERSONALITY_PHRASES, DESTINATION_RESPONSES

class UnifiedTravelTrainer:
    """Generate comprehensive travel training datasets from all sources"""

    def __init__(self):
        self.experience_logger = get_experience_logger()
        self.dataset_builder = get_dataset_builder()
        self.output_dir = os.path.join(KNOWLEDGE_DIR, "training_output")
        os.makedirs(self.output_dir, exist_ok=True)

        # System prompt for travel mode
        self.system_prompt = """You are Alexandra, an experienced travel enthusiast and advisor.

Your travel expertise includes:
- Personal experiences from destinations around the world
- Insider tips for getting the most out of trips
- Budget-friendly and luxury travel options
- Food and cultural recommendations
- Safety and practical travel advice

Your style:
- Share personal anecdotes and stories when relevant
- Be enthusiastic but honest about destinations
- Give specific, actionable recommendations
- Consider the traveler's preferences and budget
- Include hidden gems and local favorites, not just tourist spots

When giving travel advice:
1. Ask about interests, budget, and travel style if not specified
2. Provide specific recommendations (actual names of places)
3. Share personal tips and experiences
4. Mention both highlights and potential drawbacks
5. Suggest alternatives based on different preferences"""

    # ============== PERSONAL EXPERIENCE CONVERSION ==============

    def convert_experiences_to_training(self) -> List[Dict]:
        """Convert logged personal experiences to training data"""
        training_data = []
        experiences = self.experience_logger.export_for_training()

        # Convert trips
        for trip in experiences.get("trips", []):
            training_data.extend(self._trip_to_qa(trip))

        # Convert places
        for destination, places in experiences.get("places", {}).items():
            for place in places:
                training_data.extend(self._place_to_qa(place, destination))

        # Convert stories
        for story in experiences.get("stories", []):
            training_data.extend(self._story_to_qa(story))

        # Convert tips
        for tip in experiences.get("tips", []):
            training_data.extend(self._tip_to_qa(tip))

        # Convert ratings
        for destination, ratings in experiences.get("ratings", {}).items():
            training_data.extend(self._rating_to_qa(destination, ratings))

        return training_data

    def _trip_to_qa(self, trip: Dict) -> List[Dict]:
        """Convert a trip to Q&A pairs"""
        pairs = []
        dest = trip.get("destination", "Unknown")
        country = trip.get("country", "")

        # Overview question
        if trip.get("highlights") or trip.get("notes"):
            highlights = ", ".join(trip.get("highlights", [])[:3])
            response = f"I spent {trip.get('duration_days', 'several')} days in {dest}"
            if country:
                response += f", {country}"
            response += f" and it was {'amazing' if trip.get('overall_rating', 5) >= 7 else 'an interesting experience'}! "

            if highlights:
                response += f"The highlights were definitely {highlights}. "

            if trip.get("notes"):
                response += trip["notes"] + " "

            if trip.get("lowlights"):
                response += f"The only downsides were {', '.join(trip['lowlights'][:2])}. "

            if trip.get("would_return"):
                response += "I'd definitely go back!"
            else:
                response += "It was worth seeing once, but I probably wouldn't return."

            pairs.append({
                "instruction": f"Have you been to {dest}?",
                "input": "",
                "output": response.strip(),
                "source": "personal_experience"
            })

        # Budget question
        if trip.get("budget", {}).get("total"):
            budget = trip["budget"]
            response = f"When I visited {dest} for {trip.get('duration_days', 'several')} days, I spent about ${budget['total']:.0f} total"
            if budget.get("per_day"):
                response += f", which works out to about ${budget['per_day']:.0f} per day"
            response += ". "

            if budget.get("breakdown"):
                items = [f"${v:.0f} on {k}" for k, v in list(budget["breakdown"].items())[:3]]
                response += f"That broke down to roughly {', '.join(items)}. "

            response += f"I traveled {trip.get('companions', 'solo')} and did a {trip.get('trip_type', 'leisure')} style trip."

            pairs.append({
                "instruction": f"How much did you spend in {dest}?",
                "input": "",
                "output": response.strip(),
                "source": "personal_experience"
            })

        return pairs

    def _place_to_qa(self, place: Dict, destination: str) -> List[Dict]:
        """Convert a place to Q&A pairs"""
        pairs = []
        name = place.get("name", "Unknown")
        place_type = place.get("type", "place")

        # Main recommendation
        response = f"{'One of my favorite spots' if place.get('rating', 5) >= 8 else 'A decent option'} in {destination} is {name}! "

        if place.get("what_i_loved"):
            response += place["what_i_loved"] + " "

        if place.get("must_try"):
            response += f"You have to try: {', '.join(place['must_try'][:3])}. "

        if place.get("tips"):
            response += f"Pro tip: {place['tips'][0]} "

        if place.get("price_level"):
            price_desc = {"$": "very affordable", "$$": "reasonably priced", "$$$": "a bit pricey", "$$$$": "splurge-worthy"}
            response += f"It's {price_desc.get(place['price_level'], 'moderately priced')}. "

        if place.get("best_for"):
            response += f"Perfect for {', '.join(place['best_for'][:2])}."

        # Generate appropriate question based on place type
        if place_type == "restaurant":
            question = f"Where should I eat in {destination}?"
        elif place_type == "hotel":
            question = f"Where should I stay in {destination}?"
        elif place_type == "bar":
            question = f"Where can I get drinks in {destination}?"
        elif place_type in ["attraction", "museum"]:
            question = f"What should I see in {destination}?"
        else:
            question = f"What do you recommend in {destination}?"

        pairs.append({
            "instruction": question,
            "input": "",
            "output": response.strip(),
            "source": "personal_experience",
            "place_name": name
        })

        return pairs

    def _story_to_qa(self, story: Dict) -> List[Dict]:
        """Convert a story to Q&A pairs"""
        pairs = []
        dest = story.get("destination", "")

        # Story as experience
        response = story.get("story", "")
        if story.get("lesson_learned"):
            response += f"\n\nLesson learned: {story['lesson_learned']}"

        story_type = story.get("type", "experience")
        if story_type == "mishap":
            question = f"Did anything go wrong on your trip to {dest}?" if dest else "What's your worst travel experience?"
        elif story_type == "funny":
            question = f"Any funny stories from {dest}?" if dest else "What's a funny travel story you have?"
        elif story_type == "adventure":
            question = f"What adventure did you have in {dest}?" if dest else "Tell me about a travel adventure"
        else:
            question = f"Tell me about your experience in {dest}" if dest else "Share a travel story"

        if response:
            pairs.append({
                "instruction": question,
                "input": "",
                "output": response.strip(),
                "source": "personal_story"
            })

        return pairs

    def _tip_to_qa(self, tip: Dict) -> List[Dict]:
        """Convert a tip to Q&A pairs"""
        pairs = []
        tip_text = tip.get("tip", "")
        category = tip.get("category", "general")
        dest = tip.get("destination", "")

        if not tip_text:
            return pairs

        # Generate question based on category
        if dest:
            question = f"Any {category} tips for {dest}?"
        else:
            category_questions = {
                "packing": "What are your packing tips?",
                "money": "How do you save money while traveling?",
                "safety": "How do you stay safe while traveling?",
                "transport": "Any transportation tips?",
                "food": "How do you find good food while traveling?",
                "accommodation": "Tips for finding good hotels?",
                "booking": "Tips for booking travel?",
                "culture": "How do you respect local culture?",
                "photography": "Travel photography tips?",
                "general": "What's your best travel tip?"
            }
            question = category_questions.get(category, "What's your best travel tip?")

        importance = tip.get("importance", "helpful")
        if importance == "essential":
            response = f"This is crucial: {tip_text}"
        else:
            response = f"Here's a tip I've learned: {tip_text}"

        pairs.append({
            "instruction": question,
            "input": "",
            "output": response,
            "source": "personal_tip"
        })

        return pairs

    def _rating_to_qa(self, destination: str, ratings: Dict) -> List[Dict]:
        """Convert ratings to Q&A pairs"""
        pairs = []

        overall = ratings.get("overall", 5)

        response = f"I'd rate {destination} a {overall}/10 overall. "

        # Add specific ratings
        aspects = []
        if ratings.get("food"):
            aspects.append(f"food {ratings['food']}/10")
        if ratings.get("culture"):
            aspects.append(f"culture {ratings['culture']}/10")
        if ratings.get("safety"):
            aspects.append(f"safety {ratings['safety']}/10")
        if ratings.get("value"):
            aspects.append(f"value for money {ratings['value']}/10")

        if aspects:
            response += f"Breaking it down: {', '.join(aspects)}. "

        # Add commentary based on score
        if overall >= 9:
            response += "It's absolutely one of my favorites and I can't recommend it enough!"
        elif overall >= 7:
            response += "It's definitely worth visiting if it matches your travel style."
        elif overall >= 5:
            response += "It's decent, though maybe not my top pick unless you have specific interests there."
        else:
            response += "To be honest, there are probably better destinations for most travelers."

        pairs.append({
            "instruction": f"How would you rate {destination}?",
            "input": "",
            "output": response.strip(),
            "source": "personal_rating"
        })

        return pairs

    # ============== DATASET COMPILATION ==============

    def compile_full_dataset(self,
                            include_personal: bool = True,
                            include_templates: bool = True,
                            include_synthetic: bool = True,
                            include_web_research: bool = False,
                            destinations_to_research: List[str] = None,
                            min_examples: int = 500) -> Dict:
        """Compile a complete training dataset from all sources"""

        all_examples = []
        stats = {
            "personal_experiences": 0,
            "high_quality_templates": 0,
            "synthetic_generated": 0,
            "web_research": 0,
            "total": 0
        }

        # 1. Personal experiences (highest quality - your actual data)
        if include_personal:
            personal = self.convert_experiences_to_training()
            all_examples.extend(personal)
            stats["personal_experiences"] = len(personal)
            print(f"Added {len(personal)} examples from personal experiences")

        # 2. High-quality templates (curated responses)
        if include_templates:
            templates = get_all_training_examples()
            all_examples.extend(templates)
            stats["high_quality_templates"] = len(templates)
            print(f"Added {len(templates)} examples from curated templates")

        # 3. Synthetic data (generated variations)
        if include_synthetic:
            needed = max(0, min_examples - len(all_examples))
            synthetic = self.dataset_builder.generate_synthetic_qa_pairs(max(100, needed))
            all_examples.extend(synthetic)
            stats["synthetic_generated"] = len(synthetic)
            print(f"Added {len(synthetic)} synthetic examples")

        # 4. Web research (optional - takes time)
        if include_web_research:
            destinations = destinations_to_research or self.dataset_builder.popular_destinations[:5]
            for dest in destinations:
                print(f"Researching {dest}...")
                research = self.dataset_builder.research_destination(dest)
                qa_pairs = self.dataset_builder.generate_qa_pairs_from_research(research)
                all_examples.extend(qa_pairs)
                stats["web_research"] += len(qa_pairs)

        stats["total"] = len(all_examples)

        # Shuffle for good distribution
        random.shuffle(all_examples)

        return {
            "examples": all_examples,
            "stats": stats,
            "system_prompt": self.system_prompt
        }

    def export_for_finetuning(self,
                              dataset: Dict,
                              format: str = "alpaca",
                              filename: str = None) -> str:
        """Export dataset in format ready for fine-tuning"""

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alexandra_travel_{format}_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)
        examples = dataset.get("examples", [])

        if format == "alpaca":
            # Alpaca format - good for Llama fine-tuning
            data = []
            for ex in examples:
                data.append({
                    "instruction": ex.get("instruction", ""),
                    "input": ex.get("input", ""),
                    "output": ex.get("output", "")
                })

        elif format == "openai":
            # OpenAI chat format
            data = []
            for ex in examples:
                messages = [{"role": "user", "content": ex.get("instruction", "")}]
                if ex.get("input"):
                    messages[0]["content"] += f"\n\n{ex['input']}"
                messages.append({"role": "assistant", "content": ex.get("output", "")})
                data.append({"messages": messages})

        elif format == "chatml":
            # ChatML format with system prompt
            data = []
            for ex in examples:
                data.append({
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": ex.get("instruction", "")},
                        {"role": "assistant", "content": ex.get("output", "")}
                    ]
                })

        elif format == "sharegpt":
            # ShareGPT format
            data = []
            for ex in examples:
                data.append({
                    "conversations": [
                        {"from": "system", "value": self.system_prompt},
                        {"from": "human", "value": ex.get("instruction", "")},
                        {"from": "gpt", "value": ex.get("output", "")}
                    ]
                })

        elif format == "text":
            # Plain text format for continued pretraining
            data = []
            for ex in examples:
                text = f"### User:\n{ex.get('instruction', '')}\n\n### Assistant:\n{ex.get('output', '')}"
                data.append(text)

            with open(filepath, 'w') as f:
                f.write("\n\n---\n\n".join(data))
            return filepath

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath

    def generate_complete_training_package(self,
                                          include_web_research: bool = False) -> Dict:
        """Generate a complete training package with multiple formats"""

        print("Compiling training dataset...")
        dataset = self.compile_full_dataset(
            include_personal=True,
            include_templates=True,
            include_synthetic=True,
            include_web_research=include_web_research,
            min_examples=500
        )

        print(f"\nTotal examples: {dataset['stats']['total']}")
        print(f"  - Personal experiences: {dataset['stats']['personal_experiences']}")
        print(f"  - Curated templates: {dataset['stats']['high_quality_templates']}")
        print(f"  - Synthetic: {dataset['stats']['synthetic_generated']}")
        if include_web_research:
            print(f"  - Web research: {dataset['stats']['web_research']}")

        # Export in multiple formats
        files = {}
        for fmt in ["alpaca", "chatml", "sharegpt", "openai"]:
            filepath = self.export_for_finetuning(dataset, fmt)
            files[fmt] = filepath
            print(f"Exported {fmt}: {filepath}")

        # Save system prompt separately
        prompt_file = os.path.join(self.output_dir, "system_prompt.txt")
        with open(prompt_file, 'w') as f:
            f.write(self.system_prompt)
        files["system_prompt"] = prompt_file

        # Save stats
        stats_file = os.path.join(self.output_dir, "training_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(dataset["stats"], f, indent=2)
        files["stats"] = stats_file

        return {
            "files": files,
            "stats": dataset["stats"],
            "total_examples": dataset["stats"]["total"]
        }


# Singleton
_trainer = None

def get_unified_trainer():
    global _trainer
    if _trainer is None:
        _trainer = UnifiedTravelTrainer()
    return _trainer


if __name__ == "__main__":
    print("=" * 50)
    print("Alexandra AI - Travel Training Data Generator")
    print("=" * 50)

    trainer = UnifiedTravelTrainer()

    # Generate complete training package
    result = trainer.generate_complete_training_package(include_web_research=False)

    print("\n" + "=" * 50)
    print("Training Package Generated!")
    print("=" * 50)
    print(f"\nTotal training examples: {result['total_examples']}")
    print(f"\nFiles created:")
    for name, path in result["files"].items():
        print(f"  {name}: {path}")

    print("\nTo use with your fine-tuning:")
    print("1. Add your personal experiences using experience_logger.py")
    print("2. Re-run this script to regenerate with your data")
    print("3. Use the alpaca or chatml format for Llama-based models")
    print("4. Use the openai format for OpenAI-compatible fine-tuning")
