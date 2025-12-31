"""
Alexandra AI - Travel Training Data Generator
Create fine-tuning datasets for travel-focused model
"""

import os
import json
import random
from typing import List, Dict
from datetime import datetime

from travel_config import TRAINING_DATA_DIR, TRAVEL_PERSONALITY

class TravelTrainingDataGenerator:
    """Generate training data for travel fine-tuning"""

    def __init__(self):
        self.output_dir = TRAINING_DATA_DIR
        self.system_prompt = TRAVEL_PERSONALITY["system_prompt"]

        # Question templates
        self.question_templates = [
            "What's the best time to visit {destination}?",
            "Can you recommend things to do in {destination}?",
            "What should I know before visiting {destination}?",
            "Is {destination} good for {activity}?",
            "How many days do I need in {destination}?",
            "What's the food like in {destination}?",
            "Is {destination} expensive?",
            "What are the must-see attractions in {destination}?",
            "Where should I stay in {destination}?",
            "How do I get around in {destination}?",
            "Is {destination} safe for tourists?",
            "What should I pack for {destination}?",
            "Can you compare {destination1} and {destination2}?",
            "I have {budget} for a trip. Where should I go?",
            "I want a {trip_type} vacation. Any recommendations?",
            "What are some hidden gems in {destination}?",
            "Best restaurants in {destination}?",
            "Is {destination} good for families/couples/solo travelers?",
            "What's the nightlife like in {destination}?",
            "Any tips for first-time visitors to {destination}?",
        ]

        # Sample destinations for templates
        self.sample_destinations = [
            "Barcelona", "Paris", "Tokyo", "Bali", "New York",
            "Rome", "London", "Bangkok", "Sydney", "Amsterdam",
            "Prague", "Lisbon", "Iceland", "Morocco", "Greece",
            "Mexico City", "Vietnam", "Portugal", "Croatia", "Japan"
        ]

        self.activities = [
            "beaches", "hiking", "culture", "nightlife", "food",
            "history", "adventure", "relaxation", "photography", "shopping"
        ]

        self.trip_types = [
            "relaxing beach", "adventure", "cultural", "romantic",
            "budget backpacking", "luxury", "family-friendly", "foodie"
        ]

    def generate_qa_pair(self, question: str, answer: str) -> Dict:
        """Create a single Q&A training example"""
        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }

    def generate_from_knowledge(self, knowledge_base) -> List[Dict]:
        """Generate training data from travel knowledge base"""
        training_data = []

        for dest_key, dest in knowledge_base.destinations.items():
            name = dest.get('name', '')
            country = dest.get('country', '')

            # Generate various Q&A pairs for each destination

            # Best time to visit
            if dest.get('visited_date'):
                q = f"When is the best time to visit {name}?"
                a = f"Based on my visit to {name}, {country}, I'd recommend going around {dest.get('visited_date')}. "
                if dest.get('tips'):
                    a += f"A tip: {dest.get('tips')}"
                training_data.append(self.generate_qa_pair(q, a))

            # Things to do
            if dest.get('activities'):
                q = f"What should I do in {name}?"
                a = f"When I was in {name}, I really enjoyed: {', '.join(dest.get('activities'))}. "
                if dest.get('highlights'):
                    a += f"The highlights were definitely {', '.join(dest.get('highlights'))}."
                training_data.append(self.generate_qa_pair(q, a))

            # Food recommendations
            if dest.get('restaurants'):
                q = f"Where should I eat in {name}?"
                restaurants = dest.get('restaurants')
                a = f"Oh, the food in {name} is amazing! My favorite spots were: {', '.join(restaurants)}. "
                a += "I'd definitely recommend trying the local cuisine."
                training_data.append(self.generate_qa_pair(q, a))

            # Budget
            if dest.get('budget_per_day'):
                q = f"How expensive is {name}?"
                budget = dest.get('budget_per_day')
                a = f"For {name}, I spent about ${budget} per day on average. "
                a += f"That covered accommodation, food, and activities. "
                if budget < 100:
                    a += "It's quite budget-friendly!"
                elif budget > 200:
                    a += "It can be a bit pricey, but worth it for the experience."
                training_data.append(self.generate_qa_pair(q, a))

            # Rating/recommendation
            if dest.get('rating'):
                q = f"Would you recommend visiting {name}?"
                rating = dest.get('rating')
                would_return = dest.get('would_return', False)
                a = f"I'd give {name} a {rating}/10! "
                if would_return:
                    a += "I would definitely go back. "
                if dest.get('highlights'):
                    a += f"The {dest.get('highlights')[0]} alone is worth the trip."
                if dest.get('lowlights'):
                    a += f" Just be aware of: {', '.join(dest.get('lowlights'))}."
                training_data.append(self.generate_qa_pair(q, a))

        # Generate from experiences
        for exp in knowledge_base.experiences:
            q = f"Tell me about your experience in {exp.get('destination')}"
            a = f"{exp.get('title')}: {exp.get('story')}"
            training_data.append(self.generate_qa_pair(q, a))

        # Generate from tips
        for category, tips in knowledge_base.tips.items():
            if tips:
                q = f"Do you have any {category} tips for traveling?"
                tip_texts = [t.get('tip') for t in tips[:5]]
                a = f"Yes! Here are my top {category} tips:\n"
                for i, tip in enumerate(tip_texts, 1):
                    a += f"{i}. {tip}\n"
                training_data.append(self.generate_qa_pair(q, a))

        return training_data

    def generate_synthetic_data(self, num_examples: int = 100) -> List[Dict]:
        """Generate synthetic training examples"""
        training_data = []

        # This would ideally be filled with real answers
        # For now, create template-based examples

        for _ in range(num_examples):
            template = random.choice(self.question_templates)
            dest = random.choice(self.sample_destinations)
            dest2 = random.choice([d for d in self.sample_destinations if d != dest])
            activity = random.choice(self.activities)
            trip_type = random.choice(self.trip_types)
            budget = random.choice(["$1000", "$2000", "$5000", "$500"])

            question = template.format(
                destination=dest,
                destination1=dest,
                destination2=dest2,
                activity=activity,
                trip_type=trip_type,
                budget=budget
            )

            # Placeholder answer - you would fill these in or generate with an LLM
            answer = f"[NEEDS ANSWER] Question about {dest}: {question}"

            training_data.append({
                "question": question,
                "answer": answer,
                "needs_completion": True
            })

        return training_data

    def save_training_data(self, data: List[Dict], filename: str):
        """Save training data to JSONL file"""
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

        print(f"Saved {len(data)} examples to {filepath}")
        return filepath

    def load_training_data(self, filename: str) -> List[Dict]:
        """Load training data from JSONL file"""
        filepath = os.path.join(self.output_dir, filename)
        data = []

        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                for line in f:
                    data.append(json.loads(line))

        return data

    def export_for_finetuning(self, data: List[Dict], format: str = "openai") -> str:
        """Export data in format ready for fine-tuning"""
        if format == "openai":
            # OpenAI fine-tuning format
            filename = f"travel_finetune_openai_{datetime.now().strftime('%Y%m%d')}.jsonl"
        elif format == "llama":
            # Convert to Llama format
            converted = []
            for item in data:
                if "messages" in item:
                    text = ""
                    for msg in item["messages"]:
                        if msg["role"] == "system":
                            text += f"<|system|>\n{msg['content']}\n"
                        elif msg["role"] == "user":
                            text += f"<|user|>\n{msg['content']}\n"
                        elif msg["role"] == "assistant":
                            text += f"<|assistant|>\n{msg['content']}\n"
                    converted.append({"text": text})
            data = converted
            filename = f"travel_finetune_llama_{datetime.now().strftime('%Y%m%d')}.jsonl"
        else:
            filename = f"travel_finetune_{datetime.now().strftime('%Y%m%d')}.jsonl"

        return self.save_training_data(data, filename)


class TravelDataCollector:
    """Collect travel data from various sources"""

    def __init__(self):
        self.output_dir = TRAINING_DATA_DIR

    def collect_from_text(self, text: str, source: str = "manual") -> List[Dict]:
        """
        Parse travel information from text
        (e.g., copy-pasted reviews, blog posts, etc.)
        """
        # Simple extraction - would be enhanced with NLP
        entries = []

        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if len(para) > 50:  # Skip short paragraphs
                entries.append({
                    "text": para.strip(),
                    "source": source,
                    "collected": datetime.now().isoformat()
                })

        return entries

    def create_qa_from_text(self, text: str, questions: List[str]) -> List[Dict]:
        """
        Given a travel text, create Q&A pairs
        Questions are provided, answers extracted from text
        """
        # This would ideally use an LLM to extract answers
        qa_pairs = []

        for question in questions:
            qa_pairs.append({
                "question": question,
                "context": text,
                "answer": "[EXTRACT FROM CONTEXT]",
                "needs_review": True
            })

        return qa_pairs

    def import_trip_report(self, filepath: str) -> Dict:
        """Import a trip report document"""
        with open(filepath, 'r') as f:
            content = f.read()

        return {
            "filename": os.path.basename(filepath),
            "content": content,
            "imported": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Demo
    generator = TravelTrainingDataGenerator()

    # Generate synthetic examples (templates to fill in)
    synthetic = generator.generate_synthetic_data(20)
    generator.save_training_data(synthetic, "synthetic_templates.jsonl")

    print(f"\nGenerated {len(synthetic)} synthetic templates")
    print("Edit these files to add your real travel knowledge!")
    print(f"\nFiles saved to: {TRAINING_DATA_DIR}")
