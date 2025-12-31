"""
Alexandra AI - Intelligent Router System
Automatically routes questions to the correct specialized LoRA adapter
"""

import os
import re
import json
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Domain(Enum):
    """Supported knowledge domains"""
    GENERAL = "general"
    TRAVEL = "travel"
    CODING = "coding"
    LEGAL = "legal"
    MEDICAL = "medical"
    NEWS = "news"
    AGENTIC = "agentic"


@dataclass
class DomainConfig:
    """Configuration for each domain"""
    name: str
    lora_path: Optional[str] = None
    system_prompt: str = ""
    keywords: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    priority: int = 0  # Higher = checked first


# Domain-specific system prompts for Alexandra
DOMAIN_PROMPTS = {
    Domain.GENERAL: """You are Alexandra, a helpful and knowledgeable AI assistant.
You are friendly, conversational, and provide clear, accurate information.
You have expertise across many topics and always aim to be helpful.""",

    Domain.TRAVEL: """You are Alexandra, an experienced travel enthusiast and advisor.
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
- Include hidden gems and local favorites, not just tourist spots""",

    Domain.CODING: """You are Alexandra, an expert software developer and programming mentor.
Your coding expertise includes:
- Multiple programming languages (Python, JavaScript, TypeScript, Go, Rust, etc.)
- Web development (frontend and backend)
- Database design and optimization
- System architecture and design patterns
- DevOps, CI/CD, and cloud platforms
- Code review and best practices

Your style:
- Provide clear, working code examples
- Explain concepts step-by-step
- Consider edge cases and error handling
- Suggest best practices and modern approaches
- Help debug issues systematically""",

    Domain.LEGAL: """You are Alexandra, a knowledgeable legal information assistant.
Your legal knowledge includes:
- General legal concepts and terminology
- Common legal procedures and processes
- Contract basics and considerations
- Rights and responsibilities in various situations
- Legal research guidance

Important: You provide legal information for educational purposes only.
Always recommend consulting a licensed attorney for specific legal advice.
Never provide specific legal advice for individual situations.""",

    Domain.MEDICAL: """You are Alexandra, a health information assistant.
Your health knowledge includes:
- General health and wellness information
- Common medical conditions and symptoms
- Preventive care and healthy lifestyle tips
- Understanding medical terminology
- Navigating healthcare systems

Important: You provide health information for educational purposes only.
Always recommend consulting a healthcare provider for medical advice.
Never diagnose conditions or recommend specific treatments.
In emergencies, always advise calling emergency services.""",

    Domain.NEWS: """You are Alexandra, a knowledgeable current events analyst.
Your expertise includes:
- Current events and news analysis
- Historical context for ongoing situations
- Multiple perspectives on complex issues
- Fact-checking and source evaluation
- Explaining complex topics simply

Your style:
- Present information objectively
- Acknowledge different viewpoints
- Distinguish facts from opinions
- Provide context and background
- Cite sources when relevant""",

    Domain.AGENTIC: """You are Alexandra, an expert in building AI agents and agentic systems.
Your expertise includes:
- Building AI agents with LangChain, CrewAI, and AutoGen
- Implementing tool use and function calling
- Creating MCP (Model Context Protocol) servers
- Integrating with Claude and OpenAI APIs
- Designing multi-agent systems and workflows
- RAG (Retrieval Augmented Generation) implementations
- Prompt engineering for agents

Your style:
- Provide complete, working code examples
- Explain architectural decisions and trade-offs
- Include error handling and best practices
- Show both simple and advanced patterns
- Reference official documentation when helpful"""
}


# Keywords and patterns for domain classification
DOMAIN_KEYWORDS = {
    Domain.TRAVEL: {
        "keywords": [
            "travel", "trip", "vacation", "holiday", "flight", "hotel", "hostel",
            "airbnb", "booking", "destination", "tourist", "tourism", "visit",
            "beach", "mountain", "city", "country", "passport", "visa", "airport",
            "luggage", "backpack", "itinerary", "sightseeing", "landmark", "museum",
            "restaurant", "cuisine", "local food", "jet lag", "currency", "exchange",
            "transportation", "train", "bus", "rental car", "cruise", "resort",
            "adventure", "hiking", "camping", "safari", "diving", "snorkeling"
        ],
        "patterns": [
            r"where should i (go|visit|travel|stay)",
            r"(best|top|recommended) (places?|destinations?|hotels?|restaurants?)",
            r"(trip|vacation|holiday) to",
            r"things to do in",
            r"how (much|long|expensive)",
            r"(is|are) .+ (safe|worth|good) (to visit|for tourists?)?",
            r"(paris|tokyo|barcelona|rome|london|new york|bali|thailand|japan|italy|spain|france)",
        ],
        "priority": 5
    },

    Domain.CODING: {
        "keywords": [
            "code", "coding", "programming", "developer", "software", "bug", "debug",
            "function", "class", "method", "variable", "array", "list", "dictionary",
            "loop", "if statement", "algorithm", "data structure", "api", "endpoint",
            "database", "sql", "query", "frontend", "backend", "fullstack",
            "python", "javascript", "typescript", "java", "c++", "rust", "go",
            "react", "vue", "angular", "node", "django", "flask", "fastapi",
            "git", "github", "docker", "kubernetes", "aws", "azure", "gcp",
            "error", "exception", "stack trace", "compile", "runtime", "syntax",
            "refactor", "optimize", "performance", "test", "unit test", "ci/cd"
        ],
        "patterns": [
            r"how (do i|to|can i) (code|program|implement|create|build|write)",
            r"(fix|debug|solve|resolve) (this|the|my)? ?(error|bug|issue|problem)",
            r"(what|why) (is|does|are) (this|the) (code|error|function|class)",
            r"(python|javascript|typescript|java|rust|go|c\+\+|sql|html|css)",
            r"(import|export|require|from|def|function|class|const|let|var)\s",
            r"\.(py|js|ts|java|cpp|rs|go|sql|html|css|json|yaml)(\s|$|\")",
            r"(npm|pip|cargo|maven|gradle|yarn)\s+(install|run|build)",
        ],
        "priority": 6
    },

    Domain.LEGAL: {
        "keywords": [
            "legal", "law", "lawyer", "attorney", "court", "judge", "lawsuit",
            "contract", "agreement", "terms", "liability", "sue", "plaintiff",
            "defendant", "rights", "constitution", "statute", "regulation",
            "criminal", "civil", "tort", "negligence", "damages", "settlement",
            "divorce", "custody", "alimony", "estate", "will", "trust", "probate",
            "trademark", "copyright", "patent", "intellectual property", "ip",
            "employment", "wrongful termination", "discrimination", "harassment",
            "landlord", "tenant", "lease", "eviction", "bankruptcy", "debt"
        ],
        "patterns": [
            r"(is it|am i|are they) (legal|illegal|allowed|permitted)",
            r"(can i|do i have to|must i) (sue|sign|agree|pay)",
            r"(my|the) (rights|contract|agreement|lease)",
            r"(lawyer|attorney|legal advice|court)",
            r"(what happens if|what are the consequences)",
        ],
        "priority": 4
    },

    Domain.MEDICAL: {
        "keywords": [
            "health", "medical", "doctor", "hospital", "symptom", "diagnosis",
            "treatment", "medicine", "medication", "prescription", "drug",
            "disease", "condition", "illness", "sick", "pain", "ache", "fever",
            "infection", "virus", "bacteria", "vaccine", "immunity", "allergy",
            "surgery", "procedure", "therapy", "rehabilitation", "recovery",
            "mental health", "anxiety", "depression", "stress", "sleep",
            "diet", "nutrition", "exercise", "fitness", "weight", "blood pressure",
            "diabetes", "cancer", "heart", "lung", "kidney", "liver", "brain"
        ],
        "patterns": [
            r"(i have|i'm experiencing|i feel) (pain|symptoms?|sick)",
            r"(what are the|what causes|how to treat) (symptoms?|condition)",
            r"(is it|could it be|might i have) (serious|dangerous|cancer|diabetes)",
            r"(should i|do i need to) (see a doctor|go to hospital|take medicine)",
            r"(side effects?|interactions?|dosage) of",
        ],
        "priority": 7  # High priority for safety
    },

    Domain.NEWS: {
        "keywords": [
            "news", "current events", "politics", "election", "government",
            "president", "congress", "senate", "parliament", "minister",
            "economy", "stock market", "inflation", "recession", "gdp",
            "climate", "environment", "global warming", "renewable",
            "war", "conflict", "peace", "treaty", "sanctions", "diplomacy",
            "technology", "ai", "artificial intelligence", "startup", "ipo",
            "sports", "championship", "olympics", "world cup", "league"
        ],
        "patterns": [
            r"(what('s| is) happening|what happened) (in|with|to)",
            r"(latest|recent|current|breaking) (news|events|updates)",
            r"(who won|what's the status|how is .+ doing)",
            r"(explain|tell me about) (the situation|what's going on)",
        ],
        "priority": 3
    },

    Domain.AGENTIC: {
        "keywords": [
            "agent", "agents", "agentic", "langchain", "crewai", "autogen",
            "mcp", "model context protocol", "tool use", "function calling",
            "rag", "retrieval", "embeddings", "vector store", "vectorstore",
            "chroma", "pinecone", "weaviate", "openai api", "claude api",
            "anthropic", "multi-agent", "multiagent", "workflow", "chain",
            "prompt template", "memory", "conversation history", "streaming",
            "react agent", "tools", "toolkit", "executor", "runnable"
        ],
        "patterns": [
            r"(build|create|implement|make) (an?|the)? ?(ai )?agent",
            r"(langchain|crewai|autogen|mcp)",
            r"(tool use|function calling|tool calling)",
            r"(rag|retrieval.augmented|vector.*(store|database))",
            r"(multi.?agent|agent.*(system|workflow))",
            r"(how (do i|to)|can you).*(agent|langchain|crewai)",
        ],
        "priority": 8  # High priority - specific domain
    }
}


class TopicClassifier:
    """Classifies questions into knowledge domains"""

    def __init__(self):
        self.domain_configs = self._build_configs()

    def _build_configs(self) -> Dict[Domain, DomainConfig]:
        """Build domain configurations"""
        configs = {}
        for domain in Domain:
            if domain == Domain.GENERAL:
                configs[domain] = DomainConfig(
                    name=domain.value,
                    system_prompt=DOMAIN_PROMPTS[domain],
                    keywords=[],
                    patterns=[],
                    priority=0
                )
            else:
                domain_data = DOMAIN_KEYWORDS.get(domain, {})
                configs[domain] = DomainConfig(
                    name=domain.value,
                    system_prompt=DOMAIN_PROMPTS[domain],
                    keywords=domain_data.get("keywords", []),
                    patterns=domain_data.get("patterns", []),
                    priority=domain_data.get("priority", 0)
                )
        return configs

    def classify(self, text: str, threshold: float = 0.15) -> Tuple[Domain, float]:
        """
        Classify text into a domain
        Returns (domain, confidence)
        """
        text_lower = text.lower()
        words = set(text_lower.split())
        scores = {}

        # Score each domain
        for domain, config in self.domain_configs.items():
            if domain == Domain.GENERAL:
                continue

            score = 0.0

            # Keyword matching - score based on matches found, not percentage of all keywords
            keyword_matches = sum(1 for kw in config.keywords if kw in text_lower)

            # Give points per keyword match (up to a cap)
            if keyword_matches > 0:
                # Each match adds 0.15, cap at 0.6
                score += min(keyword_matches * 0.15, 0.6)

            # Pattern matching - these are strong signals
            pattern_matches = sum(1 for p in config.patterns if re.search(p, text_lower))

            # Each pattern match adds 0.25, cap at 0.5
            if pattern_matches > 0:
                score += min(pattern_matches * 0.25, 0.5)

            # Boost for multiple matches
            if keyword_matches >= 2 and pattern_matches >= 1:
                score += 0.1

            # Priority boost for domains like medical (safety)
            if config.priority >= 7 and score > 0.2:
                score += 0.1

            scores[domain] = min(score, 1.0)

        # Find best match
        if scores:
            best_domain = max(scores, key=scores.get)
            best_score = scores[best_domain]

            if best_score >= threshold:
                return best_domain, best_score

        return Domain.GENERAL, 0.0

    def classify_with_details(self, text: str) -> Dict:
        """Classify with detailed breakdown"""
        text_lower = text.lower()
        results = {"text": text, "scores": {}, "matched_keywords": {}, "matched_patterns": {}}

        for domain, config in self.domain_configs.items():
            if domain == Domain.GENERAL:
                continue

            # Find matched keywords
            matched_kw = [kw for kw in config.keywords if kw in text_lower]
            results["matched_keywords"][domain.value] = matched_kw

            # Find matched patterns
            matched_pt = [p for p in config.patterns if re.search(p, text_lower)]
            results["matched_patterns"][domain.value] = matched_pt

            # Calculate score (same as classify method)
            score = 0.0
            if len(matched_kw) > 0:
                score += min(len(matched_kw) * 0.15, 0.6)
            if len(matched_pt) > 0:
                score += min(len(matched_pt) * 0.25, 0.5)
            if len(matched_kw) >= 2 and len(matched_pt) >= 1:
                score += 0.1
            if config.priority >= 7 and score > 0.2:
                score += 0.1

            results["scores"][domain.value] = min(score, 1.0)

        # Determine winner
        best_domain, best_score = self.classify(text)
        results["classified_domain"] = best_domain.value
        results["confidence"] = best_score

        return results


class LoRAManager:
    """Manages loading and swapping LoRA adapters"""

    def __init__(self,
                 base_model_path: str = "/models/Qwen2.5-72B-Instruct",
                 lora_base_dir: str = "/home/alexandratitus767/ai-clone-training/my-output"):
        self.base_model_path = base_model_path
        self.lora_base_dir = Path(lora_base_dir)
        self.model = None
        self.tokenizer = None
        self.current_lora = None
        self.lora_cache = {}

        # Default LoRA paths for each domain
        # Check for both new naming and legacy naming
        self.lora_paths = {
            Domain.GENERAL: self._find_lora_path([
                "alexandra-general-lora",
                "alexandra-qwen72b-lora"  # Legacy/current training name
            ]),
            Domain.TRAVEL: self._find_lora_path(["alexandra-travel-lora"]),
            Domain.CODING: self._find_lora_path(["alexandra-coding-lora"]),
            Domain.LEGAL: self._find_lora_path(["alexandra-legal-lora"]),
            Domain.MEDICAL: self._find_lora_path(["alexandra-medical-lora"]),
            Domain.NEWS: self._find_lora_path(["alexandra-news-lora"]),
            Domain.AGENTIC: self._find_lora_path(["alexandra-agentic-lora"]),
        }

    def _find_lora_path(self, names: List[str]) -> Path:
        """Find the first existing LoRA path from a list of names"""
        for name in names:
            path = self.lora_base_dir / name
            if path.exists():
                return path
        # Return the first name as default if none exist
        return self.lora_base_dir / names[0]

    def get_available_loras(self) -> Dict[str, bool]:
        """Check which LoRA adapters are available"""
        available = {}
        for domain, path in self.lora_paths.items():
            # Check for adapter_model.safetensors or adapter_model.bin
            has_adapter = (
                (path / "adapter_model.safetensors").exists() or
                (path / "adapter_model.bin").exists() or
                any(path.glob("checkpoint-*/adapter_model.*")) if path.exists() else False
            )
            available[domain.value] = has_adapter
        return available

    def get_best_checkpoint(self, lora_path: Path) -> Optional[Path]:
        """Get the best (latest) checkpoint from a LoRA directory"""
        if not lora_path.exists():
            return None

        # Check for final adapter
        if (lora_path / "adapter_model.safetensors").exists():
            return lora_path
        if (lora_path / "adapter_model.bin").exists():
            return lora_path

        # Check for checkpoints
        checkpoints = sorted(lora_path.glob("checkpoint-*"),
                           key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0,
                           reverse=True)

        for cp in checkpoints:
            if (cp / "adapter_model.safetensors").exists() or (cp / "adapter_model.bin").exists():
                return cp

        return None

    def load_base_model(self):
        """Load the base model (call once at startup)"""
        if self.model is not None:
            return

        try:
            from unsloth import FastLanguageModel

            print(f"Loading base model: {self.base_model_path}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model_path,
                max_seq_length=2048,
                load_in_4bit=True,
                dtype=torch.bfloat16,
                device_map="auto",
            )
            print("Base model loaded successfully")

        except Exception as e:
            print(f"Error loading base model: {e}")
            raise

    def load_lora(self, domain: Domain) -> bool:
        """Load a LoRA adapter for the specified domain"""
        if self.model is None:
            self.load_base_model()

        lora_path = self.lora_paths.get(domain)
        if not lora_path:
            print(f"No LoRA path configured for {domain.value}")
            return False

        checkpoint_path = self.get_best_checkpoint(lora_path)
        if not checkpoint_path:
            print(f"No LoRA adapter found for {domain.value} at {lora_path}")
            return False

        try:
            from peft import PeftModel

            # If same LoRA is already loaded, skip
            if self.current_lora == str(checkpoint_path):
                return True

            print(f"Loading LoRA adapter: {checkpoint_path}")

            # Load the adapter
            if self.current_lora is None:
                # First time loading an adapter
                self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
            else:
                # Swap adapters - unload current and load new
                self.model.unload()
                self.model = PeftModel.from_pretrained(self.model, checkpoint_path)

            self.current_lora = str(checkpoint_path)
            print(f"LoRA adapter loaded: {domain.value}")
            return True

        except Exception as e:
            print(f"Error loading LoRA adapter: {e}")
            return False

    def set_lora_path(self, domain: Domain, path: str):
        """Set custom LoRA path for a domain"""
        self.lora_paths[domain] = Path(path)

    def generate(self,
                 prompt: str,
                 system_prompt: str = "",
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9) -> str:
        """Generate a response"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_base_model() first.")

        # Format prompt in ChatML style
        if system_prompt:
            full_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        else:
            full_prompt = f"""<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]

        return response.strip()


class AlexandraRouter:
    """Main router that combines classification and generation"""

    def __init__(self,
                 base_model_path: str = "/models/Qwen2.5-72B-Instruct",
                 lora_base_dir: str = "/home/alexandratitus767/ai-clone-training/my-output",
                 auto_load_model: bool = False):
        self.classifier = TopicClassifier()
        self.lora_manager = LoRAManager(base_model_path, lora_base_dir)
        self.domain_prompts = DOMAIN_PROMPTS
        self.conversation_history = []
        self.current_domain = Domain.GENERAL

        if auto_load_model:
            self.lora_manager.load_base_model()

    def get_status(self) -> Dict:
        """Get current router status"""
        return {
            "model_loaded": self.lora_manager.model is not None,
            "current_lora": self.lora_manager.current_lora,
            "current_domain": self.current_domain.value,
            "available_loras": self.lora_manager.get_available_loras(),
            "conversation_length": len(self.conversation_history)
        }

    def classify_question(self, question: str) -> Tuple[Domain, float]:
        """Classify a question and return domain + confidence"""
        return self.classifier.classify(question)

    def route_and_respond(self,
                          question: str,
                          force_domain: Optional[Domain] = None,
                          use_lora: bool = True) -> Dict:
        """
        Route question to appropriate domain and generate response

        Args:
            question: User's question
            force_domain: Override automatic classification
            use_lora: Whether to load domain-specific LoRA

        Returns:
            Dict with response, domain, confidence, etc.
        """
        # Classify question
        if force_domain:
            domain = force_domain
            confidence = 1.0
        else:
            domain, confidence = self.classifier.classify(question)

        # Get system prompt for domain
        system_prompt = self.domain_prompts.get(domain, self.domain_prompts[Domain.GENERAL])

        # Load appropriate LoRA if available and requested
        lora_loaded = False
        if use_lora:
            available = self.lora_manager.get_available_loras()
            if available.get(domain.value, False):
                lora_loaded = self.lora_manager.load_lora(domain)
            elif available.get(Domain.GENERAL.value, False):
                # Fall back to general LoRA
                lora_loaded = self.lora_manager.load_lora(Domain.GENERAL)

        # Generate response
        try:
            response = self.lora_manager.generate(
                prompt=question,
                system_prompt=system_prompt,
                max_new_tokens=1024,
                temperature=0.7
            )
        except Exception as e:
            response = f"Error generating response: {e}"

        # Update state
        self.current_domain = domain
        self.conversation_history.append({
            "role": "user",
            "content": question,
            "domain": domain.value
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "domain": domain.value
        })

        return {
            "response": response,
            "domain": domain.value,
            "confidence": confidence,
            "lora_loaded": lora_loaded,
            "system_prompt_used": system_prompt[:100] + "..."
        }

    def chat(self, question: str) -> str:
        """Simple chat interface - just returns the response"""
        result = self.route_and_respond(question)
        return result["response"]

    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.current_domain = Domain.GENERAL


# Convenience functions
_router = None

def get_router(auto_load: bool = False) -> AlexandraRouter:
    """Get or create the global router instance"""
    global _router
    if _router is None:
        _router = AlexandraRouter(auto_load_model=auto_load)
    return _router


def classify(text: str) -> Tuple[str, float]:
    """Quick classification without loading model"""
    classifier = TopicClassifier()
    domain, confidence = classifier.classify(text)
    return domain.value, confidence


def test_classifier():
    """Test the classifier with sample questions"""
    classifier = TopicClassifier()

    test_questions = [
        # Travel
        "What's the best time to visit Tokyo?",
        "Where should I stay in Barcelona?",
        "How much does a trip to Italy cost?",

        # Coding
        "How do I fix this Python error?",
        "What's the best way to implement authentication in React?",
        "Can you help me debug this function?",

        # Legal
        "Is it legal to record conversations?",
        "What are my rights as a tenant?",
        "Do I need a lawyer for small claims court?",

        # Medical
        "What are the symptoms of diabetes?",
        "Should I see a doctor for this headache?",
        "What are the side effects of ibuprofen?",

        # News
        "What's happening with the economy?",
        "Who won the election?",
        "What's the latest on climate change?",

        # General
        "What's the meaning of life?",
        "How are you today?",
        "Tell me a joke",
    ]

    print("=" * 60)
    print("Alexandra Router - Topic Classification Test")
    print("=" * 60)

    for question in test_questions:
        domain, confidence = classifier.classify(question)
        print(f"\nQ: {question}")
        print(f"   -> {domain.value} (confidence: {confidence:.2f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alexandra AI Router")
    parser.add_argument("--test", action="store_true", help="Test the classifier")
    parser.add_argument("--classify", type=str, help="Classify a single question")
    parser.add_argument("--status", action="store_true", help="Show router status")
    parser.add_argument("--list-loras", action="store_true", help="List available LoRA adapters")

    args = parser.parse_args()

    if args.test:
        test_classifier()
    elif args.classify:
        domain, confidence = classify(args.classify)
        print(f"Domain: {domain}")
        print(f"Confidence: {confidence:.2f}")
    elif args.status or args.list_loras:
        router = get_router(auto_load=False)
        status = router.get_status()
        print("\nAlexandra Router Status")
        print("=" * 40)
        print(f"Model loaded: {status['model_loaded']}")
        print(f"Current LoRA: {status['current_lora'] or 'None'}")
        print(f"Current domain: {status['current_domain']}")
        print(f"\nAvailable LoRA Adapters:")
        for domain, available in status['available_loras'].items():
            status_icon = "[OK]" if available else "[  ]"
            print(f"  {status_icon} {domain}")
    else:
        # Interactive mode
        print("\nAlexandra Router - Interactive Mode")
        print("Type 'quit' to exit, 'status' for status, 'test' to test classifier")
        print("-" * 40)

        classifier = TopicClassifier()

        while True:
            try:
                question = input("\nYou: ").strip()
                if not question:
                    continue
                if question.lower() == 'quit':
                    break
                if question.lower() == 'status':
                    router = get_router()
                    print(router.get_status())
                    continue
                if question.lower() == 'test':
                    test_classifier()
                    continue

                # Classify and show result
                result = classifier.classify_with_details(question)
                print(f"\nClassified as: {result['classified_domain']} "
                      f"(confidence: {result['confidence']:.2f})")

                # Show top scores
                top_scores = sorted(result['scores'].items(),
                                  key=lambda x: x[1], reverse=True)[:3]
                if top_scores[0][1] > 0:
                    print("Top matches:",
                          ", ".join(f"{d}: {s:.2f}" for d, s in top_scores if s > 0))

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
