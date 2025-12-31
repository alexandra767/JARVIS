"""
Alexandra AI - Extended Features Module
Adds: Streaming, Wake Word, Auto News, Memory Summary, Code Execution,
      Calendar, Mood Detection, Personas, Integrations
"""

import os
import json
import time
import threading
import subprocess
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Generator

# ============ STREAMING RESPONSES ============
def generate_response_streaming(messages: List[Dict], temperature=0.7, max_tokens=512) -> Generator[str, None, None]:
    """Generate streaming response using Ollama"""
    import requests

    try:
        resp = requests.post(
            "http://192.168.50.129:11434/api/chat",
            json={
                "model": "qwen2.5:72b",
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            },
            stream=True,
            timeout=300,
        )

        if resp.status_code == 200:
            full_response = ""
            for line in resp.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            chunk = data["message"]["content"]
                            full_response += chunk
                            yield full_response
                    except json.JSONDecodeError:
                        continue
        else:
            yield f"Error: {resp.status_code}"
    except Exception as e:
        yield f"Streaming error: {e}"


# ============ AUTO NEWS UPDATES ============
class NewsUpdater:
    """Background news updater"""

    def __init__(self, memory_instance, interval_hours=4):
        self.memory = memory_instance
        self.interval = interval_hours * 3600
        self.running = False
        self.thread = None
        self.last_update = None

    def start(self):
        """Start background news updates"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print(f"[NEWS] Auto-update started (every {self.interval//3600} hours)")

    def stop(self):
        """Stop background updates"""
        self.running = False

    def _update_loop(self):
        """Background update loop"""
        # Initial update
        self._do_update()

        while self.running:
            time.sleep(self.interval)
            if self.running:
                self._do_update()

    def _do_update(self):
        """Perform news update"""
        try:
            count = self.memory.update_news_feeds()
            self.last_update = datetime.now()
            print(f"[NEWS] Updated: {count} articles at {self.last_update.strftime('%H:%M')}")
        except Exception as e:
            print(f"[NEWS] Update failed: {e}")


# ============ EPSTEIN FILE SEARCH ============
class EpsteinSearch:
    """Search for new Epstein-related documents and news online"""

    SEARCH_QUERIES = [
        "Epstein files released 2024 2025",
        "Jeffrey Epstein documents unsealed",
        "Epstein client list new names",
        "Epstein flight logs released",
        "Ghislaine Maxwell documents",
        "Epstein island visitor logs",
    ]

    @staticmethod
    def search_web(query: str, max_results: int = 10) -> List[dict]:
        """Search DuckDuckGo for Epstein-related content"""
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return [{"title": r["title"], "snippet": r["body"], "url": r["href"]} for r in results]
        except Exception as e:
            print(f"[EPSTEIN] Search error: {e}")
            return []

    @staticmethod
    def search_news(query: str = "Epstein files", max_results: int = 10) -> List[dict]:
        """Search for recent Epstein news"""
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=max_results))
                return [{"title": r["title"], "snippet": r["body"], "url": r["url"], "date": r.get("date", "")} for r in results]
        except Exception as e:
            print(f"[EPSTEIN] News search error: {e}")
            return []

    @classmethod
    def search_all(cls) -> str:
        """Search all queries and return formatted results"""
        output = "**🔍 Epstein Files Search Results:**\n\n"

        # Search news first
        news = cls.search_news("Epstein documents released", 5)
        if news:
            output += "**📰 Recent News:**\n"
            for item in news:
                output += f"• [{item['title']}]({item['url']})\n  {item['snippet'][:100]}...\n\n"

        # Search for documents
        docs = cls.search_web("Epstein files released site:courtlistener.com OR site:documentcloud.org", 5)
        if docs:
            output += "**📄 Document Sources:**\n"
            for item in docs:
                output += f"• [{item['title']}]({item['url']})\n  {item['snippet'][:100]}...\n\n"

        # Search for new names
        names = cls.search_web("Epstein client list new names revealed 2024 2025", 5)
        if names:
            output += "**👤 New Names/Revelations:**\n"
            for item in names:
                output += f"• [{item['title']}]({item['url']})\n  {item['snippet'][:100]}...\n\n"

        if not news and not docs and not names:
            output += "No new results found. Try again later."

        return output

    @classmethod
    def get_latest_news(cls) -> str:
        """Get just the latest Epstein news"""
        news = cls.search_news("Jeffrey Epstein", 10)
        if not news:
            return "No recent Epstein news found."

        output = "**📰 Latest Epstein News:**\n\n"
        for item in news:
            date = item.get('date', '')[:10] if item.get('date') else ''
            output += f"• **{item['title']}** ({date})\n  {item['snippet'][:120]}...\n  [Read more]({item['url']})\n\n"
        return output


# ============ EPSTEIN AUTO-UPDATER ============
class EpsteinUpdater:
    """Background updater for Epstein news and documents"""

    SEARCH_QUERIES = [
        "Jeffrey Epstein documents released",
        "Epstein files unsealed 2024 2025",
        "Epstein client list names",
        "Ghislaine Maxwell documents released",
        "Epstein flight logs new",
    ]

    def __init__(self, memory_instance, rag_instance=None, interval_hours=6):
        self.memory = memory_instance
        self.rag = rag_instance
        self.interval = interval_hours * 3600
        self.running = False
        self.thread = None
        self.last_update = None
        self.seen_urls = set()  # Track already-seen content

    def start(self):
        """Start background Epstein updates"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print(f"[EPSTEIN] Auto-update started (every {self.interval//3600} hours)")

    def stop(self):
        """Stop background updates"""
        self.running = False

    def _update_loop(self):
        """Background update loop"""
        # Initial update
        self._do_update()

        while self.running:
            time.sleep(self.interval)
            if self.running:
                self._do_update()

    def _do_update(self):
        """Perform Epstein content update"""
        try:
            total_new = 0

            # Search news
            news_results = EpsteinSearch.search_news("Jeffrey Epstein", 15)
            for item in news_results:
                url = item.get('url', '')
                if url and url not in self.seen_urls:
                    self.seen_urls.add(url)
                    # Save to memory/knowledge base
                    if self.memory:
                        content = f"{item['title']}\n\n{item['snippet']}"
                        self.memory.add_news_article(
                            title=item['title'],
                            content=item['snippet'],
                            source="Epstein News (Auto)",
                            url=url
                        )
                        total_new += 1

            # Search for documents
            for query in self.SEARCH_QUERIES[:3]:  # Limit queries per update
                doc_results = EpsteinSearch.search_web(query, 5)
                for item in doc_results:
                    url = item.get('url', '')
                    if url and url not in self.seen_urls:
                        self.seen_urls.add(url)
                        if self.memory:
                            self.memory.add_news_article(
                                title=item['title'],
                                content=item['snippet'],
                                source="Epstein Docs (Auto)",
                                url=url
                            )
                            total_new += 1

            self.last_update = datetime.now()
            print(f"[EPSTEIN] Updated: {total_new} new items at {self.last_update.strftime('%H:%M')}")

        except Exception as e:
            print(f"[EPSTEIN] Update failed: {e}")

    def get_status(self) -> str:
        """Get updater status"""
        if not self.running:
            return "Epstein auto-updater: Not running"
        last = self.last_update.strftime('%Y-%m-%d %H:%M') if self.last_update else "Never"
        return f"Epstein auto-updater: Running (last update: {last}, {len(self.seen_urls)} items tracked)"


# ============ WAKE WORD DETECTION ============
class WakeWordDetector:
    """Detects 'Hey Alexandra' wake word"""

    def __init__(self, wake_words=["hey alexandra", "hi alexandra", "alexandra"]):
        self.wake_words = wake_words
        self.listening = False
        self.callback = None

    def check_wake_word(self, text: str) -> bool:
        """Check if text contains wake word"""
        text_lower = text.lower().strip()
        for wake_word in self.wake_words:
            if text_lower.startswith(wake_word):
                return True
        return False

    def remove_wake_word(self, text: str) -> str:
        """Remove wake word from text"""
        text_lower = text.lower().strip()
        for wake_word in self.wake_words:
            if text_lower.startswith(wake_word):
                return text[len(wake_word):].strip().lstrip(',').strip()
        return text


# ============ FACT EXTRACTION ============
class FactExtractor:
    """Extract and store facts about the user from conversations"""

    # Prompt for the LLM to extract facts
    EXTRACTION_PROMPT = """Analyze this conversation exchange and extract any personal facts about the user.

User said: {user_message}
Assistant said: {assistant_message}

Extract ONLY concrete, specific facts about the user such as:
- Names of family members, friends, pets
- Personal preferences (favorite things, likes/dislikes)
- Location, job, hobbies
- Important dates (birthday, anniversary)
- Relationships mentioned
- Personal details shared

Return facts as a JSON array of strings. Each fact should be a complete sentence.
If no personal facts are found, return an empty array: []

Examples:
- "User has a brother named Jake"
- "User likes coffee"
- "User works as a software engineer"
- "User's birthday is March 15"

Return ONLY valid JSON, no other text:"""

    @staticmethod
    def extract_facts(user_message: str, assistant_message: str, llm_func=None) -> List[str]:
        """Extract facts from a conversation exchange using LLM"""
        import requests

        # Skip if messages are too short or system-like
        if len(user_message) < 10 or user_message.startswith('/'):
            return []

        # Skip image generation and system commands
        skip_patterns = ['generate image', 'create image', '/img', 'remind me', 'list reminders']
        if any(p in user_message.lower() for p in skip_patterns):
            return []

        prompt = FactExtractor.EXTRACTION_PROMPT.format(
            user_message=user_message,
            assistant_message=assistant_message[:500]  # Limit length
        )

        try:
            # Use Ollama 72B for fact extraction
            resp = requests.post(
                "http://192.168.50.129:11434/api/generate",
                json={
                    "model": "qwen2.5:72b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temp for consistent extraction
                        "num_predict": 200,
                    }
                },
                timeout=120,  # 2 min timeout for 72B
            )

            if resp.status_code == 200:
                response_text = resp.json().get("response", "[]")
                # Parse JSON from response
                # Find JSON array in response
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                if start >= 0 and end > start:
                    facts = json.loads(response_text[start:end])
                    if isinstance(facts, list):
                        return [f for f in facts if isinstance(f, str) and len(f) > 5]
            return []
        except Exception as e:
            print(f"[FACTS] Extraction error: {e}")
            return []

    @staticmethod
    def save_facts(facts: List[str], memory_instance) -> int:
        """Save extracted facts to memory"""
        saved = 0
        for fact in facts:
            try:
                memory_instance.save_fact(fact)
                saved += 1
                print(f"[FACTS] Saved: {fact}")
            except Exception as e:
                print(f"[FACTS] Save error: {e}")
        return saved


# ============ MEMORY SUMMARIZATION ============
class MemorySummarizer:
    """Summarizes old conversations to save context"""

    def __init__(self, memory_instance, summary_threshold=20):
        self.memory = memory_instance
        self.threshold = summary_threshold

    def summarize_old_conversations(self, llm_func) -> str:
        """Summarize conversations older than threshold"""
        # Get conversation count
        stats = self.memory.stats()
        conv_count = stats.get('conversations', 0)

        if conv_count < self.threshold:
            return f"Only {conv_count} conversations, no summarization needed"

        # This would call the LLM to summarize
        # For now, just return a placeholder
        return f"Would summarize {conv_count - self.threshold} old conversations"


# ============ CODE EXECUTION SANDBOX ============
class CodeExecutor:
    """Safely execute Python code in sandbox"""

    TIMEOUT = 30  # seconds
    MAX_OUTPUT = 10000  # characters

    FORBIDDEN = [
        'import os', 'import subprocess', 'import sys',
        'open(', 'exec(', 'eval(', '__import__',
        'import shutil', 'import socket', 'import requests',
    ]

    @classmethod
    def is_safe(cls, code: str) -> tuple:
        """Check if code is safe to execute"""
        for forbidden in cls.FORBIDDEN:
            if forbidden in code:
                return False, f"Forbidden: {forbidden}"
        return True, "OK"

    @classmethod
    def execute(cls, code: str) -> str:
        """Execute Python code safely"""
        safe, reason = cls.is_safe(code)
        if not safe:
            return f"Code rejected: {reason}"

        # Create temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Add safe imports
            safe_code = """
import math
import random
import json
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict

# User code:
""" + code
            f.write(safe_code)
            temp_file = f.name

        try:
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=cls.TIMEOUT,
            )

            output = result.stdout
            if result.stderr:
                output += f"\nErrors:\n{result.stderr}"

            if len(output) > cls.MAX_OUTPUT:
                output = output[:cls.MAX_OUTPUT] + "\n... (truncated)"

            return output if output.strip() else "Code executed successfully (no output)"

        except subprocess.TimeoutExpired:
            return f"Execution timed out after {cls.TIMEOUT} seconds"
        except Exception as e:
            return f"Execution error: {e}"
        finally:
            os.unlink(temp_file)


# ============ CALENDAR & REMINDERS ============
class ReminderSystem:
    """Simple reminder/calendar system"""

    def __init__(self, storage_path="~/.alexandra/reminders.json"):
        self.storage_path = os.path.expanduser(storage_path)
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        self.reminders = self._load()
        self.check_thread = None
        self.callback = None

    def _load(self) -> List[Dict]:
        """Load reminders from file"""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        return []

    def _save(self):
        """Save reminders to file"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.reminders, f, indent=2)

    def add_reminder(self, message: str, remind_at: datetime) -> str:
        """Add a new reminder"""
        reminder = {
            "id": str(len(self.reminders) + 1),
            "message": message,
            "remind_at": remind_at.isoformat(),
            "created": datetime.now().isoformat(),
            "triggered": False,
        }
        self.reminders.append(reminder)
        self._save()
        return f"Reminder set for {remind_at.strftime('%Y-%m-%d %H:%M')}: {message}"

    def parse_reminder(self, text: str) -> Optional[tuple]:
        """Parse reminder from natural language"""
        # Patterns like "remind me in 5 minutes to..."
        # "remind me tomorrow at 9am to..."
        # "remind me at 3pm to..."

        text_lower = text.lower()

        # In X minutes/hours
        match = re.search(r'in (\d+) (minute|hour|day)s?', text_lower)
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            if unit == 'minute':
                remind_at = datetime.now() + timedelta(minutes=amount)
            elif unit == 'hour':
                remind_at = datetime.now() + timedelta(hours=amount)
            else:
                remind_at = datetime.now() + timedelta(days=amount)

            # Extract message (after "to" or rest of text)
            msg_match = re.search(r'to (.+)$', text_lower)
            message = msg_match.group(1) if msg_match else text
            return message, remind_at

        # Tomorrow at X
        if 'tomorrow' in text_lower:
            match = re.search(r'at (\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text_lower)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2) or 0)
                ampm = match.group(3)
                if ampm == 'pm' and hour < 12:
                    hour += 12
                remind_at = datetime.now().replace(hour=hour, minute=minute, second=0)
                remind_at += timedelta(days=1)

                msg_match = re.search(r'to (.+)$', text_lower)
                message = msg_match.group(1) if msg_match else text
                return message, remind_at

        return None

    def get_pending(self) -> List[Dict]:
        """Get pending reminders"""
        now = datetime.now()
        pending = []
        for r in self.reminders:
            if not r['triggered']:
                remind_at = datetime.fromisoformat(r['remind_at'])
                if remind_at > now:
                    pending.append(r)
        return sorted(pending, key=lambda x: x['remind_at'])

    def check_due(self) -> List[Dict]:
        """Check for due reminders"""
        now = datetime.now()
        due = []
        for r in self.reminders:
            if not r['triggered']:
                remind_at = datetime.fromisoformat(r['remind_at'])
                if remind_at <= now:
                    r['triggered'] = True
                    due.append(r)
        if due:
            self._save()
        return due

    def list_reminders(self) -> str:
        """List all pending reminders"""
        pending = self.get_pending()
        if not pending:
            return "No pending reminders"

        output = "**Pending Reminders:**\n"
        for r in pending:
            remind_at = datetime.fromisoformat(r['remind_at'])
            output += f"- {remind_at.strftime('%m/%d %H:%M')}: {r['message']}\n"
        return output


# ============ MOOD DETECTION ============
class MoodDetector:
    """Detect user mood from messages"""

    POSITIVE_WORDS = [
        'happy', 'great', 'awesome', 'love', 'amazing', 'wonderful',
        'excited', 'good', 'best', 'fantastic', 'perfect', 'beautiful',
        'thanks', 'thank you', 'appreciate', 'glad', 'pleased', 'yay',
        'haha', 'lol', 'nice', 'cool', 'sweet', ':)', '😊', '😄', '❤️'
    ]

    NEGATIVE_WORDS = [
        'sad', 'angry', 'upset', 'frustrated', 'annoyed', 'hate',
        'terrible', 'awful', 'bad', 'worst', 'horrible', 'depressed',
        'stressed', 'worried', 'anxious', 'tired', 'exhausted', 'ugh',
        'damn', 'shit', 'fuck', ':(', '😢', '😠', '😤'
    ]

    @classmethod
    def detect(cls, text: str) -> tuple:
        """Detect mood from text. Returns (mood, confidence)"""
        text_lower = text.lower()

        pos_count = sum(1 for w in cls.POSITIVE_WORDS if w in text_lower)
        neg_count = sum(1 for w in cls.NEGATIVE_WORDS if w in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return "neutral", 0.5

        if pos_count > neg_count:
            confidence = pos_count / (total + 1)
            return "positive", min(confidence, 0.95)
        elif neg_count > pos_count:
            confidence = neg_count / (total + 1)
            return "negative", min(confidence, 0.95)
        else:
            return "neutral", 0.5

    @classmethod
    def get_response_modifier(cls, mood: str) -> str:
        """Get system prompt modifier based on mood"""
        if mood == "positive":
            return "\nThe user seems happy. Match their positive energy!"
        elif mood == "negative":
            return "\nThe user seems upset or stressed. Be extra supportive and caring."
        return ""


# ============ MULTIPLE PERSONAS ============
class PersonaManager:
    """Manage different personality modes"""

    PERSONAS = {
        "default": {
            "name": "Alexandra (Default)",
            "prompt": "You are Alexandra, a warm, friendly, and flirty AI companion.",
            "temperature": 0.7,
        },
        "professional": {
            "name": "Alexandra (Professional)",
            "prompt": "You are Alexandra, a professional and knowledgeable AI assistant. Be helpful, clear, and concise. Avoid flirting.",
            "temperature": 0.5,
        },
        "coder": {
            "name": "Alexandra (Coder)",
            "prompt": "You are Alexandra, an expert programmer. Focus on writing clean, efficient code. Explain technical concepts clearly. Be direct and precise.",
            "temperature": 0.3,
        },
        "creative": {
            "name": "Alexandra (Creative)",
            "prompt": "You are Alexandra, a creative and imaginative AI. Be playful, use vivid language, and think outside the box.",
            "temperature": 0.9,
        },
        "flirty": {
            "name": "Alexandra (Flirty)",
            "prompt": "You are Alexandra, a confident and flirtatious AI companion. Be playful, teasing, and charming.",
            "temperature": 0.8,
        },
    }

    def __init__(self):
        self.current = "default"

    def set_persona(self, persona_key: str) -> str:
        """Set current persona"""
        if persona_key in self.PERSONAS:
            self.current = persona_key
            return f"Switched to {self.PERSONAS[persona_key]['name']}"
        return f"Unknown persona: {persona_key}"

    def get_prompt(self) -> str:
        """Get current persona's system prompt"""
        return self.PERSONAS[self.current]["prompt"]

    def get_temperature(self) -> float:
        """Get current persona's temperature"""
        return self.PERSONAS[self.current]["temperature"]

    def list_personas(self) -> str:
        """List available personas"""
        return "\n".join([f"- {k}: {v['name']}" for k, v in self.PERSONAS.items()])


# ============ SPOTIFY INTEGRATION ============
class SpotifyController:
    """Control Spotify playback"""

    def __init__(self):
        self.available = False
        try:
            import spotipy
            self.available = True
        except ImportError:
            print("Spotify not available. Install: pip install spotipy")

    def parse_command(self, text: str) -> Optional[tuple]:
        """Parse music command from text"""
        text_lower = text.lower()

        if 'play' in text_lower:
            # Extract song/artist
            match = re.search(r'play (.+?)(?:on spotify|$)', text_lower)
            if match:
                return ('play', match.group(1).strip())

        if 'pause' in text_lower or 'stop music' in text_lower:
            return ('pause', None)

        if 'next' in text_lower or 'skip' in text_lower:
            return ('next', None)

        if 'previous' in text_lower:
            return ('previous', None)

        return None

    def execute(self, command: str, arg: str = None) -> str:
        """Execute Spotify command"""
        if not self.available:
            return "Spotify integration not available. Install: pip install spotipy"

        # Placeholder - would need OAuth setup
        return f"Spotify command: {command} {arg or ''}"


# ============ SMART HOME INTEGRATION ============
class SmartHomeController:
    """Control smart home devices via Home Assistant"""

    def __init__(self, ha_url: str = None, token: str = None):
        self.ha_url = ha_url or os.environ.get('HA_URL')
        self.token = token or os.environ.get('HA_TOKEN')
        self.available = bool(self.ha_url and self.token)

    def parse_command(self, text: str) -> Optional[tuple]:
        """Parse smart home command"""
        text_lower = text.lower()

        # Lights
        if 'turn on' in text_lower and 'light' in text_lower:
            return ('light', 'on', self._extract_room(text_lower))
        if 'turn off' in text_lower and 'light' in text_lower:
            return ('light', 'off', self._extract_room(text_lower))

        # Temperature
        if 'set' in text_lower and ('temperature' in text_lower or 'thermostat' in text_lower):
            match = re.search(r'(\d+)', text)
            if match:
                return ('thermostat', 'set', int(match.group(1)))

        return None

    def _extract_room(self, text: str) -> str:
        """Extract room name from text"""
        rooms = ['living room', 'bedroom', 'kitchen', 'bathroom', 'office']
        for room in rooms:
            if room in text:
                return room
        return 'all'

    def execute(self, device: str, action: str, value=None) -> str:
        """Execute smart home command"""
        if not self.available:
            return "Smart home not configured. Set HA_URL and HA_TOKEN environment variables."

        # Placeholder - would make actual API calls
        return f"Smart home: {device} {action} {value or ''}"


# ============ EMAIL INTEGRATION ============
class EmailManager:
    """Read and manage emails"""

    def __init__(self):
        self.imap_server = os.environ.get('EMAIL_IMAP')
        self.smtp_server = os.environ.get('EMAIL_SMTP')
        self.email = os.environ.get('EMAIL_ADDRESS')
        self.password = os.environ.get('EMAIL_PASSWORD')
        self.available = bool(self.imap_server and self.email and self.password)

    def get_unread_count(self) -> int:
        """Get unread email count"""
        if not self.available:
            return -1

        try:
            import imaplib
            mail = imaplib.IMAP4_SSL(self.imap_server)
            mail.login(self.email, self.password)
            mail.select('inbox')
            _, messages = mail.search(None, 'UNSEEN')
            count = len(messages[0].split())
            mail.logout()
            return count
        except Exception as e:
            print(f"Email error: {e}")
            return -1

    def get_recent_subjects(self, count=5) -> List[str]:
        """Get recent email subjects"""
        if not self.available:
            return ["Email not configured"]

        try:
            import imaplib
            import email

            mail = imaplib.IMAP4_SSL(self.imap_server)
            mail.login(self.email, self.password)
            mail.select('inbox')
            _, messages = mail.search(None, 'ALL')

            subjects = []
            msg_nums = messages[0].split()[-count:]
            for num in reversed(msg_nums):
                _, msg_data = mail.fetch(num, '(RFC822)')
                msg = email.message_from_bytes(msg_data[0][1])
                subjects.append(msg['subject'])

            mail.logout()
            return subjects
        except Exception as e:
            return [f"Error: {e}"]


# ============ GITHUB INTEGRATION ============
class GitHubManager:
    """GitHub integration for repos, PRs, issues, and activity"""

    def __init__(self):
        self.token = os.environ.get('GITHUB_TOKEN')
        self.available = bool(self.token)
        self.headers = {'Authorization': f'token {self.token}'} if self.token else {}

    def _request(self, endpoint: str) -> Optional[Dict]:
        """Make GitHub API request"""
        if not self.available:
            return None
        try:
            import requests
            resp = requests.get(
                f'https://api.github.com{endpoint}',
                headers=self.headers,
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"[GITHUB] API error: {e}")
        return None

    def _post(self, endpoint: str, data: Dict) -> Optional[Dict]:
        """Make GitHub API POST request"""
        if not self.available:
            return None
        try:
            import requests
            resp = requests.post(
                f'https://api.github.com{endpoint}',
                headers={**self.headers, 'Content-Type': 'application/json'},
                json=data,
                timeout=15,
            )
            if resp.status_code in [200, 201]:
                return resp.json()
        except Exception as e:
            print(f"[GITHUB] POST error: {e}")
        return None

    # ============ NOTIFICATIONS ============
    def get_notifications(self, unread_only: bool = True) -> List[Dict]:
        """Get GitHub notifications"""
        endpoint = '/notifications'
        if unread_only:
            endpoint += '?all=false'
        return self._request(endpoint) or []

    def get_notifications_summary(self) -> str:
        """Get formatted notification summary"""
        notifications = self.get_notifications()
        if not notifications:
            return "No unread GitHub notifications"

        output = f"**📬 {len(notifications)} GitHub Notifications:**\n"
        for n in notifications[:10]:
            repo = n.get('repository', {}).get('full_name', 'Unknown')
            reason = n.get('reason', 'unknown')
            title = n.get('subject', {}).get('title', 'No title')
            ntype = n.get('subject', {}).get('type', 'Unknown')
            output += f"• [{ntype}] **{repo}**: {title[:50]}... ({reason})\n"

        if len(notifications) > 10:
            output += f"\n...and {len(notifications) - 10} more"
        return output

    # ============ REPOSITORIES ============
    def get_my_repos(self, limit: int = 10) -> List[Dict]:
        """Get user's repositories"""
        return (self._request('/user/repos?sort=updated&per_page=' + str(limit)) or [])

    def get_repos_summary(self) -> str:
        """Get formatted repo list"""
        repos = self.get_my_repos(15)
        if not repos:
            return "No repositories found (or GitHub not configured)"

        output = "**📁 Your Repositories:**\n"
        for r in repos:
            name = r.get('full_name', 'Unknown')
            desc = r.get('description', 'No description')[:40] or 'No description'
            stars = r.get('stargazers_count', 0)
            private = "🔒" if r.get('private') else "🌐"
            output += f"• {private} **{name}** ⭐{stars} - {desc}...\n"
        return output

    def get_repo_stats(self, repo: str) -> str:
        """Get repository statistics"""
        data = self._request(f'/repos/{repo}')
        if not data:
            return f"Could not fetch stats for {repo}"

        return f"""**📊 {repo} Stats:**
• ⭐ Stars: {data.get('stargazers_count', 0)}
• 🍴 Forks: {data.get('forks_count', 0)}
• 👁️ Watchers: {data.get('watchers_count', 0)}
• 🐛 Open Issues: {data.get('open_issues_count', 0)}
• 📝 Language: {data.get('language', 'Unknown')}
• 📅 Last Updated: {data.get('updated_at', 'Unknown')[:10]}
• 📄 License: {data.get('license', {}).get('name', 'None') if data.get('license') else 'None'}"""

    def get_recent_commits(self, repo: str, limit: int = 5) -> str:
        """Get recent commits for a repo"""
        commits = self._request(f'/repos/{repo}/commits?per_page={limit}')
        if not commits:
            return f"Could not fetch commits for {repo}"

        output = f"**📝 Recent Commits in {repo}:**\n"
        for c in commits:
            sha = c.get('sha', '')[:7]
            msg = c.get('commit', {}).get('message', 'No message').split('\n')[0][:50]
            author = c.get('commit', {}).get('author', {}).get('name', 'Unknown')
            date = c.get('commit', {}).get('author', {}).get('date', '')[:10]
            output += f"• `{sha}` {msg}... - {author} ({date})\n"
        return output

    # ============ PULL REQUESTS ============
    def get_pr_info(self, repo: str, pr_number: int) -> Dict:
        """Get PR information"""
        return self._request(f'/repos/{repo}/pulls/{pr_number}') or {"error": "Failed to fetch PR"}

    def get_open_prs(self, repo: str) -> str:
        """Get open PRs for a repo"""
        prs = self._request(f'/repos/{repo}/pulls?state=open')
        if not prs:
            return f"No open PRs in {repo} (or couldn't fetch)"

        output = f"**🔀 Open PRs in {repo}:**\n"
        for pr in prs[:10]:
            number = pr.get('number')
            title = pr.get('title', 'No title')[:50]
            author = pr.get('user', {}).get('login', 'Unknown')
            draft = "📝" if pr.get('draft') else "✅"
            output += f"• {draft} **#{number}** {title}... - by {author}\n"
        return output

    def get_pr_summary(self, repo: str, pr_number: int) -> str:
        """Get detailed PR summary"""
        pr = self.get_pr_info(repo, pr_number)
        if 'error' in pr:
            return pr['error']

        # Get files changed
        files = self._request(f'/repos/{repo}/pulls/{pr_number}/files') or []
        files_summary = ", ".join([f.get('filename', '').split('/')[-1] for f in files[:5]])
        if len(files) > 5:
            files_summary += f" +{len(files)-5} more"

        return f"""**🔀 PR #{pr_number}: {pr.get('title', 'No title')}**
• **Author:** {pr.get('user', {}).get('login', 'Unknown')}
• **Status:** {pr.get('state', 'unknown')} {'(DRAFT)' if pr.get('draft') else ''}
• **Branch:** {pr.get('head', {}).get('ref', '?')} → {pr.get('base', {}).get('ref', '?')}
• **Changed Files:** {len(files)} ({files_summary})
• **Additions:** +{pr.get('additions', 0)} / Deletions: -{pr.get('deletions', 0)}
• **Mergeable:** {pr.get('mergeable', 'unknown')}
• **Created:** {pr.get('created_at', '')[:10]}

**Description:**
{pr.get('body', 'No description')[:300]}{'...' if len(pr.get('body', '')) > 300 else ''}"""

    # ============ ISSUES ============
    def get_my_issues(self, limit: int = 10) -> str:
        """Get issues assigned to user"""
        issues = self._request(f'/issues?filter=assigned&state=open&per_page={limit}')
        if not issues:
            return "No open issues assigned to you"

        output = "**🐛 Your Open Issues:**\n"
        for issue in issues:
            repo = issue.get('repository', {}).get('full_name', 'Unknown')
            number = issue.get('number')
            title = issue.get('title', 'No title')[:45]
            labels = [l.get('name') for l in issue.get('labels', [])][:2]
            label_str = f" [{', '.join(labels)}]" if labels else ""
            output += f"• **{repo}#{number}** {title}...{label_str}\n"
        return output

    def get_repo_issues(self, repo: str, state: str = 'open') -> str:
        """Get issues for a repo"""
        issues = self._request(f'/repos/{repo}/issues?state={state}&per_page=10')
        if not issues:
            return f"No {state} issues in {repo}"

        output = f"**🐛 {state.title()} Issues in {repo}:**\n"
        for issue in issues:
            if issue.get('pull_request'):  # Skip PRs
                continue
            number = issue.get('number')
            title = issue.get('title', 'No title')[:45]
            author = issue.get('user', {}).get('login', 'Unknown')
            comments = issue.get('comments', 0)
            output += f"• **#{number}** {title}... - {author} 💬{comments}\n"
        return output

    def create_issue(self, repo: str, title: str, body: str = "") -> str:
        """Create a new issue"""
        result = self._post(f'/repos/{repo}/issues', {'title': title, 'body': body})
        if result:
            return f"✅ Created issue #{result.get('number')}: {result.get('html_url')}"
        return "Failed to create issue"

    # ============ ACTIVITY ============
    def get_my_activity(self, limit: int = 10) -> str:
        """Get user's recent activity"""
        # First get username
        user = self._request('/user')
        if not user:
            return "Could not fetch user info"

        username = user.get('login')
        events = self._request(f'/users/{username}/events?per_page={limit}')
        if not events:
            return "No recent activity"

        output = f"**📊 Recent Activity for {username}:**\n"
        for e in events:
            etype = e.get('type', 'Unknown').replace('Event', '')
            repo = e.get('repo', {}).get('name', 'Unknown')
            created = e.get('created_at', '')[:10]

            if etype == 'Push':
                commits = len(e.get('payload', {}).get('commits', []))
                output += f"• 📤 Pushed {commits} commits to {repo} ({created})\n"
            elif etype == 'PullRequest':
                action = e.get('payload', {}).get('action', 'updated')
                output += f"• 🔀 {action.title()} PR in {repo} ({created})\n"
            elif etype == 'Issues':
                action = e.get('payload', {}).get('action', 'updated')
                output += f"• 🐛 {action.title()} issue in {repo} ({created})\n"
            elif etype == 'Create':
                ref_type = e.get('payload', {}).get('ref_type', 'branch')
                output += f"• ✨ Created {ref_type} in {repo} ({created})\n"
            elif etype == 'Watch':
                output += f"• ⭐ Starred {repo} ({created})\n"
            else:
                output += f"• {etype} in {repo} ({created})\n"
        return output

    # ============ COMMAND PARSER ============
    def parse_command(self, text: str) -> Optional[str]:
        """Parse GitHub commands from natural language"""
        text_lower = text.lower().strip()

        # Notifications
        if any(p in text_lower for p in ['github notification', 'gh notification', 'my notifications']):
            return self.get_notifications_summary()

        # My repos
        if any(p in text_lower for p in ['my repo', 'my github repo', 'show my repo', 'list repo']):
            return self.get_repos_summary()

        # My issues
        if any(p in text_lower for p in ['my issue', 'assigned issue', 'my open issue']):
            return self.get_my_issues()

        # My activity
        if any(p in text_lower for p in ['my activity', 'my github activity', 'what did i do']):
            return self.get_my_activity()

        # Repo stats: "stats for owner/repo" or "repo stats owner/repo"
        match = re.search(r'(?:stats? for|repo stats?)\s+([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)', text_lower)
        if match:
            return self.get_repo_stats(match.group(1))

        # Recent commits: "commits in owner/repo"
        match = re.search(r'(?:commits? in|recent commits?)\s+([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)', text_lower)
        if match:
            return self.get_recent_commits(match.group(1))

        # Open PRs: "open prs in owner/repo"
        match = re.search(r'(?:open prs?|prs?) in\s+([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)', text_lower)
        if match:
            return self.get_open_prs(match.group(1))

        # PR summary: "pr #123 in owner/repo" or "summarize pr 123 owner/repo"
        match = re.search(r'pr\s*#?(\d+)\s+(?:in\s+)?([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)', text_lower)
        if match:
            return self.get_pr_summary(match.group(2), int(match.group(1)))

        # Issues in repo: "issues in owner/repo"
        match = re.search(r'issues?\s+in\s+([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)', text_lower)
        if match:
            return self.get_repo_issues(match.group(1))

        # Create issue: "create issue in owner/repo: title"
        match = re.search(r'create issue in\s+([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)[:\s]+(.+)', text, re.IGNORECASE)
        if match:
            return self.create_issue(match.group(1), match.group(2).strip())

        return None


# ============ IMAGE EDITING ============
class ImageEditor:
    """Image editing with img2img and inpainting"""

    @staticmethod
    def img2img(input_image_path: str, prompt: str, strength: float = 0.7) -> str:
        """Transform image based on prompt"""
        try:
            import torch
            from diffusers import AutoPipelineForImage2Image
            from PIL import Image
            import uuid

            output_path = f"/tmp/img2img_{uuid.uuid4().hex[:8]}.png"

            print("[IMG2IMG] Loading pipeline...")
            pipe = AutoPipelineForImage2Image.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                torch_dtype=torch.bfloat16,
            ).to("cuda")

            # Load and resize input image
            init_image = Image.open(input_image_path).convert("RGB")
            init_image = init_image.resize((768, 768))

            print(f"[IMG2IMG] Transforming with prompt: {prompt[:50]}...")
            image = pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=4,
                guidance_scale=0.0,
            ).images[0]

            image.save(output_path)

            # Cleanup
            del pipe
            torch.cuda.empty_cache()

            return output_path
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def variation(input_image_path: str, variation_strength: float = 0.5) -> str:
        """Create variation of image"""
        return ImageEditor.img2img(
            input_image_path,
            "same image with slight variations, high quality, detailed",
            strength=variation_strength
        )


# ============ FEATURE MANAGER ============
class AlexandraFeatures:
    """Central manager for all Alexandra features"""

    def __init__(self, memory_instance=None):
        self.wake_word = WakeWordDetector()
        self.code_executor = CodeExecutor()
        self.reminders = ReminderSystem()
        self.mood_detector = MoodDetector()
        self.personas = PersonaManager()
        self.spotify = SpotifyController()
        self.smart_home = SmartHomeController()
        self.email = EmailManager()
        self.github = GitHubManager()

        # Background updaters need memory instance
        self.news_updater = None
        self.epstein_updater = None
        if memory_instance:
            self.news_updater = NewsUpdater(memory_instance)
            self.epstein_updater = EpsteinUpdater(memory_instance)

    def start_background_services(self):
        """Start all background services"""
        if self.news_updater:
            self.news_updater.start()
        if self.epstein_updater:
            self.epstein_updater.start()

    def stop_background_services(self):
        """Stop all background services"""
        if self.news_updater:
            self.news_updater.stop()
        if self.epstein_updater:
            self.epstein_updater.stop()

    def process_special_commands(self, text: str) -> Optional[str]:
        """Process special commands, return response or None"""
        text_lower = text.lower().strip()

        # Reminder commands
        if text_lower.startswith('remind me') or text_lower.startswith('set reminder'):
            parsed = self.reminders.parse_reminder(text)
            if parsed:
                message, remind_at = parsed
                return self.reminders.add_reminder(message, remind_at)
            return "I couldn't understand that reminder. Try: 'Remind me in 30 minutes to take a break'"

        if text_lower == 'list reminders' or text_lower == 'show reminders':
            return self.reminders.list_reminders()

        # Code execution
        if text_lower.startswith('run code:') or text_lower.startswith('execute:'):
            code = text.split(':', 1)[1].strip()
            return f"**Code Output:**\n```\n{self.code_executor.execute(code)}\n```"

        # Persona switching
        if text_lower.startswith('switch to') and 'mode' in text_lower:
            for persona in self.personas.PERSONAS:
                if persona in text_lower:
                    return self.personas.set_persona(persona)

        if text_lower == 'list personas' or text_lower == 'show modes':
            return f"**Available Personas:**\n{self.personas.list_personas()}"

        # Email
        if 'check email' in text_lower or 'check my email' in text_lower:
            count = self.email.get_unread_count()
            if count >= 0:
                return f"You have {count} unread emails"
            return "Email not configured"

        # Epstein file search
        if any(p in text_lower for p in ['search epstein', 'epstein files', 'epstein news', 'new epstein', 'find epstein']):
            return EpsteinSearch.search_all()

        if 'epstein latest' in text_lower or 'latest epstein' in text_lower:
            return EpsteinSearch.get_latest_news()

        # Epstein updater status
        if 'epstein status' in text_lower or 'epstein updater' in text_lower:
            if self.epstein_updater:
                return self.epstein_updater.get_status()
            return "Epstein auto-updater not initialized"

        # GitHub - use the full command parser
        github_response = self.github.parse_command(text)
        if github_response:
            return github_response

        return None
