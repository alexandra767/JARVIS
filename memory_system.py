import chromadb
from chromadb.config import Settings
import datetime
import hashlib
import os

class AlexandraMemory:
    def __init__(self, db_path="~/ai-clone-chat/memory/db"):
        db_path = os.path.expanduser(db_path)
        os.makedirs(db_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Two collections: facts and conversations
        self.facts = self.client.get_or_create_collection(
            name="facts",
            metadata={"description": "Key facts about the user"}
        )
        self.conversations = self.client.get_or_create_collection(
            name="conversations", 
            metadata={"description": "Conversation history"}
        )
    
    def _make_id(self, text):
        return hashlib.md5(text.encode()).hexdigest()[:12]
    
    def save_exchange(self, user_msg, assistant_msg):
        """Save a conversation exchange"""
        timestamp = datetime.datetime.now().isoformat()
        
        # Save user message
        self.conversations.upsert(
            ids=[self._make_id(f"user_{timestamp}")],
            documents=[f"User said: {user_msg}"],
            metadatas=[{"role": "user", "timestamp": timestamp}]
        )
        
        # Save assistant response
        self.conversations.upsert(
            ids=[self._make_id(f"assistant_{timestamp}")],
            documents=[f"Alexandra said: {assistant_msg}"],
            metadatas=[{"role": "assistant", "timestamp": timestamp}]
        )
    
    def save_fact(self, fact):
        """Save a key fact about the user"""
        timestamp = datetime.datetime.now().isoformat()
        self.facts.upsert(
            ids=[self._make_id(fact)],
            documents=[fact],
            metadatas=[{"timestamp": timestamp}]
        )
    
    def recall(self, query, n_results=5):
        """Retrieve relevant memories for a query"""
        memories = []
        
        # Search facts
        if self.facts.count() > 0:
            fact_results = self.facts.query(
                query_texts=[query],
                n_results=min(n_results, self.facts.count())
            )
            for doc in fact_results['documents'][0]:
                memories.append(f"[Fact] {doc}")
        
        # Search conversations
        if self.conversations.count() > 0:
            conv_results = self.conversations.query(
                query_texts=[query],
                n_results=min(n_results, self.conversations.count())
            )
            for doc, meta in zip(conv_results['documents'][0], conv_results['metadatas'][0]):
                memories.append(f"[{meta.get('timestamp', 'past')}] {doc}")
        
        return memories
    
    def get_recent(self, n=10):
        """Get most recent conversation entries"""
        if self.conversations.count() == 0:
            return []
        
        results = self.conversations.get(
            limit=min(n, self.conversations.count()),
            include=["documents", "metadatas"]
        )
        
        # Sort by timestamp
        items = list(zip(results['documents'], results['metadatas']))
        items.sort(key=lambda x: x[1].get('timestamp', ''), reverse=True)
        
        return [doc for doc, _ in items]
    
    def stats(self):
        return {
            "facts": self.facts.count(),
            "conversations": self.conversations.count()
        }

# Fact extractor - uses the LLM to pull out key facts
EXTRACT_PROMPT = """Extract any key facts about the user from this exchange. 
Return ONLY a JSON list of facts, or empty list if none.
Examples of facts: name, job, pets, preferences, family, hobbies, important events.

User: {user_msg}
Assistant: {assistant_msg}

Return format: ["fact 1", "fact 2"] or []
"""

def extract_facts_from_exchange(user_msg, assistant_msg, llm_func):
    """Use LLM to extract facts from conversation"""
    try:
        prompt = EXTRACT_PROMPT.format(user_msg=user_msg, assistant_msg=assistant_msg)
        response = llm_func(prompt)
        
        # Parse JSON list from response
        import json
        import re
        
        # Find JSON list in response
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            facts = json.loads(match.group())
            return facts if isinstance(facts, list) else []
    except:
        pass
    return []
