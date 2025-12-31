"""
Enhanced RAG System with Web Search for Current Events
"""
import chromadb
from chromadb.config import Settings
import datetime
import hashlib
import os
import json
import requests
from typing import List, Optional

class EnhancedAlexandraMemory:
    def __init__(self, db_path="~/ai-clone-chat/memory/db"):
        db_path = os.path.expanduser(db_path)
        os.makedirs(db_path, exist_ok=True)

        self.client = chromadb.PersistentClient(path=db_path)

        # Collections
        self.facts = self.client.get_or_create_collection(name="facts")
        self.conversations = self.client.get_or_create_collection(name="conversations")
        self.knowledge = self.client.get_or_create_collection(name="knowledge")  # NEW: General knowledge
        self.news = self.client.get_or_create_collection(name="news")  # NEW: News articles

    def _make_id(self, text):
        return hashlib.md5(text.encode()).hexdigest()[:12]

    # ============ CONVERSATION MEMORY ============
    def save_exchange(self, user_msg, assistant_msg):
        """Save a conversation exchange"""
        timestamp = datetime.datetime.now().isoformat()

        self.conversations.upsert(
            ids=[self._make_id(f"user_{timestamp}")],
            documents=[f"User said: {user_msg}"],
            metadatas=[{"role": "user", "timestamp": timestamp}]
        )

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

    # ============ KNOWLEDGE BASE ============
    def add_knowledge(self, content: str, source: str = "manual", category: str = "general"):
        """Add knowledge to the database"""
        timestamp = datetime.datetime.now().isoformat()
        self.knowledge.upsert(
            ids=[self._make_id(content)],
            documents=[content],
            metadatas=[{"source": source, "category": category, "timestamp": timestamp}]
        )

    def add_news_article(self, title: str, content: str, source: str, url: str = ""):
        """Add a news article"""
        timestamp = datetime.datetime.now().isoformat()
        doc = f"{title}\n\n{content}"
        self.news.upsert(
            ids=[self._make_id(url or title)],
            documents=[doc],
            metadatas=[{"title": title, "source": source, "url": url, "timestamp": timestamp}]
        )

    # ============ WEB SEARCH (Current Events) ============
    def search_web_duckduckgo(self, query: str, max_results: int = 5) -> List[dict]:
        """Search DuckDuckGo for current information (FREE, no API key)"""
        try:
            # Using new ddgs package (renamed from duckduckgo_search)
            from ddgs import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return [{"title": r["title"], "snippet": r["body"], "url": r["href"]} for r in results]
        except ImportError:
            print("Install: pip install ddgs")
            return []
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def search_web_tavily(self, query: str, api_key: str, max_results: int = 5) -> List[dict]:
        """Search using Tavily API (better quality, needs API key)"""
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": True
                }
            )
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            print(f"Tavily error: {e}")
            return []

    # ============ NEWS FEEDS ============
    def fetch_news_rss(self, feed_url: str, max_items: int = 10):
        """Fetch and store news from RSS feed"""
        try:
            import feedparser
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:max_items]:
                self.add_news_article(
                    title=entry.get("title", ""),
                    content=entry.get("summary", entry.get("description", "")),
                    source=feed.feed.get("title", feed_url),
                    url=entry.get("link", "")
                )
            return len(feed.entries[:max_items])
        except ImportError:
            print("Install: pip install feedparser")
            return 0

    def update_news_feeds(self, max_per_feed: int = 15):
        """Update from common news RSS feeds"""
        feeds = [
            # Major News
            "http://feeds.bbci.co.uk/news/rss.xml",  # BBC
            "http://rss.cnn.com/rss/cnn_topstories.rss",  # CNN
            "https://feeds.npr.org/1001/rss.xml",  # NPR
            "http://feeds.reuters.com/reuters/topNews",  # Reuters
            "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",  # NY Times
            "https://feeds.washingtonpost.com/rss/national",  # Washington Post
            # Tech News
            "https://feeds.arstechnica.com/arstechnica/index",  # Ars Technica
            "https://www.wired.com/feed/rss",  # Wired
            "https://techcrunch.com/feed/",  # TechCrunch
            # Science
            "https://www.sciencedaily.com/rss/all.xml",  # Science Daily
            # Business
            "https://feeds.bloomberg.com/markets/news.rss",  # Bloomberg
        ]

        total = 0
        for feed in feeds:
            try:
                count = self.fetch_news_rss(feed, max_items=max_per_feed)
                total += count
                print(f"Fetched {count} from {feed}")
            except Exception as e:
                print(f"Failed to fetch {feed}: {e}")
        return total

    # ============ UNIFIED RECALL ============
    def recall(self, query: str, n_results: int = 5, include_web: bool = False) -> dict:
        """Retrieve all relevant context for a query"""
        results = {
            "facts": [],
            "conversations": [],
            "knowledge": [],
            "news": [],
            "web": []
        }

        # Search facts
        if self.facts.count() > 0:
            fact_results = self.facts.query(query_texts=[query], n_results=min(n_results, self.facts.count()))
            results["facts"] = fact_results['documents'][0]

        # Search conversations
        if self.conversations.count() > 0:
            conv_results = self.conversations.query(query_texts=[query], n_results=min(n_results, self.conversations.count()))
            results["conversations"] = conv_results['documents'][0]

        # Search knowledge base
        if self.knowledge.count() > 0:
            know_results = self.knowledge.query(query_texts=[query], n_results=min(n_results, self.knowledge.count()))
            results["knowledge"] = know_results['documents'][0]

        # Search news
        if self.news.count() > 0:
            news_results = self.news.query(query_texts=[query], n_results=min(n_results, self.news.count()))
            results["news"] = news_results['documents'][0]

        # Web search for current info
        if include_web:
            results["web"] = self.search_web_duckduckgo(query, max_results=3)

        return results

    def build_context(self, query: str, include_web: bool = True) -> str:
        """Build context string for LLM prompt"""
        recall = self.recall(query, n_results=3, include_web=include_web)

        context_parts = []

        if recall["facts"]:
            context_parts.append("**Known facts:**\n" + "\n".join(f"- {f}" for f in recall["facts"]))

        if recall["conversations"]:
            context_parts.append("**Relevant past conversations:**\n" + "\n".join(recall["conversations"][:3]))

        if recall["knowledge"]:
            context_parts.append("**Knowledge base:**\n" + "\n".join(recall["knowledge"][:3]))

        if recall["news"]:
            context_parts.append("**Recent news:**\n" + "\n".join(recall["news"][:3]))

        if recall["web"]:
            web_text = "\n".join([f"- {r['title']}: {r['snippet']}" for r in recall["web"]])
            context_parts.append(f"**Current web results:**\n{web_text}")

        return "\n\n".join(context_parts)

    def stats(self):
        return {
            "facts": self.facts.count(),
            "conversations": self.conversations.count(),
            "knowledge": self.knowledge.count(),
            "news": self.news.count()
        }


# ============ EXAMPLE USAGE ============
if __name__ == "__main__":
    memory = EnhancedAlexandraMemory()

    # Update news feeds
    print("Updating news feeds...")
    memory.update_news_feeds()

    # Test recall with web search
    print("\nTesting recall for 'latest technology news'...")
    context = memory.build_context("latest technology news", include_web=True)
    print(context)

    print("\nStats:", memory.stats())
