#!/usr/bin/env python3
"""
JARVIS Tools - Real-time Information Services
Weather, Traffic, Search, and more
"""

import asyncio
import aiohttp
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import json

logger = logging.getLogger("JarvisTools")


# ============================================================================
# CONFIGURATION - Add your API keys here or use environment variables
# ============================================================================

# OpenWeatherMap - Free tier: https://openweathermap.org/api
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

# Google Maps - For traffic: https://developers.google.com/maps
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# Default location (can be overridden)
DEFAULT_LOCATION = "Ridgway, PA"
DEFAULT_LAT = 51.5074
DEFAULT_LON = -0.1278


# ============================================================================
# WEATHER SERVICE
# ============================================================================

class WeatherService:
    """Real-time weather information using OpenWeatherMap API"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENWEATHER_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5"

    async def get_current_weather(self, location: str = None, lat: float = None, lon: float = None) -> Dict[str, Any]:
        """Get current weather for a location"""

        if not self.api_key:
            # Fallback to web scraping
            return await self._scrape_weather(location or DEFAULT_LOCATION)

        params = {
            "appid": self.api_key,
            "units": "metric"  # Use Celsius
        }

        if lat and lon:
            params["lat"] = lat
            params["lon"] = lon
        else:
            params["q"] = location or DEFAULT_LOCATION

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/weather",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_weather(data)
                    else:
                        logger.error(f"Weather API error: {response.status}")
                        return await self._scrape_weather(location or DEFAULT_LOCATION)

        except Exception as e:
            logger.error(f"Weather fetch error: {e}")
            return await self._scrape_weather(location or DEFAULT_LOCATION)

    async def get_forecast(self, location: str = None, days: int = 3) -> List[Dict[str, Any]]:
        """Get weather forecast from weather.gov (free, no API key needed)"""
        # Ridgway PA coordinates - could be made configurable
        lat, lon = 41.4259, -78.7286

        try:
            import aiohttp
            headers = {'User-Agent': 'JARVIS Weather App (contact: jarvis@local)'}

            async with aiohttp.ClientSession() as session:
                # Get grid endpoint for location
                point_url = f'https://api.weather.gov/points/{lat},{lon}'
                async with session.get(point_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status != 200:
                        return [{"error": f"Weather.gov API error: {r.status}"}]
                    point_data = await r.json()
                    forecast_url = point_data['properties']['forecast']

                # Get actual forecast
                async with session.get(forecast_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status != 200:
                        return [{"error": f"Forecast fetch error: {r.status}"}]
                    forecast_data = await r.json()

                # Parse periods into friendly format
                forecasts = []
                for period in forecast_data['properties']['periods'][:days * 2]:  # 2 periods per day
                    forecasts.append({
                        "name": period['name'],
                        "temperature": period['temperature'],
                        "unit": period['temperatureUnit'],
                        "description": period['shortForecast'],
                        "detailed": period['detailedForecast'],
                        "wind": f"{period['windSpeed']} {period['windDirection']}",
                        "icon": period.get('icon', '')
                    })
                return forecasts

        except Exception as e:
            logger.error(f"Weather.gov forecast error: {e}")
            return [{"error": str(e)}]

    async def get_forecast_old(self, location: str = None, days: int = 3) -> List[Dict[str, Any]]:
        """Get weather forecast (old OpenWeatherMap version)"""

        if not self.api_key:
            return [{"error": "No API key - forecast requires OpenWeatherMap API key"}]

        params = {
            "appid": self.api_key,
            "units": "metric",
            "q": location or DEFAULT_LOCATION,
            "cnt": days * 8  # 3-hour intervals
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/forecast",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_forecast(data)
                    else:
                        return [{"error": f"API error: {response.status}"}]

        except Exception as e:
            logger.error(f"Forecast fetch error: {e}")
            return [{"error": str(e)}]

    def _parse_weather(self, data: Dict) -> Dict[str, Any]:
        """Parse OpenWeatherMap response"""
        temp_c = data["main"]["temp"]
        feels_c = data["main"]["feels_like"]
        return {
            "location": data.get("name", "Unknown"),
            "country": data.get("sys", {}).get("country", ""),
            "temperature": round(temp_c, 1),
            "temperature_f": round(temp_c * 9/5 + 32, 1),
            "feels_like": round(feels_c, 1),
            "feels_like_f": round(feels_c * 9/5 + 32, 1),
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"].title(),
            "wind_speed": round(data["wind"]["speed"] * 2.237, 1),  # m/s to mph
            "wind_direction": data["wind"].get("deg", 0),
            "clouds": data["clouds"]["all"],
            "pressure": data["main"]["pressure"],
            "visibility": data.get("visibility", 10000) / 1000,  # meters to km
            "sunrise": datetime.fromtimestamp(data["sys"]["sunrise"]).strftime("%I:%M %p"),
            "sunset": datetime.fromtimestamp(data["sys"]["sunset"]).strftime("%I:%M %p"),
            "timestamp": datetime.now().isoformat()
        }

    def _parse_forecast(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse forecast response"""
        forecasts = []
        for item in data.get("list", []):
            forecasts.append({
                "datetime": item["dt_txt"],
                "temperature": round(item["main"]["temp"], 1),
                "description": item["weather"][0]["description"].title(),
                "humidity": item["main"]["humidity"],
                "wind_speed": round(item["wind"]["speed"] * 3.6, 1)
            })
        return forecasts

    async def _scrape_weather(self, location: str) -> Dict[str, Any]:
        """Fallback: Scrape weather from wttr.in (no API key needed)"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://wttr.in/{location}?format=j1",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        current = data["current_condition"][0]
                        temp_c = int(current["temp_C"])
                        feels_c = int(current["FeelsLikeC"])
                        return {
                            "location": location,
                            "temperature": temp_c,
                            "temperature_f": int(current["temp_F"]),
                            "feels_like": feels_c,
                            "feels_like_f": int(current["FeelsLikeF"]),
                            "humidity": int(current["humidity"]),
                            "description": current["weatherDesc"][0]["value"],
                            "wind_speed": int(int(current["windspeedKmph"]) * 0.621),  # km/h to mph
                            "visibility": int(current["visibility"]),
                            "source": "wttr.in",
                            "timestamp": datetime.now().isoformat()
                        }
        except Exception as e:
            logger.error(f"Weather scrape error: {e}")

        return {"error": "Could not fetch weather", "location": location}

    def format_weather(self, weather: Dict) -> str:
        """Format weather data for voice response"""
        if "error" in weather:
            return f"Sorry, I couldn't get the weather: {weather['error']}"

        temp = weather.get('temperature_f', weather.get('temperature', 'N/A'))
        feels = weather.get('feels_like_f', weather.get('feels_like', 'N/A'))
        return (
            f"The current weather in {weather['location']} is {weather['description']}. "
            f"Temperature is {temp} degrees, feels like {feels}. "
            f"Humidity is {weather['humidity']}% with wind at {weather['wind_speed']} mph."
        )


# ============================================================================
# TRAFFIC SERVICE
# ============================================================================

class TrafficService:
    """Real-time traffic information"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or GOOGLE_MAPS_API_KEY

    async def get_traffic(self, origin: str, destination: str) -> Dict[str, Any]:
        """Get traffic/route information between two points"""

        if self.api_key:
            return await self._get_google_traffic(origin, destination)
        else:
            # Fallback to web search
            return await self._search_traffic(origin, destination)

    async def _get_google_traffic(self, origin: str, destination: str) -> Dict[str, Any]:
        """Get traffic from Google Maps API"""
        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            "origin": origin,
            "destination": destination,
            "departure_time": "now",
            "traffic_model": "best_guess",
            "key": self.api_key
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data["status"] == "OK":
                            route = data["routes"][0]["legs"][0]
                            return {
                                "origin": route["start_address"],
                                "destination": route["end_address"],
                                "distance": route["distance"]["text"],
                                "duration": route["duration"]["text"],
                                "duration_in_traffic": route.get("duration_in_traffic", {}).get("text", route["duration"]["text"]),
                                "summary": data["routes"][0].get("summary", ""),
                                "timestamp": datetime.now().isoformat()
                            }
        except Exception as e:
            logger.error(f"Traffic API error: {e}")

        return await self._search_traffic(origin, destination)

    async def _search_traffic(self, origin: str, destination: str) -> Dict[str, Any]:
        """Fallback: Search for traffic info"""
        return {
            "origin": origin,
            "destination": destination,
            "note": "For real-time traffic, add a Google Maps API key",
            "suggestion": f"Search Google Maps for directions from {origin} to {destination}",
            "timestamp": datetime.now().isoformat()
        }

    def format_traffic(self, traffic: Dict) -> str:
        """Format traffic data for voice response"""
        if "note" in traffic:
            return f"I don't have real-time traffic access. You can check Google Maps for the route from {traffic['origin']} to {traffic['destination']}."

        return (
            f"The route from {traffic['origin']} to {traffic['destination']} "
            f"is {traffic['distance']}. "
            f"With current traffic, it will take approximately {traffic['duration_in_traffic']}."
        )


# ============================================================================
# WEB SEARCH SERVICE - Enhanced with duckduckgo_search library
# ============================================================================

class WebSearchService:
    """Web search using DuckDuckGo (no API key needed)"""

    def __init__(self):
        self._ddgs = None
        self._ddgs_available = False
        try:
            from ddgs import DDGS
            self._ddgs_available = True
            logger.info("DuckDuckGo Search library available - enhanced search enabled")
        except ImportError:
            logger.warning("duckduckgo_search not installed - using basic search. Install with: pip install duckduckgo_search")

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web using DuckDuckGo"""
        # Try enhanced search first
        if self._ddgs_available:
            try:
                results = await self._enhanced_search(query, num_results)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Enhanced search failed: {e}, falling back to basic")

        # Fallback to basic API
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": 1,
                        "skip_disambig": 1
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []

                        # Abstract (main answer)
                        if data.get("Abstract"):
                            results.append({
                                "title": data.get("Heading", "Answer"),
                                "snippet": data["Abstract"],
                                "url": data.get("AbstractURL", ""),
                                "source": data.get("AbstractSource", "DuckDuckGo")
                            })

                        # Related topics
                        for topic in data.get("RelatedTopics", [])[:num_results]:
                            if isinstance(topic, dict) and "Text" in topic:
                                results.append({
                                    "title": topic.get("Text", "")[:50],
                                    "snippet": topic.get("Text", ""),
                                    "url": topic.get("FirstURL", "")
                                })

                        return results if results else await self._fallback_search(query)

        except Exception as e:
            logger.error(f"Search error: {e}")

        return await self._fallback_search(query)

    async def _enhanced_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Enhanced search using duckduckgo_search library - real search results with sources"""
        from ddgs import DDGS
        import asyncio

        def sync_search():
            results = []
            try:
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=num_results):
                        results.append({
                            "title": r.get("title", ""),
                            "snippet": r.get("body", ""),
                            "url": r.get("href", ""),
                            "source": "DuckDuckGo"
                        })
            except Exception as e:
                logger.error(f"DDGS search error: {e}")
            return results

        # Run sync search in thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, sync_search)
        return results

    async def search_news(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search news specifically"""
        if not self._ddgs_available:
            return await self.search(f"{query} news", num_results)

        from ddgs import DDGS
        import asyncio

        def sync_news_search():
            results = []
            try:
                with DDGS() as ddgs:
                    for r in ddgs.news(query, max_results=num_results):
                        results.append({
                            "title": r.get("title", ""),
                            "snippet": r.get("body", ""),
                            "url": r.get("url", ""),
                            "date": r.get("date", ""),
                            "source": r.get("source", "News")
                        })
            except Exception as e:
                logger.error(f"News search error: {e}")
            return results

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_news_search)

    async def _fallback_search(self, query: str) -> List[Dict[str, Any]]:
        """Fallback search using DuckDuckGo HTML (scraping)"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        html = await response.text()
                        # Basic parsing - extract result links
                        import re
                        results = []
                        # Find result snippets
                        snippets = re.findall(r'class="result__snippet"[^>]*>([^<]+)<', html)
                        titles = re.findall(r'class="result__a"[^>]*>([^<]+)<', html)
                        urls = re.findall(r'class="result__url"[^>]*href="([^"]+)"', html)

                        for i in range(min(5, len(snippets))):
                            results.append({
                                "title": titles[i] if i < len(titles) else "Result",
                                "snippet": snippets[i] if i < len(snippets) else "",
                                "url": urls[i] if i < len(urls) else f"https://duckduckgo.com/?q={query}"
                            })

                        if results:
                            return results

                        return [{
                            "title": "Search Results",
                            "snippet": f"Found results for: {query}. Use browser agent for detailed search.",
                            "url": f"https://duckduckgo.com/?q={query}"
                        }]
        except Exception as e:
            logger.error(f"Fallback search error: {e}")

        return [{"error": "Search failed", "query": query}]

    async def get_answer(self, question: str) -> str:
        """Get a direct answer to a question"""
        results = await self.search(question)
        if results and not results[0].get("error"):
            # Format results nicely
            answer_parts = []
            for i, r in enumerate(results[:3], 1):
                if r.get("snippet"):
                    answer_parts.append(f"{i}. {r['title']}: {r['snippet'][:200]}...")
                    if r.get("url"):
                        answer_parts.append(f"   Source: {r['url']}")
            if answer_parts:
                return "\n".join(answer_parts)
        return f"I couldn't find a direct answer. Try asking me to search the web for: {question}"

    async def fetch_article(self, url: str) -> Dict[str, Any]:
        """Fetch and extract main content from an article URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    },
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        html = await response.text()

                        # Try to extract article content
                        import re

                        # Remove scripts and styles
                        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
                        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
                        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

                        # Extract title
                        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
                        title = title_match.group(1).strip() if title_match else "Article"

                        # Try to find article content in common containers
                        content = ""

                        # Look for article, main, or content divs
                        for pattern in [
                            r'<article[^>]*>(.*?)</article>',
                            r'<main[^>]*>(.*?)</main>',
                            r'class="[^"]*article[^"]*"[^>]*>(.*?)</div>',
                            r'class="[^"]*content[^"]*"[^>]*>(.*?)</div>',
                            r'class="[^"]*post[^"]*"[^>]*>(.*?)</div>',
                        ]:
                            match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
                            if match:
                                content = match.group(1)
                                break

                        # If no article container found, use body
                        if not content:
                            body_match = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL | re.IGNORECASE)
                            if body_match:
                                content = body_match.group(1)

                        # Extract paragraphs
                        paragraphs = re.findall(r'<p[^>]*>([^<]+(?:<[^/][^>]*>[^<]*</[^>]+>[^<]*)*)</p>', content, re.IGNORECASE)

                        # Clean up text
                        text_parts = []
                        for p in paragraphs:
                            # Remove remaining HTML tags
                            clean = re.sub(r'<[^>]+>', ' ', p)
                            clean = re.sub(r'\s+', ' ', clean).strip()
                            if len(clean) > 50:  # Skip very short paragraphs
                                text_parts.append(clean)

                        article_text = "\n\n".join(text_parts[:20])  # Limit to first 20 paragraphs

                        if article_text:
                            return {
                                "success": True,
                                "title": title,
                                "content": article_text[:8000],  # Limit content length
                                "url": url
                            }
                        else:
                            return {
                                "success": False,
                                "error": "Could not extract article content",
                                "url": url
                            }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}",
                            "url": url
                        }
        except Exception as e:
            logger.error(f"Article fetch error: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }

    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for display"""
        if not results or results[0].get("error"):
            return "No results found."

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Result")
            snippet = r.get("snippet", "")[:200]
            url = r.get("url", "")
            formatted.append(f"**{i}. {title}**\n{snippet}\n[{url}]({url})\n")

        return "\n".join(formatted)


# ============================================================================
# NEWS SERVICE
# ============================================================================

class NewsService:
    """Get latest news headlines"""

    async def get_headlines(self, topic: str = None, country: str = "us") -> List[Dict[str, Any]]:
        """Get news headlines"""
        try:
            # Using Google News RSS (no API key needed)
            url = "https://news.google.com/rss"
            if topic:
                url = f"https://news.google.com/rss/search?q={topic}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        # Parse RSS (simplified)
                        text = await response.text()
                        headlines = []

                        # Basic XML parsing
                        import re
                        items = re.findall(r'<item>.*?</item>', text, re.DOTALL)

                        for item in items[:5]:
                            title_match = re.search(r'<title>(.*?)</title>', item)
                            link_match = re.search(r'<link>(.*?)</link>', item)
                            pub_match = re.search(r'<pubDate>(.*?)</pubDate>', item)

                            if title_match:
                                headlines.append({
                                    "title": title_match.group(1).replace('<![CDATA[', '').replace(']]>', ''),
                                    "url": link_match.group(1) if link_match else "",
                                    "published": pub_match.group(1) if pub_match else ""
                                })

                        return headlines

        except Exception as e:
            logger.error(f"News fetch error: {e}")

        return [{"error": "Could not fetch news"}]

    def format_headlines(self, headlines: List[Dict]) -> str:
        """Format headlines for voice response"""
        if not headlines or "error" in headlines[0]:
            return "Sorry, I couldn't fetch the latest news."

        response = "Here are the top headlines: "
        for i, h in enumerate(headlines[:3], 1):
            response += f"{i}. {h['title']}. "
        return response


# ============================================================================
# TIME & DATE SERVICE
# ============================================================================

class TimeService:
    """Time and date information"""

    async def get_time(self, timezone: str = None) -> Dict[str, Any]:
        """Get current time"""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        try:
            # Default to Eastern Time
            if timezone:
                tz = ZoneInfo(timezone)
            else:
                tz = ZoneInfo("America/New_York")

            now = datetime.now(tz)

            return {
                "time": now.strftime("%I:%M %p").lstrip("0"),
                "date": now.strftime("%A, %B %d, %Y"),
                "timezone": timezone or "Eastern",
                "timestamp": now.isoformat()
            }
        except Exception as e:
            now = datetime.now()
            return {
                "time": now.strftime("%I:%M %p"),
                "date": now.strftime("%A, %B %d, %Y"),
                "timezone": "local",
                "error": str(e)
            }

    def format_time(self, time_data: Dict) -> str:
        """Format time for voice response"""
        return f"It's currently {time_data['time']} on {time_data['date']}."


# ============================================================================
# UNIFIED TOOLS INTERFACE
# ============================================================================

class JarvisTools:
    """
    Unified interface for all Jarvis tools.
    Call these from the main Jarvis system.
    """

    def __init__(self):
        self.weather = WeatherService()
        self.traffic = TrafficService()
        self.search = WebSearchService()
        self.news = NewsService()
        self.time = TimeService()

    async def handle_query(self, query: str) -> str:
        """
        Automatically detect query type and return appropriate response.
        """
        query_lower = query.lower()

        # Weather queries
        if any(kw in query_lower for kw in ["weather", "temperature", "forecast", "rain", "sunny", "cold", "hot"]):
            # Extract location if mentioned
            location = self._extract_location(query) or DEFAULT_LOCATION

            # Check if asking for future weather (tomorrow, next week, specific date, etc.)
            future_keywords = ['tomorrow', 'next', 'forecast', 'week', 'weekend', 'monday', 'tuesday',
                             'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'later', 'tonight']
            import re
            has_future_date = re.search(r'\d{1,2}(?:st|nd|rd|th)?', query_lower) or \
                            any(kw in query_lower for kw in future_keywords)

            if has_future_date:
                # Use weather.gov forecast
                forecasts = await self.weather.get_forecast(location, days=3)
                if forecasts and not forecasts[0].get('error'):
                    # Format forecast nicely
                    result = f"Weather forecast for {location}:\n"
                    for f in forecasts[:4]:  # Show next 4 periods
                        result += f"\n{f['name']}: {f['temperature']}°{f['unit']}\n"
                        result += f"  {f['description']}\n"
                        result += f"  Wind: {f['wind']}\n"
                    return result
                # Fall through to current weather if forecast fails

            # Default to current weather
            weather = await self.weather.get_current_weather(location)
            return self.weather.format_weather(weather)

        # Traffic queries
        if any(kw in query_lower for kw in ["traffic", "commute", "drive to", "route to", "how long to"]):
            # This would need origin/destination parsing
            return "For traffic information, please tell me where you're going from and to."

        # News queries
        if any(kw in query_lower for kw in ["news", "headlines", "what's happening"]):
            topic = self._extract_topic(query)
            headlines = await self.news.get_headlines(topic)
            return self.news.format_headlines(headlines)

        # Time queries
        if any(kw in query_lower for kw in ["time", "date", "what day"]):
            time_data = await self.time.get_time()
            return self.time.format_time(time_data)

        # Skip personal questions - these should use memory, not web search
        is_personal = 'my ' in query_lower and any(word in query_lower for word in [
            'aunt', 'uncle', 'mother', 'father', 'mom', 'dad', 'sister', 'brother',
            'son', 'daughter', 'name', 'birthday', 'favorite', 'wife', 'husband',
            'friend', 'boss', 'pet', 'dog', 'cat', 'car', 'address', 'phone'
        ])

        if is_personal:
            return None  # Let the LLM handle with memory context

        # General search
        if any(kw in query_lower for kw in ["search", "look up", "find", "what is", "who is", "how to"]):
            answer = await self.search.get_answer(query)
            return answer

        # Default: return None to let LLM handle
        return None

    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location from query"""
        # Simple extraction - could be improved with NER
        location_indicators = ["in", "at", "for"]
        words = query.split()

        for i, word in enumerate(words):
            if word.lower() in location_indicators and i + 1 < len(words):
                return " ".join(words[i+1:]).strip("?.,!")

        return None

    def _extract_topic(self, query: str) -> Optional[str]:
        """Extract topic from news query"""
        # Remove common news-related words
        stopwords = ["news", "headlines", "about", "on", "regarding", "what's", "happening", "the", "latest"]
        words = query.lower().split()
        topic_words = [w for w in words if w not in stopwords]

        if topic_words:
            return " ".join(topic_words)
        return None


# ============================================================================
# FILE SYSTEM SERVICE - Create, List, Edit, Delete files and folders
# ============================================================================

class FileSystemService:
    """File and folder operations"""

    def __init__(self, workspace: str = None):
        self.workspace = workspace or os.path.expanduser("~/jarvis_workspace")
        os.makedirs(self.workspace, exist_ok=True)

    def _safe_path(self, path: str) -> str:
        """Ensure path is within workspace for safety"""
        if os.path.isabs(path):
            return path
        return os.path.join(self.workspace, path)

    def create_folder(self, folder_path: str) -> Dict[str, Any]:
        """Create a new folder"""
        try:
            full_path = self._safe_path(folder_path)
            os.makedirs(full_path, exist_ok=True)
            return {"success": True, "message": f"Created folder: {folder_path}", "path": full_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_file(self, file_path: str, content: str = "") -> Dict[str, Any]:
        """Create a new file with optional content"""
        try:
            full_path = self._safe_path(file_path)
            # Create parent directories if needed
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
            return {"success": True, "message": f"Created file: {file_path}", "path": full_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_files(self, folder_path: str = ".") -> Dict[str, Any]:
        """List files and folders in a directory"""
        try:
            full_path = self._safe_path(folder_path)
            if not os.path.exists(full_path):
                return {"success": False, "error": f"Path does not exist: {folder_path}"}

            items = []
            for item in os.listdir(full_path):
                item_path = os.path.join(full_path, item)
                items.append({
                    "name": item,
                    "type": "folder" if os.path.isdir(item_path) else "file",
                    "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None
                })
            return {"success": True, "path": folder_path, "items": items}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def read_file(self, file_path: str) -> Dict[str, Any]:
        """Read contents of a file"""
        try:
            full_path = self._safe_path(file_path)
            if not os.path.exists(full_path):
                return {"success": False, "error": f"File does not exist: {file_path}"}
            with open(full_path, 'r') as f:
                content = f.read()
            return {"success": True, "path": file_path, "content": content}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def edit_file(self, file_path: str, content: str, append: bool = False) -> Dict[str, Any]:
        """Edit/write content to a file"""
        try:
            full_path = self._safe_path(file_path)
            mode = 'a' if append else 'w'
            with open(full_path, mode) as f:
                f.write(content)
            action = "Appended to" if append else "Updated"
            return {"success": True, "message": f"{action} file: {file_path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_file(self, file_path: str) -> Dict[str, Any]:
        """Delete a file"""
        try:
            full_path = self._safe_path(file_path)
            if os.path.isfile(full_path):
                os.remove(full_path)
                return {"success": True, "message": f"Deleted file: {file_path}"}
            elif os.path.isdir(full_path):
                import shutil
                shutil.rmtree(full_path)
                return {"success": True, "message": f"Deleted folder: {file_path}"}
            else:
                return {"success": False, "error": f"Path does not exist: {file_path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# HOST CONTROL SERVICE - Control host machine from Docker
# ============================================================================

class HostControlService:
    """Control host machine via HTTP bridge (for Docker containers)"""

    def __init__(self, host_url: str = None):
        # Default to Docker host IP
        self.host_url = host_url or os.environ.get("HOST_CONTROL_URL", "http://172.17.0.1:7899")

    def send_command(self, command: str, url: str = "") -> Dict[str, Any]:
        """Send a command to the host control bridge"""
        import requests

        try:
            response = requests.post(
                self.host_url,
                json={"command": command, "url": url},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": f"Host returned {response.status_code}"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Host control bridge not running. Start ada_host_control.py on the host machine."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def open_chrome(self, url: str = "") -> Dict[str, Any]:
        """Open Chrome browser, optionally with a URL"""
        return self.send_command("open_chrome", url)

    def open_firefox(self, url: str = "") -> Dict[str, Any]:
        """Open Firefox browser, optionally with a URL"""
        return self.send_command("open_firefox", url)

    def open_url(self, url: str) -> Dict[str, Any]:
        """Open URL in default browser"""
        return self.send_command("open_url", url)

    def open_terminal(self) -> Dict[str, Any]:
        """Open terminal"""
        return self.send_command("open_terminal")

    def open_files(self) -> Dict[str, Any]:
        """Open file manager"""
        return self.send_command("open_files")


# ============================================================================
# APPLICATION SERVICE - Open applications
# ============================================================================

class ApplicationService:
    """Open and manage applications"""

    def __init__(self):
        import platform
        self.system = platform.system().lower()
        # Try host control first (for Docker)
        self.host_control = HostControlService()

    def open_application(self, app_name: str, url: str = "") -> Dict[str, Any]:
        """Open an application by name"""
        import subprocess

        app_lower = app_name.lower()

        # Try host control bridge first (works from Docker)
        if any(x in app_lower for x in ["chrome", "firefox", "browser", "terminal", "files"]):
            result = self.host_control.send_command(f"open_{app_lower.replace(' ', '_')}", url)
            if result.get("status") == "ok" or result.get("success"):
                return {"success": True, "message": result.get("result", f"Opened {app_name}")}
            # If bridge not running, fall through to local attempt

        # Map common app names to commands
        app_map_linux = {
            "notepad": "gedit",
            "text editor": "gedit",
            "terminal": "gnome-terminal",
            "file manager": "nautilus",
            "files": "nautilus",
            "calculator": "gnome-calculator",
            "chrome": "google-chrome",
            "google chrome": "google-chrome",
            "firefox": "firefox",
            "browser": "firefox",
            "vscode": "code",
            "visual studio code": "code",
        }

        app_map_mac = {
            "notepad": "TextEdit",
            "text editor": "TextEdit",
            "terminal": "Terminal",
            "file manager": "Finder",
            "files": "Finder",
            "calculator": "Calculator",
            "chrome": "Google Chrome",
            "google chrome": "Google Chrome",
            "firefox": "Firefox",
            "safari": "Safari",
            "browser": "Safari",
            "vscode": "Visual Studio Code",
            "visual studio code": "Visual Studio Code",
        }

        try:
            if self.system == "linux":
                cmd = app_map_linux.get(app_lower, app_name)
                subprocess.Popen([cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif self.system == "darwin":  # macOS
                cmd = app_map_mac.get(app_lower, app_name)
                subprocess.Popen(["open", "-a", cmd])
            elif self.system == "windows":
                subprocess.Popen(["start", app_name], shell=True)
            else:
                return {"success": False, "error": f"Unsupported system: {self.system}"}

            return {"success": True, "message": f"Opened {app_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# BROWSER SERVICE - Open URLs and websites
# ============================================================================

class BrowserService:
    """Open URLs in web browser"""

    def open_url(self, url: str) -> Dict[str, Any]:
        """Open a URL in the default browser"""
        import webbrowser

        try:
            # Add https:// if no protocol specified
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            webbrowser.open(url)
            return {"success": True, "message": f"Opened {url}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_google(self, query: str) -> Dict[str, Any]:
        """Search the web for a query using DuckDuckGo"""
        try:
            from ddgs import DDGS

            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=5))

            if results:
                # Format results for display
                formatted = []
                for r in results[:5]:
                    title = r.get('title', 'No title')
                    body = r.get('body', '')[:200]
                    url = r.get('href', '')
                    formatted.append(f"• {title}\n  {body}\n  Source: {url}")

                return {
                    "success": True,
                    "message": f"Found {len(results)} results for '{query}':\n\n" + "\n\n".join(formatted),
                    "results": results
                }
            else:
                return {"success": False, "message": f"No results found for: {query}"}
        except Exception as e:
            return {"success": False, "error": f"Search failed: {str(e)}"}


# ============================================================================
# SCREEN CAPTURE SERVICE - Capture screen for AI vision
# ============================================================================

class ScreenCaptureService:
    """Capture screen for AI to analyze"""

    def capture_screen(self) -> Dict[str, Any]:
        """Capture the entire screen"""
        try:
            from PIL import ImageGrab
            import numpy as np
            import tempfile
            import time

            screenshot = ImageGrab.grab()
            # Convert to numpy array for vision model
            screen_array = np.array(screenshot)

            # Also save to temp file
            temp_path = os.path.join(tempfile.gettempdir(), f"jarvis_screen_{int(time.time())}.png")
            screenshot.save(temp_path)

            return {
                "success": True,
                "image": screen_array,
                "path": temp_path,
                "size": screenshot.size
            }
        except ImportError:
            # Try alternative method for Linux without display
            try:
                import subprocess
                import tempfile
                import time

                temp_path = os.path.join(tempfile.gettempdir(), f"jarvis_screen_{int(time.time())}.png")
                subprocess.run(["scrot", temp_path], check=True)

                from PIL import Image
                import numpy as np
                img = Image.open(temp_path)
                screen_array = np.array(img)

                return {
                    "success": True,
                    "image": screen_array,
                    "path": temp_path,
                    "size": img.size
                }
            except Exception as e:
                return {"success": False, "error": f"Screen capture not available: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def capture_window(self, window_title: str = None) -> Dict[str, Any]:
        """Capture a specific window"""
        # For now, just capture full screen
        return self.capture_screen()


# ============================================================================
# CODE EXECUTION SERVICE - Run Python code
# ============================================================================

class CodeExecutionService:
    """Execute Python code safely"""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def execute_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code and return the result"""
        import subprocess
        import tempfile
        import os

        try:
            # Create temp file with the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Run the code with timeout
                result = subprocess.run(
                    ['python3', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )

                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                }
            finally:
                # Clean up temp file
                os.unlink(temp_file)

        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Code execution timed out after {self.timeout} seconds"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def evaluate_expression(self, expression: str) -> Dict[str, Any]:
        """Safely evaluate a mathematical expression"""
        import ast
        import operator

        # Safe operators
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.Mod: operator.mod,
        }

        def eval_node(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                return operators[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                return operators[type(node.op)](operand)
            else:
                raise ValueError(f"Unsupported operation: {type(node)}")

        try:
            tree = ast.parse(expression, mode='eval')
            result = eval_node(tree.body)
            return {"success": True, "expression": expression, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# ENHANCED JARVIS TOOLS - Combined service with all tools
# ============================================================================

class EnhancedJarvisTools(JarvisTools):
    """Extended JarvisTools with ADA-style capabilities"""

    def __init__(self):
        super().__init__()
        self.filesystem = FileSystemService()
        self.apps = ApplicationService()
        self.browser = BrowserService()
        self.screen = ScreenCaptureService()
        self.code = CodeExecutionService()

    async def handle_command(self, command: str, **kwargs) -> str:
        """Handle file/app/browser commands"""
        command_lower = command.lower()

        # File operations
        if "create folder" in command_lower or "make folder" in command_lower or "new folder" in command_lower:
            folder_name = kwargs.get("folder_name") or self._extract_name(command, ["folder", "directory"])
            if folder_name:
                result = self.filesystem.create_folder(folder_name)
                return result.get("message") or result.get("error")
            return "What should I name the folder?"

        if "create file" in command_lower or "make file" in command_lower or "new file" in command_lower:
            file_name = kwargs.get("file_name") or self._extract_name(command, ["file"])
            content = kwargs.get("content", "")
            if file_name:
                result = self.filesystem.create_file(file_name, content)
                return result.get("message") or result.get("error")
            return "What should I name the file?"

        if "list files" in command_lower or "show files" in command_lower or "what files" in command_lower:
            folder = kwargs.get("folder", ".")
            result = self.filesystem.list_files(folder)
            if result.get("success"):
                items = result.get("items", [])
                if not items:
                    return f"The folder is empty."
                item_list = ", ".join([f"{i['name']} ({i['type']})" for i in items])
                return f"Found: {item_list}"
            return result.get("error")

        if "read file" in command_lower or "show file" in command_lower or "open file" in command_lower:
            file_name = kwargs.get("file_name") or self._extract_name(command, ["file"])
            if file_name:
                result = self.filesystem.read_file(file_name)
                if result.get("success"):
                    return f"Contents of {file_name}:\n{result.get('content')}"
                return result.get("error")

        if "edit file" in command_lower or "write to" in command_lower or "add to" in command_lower:
            file_name = kwargs.get("file_name") or self._extract_name(command, ["file"])
            content = kwargs.get("content", "")
            append = "append" in command_lower or "add" in command_lower
            if file_name and content:
                result = self.filesystem.edit_file(file_name, content, append)
                return result.get("message") or result.get("error")

        if "delete file" in command_lower or "remove file" in command_lower:
            file_name = kwargs.get("file_name") or self._extract_name(command, ["file", "folder"])
            if file_name:
                result = self.filesystem.delete_file(file_name)
                return result.get("message") or result.get("error")

        # Application operations
        if "open" in command_lower and any(app in command_lower for app in ["notepad", "chrome", "firefox", "terminal", "vscode", "calculator", "browser"]):
            app_name = self._extract_app_name(command)
            if app_name:
                result = self.apps.open_application(app_name)
                return result.get("message") or result.get("error")

        # Browser operations
        if ("open" in command_lower or "go to" in command_lower) and ".com" in command_lower or ".org" in command_lower or "http" in command_lower:
            url = self._extract_url(command)
            if url:
                result = self.browser.open_url(url)
                return result.get("message") or result.get("error")

        if "search google" in command_lower or "google search" in command_lower:
            query = kwargs.get("query") or command.split("search")[-1].strip()
            if query:
                result = self.browser.search_google(query)
                return result.get("message") or result.get("error")

        # Code execution
        if "calculate" in command_lower or "solve" in command_lower or "what is" in command_lower:
            # Try to extract math expression
            expression = self._extract_math(command)
            if expression:
                result = self.code.evaluate_expression(expression)
                if result.get("success"):
                    return f"The answer is: {result.get('result')}"
                return result.get("error")

        if "run code" in command_lower or "execute code" in command_lower:
            code = kwargs.get("code", "")
            if code:
                result = self.code.execute_python(code)
                if result.get("success"):
                    return f"Output:\n{result.get('stdout')}"
                return f"Error:\n{result.get('stderr') or result.get('error')}"

        # Fall back to base handler
        return await self.handle_query(command)

    def _extract_name(self, text: str, keywords: List[str]) -> Optional[str]:
        """Extract name after keywords like 'folder', 'file', etc."""
        text_lower = text.lower()
        for kw in keywords:
            if kw in text_lower:
                # Find text after "called X" or "named X"
                for marker in ["called ", "named ", f"{kw} "]:
                    if marker in text_lower:
                        idx = text_lower.find(marker) + len(marker)
                        name = text[idx:].split()[0].strip('.,!?')
                        return name
        return None

    def _extract_app_name(self, text: str) -> Optional[str]:
        """Extract application name from command"""
        apps = ["notepad", "chrome", "google chrome", "firefox", "terminal", "vscode", "visual studio code", "calculator", "browser", "file manager"]
        text_lower = text.lower()
        for app in apps:
            if app in text_lower:
                return app
        return None

    def _extract_url(self, text: str) -> Optional[str]:
        """Extract URL from text"""
        import re
        # Match URLs or domain names
        url_pattern = r'(https?://[^\s]+|[a-zA-Z0-9][a-zA-Z0-9-]*\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?(?:/[^\s]*)?)'
        match = re.search(url_pattern, text)
        if match:
            return match.group(1)
        return None

    def _extract_math(self, text: str) -> Optional[str]:
        """Extract mathematical expression from text"""
        import re
        # Remove common words
        text = text.lower().replace("what is", "").replace("calculate", "").replace("solve", "")
        # Find math-like patterns
        math_pattern = r'[\d\s\+\-\*\/\^\(\)\.]+'
        match = re.search(math_pattern, text)
        if match:
            expr = match.group().strip()
            if expr and any(c.isdigit() for c in expr):
                return expr
        return None


# ============================================================================
# TOOL DEFINITIONS FOR LLM
# ============================================================================

JARVIS_TOOL_DEFINITIONS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location (e.g., 'London, UK')"
                }
            }
        }
    },
    {
        "name": "get_traffic",
        "description": "Get traffic/route information between two locations",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Starting location"},
                "destination": {"type": "string", "description": "Destination"}
            },
            "required": ["origin", "destination"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the internet for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_news",
        "description": "Get latest news headlines, optionally filtered by topic",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Optional topic to filter news"}
            }
        }
    },
    {
        "name": "get_time",
        "description": "Get current time and date",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "Optional timezone"}
            }
        }
    },
    # ========== NEW ADA-STYLE TOOLS ==========
    {
        "name": "create_folder",
        "description": "Create a new folder/directory",
        "parameters": {
            "type": "object",
            "properties": {
                "folder_path": {"type": "string", "description": "Path for the new folder"}
            },
            "required": ["folder_path"]
        }
    },
    {
        "name": "create_file",
        "description": "Create a new file with optional content",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path for the new file"},
                "content": {"type": "string", "description": "File content (optional)"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "list_files",
        "description": "List files and folders in a directory",
        "parameters": {
            "type": "object",
            "properties": {
                "folder_path": {"type": "string", "description": "Path to list (default: workspace)"}
            }
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file to read"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "edit_file",
        "description": "Edit/write content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "New content to write"},
                "append": {"type": "boolean", "description": "Append instead of overwrite"}
            },
            "required": ["file_path", "content"]
        }
    },
    {
        "name": "delete_file",
        "description": "Delete a file or folder",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to delete"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "open_application",
        "description": "Open an application by name (e.g., Chrome, Firefox, Terminal, VS Code)",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {"type": "string", "description": "Name of the application to open"}
            },
            "required": ["app_name"]
        }
    },
    {
        "name": "open_url",
        "description": "Open a URL in the default web browser",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to open"}
            },
            "required": ["url"]
        }
    },
    {
        "name": "google_search",
        "description": "Search Google in the browser",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "capture_screen",
        "description": "Capture a screenshot of the current screen",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "execute_python",
        "description": "Execute Python code and return the result",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"]
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression (e.g., '2 + 2 * 3')"}
            },
            "required": ["expression"]
        }
    }
]


# ============================================================================
# TEST
# ============================================================================

async def test_tools():
    """Test the tools"""
    tools = JarvisTools()

    print("Testing Weather...")
    weather = await tools.weather.get_current_weather("London")
    print(f"Weather: {tools.weather.format_weather(weather)}")
    print()

    print("Testing Search...")
    answer = await tools.search.get_answer("What is Python programming language?")
    print(f"Answer: {answer}")
    print()

    print("Testing News...")
    headlines = await tools.news.get_headlines()
    print(f"News: {tools.news.format_headlines(headlines)}")
    print()

    print("Testing Time...")
    time_data = await tools.time.get_time()
    print(f"Time: {tools.time.format_time(time_data)}")
    print()

    print("Testing Auto-detect...")
    queries = [
        "What's the weather in New York?",
        "What time is it?",
        "What's in the news today?",
        "Search for Python tutorials"
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        response = await tools.handle_query(q)
        print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(test_tools())
