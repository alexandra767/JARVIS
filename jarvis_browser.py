#!/usr/bin/env python3
"""
JARVIS Browser Agent - Autonomous Web Browser Control
Uses Playwright for browser automation with LLM-guided actions
"""

import asyncio
import base64
import json
import os
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger("JarvisBrowser")


@dataclass
class BrowserConfig:
    headless: bool = False
    viewport_width: int = 1280
    viewport_height: int = 720
    timeout: int = 30000  # ms
    ollama_url: str = "http://192.168.50.129:11434"
    model: str = "qwen2.5:72b"


class BrowserAction:
    """Represents an action to take in the browser"""

    def __init__(self, action_type: str, **kwargs):
        self.type = action_type
        self.params = kwargs

    @classmethod
    def click(cls, x: int, y: int):
        return cls("click", x=x, y=y)

    @classmethod
    def type_text(cls, text: str, selector: str = None):
        return cls("type", text=text, selector=selector)

    @classmethod
    def scroll(cls, direction: str = "down", amount: int = 300):
        return cls("scroll", direction=direction, amount=amount)

    @classmethod
    def navigate(cls, url: str):
        return cls("navigate", url=url)

    @classmethod
    def wait(cls, seconds: float = 1.0):
        return cls("wait", seconds=seconds)

    @classmethod
    def done(cls, result: str = ""):
        return cls("done", result=result)

    @classmethod
    def error(cls, message: str):
        return cls("error", message=message)


class LLMBrowserPlanner:
    """
    Uses LLM to plan browser actions based on screenshots.
    """

    def __init__(self, config: BrowserConfig):
        self.config = config
        self.action_history = []

    async def plan_action(
        self,
        task: str,
        screenshot_base64: str,
        page_url: str,
        previous_actions: List[Dict] = None
    ) -> BrowserAction:
        """
        Given a task and current screenshot, plan the next action.
        """
        import requests

        # Build prompt
        prompt = self._build_prompt(task, page_url, previous_actions)

        try:
            # Call Ollama with vision model
            response = requests.post(
                f"{self.config.ollama_url}/api/generate",
                json={
                    "model": "llava",  # Vision model for screenshot analysis
                    "prompt": prompt,
                    "images": [screenshot_base64],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 500
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json().get("response", "")
                return self._parse_action(result)
            else:
                logger.error(f"LLM request failed: {response.status_code}")
                return BrowserAction.error(f"LLM error: {response.status_code}")

        except Exception as e:
            logger.error(f"LLM planning error: {e}")
            return BrowserAction.error(str(e))

    def _build_prompt(self, task: str, page_url: str, previous_actions: List[Dict]) -> str:
        """Build prompt for LLM"""
        history_text = ""
        if previous_actions:
            history_text = "Previous actions:\n"
            for i, action in enumerate(previous_actions[-5:]):  # Last 5 actions
                history_text += f"  {i+1}. {action['type']}: {action.get('params', {})}\n"

        return f"""You are a web browser automation agent. Analyze the screenshot and determine the next action to complete the task.

TASK: {task}

CURRENT URL: {page_url}

{history_text}

Available actions:
1. CLICK x y - Click at coordinates (x, y)
2. TYPE "text" - Type text into the focused input field
3. SCROLL down/up amount - Scroll the page
4. NAVIGATE url - Go to a URL
5. WAIT seconds - Wait for page to load
6. DONE result - Task is complete, with result description
7. ERROR message - Cannot complete task

Analyze the screenshot and respond with ONLY the next action in the format:
ACTION: <action_type> <parameters>

Example responses:
- ACTION: CLICK 640 350
- ACTION: TYPE "search query"
- ACTION: SCROLL down 300
- ACTION: DONE Successfully added item to cart
- ACTION: NAVIGATE https://amazon.com

Think step by step about what you see and what action will help complete the task."""

    def _parse_action(self, response: str) -> BrowserAction:
        """Parse LLM response into BrowserAction"""
        response = response.strip()

        # Look for ACTION: pattern
        match = re.search(r'ACTION:\s*(\w+)\s*(.*)', response, re.IGNORECASE)
        if not match:
            # Try to extract action from response
            logger.warning(f"Could not parse action from: {response}")
            return BrowserAction.wait(1.0)

        action_type = match.group(1).upper()
        params = match.group(2).strip()

        if action_type == "CLICK":
            coords = re.findall(r'\d+', params)
            if len(coords) >= 2:
                return BrowserAction.click(int(coords[0]), int(coords[1]))

        elif action_type == "TYPE":
            # Extract quoted text
            text_match = re.search(r'"([^"]*)"', params)
            if text_match:
                return BrowserAction.type_text(text_match.group(1))
            return BrowserAction.type_text(params)

        elif action_type == "SCROLL":
            parts = params.split()
            direction = parts[0] if parts else "down"
            amount = int(parts[1]) if len(parts) > 1 else 300
            return BrowserAction.scroll(direction, amount)

        elif action_type == "NAVIGATE":
            return BrowserAction.navigate(params)

        elif action_type == "WAIT":
            seconds = float(params) if params else 1.0
            return BrowserAction.wait(seconds)

        elif action_type == "DONE":
            return BrowserAction.done(params)

        elif action_type == "ERROR":
            return BrowserAction.error(params)

        return BrowserAction.wait(1.0)


class JarvisBrowser:
    """
    Autonomous browser agent for Jarvis.
    Takes high-level tasks and executes them in the browser.
    """

    def __init__(self, config: BrowserConfig = None):
        self.config = config or BrowserConfig()
        self.browser = None
        self.context = None
        self.page = None
        self.planner = LLMBrowserPlanner(self.config)
        self.running = False
        self.current_task = None
        self.action_history = []

    async def start(self):
        """Start browser"""
        try:
            from playwright.async_api import async_playwright

            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.config.headless
            )
            self.context = await self.browser.new_context(
                viewport={
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height
                }
            )
            self.page = await self.context.new_page()
            self.running = True
            logger.info("Browser started")

        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            raise

    async def stop(self):
        """Stop browser"""
        self.running = False
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("Browser stopped")

    async def execute(self, task: str, start_url: str = None) -> Dict[str, Any]:
        """
        Execute a high-level task in the browser.
        Returns result dict with success status and details.
        """
        if not self.running:
            await self.start()

        self.current_task = task
        self.action_history = []

        logger.info(f"Executing browser task: {task}")

        # Navigate to start URL if provided
        if start_url:
            await self.page.goto(start_url, wait_until="networkidle")
        elif not self.page.url or self.page.url == "about:blank":
            await self.page.goto("https://google.com", wait_until="networkidle")

        max_steps = 30  # Prevent infinite loops
        step = 0

        while step < max_steps:
            step += 1

            try:
                # Take screenshot
                screenshot = await self.page.screenshot(type="png")
                screenshot_b64 = base64.b64encode(screenshot).decode()

                # Get current URL
                current_url = self.page.url

                # Plan next action
                action = await self.planner.plan_action(
                    task=task,
                    screenshot_base64=screenshot_b64,
                    page_url=current_url,
                    previous_actions=self.action_history
                )

                # Log action
                logger.info(f"Step {step}: {action.type} {action.params}")
                self.action_history.append({
                    "type": action.type,
                    "params": action.params,
                    "url": current_url
                })

                # Execute action
                if action.type == "done":
                    return {
                        "success": True,
                        "result": action.params.get("result", "Task completed"),
                        "steps": step,
                        "history": self.action_history
                    }

                elif action.type == "error":
                    return {
                        "success": False,
                        "error": action.params.get("message", "Unknown error"),
                        "steps": step,
                        "history": self.action_history
                    }

                elif action.type == "click":
                    await self.page.mouse.click(
                        action.params["x"],
                        action.params["y"]
                    )
                    await asyncio.sleep(0.5)

                elif action.type == "type":
                    text = action.params.get("text", "")
                    selector = action.params.get("selector")
                    if selector:
                        await self.page.fill(selector, text)
                    else:
                        await self.page.keyboard.type(text)
                    await asyncio.sleep(0.3)

                elif action.type == "scroll":
                    direction = action.params.get("direction", "down")
                    amount = action.params.get("amount", 300)
                    if direction == "up":
                        amount = -amount
                    await self.page.evaluate(f"window.scrollBy(0, {amount})")
                    await asyncio.sleep(0.3)

                elif action.type == "navigate":
                    url = action.params.get("url", "")
                    if url:
                        await self.page.goto(url, wait_until="networkidle")
                    await asyncio.sleep(1.0)

                elif action.type == "wait":
                    seconds = action.params.get("seconds", 1.0)
                    await asyncio.sleep(seconds)

                # Wait for any navigation/loading
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Action execution error: {e}")
                self.action_history.append({
                    "type": "error",
                    "message": str(e)
                })
                # Continue trying

        return {
            "success": False,
            "error": "Maximum steps reached",
            "steps": step,
            "history": self.action_history
        }

    async def screenshot(self) -> bytes:
        """Take screenshot of current page"""
        if self.page:
            return await self.page.screenshot(type="png")
        return None

    async def get_page_content(self) -> str:
        """Get text content of current page"""
        if self.page:
            return await self.page.content()
        return ""

    async def simple_navigate(self, url: str):
        """Simple navigation without LLM planning"""
        if not self.running:
            await self.start()
        await self.page.goto(url, wait_until="networkidle")

    async def simple_search(self, query: str, engine: str = "google") -> str:
        """Simple web search"""
        if not self.running:
            await self.start()

        search_urls = {
            "google": f"https://www.google.com/search?q={query}",
            "duckduckgo": f"https://duckduckgo.com/?q={query}",
            "bing": f"https://www.bing.com/search?q={query}"
        }

        url = search_urls.get(engine, search_urls["google"])
        await self.page.goto(url, wait_until="networkidle")

        # Get search results text
        content = await self.page.inner_text("body")
        return content[:5000]  # Limit content


# Tool definitions for Alexandra integration
BROWSER_TOOLS = [
    {
        "name": "browse_web",
        "description": "Execute a task in the web browser. Can shop, research, book, or interact with any website.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to complete (e.g., 'go to amazon and add a 25kg servo motor to cart')"
                },
                "start_url": {
                    "type": "string",
                    "description": "Optional starting URL"
                }
            },
            "required": ["task"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "engine": {
                    "type": "string",
                    "enum": ["google", "duckduckgo", "bing"],
                    "default": "google"
                }
            },
            "required": ["query"]
        }
    }
]


# Test function
async def test_browser():
    """Test browser automation"""
    print("Testing browser automation...")

    browser = JarvisBrowser(BrowserConfig(headless=False))
    await browser.start()

    try:
        # Simple navigation test
        print("Navigating to Google...")
        await browser.simple_navigate("https://google.com")
        await asyncio.sleep(2)

        # Search test
        print("Searching...")
        results = await browser.simple_search("Python programming")
        print(f"Results preview: {results[:500]}...")

        # Task execution test (requires LLM)
        # result = await browser.execute("Search for the weather in London")
        # print(f"Task result: {result}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await browser.stop()


if __name__ == "__main__":
    asyncio.run(test_browser())
