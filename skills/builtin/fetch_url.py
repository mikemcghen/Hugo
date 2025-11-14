"""
Fetch URL Skill
---------------
Downloads webpages and extracts readable content.

Actions:
- fetch: Download and extract content from a URL
- summarize: Fetch and create a summary
"""

import aiohttp
import asyncio
import re
from typing import Dict, Any, List
from datetime import datetime
from bs4 import BeautifulSoup

from skills.base_skill import BaseSkill, SkillResult


class FetchUrlSkill(BaseSkill):
    """
    URL fetching and content extraction skill.

    Downloads webpages, extracts readable text, and optionally summarizes.
    """

    def __init__(self, logger=None, sqlite_manager=None, memory_manager=None):
        super().__init__(logger, sqlite_manager, memory_manager)

        self.name = "fetch_url"
        self.description = "Downloads a webpage and summarizes it"
        self.version = "1.0.0"

    async def run(self, action: str = "fetch", **kwargs) -> SkillResult:
        """
        Execute fetch URL skill.

        Args:
            action: Action to perform (fetch, summarize, help)
            **kwargs: Action-specific arguments

        Returns:
            SkillResult with fetched content
        """
        if action == "fetch":
            return await self._fetch(**kwargs)
        elif action == "summarize":
            return await self._summarize(**kwargs)
        elif action == "help":
            return self._help()
        else:
            return SkillResult(
                success=False,
                output=None,
                message=f"Unknown action: {action}"
            )

    async def _fetch(self, url: str = None, **kwargs) -> SkillResult:
        """
        Fetch and extract content from URL.

        Args:
            url: URL to fetch

        Returns:
            SkillResult with extracted content
        """
        if not url:
            return SkillResult(
                success=False,
                output=None,
                message="Missing required argument: url"
            )

        try:
            # Fetch URL
            timeout = aiohttp.ClientTimeout(total=15)
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; HugoBot/1.0)"
            }

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        return SkillResult(
                            success=False,
                            output=None,
                            message=f"Failed to fetch URL: status {response.status}"
                        )

                    html = await response.text()

            # Extract readable content using BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()

            # Get title
            title = soup.title.string if soup.title else "No title"

            # Extract text from main content areas
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)

            # Clean up text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)

            # Truncate if too long
            max_length = 5000
            if len(text) > max_length:
                text = text[:max_length] + "..."

            results = {
                "url": url,
                "title": title,
                "content": text,
                "length": len(text),
                "timestamp": datetime.now().isoformat()
            }

            # Store in memory if available
            if self.memory:
                from core.memory import MemoryEntry

                content_preview = text[:500] if len(text) > 500 else text

                fetch_entry = MemoryEntry(
                    id=None,
                    session_id="fetch_url",
                    timestamp=datetime.now(),
                    memory_type="knowledge",
                    content=f"Fetched: {title} - {content_preview}",
                    embedding=None,
                    metadata={
                        "skill": "fetch_url",
                        "url": url,
                        "title": title,
                        "content_length": len(text)
                    },
                    importance_score=0.65,
                    is_fact=True
                )

                await self.memory.store(fetch_entry, persist_long_term=True)

            return SkillResult(
                success=True,
                output=results,
                message=f"Fetched '{title}' from {url}",
                metadata={"url": url, "title": title, "length": len(text)}
            )

        except asyncio.TimeoutError:
            return SkillResult(
                success=False,
                output=None,
                message="Request timed out"
            )
        except aiohttp.ClientError as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "fetch_url", "url": url})

            return SkillResult(
                success=False,
                output=None,
                message=f"Network error: {str(e)}"
            )
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "fetch_url", "url": url})

            return SkillResult(
                success=False,
                output=None,
                message=f"Fetch failed: {str(e)}"
            )

    async def _summarize(self, url: str = None, **kwargs) -> SkillResult:
        """
        Fetch URL and create a summary.

        Args:
            url: URL to fetch and summarize

        Returns:
            SkillResult with summary
        """
        # First fetch the content
        fetch_result = await self._fetch(url=url)

        if not fetch_result.success:
            return fetch_result

        # Extract content
        content = fetch_result.output["content"]
        title = fetch_result.output["title"]

        # Create simple extractive summary (first 3 paragraphs)
        paragraphs = [p for p in content.split('\n\n') if len(p) > 50]
        summary = '\n\n'.join(paragraphs[:3])

        if len(summary) > 1000:
            summary = summary[:1000] + "..."

        results = {
            "url": url,
            "title": title,
            "summary": summary,
            "full_length": len(content),
            "summary_length": len(summary),
            "timestamp": datetime.now().isoformat()
        }

        # Store summary in memory
        if self.memory:
            from core.memory import MemoryEntry

            summary_entry = MemoryEntry(
                id=None,
                session_id="fetch_url",
                timestamp=datetime.now(),
                memory_type="knowledge",
                content=f"Summary of {title}: {summary}",
                embedding=None,
                metadata={
                    "skill": "fetch_url",
                    "action": "summarize",
                    "url": url,
                    "title": title
                },
                importance_score=0.7,
                is_fact=True
            )

            await self.memory.store(summary_entry, persist_long_term=True)

        return SkillResult(
            success=True,
            output=results,
            message=f"Summarized '{title}'",
            metadata={"url": url, "title": title}
        )

    def _help(self) -> SkillResult:
        """
        Return help information.

        Returns:
            SkillResult with usage information
        """
        help_text = """
Fetch URL Skill Usage:
======================

Actions:
  fetch       Download and extract content from a URL
              Args: url (required)
              Example: /net fetch "https://example.com"

  summarize   Fetch URL and create a summary
              Args: url (required)
              Example: /net fetch "https://example.com" --summarize

  help        Show this help message
              Example: /skill run fetch_url help
"""

        return SkillResult(
            success=True,
            output=help_text.strip(),
            message="Fetch URL skill help"
        )

    def requires_permissions(self) -> List[str]:
        """Return required permissions"""
        return ["external_http", "internet_access"]
