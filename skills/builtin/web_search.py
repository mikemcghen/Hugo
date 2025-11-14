"""
Web Search Skill
----------------
Searches the web using DuckDuckGo Instant Answer API.

Actions:
- search: Perform a web search and return structured results
"""

import aiohttp
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from skills.base_skill import BaseSkill, SkillResult


class WebSearchSkill(BaseSkill):
    """
    Web search skill using DuckDuckGo Instant Answer API.

    Provides structured web search results without tracking.
    """

    def __init__(self, logger=None, sqlite_manager=None, memory_manager=None):
        super().__init__(logger, sqlite_manager, memory_manager)

        self.name = "web_search"
        self.description = "Searches the web and returns structured results"
        self.version = "1.0.0"

        # DuckDuckGo Instant Answer API (no API key required)
        self.api_url = "https://api.duckduckgo.com/"

        # Polling configuration for 202 ACCEPTED responses
        self.max_poll_attempts = 5
        self.poll_delay_ms = 500

    async def run(self, action: str = "search", **kwargs) -> SkillResult:
        """
        Execute web search skill.

        Args:
            action: Action to perform (search, help)
            **kwargs: Action-specific arguments

        Returns:
            SkillResult with search results
        """
        if action == "search":
            return await self._search(**kwargs)
        elif action == "help":
            return self._help()
        else:
            return SkillResult(
                success=False,
                output=None,
                message=f"Unknown action: {action}"
            )

    async def _search(self, query: str = None, **kwargs) -> SkillResult:
        """
        Perform web search with automatic polling for 202 ACCEPTED responses.

        Args:
            query: Search query string

        Returns:
            SkillResult with search results
        """
        if not query:
            return SkillResult(
                success=False,
                output=None,
                message="Missing required argument: query"
            )

        try:
            # Call DuckDuckGo Instant Answer API with retry logic for 202
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }

            timeout = aiohttp.ClientTimeout(total=10)
            attempt = 0
            data = None

            async with aiohttp.ClientSession(timeout=timeout) as session:
                while attempt < self.max_poll_attempts:
                    attempt += 1

                    async with session.get(self.api_url, params=params) as response:
                        status = response.status

                        # HTTP 200 OK - Success
                        if status == 200:
                            if self.logger:
                                if attempt > 1:
                                    self.logger.log_event("web_search", "poll_success", {
                                        "query": query,
                                        "attempts": attempt
                                    })
                            data = await response.json()
                            break

                        # HTTP 202 ACCEPTED - Result not ready, poll again
                        elif status == 202:
                            if self.logger:
                                if attempt == 1:
                                    self.logger.log_event("web_search", "poll_started", {
                                        "query": query,
                                        "status": 202
                                    })
                                else:
                                    self.logger.log_event("web_search", "poll_retry", {
                                        "query": query,
                                        "attempt": attempt,
                                        "max_attempts": self.max_poll_attempts
                                    })

                            # Wait before retrying
                            if attempt < self.max_poll_attempts:
                                await asyncio.sleep(self.poll_delay_ms / 1000.0)
                            continue

                        # Any other status - Error
                        else:
                            if self.logger:
                                self.logger.log_event("web_search", "poll_failed", {
                                    "query": query,
                                    "status": status,
                                    "attempt": attempt
                                })

                            return SkillResult(
                                success=False,
                                output=None,
                                message=f"Search API returned status {status}"
                            )

            # If we exhausted all retries without getting 200
            if data is None:
                if self.logger:
                    self.logger.log_event("web_search", "poll_failed", {
                        "query": query,
                        "reason": "max_attempts_reached",
                        "attempts": self.max_poll_attempts
                    })

                return SkillResult(
                    success=False,
                    output=None,
                    message="Search results not ready (API stuck in 202)."
                )

            # Extract relevant information
            results = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "abstract": data.get("Abstract", ""),
                "abstract_text": data.get("AbstractText", ""),
                "abstract_source": data.get("AbstractSource", ""),
                "abstract_url": data.get("AbstractURL", ""),
                "heading": data.get("Heading", ""),
                "answer": data.get("Answer", ""),
                "definition": data.get("Definition", ""),
                "definition_source": data.get("DefinitionSource", ""),
                "definition_url": data.get("DefinitionURL", ""),
                "related_topics": []
            }

            # Extract related topics
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and "Text" in topic:
                    results["related_topics"].append({
                        "text": topic.get("Text", ""),
                        "url": topic.get("FirstURL", "")
                    })

            # Extract URLs from related topics for downstream processing
            results["urls"] = []

            # Add abstract URL if available
            if results["abstract_url"]:
                results["urls"].append(results["abstract_url"])

            # Add definition URL if available
            if results["definition_url"]:
                results["urls"].append(results["definition_url"])

            # Add related topic URLs
            for topic in results["related_topics"]:
                if topic.get("url"):
                    results["urls"].append(topic["url"])

            # Log result status
            if self.logger:
                self.logger.log_event("web_search", "search_complete", {
                    "query": query,
                    "has_abstract": bool(results["abstract_text"]),
                    "has_answer": bool(results["answer"]),
                    "has_definition": bool(results["definition"]),
                    "url_count": len(results["urls"]),
                    "status": "success" if results["urls"] else "no_results"
                })

            # Store in memory if available
            if self.memory:
                from core.memory import MemoryEntry

                content = f"Web search: {query}"
                if results["abstract_text"]:
                    content += f" - {results['abstract_text'][:200]}"

                search_entry = MemoryEntry(
                    id=None,
                    session_id="web_search",
                    timestamp=datetime.now(),
                    memory_type="knowledge",
                    content=content,
                    embedding=None,
                    metadata={
                        "skill": "web_search",
                        "query": query,
                        "source": results["abstract_source"],
                        "url": results["abstract_url"]
                    },
                    importance_score=0.6,
                    is_fact=True
                )

                await self.memory.store(search_entry, persist_long_term=True)

            return SkillResult(
                success=True,
                output=results,
                message=f"Search completed for '{query}'",
                metadata={
                    "query": query,
                    "has_results": bool(results["abstract_text"] or results["answer"] or results["definition"]),
                    "url_count": len(results["urls"]),
                    "status": "success" if results["urls"] else "no_results"
                }
            )

        except asyncio.TimeoutError:
            return SkillResult(
                success=False,
                output=None,
                message="Search request timed out"
            )
        except aiohttp.ClientError as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "web_search", "query": query})

            return SkillResult(
                success=False,
                output=None,
                message=f"Network error: {str(e)}"
            )
        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "web_search", "query": query})

            return SkillResult(
                success=False,
                output=None,
                message=f"Search failed: {str(e)}"
            )

    def _help(self) -> SkillResult:
        """
        Return help information.

        Returns:
            SkillResult with usage information
        """
        help_text = """
Web Search Skill Usage:
========================

Actions:
  search    Search the web using DuckDuckGo
            Args: query (required)
            Example: /net search "Python asyncio tutorial"

  help      Show this help message
            Example: /skill run web_search help
"""

        return SkillResult(
            success=True,
            output=help_text.strip(),
            message="Web search skill help"
        )

    def requires_permissions(self) -> List[str]:
        """Return required permissions"""
        return ["external_http", "internet_access"]
