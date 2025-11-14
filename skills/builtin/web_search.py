"""
Web Search Skill
----------------
Agent-based multi-source web search system.

This skill delegates all search operations to an autonomous SearchAgent
that performs comprehensive investigations across multiple sources.

Actions:
- search: Deploy SearchAgent for multi-source investigation
"""

from typing import Dict, Any, List
from datetime import datetime

from skills.base_skill import BaseSkill, SkillResult
from core.agents import SearchAgent


class WebSearchSkill(BaseSkill):
    """
    Agent-based web search skill.

    Deploys an autonomous SearchAgent to perform multi-source investigations
    including DuckDuckGo, Wikipedia, IMDb, and optional Google search.
    """

    def __init__(self, logger=None, sqlite_manager=None, memory_manager=None, cognition=None):
        super().__init__(logger, sqlite_manager, memory_manager)

        self.name = "web_search"
        self.description = "Agent-based multi-source web search"
        self.version = "2.1.0"
        self.cognition = cognition

        # Initialize search agent
        self.search_agent = None

    async def run(self, action: str = "search", **kwargs) -> SkillResult:
        """
        Execute web search skill.

        Args:
            action: Action to perform (search, help)
            **kwargs: Action-specific arguments

        Returns:
            SkillResult with investigation report
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
        Deploy SearchAgent for investigation.

        Args:
            query: Search query string

        Returns:
            SkillResult with investigation findings
        """
        if not query:
            return SkillResult(
                success=False,
                output=None,
                message="Missing required argument: query"
            )

        try:
            # Initialize search agent if not already done
            if not self.search_agent:
                # Import extract_and_answer skill for content extraction
                extract_skill = None
                try:
                    from skills.skill_registry import SkillRegistry
                    registry = SkillRegistry()
                    extract_skill = registry.get("extract_and_answer")
                except Exception:
                    pass

                self.search_agent = SearchAgent(
                    logger=self.logger,
                    extract_skill=extract_skill,
                    cognition=self.cognition
                )

            if self.logger:
                self.logger.log_event("web_search", "deploying_agent", {
                    "query": query
                })

            # Deploy agent for investigation
            report = await self.search_agent.investigate(query)

            # Check if agent encountered an error
            if report.get("error"):
                if self.logger:
                    self.logger.log_event("web_search", "reporting_agent_error", {
                        "query": query,
                        "error": report["error"]
                    })

                return SkillResult(
                    success=False,
                    output={"error_passthrough": True, "agent_error": report["error"]},
                    message=f"Agent encountered an error: {report['error']}",
                    metadata={"error_passthrough": True}
                )

            # Check if investigation was successful
            if not report["success"]:
                return SkillResult(
                    success=False,
                    output=None,
                    message=f"Investigation found no results for '{query}'"
                )

            # Format results for consumption with synthesized answer
            results = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "urls": report["urls_checked"],
                "sources_used": report["sources_used"],
                "synthesized_answer": report.get("synthesized_answer", ""),
                "answer_support": report.get("answer_support", ""),
                "combined_evidence": report["combined_evidence"],
                "passages_count": len(report["extracted_passages"]),
                "extracted_passages": report["extracted_passages"]
            }

            # Store synthesized answer in memory if available
            if self.memory:
                from core.memory import MemoryEntry

                # Store the concise synthesized answer, not raw evidence
                content = f"Q: {query}\nA: {report.get('synthesized_answer', 'No answer available')}"

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
                        "sources": report["sources_used"],
                        "urls_count": len(report["urls_checked"]),
                        "agent_investigation": True,
                        "synthesized": True
                    },
                    importance_score=0.8,
                    is_fact=True
                )

                await self.memory.store(search_entry, persist_long_term=True)

            return SkillResult(
                success=True,
                output=results,
                message=f"Investigation completed: {len(report['urls_checked'])} sources checked",
                metadata={
                    "query": query,
                    "sources_used": report["sources_used"],
                    "passages_count": len(report["extracted_passages"]),
                    "agent_deployed": True
                }
            )

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "web_search", "query": query})

            return SkillResult(
                success=False,
                output=None,
                message=f"Search investigation failed: {str(e)}"
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

This skill deploys an autonomous SearchAgent that performs
multi-source investigations including:
  - DuckDuckGo HTML scraping
  - Wikipedia search and extraction
  - IMDb search for entertainment queries
  - Optional Google Programmable Search

Actions:
  search    Deploy agent for investigation
            Args: query (required)
            Example: /net search "Python asyncio tutorial"

  help      Show this help message
            Example: /skill run web_search help

The agent autonomously collects URLs, extracts content,
and produces a structured evidence report.
"""

        return SkillResult(
            success=True,
            output=help_text.strip(),
            message="Web search skill help"
        )

    def requires_permissions(self) -> List[str]:
        """Return required permissions"""
        return ["external_http", "internet_access", "agent_deployment"]
