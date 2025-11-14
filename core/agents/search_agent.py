"""
Search Agent
------------
Autonomous agent for multi-source web investigation.

The SearchAgent performs comprehensive searches across multiple sources:
- DuckDuckGo HTML scraping
- Wikipedia search and extraction
- IMDb search for entertainment queries
- Google Programmable Search API (optional)
"""

import aiohttp
import asyncio
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import quote_plus, urljoin
from bs4 import BeautifulSoup


class SearchAgent:
    """
    Autonomous search agent for multi-source web investigation.

    This agent independently investigates queries across multiple sources,
    extracts content, and produces structured evidence reports.
    """

    def __init__(self, logger=None, extract_skill=None):
        """
        Initialize the search agent.

        Args:
            logger: HugoLogger instance for structured logging
            extract_skill: ExtractAndAnswerSkill instance for content extraction
        """
        self.logger = logger
        self.extract_skill = extract_skill

        # Configuration
        self.max_urls_per_source = 5
        self.max_total_urls = 15
        self.timeout = 15
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

    async def investigate(self, query: str) -> Dict[str, Any]:
        """
        Perform autonomous multi-source investigation.

        Args:
            query: Search query

        Returns:
            Structured investigation report with findings
        """
        try:
            if self.logger:
                self.logger.log_event("agent", "search_agent_started", {
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                })

            # Step 1: Collect URLs from multiple sources
            urls = await self._collect_urls(query)

            if self.logger:
                self.logger.log_event("agent", "urls_collected", {
                    "query": query,
                    "total_urls": len(urls),
                    "sources": list(set(u.get("source") for u in urls))
                })

            # Step 2: Extract content from each URL
            extracted_passages = await self._extract_content(query, urls)

            if self.logger:
                self.logger.log_event("agent", "extraction_complete", {
                    "query": query,
                    "successful_extractions": len(extracted_passages),
                    "total_urls": len(urls)
                })

            # Step 3: Combine evidence
            combined_evidence = self._combine_evidence(extracted_passages)

            # Step 4: Generate report
            report = {
                "query": query,
                "urls_checked": [u["url"] for u in urls],
                "extracted_passages": extracted_passages,
                "combined_evidence": combined_evidence,
                "sources_used": list(set(u.get("source") for u in urls)),
                "timestamp": datetime.now().isoformat(),
                "success": len(extracted_passages) > 0,
                "error": None
            }

            if self.logger:
                self.logger.log_event("agent", "report_ready", {
                    "query": query,
                    "passages_count": len(extracted_passages),
                    "evidence_length": len(combined_evidence),
                    "success": report["success"]
                })

            return report

        except Exception as e:
            # Log the error
            if self.logger:
                self.logger.log_event("agent", "error_detected", {
                    "query": query,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                self.logger.log_error(e, {"phase": "search_agent_investigate", "query": query})

            # Return error report instead of raising
            return {
                "query": query,
                "urls_checked": [],
                "extracted_passages": [],
                "combined_evidence": "",
                "sources_used": [],
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}"
            }

    async def _collect_urls(self, query: str) -> List[Dict[str, str]]:
        """
        Collect URLs from multiple sources.

        Args:
            query: Search query

        Returns:
            List of URL dictionaries with source information
        """
        urls = []

        # Parallel collection from all sources
        tasks = [
            self._search_duckduckgo(query),
            self._search_wikipedia(query),
        ]

        # Add IMDb search for entertainment-related queries
        if self._is_entertainment_query(query):
            tasks.append(self._search_imdb(query))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results from all sources
        for result in results:
            if isinstance(result, list):
                urls.extend(result)
            elif isinstance(result, Exception):
                if self.logger:
                    self.logger.log_error(result, {"phase": "url_collection"})

        # Remove duplicates and limit total URLs
        seen = set()
        unique_urls = []
        for url_dict in urls:
            url = url_dict["url"]
            if url not in seen and len(unique_urls) < self.max_total_urls:
                seen.add(url)
                unique_urls.append(url_dict)

        return unique_urls

    async def _search_duckduckgo(self, query: str) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo via HTML scraping.

        Args:
            query: Search query

        Returns:
            List of URL dictionaries from DuckDuckGo
        """
        urls = []

        try:
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {"User-Agent": self.user_agent}

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(search_url, headers=headers) as response:
                    if response.status != 200:
                        return urls

                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Find result links
                    results = soup.find_all('a', class_='result__a')[:self.max_urls_per_source]

                    for result in results:
                        url = result.get('href')
                        if url:
                            # Convert protocol-relative URLs to https
                            if url.startswith('//'):
                                url = 'https:' + url
                            # Convert relative URLs to absolute
                            elif not url.startswith('http'):
                                url = urljoin("https://duckduckgo.com", url)

                            urls.append({
                                "url": url,
                                "source": "duckduckgo",
                                "title": result.get_text(strip=True)
                            })

            if self.logger:
                self.logger.log_event("agent", "duckduckgo_search_complete", {
                    "query": query,
                    "urls_found": len(urls)
                })

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "duckduckgo_search", "query": query})

        return urls

    async def _search_wikipedia(self, query: str) -> List[Dict[str, str]]:
        """
        Search Wikipedia and return article URLs.

        Args:
            query: Search query

        Returns:
            List of Wikipedia article URLs
        """
        urls = []

        try:
            # Use Wikipedia API for search
            api_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "opensearch",
                "search": query,
                "limit": min(3, self.max_urls_per_source),
                "format": "json"
            }

            timeout = aiohttp.ClientTimeout(total=self.timeout)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(api_url, params=params) as response:
                    if response.status != 200:
                        return urls

                    data = await response.json()

                    # Extract URLs from response
                    # Format: [query, [titles], [descriptions], [urls]]
                    if len(data) >= 4:
                        titles = data[1]
                        article_urls = data[3]

                        for title, url in zip(titles, article_urls):
                            urls.append({
                                "url": url,
                                "source": "wikipedia",
                                "title": title
                            })

            if self.logger:
                self.logger.log_event("agent", "wikipedia_search_complete", {
                    "query": query,
                    "urls_found": len(urls)
                })

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "wikipedia_search", "query": query})

        return urls

    async def _search_imdb(self, query: str) -> List[Dict[str, str]]:
        """
        Search IMDb for entertainment content.

        Args:
            query: Search query

        Returns:
            List of IMDb URLs
        """
        urls = []

        try:
            # IMDb search URL
            search_url = f"https://www.imdb.com/find?q={quote_plus(query)}&s=tt"

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {"User-Agent": self.user_agent}

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(search_url, headers=headers) as response:
                    if response.status != 200:
                        return urls

                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Find title results
                    results = soup.find_all('td', class_='result_text')[:self.max_urls_per_source]

                    for result in results:
                        link = result.find('a')
                        if link:
                            href = link.get('href')
                            if href:
                                # Convert relative URL to absolute
                                full_url = urljoin("https://www.imdb.com", href)
                                urls.append({
                                    "url": full_url,
                                    "source": "imdb",
                                    "title": link.get_text(strip=True)
                                })

            if self.logger:
                self.logger.log_event("agent", "imdb_search_complete", {
                    "query": query,
                    "urls_found": len(urls)
                })

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "imdb_search", "query": query})

        return urls

    def _is_entertainment_query(self, query: str) -> bool:
        """
        Detect if query is entertainment-related.

        Args:
            query: Search query

        Returns:
            True if entertainment-related
        """
        entertainment_keywords = [
            'movie', 'film', 'actor', 'actress', 'director', 'cast',
            'tv show', 'series', 'episode', 'season', 'release date',
            'premiere', 'starring', 'wicked', 'dune', 'avatar',
            'marvel', 'dc', 'star wars', 'netflix', 'hbo'
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in entertainment_keywords)

    async def _extract_content(self, query: str, urls: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Extract content from URLs using the extraction pipeline.

        Args:
            query: Original search query
            urls: List of URL dictionaries

        Returns:
            List of extracted content passages
        """
        passages = []

        # If extract_skill is not available, use direct extraction
        if not self.extract_skill:
            return await self._extract_content_direct(urls)

        # Use extract_skill's extraction methods
        for url_dict in urls:
            try:
                # Use the extract_and_answer skill's extraction method
                extracted = await self.extract_skill._fetch_and_extract_direct(url_dict["url"])

                if extracted and extracted.get("content"):
                    passages.append({
                        "url": url_dict["url"],
                        "source": url_dict["source"],
                        "title": extracted.get("title", url_dict.get("title", "")),
                        "content": extracted["content"][:2000],  # Limit content size
                        "extraction_method": "extract_skill"
                    })

                    if self.logger:
                        self.logger.log_event("agent", "url_extracted", {
                            "url": url_dict["url"],
                            "source": url_dict["source"],
                            "content_length": len(extracted["content"])
                        })

            except Exception as e:
                if self.logger:
                    self.logger.log_error(e, {
                        "phase": "content_extraction",
                        "url": url_dict["url"]
                    })

        return passages

    async def _extract_content_direct(self, urls: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Direct content extraction fallback.

        Args:
            urls: List of URL dictionaries

        Returns:
            List of extracted content passages
        """
        passages = []

        for url_dict in urls:
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                headers = {"User-Agent": self.user_agent}

                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url_dict["url"], headers=headers) as response:
                        if response.status != 200:
                            continue

                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')

                        # Remove unwanted elements
                        for element in soup(["script", "style", "nav", "header", "footer"]):
                            element.decompose()

                        # Extract text
                        text = soup.get_text(separator='\n', strip=True)
                        lines = [line.strip() for line in text.split('\n') if line.strip()]
                        content = '\n'.join(lines)[:2000]

                        if len(content) > 100:
                            passages.append({
                                "url": url_dict["url"],
                                "source": url_dict["source"],
                                "title": url_dict.get("title", ""),
                                "content": content,
                                "extraction_method": "direct"
                            })

            except Exception as e:
                if self.logger:
                    self.logger.log_error(e, {
                        "phase": "direct_extraction",
                        "url": url_dict["url"]
                    })

        return passages

    def _combine_evidence(self, passages: List[Dict[str, Any]]) -> str:
        """
        Combine extracted passages into coherent evidence.

        Args:
            passages: List of extracted content passages

        Returns:
            Combined evidence string
        """
        if not passages:
            return ""

        evidence_parts = []

        for i, passage in enumerate(passages[:5], 1):  # Limit to top 5
            source_label = f"[{passage['source'].upper()}]"
            title = passage.get('title', 'Untitled')
            content = passage['content'][:500]  # Limit each passage

            evidence_parts.append(
                f"{source_label} {title}\n{content}\n"
            )

        return "\n\n".join(evidence_parts)
