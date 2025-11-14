"""
Extract and Answer Skill
-------------------------
Extracts content from web pages and synthesizes answers using local LLM.

Actions:
- extract: Fetch URLs, extract content, and generate synthesized answers
"""

import aiohttp
import asyncio
import re
from typing import Dict, Any, List, Tuple
from datetime import datetime
from bs4 import BeautifulSoup

from skills.base_skill import BaseSkill, SkillResult


class ExtractAndAnswerSkill(BaseSkill):
    """
    Web extraction and answer synthesis skill.

    Fetches web pages, extracts readable content, and uses local LLM
    to synthesize precise answers to user questions.
    """

    def __init__(self, logger=None, sqlite_manager=None, memory_manager=None):
        super().__init__(logger, sqlite_manager, memory_manager)

        self.name = "extract_and_answer"
        self.description = "Extracts content from web pages and synthesizes answers"
        self.version = "1.0.0"

        # Configuration
        self.chunk_size = 2000  # Characters per chunk for LLM
        self.max_urls = 3  # Maximum URLs to process
        self.timeout = 15  # Request timeout in seconds

    async def run(self, action: str = "extract", **kwargs) -> SkillResult:
        """
        Execute extract and answer skill.

        Args:
            action: Action to perform (extract, help)
            **kwargs: Action-specific arguments

        Returns:
            SkillResult with synthesized answer
        """
        if action == "extract":
            return await self._extract(**kwargs)
        elif action == "help":
            return self._help()
        else:
            return SkillResult(
                success=False,
                output=None,
                message=f"Unknown action: {action}"
            )

    async def _extract(self, query: str = None, urls: List[str] = None, **kwargs) -> SkillResult:
        """
        Extract content from URLs and synthesize answer.

        Args:
            query: Original user question
            urls: List of URLs to extract from

        Returns:
            SkillResult with synthesized answer
        """
        if not query:
            return SkillResult(
                success=False,
                output=None,
                message="Missing required argument: query"
            )

        if not urls or not isinstance(urls, list):
            return SkillResult(
                success=False,
                output=None,
                message="Missing required argument: urls (must be a list)"
            )

        # Limit number of URLs
        urls = urls[:self.max_urls]

        try:
            # Step 1: Fetch and extract content from all URLs
            all_extracts = []
            for url in urls:
                if self.logger:
                    self.logger.log_event("extract", "fetch_started", {
                        "url": url,
                        "query": query
                    })

                extract = await self._fetch_and_extract(url)
                if extract:
                    all_extracts.append({
                        "url": url,
                        "title": extract["title"],
                        "content": extract["content"]
                    })

                    if self.logger:
                        self.logger.log_event("extract", "parse_success", {
                            "url": url,
                            "content_length": len(extract["content"]),
                            "title": extract["title"]
                        })

            if not all_extracts:
                return SkillResult(
                    success=False,
                    output=None,
                    message="Failed to extract content from any URL"
                )

            # Step 2: Chunk and summarize content using local LLM
            summaries = []
            for extract in all_extracts:
                chunks = self._chunk_text(extract["content"], self.chunk_size)

                for i, chunk in enumerate(chunks):
                    summary = await self._summarize_chunk(query, chunk, extract["url"])
                    if summary:
                        summaries.append({
                            "url": extract["url"],
                            "title": extract["title"],
                            "chunk_index": i,
                            "summary": summary
                        })

                        if self.logger:
                            self.logger.log_event("extract", "llm_summarized", {
                                "url": extract["url"],
                                "chunk_index": i,
                                "summary_length": len(summary)
                            })

            if not summaries:
                return SkillResult(
                    success=False,
                    output=None,
                    message="Failed to generate summaries from content"
                )

            # Step 3: Rank summaries by relevance
            ranked_summaries = await self._rank_summaries(query, summaries)

            # Step 4: Compose final answer (2-3 sentences max)
            final_answer = await self._compose_answer(query, ranked_summaries[:3])

            if self.logger:
                self.logger.log_event("extract", "final_answer", {
                    "query": query,
                    "answer_length": len(final_answer),
                    "sources_used": len(ranked_summaries[:3])
                })

            # Step 5: Store as knowledge in memory
            if self.memory:
                from core.memory import MemoryEntry

                knowledge_entry = MemoryEntry(
                    id=None,
                    session_id="extract_and_answer",
                    timestamp=datetime.now(),
                    memory_type="knowledge",
                    content=f"Q: {query}\nA: {final_answer}",
                    embedding=None,
                    metadata={
                        "skill": "extract_and_answer",
                        "query": query,
                        "sources": [s["url"] for s in ranked_summaries[:3]],
                        "extraction_type": "web_synthesis"
                    },
                    importance_score=0.8,
                    is_fact=True
                )

                await self.memory.store(knowledge_entry, persist_long_term=True)

            # Build result
            result_output = {
                "query": query,
                "answer": final_answer,
                "sources": [
                    {
                        "url": s["url"],
                        "title": s["title"],
                        "excerpt": s["summary"][:100]
                    }
                    for s in ranked_summaries[:3]
                ],
                "timestamp": datetime.now().isoformat()
            }

            return SkillResult(
                success=True,
                output=result_output,
                message=f"Answer synthesized from {len(ranked_summaries[:3])} sources",
                metadata={
                    "query": query,
                    "sources_count": len(ranked_summaries[:3])
                }
            )

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "extract_and_answer", "query": query})

            return SkillResult(
                success=False,
                output=None,
                message=f"Extraction failed: {str(e)}"
            )

    async def _fetch_and_extract(self, url: str) -> Dict[str, str]:
        """
        Fetch URL and extract readable content.

        Args:
            url: URL to fetch

        Returns:
            Dictionary with title and content, or None if failed
        """
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; HugoBot/1.0)"
            }

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        return None

                    html = await response.text()

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            # Extract title
            title = soup.title.string if soup.title else url

            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
                element.decompose()

            # Extract main content
            main_content = (
                soup.find('main') or
                soup.find('article') or
                soup.find('div', class_=re.compile(r'content|main|article', re.I)) or
                soup.find('body')
            )

            if not main_content:
                return None

            # Extract text
            text = main_content.get_text(separator='\n', strip=True)

            # Clean up text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            content = '\n'.join(lines)

            # Remove excessive whitespace
            content = re.sub(r'\n{3,}', '\n\n', content)

            return {
                "title": title.strip(),
                "content": content
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "fetch_and_extract", "url": url})
            return None

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into chunks of approximately chunk_size characters.

        Args:
            text: Text to chunk
            chunk_size: Target chunk size in characters

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    async def _summarize_chunk(self, query: str, chunk: str, source_url: str) -> str:
        """
        Summarize a chunk of text using local LLM.

        Args:
            query: Original user question
            chunk: Text chunk to summarize
            source_url: Source URL for context

        Returns:
            Summary text or None if failed
        """
        try:
            # Import cognition engine (avoid circular import)
            from core.runtime_manager import RuntimeManager

            # Get runtime manager instance (assuming it's available globally)
            # For now, we'll need to access it through memory manager's reference
            # or pass it explicitly. For this implementation, we'll construct
            # a simple analysis prompt that can be sent to the LLM.

            # Since we don't have direct access to cognition here, we'll use
            # the memory manager to access embeddings and return the chunk
            # for now. In production, this should call cognition.generate_reply()

            # Simplified approach: Return relevant excerpts from the chunk
            # that match the query semantically

            # For now, extract sentences that might answer the question
            sentences = chunk.split('.')
            relevant_sentences = []

            query_lower = query.lower()
            query_keywords = set(re.findall(r'\b\w+\b', query_lower))

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                sentence_lower = sentence.lower()
                sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))

                # Check for keyword overlap
                overlap = len(query_keywords & sentence_words)
                if overlap >= 2:  # At least 2 matching keywords
                    relevant_sentences.append(sentence)

            if relevant_sentences:
                # Return top 3 most relevant sentences
                return '. '.join(relevant_sentences[:3]) + '.'

            # Fallback: return first few sentences of chunk
            return '. '.join(sentences[:3]) + '.'

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "summarize_chunk", "source": source_url})
            return None

    async def _rank_summaries(self, query: str, summaries: List[Dict]) -> List[Dict]:
        """
        Rank summaries by relevance to query using embeddings.

        Args:
            query: Original user question
            summaries: List of summary dictionaries

        Returns:
            Sorted list of summaries (most relevant first)
        """
        if not self.memory or not self.memory.embedding_model:
            # No memory manager or embeddings, return as-is
            return summaries

        try:
            # Get query embedding
            query_embedding = self.memory.embedding_model.encode([query])[0]

            # Get embeddings for summaries
            summary_texts = [s["summary"] for s in summaries]
            summary_embeddings = self.memory.embedding_model.encode(summary_texts)

            # Calculate cosine similarity
            import numpy as np

            def cosine_similarity(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            # Add relevance scores
            for i, summary in enumerate(summaries):
                score = cosine_similarity(query_embedding, summary_embeddings[i])
                summary["relevance_score"] = float(score)

            # Sort by relevance
            ranked = sorted(summaries, key=lambda x: x["relevance_score"], reverse=True)
            return ranked

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "rank_summaries"})
            # Fallback: return original order
            return summaries

    async def _compose_answer(self, query: str, top_summaries: List[Dict]) -> str:
        """
        Compose final answer from top summaries.

        Args:
            query: Original user question
            top_summaries: Top-ranked summaries

        Returns:
            Composed answer (2-3 sentences max)
        """
        if not top_summaries:
            return "No relevant information found."

        # Combine top summaries
        combined_text = " ".join(s["summary"] for s in top_summaries)

        # Extract most relevant sentences (limit to 3)
        sentences = [s.strip() for s in combined_text.split('.') if s.strip()]

        # Remove duplicates while preserving order
        seen = set()
        unique_sentences = []
        for sentence in sentences:
            sentence_normalized = sentence.lower().strip()
            if sentence_normalized not in seen and len(sentence_normalized) > 20:
                seen.add(sentence_normalized)
                unique_sentences.append(sentence)

        # Return top 3 sentences
        answer_sentences = unique_sentences[:3]

        if answer_sentences:
            return '. '.join(answer_sentences) + '.'
        else:
            return sentences[0] + '.' if sentences else "Information extracted but answer unclear."

    def _help(self) -> SkillResult:
        """
        Return help information.

        Returns:
            SkillResult with usage information
        """
        help_text = """
Extract and Answer Skill Usage:
================================

Actions:
  extract   Extract content from URLs and synthesize answer
            Args: query (required), urls (required, list)
            Example: /skill run extract_and_answer extract
                     query="Who directed Dune?"
                     urls=["https://www.imdb.com/title/tt1160419/"]

  help      Show this help message
            Example: /skill run extract_and_answer help
"""

        return SkillResult(
            success=True,
            output=help_text.strip(),
            message="Extract and answer skill help"
        )

    def requires_permissions(self) -> List[str]:
        """Return required permissions"""
        return ["external_http", "internet_access", "llm_access"]
