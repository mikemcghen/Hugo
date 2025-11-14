"""
Extract and Answer Skill
-------------------------
Extracts content from web pages and synthesizes answers using local LLM.

Actions:
- extract: Fetch URLs, extract content, and generate synthesized answers
"""

import aiohttp
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from bs4 import BeautifulSoup

from skills.base_skill import BaseSkill, SkillResult


class ExtractAndAnswerSkill(BaseSkill):
    """
    Web extraction and answer synthesis skill.

    Fetches web pages, extracts readable content, and uses local LLM to synthesize precise answers.
    """

    def __init__(self, logger=None, sqlite_manager=None, memory_manager=None, cognition=None):
        super().__init__(logger, sqlite_manager, memory_manager)

        self.name = "extract_and_answer"
        self.description = "Extracts content from web pages and synthesizes answers"
        self.version = "2.0.0"
        self.cognition = cognition

        # Configuration
        self.min_chunk_size = 1800  # Minimum characters per chunk
        self.max_chunk_size = 2200  # Maximum characters per chunk
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

                extract = await self._fetch_and_extract_with_fallback(url)
                if extract and extract.get("content") and len(extract["content"]) > 100:
                    all_extracts.append({
                        "url": url,
                        "title": extract["title"],
                        "content": extract["content"],
                        "method": extract.get("method", "direct")
                    })

                    if self.logger:
                        self.logger.log_event("extract", "parse_success", {
                            "url": url,
                            "content_length": len(extract["content"]),
                            "title": extract["title"],
                            "method": extract.get("method", "direct")
                        })
                else:
                    if self.logger:
                        self.logger.log_event("extract", "parse_failed", {
                            "url": url,
                            "reason": "No content extracted or too short"
                        })

            if not all_extracts:
                return SkillResult(
                    success=False,
                    output=None,
                    message="Failed to extract content from any URL"
                )

            # Step 2: Chunk and summarize content
            summaries = []
            total_chunks = 0

            for extract in all_extracts:
                chunks = self._chunk_text(extract["content"])
                total_chunks += len(chunks)

                if self.logger:
                    self.logger.log_event("extract", "chunk_count", {
                        "url": extract["url"],
                        "chunks": len(chunks),
                        "total_chunks": total_chunks
                    })

                for i, chunk in enumerate(chunks):
                    summary = await self._summarize_chunk(query, chunk, extract["url"], i)
                    if summary and summary.strip():
                        summaries.append({
                            "url": extract["url"],
                            "title": extract["title"],
                            "chunk_index": i,
                            "summary": summary.strip()
                        })

                        if self.logger:
                            self.logger.log_event("extract", "summary_generated", {
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
                        "excerpt": s["summary"][:100] if len(s["summary"]) > 100 else s["summary"]
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

    async def _fetch_and_extract_with_fallback(self, url: str) -> Optional[Dict[str, str]]:
        """
        Fetch URL and extract content.

        Args:
            url: URL to fetch

        Returns:
            Dictionary with title and content, or None if failed
        """
        # Direct fetch only
        result = await self._fetch_and_extract_direct(url)
        if result and len(result.get("content", "")) > 100:
            result["method"] = "direct"
            return result

        return None

    async def _fetch_and_extract_direct(self, url: str) -> Optional[Dict[str, str]]:
        """
        Fetch URL directly and extract readable content.

        Args:
            url: URL to fetch

        Returns:
            Dictionary with title and content, or None if failed
        """
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9"
            }

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, allow_redirects=True) as response:
                    if response.status != 200:
                        if self.logger:
                            self.logger.log_event("extract", "fetch_failed", {
                                "url": url,
                                "status": response.status
                            })
                        return None

                    html = await response.text()

            if not html or len(html) < 100:
                return None

            return self._extract_content_from_html(html, url)

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "fetch_and_extract_direct", "url": url})
            return None


    def _extract_content_from_html(self, html: str, source_url: str) -> Optional[Dict[str, str]]:
        """
        Extract readable content from HTML.

        Args:
            html: HTML content
            source_url: Source URL for title fallback

        Returns:
            Dictionary with title and content
        """
        try:
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            # Extract title
            title = soup.title.string if soup.title else source_url

            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside", "iframe", "noscript"]):
                element.decompose()

            # Extract main content
            main_content = (
                soup.find('main') or
                soup.find('article') or
                soup.find('div', class_=re.compile(r'content|main|article', re.I)) or
                soup.find('body')
            )

            if not main_content:
                return {"title": title.strip() if title else source_url, "content": ""}

            # Extract text
            text = main_content.get_text(separator='\n', strip=True)

            # Clean up text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            content = '\n'.join(lines)

            # Remove excessive whitespace
            content = re.sub(r'\n{3,}', '\n\n', content)

            # Remove very short lines (likely navigation/UI text)
            paragraphs = content.split('\n\n')
            filtered_paragraphs = [p for p in paragraphs if len(p) > 30]
            content = '\n\n'.join(filtered_paragraphs)

            return {
                "title": title.strip() if title else source_url,
                "content": content
            }

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "extract_content_from_html"})
            return None

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks of 1800-2200 characters.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if len(text) <= self.max_chunk_size:
            return [text]

        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""

        for paragraph in paragraphs:
            # If adding this paragraph would exceed max_chunk_size
            if len(current_chunk) + len(paragraph) + 2 > self.max_chunk_size:
                # Save current chunk if it meets minimum size
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"
                else:
                    # Chunk too small, add paragraph anyway
                    current_chunk += paragraph + "\n\n"
                    if len(current_chunk) >= self.max_chunk_size:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
            else:
                current_chunk += paragraph + "\n\n"

        # Add remaining chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If no chunks created, return original text
        if not chunks:
            chunks = [text]

        return chunks

    async def _summarize_chunk(self, query: str, chunk: str, source_url: str, chunk_index: int) -> str:
        """
        Extract relevant content from chunk that answers the query.

        Uses keyword matching and sentence extraction to find relevant information.

        Args:
            query: Original user question
            chunk: Text chunk to process
            source_url: Source URL for context
            chunk_index: Index of this chunk

        Returns:
            Extracted relevant text or None if failed
        """
        try:
            # Split into sentences
            # Handle multiple sentence endings
            text = chunk.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|')
            sentences = [s.strip() for s in text.split('|') if s.strip()]

            # Extract query keywords (remove common words)
            query_lower = query.lower()
            common_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'was', 'were',
                           'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
            query_keywords = set(re.findall(r'\b\w+\b', query_lower)) - common_words

            # Score each sentence by keyword matches
            scored_sentences = []
            for sentence in sentences:
                if len(sentence) < 20:  # Skip very short sentences
                    continue

                sentence_lower = sentence.lower()
                sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))

                # Count keyword matches
                matches = len(query_keywords & sentence_words)

                # Boost score if sentence appears to be answering the question
                if any(word in sentence_lower for word in ['is', 'was', 'are', 'were', 'has', 'have', 'did', 'will']):
                    matches += 0.5

                if matches > 0:
                    scored_sentences.append((matches, sentence))

            # Sort by score and take top sentences
            scored_sentences.sort(reverse=True, key=lambda x: x[0])
            top_sentences = [s[1] for s in scored_sentences[:5]]  # Top 5 sentences

            if top_sentences:
                return ' '.join(top_sentences)

            # Fallback: return first few sentences of chunk
            fallback_sentences = sentences[:3] if len(sentences) >= 3 else sentences
            return ' '.join(fallback_sentences) if fallback_sentences else chunk[:500]

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "summarize_chunk", "source": source_url, "chunk_index": chunk_index})
            # Return first part of chunk as fallback
            return chunk[:500] if len(chunk) > 500 else chunk

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
        Compose final answer from top summaries using LLM synthesis.

        Args:
            query: Original user question
            top_summaries: Top-ranked summaries

        Returns:
            Synthesized answer (2-3 sentences max)
        """
        if not top_summaries:
            return "No relevant information found."

        # Combine top summaries
        combined_summary = " ".join(s["summary"] for s in top_summaries)

        # If summary is empty or too short, return no information
        if not combined_summary or len(combined_summary.strip()) < 50:
            return "No clear information available."

        # Synthesize final answer using cognition
        if self.logger:
            self.logger.log_event("extract", "synthesis_started", {
                "query": query,
                "combined_length": len(combined_summary)
            })

        try:
            # Check if cognition engine is available
            if not self.cognition:
                if self.logger:
                    self.logger.log_event("extract", "no_cognition_available", {
                        "query": query
                    })
                return "No clear information available."

            synthesis_prompt = (
                f"Based on the extracted text below, answer the user's question:\n\n"
                f"User question: {query}\n"
                f"Extracted info: {combined_summary}\n\n"
                "Respond with a direct, concise answer in 2-3 sentences. If no answer exists in the text, say 'No clear information available.'"
            )

            # Generate reply through cognition engine using extraction_synthesis mode
            response_package = await self.cognition.generate_reply(
                message=synthesis_prompt,
                session_id="extract_and_answer_synthesis",
                streaming=False,
                mode="extraction_synthesis"
            )

            final_answer = response_package.content if hasattr(response_package, 'content') else str(response_package)

            if self.logger:
                self.logger.log_event("extract", "synthesis_completed", {
                    "query": query,
                    "answer_length": len(final_answer)
                })

            # Clean the answer
            cleaned_answer = self._clean_final_answer(final_answer)

            if self.logger:
                self.logger.log_event("extract", "final_answer_cleaned", {
                    "original_length": len(final_answer),
                    "cleaned_length": len(cleaned_answer)
                })

            return cleaned_answer

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "compose_answer_synthesis"})

            return "No clear information available."


    def _clean_final_answer(self, answer: str) -> str:
        """
        Clean and format the final answer.

        Args:
            answer: Raw answer from LLM

        Returns:
            Cleaned answer
        """
        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()

        # Remove markdown formatting if present
        answer = re.sub(r'\*\*', '', answer)
        answer = re.sub(r'__', '', answer)

        # Remove leading labels like "Answer:", "Response:", etc.
        answer = re.sub(r'^(Answer|Response|Result):\s*', '', answer, flags=re.IGNORECASE)

        # Ensure proper sentence ending
        if answer and not answer[-1] in '.!?':
            answer += '.'

        return answer


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
