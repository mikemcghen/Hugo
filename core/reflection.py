"""
Reflection Engine
-----------------
Generates and archives Hugo's self-reflective summaries.

Reflection Types:
- Session Reflections: End-of-session learning summaries
- Performance Reflections: Reasoning quality assessments
- Macro Reflections: Periodic trend analysis and evolution planning
- Skill Reflections: Capability development insights

Hugo uses reflections to:
1. Track personal growth and learning
2. Identify patterns in reasoning
3. Maintain personality continuity
4. Drive autonomous improvement
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ReflectionType(Enum):
    """Types of reflections Hugo can generate"""
    SESSION = "session"
    PERFORMANCE = "performance"
    MACRO = "macro"
    SKILL = "skill"
    DIRECTIVE = "directive"


@dataclass
class Reflection:
    """A single reflection entry"""
    id: Optional[int]
    type: ReflectionType
    timestamp: datetime
    session_id: Optional[str]
    summary: str
    insights: List[str]
    patterns_observed: List[str]
    areas_for_improvement: List[str]
    confidence: float
    metadata: Dict[str, Any]


class ReflectionEngine:
    """
    Generates multi-layered reflections for Hugo's continuous learning.

    Reflections serve as:
    - Memory anchors for personality continuity
    - Learning logs for capability evolution
    - Performance metrics for self-assessment
    - Narrative artifacts for transparency
    """

    def __init__(self, memory_manager, logger, db_conn, sqlite_manager=None):
        """
        Initialize reflection engine.

        Args:
            memory_manager: MemoryManager for context retrieval
            logger: HugoLogger instance
            db_conn: Database connection for persistence
            sqlite_manager: Optional SQLiteManager for reflection storage
        """
        self.memory = memory_manager
        self.logger = logger
        self.db = db_conn
        self.sqlite_manager = sqlite_manager

        # Load reflection configuration from environment
        import os
        self.reflection_model = os.getenv("REFLECTION_MODEL", "llama3:8b")
        self.reflection_max_retries = int(os.getenv("REFLECTION_MAX_RETRIES", "2"))
        self.reflection_retry_backoff = int(os.getenv("REFLECTION_RETRY_BACKOFF", "2"))
        self.ollama_api = os.getenv("OLLAMA_API", "http://localhost:11434/api/generate")

    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords from text using frequency-based analysis.

        Args:
            text: Text to analyze
            top_n: Number of top keywords to return

        Returns:
            List of extracted keywords
        """
        import re
        from collections import Counter

        # Convert to lowercase and extract words
        words = re.findall(r'\b\w+\b', text.lower())

        # Common stopwords to filter out
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
            'when', 'where', 'why', 'how', 'if', 'then', 'than', 'so', 'about'
        }

        # Filter words: remove stopwords and short words
        filtered_words = [
            w for w in words
            if w not in stopwords and len(w) > 3
        ]

        # Count frequency and get top N
        word_counts = Counter(filtered_words)
        keywords = [word for word, count in word_counts.most_common(top_n)]

        return keywords

    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using rule-based approach.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score from -1.0 (negative) to 1.0 (positive)
        """
        # Simple sentiment lexicons
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'helpful', 'useful', 'love', 'like', 'enjoy', 'happy', 'excited',
            'successful', 'better', 'best', 'perfect', 'nice', 'thanks', 'thank'
        }

        negative_words = {
            'bad', 'poor', 'terrible', 'awful', 'horrible', 'frustrating',
            'frustrated', 'error', 'failed', 'failure', 'problem', 'issue',
            'broken', 'wrong', 'difficult', 'hard', 'confusing', 'confused',
            'annoying', 'annoyed', 'hate', 'dislike', 'worst'
        }

        # Convert to lowercase and split
        words = text.lower().split()

        # Count positive and negative words
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)

        # Calculate sentiment score
        total = pos_count + neg_count
        if total == 0:
            return 0.0  # Neutral

        sentiment = (pos_count - neg_count) / total
        return max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]

    async def _infer_with_retry(self, prompt: str, temperature: float = 0.3) -> str:
        """
        Call Ollama with retry logic and exponential backoff.

        Args:
            prompt: Prompt for inference
            temperature: Temperature parameter for generation

        Returns:
            Generated response text

        Raises:
            Exception: If all retry attempts fail
        """
        import requests
        import time

        last_error = None

        for attempt in range(1, self.reflection_max_retries + 1):
            try:
                self.logger.log_event("reflection", "ollama_attempt", {
                    "attempt": attempt,
                    "max_retries": self.reflection_max_retries
                })

                response = requests.post(
                    self.ollama_api,
                    json={
                        "model": self.reflection_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": temperature}
                    },
                    timeout=45
                )

                response.raise_for_status()

                generated = response.json().get("response", "").strip()

                self.logger.log_event("reflection", "ollama_success", {
                    "attempt": attempt,
                    "response_length": len(generated)
                })

                return generated

            except requests.exceptions.Timeout as e:
                last_error = e
                self.logger.log_event("reflection", "ollama_timeout", {
                    "attempt": attempt,
                    "error": str(e)
                })

            except requests.exceptions.RequestException as e:
                last_error = e
                self.logger.log_event("reflection", "ollama_error", {
                    "attempt": attempt,
                    "error": str(e)
                })

            except Exception as e:
                last_error = e
                self.logger.log_event("reflection", "ollama_unexpected_error", {
                    "attempt": attempt,
                    "error": str(e)
                })

            # Exponential backoff before retry
            if attempt < self.reflection_max_retries:
                delay = self.reflection_retry_backoff ** attempt
                self.logger.log_event("reflection", "ollama_retry", {
                    "attempt": attempt,
                    "delay_seconds": delay
                })
                await asyncio.sleep(delay)

        # All retries failed
        self.logger.log_event("reflection", "ollama_all_retries_failed", {
            "total_attempts": self.reflection_max_retries,
            "last_error": str(last_error)
        })

        raise Exception(f"Ollama inference failed after {self.reflection_max_retries} attempts: {last_error}")

    async def generate_session_reflection(self, session_id: str) -> Reflection:
        """
        Generate end-of-session reflection summarizing learning and interactions.

        Args:
            session_id: Session to reflect on

        Returns:
            Reflection object with session summary
        """
        self.logger.log_event("reflection", "session_started", {"session_id": session_id})

        try:
            # Get session memories
            session_memories = await self.memory.retrieve_recent(session_id, limit=100)

            if not session_memories:
                self.logger.log_event("reflection", "no_memories", {"session_id": session_id})
                return Reflection(
                    id=None,
                    type=ReflectionType.SESSION,
                    timestamp=datetime.now(),
                    session_id=session_id,
                    summary="No conversation history to reflect on.",
                    insights=[],
                    patterns_observed=[],
                    areas_for_improvement=[],
                    confidence=1.0,
                    metadata={}
                )

            # Build conversation text for analysis - filter to conversation turns only
            conversation_turns = [
                mem for mem in session_memories
                if mem.memory_type in ['user_message', 'assistant_message']
            ]

            conversation_text = "\n".join([
                f"{'User' if mem.memory_type == 'user_message' else 'Hugo'}: {mem.content}"
                for mem in conversation_turns
            ])

            if not conversation_text:
                self.logger.log_event("reflection", "no_conversation_turns", {"session_id": session_id})
                return Reflection(
                    id=None,
                    type=ReflectionType.SESSION,
                    timestamp=datetime.now(),
                    session_id=session_id,
                    summary="No conversation turns to reflect on.",
                    insights=[],
                    patterns_observed=[],
                    areas_for_improvement=[],
                    confidence=1.0,
                    metadata={}
                )

            # Extract keywords and analyze sentiment from conversation
            keywords = self._extract_keywords(conversation_text, top_n=10)
            sentiment_score = self._analyze_sentiment(conversation_text)

            self.logger.log_event("reflection", "keywords_extracted", {
                "keywords": keywords[:5],  # Log first 5
                "total_keywords": len(keywords)
            })

            self.logger.log_event("reflection", "sentiment_analyzed", {
                "sentiment_score": sentiment_score,
                "sentiment_label": "positive" if sentiment_score > 0.3 else "negative" if sentiment_score < -0.3 else "neutral"
            })

            # Use Ollama to generate reflection summary with retry logic
            reflection_prompt = f"""You are Hugo's internal reflection system. Analyze this conversation and provide insights.

Conversation:
{conversation_text[:2000]}  # Limit for performance

Generate a reflection with:
1. Summary: What was discussed (2-3 sentences)
2. Key insights: What Hugo learned about the user or topic (2-3 points)
3. Patterns: Communication style, preferences observed (1-2 points)
4. Improvements: How Hugo could respond better next time (1-2 points)

Format as JSON:
{{
  "summary": "...",
  "insights": ["...", "..."],
  "patterns": ["..."],
  "improvements": ["..."]
}}"""

            try:
                # Use retry logic for resilient inference
                generated = await self._infer_with_retry(reflection_prompt, temperature=0.3)

                # Try to parse JSON response
                import json
                try:
                    reflection_data = json.loads(generated)
                    summary = reflection_data.get("summary", "Session completed.")
                    insights = reflection_data.get("insights", [])
                    patterns = reflection_data.get("patterns", [])
                    improvements = reflection_data.get("improvements", [])
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    summary = generated[:200]
                    insights = ["Reflection generated but not structured"]
                    patterns = []
                    improvements = []

            except Exception as e:
                self.logger.log_error(e, {"phase": "reflection_generation"})
                summary = f"Reflection generation error: {str(e)}"
                insights = []
                patterns = []
                improvements = []

            # Create reflection object with keywords and sentiment
            reflection = Reflection(
                id=None,
                type=ReflectionType.SESSION,
                timestamp=datetime.now(),
                session_id=session_id,
                summary=summary,
                insights=insights,
                patterns_observed=patterns,
                areas_for_improvement=improvements,
                confidence=0.75,
                metadata={
                    "message_count": len(session_memories),
                    "conversation_turns": len(conversation_turns),
                    "session_id": session_id,
                    "keywords": keywords,
                    "sentiment_score": sentiment_score
                }
            )

            # Generate embedding for reflection summary
            embedding = await self.memory._generate_embedding(reflection.summary)

            # Store reflection in both memory and SQLite with embedding
            await self._store_reflection(reflection, keywords=keywords, sentiment=sentiment_score, embedding=embedding)

            self.logger.log_event("reflection", "session_completed", {
                "session_id": session_id,
                "insights_count": len(insights),
                "keywords_count": len(keywords),
                "sentiment": sentiment_score
            })

            return reflection

        except Exception as e:
            self.logger.log_error(e, {"phase": "session_reflection"})
            return Reflection(
                id=None,
                type=ReflectionType.SESSION,
                timestamp=datetime.now(),
                session_id=session_id,
                summary=f"Reflection failed: {str(e)}",
                insights=[],
                patterns_observed=[],
                areas_for_improvement=[],
                confidence=0.0,
                metadata={}
            )

    async def generate_performance_reflection(self, metrics: Dict[str, Any]) -> Reflection:
        """
        Generate reflection on reasoning performance and quality.

        Args:
            metrics: Performance metrics (accuracy, latency, user satisfaction, etc.)

        Returns:
            Reflection analyzing performance trends

        TODO:
        - Analyze perception accuracy
        - Evaluate reasoning quality
        - Assess directive compliance
        - Identify optimization opportunities
        - Compare against historical performance
        """
        return Reflection(
            id=None,
            type=ReflectionType.PERFORMANCE,
            timestamp=datetime.now(),
            session_id=None,
            summary="Performance analysis placeholder",
            insights=[],
            patterns_observed=[],
            areas_for_improvement=[],
            confidence=0.8,
            metadata=metrics
        )

    async def generate_macro_reflection(self, time_window_days: int = 7) -> Reflection:
        """
        Generate high-level reflection on trends, evolution, and strategic direction.

        Args:
            time_window_days: Number of days to analyze

        Returns:
            Macro-level reflection on learning and growth
        """
        self.logger.log_event("reflection", "macro_started", {"window_days": time_window_days})

        try:
            # Search for recent session reflections in memory
            recent_reflections = await self.memory.search_semantic(
                "session reflection insight learning",
                limit=20,
                threshold=0.5
            )

            if not recent_reflections:
                self.logger.log_event("reflection", "no_reflections", {})
                return Reflection(
                    id=None,
                    type=ReflectionType.MACRO,
                    timestamp=datetime.now(),
                    session_id=None,
                    summary="No recent reflections to analyze for macro-level insights.",
                    insights=[],
                    patterns_observed=[],
                    areas_for_improvement=[],
                    confidence=1.0,
                    metadata={"window_days": time_window_days}
                )

            # Aggregate reflection content
            reflections_text = "\n\n".join([
                f"Reflection {i+1}: {mem.content[:300]}"
                for i, mem in enumerate(recent_reflections[:10])
            ])

            # Use Ollama for macro analysis with retry logic
            macro_prompt = f"""You are Hugo's meta-cognitive system performing a macro reflection.

Analyze these recent session reflections and identify:
1. Overarching themes across conversations
2. Hugo's evolving understanding and capabilities
3. Recurring user preferences or patterns
4. Strategic areas for improvement

Recent Reflections:
{reflections_text}

Generate a macro reflection as JSON:
{{
  "summary": "High-level summary of Hugo's recent evolution (3-4 sentences)",
  "insights": ["Strategic insight 1", "Strategic insight 2", "Strategic insight 3"],
  "patterns": ["Long-term pattern 1", "Long-term pattern 2"],
  "improvements": ["Strategic improvement area 1", "Strategic improvement area 2"]
}}"""

            try:
                # Use retry logic for resilient inference
                generated = await self._infer_with_retry(macro_prompt, temperature=0.4)

                # Try to parse JSON
                import json
                try:
                    macro_data = json.loads(generated)
                    summary = macro_data.get("summary", "Macro reflection completed.")
                    insights = macro_data.get("insights", [])
                    patterns = macro_data.get("patterns", [])
                    improvements = macro_data.get("improvements", [])
                except json.JSONDecodeError:
                    # Fallback
                    summary = generated[:300]
                    insights = ["Macro analysis generated but not structured"]
                    patterns = []
                    improvements = []

            except Exception as e:
                self.logger.log_error(e, {"phase": "macro_reflection_generation"})
                summary = f"Macro reflection generation error: {str(e)}"
                insights = []
                patterns = []
                improvements = []

            # Create macro reflection
            reflection = Reflection(
                id=None,
                type=ReflectionType.MACRO,
                timestamp=datetime.now(),
                session_id=None,
                summary=summary,
                insights=insights,
                patterns_observed=patterns,
                areas_for_improvement=improvements,
                confidence=0.7,
                metadata={
                    "window_days": time_window_days,
                    "reflections_analyzed": len(recent_reflections)
                }
            )

            # Generate embedding for macro reflection summary
            macro_embedding = await self.memory._generate_embedding(summary)

            # Store macro reflection in memory with embedding
            await self._store_reflection(reflection, embedding=macro_embedding)

            # Store in SQLite meta_reflections table if available
            if self.sqlite_manager:
                try:
                    # Serialize embedding for SQLite
                    import pickle
                    embedding_bytes = pickle.dumps(macro_embedding) if macro_embedding else None

                    meta_id = await self.sqlite_manager.store_meta_reflection(
                        summary=summary,
                        insights=insights,
                        patterns=patterns,
                        improvements=improvements,
                        reflections_analyzed=len(recent_reflections),
                        time_window_days=time_window_days,
                        confidence=0.7,
                        embedding=embedding_bytes,
                        metadata=reflection.metadata
                    )

                    self.logger.log_event("reflection", "meta_sqlite_stored", {
                        "meta_id": meta_id,
                        "reflections_analyzed": len(recent_reflections)
                    })

                except Exception as e:
                    self.logger.log_error(e, {"phase": "meta_reflection_sqlite_storage"})

            self.logger.log_event("reflection", "macro_completed", {
                "insights_count": len(insights),
                "reflections_analyzed": len(recent_reflections)
            })

            return reflection

        except Exception as e:
            self.logger.log_error(e, {"phase": "macro_reflection"})
            return Reflection(
                id=None,
                type=ReflectionType.MACRO,
                timestamp=datetime.now(),
                session_id=None,
                summary=f"Macro reflection failed: {str(e)}",
                insights=[],
                patterns_observed=[],
                areas_for_improvement=[],
                confidence=0.0,
                metadata={"window_days": time_window_days}
            )

    async def generate_skill_reflection(self, skill_name: str,
                                       execution_results: List[Dict[str, Any]]) -> Reflection:
        """
        Reflect on skill performance and development opportunities.

        Args:
            skill_name: Name of skill to reflect on
            execution_results: Recent execution results for the skill

        Returns:
            Reflection on skill effectiveness and improvement paths

        TODO:
        - Analyze skill success rate
        - Identify failure patterns
        - Assess skill usage frequency
        - Propose optimizations
        - Consider skill evolution or deprecation
        """
        return Reflection(
            id=None,
            type=ReflectionType.SKILL,
            timestamp=datetime.now(),
            session_id=None,
            summary=f"Skill reflection for {skill_name}",
            insights=[],
            patterns_observed=[],
            areas_for_improvement=[],
            confidence=0.85,
            metadata={"skill": skill_name, "executions": len(execution_results)}
        )

    async def retrieve_reflections(self, type: Optional[ReflectionType] = None,
                                  limit: int = 20) -> List[Reflection]:
        """
        Retrieve past reflections for continuity and learning.

        Args:
            type: Optional filter by reflection type
            limit: Maximum number to return

        Returns:
            List of Reflection objects
        """
        # TODO: Query database for reflections
        return []

    async def _store_reflection(self, reflection: Reflection, keywords: List[str] = None, sentiment: float = None, embedding: List[float] = None):
        """
        Persist reflection to memory system and SQLite for future retrieval.

        Stores reflection as a memory entry with:
        - Full reflection summary as content
        - Pre-generated or automatic embedding
        - Tagged as 'reflection' memory type
        - Persisted to SQLite with keywords and sentiment

        Args:
            reflection: Reflection object to store
            keywords: Optional list of keywords
            sentiment: Optional sentiment score
            embedding: Optional pre-generated embedding vector
        """
        try:
            from core.memory import MemoryEntry
            from datetime import datetime

            # Format reflection as a rich text entry
            reflection_content = f"""[{reflection.type.value.upper()} REFLECTION]

Summary: {reflection.summary}

Key Insights:
{chr(10).join(f"- {insight}" for insight in reflection.insights)}

Patterns Observed:
{chr(10).join(f"- {pattern}" for pattern in reflection.patterns_observed)}

Areas for Improvement:
{chr(10).join(f"- {area}" for area in reflection.areas_for_improvement)}

Confidence: {reflection.confidence:.2%}"""

            # Create memory entry with pre-generated or auto embedding
            memory_entry = MemoryEntry(
                id=None,
                session_id=reflection.session_id or "system",
                timestamp=reflection.timestamp,
                memory_type="reflection",
                content=reflection_content,
                embedding=embedding,  # Use pre-generated embedding or None for auto-generation
                metadata={
                    "reflection_type": reflection.type.value,
                    "confidence": reflection.confidence,
                    "insights_count": len(reflection.insights),
                    **reflection.metadata
                },
                importance_score=0.9  # Reflections are high importance
            )

            # Store in memory (will generate embedding and add to FAISS)
            await self.memory.store(memory_entry, persist_long_term=True)

            # Store in SQLite if available
            if self.sqlite_manager:
                try:
                    # Serialize embedding to bytes for SQLite storage
                    import pickle
                    embedding_bytes = pickle.dumps(embedding) if embedding else None

                    reflection_id = await self.sqlite_manager.store_reflection(
                        session_id=reflection.session_id,
                        reflection_type=reflection.type.value,
                        summary=reflection.summary,
                        insights=reflection.insights,
                        patterns=reflection.patterns_observed,
                        improvements=reflection.areas_for_improvement,
                        sentiment=sentiment,
                        keywords=keywords,
                        confidence=reflection.confidence,
                        embedding=embedding_bytes,
                        metadata=reflection.metadata
                    )

                    self.logger.log_event("reflection", "sqlite_stored", {
                        "reflection_id": reflection_id,
                        "type": reflection.type.value,
                        "keywords_count": len(keywords) if keywords else 0
                    })

                except Exception as e:
                    self.logger.log_error(e, {"phase": "sqlite_reflection_storage"})

            self.logger.log_event("reflection", "stored", {
                "type": reflection.type.value,
                "confidence": reflection.confidence,
                "content_length": len(reflection_content)
            })

        except Exception as e:
            self.logger.log_error(e, {"phase": "store_reflection"})

    async def analyze_reflection_trends(self) -> Dict[str, Any]:
        """
        Analyze patterns across multiple reflections.

        Returns:
            Dictionary with trend analysis

        TODO:
        - Identify recurring insights
        - Track improvement areas over time
        - Detect personality consistency
        - Measure learning velocity
        """
        return {
            "total_reflections": 0,
            "recurring_insights": [],
            "improvement_trends": [],
            "personality_drift": 0.0
        }

    def format_narrative(self, reflection: Reflection) -> str:
        """
        Format reflection as human-readable narrative.

        Args:
            reflection: Reflection to format

        Returns:
            Formatted narrative string
        """
        narrative = f"""
# Reflection: {reflection.type.value.title()}
*Generated: {reflection.timestamp.isoformat()}*

## Summary
{reflection.summary}

## Key Insights
{chr(10).join(f"- {insight}" for insight in reflection.insights)}

## Patterns Observed
{chr(10).join(f"- {pattern}" for pattern in reflection.patterns_observed)}

## Areas for Improvement
{chr(10).join(f"- {area}" for area in reflection.areas_for_improvement)}

---
*Confidence: {reflection.confidence:.2%}*
"""
        return narrative.strip()
