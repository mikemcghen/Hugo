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

    def __init__(self, memory_manager, logger, db_conn):
        """
        Initialize reflection engine.

        Args:
            memory_manager: MemoryManager for context retrieval
            logger: HugoLogger instance
            db_conn: Database connection for persistence
        """
        self.memory = memory_manager
        self.logger = logger
        self.db = db_conn

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

            # Use Ollama to generate reflection summary
            from core.cognition import CognitionEngine
            import os
            import requests

            ollama_api = os.getenv("OLLAMA_API", "http://localhost:11434/api/generate")
            model_name = os.getenv("MODEL_NAME", "llama3:8b")

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
                response = requests.post(
                    ollama_api,
                    json={
                        "model": model_name,
                        "prompt": reflection_prompt,
                        "stream": False,
                        "options": {"temperature": 0.3}  # Lower for more consistent structured output
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    generated = response.json().get("response", "").strip()

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
                else:
                    summary = "Ollama unavailable for reflection"
                    insights = []
                    patterns = []
                    improvements = []

            except Exception as e:
                self.logger.log_error(e, {"phase": "reflection_generation"})
                summary = f"Reflection generation error: {str(e)}"
                insights = []
                patterns = []
                improvements = []

            # Create reflection object
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
                    "session_id": session_id
                }
            )

            # Store reflection as a memory
            await self._store_reflection(reflection)

            self.logger.log_event("reflection", "session_completed", {
                "session_id": session_id,
                "insights_count": len(insights)
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

            # Use Ollama for macro analysis
            import os
            import requests

            ollama_api = os.getenv("OLLAMA_API", "http://localhost:11434/api/generate")
            model_name = os.getenv("MODEL_NAME", "llama3:8b")

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
                response = requests.post(
                    ollama_api,
                    json={
                        "model": model_name,
                        "prompt": macro_prompt,
                        "stream": False,
                        "options": {"temperature": 0.4}
                    },
                    timeout=45
                )

                if response.status_code == 200:
                    generated = response.json().get("response", "").strip()

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
                else:
                    summary = "Ollama unavailable for macro reflection"
                    insights = []
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

            # Store macro reflection
            await self._store_reflection(reflection)

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

    async def _store_reflection(self, reflection: Reflection):
        """
        Persist reflection to memory system for future retrieval.

        Stores reflection as a memory entry with:
        - Full reflection summary as content
        - Automatic embedding generation
        - Tagged as 'reflection' memory type
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

            # Create memory entry
            memory_entry = MemoryEntry(
                id=None,
                session_id=reflection.session_id or "system",
                timestamp=reflection.timestamp,
                memory_type="reflection",
                content=reflection_content,
                embedding=None,  # Will be auto-generated by MemoryManager
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
