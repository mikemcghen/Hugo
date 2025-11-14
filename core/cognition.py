"""
Cognition Engine
----------------
Implements Hugo's multi-layered reasoning pipeline:
- Perception Layer: Intent recognition and emotional context mapping
- Context Assembly: Memory retrieval and directive filtering
- Synthesis Layer: Internal reasoning chain construction
- Output Construction: Response generation with tone adjustment
- Post Reflection: Performance evaluation and heuristic updates
"""

import asyncio
import os
import re
import time
import requests
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Optional async support
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Load environment variables
load_dotenv()


class MoodSpectrum(Enum):
    """Hugo's adaptive mood states"""
    FOCUSED = "focused"
    REFLECTIVE = "reflective"
    CONVERSATIONAL = "conversational"
    OPERATIONAL = "operational"
    LOW_POWER = "low_power"


@dataclass
class PerceptionResult:
    """Results from the perception layer"""
    user_intent: str
    tone: str
    emotional_context: Dict[str, Any]
    detected_mood: MoodSpectrum
    confidence: float
    corrected_input: Optional[str] = None


@dataclass
class ContextAssembly:
    """Assembled context for reasoning"""
    short_term_memory: List[Dict[str, Any]]
    long_term_memory: List[Dict[str, Any]]
    active_tasks: List[Dict[str, Any]]
    session_state: Dict[str, Any]


@dataclass
class ReasoningChain:
    """Internal reasoning process"""
    steps: List[str]
    assumptions: List[str]
    alternatives_considered: List[str]
    selected_approach: str
    confidence_score: float


@dataclass
class ResponsePackage:
    """Complete response with metadata"""
    content: str
    tone: MoodSpectrum
    reasoning_chain: ReasoningChain
    directive_checks: List[str]
    metadata: Dict[str, Any]


class CognitionEngine:
    """
    Core reasoning engine implementing Hugo's cognitive architecture.

    This engine orchestrates the full perception → reasoning → response pipeline,
    maintaining personality consistency while adapting to context.
    """

    def __init__(self, memory_manager, logger, runtime_manager=None):
        """
        Initialize the cognition engine.

        Args:
            memory_manager: MemoryManager instance for context retrieval
            logger: HugoLogger instance
            runtime_manager: Optional RuntimeManager for worker agent delegation
        """
        self.memory = memory_manager
        self.logger = logger
        self.current_mood = MoodSpectrum.CONVERSATIONAL
        self.runtime_manager = runtime_manager

        # Ollama configuration
        self.ollama_api = os.getenv("OLLAMA_API", "http://localhost:11434/api/generate")
        self.model_name = os.getenv("MODEL_NAME", "llama3:8b")
        self.model_engine = os.getenv("MODEL_ENGINE", "ollama")

        # Ollama connection settings
        self.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "60"))
        self.ollama_max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
        self.ollama_retry_backoff = float(os.getenv("OLLAMA_RETRY_BACKOFF", "2"))
        self.ollama_async_mode = os.getenv("OLLAMA_ASYNC_MODE", "true").lower() == "true"

        # Agent delegation settings
        self.agent_delegation_enabled = os.getenv("AGENT_DELEGATION_ENABLED", "true").lower() == "true"

        # Fallback mode tracking
        self.ollama_available = True
        self.last_connection_attempt = None

        # Worker agent (lazy initialization)
        self._worker_agent = None

        # Load Hugo's personality manifest
        self.persona = self._load_persona()
        self.logger.log_event("cognition", "persona_loaded", {
            "name": self.persona.get("name", "Hugo"),
            "role": self.persona.get("identity", {}).get("role", "Unknown"),
            "mood": self.current_mood.value,
            "agent_delegation": self.agent_delegation_enabled
        })

    def _load_persona(self) -> Dict[str, Any]:
        """
        Load Hugo's personality manifest from YAML configuration.

        Returns:
            Dictionary containing persona data (identity, traits, directives, etc.)
        """
        try:
            manifest_path = Path("configs/hugo_manifest.yaml")
            if not manifest_path.exists():
                self.logger.log_event("cognition", "persona_load_failed", {
                    "reason": "manifest_not_found",
                    "path": str(manifest_path)
                })
                return self._default_persona()

            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # Extract manifest section
            manifest = data.get("manifest", {})
            identity = data.get("identity", {})
            personality = data.get("personality", {})
            mood_spectrum = data.get("mood_spectrum", {})
            directives = data.get("directives", {})

            return {
                "name": manifest.get("name", "Hugo"),
                "codename": manifest.get("codename", "The Right Hand"),
                "identity": identity,
                "personality": personality,
                "mood_spectrum": mood_spectrum,
                "directives": directives,
                "overview": manifest.get("overview", "")
            }

        except Exception as e:
            self.logger.log_error(e, {"phase": "persona_loading"})
            return self._default_persona()

    def _default_persona(self) -> Dict[str, Any]:
        """
        Return default persona if manifest loading fails.

        Returns:
            Minimal persona dictionary
        """
        return {
            "name": "Hugo",
            "codename": "The Right Hand",
            "identity": {
                "role": "Right Hand / Second in Command",
                "core_traits": ["Loyal", "Reflective", "Analytical"]
            },
            "personality": {
                "communication_style": ["Conversational and pragmatic"]
            },
            "directives": {
                "core_ethics": ["Privacy First", "Truthfulness", "Transparency"]
            },
            "mood_spectrum": {}
        }

    async def generate_reply(self, message: str, session_id: str, streaming: bool = False):
        """
        Main public API for generating replies.

        This is the primary entry point for the REPL and other clients.
        Handles both streaming and non-streaming modes.

        Args:
            message: User message
            session_id: Current session identifier
            streaming: If True, yield chunks; if False, return complete response

        Returns:
            If streaming=False: ResponsePackage
            If streaming=True: Generator yielding chunks, then ResponsePackage
        """
        self.logger.log_event("cognition", "generate_reply_started", {
            "session_id": session_id,
            "streaming": streaming,
            "message_length": len(message)
        })

        # Save user message to memory first
        await self._save_user_message(message, session_id)

        if streaming:
            # Return streaming generator
            return self.process_input_streaming(message, session_id)
        else:
            # Return complete response
            response_package = await self.process_input(message, session_id)

            # Post-process: Save assistant response to memory
            await self.post_process(response_package.content, session_id)

            return response_package

    async def _save_user_message(self, message: str, session_id: str):
        """
        Save user message to memory before processing.

        Args:
            message: User message
            session_id: Session identifier
        """
        try:
            from core.memory import MemoryEntry
            from datetime import datetime

            user_entry = MemoryEntry(
                id=None,
                session_id=session_id,
                timestamp=datetime.now(),
                memory_type="user_message",
                content=message,
                embedding=None,
                metadata={},
                importance_score=0.5,
                is_fact=False
            )

            await self.memory.store(user_entry, persist_long_term=False)
            self.logger.log_event("cognition", "user_message_saved", {
                "session_id": session_id,
                "content_length": len(message)
            })

        except Exception as e:
            self.logger.log_error(e, {"phase": "save_user_message"})

    async def process_input(self, user_input: str, session_id: str) -> ResponsePackage:
        """
        Main entry point for processing user input through the cognitive pipeline.

        Includes agent delegation for heavy technical work.

        Args:
            user_input: Raw user message
            session_id: Current session identifier

        Returns:
            ResponsePackage with generated response and metadata
        """
        self.logger.log_event("cognition", "processing_input", {"session_id": session_id})

        # Check if this requires agent delegation
        if self._detect_heavy_work(user_input):
            return await self._delegate_to_worker(user_input, session_id)

        # Standard cognitive pipeline
        # Step 1: Perception
        perception = await self._perceive(user_input)

        # Step 2: Context Assembly
        context = await self._assemble_context(perception, session_id)

        # Step 3: Synthesis
        reasoning, generated_text, prompt_metadata = await self._synthesize(perception, context, session_id, user_input)

        # Step 4: Output Construction
        response = await self._construct_output(reasoning, perception, generated_text, prompt_metadata)

        # Step 5: Post Reflection
        await self._post_reflect(perception, reasoning, response)

        return response

    def _detect_heavy_work(self, user_input: str) -> bool:
        """
        Detect if user input requires heavy technical work (agent delegation).

        Returns True if the input contains keywords suggesting:
        - Implementation tasks
        - Refactoring requests
        - Feature development
        - Complex technical analysis

        Args:
            user_input: User message

        Returns:
            bool: True if heavy work detected
        """
        if not self.agent_delegation_enabled:
            return False

        heavy_work_keywords = [
            "implement", "refactor", "add feature", "create feature",
            "build", "develop", "design system", "architect",
            "write code", "generate code", "create function",
            "add class", "modify", "update code", "fix bug",
            "optimize", "improve performance", "add test"
        ]

        user_input_lower = user_input.lower()

        for keyword in heavy_work_keywords:
            if keyword in user_input_lower:
                self.logger.log_event("cognition", "heavy_work_detected", {
                    "keyword": keyword,
                    "input_preview": user_input[:100]
                })
                return True

        return False

    @property
    def worker_agent(self):
        """Lazy initialization of worker agent"""
        if self._worker_agent is None and self.runtime_manager:
            try:
                from agents.worker_agent import WorkerAgent
                self._worker_agent = WorkerAgent(self.runtime_manager)
            except ImportError as e:
                self.logger.log_error(e, {"phase": "worker_agent_initialization"})
        return self._worker_agent

    async def _delegate_to_worker(self, user_input: str, session_id: str) -> ResponsePackage:
        """
        Delegate heavy technical work to the worker agent.

        Args:
            user_input: User message
            session_id: Current session ID

        Returns:
            ResponsePackage with worker agent's response
        """
        self.logger.log_event("cognition", "delegating_to_worker", {
            "session_id": session_id,
            "input_length": len(user_input)
        })

        try:
            # Get worker agent
            if self.worker_agent is None:
                # Fallback: Worker agent not available, process normally
                self.logger.log_event("cognition", "worker_delegation_failed", {
                    "reason": "worker_agent_not_available"
                })
                return await self.process_input(user_input, session_id)

            # Delegate to worker agent
            result = await self.worker_agent.run_task(user_input, context={
                "session_id": session_id,
                "delegated_from": "cognition_engine"
            })

            # Build response package
            from datetime import datetime

            reasoning_chain = ReasoningChain(
                steps=[
                    "Detected heavy technical work",
                    "Delegated to worker agent",
                    "Worker agent completed task"
                ],
                assumptions=["Task requires focused technical attention"],
                alternatives_considered=["Direct response", "Worker delegation"],
                selected_approach="worker_delegation",
                confidence_score=0.85
            )

            response_package = ResponsePackage(
                content=result,
                tone=MoodSpectrum.OPERATIONAL,
                reasoning_chain=reasoning_chain,
                directive_checks=["worker_delegation_ok"],
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model_name,
                    "engine": "worker_agent",
                    "delegated": True,
                    "session_id": session_id
                }
            )

            self.logger.log_event("cognition", "worker_delegation_complete", {
                "session_id": session_id,
                "result_length": len(result)
            })

            return response_package

        except Exception as e:
            self.logger.log_error(e, {
                "phase": "worker_delegation",
                "session_id": session_id
            })

            # Fallback to normal processing
            return await self.process_input(user_input, session_id)

    async def process_input_streaming(self, user_input: str, session_id: str):
        """
        Stream-enabled input processing for real-time response generation.

        This method follows the same cognitive pipeline as process_input,
        but yields response chunks as they're generated rather than waiting
        for the full response.

        Args:
            user_input: Raw user message
            session_id: Current session identifier

        Yields:
            str: Response chunks as they're generated
            ResponsePackage: Final response package with metadata (last yield)
        """
        self.logger.log_event("cognition", "processing_input_streaming", {"session_id": session_id})

        # Step 1: Perception
        perception = await self._perceive(user_input)

        # Step 2: Context Assembly
        context = await self._assemble_context(perception, session_id)

        # Step 3: Assemble prompt
        user_message = perception.corrected_input if perception.corrected_input else user_input
        prompt_data = await self.assemble_prompt(user_message, perception, context, session_id)
        prompt = prompt_data["prompt"]
        prompt_metadata = prompt_data["metadata"]

        # Step 4: Stream generation
        generated_chunks = []

        for chunk in self.stream_local_infer(prompt, temperature=0.7):
            generated_chunks.append(chunk)
            yield chunk  # Yield each chunk to REPL

        # Combine full response
        generated_response = "".join(generated_chunks)

        # Build reasoning chain
        reasoning_steps = [
            f"Detected user sentiment: {prompt_metadata.get('user_sentiment', {}).get('primary_sentiment', 'neutral')}",
            f"Retrieved {prompt_metadata.get('conversation_turns', 0)} conversation turns",
            f"Retrieved {prompt_metadata.get('factual_memories', 0)} factual memories",
            f"Retrieved {prompt_metadata.get('reflection_insights', 0)} reflection insights",
            f"Retrieved {prompt_metadata.get('semantic_memories', 0)} semantic memories",
            f"Applied persona: {prompt_metadata.get('persona_name', 'Hugo')} in {prompt_metadata.get('mood', 'conversational')} mood",
            f"Adjusted tone: {prompt_metadata.get('tone_adjustment', 'conversational')}"
        ]

        reasoning_chain = ReasoningChain(
            steps=reasoning_steps,
            assumptions=[
                f"User expects {perception.tone} tone",
                f"User sentiment: {prompt_metadata.get('user_sentiment', {}).get('primary_sentiment', 'neutral')}"
            ],
            alternatives_considered=["Direct factual response", "Detailed explanation", "Conversational engagement"],
            selected_approach=f"persona_driven_{prompt_metadata.get('mood', 'conversational')}",
            confidence_score=perception.confidence
        )

        # Construct final response package
        from datetime import datetime

        response = ResponsePackage(
            content=generated_response,
            tone=perception.detected_mood,
            reasoning_chain=reasoning_chain,
            directive_checks=[],
            metadata={
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "engine": self.model_engine,
                "confidence": reasoning_chain.confidence_score,
                "persona_name": prompt_metadata.get("persona_name", "Hugo"),
                "mood": prompt_metadata.get("mood", "conversational"),
                "user_sentiment": prompt_metadata.get("user_sentiment", {}).get("primary_sentiment", "neutral"),
                "tone_adjustment": prompt_metadata.get("tone_adjustment", "conversational"),
                "conversation_turns": prompt_metadata.get("conversation_turns", 0),
                "factual_memories": prompt_metadata.get("factual_memories", 0),
                "reflection_insights": prompt_metadata.get("reflection_insights", 0),
                "semantic_memories": prompt_metadata.get("semantic_memories", 0),
                "prompt_tokens": prompt_metadata.get("prompt_tokens", 0),
                "streaming": True
            }
        )

        # Post-process: Save assistant response to memory
        await self.post_process(generated_response, session_id)

        # Post reflection
        await self._post_reflect(perception, reasoning_chain, response)

        # Yield final response package for metadata storage
        yield response

    async def post_process(self, response_text: str, session_id: str):
        """
        Post-process the response: save to memory, update conversation state.

        This ensures every assistant response is persisted to SQLite and FAISS
        for future recall and reflection.

        Args:
            response_text: Generated response text
            session_id: Current session identifier
        """
        try:
            from core.memory import MemoryEntry
            from datetime import datetime

            # Create memory entry for assistant response
            assistant_entry = MemoryEntry(
                id=None,
                session_id=session_id,
                timestamp=datetime.now(),
                memory_type="assistant_response",
                content=response_text,
                embedding=None,  # Will be generated by memory manager
                metadata={
                    "model": self.model_name,
                    "engine": self.model_engine,
                    "persona_name": self.persona.get("name", "Hugo"),
                    "mood": self.current_mood.value
                },
                importance_score=0.5,
                is_fact=False
            )

            # Store in memory (SQLite + FAISS)
            await self.memory.store(assistant_entry, persist_long_term=True)

            self.logger.log_event("cognition", "response_saved_to_memory", {
                "session_id": session_id,
                "content_length": len(response_text),
                "model": self.model_name,
                "mood": self.current_mood.value
            })

        except Exception as e:
            self.logger.log_error(e, {
                "phase": "post_process",
                "session_id": session_id
            })

    def _detect_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Detect user sentiment using keyword matching and pattern analysis.

        Args:
            text: User input text

        Returns:
            Dictionary with sentiment analysis (primary_sentiment, intensity, keywords)
        """
        # Sentiment keyword patterns
        sentiment_patterns = {
            "frustrated": ["frustrated", "annoying", "annoyed", "irritating", "stuck", "confusing", "broken"],
            "excited": ["excited", "amazing", "awesome", "love", "fantastic", "great", "wonderful"],
            "urgent": ["urgent", "asap", "quickly", "hurry", "immediately", "critical", "emergency"],
            "curious": ["how", "why", "what", "tell me", "explain", "curious", "wondering"],
            "grateful": ["thanks", "thank you", "appreciate", "helpful", "grateful"],
            "concerned": ["worried", "concerned", "anxious", "nervous", "uncertain"]
        }

        text_lower = text.lower()
        detected = []

        for sentiment, keywords in sentiment_patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected.append(sentiment)
                    break

        # Determine primary sentiment
        primary = detected[0] if detected else "neutral"
        intensity = len(detected) / 3.0  # Normalize 0-1

        return {
            "primary_sentiment": primary,
            "intensity": min(intensity, 1.0),
            "detected_sentiments": detected,
            "is_neutral": len(detected) == 0
        }

    async def assemble_prompt(self, user_message: str, perception: PerceptionResult,
                             context: ContextAssembly, session_id: str) -> Dict[str, Any]:
        """
        Assemble a persona-driven contextual prompt for inference.

        Args:
            user_message: Corrected user input
            perception: Perception analysis results
            context: Assembled context from memory
            session_id: Current session ID

        Returns:
            Dictionary containing:
              - prompt: Formatted prompt string
              - metadata: Context metadata (memories_used, sentiment, tone, etc.)
        """
        # Detect user sentiment
        sentiment = self._detect_sentiment(user_message)

        # Determine Hugo's tone based on sentiment and current mood
        tone_adjustment = self._adjust_tone(sentiment, perception.detected_mood)

        # Retrieve factual memories, semantic memory, and reflections
        factual_memories = []
        semantic_context = []
        reflection_insights = []

        try:
            # Get factual memories about the user
            if hasattr(self.memory, 'get_factual_memories'):
                factual_entries = await self.memory.get_factual_memories(limit=10)
                for fact in factual_entries:
                    factual_memories.append({
                        "content": fact.content,
                        "entity_type": fact.entity_type,
                        "importance": fact.importance_score
                    })

            # Get reflection insights from reflection system
            if hasattr(self, 'runtime_manager') and self.runtime_manager:
                if hasattr(self.runtime_manager, 'reflection'):
                    reflection_insights = await self.runtime_manager.reflection.get_reflection_insights(limit=5)

            # Search for relevant semantic memories
            if hasattr(self.memory, 'search_semantic'):
                semantic_results = await self.memory.search_semantic(
                    user_message,
                    limit=5,
                    threshold=0.6
                )

                # Filter for non-reflection, non-factual memories
                for mem in semantic_results:
                    if mem.memory_type != "reflection" and not mem.is_fact:
                        semantic_context.append(mem.content[:150])

                # Limit to top 3
                semantic_context = semantic_context[:3]

        except Exception as e:
            self.logger.log_error(e, {"phase": "memory_retrieval"})

        # Build conversation history
        conversation_turns = []
        if context.short_term_memory:
            for mem in context.short_term_memory[-5:]:
                role = mem.get('role', 'user')
                content = mem.get('content', '')
                conversation_turns.append(f"{role.capitalize()}: {content}")

        # Extract persona details
        identity = self.persona.get("identity", {})
        personality = self.persona.get("personality", {})

        persona_name = self.persona.get("name", "Hugo")
        persona_role = identity.get("role", "Assistant")
        core_traits = ", ".join(identity.get("core_traits", ["Helpful"]))
        persona_desc = identity.get("persona_description", "I am a helpful AI assistant.")

        # Build mood description
        mood_spectrum = self.persona.get("mood_spectrum", {})
        current_mood_desc = mood_spectrum.get(self.current_mood.value, "Engaged and helpful")

        # Assemble the prompt
        prompt_parts = [
            f"[Persona: {persona_name} — {persona_role}]",
            f"[Core Traits: {core_traits}]",
            f"[Current Mood: {self.current_mood.value.title()} - {current_mood_desc}]",
            "",
            f"{persona_desc}",
            "",
            "[Memory Policy]",
            "CRITICAL: When responding about user information or past conversations:",
            "- If a memory exists, use it EXACTLY as written",
            "- If no memory exists, say 'I'm not certain' rather than guessing",
            "- NEVER fabricate or invent facts about the user",
            "- Only reference information from the sections below",
            ""
        ]

        # Add factual memories about the user
        if factual_memories:
            prompt_parts.append("[Known Facts About the User]")
            for i, fact in enumerate(factual_memories, 1):
                entity_label = f"[{fact['entity_type']}]" if fact['entity_type'] else ""
                prompt_parts.append(f"{i}. {entity_label} {fact['content']}")
            prompt_parts.append("")

        # Add long-term reflection insights
        if reflection_insights:
            prompt_parts.append("[Long-Term Reflections Summary]")
            for i, refl in enumerate(reflection_insights, 1):
                prompt_parts.append(f"\nReflection {i}:")
                prompt_parts.append(f"  Summary: {refl['summary']}")
                if refl.get('insights'):
                    prompt_parts.append(f"  Key Insights: {', '.join(refl['insights'][:3])}")
                if refl.get('keywords'):
                    prompt_parts.append(f"  Keywords: {', '.join(refl['keywords'][:5])}")
            prompt_parts.append("")

        # Add conversation history
        if conversation_turns:
            prompt_parts.append("[Recent Conversation]")
            prompt_parts.extend(conversation_turns)
            prompt_parts.append("")

        # Add semantic context if available
        if semantic_context:
            prompt_parts.append("[Relevant Context from Memory]")
            for i, ctx in enumerate(semantic_context, 1):
                prompt_parts.append(f"{i}. {ctx}...")
            prompt_parts.append("")

        # Add user sentiment context
        if not sentiment["is_neutral"]:
            prompt_parts.append(f"[User Sentiment: {sentiment['primary_sentiment'].title()}]")
            prompt_parts.append(f"[Suggested Tone: {tone_adjustment}]")
            prompt_parts.append("")

        # Add current user input
        prompt_parts.append(f"User: {user_message}")
        prompt_parts.append(f"{persona_name}:")

        prompt = "\n".join(prompt_parts)

        # Log prompt assembly
        self.logger.log_event("cognition", "prompt_assembled", {
            "session_id": session_id,
            "persona_name": persona_name,
            "mood": self.current_mood.value,
            "conversation_turns": len(conversation_turns),
            "factual_memories": len(factual_memories),
            "reflection_insights": len(reflection_insights),
            "semantic_memories": len(semantic_context),
            "user_sentiment": sentiment["primary_sentiment"],
            "tone_adjustment": tone_adjustment,
            "prompt_length": len(prompt)
        })

        return {
            "prompt": prompt,
            "metadata": {
                "persona_name": persona_name,
                "mood": self.current_mood.value,
                "conversation_turns": len(conversation_turns),
                "factual_memories": len(factual_memories),
                "reflection_insights": len(reflection_insights),
                "semantic_memories": len(semantic_context),
                "user_sentiment": sentiment,
                "tone_adjustment": tone_adjustment,
                "prompt_tokens": len(prompt.split())
            }
        }

    def _adjust_tone(self, sentiment: Dict[str, Any], detected_mood: MoodSpectrum) -> str:
        """
        Adjust Hugo's response tone based on user sentiment and current mood.

        Args:
            sentiment: Detected user sentiment
            detected_mood: Current detected mood

        Returns:
            Tone adjustment description
        """
        primary = sentiment["primary_sentiment"]

        tone_map = {
            "frustrated": "Calm, patient, and solution-oriented",
            "excited": "Upbeat and enthusiastically engaged",
            "urgent": "Direct, focused, and efficient",
            "curious": "Thoughtful and exploratory",
            "grateful": "Warm and appreciative",
            "concerned": "Reassuring and supportive",
            "neutral": "Balanced and conversational"
        }

        base_tone = tone_map.get(primary, "Balanced and conversational")

        # Modify based on Hugo's current mood
        if detected_mood == MoodSpectrum.FOCUSED:
            return f"{base_tone}, with precision"
        elif detected_mood == MoodSpectrum.REFLECTIVE:
            return f"{base_tone}, with depth"
        elif detected_mood == MoodSpectrum.LOW_POWER:
            return f"{base_tone}, with gentleness"

        return base_tone

    async def build_prompt(self, user_message: str, session_id: str,
                          include_facts: bool = True,
                          include_reflections: bool = True,
                          include_conversation: bool = True) -> str:
        """
        Build a complete prompt with persona, memories, and context.

        This is a public wrapper around the internal prompt assembly logic,
        useful for testing, debugging, or custom prompt generation.

        Args:
            user_message: User message to respond to
            session_id: Current session ID
            include_facts: Include factual memories (default: True)
            include_reflections: Include reflection insights (default: True)
            include_conversation: Include recent conversation (default: True)

        Returns:
            Complete formatted prompt string
        """
        # Use internal perception and context assembly
        perception = await self._perceive(user_message)
        context = await self._assemble_context(perception, session_id)

        # Assemble full prompt
        prompt_data = await self.assemble_prompt(user_message, perception, context, session_id)

        self.logger.log_event("cognition", "prompt_built", {
            "session_id": session_id,
            "prompt_length": len(prompt_data["prompt"]),
            "include_facts": include_facts,
            "include_reflections": include_reflections,
            "include_conversation": include_conversation
        })

        return prompt_data["prompt"]

    async def retrieve_relevant_memories(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Retrieve relevant memories for a given query.

        Returns factual memories, semantic search results, and reflection insights.

        Args:
            query: Search query
            limit: Maximum results per category

        Returns:
            Dictionary with:
              - factual_memories: List of factual memory entries
              - semantic_results: List of semantically similar memories
              - reflections: List of reflection insights
        """
        result = {
            "factual_memories": [],
            "semantic_results": [],
            "reflections": []
        }

        try:
            # Get factual memories
            if hasattr(self.memory, 'get_factual_memories'):
                facts = await self.memory.get_factual_memories(limit=limit)
                result["factual_memories"] = [
                    {
                        "content": fact.content,
                        "entity_type": fact.entity_type,
                        "importance": fact.importance_score
                    }
                    for fact in facts
                ]

            # Get semantic search results
            if hasattr(self.memory, 'search_semantic'):
                semantic_results = await self.memory.search_semantic(
                    query,
                    limit=limit,
                    threshold=0.5
                )
                result["semantic_results"] = [
                    {
                        "content": mem.content,
                        "memory_type": mem.memory_type,
                        "importance": mem.importance_score,
                        "is_fact": mem.is_fact
                    }
                    for mem in semantic_results
                ]

            # Get reflection insights
            if hasattr(self, 'runtime_manager') and self.runtime_manager:
                if hasattr(self.runtime_manager, 'reflection'):
                    reflections = await self.runtime_manager.reflection.get_reflection_insights(limit=5)
                    result["reflections"] = reflections

            self.logger.log_event("cognition", "memories_retrieved", {
                "query": query[:50],
                "factual_count": len(result["factual_memories"]),
                "semantic_count": len(result["semantic_results"]),
                "reflection_count": len(result["reflections"])
            })

        except Exception as e:
            self.logger.log_error(e, {"phase": "retrieve_relevant_memories"})

        return result

    async def call_ollama(self, prompt: str, streaming: bool = False, temperature: float = 0.7):
        """
        Call Ollama API directly with a prompt.

        Public wrapper for direct model inference, useful for testing
        or custom inference scenarios.

        Args:
            prompt: Input prompt
            streaming: If True, return generator; if False, return complete response
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            If streaming=False: Complete response string
            If streaming=True: Generator yielding chunks
        """
        self.logger.log_event("cognition", "ollama_direct_call", {
            "streaming": streaming,
            "temperature": temperature,
            "prompt_length": len(prompt)
        })

        if streaming:
            # Return streaming generator
            return self.stream_local_infer(prompt, temperature=temperature)
        else:
            # Return complete response
            if self.ollama_async_mode:
                return await self._local_infer_async(prompt, temperature=temperature)
            else:
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    self._local_infer,
                    prompt,
                    temperature
                )

    async def _perceive(self, user_input: str) -> PerceptionResult:
        """
        Perception Layer: Recognize intent, tone, and emotional context.

        TODO:
        - Implement NLP-based intent classification
        - Add tone analysis (formal, casual, urgent, etc.)
        - Map emotional signals to mood spectrum
        - Calculate confidence scores
        """
        # Simple typo autocorrect (common low-edit-distance fixes)
        corrections = {
            "squre": "square",
            "recieve": "receive",
            "definately": "definitely",
            "teh": "the",
            "adress": "address",
            "occured": "occurred",
            "seperate": "separate",
            "wierd": "weird",
            "untill": "until",
            "basicly": "basically"
        }

        corrected_input = user_input
        for wrong, right in corrections.items():
            corrected_input = re.sub(rf"\b{wrong}\b", right, corrected_input, flags=re.IGNORECASE)

        # Log corrections if any were made
        if corrected_input != user_input:
            self.logger.log_event("cognition", "typo_correction", {
                "original": user_input,
                "corrected": corrected_input
            })

        # Placeholder implementation
        return PerceptionResult(
            user_intent="general_query",
            tone="conversational",
            emotional_context={},
            detected_mood=MoodSpectrum.CONVERSATIONAL,
            confidence=0.85,
            corrected_input=corrected_input
        )

    async def _assemble_context(self, perception: PerceptionResult, session_id: str) -> ContextAssembly:
        """
        Context Assembly: Retrieve relevant memories and apply directive filters.

        TODO:
        - Query short-term memory for recent context
        - Search long-term memory using vector similarity
        - Load relevant directives based on intent
        - Fetch active tasks and session state
        """
        # Retrieve recent conversation history
        short_term_memory = []
        try:
            recent_memories = await self.memory.retrieve_recent(session_id, limit=10)
            # Format memories as conversation turns
            for mem in recent_memories:
                role = "user" if mem.memory_type == "user_message" else "assistant"
                short_term_memory.append({
                    "role": role,
                    "content": mem.content,
                    "timestamp": mem.timestamp.isoformat() if hasattr(mem.timestamp, 'isoformat') else str(mem.timestamp)
                })

            self.logger.log_event("cognition", "context_assembled", {
                "session_id": session_id,
                "memory_count": len(short_term_memory)
            })
        except Exception as e:
            self.logger.log_error(e, {"phase": "context_assembly"})

        return ContextAssembly(
            short_term_memory=short_term_memory,
            long_term_memory=[],
            active_tasks=[],
            session_state={}
        )

    def _local_infer(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Perform local inference using Ollama API with retry logic and fallback.

        Features:
        - Configurable timeout and retry attempts
        - Exponential backoff on failures
        - Enhanced logging for each attempt
        - Graceful fallback mode when Ollama is unavailable

        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated response text or fallback message
        """
        attempt = 0
        last_error = None

        while attempt < self.ollama_max_retries:
            attempt += 1
            start_time = time.time()

            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature
                    }
                }

                self.logger.log_event("cognition", "ollama_inference_attempt", {
                    "attempt": attempt,
                    "max_retries": self.ollama_max_retries,
                    "timeout": self.ollama_timeout
                })

                response = requests.post(
                    self.ollama_api,
                    json=payload,
                    timeout=self.ollama_timeout
                )
                response.raise_for_status()

                result = response.json()
                generated_text = result.get("response", "").strip()

                duration = time.time() - start_time
                self.logger.log_event("cognition", "ollama_inference", {
                    "attempt": attempt,
                    "duration": round(duration, 2),
                    "status": "success",
                    "response_length": len(generated_text)
                })

                # Mark Ollama as available
                self.ollama_available = True
                self.last_connection_attempt = time.time()

                return generated_text

            except requests.exceptions.ReadTimeout as e:
                duration = time.time() - start_time
                last_error = e
                self.logger.log_event("cognition", "ollama_inference", {
                    "attempt": attempt,
                    "duration": round(duration, 2),
                    "status": "timeout",
                    "error": str(e)
                })

                if attempt < self.ollama_max_retries:
                    backoff_time = self.ollama_retry_backoff ** attempt
                    self.logger.log_event("cognition", "ollama_retry", {
                        "attempt": attempt,
                        "backoff_seconds": backoff_time
                    })
                    time.sleep(backoff_time)

            except requests.exceptions.ConnectionError as e:
                duration = time.time() - start_time
                last_error = e
                self.logger.log_event("cognition", "ollama_inference", {
                    "attempt": attempt,
                    "duration": round(duration, 2),
                    "status": "connection_error",
                    "error": str(e)
                })

                if attempt < self.ollama_max_retries:
                    backoff_time = self.ollama_retry_backoff ** attempt
                    self.logger.log_event("cognition", "ollama_retry", {
                        "attempt": attempt,
                        "backoff_seconds": backoff_time
                    })
                    time.sleep(backoff_time)

            except requests.exceptions.RequestException as e:
                duration = time.time() - start_time
                last_error = e
                self.logger.log_event("cognition", "ollama_inference", {
                    "attempt": attempt,
                    "duration": round(duration, 2),
                    "status": "request_error",
                    "error": str(e)
                })

                if attempt < self.ollama_max_retries:
                    backoff_time = self.ollama_retry_backoff ** attempt
                    time.sleep(backoff_time)

            except Exception as e:
                duration = time.time() - start_time
                last_error = e
                self.logger.log_error(e, {
                    "phase": "ollama_inference",
                    "attempt": attempt,
                    "duration": round(duration, 2)
                })

                if attempt < self.ollama_max_retries:
                    backoff_time = self.ollama_retry_backoff ** attempt
                    time.sleep(backoff_time)

        # All retries exhausted - enter fallback mode
        self.ollama_available = False
        self.last_connection_attempt = time.time()

        self.logger.log_event("cognition", "ollama_fallback_mode", {
            "total_attempts": attempt,
            "last_error": str(last_error) if last_error else "Unknown"
        })

        return self._fallback_response(prompt)

    def _fallback_response(self, prompt: str) -> str:
        """
        Generate a graceful fallback response when Ollama is unavailable.

        Args:
            prompt: The original prompt (for context)

        Returns:
            A reflective acknowledgment message
        """
        fallback_messages = [
            "I'm having trouble connecting to my reasoning core. Let's pause for a moment.",
            "My reasoning system seems to be taking a break. Could you try again in a moment?",
            "I'm experiencing some difficulty accessing my core processes right now.",
            "Connection to my inference engine is temporarily unavailable. Please give me a moment."
        ]

        # Simple rotation based on timestamp
        index = int(time.time()) % len(fallback_messages)
        return fallback_messages[index]

    async def _local_infer_async(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Async version of local inference using aiohttp for non-blocking operation.

        Features:
        - Non-blocking HTTP requests using aiohttp
        - Same retry logic and fallback as synchronous version
        - Maintains REPL responsiveness during inference
        - Falls back to synchronous version if aiohttp not available

        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated response text or fallback message
        """
        # Fallback to synchronous if aiohttp not available
        if not AIOHTTP_AVAILABLE:
            self.logger.log_event("cognition", "async_fallback_sync", {
                "reason": "aiohttp_not_available"
            })
            # Run synchronous version in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._local_infer, prompt, temperature)

        attempt = 0
        last_error = None

        while attempt < self.ollama_max_retries:
            attempt += 1
            start_time = time.time()

            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature
                    }
                }

                self.logger.log_event("cognition", "ollama_inference_attempt_async", {
                    "attempt": attempt,
                    "max_retries": self.ollama_max_retries,
                    "timeout": self.ollama_timeout
                })

                timeout = aiohttp.ClientTimeout(total=self.ollama_timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(self.ollama_api, json=payload) as response:
                        response.raise_for_status()
                        result = await response.json()
                        generated_text = result.get("response", "").strip()

                        duration = time.time() - start_time
                        self.logger.log_event("cognition", "ollama_inference_async", {
                            "attempt": attempt,
                            "duration": round(duration, 2),
                            "status": "success",
                            "response_length": len(generated_text)
                        })

                        # Mark Ollama as available
                        self.ollama_available = True
                        self.last_connection_attempt = time.time()

                        return generated_text

            except asyncio.TimeoutError as e:
                duration = time.time() - start_time
                last_error = e
                self.logger.log_event("cognition", "ollama_inference_async", {
                    "attempt": attempt,
                    "duration": round(duration, 2),
                    "status": "timeout",
                    "error": str(e)
                })

                if attempt < self.ollama_max_retries:
                    backoff_time = self.ollama_retry_backoff ** attempt
                    self.logger.log_event("cognition", "ollama_retry_async", {
                        "attempt": attempt,
                        "backoff_seconds": backoff_time
                    })
                    await asyncio.sleep(backoff_time)

            except aiohttp.ClientError as e:
                duration = time.time() - start_time
                last_error = e
                self.logger.log_event("cognition", "ollama_inference_async", {
                    "attempt": attempt,
                    "duration": round(duration, 2),
                    "status": "client_error",
                    "error": str(e)
                })

                if attempt < self.ollama_max_retries:
                    backoff_time = self.ollama_retry_backoff ** attempt
                    self.logger.log_event("cognition", "ollama_retry_async", {
                        "attempt": attempt,
                        "backoff_seconds": backoff_time
                    })
                    await asyncio.sleep(backoff_time)

            except Exception as e:
                duration = time.time() - start_time
                last_error = e
                self.logger.log_error(e, {
                    "phase": "ollama_inference_async",
                    "attempt": attempt,
                    "duration": round(duration, 2)
                })

                if attempt < self.ollama_max_retries:
                    backoff_time = self.ollama_retry_backoff ** attempt
                    await asyncio.sleep(backoff_time)

        # All retries exhausted - enter fallback mode
        self.ollama_available = False
        self.last_connection_attempt = time.time()

        self.logger.log_event("cognition", "ollama_fallback_mode_async", {
            "total_attempts": attempt,
            "last_error": str(last_error) if last_error else "Unknown"
        })

        return self._fallback_response(prompt)

    def stream_local_infer(self, prompt: str, temperature: float = 0.7):
        """
        Perform streaming local inference using Ollama API.

        This generator yields text chunks as they arrive from the model,
        enabling real-time token-by-token display in the REPL.

        Features:
        - Yields chunks as they arrive from Ollama
        - Maintains retry logic for connection failures
        - Logs streaming start/completion events
        - Falls back gracefully if streaming fails

        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature (0.0-1.0)

        Yields:
            str: Text chunks as they arrive from Ollama
        """
        attempt = 0
        last_error = None

        while attempt < self.ollama_max_retries:
            attempt += 1
            start_time = time.time()

            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": True,  # Enable streaming
                    "options": {
                        "temperature": temperature
                    }
                }

                self.logger.log_event("cognition", "ollama_streaming_attempt", {
                    "attempt": attempt,
                    "max_retries": self.ollama_max_retries,
                    "timeout": self.ollama_timeout
                })

                response = requests.post(
                    self.ollama_api,
                    json=payload,
                    stream=True,
                    timeout=self.ollama_timeout
                )
                response.raise_for_status()

                # Track total generated text
                total_generated = []

                # Stream chunks line by line
                for line in response.iter_lines():
                    if line:
                        try:
                            import json
                            chunk_data = json.loads(line.decode('utf-8'))

                            # Extract response chunk
                            chunk_text = chunk_data.get("response", "")
                            if chunk_text:
                                total_generated.append(chunk_text)
                                yield chunk_text

                            # Check if done
                            if chunk_data.get("done", False):
                                break

                        except json.JSONDecodeError:
                            continue

                duration = time.time() - start_time
                full_response = "".join(total_generated)

                self.logger.log_event("cognition", "ollama_streaming_complete", {
                    "attempt": attempt,
                    "duration": round(duration, 2),
                    "status": "success",
                    "response_length": len(full_response),
                    "chunks": len(total_generated)
                })

                # Mark Ollama as available
                self.ollama_available = True
                self.last_connection_attempt = time.time()

                return  # Successful completion

            except requests.exceptions.ReadTimeout as e:
                duration = time.time() - start_time
                last_error = e
                self.logger.log_event("cognition", "ollama_streaming", {
                    "attempt": attempt,
                    "duration": round(duration, 2),
                    "status": "timeout",
                    "error": str(e)
                })

                if attempt < self.ollama_max_retries:
                    backoff_time = self.ollama_retry_backoff ** attempt
                    self.logger.log_event("cognition", "ollama_retry", {
                        "attempt": attempt,
                        "backoff_seconds": backoff_time
                    })
                    time.sleep(backoff_time)

            except requests.exceptions.ConnectionError as e:
                duration = time.time() - start_time
                last_error = e
                self.logger.log_event("cognition", "ollama_streaming", {
                    "attempt": attempt,
                    "duration": round(duration, 2),
                    "status": "connection_error",
                    "error": str(e)
                })

                if attempt < self.ollama_max_retries:
                    backoff_time = self.ollama_retry_backoff ** attempt
                    time.sleep(backoff_time)

            except Exception as e:
                duration = time.time() - start_time
                last_error = e
                self.logger.log_error(e, {
                    "phase": "ollama_streaming",
                    "attempt": attempt,
                    "duration": round(duration, 2)
                })

                if attempt < self.ollama_max_retries:
                    backoff_time = self.ollama_retry_backoff ** attempt
                    time.sleep(backoff_time)

        # All retries exhausted - yield fallback
        self.ollama_available = False
        self.last_connection_attempt = time.time()

        self.logger.log_event("cognition", "ollama_streaming_fallback", {
            "total_attempts": attempt,
            "last_error": str(last_error) if last_error else "Unknown"
        })

        yield self._fallback_response(prompt)

    async def _synthesize(self, perception: PerceptionResult, context: ContextAssembly, session_id: str, user_input: str) -> tuple[ReasoningChain, str, Dict[str, Any]]:
        """
        Synthesis Layer: Construct internal reasoning chain with personality injection.

        Args:
            perception: Perception analysis results
            context: Assembled context from memory
            session_id: Current session identifier
            user_input: Original user input

        Returns:
            Tuple of (ReasoningChain, generated_response, prompt_metadata)
        """
        # Assemble persona-driven contextual prompt
        user_message = perception.corrected_input if perception.corrected_input else user_input
        prompt_data = await self.assemble_prompt(user_message, perception, context, session_id)
        prompt = prompt_data["prompt"]
        prompt_metadata = prompt_data["metadata"]

        # Generate response using local model
        if self.model_engine == "ollama":
            # Use async inference if enabled and available
            if self.ollama_async_mode:
                generated_response = await self._local_infer_async(prompt, temperature=0.7)
            else:
                generated_response = self._local_infer(prompt, temperature=0.7)

            self.logger.log_event("cognition", "ollama_inference_complete", {
                "response_length": len(generated_response),
                "response_preview": generated_response[:100] + "..." if len(generated_response) > 100 else generated_response,
                "async_mode": self.ollama_async_mode,
                "prompt_tokens": prompt_metadata.get("prompt_tokens", 0),
                "persona_name": prompt_metadata.get("persona_name", "Unknown")
            })
        else:
            generated_response = "Model engine not configured. Please set MODEL_ENGINE=ollama in .env"

        # Build reasoning chain with persona context
        reasoning_steps = [
            f"Detected user sentiment: {prompt_metadata.get('user_sentiment', {}).get('primary_sentiment', 'neutral')}",
            f"Retrieved {prompt_metadata.get('conversation_turns', 0)} conversation turns",
            f"Retrieved {prompt_metadata.get('factual_memories', 0)} factual memories",
            f"Retrieved {prompt_metadata.get('reflection_insights', 0)} reflection insights",
            f"Retrieved {prompt_metadata.get('semantic_memories', 0)} semantic memories",
            f"Applied persona: {prompt_metadata.get('persona_name', 'Hugo')} in {prompt_metadata.get('mood', 'conversational')} mood",
            f"Adjusted tone: {prompt_metadata.get('tone_adjustment', 'conversational')}"
        ]

        reasoning_chain = ReasoningChain(
            steps=reasoning_steps,
            assumptions=[
                f"User expects {perception.tone} tone",
                f"User sentiment: {prompt_metadata.get('user_sentiment', {}).get('primary_sentiment', 'neutral')}"
            ],
            alternatives_considered=["Direct factual response", "Detailed explanation", "Conversational engagement"],
            selected_approach=f"persona_driven_{prompt_metadata.get('mood', 'conversational')}",
            confidence_score=perception.confidence
        )

        return reasoning_chain, generated_response, prompt_metadata

    async def _construct_output(self, reasoning: ReasoningChain, perception: PerceptionResult,
                               generated_text: str, prompt_metadata: Dict[str, Any]) -> ResponsePackage:
        """
        Output Construction: Generate response, apply directive checks, adjust tone.

        Args:
            reasoning: Reasoning chain from synthesis
            perception: Perception results
            generated_text: Generated response from model
            prompt_metadata: Metadata from prompt assembly

        Returns:
            Complete response package with enriched metadata
        """
        from datetime import datetime

        return ResponsePackage(
            content=generated_text,
            tone=perception.detected_mood,
            reasoning_chain=reasoning,
            directive_checks=[],
            metadata={
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "engine": self.model_engine,
                "confidence": reasoning.confidence_score,
                "persona_name": prompt_metadata.get("persona_name", "Hugo"),
                "mood": prompt_metadata.get("mood", "conversational"),
                "user_sentiment": prompt_metadata.get("user_sentiment", {}).get("primary_sentiment", "neutral"),
                "tone_adjustment": prompt_metadata.get("tone_adjustment", "conversational"),
                "conversation_turns": prompt_metadata.get("conversation_turns", 0),
                "factual_memories": prompt_metadata.get("factual_memories", 0),
                "reflection_insights": prompt_metadata.get("reflection_insights", 0),
                "semantic_memories": prompt_metadata.get("semantic_memories", 0),
                "prompt_tokens": prompt_metadata.get("prompt_tokens", 0)
            }
        )

    async def _post_reflect(self, perception: PerceptionResult, reasoning: ReasoningChain, response: ResponsePackage):
        """
        Post Reflection: Evaluate performance and log for future learning.

        TODO:
        - Assess reasoning quality
        - Log successful patterns
        - Identify areas for improvement
        - Update heuristics if needed
        - Trigger macro reflection if patterns emerge
        """
        await self.logger.log_reflection({
            "perception_confidence": perception.confidence,
            "reasoning_confidence": reasoning.confidence_score,
            "mood": response.tone.value
        })

    def set_mood(self, mood: MoodSpectrum):
        """Manually set Hugo's current mood/operational mode"""
        self.current_mood = mood
        self.logger.log_event("cognition", "mood_change", {"new_mood": mood.value})

    async def macro_reflect(self):
        """
        Periodic macro reflection on reasoning patterns and performance.

        TODO:
        - Analyze trends in perception accuracy
        - Identify frequently used reasoning patterns
        - Detect opportunities for optimization
        - Generate self-improvement proposals
        """
        self.logger.log_event("cognition", "macro_reflection_started", {})
        # Placeholder for macro reflection logic
        pass
