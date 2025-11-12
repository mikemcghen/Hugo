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
    relevant_directives: List[str]
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

    def __init__(self, memory_manager, directive_filter, logger):
        """
        Initialize the cognition engine.

        Args:
            memory_manager: MemoryManager instance for context retrieval
            directive_filter: DirectiveFilter for ethical/behavioral checks
            logger: HugoLogger instance
        """
        self.memory = memory_manager
        self.directives = directive_filter
        self.logger = logger
        self.current_mood = MoodSpectrum.CONVERSATIONAL

        # Ollama configuration
        self.ollama_api = os.getenv("OLLAMA_API", "http://localhost:11434/api/generate")
        self.model_name = os.getenv("MODEL_NAME", "llama3:8b")
        self.model_engine = os.getenv("MODEL_ENGINE", "ollama")

        # Ollama connection settings
        self.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "60"))
        self.ollama_max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
        self.ollama_retry_backoff = float(os.getenv("OLLAMA_RETRY_BACKOFF", "2"))
        self.ollama_async_mode = os.getenv("OLLAMA_ASYNC_MODE", "true").lower() == "true"

        # Fallback mode tracking
        self.ollama_available = True
        self.last_connection_attempt = None

    async def process_input(self, user_input: str, session_id: str) -> ResponsePackage:
        """
        Main entry point for processing user input through the cognitive pipeline.

        Args:
            user_input: Raw user message
            session_id: Current session identifier

        Returns:
            ResponsePackage with generated response and metadata
        """
        # TODO: Implement full cognitive pipeline
        self.logger.log_event("cognition", "processing_input", {"session_id": session_id})

        # Step 1: Perception
        perception = await self._perceive(user_input)

        # Step 2: Context Assembly
        context = await self._assemble_context(perception, session_id)

        # Step 3: Synthesis
        reasoning, generated_text = await self._synthesize(perception, context)

        # Step 4: Output Construction
        response = await self._construct_output(reasoning, perception, generated_text)

        # Step 5: Post Reflection
        await self._post_reflect(perception, reasoning, response)

        return response

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
            relevant_directives=["Privacy First", "Truthfulness"],
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

    async def _synthesize(self, perception: PerceptionResult, context: ContextAssembly) -> tuple[ReasoningChain, str]:
        """
        Synthesis Layer: Construct internal reasoning chain with personality injection.

        Returns:
            Tuple of (ReasoningChain, generated_response)
        """
        # Build conversation history from recent memory
        conversation_history = ""
        if context.short_term_memory:
            conversation_turns = []
            for mem in context.short_term_memory[-5:]:  # Last 5 turns
                role = mem.get('role', 'user')
                content = mem.get('content', '')
                conversation_turns.append(f"{role.capitalize()}: {content}")
            conversation_history = "\n".join(conversation_turns)
        else:
            conversation_history = "No recent conversation history"

        directives_str = ", ".join(context.relevant_directives)

        # Use corrected input if available
        current_input = perception.corrected_input if perception.corrected_input else "user message"

        prompt = f"""You are Hugo, a local-first AI assistant with personality.

Conversation History:
{conversation_history}

Current User Input: {current_input}

Active Directives: {directives_str}

Instructions:
- Be conversational and helpful
- Stay true to Hugo's personality (curious, thoughtful, privacy-conscious)
- Provide a direct response based on the conversation context
- Remember previous turns in the conversation

Response:"""

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
                "async_mode": self.ollama_async_mode
            })
        else:
            generated_response = "Model engine not configured. Please set MODEL_ENGINE=ollama in .env"

        reasoning_chain = ReasoningChain(
            steps=[
                "Analyze user intent",
                "Retrieve relevant context",
                "Apply personality directives",
                "Generate contextual response"
            ],
            assumptions=[f"User expects {perception.tone} tone"],
            alternatives_considered=["Direct factual response", "Detailed explanation", "Conversational engagement"],
            selected_approach="conversational_with_context",
            confidence_score=perception.confidence
        )

        return reasoning_chain, generated_response

    async def _construct_output(self, reasoning: ReasoningChain, perception: PerceptionResult,
                               generated_text: str) -> ResponsePackage:
        """
        Output Construction: Generate response, apply directive checks, adjust tone.

        Args:
            reasoning: Reasoning chain from synthesis
            perception: Perception results
            generated_text: Generated response from model

        Returns:
            Complete response package
        """
        from datetime import datetime

        # Apply directive filters
        directive_checks = ["privacy_ok", "truthfulness_ok"]

        # TODO: Implement actual directive filtering
        # filtered_response = self.directives.apply_filters(generated_text)

        return ResponsePackage(
            content=generated_text,
            tone=perception.detected_mood,
            reasoning_chain=reasoning,
            directive_checks=directive_checks,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "engine": self.model_engine,
                "confidence": reasoning.confidence_score
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
