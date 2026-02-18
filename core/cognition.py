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

# Ollama stability module
from core.ollama_stability import OllamaStabilityManager

# Persona engine
from core.persona_engine import HugoPersonaEngine, PersonaContext, Domain

# Configuration helper
from core.config import is_core_mode, is_full_mode, get_hugo_mode, HugoMode

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

        # Intent parser + action router (lazy initialization, FULL mode only)
        self._intent_parser = None
        self._action_router = None
        # Pending confirmation: stores a ParsedIntent awaiting user yes/no
        self._pending_confirmation = None
        # Delegation agent for compound/multi-step infrastructure requests
        self._delegation_agent = None

        # Load Hugo's personality manifest
        self.persona = self._load_persona()

        # Load Jarvis mode configuration
        self.jarvis_mode_enabled = self.persona.get("jarvis_mode", {}).get("enabled", False)
        self.jarvis_config = self.persona.get("jarvis_mode", {})

        # Initialize Ollama stability manager
        self.ollama_stability = OllamaStabilityManager(
            api_url=self.ollama_api,
            model_name=self.model_name,
            logger=self.logger,
            max_retries=self.ollama_max_retries
        )

        # Initialize Persona Engine
        self.persona_engine = HugoPersonaEngine(jarvis_mode=self.jarvis_mode_enabled)

        self.logger.log_event("cognition", "persona_loaded", {
            "name": self.persona.get("name", "Hugo"),
            "role": self.persona.get("identity", {}).get("role", "Unknown"),
            "mood": self.current_mood.value,
            "agent_delegation": self.agent_delegation_enabled,
            "jarvis_mode": self.jarvis_mode_enabled,
            "ollama_stability": True,
            "persona_engine": True
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

    # ── Infrastructure helpers (FULL mode only) ────────────────────────────

    def _is_compound_request(self, message: str, intent) -> bool:
        """
        Heuristic: does this message describe a multi-step infrastructure task?

        Compound requests are routed to DelegationAgent instead of ActionRouter
        so they can be broken into parallel subtasks.
        """
        lowered = message.lower()
        compound_signals = [
            " and then ", " after that ", " then ", " also ",
            " and restart", " and check", " and list", " and stop",
            " and start", " and run",
            "check all", "for each", "all containers", "all hosts",
            "everything that", "any that are", "ones that are",
        ]
        return any(sig in lowered for sig in compound_signals)

    async def _handle_memory_intent(self, intent, session_id: str) -> str:
        """Handle a memory-domain intent directly using Hugo's memory system."""
        try:
            action = intent.action or "search"
            query = (
                intent.parameters.get("query")
                or intent.parameters.get("topic")
                or intent.target
                or ""
            )

            if action in ("search", "recall") and query:
                memories = await self.memory.search_memories(query, k=5)
                if memories:
                    items = "\n".join(
                        f"- {m.get('content', str(m))}" for m in memories
                    )
                    return f"Here's what I remember about '{query}':\n{items}"
                return f"I don't have any memories matching '{query}'."

            elif action == "list_recent":
                memories = await self.memory.get_all_factual_memories(limit=10)
                if memories:
                    items = "\n".join(
                        f"- {m.get('content', str(m))}" for m in memories[:10]
                    )
                    return f"Recent memories:\n{items}"
                return "No recent memories found."

            elif action == "forget" and query:
                try:
                    memory_id = int(query)
                    deleted = await self.memory.delete_memory(memory_id)
                    return "Memory removed." if deleted else f"No memory found with ID {memory_id}."
                except ValueError:
                    return f"Please provide a numeric memory ID to forget. Got: '{query}'"

            else:
                return (
                    f"Memory action '{action}' not supported directly. "
                    "Try: 'recall <topic>', 'what do you remember about <topic>', or 'list recent memories'."
                )

        except Exception as e:
            self.logger.log_event("cognition", "memory_intent_error", {
                "error": str(e), "session_id": session_id
            })
            return f"I had trouble accessing memory: {str(e)}"

    async def _handle_compound_request(self, message: str, session_id: str) -> str:
        """Route a compound (multi-step) infrastructure request to DelegationAgent."""
        try:
            if self._delegation_agent is None:
                from agents.delegation_agent import DelegationAgent
                self._delegation_agent = DelegationAgent(
                    ollama_url=self.ollama_api,
                    model=self.model_name,
                )

            from agents.base_agent import AgentTask
            task = AgentTask(
                type="delegate",
                description=message,
                context={"session_id": session_id, "source": "cognition_compound"},
            )

            # DelegationAgent.execute() is synchronous — run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self._delegation_agent.execute(task)
            )

            if result.success and result.output:
                return str(result.output)
            elif result.errors:
                return f"Compound task partially failed: {'; '.join(result.errors)}"
            else:
                return "Compound task completed."

        except Exception as e:
            self.logger.log_event("cognition", "compound_request_error", {
                "error": str(e), "session_id": session_id
            })
            return f"I had trouble coordinating that multi-step request: {str(e)}"

    async def generate_reply(self, message: str, session_id: str, streaming: bool = False, mode: str = None):
        """
        Main public API for generating replies.

        This is the primary entry point for the REPL and other clients.
        ALWAYS returns an async iterator for uniform interface.

        Args:
            message: User message
            session_id: Current session identifier
            streaming: If True, yield chunks; if False, yield single response
            mode: Optional processing mode override (e.g., "extraction_synthesis")

        Returns:
            AsyncIterator that yields:
            - If streaming=True: string chunks, then final ResponsePackage
            - If streaming=False: single ResponsePackage
        """
        # Import async helper
        from runtime.utils.async_helpers import stream_single

        # Early bypass for extraction synthesis mode
        if mode == "extraction_synthesis":
            result = await self._generate_extraction_synthesis(message)
            return stream_single(result)

        # Core mode delegation: minimal, stable pipeline
        if is_core_mode():
            self.logger.log_event("cognition", "core_mode_delegation", {
                "session_id": session_id,
                "streaming": streaming
            })

            # Save user message to memory first
            await self._save_user_message(message, session_id)

            if streaming:
                # Return streaming generator directly
                return self._generate_core_reply_streaming(message, session_id)
            else:
                # Get complete response and wrap in async iterator
                response_text = await self._generate_core_reply_nonstreaming(message, session_id)

                # Save assistant response to memory
                await self.post_process(response_text, session_id)

                # Wrap in ResponsePackage and stream
                response_package = ResponsePackage(
                    content=response_text,
                    tone=MoodSpectrum.CONVERSATIONAL,
                    reasoning_chain=ReasoningChain(
                        steps=["Core mode response"],
                        assumptions=[],
                        alternatives_considered=[],
                        selected_approach="direct_response",
                        confidence_score=0.9
                    ),
                    directive_checks=[],
                    metadata={
                        "mode": "core",
                        "session_id": session_id,
                        "streaming": False
                    }
                )
                return stream_single(response_package)

        self.logger.log_event("cognition", "generate_reply_started", {
            "session_id": session_id,
            "streaming": streaming,
            "message_length": len(message)
        })

        # Classify message to check for skill triggers (internet queries, notes, etc.)
        classification = self.memory.classify_memory(message)

        # Check if this is an internet query or skill trigger that should bypass LLM
        if classification.metadata and "skill_trigger" in classification.metadata:
            skill_name = classification.metadata["skill_trigger"]
            skill_action = classification.metadata.get("skill_action", "help")
            skill_payload = classification.metadata.get("skill_payload", {})

            # Internet queries (web_search, fetch_url) bypass LLM entirely
            if skill_name in ["web_search", "fetch_url"]:
                self.logger.log_event("cognition", "internet_query_detected", {
                    "skill": skill_name,
                    "action": skill_action,
                    "session_id": session_id
                })

                # Execute skill directly
                response_package = await self._execute_skill_bypass(
                    skill_name, skill_action, skill_payload, message, session_id
                )

                # Save assistant response to memory
                await self.post_process(response_package.content, session_id)

                # Wrap in async iterator for uniform interface
                self.logger.log_event("cognition", "skill_bypass_wrapped_in_stream", {
                    "skill": skill_name,
                    "streaming": False
                })
                return stream_single(response_package)

        # ── FULL MODE: pending confirmation resolution ─────────────────────────
        # If the user is responding to a confirmation prompt (yes/no), resolve it.
        if self._pending_confirmation is not None:
            lowered = message.strip().lower()
            if lowered in ("yes", "y", "confirm", "do it", "go ahead", "ok", "sure"):
                pending = self._pending_confirmation
                self._pending_confirmation = None
                from core.actions.action_router import ActionRouter
                if self._action_router is None:
                    self._action_router = ActionRouter(logger=self.logger)
                # Re-route with permission bypass (user confirmed)
                from core.actions.permission import PermissionLevel
                exec_result = await self._action_router.route.__func__(
                    self._action_router, pending
                ) if False else None  # placeholder — use direct executor below

                # Execute directly on executor, bypassing gate
                from core.executors.executor_registry import ExecutorRegistry
                import core.executors.ssh    # noqa
                import core.executors.docker  # noqa
                import core.executors.monitor  # noqa
                executor = ExecutorRegistry.get_executor(pending.domain)
                params = dict(pending.parameters or {})
                if pending.target:
                    if pending.domain == "docker":
                        params.setdefault("container", pending.target)
                    elif pending.domain in ("ssh", "monitor"):
                        params.setdefault("host", pending.target)
                exec_res = await executor.execute_async(pending.action, **params)

                from core.actions.action_result import ActionResult
                ar = ActionResult(
                    success=exec_res.success,
                    data=exec_res.data,
                    error=exec_res.error,
                    domain=pending.domain,
                    action=pending.action,
                )
                response_text = self._action_router.format_for_cognition(ar)
                await self.post_process(response_text, session_id)
                response_package = ResponsePackage(
                    content=response_text,
                    tone=MoodSpectrum.OPERATIONAL,
                    reasoning_chain=ReasoningChain(
                        steps=["Confirmation received", "Executing confirmed action"],
                        assumptions=["User confirmed action"],
                        alternatives_considered=[],
                        selected_approach="confirmed_execution",
                        confidence_score=1.0
                    ),
                    directive_checks=["user_confirmed"],
                    metadata={"mode": "confirmed_intent", "session_id": session_id}
                )
                return stream_single(response_package)

            elif lowered in ("no", "n", "cancel", "abort", "stop", "nope", "nah"):
                self._pending_confirmation = None
                response_package = ResponsePackage(
                    content="Cancelled.",
                    tone=MoodSpectrum.CONVERSATIONAL,
                    reasoning_chain=ReasoningChain(
                        steps=["User cancelled"], assumptions=[], alternatives_considered=[],
                        selected_approach="cancel", confidence_score=1.0
                    ),
                    directive_checks=[],
                    metadata={"mode": "cancelled", "session_id": session_id}
                )
                return stream_single(response_package)
            else:
                # Not a confirmation — clear pending and fall through to normal pipeline
                self._pending_confirmation = None

        # ── FULL MODE: natural language intent routing ─────────────────────────
        # Only applies when message is NOT an explicit /skill command.
        if not message.strip().startswith('/'):
            try:
                if self._intent_parser is None:
                    from core.intent.intent_parser import HugoIntentParser
                    self._intent_parser = HugoIntentParser(
                        ollama_url=self.ollama_api,
                        model_name=self.model_name,
                        logger=self.logger,
                    )
                if self._action_router is None:
                    from core.actions.action_router import ActionRouter
                    self._action_router = ActionRouter(logger=self.logger)

                intent = self._intent_parser.parse(message, conversation_context=[])

                if intent.requires_action and intent.domain:
                    self.logger.log_event("cognition", "intent_action_detected", {
                        "domain": intent.domain,
                        "action": intent.action,
                        "confidence": intent.confidence,
                        "session_id": session_id,
                    })

                    # ── Memory domain: handle directly via Hugo's memory system ──
                    if intent.domain == "memory":
                        response_text = await self._handle_memory_intent(intent, session_id)
                        await self.post_process(response_text, session_id)
                        response_package = ResponsePackage(
                            content=response_text,
                            tone=MoodSpectrum.REFLECTIVE,
                            reasoning_chain=ReasoningChain(
                                steps=["Memory intent parsed", f"Action: {intent.action}", "Memory queried"],
                                assumptions=[intent.reasoning],
                                alternatives_considered=[],
                                selected_approach="memory_direct",
                                confidence_score=intent.confidence
                            ),
                            directive_checks=[],
                            metadata={
                                "mode": "memory_intent",
                                "action": intent.action,
                                "session_id": session_id,
                            }
                        )
                        return stream_single(response_package)

                    # ── Compound request: route to DelegationAgent ──────────────
                    if self._is_compound_request(message, intent):
                        self.logger.log_event("cognition", "compound_request_detected", {
                            "session_id": session_id,
                            "domain": intent.domain,
                        })
                        response_text = await self._handle_compound_request(message, session_id)
                        await self.post_process(response_text, session_id)
                        response_package = ResponsePackage(
                            content=response_text,
                            tone=MoodSpectrum.OPERATIONAL,
                            reasoning_chain=ReasoningChain(
                                steps=["Compound request detected", "Delegated to DelegationAgent"],
                                assumptions=["Multiple infrastructure actions required"],
                                alternatives_considered=["Single action routing"],
                                selected_approach="delegation",
                                confidence_score=intent.confidence
                            ),
                            directive_checks=["permission_checked"],
                            metadata={"mode": "delegation", "session_id": session_id}
                        )
                        return stream_single(response_package)

                    # ── Single action: route through ActionRouter ───────────────
                    action_result = await self._action_router.route(intent)

                    if not action_result.success and action_result.error == "confirmation_required":
                        # Store pending intent and return the confirmation prompt
                        self._pending_confirmation = intent
                        response_text = self._action_router.format_for_cognition(action_result)
                    else:
                        response_text = self._action_router.format_for_cognition(action_result)
                        await self.post_process(response_text, session_id)

                    response_package = ResponsePackage(
                        content=response_text,
                        tone=MoodSpectrum.OPERATIONAL,
                        reasoning_chain=ReasoningChain(
                            steps=["NLP intent parsed", f"Domain: {intent.domain}",
                                   f"Action: {intent.action}", "Executor dispatched"],
                            assumptions=[intent.reasoning],
                            alternatives_considered=["LLM generation", "/skill trigger"],
                            selected_approach="nlp_intent_routing",
                            confidence_score=intent.confidence
                        ),
                        directive_checks=["permission_checked"],
                        metadata={
                            "mode": "intent_routing",
                            "domain": intent.domain,
                            "action": intent.action,
                            "session_id": session_id,
                            "permission_level": action_result.permission_level,
                        }
                    )
                    return stream_single(response_package)

            except Exception as e:
                # Intent parsing failure must never crash the main pipeline
                self.logger.log_event("cognition", "intent_parse_error", {
                    "error": str(e), "session_id": session_id
                })

        # Save user message to memory (with skill trigger metadata if present)
        await self._save_user_message(message, session_id)

        if streaming:
            # Return streaming generator
            return self.process_input_streaming(message, session_id)
        else:
            # Return complete response wrapped in async iterator
            response_package = await self.process_input(message, session_id)

            # Post-process: Save assistant response to memory
            await self.post_process(response_package.content, session_id)

            self.logger.log_event("cognition", "non_streaming_wrapped_in_stream", {
                "response_length": len(response_package.content)
            })
            return stream_single(response_package)

    async def _generate_extraction_synthesis(self, message: str):
        """
        Special-purpose LLM call for extract_and_answer.
        No memory writes, no reflections, no persona,
        no fallback chatter. One-shot factual synthesis.

        This mode bypasses ALL normal conversation logic:
        - No memory saving
        - No reflection
        - No persona injection
        - No conversational tone
        - No fallback chatter

        Args:
            message: The synthesis prompt containing extracted text and question

        Returns:
            SimpleNamespace with content attribute (minimal interface compatible with ResponsePackage)
        """
        from types import SimpleNamespace

        prompt = (
            f"You are a factual synthesis engine. "
            f"Given extracted webpage text, produce the shortest direct answer possible.\n\n"
            f"{message}\n\n"
            f"Return ONLY the answer. Do not explain your steps."
        )

        # Use LLM with zero temperature for deterministic output
        if self.model_engine == "ollama":
            if self.ollama_async_mode:
                response = await self._local_infer_async(prompt, temperature=0.0)
            else:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    self._local_infer,
                    prompt,
                    0.0
                )
        else:
            response = "Model engine not configured."

        # Safety: ensure clean final output
        text = (response or "").strip()

        # Minimal error fallback
        if not text or len(text) < 3:
            text = "No clear information available."

        self.logger.log_event("cognition", "extraction_synthesis_complete", {
            "mode": "extraction_synthesis",
            "response_length": len(text),
            "bypassed_normal_flow": True
        })

        # Return minimal interface matching ResponsePackage
        return SimpleNamespace(content=text)

    async def _save_user_message(self, message: str, session_id: str):
        """
        Save user message to memory before processing.

        This method also checks for skill trigger metadata attached by the
        memory classification system and auto-executes skills when detected.

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

            # Store in memory (classification happens inside store())
            await self.memory.store(user_entry, persist_long_term=False)

            self.logger.log_event("cognition", "user_message_saved", {
                "session_id": session_id,
                "content_length": len(message)
            })

            # Check for skill trigger metadata attached during classification
            if "skill_trigger" in user_entry.metadata:
                self.logger.log_event("cognition", "skill_trigger_detected", {
                    "skill_name": user_entry.metadata["skill_trigger"],
                    "action": user_entry.metadata.get("skill_action", "unknown"),
                    "session_id": session_id
                })

                # Auto-execute skill if skill manager is available
                if self.runtime_manager and hasattr(self.runtime_manager, 'skills') and self.runtime_manager.skills:
                    skill_name = user_entry.metadata["skill_trigger"]
                    skill_action = user_entry.metadata.get("skill_action", "help")
                    skill_payload = user_entry.metadata.get("skill_payload", {})

                    self.logger.log_event("cognition", "skill_autorun_started", {
                        "skill": skill_name,
                        "action": skill_action,
                        "session_id": session_id
                    })

                    try:
                        # Execute the skill
                        result = await self.runtime_manager.skills.run_skill(
                            skill_name,
                            action=skill_action,
                            **skill_payload
                        )

                        self.logger.log_event("cognition", "skill_autorun_completed", {
                            "skill": skill_name,
                            "action": skill_action,
                            "success": result.success,
                            "message": result.message,
                            "session_id": session_id
                        })

                        # Store skill result in memory if successful
                        if result.success and result.output:
                            skill_result_entry = MemoryEntry(
                                id=None,
                                session_id=session_id,
                                timestamp=datetime.now(),
                                memory_type="skill_execution",
                                content=f"Skill '{skill_name}' executed: {result.message}",
                                embedding=None,
                                metadata={
                                    "skill_name": skill_name,
                                    "skill_action": skill_action,
                                    "skill_output": result.output,
                                    "auto_triggered": True
                                },
                                importance_score=0.6,
                                is_fact=False
                            )
                            await self.memory.store(skill_result_entry, persist_long_term=False)

                    except Exception as skill_error:
                        self.logger.log_error(skill_error, {
                            "phase": "skill_autorun",
                            "skill": skill_name,
                            "action": skill_action,
                            "session_id": session_id
                        })
                else:
                    self.logger.log_event("cognition", "skill_autorun_skipped", {
                        "reason": "skill_manager_not_available",
                        "skill": user_entry.metadata["skill_trigger"],
                        "session_id": session_id
                    })

        except Exception as e:
            self.logger.log_error(e, {"phase": "save_user_message"})

    async def _execute_skill_bypass(self, skill_name: str, skill_action: str,
                                    skill_payload: dict, original_message: str,
                                    session_id: str) -> ResponsePackage:
        """
        Execute a skill and bypass the LLM entirely.

        Used for internet queries where we want direct factual results
        without LLM hallucination risk.

        Args:
            skill_name: Name of skill to execute
            skill_action: Action to perform
            skill_payload: Skill parameters
            original_message: Original user message
            session_id: Session identifier

        Returns:
            ResponsePackage with skill result as content
        """
        try:
            from datetime import datetime

            self.logger.log_event("cognition", "skill_bypass_started", {
                "skill": skill_name,
                "action": skill_action,
                "session_id": session_id
            })

            # Execute skill
            if self.runtime_manager and hasattr(self.runtime_manager, 'skills') and self.runtime_manager.skills:
                result = await self.runtime_manager.skills.run_skill(
                    skill_name,
                    action=skill_action,
                    **skill_payload
                )

                self.logger.log_event("cognition", "skill_bypass_completed", {
                    "skill": skill_name,
                    "action": skill_action,
                    "success": result.success,
                    "session_id": session_id
                })

                # Format response based on skill result
                if result.success:
                    response_content = self._format_skill_response(skill_name, result)
                else:
                    response_content = f"I tried to look that up but encountered an issue: {result.message}"

                # Build response package with agent deployment info
                steps = [
                    f"Detected internet query: {original_message}",
                    f"Triggered skill: {skill_name}"
                ]

                if skill_name == "web_search":
                    steps.append("Deployed SearchAgent for multi-source investigation")
                    steps.append("Agent collected URLs from DuckDuckGo, Wikipedia, IMDb")
                    steps.append("Agent extracted and synthesized evidence")
                else:
                    steps.append(f"Executed action: {skill_action}")

                steps.append("Bypassed LLM to avoid hallucination")

                reasoning_chain = ReasoningChain(
                    steps=steps,
                    assumptions=["User needs factual real-time information"],
                    alternatives_considered=["LLM generation", "Agent-based investigation"],
                    selected_approach="autonomous_agent" if skill_name == "web_search" else "direct_skill_bypass",
                    confidence_score=0.95 if result.success else 0.5
                )

                response_package = ResponsePackage(
                    content=response_content,
                    tone=MoodSpectrum.OPERATIONAL,
                    reasoning_chain=reasoning_chain,
                    directive_checks=["internet_query_bypass"],
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "model": "skill_bypass",
                        "engine": skill_name,
                        "skill": skill_name,
                        "action": skill_action,
                        "success": result.success,
                        "bypassed_llm": True,
                        "session_id": session_id
                    }
                )

                return response_package

            else:
                # Skill manager not available - fallback to LLM
                self.logger.log_event("cognition", "skill_bypass_unavailable", {
                    "reason": "skill_manager_not_available",
                    "session_id": session_id
                })

                # Fall back to normal processing
                return await self.process_input(original_message, session_id)

        except Exception as e:
            self.logger.log_error(e, {
                "phase": "skill_bypass",
                "skill": skill_name,
                "session_id": session_id
            })

            # Return error response
            from datetime import datetime

            return ResponsePackage(
                content=f"I encountered an error trying to look that up: {str(e)}",
                tone=MoodSpectrum.OPERATIONAL,
                reasoning_chain=ReasoningChain(
                    steps=["Attempted skill bypass", "Encountered error"],
                    assumptions=[],
                    alternatives_considered=[],
                    selected_approach="error_fallback",
                    confidence_score=0.0
                ),
                directive_checks=[],
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "bypassed_llm": True,
                    "session_id": session_id
                }
            )

    def _format_skill_response(self, skill_name: str, result) -> str:
        """
        Format skill result into a natural language response.

        Args:
            skill_name: Name of the executed skill
            result: SkillResult object

        Returns:
            Formatted response string
        """
        if skill_name == "web_search":
            output = result.output
            if not output:
                return "I couldn't find any information about that."

            # Check for error passthrough from agent
            if output.get('error_passthrough'):
                agent_error = output.get('agent_error', 'Unknown error')
                return f"Agent encountered an error: {agent_error}"

            # Check if this is agent-based search with synthesized answer
            if output.get('synthesized_answer'):
                # Return concise, Jarvis-like synthesized answer
                answer = output.get('synthesized_answer', '')
                support = output.get('answer_support', '')

                # Format: Direct answer, then optional source info
                if support:
                    return f"{answer}\n\n({support})"
                else:
                    return answer

            # Fallback to combined evidence if synthesis not available
            elif output.get('combined_evidence'):
                response_parts = []

                # Commander-style report
                response_parts.append("Investigation complete.")

                # Report sources used
                sources = output.get('sources_used', [])
                if sources:
                    sources_str = ", ".join(s.upper() for s in sources)
                    response_parts.append(f"Sources checked: {sources_str}")

                # Report findings
                passages_count = output.get('passages_count', 0)
                if passages_count > 0:
                    response_parts.append(f"Evidence collected from {passages_count} sources.")

                    # Include combined evidence for synthesis
                    evidence = output.get('combined_evidence', '')
                    if evidence:
                        response_parts.append(f"\nFindings:\n{evidence[:1000]}")

                return "\n".join(response_parts)

            # Legacy format support (if any old code still uses it)
            response_parts = []

            if output.get('abstract_text'):
                response_parts.append(output['abstract_text'])
                if output.get('abstract_source'):
                    response_parts.append(f"\n\nSource: {output['abstract_source']}")

            elif output.get('answer'):
                response_parts.append(output['answer'])

            elif output.get('definition'):
                response_parts.append(output['definition'])
                if output.get('definition_source'):
                    response_parts.append(f"\n\nSource: {output['definition_source']}")

            if response_parts:
                return "\n".join(response_parts)

            return "I couldn't find any information about that."

        elif skill_name == "fetch_url":
            output = result.output
            if not output:
                return "I couldn't fetch that URL."

            response_parts = []
            response_parts.append(f"Fetched: {output.get('title', 'Untitled')}\n")

            content = output.get('content', '')
            # Return first 800 characters
            if len(content) > 800:
                response_parts.append(content[:800] + "...")
            else:
                response_parts.append(content)

            return "\n".join(response_parts)

        else:
            # Generic skill response
            if result.output:
                return str(result.output)
            else:
                return result.message

    async def process_input(self, user_input: str, session_id: str) -> ResponsePackage:
        """
        Main entry point for processing user input through the cognitive pipeline.

        Includes agent delegation for heavy technical work.

        NOTE: This method is called internally by generate_reply(). For external
        entry points, prefer generate_reply() which includes skill bypass logic.

        Args:
            user_input: Raw user message
            session_id: Current session identifier

        Returns:
            ResponsePackage with generated response and metadata
        """
        self.logger.log_event("cognition", "processing_input", {"session_id": session_id})

        # Check for skill triggers BEFORE processing
        # (This is a safety net in case process_input is called directly)
        classification = self.memory.classify_memory(user_input)
        if classification.metadata and "skill_trigger" in classification.metadata:
            skill_name = classification.metadata["skill_trigger"]
            skill_action = classification.metadata.get("skill_action", "help")
            skill_payload = classification.metadata.get("skill_payload", {})

            # Internet queries (web_search, fetch_url) bypass LLM entirely
            if skill_name in ["web_search", "fetch_url"]:
                self.logger.log_event("cognition", "internet_query_detected_in_process_input", {
                    "skill": skill_name,
                    "action": skill_action,
                    "session_id": session_id
                })

                # Execute skill directly and return result
                return await self._execute_skill_bypass(
                    skill_name, skill_action, skill_payload, user_input, session_id
                )

        # Check if this requires agent delegation
        if self._detect_heavy_work(user_input):
            return await self._delegate_to_worker(user_input, session_id)

        # Standard cognitive pipeline
        # Step 1: Perception
        perception = await self._perceive(user_input)

        # Step 2: Context Assembly
        context = await self._assemble_context(perception, session_id)

        # Step 2.5: Jarvis Mode - Contextual Anticipation
        anticipation = None
        if self.jarvis_mode_enabled:
            anticipation = await self._contextual_anticipation(user_input, context)

            # If we have a direct template response, return it immediately
            template_response = self._apply_jarvis_template(user_input, anticipation)
            if template_response:
                # Create simplified response package for template responses
                simple_reasoning = ReasoningChain(
                    steps=["Jarvis template applied"],
                    assumptions=[f"Pattern detected: {anticipation.get('pattern')}"],
                    alternatives_considered=[],
                    selected_approach="jarvis_template",
                    confidence_score=0.95
                )
                return ResponsePackage(
                    content=template_response,
                    tone=perception.detected_mood,
                    reasoning_chain=simple_reasoning,
                    directive_checks=[],
                    metadata={
                        "timestamp": __import__('datetime').datetime.now().isoformat(),
                        "jarvis_mode": True,
                        "template_applied": True,
                        "anticipation_pattern": anticipation.get("pattern")
                    }
                )

        # Step 3: Synthesis
        reasoning, generated_text, prompt_metadata = await self._synthesize(perception, context, session_id, user_input)

        # Step 3.5: Jarvis Mode - Response Compression
        if self.jarvis_mode_enabled:
            generated_text = self._compress_response(generated_text)

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

        NOTE: For external entry points, prefer generate_reply(streaming=True)
        which includes skill bypass logic.

        Args:
            user_input: Raw user message
            session_id: Current session identifier

        Yields:
            str: Response chunks as they're generated
            ResponsePackage: Final response package with metadata (last yield)
        """
        self.logger.log_event("cognition", "processing_input_streaming", {"session_id": session_id})

        # Check for skill triggers BEFORE processing
        # (This is a safety net in case process_input_streaming is called directly)
        classification = self.memory.classify_memory(user_input)
        if classification.metadata and "skill_trigger" in classification.metadata:
            skill_name = classification.metadata["skill_trigger"]
            skill_action = classification.metadata.get("skill_action", "help")
            skill_payload = classification.metadata.get("skill_payload", {})

            # Internet queries (web_search, fetch_url) bypass LLM entirely
            if skill_name in ["web_search", "fetch_url"]:
                self.logger.log_event("cognition", "internet_query_detected_in_streaming", {
                    "skill": skill_name,
                    "action": skill_action,
                    "session_id": session_id
                })

                # Execute skill and yield the result as a single chunk
                response_package = await self._execute_skill_bypass(
                    skill_name, skill_action, skill_payload, user_input, session_id
                )

                # Yield the response content as a single chunk
                yield response_package.content
                # Yield the final response package
                yield response_package
                return

        # Step 1: Perception
        perception = await self._perceive(user_input)

        # Step 2: Context Assembly
        context = await self._assemble_context(perception, session_id)

        # Step 2.5: Jarvis Mode - Contextual Anticipation
        anticipation = None
        if self.jarvis_mode_enabled:
            anticipation = await self._contextual_anticipation(user_input, context)

            # If we have a direct template response, return it immediately
            template_response = self._apply_jarvis_template(user_input, anticipation)
            if template_response:
                # Yield template response
                yield template_response

                # Create simplified response package
                simple_reasoning = ReasoningChain(
                    steps=["Jarvis template applied"],
                    assumptions=[f"Pattern detected: {anticipation.get('pattern')}"],
                    alternatives_considered=[],
                    selected_approach="jarvis_template",
                    confidence_score=0.95
                )
                response_package = ResponsePackage(
                    content=template_response,
                    tone=perception.detected_mood,
                    reasoning_chain=simple_reasoning,
                    directive_checks=[],
                    metadata={
                        "timestamp": __import__('datetime').datetime.now().isoformat(),
                        "jarvis_mode": True,
                        "template_applied": True,
                        "anticipation_pattern": anticipation.get("pattern")
                    }
                )
                yield response_package
                return

        # Step 3: Assemble prompt
        user_message = perception.corrected_input if perception.corrected_input else user_input

        # FULL mode only: Apply skill routing if enabled
        if is_full_mode():
            user_message = await self._apply_full_mode_skill_routing(user_message)

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

        # Step 4.5: Jarvis Mode - Response Compression
        if self.jarvis_mode_enabled:
            generated_response = self._compress_response(generated_response)

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

    # ============================================================================
    # FULL MODE: Skill Routing (CORE-2 Integration)
    # ============================================================================

    async def _apply_full_mode_skill_routing(self, user_text: str) -> str:
        """
        FULL-mode only: Apply skill routing if a skill trigger is detected.

        If the user_text starts with a valid skill trigger (e.g. /search),
        run the CORE-2 skill router and inject the resulting block before
        the user text. Otherwise, return the original user_text unchanged.

        This is FULL-mode only and MUST NOT be called from CORE mode.

        Args:
            user_text: User input text

        Returns:
            Original text or skill block + text if skill was triggered

        Example:
            Input: "/search python asyncio"
            Output: "[SkillResult]\nskill: search\n...\n\n/search python asyncio"
        """
        from core.skills.trigger_detector import detect_skill
        from core.skills.router import route_skill
        from core.skills.prompt_injection import inject_skill_block

        # Detect skill trigger
        trigger = detect_skill(user_text)
        if not trigger:
            # No skill detected, return original text
            return user_text

        self.logger.log_event("cognition", "skill_trigger_detected_full_mode", {
            "skill_name": trigger.skill_name,
            "args": trigger.args
        })

        # Route skill to handler
        skill_block = await route_skill(trigger)

        # Inject skill block before user text
        augmented_text = inject_skill_block(skill_block.block, user_text)

        self.logger.log_event("cognition", "skill_routing_complete_full_mode", {
            "skill_name": trigger.skill_name,
            "block_length": len(skill_block.block),
            "augmented_length": len(augmented_text)
        })

        return augmented_text

    # ============================================================================
    # CORE MODE: Minimal, Clean, Stable Pipeline
    # ============================================================================

    def _build_core_prompt(self, user_message: str, session_id: str, context: Optional[Any] = None) -> str:
        """
        Build a clean, minimal prompt for core mode.

        Core mode prompt structure:
        1. System block: persona description + core rules
        2. Context block: recent conversation (if any)
        3. User block: latest message

        Args:
            user_message: User's message
            session_id: Current session ID
            context: Optional context assembly (with memory)

        Returns:
            Formatted prompt string
        """
        # Get persona
        persona_name = self.persona.get("name", "Hugo")
        persona_role = self.persona.get("identity", {}).get("role", "Assistant")
        persona_desc = self.persona.get("identity", {}).get("archetype", "Helpful AI assistant")

        # Start with system block
        prompt_parts = [
            f"[Persona: {persona_name} — {persona_role}]",
            f"{persona_desc}",
            "",
            "[Core Rules]",
            "- Provide clear, helpful responses",
            "- Stay focused on the user's question",
            "- Be concise and direct",
            "",
        ]

        # Add recent conversation if available
        if context and hasattr(context, 'short_term_memory') and context.short_term_memory:
            prompt_parts.append("[Recent Conversation]")
            # Last 5 turns
            for mem in context.short_term_memory[-5:]:
                role = mem.get('metadata', {}).get('role', 'unknown')
                content = mem.get('content', '')
                if role == 'user':
                    prompt_parts.append(f"User: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"{persona_name}: {content}")
            prompt_parts.append("")

        # Add user message
        prompt_parts.append(f"User: {user_message}")
        prompt_parts.append(f"{persona_name}:")

        return "\n".join(prompt_parts)

    async def _generate_core_reply_streaming(self, message: str, session_id: str):
        """
        Generate reply in core mode (streaming).

        Core mode pipeline:
        1. Load persona
        2. Fetch minimal memory
        3. Build clean prompt
        4. Call Ollama via stability manager
        5. Stream chunks back

        Args:
            message: User message
            session_id: Session ID

        Yields:
            Text chunks from LLM, then final ResponsePackage
        """
        self.logger.log_event("cognition", "core_mode_active", {
            "session_id": session_id,
            "streaming": True,
            "mode": "core"
        })

        accumulated_response = []

        try:
            # Fetch minimal context (recent turns only)
            context = None
            if self.memory:
                recent_turns = await self.memory.retrieve_recent(session_id, limit=5)
                if recent_turns:
                    from types import SimpleNamespace
                    context = SimpleNamespace(short_term_memory=recent_turns)

            # Build core prompt
            prompt = self._build_core_prompt(message, session_id, context)

            self.logger.log_event("cognition", "core_prompt_built", {
                "prompt_length": len(prompt),
                "has_context": context is not None
            })

            # Stream from Ollama via stability manager
            for chunk in self.ollama_stability.stream_with_recovery(prompt, temperature=0.7):
                accumulated_response.append(chunk)
                yield chunk

            # Save assistant response to memory
            full_response = "".join(accumulated_response)
            await self.post_process(full_response, session_id)

            self.logger.log_event("cognition", "core_mode_complete", {
                "session_id": session_id,
                "status": "success",
                "response_length": len(full_response)
            })

            # Yield final ResponsePackage
            yield ResponsePackage(
                content=full_response,
                tone=MoodSpectrum.CONVERSATIONAL,
                reasoning_chain=ReasoningChain(
                    steps=["Core mode response"],
                    assumptions=[],
                    alternatives_considered=[],
                    selected_approach="direct_response",
                    confidence_score=0.9
                ),
                directive_checks=[],
                metadata={
                    "mode": "core",
                    "session_id": session_id,
                    "streaming": True
                }
            )

        except Exception as e:
            self.logger.log_error(e, {
                "phase": "core_mode_streaming",
                "session_id": session_id
            })
            # Yield soft fallback
            fallback_msg = self.ollama_stability.soft_fallback_message("general")
            yield fallback_msg

            # Save fallback to memory
            await self.post_process(fallback_msg, session_id)

            # Yield fallback ResponsePackage
            yield ResponsePackage(
                content=fallback_msg,
                tone=MoodSpectrum.CONVERSATIONAL,
                reasoning_chain=ReasoningChain(
                    steps=["Fallback triggered"],
                    assumptions=[],
                    alternatives_considered=[],
                    selected_approach="soft_fallback",
                    confidence_score=0.5
                ),
                directive_checks=[],
                metadata={
                    "mode": "core",
                    "session_id": session_id,
                    "streaming": True,
                    "fallback": True
                }
            )

    async def _generate_core_reply_nonstreaming(self, message: str, session_id: str) -> str:
        """
        Generate reply in core mode (non-streaming).

        Same pipeline as streaming, but returns complete response.

        Args:
            message: User message
            session_id: Session ID

        Returns:
            Complete response text
        """
        self.logger.log_event("cognition", "core_mode_active", {
            "session_id": session_id,
            "streaming": False,
            "mode": "core"
        })

        try:
            # Fetch minimal context
            context = None
            if self.memory:
                recent_turns = await self.memory.retrieve_recent(session_id, limit=5)
                if recent_turns:
                    from types import SimpleNamespace
                    context = SimpleNamespace(short_term_memory=recent_turns)

            # Build core prompt
            prompt = self._build_core_prompt(message, session_id, context)

            self.logger.log_event("cognition", "core_prompt_built", {
                "prompt_length": len(prompt),
                "has_context": context is not None
            })

            # Non-streaming call via stability manager
            result = self.ollama_stability.non_stream_fallback(prompt, temperature=0.7)

            if result.success:
                response = result.content
            else:
                response = self.ollama_stability.soft_fallback_message("general")

            self.logger.log_event("cognition", "core_mode_complete", {
                "session_id": session_id,
                "status": "success" if result.success else "fallback",
                "response_length": len(response)
            })

            return response

        except Exception as e:
            self.logger.log_error(e, {
                "phase": "core_mode_nonstreaming",
                "session_id": session_id
            })
            return self.ollama_stability.soft_fallback_message("general")

    # ============================================================================
    # END CORE MODE
    # ============================================================================

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

    async def _contextual_anticipation(self, user_input: str, context: ContextAssembly) -> Optional[Dict[str, Any]]:
        """
        Jarvis-style contextual anticipation for ambiguous commands.

        Analyzes recent context to infer likely meaning of vague commands like:
        - "Fix that"
        - "Continue"
        - "Next"
        - "Run it again"
        - "Help me with the Metix widget"

        Returns:
            Dictionary with inferred context or None if no anticipation possible
        """
        if not self.jarvis_mode_enabled:
            return None

        user_lower = user_input.lower().strip()

        # Pattern matching for ambiguous commands
        ambiguous_patterns = {
            "fix_that": ["fix that", "fix this", "fix it"],
            "continue": ["continue", "keep going", "go on", "next"],
            "run_again": ["run it", "run again", "try again", "do it again"],
            "help_with": ["help me with", "help with", "work on"],
            "phase": ["phase", "start phase"],
            "patch": ["generate patch", "create patch", "make patch"],
            "next_step": ["next step", "move forward", "what's next"]
        }

        detected_pattern = None
        for pattern_name, patterns in ambiguous_patterns.items():
            if any(p in user_lower for p in patterns):
                detected_pattern = pattern_name
                break

        if not detected_pattern:
            return None

        # Extract recent context (last 3-5 messages)
        recent_context = []
        recency_depth = self.jarvis_config.get("anticipation", {}).get("context_depth", 5)

        if context.short_term_memory:
            for mem in context.short_term_memory[-recency_depth:]:
                content = mem.get('content', '').lower()
                recent_context.append(content)

        # Infer context based on pattern and recent messages
        inferred = {
            "pattern": detected_pattern,
            "possibilities": [],
            "suggested_disambiguation": None
        }

        # Pattern-specific inference
        if detected_pattern == "fix_that":
            # Look for error mentions, SQL, bugs in recent context
            possibilities = []
            if any("error" in ctx or "bug" in ctx for ctx in recent_context):
                possibilities.append("the recent error")
            if any("sql" in ctx or "query" in ctx for ctx in recent_context):
                possibilities.append("the SQL query")
            if any("refactor" in ctx for ctx in recent_context):
                possibilities.append("the refactoring")

            if len(possibilities) > 1:
                inferred["possibilities"] = possibilities
                inferred["suggested_disambiguation"] = f"Two possibilities: {possibilities[0]} or {possibilities[1]} — which one?"
            elif len(possibilities) == 1:
                inferred["possibilities"] = possibilities
                inferred["suggested_disambiguation"] = None  # Clear intent

        elif detected_pattern == "continue":
            # Look for ongoing tasks
            if any("phase" in ctx for ctx in recent_context):
                inferred["possibilities"] = ["next phase step"]
                inferred["suggested_disambiguation"] = "Next stage: synthesis. Running now."
            elif any("refactor" in ctx or "implement" in ctx for ctx in recent_context):
                inferred["possibilities"] = ["continue implementation"]

        elif detected_pattern == "help_with":
            # Extract widget/component references
            if "metix" in user_lower:
                # Common Metix widget IDs
                inferred["possibilities"] = ["widget 41", "widget 48", "widget 45", "widget 50"]
                inferred["suggested_disambiguation"] = "Which one — 41, 48, 45, or 50?"
            elif "hugo" in user_lower:
                if "phase" in user_lower:
                    inferred["possibilities"] = ["Phase 4.1", "Phase 4.2", "Phase 5"]
                    inferred["suggested_disambiguation"] = "Which phase — 4.1, 4.2, or 5?"

        elif detected_pattern == "phase":
            # Extract phase number if present
            import re
            phase_match = re.search(r'phase\s*(\d+\.?\d*)', user_lower)
            if phase_match:
                phase_num = phase_match.group(1)
                inferred["possibilities"] = [f"Phase {phase_num}"]
                inferred["suggested_disambiguation"] = f"Ready. Phase {phase_num} initialized."
            else:
                inferred["possibilities"] = ["Phase 4.1", "Phase 4.2", "Phase 5"]
                inferred["suggested_disambiguation"] = "Which phase — 4.1, 4.2, or 5?"

        elif detected_pattern == "patch":
            inferred["possibilities"] = ["generate code patch"]
            inferred["suggested_disambiguation"] = "On it."

        elif detected_pattern == "next_step":
            inferred["suggested_disambiguation"] = "Running now."

        return inferred if inferred["possibilities"] or inferred["suggested_disambiguation"] else None

    def _compress_response(self, response: str) -> str:
        """
        Compress response to Jarvis-style concise format.

        Removes:
        - Hedging ("maybe", "possibly", "I think")
        - Filler phrases
        - Excessive pleasantries
        - Redundancy

        Args:
            response: Generated response text

        Returns:
            Compressed response
        """
        if not self.jarvis_mode_enabled:
            return response

        # Remove hedging phrases
        hedging_patterns = [
            r'\b(maybe|possibly|perhaps|might|could be|I think|I believe)\b',
            r'\b(it seems|it appears|probably|likely)\b',
            r'\bsort of\b',
            r'\bkind of\b'
        ]

        compressed = response
        for pattern in hedging_patterns:
            compressed = re.sub(pattern, '', compressed, flags=re.IGNORECASE)

        # Remove filler prefaces
        filler_patterns = [
            r'^(Sure,?\s*)',
            r'^(Okay,?\s*)',
            r'^(Alright,?\s*)',
            r'^(Here (it is|you go):?\s*)',
            r'^(Let me\s+)',
            r'^(I\'ll\s+)',
            r'^(I can\s+)'
        ]

        for pattern in filler_patterns:
            compressed = re.sub(pattern, '', compressed, flags=re.IGNORECASE)

        # Clean up extra whitespace
        compressed = re.sub(r'\s+', ' ', compressed).strip()

        # If response is too long, truncate to key points (unless it's code or data)
        max_sentences = 3
        if not any(indicator in compressed for indicator in ['```', 'SELECT', 'FROM', 'def ', 'class ']):
            sentences = re.split(r'[.!?]\s+', compressed)
            if len(sentences) > max_sentences:
                compressed = '. '.join(sentences[:max_sentences]) + '.'

        return compressed

    def _apply_jarvis_template(self, user_input: str, anticipation: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Apply Jarvis-style response templates for common patterns.

        Args:
            user_input: User input
            anticipation: Contextual anticipation result

        Returns:
            Template response or None if no template applies
        """
        if not self.jarvis_mode_enabled or not anticipation:
            return None

        templates = self.jarvis_config.get("templates", {})

        if anticipation.get("suggested_disambiguation"):
            return anticipation["suggested_disambiguation"]

        # Pattern-specific templates
        pattern = anticipation.get("pattern")

        if pattern == "patch":
            return templates.get("on_it", "On it.")
        elif pattern == "next_step":
            return templates.get("running", "Running now.")
        elif pattern == "continue":
            return templates.get("confirmed", "Confirmed.")

        return None

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
            ""
        ]

        # Add Jarvis mode instructions if enabled
        if self.jarvis_mode_enabled:
            prompt_parts.extend([
                "[Jarvis Mode: ACTIVE]",
                "Response Style:",
                "- Default length: 1-3 sentences (expand only if user explicitly asks for detail)",
                "- NEVER hedge with 'maybe', 'possibly', 'I think'",
                "- Give direct recommendations, no filler",
                "- When unsure: ask tight disambiguation question",
                "- Prefer action over analysis",
                "- No prefaces like 'Sure, here it is:' — just deliver",
                ""
            ])

        prompt_parts.extend([
            "[Memory Policy]",
            "CRITICAL: When responding about user information or past conversations:",
            "- If a memory exists, use it EXACTLY as written",
            "- If no memory exists, say 'I'm not certain' rather than guessing",
            "- NEVER fabricate or invent facts about the user",
            "- Only reference information from the sections below",
            ""
        ])

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
        Perform streaming local inference using Ollama API with enhanced stability.

        This generator yields text chunks as they arrive from the model,
        with automatic error recovery, context reduction, and fallback handling.

        NEW in Phase 5.2:
        - Automatic 500 error recovery with context reduction
        - Streaming to non-streaming fallback
        - Server health detection
        - Payload validation
        - Soft fallback messages
        - Enhanced logging

        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature (0.0-1.0)

        Yields:
            str: Text chunks as they arrive from Ollama (or fallback message)
        """
        # Delegate to stability manager
        for chunk in self.ollama_stability.stream_with_recovery(prompt, temperature):
            yield chunk

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

        # FULL mode only: Apply skill routing if enabled
        if is_full_mode():
            user_message = await self._apply_full_mode_skill_routing(user_message)

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

            # Apply persona transformation (Hugo's Right Hand style)
            original_length = len(generated_response)

            # Build persona context from memory
            persona_context = PersonaContext(
                recent_turns=[m for m in context.short_term_memory[-5:] if m],
                last_domain=None,  # Will be detected from user_input
                ongoing_task=None,  # TODO: Extract from context
                user_preferences={}  # TODO: Extract from factual memories
            )

            # Transform response through persona engine
            generated_response = self.persona_engine.detect_and_transform(
                response_text=generated_response,
                user_input=user_input,
                persona_context=persona_context
            )

            self.logger.log_event("cognition", "persona_transform_applied", {
                "original_length": original_length,
                "transformed_length": len(generated_response),
                "compression_ratio": round(len(generated_response) / original_length, 2) if original_length > 0 else 1.0,
                "jarvis_mode": self.jarvis_mode_enabled
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
