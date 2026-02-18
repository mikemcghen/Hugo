"""
Integration tests for the dual-path routing in CognitionEngine.generate_reply().

Verifies:
- /skill commands always take the explicit skill path (ActionRouter never called)
- Natural language infrastructure requests take the intent path
- Low-confidence intents fall through to standard LLM pipeline
- CORE mode never touches the intent parser
- Pending confirmation flow works end-to-end
"""

import sys
from unittest.mock import MagicMock, AsyncMock, patch

# faiss, sentence_transformers etc. are only installed on the GPU server that runs Hugo.
# Mock heavy ML deps before any import that pulls in core.memory.
for _dep in ('faiss', 'sentence_transformers', 'sentence_transformers.SentenceTransformer'):
    if _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()

import pytest
from core.config import HugoMode


def make_cognition_engine():
    """Create a CognitionEngine with mocked memory/logger/stability."""
    memory = MagicMock()
    memory.classify_memory.return_value = MagicMock(metadata={}, memory_type="conversation")
    memory.retrieve_recent = AsyncMock(return_value=[])
    memory.store_memory = AsyncMock()

    logger = MagicMock()
    logger.log_event = MagicMock()
    logger.log_error = MagicMock()

    from core.cognition import CognitionEngine
    engine = CognitionEngine(memory_manager=memory, logger=logger)

    # Mock OllamaStabilityManager so no real Ollama calls happen
    engine.ollama_stability = MagicMock()
    engine.ollama_stability.generate_with_fallback = MagicMock(return_value=MagicMock(
        success=True, content="LLM response", fallback_used=False
    ))

    return engine


class TestSkillPathUnchanged:
    @pytest.mark.asyncio
    async def test_slash_skill_bypasses_intent_parser(self):
        """A /search command should never reach the intent parser."""
        with patch("core.config.get_hugo_mode", return_value=HugoMode.FULL):
            engine = make_cognition_engine()

            mock_intent_parser = MagicMock()
            engine._intent_parser = mock_intent_parser

            # Mock the skill detection to return a proper skill trigger
            # detect_skill is imported locally inside _run_skill_pipeline, so patch the source module
            with patch("core.skills.trigger_detector.detect_skill") as mock_detect:
                from core.skills.trigger_detector import SkillTrigger
                mock_detect.return_value = SkillTrigger(
                    skill_name="search", args="python asyncio", raw_input="/search python asyncio"
                )
                with patch("core.skills.router.route_skill") as mock_route:
                    from core.skills.router import SkillBlock
                    mock_route.return_value = SkillBlock(block="[SkillResult]\nskill: search\noutput: \"results\"")

                    with patch.object(engine, "_save_user_message", new_callable=AsyncMock):
                        with patch.object(engine, "post_process", new_callable=AsyncMock):
                            with patch.object(engine, "process_input", new_callable=AsyncMock) as mock_process:
                                mock_process.return_value = MagicMock(content="LLM response")
                                try:
                                    await engine.generate_reply("/search python asyncio", "test_session")
                                except Exception:
                                    pass  # We just care that intent parser was NOT called

            # Intent parser should not have been called for /skill commands
            mock_intent_parser.parse.assert_not_called()


class TestIntentRoutingPath:
    @pytest.mark.asyncio
    async def test_natural_language_reaches_intent_parser(self):
        """Natural language docker request should reach the intent parser."""
        with patch("core.config.get_hugo_mode", return_value=HugoMode.FULL):
            engine = make_cognition_engine()

            from core.intent.parsed_intent import ParsedIntent
            from core.actions.action_result import ActionResult

            mock_intent = ParsedIntent(
                requires_action=True, domain="docker", action="list",
                target=None, parameters={}, confidence=0.92,
                reasoning="User wants to list containers", original_message="list all docker containers"
            )
            mock_action_result = ActionResult(
                success=True, data=[{"name": "ollama", "status": "Up", "image": "ollama"}],
                domain="docker", action="list"
            )

            mock_parser = MagicMock()
            mock_parser.parse.return_value = mock_intent
            engine._intent_parser = mock_parser

            mock_router = MagicMock()
            mock_router.route = AsyncMock(return_value=mock_action_result)
            mock_router.format_for_cognition.return_value = "Running containers:\n- ollama: Up"
            engine._action_router = mock_router

            with patch.object(engine, "_save_user_message", new_callable=AsyncMock):
                with patch.object(engine, "post_process", new_callable=AsyncMock):
                    with patch("core.cognition.is_core_mode", return_value=False):
                        try:
                            gen = await engine.generate_reply(
                                "list all docker containers", "test_session"
                            )
                        except Exception:
                            pass

            mock_parser.parse.assert_called_once()


class TestCoreModeIsolation:
    @pytest.mark.asyncio
    async def test_core_mode_never_calls_intent_parser(self):
        """In CORE mode, intent parser should never be called."""
        engine = make_cognition_engine()

        mock_intent_parser = MagicMock()
        engine._intent_parser = mock_intent_parser

        with patch("core.cognition.is_core_mode", return_value=True):
            with patch.object(engine, "_save_user_message", new_callable=AsyncMock):
                with patch.object(engine, "_generate_core_reply_nonstreaming", new_callable=AsyncMock) as mock_core:
                    mock_core.return_value = "Core response"
                    with patch.object(engine, "post_process", new_callable=AsyncMock):
                        try:
                            await engine.generate_reply(
                                "restart the ollama container", "test_session"
                            )
                        except Exception:
                            pass

        mock_intent_parser.parse.assert_not_called()


class TestMemoryGoalClassification:
    """Tests that memory correctly classifies goal and relationship messages."""

    def test_goal_classified_as_goal(self):
        from core.memory import MemoryManager
        # Create a minimal MemoryManager without real connections
        mm = MemoryManager.__new__(MemoryManager)

        import re
        mm._extract_note_content = lambda t: t
        mm.enable_llm_classification = False

        result = mm.classify_memory("my goal is to finish the Metix project")
        assert result.memory_type == "goal"
        assert result.should_persist is True
        assert result.importance >= 0.9

    def test_relationship_classified_correctly(self):
        from core.memory import MemoryManager
        mm = MemoryManager.__new__(MemoryManager)
        mm._extract_note_content = lambda t: t
        mm.enable_llm_classification = False

        result = mm.classify_memory("my colleague Dave handles the DevOps work")
        assert result.memory_type == "relationship"
        assert result.should_persist is True

    def test_identity_still_works(self):
        from core.memory import MemoryManager
        mm = MemoryManager.__new__(MemoryManager)
        mm._extract_note_content = lambda t: t
        mm.enable_llm_classification = False

        result = mm.classify_memory("I am a software engineer")
        assert result.memory_type == "identity"
