"""
Tests for Hugo Core Mode Pipeline
----------------------------------
Tests the minimal, clean, stable cognition pipeline (CORE mode).

Requirements:
- Core mode uses OllamaStabilityManager
- Core mode returns async iterator
- Core mode handles Ollama errors gracefully
- Core mode prompt assembly is sane
- Core mode integrates with REPL properly
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock
from types import SimpleNamespace


@pytest.fixture
def mock_logger():
    """Mock HugoLogger for testing"""
    logger = Mock()
    logger.log_event = Mock()
    logger.log_error = Mock()
    return logger


@pytest.fixture
def mock_memory():
    """Mock MemoryManager for testing"""
    memory = Mock()
    memory.retrieve_recent = AsyncMock(return_value=[
        {
            "content": "Hello Hugo",
            "metadata": {"role": "user"}
        },
        {
            "content": "Hi! How can I help?",
            "metadata": {"role": "assistant"}
        }
    ])
    return memory


@pytest.fixture
def mock_ollama_stability():
    """Mock OllamaStabilityManager for testing"""
    manager = Mock()

    # Mock streaming response
    manager.stream_with_recovery = Mock(return_value=iter([
        "This ", "is ", "a ", "test ", "response."
    ]))

    # Mock non-streaming response
    manager.non_stream_fallback = Mock(return_value=SimpleNamespace(
        success=True,
        content="This is a test response."
    ))

    # Mock soft fallback
    manager.soft_fallback_message = Mock(
        return_value="(My reasoning engine is warming up… let me try that again.)"
    )

    return manager


@pytest.fixture
def mock_persona():
    """Mock persona for testing"""
    return {
        "name": "Hugo",
        "codename": "The Right Hand",
        "identity": {
            "role": "Right Hand / Second in Command",
            "archetype": "A loyal, reflective second-in-command"
        }
    }


@pytest.fixture
def cognition_engine(mock_logger, mock_memory, mock_ollama_stability, mock_persona):
    """Create CognitionEngine instance with mocked dependencies"""
    from core.cognition import CognitionEngine

    # Create engine with proper constructor
    engine = CognitionEngine(
        memory_manager=mock_memory,
        logger=mock_logger,
        runtime_manager=None
    )

    # Inject mocked dependencies
    engine.persona = mock_persona
    engine.ollama_stability = mock_ollama_stability
    engine.post_process = AsyncMock()  # Mock post_process
    engine._save_user_message = AsyncMock()  # Mock user message saving

    return engine


class TestCoreMode:
    """Test core mode functionality"""

    @pytest.mark.asyncio
    async def test_core_mode_detection(self):
        """Test that is_core_mode() detects HUGO_MODE correctly"""
        import sys
        import importlib.util

        # Direct import of core.config module to avoid importing entire core package
        spec = importlib.util.spec_from_file_location(
            "core.config",
            "core/config.py"
        )
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        is_core_mode = config_module.is_core_mode
        is_full_mode = config_module.is_full_mode
        get_hugo_mode = config_module.get_hugo_mode
        HugoMode = config_module.HugoMode

        # Default should be core mode
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HUGO_MODE", None)
            assert is_core_mode() == True
            assert is_full_mode() == False
            assert get_hugo_mode() == HugoMode.CORE

        # Explicit core mode
        with patch.dict(os.environ, {"HUGO_MODE": "core"}):
            assert is_core_mode() == True
            assert is_full_mode() == False

        # Full mode
        with patch.dict(os.environ, {"HUGO_MODE": "full"}):
            assert is_core_mode() == False
            assert is_full_mode() == True

    @pytest.mark.asyncio
    async def test_build_core_prompt(self, cognition_engine, mock_memory):
        """Test core prompt building with context"""
        message = "What's the weather like?"
        session_id = "test_session"

        # Create context with recent conversation
        context = SimpleNamespace(short_term_memory=[
            {"content": "Hello", "metadata": {"role": "user"}},
            {"content": "Hi!", "metadata": {"role": "assistant"}}
        ])

        # Build prompt
        prompt = cognition_engine._build_core_prompt(message, session_id, context)

        # Verify prompt structure
        assert "[Persona: Hugo" in prompt
        assert "Right Hand" in prompt
        assert "[Core Rules]" in prompt
        assert "[Recent Conversation]" in prompt
        assert "User: Hello" in prompt
        assert "Hugo: Hi!" in prompt
        assert f"User: {message}" in prompt
        assert "Hugo:" in prompt

    @pytest.mark.asyncio
    async def test_build_core_prompt_no_context(self, cognition_engine):
        """Test core prompt building without context"""
        message = "Hello"
        session_id = "test_session"

        # Build prompt without context
        prompt = cognition_engine._build_core_prompt(message, session_id, None)

        # Should have persona and rules but no conversation history
        assert "[Persona: Hugo" in prompt
        assert "[Core Rules]" in prompt
        assert "[Recent Conversation]" not in prompt
        assert f"User: {message}" in prompt

    @pytest.mark.asyncio
    async def test_core_streaming_uses_stability_manager(self, cognition_engine, mock_ollama_stability):
        """Test that core streaming mode uses OllamaStabilityManager"""
        message = "Test message"
        session_id = "test_session"

        # Call core streaming
        chunks = []
        async for chunk in cognition_engine._generate_core_reply_streaming(message, session_id):
            if isinstance(chunk, str):
                chunks.append(chunk)

        # Verify stability manager was called
        assert mock_ollama_stability.stream_with_recovery.called

        # Verify chunks were yielded
        assert len(chunks) > 0
        assert "".join(chunks) == "This is a test response."

    @pytest.mark.asyncio
    async def test_core_nonstreaming_uses_stability_manager(self, cognition_engine, mock_ollama_stability):
        """Test that core non-streaming mode uses OllamaStabilityManager"""
        message = "Test message"
        session_id = "test_session"

        # Call core non-streaming
        response = await cognition_engine._generate_core_reply_nonstreaming(message, session_id)

        # Verify stability manager was called
        assert mock_ollama_stability.non_stream_fallback.called

        # Verify response
        assert response == "This is a test response."

    @pytest.mark.asyncio
    async def test_core_streaming_handles_errors(self, cognition_engine, mock_ollama_stability):
        """Test that core streaming handles errors with soft fallback"""
        # Make stability manager raise exception
        mock_ollama_stability.stream_with_recovery = Mock(side_effect=Exception("Test error"))

        message = "Test message"
        session_id = "test_session"

        # Call core streaming
        chunks = []
        async for chunk in cognition_engine._generate_core_reply_streaming(message, session_id):
            if isinstance(chunk, str):
                chunks.append(chunk)

        # Should get soft fallback message
        response = "".join(chunks)
        assert "reasoning engine" in response or "warming up" in response

    @pytest.mark.asyncio
    async def test_core_nonstreaming_handles_errors(self, cognition_engine, mock_ollama_stability):
        """Test that core non-streaming handles errors with soft fallback"""
        # Make stability manager return failure
        mock_ollama_stability.non_stream_fallback = Mock(return_value=SimpleNamespace(
            success=False,
            content=None
        ))

        message = "Test message"
        session_id = "test_session"

        # Call core non-streaming
        response = await cognition_engine._generate_core_reply_nonstreaming(message, session_id)

        # Should get soft fallback message
        assert "reasoning engine" in response or "warming up" in response

    @pytest.mark.asyncio
    async def test_generate_reply_delegates_to_core_streaming(self, cognition_engine):
        """Test that generate_reply() delegates to core mode when streaming=True"""
        with patch.dict(os.environ, {"HUGO_MODE": "core"}):
            with patch.object(cognition_engine, '_generate_core_reply_streaming') as mock_core:
                # Setup mock to return async generator
                async def fake_generator():
                    yield "test"

                mock_core.return_value = fake_generator()

                # Call generate_reply with streaming=True
                result = await cognition_engine.generate_reply(
                    "Test message",
                    "test_session",
                    streaming=True
                )

                # Verify core streaming was called
                assert mock_core.called

    @pytest.mark.asyncio
    async def test_generate_reply_delegates_to_core_nonstreaming(self, cognition_engine):
        """Test that generate_reply() delegates to core mode when streaming=False"""
        with patch.dict(os.environ, {"HUGO_MODE": "core"}):
            with patch.object(cognition_engine, '_generate_core_reply_nonstreaming') as mock_core:
                # Setup mock to return string
                mock_core.return_value = "test response"

                # Call generate_reply with streaming=False
                result = await cognition_engine.generate_reply(
                    "Test message",
                    "test_session",
                    streaming=False
                )

                # Consume iterator
                chunks = []
                async for chunk in result:
                    chunks.append(chunk)

                # Verify core non-streaming was called
                assert mock_core.called

    @pytest.mark.asyncio
    async def test_core_mode_saves_to_memory_streaming(self, cognition_engine, mock_memory):
        """Test that core streaming mode saves responses to memory"""
        message = "Test message"
        session_id = "test_session"

        # Call core streaming and consume
        chunks = []
        async for chunk in cognition_engine._generate_core_reply_streaming(message, session_id):
            if isinstance(chunk, str):
                chunks.append(chunk)

        # Verify post_process (memory save) was called
        assert cognition_engine.post_process.called

        # Verify it was called with the full response
        call_args = cognition_engine.post_process.call_args
        assert "This is a test response." in call_args[0][0]

    @pytest.mark.asyncio
    async def test_core_mode_saves_to_memory_nonstreaming(self, cognition_engine, mock_memory):
        """Test that core non-streaming mode saves responses to memory"""
        # Note: In the current implementation, non-streaming memory save happens
        # in generate_reply(), not in _generate_core_reply_nonstreaming()

        with patch.dict(os.environ, {"HUGO_MODE": "core"}):
            # Call generate_reply (which handles memory for non-streaming)
            result = await cognition_engine.generate_reply(
                "Test message",
                "test_session",
                streaming=False
            )

            # Consume iterator
            async for chunk in result:
                pass

            # Verify post_process was called
            assert cognition_engine.post_process.called

    @pytest.mark.asyncio
    async def test_core_prompt_includes_temperature(self, cognition_engine, mock_ollama_stability):
        """Test that core mode calls stability manager with correct temperature"""
        message = "Test message"
        session_id = "test_session"

        # Call core streaming
        async for chunk in cognition_engine._generate_core_reply_streaming(message, session_id):
            pass

        # Verify temperature was passed
        call_kwargs = mock_ollama_stability.stream_with_recovery.call_args[1]
        assert call_kwargs.get('temperature') == 0.7

    @pytest.mark.asyncio
    async def test_core_mode_fetches_recent_memory(self, cognition_engine, mock_memory):
        """Test that core mode fetches recent memory (limit=5)"""
        message = "Test message"
        session_id = "test_session"

        # Call core streaming
        async for chunk in cognition_engine._generate_core_reply_streaming(message, session_id):
            pass

        # Verify retrieve_recent was called with limit=5
        assert mock_memory.retrieve_recent.called
        call_args = mock_memory.retrieve_recent.call_args
        assert call_args[0][0] == session_id
        assert call_args[1]['limit'] == 5

    @pytest.mark.asyncio
    async def test_core_mode_logs_events(self, cognition_engine, mock_logger):
        """Test that core mode logs important events"""
        message = "Test message"
        session_id = "test_session"

        # Call core streaming
        async for chunk in cognition_engine._generate_core_reply_streaming(message, session_id):
            pass

        # Verify log events were called
        assert mock_logger.log_event.called

        # Check for specific events
        event_names = [call[0][1] for call in mock_logger.log_event.call_args_list]
        assert "core_mode_active" in event_names
        assert "core_prompt_built" in event_names
        assert "core_mode_complete" in event_names


class TestCoreIntegration:
    """Integration tests for core mode with full pipeline"""

    @pytest.mark.asyncio
    async def test_full_pipeline_core_mode_streaming(self, cognition_engine):
        """Test complete pipeline: generate_reply -> core streaming -> memory save"""
        with patch.dict(os.environ, {"HUGO_MODE": "core"}):
            message = "Hello Hugo"
            session_id = "integration_test"

            # Call full pipeline
            result = await cognition_engine.generate_reply(
                message,
                session_id,
                streaming=True
            )

            # Consume and verify
            chunks = []
            response_pkg = None
            async for chunk in result:
                if isinstance(chunk, str):
                    chunks.append(chunk)
                else:
                    response_pkg = chunk

            # Verify we got chunks
            assert len(chunks) > 0

            # Verify we got ResponsePackage
            assert response_pkg is not None
            assert response_pkg.metadata.get("mode") == "core"

    @pytest.mark.asyncio
    async def test_full_pipeline_core_mode_nonstreaming(self, cognition_engine):
        """Test complete pipeline: generate_reply -> core non-streaming -> memory save"""
        with patch.dict(os.environ, {"HUGO_MODE": "core"}):
            message = "Hello Hugo"
            session_id = "integration_test"

            # Call full pipeline
            result = await cognition_engine.generate_reply(
                message,
                session_id,
                streaming=False
            )

            # Consume and verify
            response_pkg = None
            async for chunk in result:
                response_pkg = chunk

            # Verify we got ResponsePackage
            assert response_pkg is not None
            assert response_pkg.metadata.get("mode") == "core"
            assert response_pkg.content == "This is a test response."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
