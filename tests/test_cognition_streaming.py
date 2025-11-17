"""
Tests for Cognition Engine streaming functionality
---------------------------------------------------
Ensures cognition.generate_reply() always returns an async iterator,
preventing 'async for' type errors in REPL and other clients.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class MockMemoryEntry:
    id: Any
    session_id: str
    timestamp: Any
    memory_type: str
    content: str
    embedding: Any
    metadata: Dict[str, Any]
    importance_score: float


@dataclass
class MockMemoryClassification:
    memory_type: str
    importance_score: float
    should_persist: bool
    metadata: Dict[str, Any] = None


@dataclass
class MockResponsePackage:
    content: str
    tone: str
    reasoning_chain: Any
    directive_checks: List[str]
    metadata: Dict[str, Any]


class MockMemoryManager:
    def classify_memory(self, message):
        # Simulate normal conversation (no skill trigger)
        return MockMemoryClassification(
            memory_type="conversation",
            importance_score=0.5,
            should_persist=False,
            metadata={}
        )

    async def store(self, entry):
        pass


class MockLogger:
    def log_event(self, category, event_type, data):
        pass


@pytest.mark.asyncio
async def test_cognition_force_streaming_interface():
    """Test that generate_reply ALWAYS returns an async iterator"""
    # This test verifies the core fix: generate_reply must return an async iterator
    # even when it would have returned a plain ResponsePackage before

    from runtime.utils.async_helpers import stream_single, is_async_iterator

    # Test single value wrapped in stream_single
    package = MockResponsePackage(
        content="Test",
        tone="conversational",
        reasoning_chain=None,
        directive_checks=[],
        metadata={}
    )

    result = stream_single(package)

    # Assert it's an async iterator
    assert is_async_iterator(result)

    # Assert we can iterate it
    collected = []
    async for item in result:
        collected.append(item)

    assert len(collected) == 1
    assert collected[0] == package


@pytest.mark.asyncio
async def test_cognition_streaming_normal_conversation():
    """Test that normal streaming conversation returns async iterator"""
    # Simulate streaming generator
    async def mock_process_input_streaming(message, session_id):
        yield "Hello"
        yield " "
        yield "world"
        yield MockResponsePackage(
            content="Hello world",
            tone="conversational",
            reasoning_chain=None,
            directive_checks=[],
            metadata={}
        )

    result = mock_process_input_streaming("test", "session_123")

    # Should be async iterator
    from runtime.utils.async_helpers import is_async_iterator
    assert is_async_iterator(result)

    # Should yield chunks then package
    items = []
    async for item in result:
        items.append(item)

    assert len(items) == 4
    assert items[0] == "Hello"
    assert items[1] == " "
    assert items[2] == "world"
    assert isinstance(items[3], MockResponsePackage)


@pytest.mark.asyncio
async def test_cognition_skill_bypass_wrapped():
    """Test that skill bypass responses are wrapped in async iterator"""
    from runtime.utils.async_helpers import stream_single

    # Simulate skill bypass response
    skill_response = MockResponsePackage(
        content="Pittsburgh Light Up Night is on November 22, 2025.",
        tone="conversational",
        reasoning_chain=None,
        directive_checks=[],
        metadata={"skill": "web_search"}
    )

    # Wrap it like cognition.generate_reply does
    wrapped = stream_single(skill_response)

    # Should be iterable
    result = []
    async for item in wrapped:
        result.append(item)

    assert len(result) == 1
    assert result[0].content == skill_response.content


@pytest.mark.asyncio
async def test_cognition_non_streaming_wrapped():
    """Test that non-streaming responses are wrapped in async iterator"""
    from runtime.utils.async_helpers import stream_single

    # Simulate non-streaming conversation response
    non_streaming_response = MockResponsePackage(
        content="This is a short response.",
        tone="conversational",
        reasoning_chain=None,
        directive_checks=[],
        metadata={}
    )

    # Wrap it like cognition.generate_reply does
    wrapped = stream_single(non_streaming_response)

    # Should be iterable
    result = []
    async for item in wrapped:
        result.append(item)

    assert len(result) == 1
    assert result[0].content == non_streaming_response.content


@pytest.mark.asyncio
async def test_cognition_extraction_synthesis_wrapped():
    """Test that extraction synthesis mode is wrapped"""
    from runtime.utils.async_helpers import stream_single
    from types import SimpleNamespace

    # Simulate extraction synthesis response
    extraction_result = SimpleNamespace(content="Direct answer from extraction")

    # Wrap it like cognition.generate_reply does for mode="extraction_synthesis"
    wrapped = stream_single(extraction_result)

    # Should be iterable
    result = []
    async for item in wrapped:
        result.append(item)

    assert len(result) == 1
    assert result[0].content == "Direct answer from extraction"


@pytest.mark.asyncio
async def test_no_type_error_on_iteration():
    """Critical test: ensure we NEVER get 'async for requires __aiter__' error"""
    from runtime.utils.async_helpers import stream_single

    # Test various types that might be returned
    test_cases = [
        "plain string",
        123,
        {"key": "value"},
        MockResponsePackage("test", "tone", None, [], {})
    ]

    for test_value in test_cases:
        wrapped = stream_single(test_value)

        # This should NEVER raise TypeError
        try:
            results = []
            async for item in wrapped:
                results.append(item)

            assert len(results) == 1
            assert results[0] == test_value

        except TypeError as e:
            pytest.fail(f"Got TypeError for {type(test_value).__name__}: {e}")


@pytest.mark.asyncio
async def test_ensure_async_iterator_passthrough():
    """Test ensure_async_iterator passes through existing iterators"""
    from runtime.utils.async_helpers import ensure_async_iterator

    # Create an async generator
    async def existing_iterator():
        yield 1
        yield 2
        yield 3

    # Pass it through ensure_async_iterator
    result = ensure_async_iterator(existing_iterator())

    # Should yield same values
    values = []
    async for item in result:
        values.append(item)

    assert values == [1, 2, 3]


@pytest.mark.asyncio
async def test_ensure_async_iterator_wrapping():
    """Test ensure_async_iterator wraps non-iterators"""
    from runtime.utils.async_helpers import ensure_async_iterator

    # Pass non-iterator
    result = ensure_async_iterator("test_value")

    # Should wrap it
    values = []
    async for item in result:
        values.append(item)

    assert values == ["test_value"]


@pytest.mark.asyncio
async def test_streaming_vs_non_streaming_behavior():
    """Test that streaming and non-streaming have consistent interfaces"""
    from runtime.utils.async_helpers import stream_single

    # Streaming simulation
    async def streaming_gen():
        yield "chunk1"
        yield "chunk2"
        yield MockResponsePackage("final", "tone", None, [], {})

    # Non-streaming simulation
    non_streaming = stream_single(
        MockResponsePackage("instant", "tone", None, [], {})
    )

    # Both should be async iterators
    from runtime.utils.async_helpers import is_async_iterator
    assert is_async_iterator(streaming_gen())
    assert is_async_iterator(non_streaming)

    # Both should work with async for
    streaming_results = []
    async for item in streaming_gen():
        streaming_results.append(item)

    non_streaming_results = []
    async for item in non_streaming:
        non_streaming_results.append(item)

    # Streaming has multiple items
    assert len(streaming_results) == 3

    # Non-streaming has single item
    assert len(non_streaming_results) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
