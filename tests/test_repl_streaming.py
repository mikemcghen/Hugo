"""
Tests for REPL streaming functionality
---------------------------------------
Ensures REPL correctly handles both streaming and non-streaming responses
without raising 'async for' type errors.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any


# Mock ResponsePackage for testing
@dataclass
class MockResponsePackage:
    content: str
    tone: str = "conversational"
    reasoning_chain: Any = None
    directive_checks: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.directive_checks is None:
            self.directive_checks = []
        if self.metadata is None:
            self.metadata = {
                "persona_name": "Hugo",
                "mood": "conversational",
                "user_sentiment": "neutral",
                "tone_adjustment": "conversational",
                "conversation_turns": 0,
                "semantic_memories": 0,
                "confidence": 0.85
            }


async def mock_streaming_generator(chunks: List[str], final_package: MockResponsePackage):
    """Mock streaming response generator"""
    for chunk in chunks:
        yield chunk
    yield final_package


async def mock_single_shot_generator(package: MockResponsePackage):
    """Mock single-shot response generator"""
    yield package


@pytest.mark.asyncio
async def test_repl_streaming_normal():
    """Test REPL handles normal streaming responses correctly"""
    # Arrange
    chunks = ["Hello", " ", "world", "!"]
    final_package = MockResponsePackage(content="Hello world!")

    # Simulate what cognition.generate_reply returns
    reply_iterator = mock_streaming_generator(chunks, final_package)

    # Act
    collected_chunks = []
    response_package = None

    async for chunk in reply_iterator:
        if isinstance(chunk, str):
            collected_chunks.append(chunk)
        else:
            response_package = chunk

    # Assert
    assert collected_chunks == chunks
    assert response_package is not None
    assert response_package.content == "Hello world!"


@pytest.mark.asyncio
async def test_repl_skill_bypass_single_shot():
    """Test REPL handles skill bypass (non-streaming) responses correctly"""
    # Arrange
    final_package = MockResponsePackage(
        content="Pittsburgh Light Up Night is on Saturday, November 22, 2025."
    )

    # Simulate what cognition.generate_reply returns for skill bypass
    reply_iterator = mock_single_shot_generator(final_package)

    # Act
    collected_chunks = []
    response_package = None

    async for chunk in reply_iterator:
        if isinstance(chunk, str):
            collected_chunks.append(chunk)
        else:
            response_package = chunk

    # Assert
    assert collected_chunks == []  # No string chunks for single-shot
    assert response_package is not None
    assert "Pittsburgh Light Up Night" in response_package.content


@pytest.mark.asyncio
async def test_repl_no_crash_on_responsepackage():
    """Test that REPL never crashes with 'async for' type error"""
    # Arrange
    final_package = MockResponsePackage(content="Test response")

    # This is the key test: even if we get a ResponsePackage directly,
    # it should be wrapped in an async iterator
    from runtime.utils.async_helpers import stream_single

    reply_iterator = stream_single(final_package)

    # Act & Assert - should not raise TypeError
    try:
        async for chunk in reply_iterator:
            assert isinstance(chunk, MockResponsePackage)
            assert chunk.content == "Test response"
    except TypeError as e:
        pytest.fail(f"Got TypeError when iterating: {e}")


@pytest.mark.asyncio
async def test_repl_unified_interface():
    """Test that both streaming and non-streaming use the same async for loop"""
    # Arrange
    streaming_chunks = ["chunk1", "chunk2"]
    streaming_package = MockResponsePackage(content="chunk1chunk2")
    non_streaming_package = MockResponsePackage(content="instant response")

    from runtime.utils.async_helpers import stream_single

    # Act - Test streaming
    streaming_result = []
    async for chunk in mock_streaming_generator(streaming_chunks, streaming_package):
        streaming_result.append(chunk)

    # Act - Test non-streaming
    non_streaming_result = []
    async for chunk in stream_single(non_streaming_package):
        non_streaming_result.append(chunk)

    # Assert
    assert len(streaming_result) == 3  # 2 chunks + 1 package
    assert len(non_streaming_result) == 1  # 1 package only
    assert isinstance(non_streaming_result[0], MockResponsePackage)


@pytest.mark.asyncio
async def test_async_helpers_stream_single():
    """Test stream_single utility function"""
    from runtime.utils.async_helpers import stream_single

    # Arrange
    test_value = "test"

    # Act
    results = []
    async for item in stream_single(test_value):
        results.append(item)

    # Assert
    assert results == ["test"]
    assert len(results) == 1


@pytest.mark.asyncio
async def test_async_helpers_ensure_async_iterator():
    """Test ensure_async_iterator utility function"""
    from runtime.utils.async_helpers import ensure_async_iterator

    # Test with non-iterator
    result1 = []
    async for item in ensure_async_iterator("plain string"):
        result1.append(item)
    assert result1 == ["plain string"]

    # Test with async iterator
    async def async_gen():
        yield 1
        yield 2
        yield 3

    result2 = []
    async for item in ensure_async_iterator(async_gen()):
        result2.append(item)
    assert result2 == [1, 2, 3]


@pytest.mark.asyncio
async def test_async_helpers_is_async_iterator():
    """Test is_async_iterator utility function"""
    from runtime.utils.async_helpers import is_async_iterator

    # Async iterator
    async def async_gen():
        yield 1

    gen = async_gen()
    assert is_async_iterator(gen) is True

    # Not an async iterator
    assert is_async_iterator("string") is False
    assert is_async_iterator(123) is False
    assert is_async_iterator([1, 2, 3]) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
