"""
Async Helpers
-------------
Utilities for handling async iteration and streaming responses uniformly.
"""

from typing import AsyncIterator, Any, TypeVar

T = TypeVar('T')


async def stream_single(value: T) -> AsyncIterator[T]:
    """
    Wrap a single value in an async iterator.

    This allows non-streaming responses to be treated uniformly
    with streaming responses in async for loops.

    Args:
        value: Single value to yield

    Yields:
        The value once

    Example:
        response = ResponsePackage(...)
        async for chunk in stream_single(response):
            # chunk will be the ResponsePackage
            pass
    """
    yield value


async def ensure_async_iterator(obj: Any) -> AsyncIterator[Any]:
    """
    Ensure an object is an async iterator, wrapping if necessary.

    If obj is already an async iterator, return it as-is.
    If obj is not an async iterator, wrap it with stream_single().

    Args:
        obj: Object that may or may not be an async iterator

    Returns:
        Async iterator

    Example:
        result = await cognition.generate_reply(...)
        async for chunk in ensure_async_iterator(result):
            print(chunk)
    """
    # Check if it's an async iterator
    if hasattr(obj, '__aiter__'):
        # It's already an async iterator, return as-is
        async for item in obj:
            yield item
    else:
        # Not an async iterator, wrap it
        yield obj


def is_async_iterator(obj: Any) -> bool:
    """
    Check if an object is an async iterator.

    Args:
        obj: Object to check

    Returns:
        True if obj has __aiter__ method
    """
    return hasattr(obj, '__aiter__')
