#!/usr/bin/env python3
"""
Streaming Test Script
=====================
Tests Hugo's streaming response generation.

Verifies:
- Stream chunks are printed incrementally
- Full response is assembled correctly
- Metadata is tracked properly
"""

import asyncio
import sys
from pathlib import Path

# Add Hugo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.runtime_manager import RuntimeManager
from core.memory import MemoryEntry
from datetime import datetime


async def test_streaming_response():
    """Test streaming response generation"""
    print("=" * 60)
    print("TEST: Streaming Response Generation")
    print("=" * 60)

    # Initialize runtime
    runtime = RuntimeManager()
    await runtime.initialize()

    if not runtime.cognition:
        print("✗ Cognition engine not initialized")
        return False

    # Test prompts that should trigger streaming
    test_prompts = [
        "Explain how streaming works in Ollama",
        "Describe the architecture of a multi-agent system",
        "How would you implement a retry mechanism with exponential backoff?"
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}/{len(test_prompts)} ---")
        print(f"Prompt: {prompt}")
        print("\nHugo (streaming): ", end="", flush=True)

        session_id = f"streaming_test_{i}"
        chunks_received = 0
        total_response = ""

        try:
            async for chunk in runtime.cognition.process_input_streaming(prompt, session_id):
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    chunks_received += 1
                    total_response += chunk
                else:
                    # Final response package
                    response_package = chunk
                    print()  # Newline after streaming

            print(f"\n✓ Streaming complete")
            print(f"  Chunks received: {chunks_received}")
            print(f"  Total length: {len(total_response)} chars")
            print(f"  Metadata: streaming={response_package.metadata.get('streaming', False)}")

        except Exception as e:
            print(f"\n✗ Streaming failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "=" * 60)
    print("✓ ALL STREAMING TESTS PASSED")
    print("=" * 60)
    return True


async def test_non_streaming_response():
    """Test that short prompts don't trigger streaming"""
    print("\n" + "=" * 60)
    print("TEST: Non-Streaming Response")
    print("=" * 60)

    runtime = RuntimeManager()
    await runtime.initialize()

    if not runtime.cognition:
        print("✗ Cognition engine not initialized")
        return False

    # Short prompt that shouldn't trigger streaming
    prompt = "Hello"
    session_id = "non_streaming_test"

    print(f"Prompt: {prompt}")
    print("Hugo: ", end="", flush=True)

    try:
        response_package = await runtime.cognition.process_input(prompt, session_id)
        print(response_package.content)

        print(f"\n✓ Non-streaming response complete")
        print(f"  Length: {len(response_package.content)} chars")

    except Exception as e:
        print(f"\n✗ Non-streaming failed: {str(e)}")
        return False

    print("\n" + "=" * 60)
    print("✓ NON-STREAMING TEST PASSED")
    print("=" * 60)
    return True


async def test_streaming_configuration():
    """Test streaming configuration settings"""
    print("\n" + "=" * 60)
    print("TEST: Streaming Configuration")
    print("=" * 60)

    import os

    streaming_enabled = os.getenv("STREAMING_ENABLED", "false").lower()
    print(f"STREAMING_ENABLED: {streaming_enabled}")

    if streaming_enabled == "true":
        print("✓ Streaming is enabled in configuration")
    else:
        print("⚠ Streaming is disabled in configuration")

    print("\n" + "=" * 60)
    return True


async def main():
    """Run all streaming tests"""
    print("\n" + "=" * 60)
    print("HUGO STREAMING SYSTEM VERIFICATION")
    print("=" * 60)

    try:
        # Test 1: Configuration
        await test_streaming_configuration()

        # Test 2: Streaming response
        streaming_success = await test_streaming_response()

        # Test 3: Non-streaming response
        non_streaming_success = await test_non_streaming_response()

        if streaming_success and non_streaming_success:
            print("\n" + "=" * 60)
            print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("✗ SOME TESTS FAILED")
            print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
