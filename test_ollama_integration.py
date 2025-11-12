#!/usr/bin/env python3
"""
Test Ollama Integration
=======================
Quick test to verify Hugo can generate responses using Ollama.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_ollama_connection():
    """Test direct connection to Ollama API"""
    print("=" * 60)
    print("TEST 1: Ollama API Connection")
    print("=" * 60)

    import requests
    from dotenv import load_dotenv
    load_dotenv()

    ollama_api = os.getenv("OLLAMA_API", "http://localhost:11434/api/generate")
    model_name = os.getenv("MODEL_NAME", "llama3:8b")

    print(f"API: {ollama_api}")
    print(f"Model: {model_name}")
    print()

    try:
        print("Sending test prompt to Ollama...")
        response = requests.post(
            ollama_api,
            json={
                "model": model_name,
                "prompt": "Say hello in one friendly sentence.",
                "stream": False
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            reply = result.get("response", "").strip()
            print(f"✓ Ollama Response: {reply}")
            print()
            return True
        else:
            print(f"✗ Error: Status {response.status_code}")
            return False

    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print()
        print("Make sure Ollama is running:")
        print("  1. ollama serve")
        print("  2. ollama pull llama3:8b")
        return False


def test_cognition_engine():
    """Test CognitionEngine with Ollama"""
    print("=" * 60)
    print("TEST 2: CognitionEngine Integration")
    print("=" * 60)

    try:
        import asyncio
        from core.cognition import CognitionEngine
        from core.logger import HugoLogger

        # Create minimal mocks
        class MockMemory:
            async def retrieve_recent(self, session_id, limit=20):
                return []

            async def search_semantic(self, query, limit=10):
                return []

        class MockDirectives:
            pass

        logger = HugoLogger()
        memory = MockMemory()
        directives = MockDirectives()

        print("Initializing CognitionEngine...")
        engine = CognitionEngine(memory, directives, logger)
        print(f"✓ Engine configured for: {engine.model_engine} / {engine.model_name}")
        print()

        async def test_inference():
            print("Processing test input: 'Hello Hugo, introduce yourself'")
            print("(This will call Ollama...)")
            print()

            response = await engine.process_input(
                "Hello Hugo, introduce yourself briefly",
                "test-session-001"
            )

            print("=" * 60)
            print("RESPONSE FROM HUGO:")
            print("=" * 60)
            print(response.content)
            print()
            print("=" * 60)
            print(f"Model: {response.metadata.get('model')}")
            print(f"Engine: {response.metadata.get('engine')}")
            print(f"Confidence: {response.metadata.get('confidence')}")
            print(f"Tone: {response.tone.value}")
            print()

            return True

        result = asyncio.run(test_inference())
        return result

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print()
    print("#" * 60)
    print("# Hugo Ollama Integration Test")
    print("#" * 60)
    print()

    results = []

    # Test 1: Direct Ollama connection
    results.append(("Ollama API", test_ollama_connection()))

    if results[0][1]:
        # Test 2: CognitionEngine integration
        results.append(("CognitionEngine", test_cognition_engine()))
    else:
        print("Skipping CognitionEngine test (Ollama not available)")
        results.append(("CognitionEngine", False))

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")

    print()

    if all(r[1] for r in results):
        print("🎉 All tests passed! Hugo is ready to chat.")
        print()
        print("Start Hugo with: python -m runtime.cli shell")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
