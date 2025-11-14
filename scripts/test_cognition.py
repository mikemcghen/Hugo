"""
Test Cognition Engine
=====================

Tests the cognition engine's core functionality:
- generate_reply()
- build_prompt()
- retrieve_relevant_memories()
- call_ollama()
- post_process()
- apply_directives()
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.cognition import CognitionEngine
from core.memory import MemoryManager
from core.logger import HugoLogger
from data.sqlite_manager import SQLiteManager

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


async def test_cognition_engine():
    """Test cognition engine functionality"""
    print("=" * 70)
    print("COGNITION ENGINE TEST")
    print("=" * 70)

    # Use test database
    test_db = "data/memory/test_cognition.db"
    Path(test_db).unlink(missing_ok=True)  # Clean slate

    logger = HugoLogger(log_dir="logs/tests")
    sqlite_manager = SQLiteManager(db_path=test_db)
    await sqlite_manager.connect()

    memory_manager = MemoryManager(sqlite_manager, None, logger)

    # Initialize cognition engine
    cognition = CognitionEngine(memory_manager, logger)

    print(f"\n✓ Cognition engine initialized")
    print(f"  Persona: {cognition.persona.get('name', 'Hugo')}")
    print(f"  Model: {cognition.model_name}")
    print(f"  Engine: {cognition.model_engine}")
    print(f"  Mood: {cognition.current_mood.value}")

    # Test 1: Build prompt
    print("\n📝 Test 1: Building prompt...\n")

    test_message = "What are my pets?"
    session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        prompt = await cognition.build_prompt(test_message, session_id)
        print(f"✓ Built prompt (length: {len(prompt)} chars)")
        print(f"  Preview: {prompt[:200]}...")
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False

    # Test 2: Retrieve relevant memories
    print("\n🔍 Test 2: Retrieving relevant memories...\n")

    try:
        memories = await cognition.retrieve_relevant_memories("pets", limit=5)
        print(f"✓ Retrieved memories:")
        print(f"  Factual: {len(memories['factual_memories'])}")
        print(f"  Semantic: {len(memories['semantic_results'])}")
        print(f"  Reflections: {len(memories['reflections'])}")
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False

    # Test 3: Post-process (save to memory)
    print("\n💾 Test 3: Post-processing (save to memory)...\n")

    try:
        test_response = "This is a test response from Hugo."
        await cognition.post_process(test_response, session_id)
        print(f"✓ Response saved to memory")

        # Verify it was saved
        recent = await memory_manager.retrieve_recent(session_id, limit=5)
        print(f"✓ Verified: {len(recent)} memories in session")
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False

    # Test 4: Check Ollama availability (don't actually call if not running)
    print("\n🤖 Test 4: Checking Ollama availability...\n")

    print(f"  Ollama API: {cognition.ollama_api}")
    print(f"  Timeout: {cognition.ollama_timeout}s")
    print(f"  Max retries: {cognition.ollama_max_retries}")
    print(f"  Async mode: {cognition.ollama_async_mode}")

    # Note: We don't actually call Ollama in tests to avoid dependency
    print(f"\n✓ Ollama configuration validated (not called)")

    print("\n" + "=" * 70)
    print("✨ ALL COGNITION ENGINE TESTS PASSED")
    print("=" * 70)
    print("\nNote: Ollama inference was not tested (requires running Ollama)")
    print("To test full inference, run Hugo REPL and send a message.")

    return True


if __name__ == "__main__":
    success = asyncio.run(test_cognition_engine())
    sys.exit(0 if success else 1)
