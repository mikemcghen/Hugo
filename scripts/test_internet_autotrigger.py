"""
Test internet query auto-triggering via generate_reply().

This test verifies that:
1. Internet queries are classified with skill_trigger="web_search"
2. generate_reply() detects the trigger and bypasses LLM
3. The web_search skill is executed
4. Results are returned without LLM processing
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.memory import MemoryManager
from core.cognition import CognitionEngine
from data.sqlite_manager import SQLiteManager
from core.logger import HugoLogger
from skills.skill_manager import SkillManager
from core.runtime_manager import RuntimeManager


async def test_internet_query_bypass():
    """Test that internet queries trigger web_search skill and bypass LLM."""

    print("\n" + "=" * 70)
    print("INTERNET QUERY AUTO-TRIGGER TEST")
    print("=" * 70 + "\n")

    # Initialize components
    logger = HugoLogger()
    sqlite = SQLiteManager()
    await sqlite.initialize()
    print("[OK] SQLite manager initialized")

    memory = MemoryManager(sqlite, logger)
    await memory.initialize()
    print("[OK] Memory manager initialized")

    skill_manager = SkillManager(logger, sqlite, memory)
    skill_manager.load_skills()
    print(f"[OK] Skill manager loaded with {skill_manager.registry.count()} skills")

    # Create runtime manager
    runtime = RuntimeManager(logger)
    runtime.skills = skill_manager

    cognition = CognitionEngine(memory, logger, runtime)
    print("[OK] Cognition engine initialized\n")

    # Test 1: Classification check
    print("-" * 70)
    print("TEST 1: Memory classification for internet query")
    print("-" * 70)

    test_query = "When does Wicked Part 2 come out?"
    print(f"Query: {test_query}\n")

    classification = memory.classify_memory(test_query)

    print(f"Classification:")
    print(f"  - Type: {classification.memory_type}")
    print(f"  - Reasoning: {classification.reasoning}")
    print(f"  - Skill trigger: {classification.metadata.get('skill_trigger')}")
    print(f"  - Skill action: {classification.metadata.get('skill_action')}")
    print(f"  - Skill payload: {classification.metadata.get('skill_payload')}\n")

    if classification.metadata.get('skill_trigger') == 'web_search':
        print("[PASS] ✓ Query classified as web_search trigger\n")
    else:
        print("[FAIL] ✗ Query NOT classified as web_search trigger\n")
        return False

    # Test 2: generate_reply() bypass check
    print("-" * 70)
    print("TEST 2: generate_reply() skill bypass")
    print("-" * 70)
    print(f"Calling: cognition.generate_reply('{test_query}')\n")

    try:
        response = await cognition.generate_reply(
            message=test_query,
            session_id="test_internet_session",
            streaming=False
        )

        print(f"Response received:")
        print(f"  - Content: {response.content[:100]}...")
        print(f"  - Metadata engine: {response.metadata.get('engine')}")
        print(f"  - Bypassed LLM: {response.metadata.get('bypassed_llm')}")
        print(f"  - Skill: {response.metadata.get('skill')}\n")

        if response.metadata.get('bypassed_llm'):
            print("[PASS] ✓ LLM was bypassed\n")
        else:
            print("[FAIL] ✗ LLM was NOT bypassed\n")
            return False

        if response.metadata.get('skill') == 'web_search':
            print("[PASS] ✓ web_search skill was executed\n")
        else:
            print(f"[FAIL] ✗ Wrong skill executed: {response.metadata.get('skill')}\n")
            return False

    except Exception as e:
        print(f"[ERROR] generate_reply() failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Verify NO LLM processing occurred
    print("-" * 70)
    print("TEST 3: Verify LLM was not called")
    print("-" * 70)

    # Check logs for ollama calls
    import json
    logs_path = project_root / "data" / "logs" / "structured.jsonl"

    if logs_path.exists():
        with open(logs_path, 'r') as f:
            lines = f.readlines()

        # Get last 50 lines
        recent_logs = lines[-50:]

        ollama_calls = 0
        internet_query_detected = 0
        skill_bypass = 0

        for line in recent_logs:
            try:
                log = json.loads(line)
                if log.get('event_type') == 'ollama_inference_attempt_async':
                    ollama_calls += 1
                if log.get('event_type') == 'internet_query_detected':
                    internet_query_detected += 1
                if log.get('event_type') == 'skill_bypass_started':
                    skill_bypass += 1
            except:
                continue

        print(f"Recent log events:")
        print(f"  - internet_query_detected: {internet_query_detected}")
        print(f"  - skill_bypass_started: {skill_bypass}")
        print(f"  - ollama_inference_attempt: {ollama_calls}\n")

        if internet_query_detected > 0:
            print("[PASS] ✓ Internet query was detected\n")
        else:
            print("[WARN] Internet query detection not logged\n")

        if ollama_calls == 0:
            print("[PASS] ✓ Ollama was NOT called (LLM bypass successful)\n")
        else:
            print(f"[FAIL] ✗ Ollama was called {ollama_calls} times (bypass failed)\n")
            return False

    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70 + "\n")

    print("Summary:")
    print("  1. ✓ Internet queries are classified with skill_trigger='web_search'")
    print("  2. ✓ generate_reply() detects trigger and bypasses LLM")
    print("  3. ✓ web_search skill is executed")
    print("  4. ✓ Ollama is NOT called (no hallucination risk)")
    print()

    return True


if __name__ == "__main__":
    result = asyncio.run(test_internet_query_bypass())
    sys.exit(0 if result else 1)
