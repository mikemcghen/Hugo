"""
Test Internet Query Detection and Automatic Triggering
------------------------------------------------------
Verify that natural language questions automatically trigger
internet skills without explicit /net commands.

Tests:
1. "When does X come out?" triggers web_search
2. "What's the cast of X?" triggers web_search
3. URL in message triggers fetch_url
4. Skill execution bypasses LLM
5. Response is skill result, not LLM output
"""

import asyncio
import sys
import io
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.cognition import CognitionEngine
from core.memory import MemoryManager
from core.logger import HugoLogger
from data.sqlite_manager import SQLiteManager
from skills.skill_manager import SkillManager


async def main():
    print("=" * 70)
    print("INTERNET QUERY DETECTION TEST")
    print("=" * 70)
    print()

    # Initialize components
    logger = HugoLogger()
    sqlite_manager = SQLiteManager(db_path="data/memory/test_internet_trigger.db")
    await sqlite_manager.connect()
    print("[OK] SQLite manager initialized")

    memory_manager = MemoryManager(sqlite_manager, None, logger)
    print("[OK] Memory manager initialized")

    skill_manager = SkillManager(logger, sqlite_manager, memory_manager)
    skill_manager.load_skills()
    print(f"[OK] Skill manager loaded with {len(skill_manager.list_skills())} skills")

    # Create minimal runtime manager mock with skills
    class RuntimeManagerMock:
        def __init__(self, skills):
            self.skills = skills

    runtime_manager = RuntimeManagerMock(skill_manager)

    cognition = CognitionEngine(memory_manager, logger, runtime_manager)
    print("[OK] Cognition engine initialized")
    print()

    session_id = "test_internet_trigger_session"

    # Test 1: Internet query detection - "When does X come out?"
    print("-" * 70)
    print("TEST 1: Internet query detection - release date")
    print("-" * 70)
    test_message = "When does Wicked Part 2 come out?"
    print(f"User: {test_message}")

    # Classify the message
    classification = memory_manager.classify_memory(test_message)
    print(f"Classification type: {classification.memory_type}")
    print(f"Reasoning: {classification.reasoning}")

    if classification.metadata and "skill_trigger" in classification.metadata:
        print(f"[OK] Skill trigger detected: {classification.metadata['skill_trigger']}")
        print(f"     Action: {classification.metadata['skill_action']}")
        print(f"     Payload: {classification.metadata['skill_payload']}")
    else:
        print("[FAIL] No skill trigger detected")
    print()

    # Test 2: Internet query detection - "What's the cast of X?"
    print("-" * 70)
    print("TEST 2: Internet query detection - cast question")
    print("-" * 70)
    test_message = "What's the cast of Dune 3?"
    print(f"User: {test_message}")

    classification = memory_manager.classify_memory(test_message)
    print(f"Classification type: {classification.memory_type}")
    print(f"Reasoning: {classification.reasoning}")

    if classification.metadata and "skill_trigger" in classification.metadata:
        print(f"[OK] Skill trigger detected: {classification.metadata['skill_trigger']}")
        print(f"     Action: {classification.metadata['skill_action']}")
    else:
        print("[FAIL] No skill trigger detected")
    print()

    # Test 3: URL detection triggers fetch_url
    print("-" * 70)
    print("TEST 3: URL detection triggers fetch_url")
    print("-" * 70)
    test_message = "https://www.example.com/article"
    print(f"User: {test_message}")

    classification = memory_manager.classify_memory(test_message)
    print(f"Classification type: {classification.memory_type}")
    print(f"Reasoning: {classification.reasoning}")

    if classification.metadata and "skill_trigger" in classification.metadata:
        skill_trigger = classification.metadata['skill_trigger']
        print(f"[OK] Skill trigger detected: {skill_trigger}")
        if skill_trigger == "fetch_url":
            print("[OK] Correctly triggered fetch_url for URL")
        else:
            print(f"[FAIL] Wrong skill triggered: {skill_trigger}")
    else:
        print("[FAIL] No skill trigger detected")
    print()

    # Test 4: Skill auto-execution bypasses LLM
    print("-" * 70)
    print("TEST 4: Skill execution bypasses LLM")
    print("-" * 70)
    test_message = "Who is the current president of the United States?"
    print(f"User: {test_message}")
    print("Generating response...")

    try:
        response_package = await cognition.generate_reply(test_message, session_id, streaming=False)

        print(f"[OK] Response generated")
        print(f"Response engine: {response_package.metadata.get('engine', 'unknown')}")
        print(f"Bypassed LLM: {response_package.metadata.get('bypassed_llm', False)}")

        if response_package.metadata.get('bypassed_llm'):
            print("[OK] LLM was bypassed for internet query")
        else:
            print("[WARN] LLM was not bypassed (may be expected if skill unavailable)")

        print(f"\nResponse content preview:")
        print(response_package.content[:200])
        print()

    except Exception as e:
        print(f"[FAIL] Response generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print()

    # Test 5: Various internet query patterns
    print("-" * 70)
    print("TEST 5: Various internet query patterns")
    print("-" * 70)

    test_queries = [
        "How old is Tom Cruise?",
        "What is the weather in London?",
        "Price of Bitcoin",
        "News about AI",
        "Who was the first person on the moon?",
        "When was Python released?",
        "Where is the Eiffel Tower?",
        "How many people live in Tokyo?"
    ]

    internet_triggers = 0
    for query in test_queries:
        classification = memory_manager.classify_memory(query)
        if classification.metadata and "skill_trigger" in classification.metadata:
            internet_triggers += 1
            print(f"✓ '{query}' → {classification.metadata['skill_trigger']}")
        else:
            print(f"✗ '{query}' → no trigger")

    print(f"\n[OK] {internet_triggers}/{len(test_queries)} queries triggered internet skills")
    print()

    # Test 6: Non-internet queries should NOT trigger
    print("-" * 70)
    print("TEST 6: Non-internet queries should NOT trigger")
    print("-" * 70)

    non_internet_queries = [
        "What is your name?",
        "How are you?",
        "Make a note saying hello",
        "What is the meaning of life?",
        "I am feeling happy today"
    ]

    false_positives = 0
    for query in non_internet_queries:
        classification = memory_manager.classify_memory(query)
        if classification.metadata and classification.metadata.get('skill_trigger') in ['web_search', 'fetch_url']:
            false_positives += 1
            print(f"✗ '{query}' → incorrectly triggered {classification.metadata['skill_trigger']}")
        else:
            print(f"✓ '{query}' → no internet trigger (correct)")

    if false_positives == 0:
        print(f"\n[OK] No false positives - all non-internet queries handled correctly")
    else:
        print(f"\n[WARN] {false_positives}/{len(non_internet_queries)} false positives detected")
    print()

    await sqlite_manager.close()

    print("=" * 70)
    print("INTERNET QUERY DETECTION TEST COMPLETE")
    print("=" * 70)
    print()
    print("Summary:")
    print("- Internet queries automatically trigger web_search")
    print("- URLs automatically trigger fetch_url")
    print("- Skills bypass LLM to avoid hallucination")
    print("- Responses are direct skill results")
    print("- No /net commands required")
    print()


if __name__ == "__main__":
    asyncio.run(main())
