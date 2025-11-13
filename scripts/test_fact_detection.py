"""
Test Fact Detection - Questions vs Declarations
================================================

Tests the improved _detect_facts() logic to ensure:
1. Questions are never stored as facts (even if they mention animals, people, etc.)
2. Declarative sentences are properly detected as facts
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.memory import MemoryManager
from core.logger import HugoLogger

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def test_fact_detection():
    """Test fact detection logic with various inputs"""
    print("=" * 70)
    print("FACT DETECTION TEST")
    print("=" * 70)

    logger = HugoLogger(log_dir="logs/tests")
    memory = MemoryManager(None, None, logger)

    # Test cases: (message, expected_is_fact, expected_entity_type, category)
    test_cases = [
        # Questions - should NOT be detected as facts
        ("Do you remember my pets names?", False, None, "question-?"),
        ("Hi there Hugo! Do you remember my pets names?", False, None, "question-do-you"),
        ("Can you tell me about my cats?", False, None, "question-can-you"),
        ("Will you remember this?", False, None, "question-will-you"),
        ("Would you like to know about my dog?", False, None, "question-would-you"),
        ("What are my pets?", False, None, "question-what"),
        ("Where do I live?", False, None, "question-where"),
        ("How many cats do I have?", False, None, "question-how"),

        # Declarative sentences - SHOULD be detected as facts
        ("I have 2 cats Will and Tonks and 2 bunnies Oswald and Keely", True, "animal", "declaration"),
        ("I have two bunnies named Oswald and Keely", True, "animal", "declaration"),
        ("My cats are Will and Tonks", True, "animal", "declaration"),
        ("I live in San Francisco", True, "location", "declaration"),
        ("My favorite programming language is Python", True, "preference", "declaration"),
        ("I own a red Tesla Model 3", True, "possession", "declaration"),
        ("I work as a software engineer", True, "occupation", "declaration"),

        # Edge cases
        ("I have a question about my pets", True, "possession", "edge-case-have"),  # "have" triggers possession
        ("Tell me, do I have any pets?", False, None, "edge-case-question"),  # Still a question
    ]

    passed = 0
    failed = 0

    print("\n🧪 Testing fact detection logic...\n")

    for message, expected_fact, expected_entity, category in test_cases:
        is_fact, entity_type = memory._detect_facts(message)

        # Check if result matches expectation
        if is_fact == expected_fact and entity_type == expected_entity:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1

        # Display result
        fact_label = f"FACT:{entity_type}" if is_fact and entity_type else "NOT FACT" if not is_fact else "FACT"
        print(f"{status} [{category}]")
        print(f"     Message: '{message[:60]}{'...' if len(message) > 60 else ''}'")
        print(f"     Expected: {('FACT:' + expected_entity) if expected_fact else 'NOT FACT'}")
        print(f"     Got: {fact_label}")
        print()

    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\n✨ All tests passed! Fact detection is working correctly.")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed. Review the logic.")
        return False


if __name__ == "__main__":
    success = test_fact_detection()
    sys.exit(0 if success else 1)
