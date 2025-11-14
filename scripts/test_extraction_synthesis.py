"""
Test script for extraction synthesis mode
"""
import asyncio
import sys
import os
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cognition import CognitionEngine
from core.logger import HugoLogger


async def test_extraction_synthesis():
    """Test the extraction synthesis mode"""

    print("Initializing Hugo components...")

    # Initialize logger
    logger = HugoLogger()

    # Create a minimal mock memory manager
    memory = MagicMock()
    memory.classify_memory = MagicMock(return_value=MagicMock(metadata={}))

    async def mock_retrieve_recent(*args, **kwargs):
        return []
    memory.retrieve_recent = mock_retrieve_recent

    # Initialize cognition engine
    cognition = CognitionEngine(
        memory_manager=memory,
        logger=logger,
        runtime_manager=None
    )

    print("[OK] Components initialized\n")

    # Test 1: Simple extraction synthesis
    print("=" * 60)
    print("TEST 1: Extraction Synthesis Mode")
    print("=" * 60)

    test_prompt = """Based on the extracted text below, answer the user's question:

User question: When does Wicked Part 2 come out?
Extracted info: Wicked Part Two is scheduled to be released on November 21, 2025. The film is a musical fantasy adaptation directed by Jon M. Chu. It is the second and final installment of the two-part film adaptation of the Broadway musical Wicked.

Respond with a direct, concise answer in 2-3 sentences. If no answer exists in the text, say 'No clear information available.'"""

    print(f"Prompt:\n{test_prompt}\n")
    print("Calling cognition.generate_reply with mode='extraction_synthesis'...\n")

    response = await cognition.generate_reply(
        message=test_prompt,
        session_id="test_extraction",
        streaming=False,
        mode="extraction_synthesis"
    )

    print(f"Response type: {type(response)}")
    print(f"Response content:\n{response.content}\n")

    # Check results
    print("=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)

    checks = {
        "Short answer (< 500 chars)": len(response.content) < 500,
        "Non-empty response": len(response.content) > 0,
        "No personality detected": "I" not in response.content and "Hugo" not in response.content,
        "Factual tone": not any(word in response.content.lower() for word in ["i think", "in my opinion", "personally"])
    }

    for check, passed in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {check}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_extraction_synthesis())
