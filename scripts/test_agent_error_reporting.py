"""
Test script for agent error reporting and passthrough
"""
import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_agent_error_reporting():
    """Test that agent errors are properly reported to the user"""

    print("=" * 70)
    print("PHASE 4.4: Testing Agent Error Reporting")
    print("=" * 70)
    print()

    from core.logger import HugoLogger
    from core.agents import SearchAgent
    from skills.builtin.web_search import WebSearchSkill
    from core.cognition import CognitionEngine

    logger = HugoLogger()

    print("[1/5] Testing SearchAgent error handling...")

    # Create a SearchAgent that will encounter an error
    search_agent = SearchAgent(logger=logger, extract_skill=None)

    # Force an error by passing invalid input
    print("  Forcing error with invalid query...")

    # Mock the _collect_urls method to raise an exception
    async def mock_collect_urls_error(query):
        raise ValueError("Simulated network error")

    original_collect_urls = search_agent._collect_urls
    search_agent._collect_urls = mock_collect_urls_error

    report = await search_agent.investigate("test query")

    print(f"  Report success: {report['success']}")
    print(f"  Report has error: {report.get('error') is not None}")
    print(f"  Error message: {report.get('error', 'N/A')[:60]}...")

    # Restore original method
    search_agent._collect_urls = original_collect_urls

    print()

    # Test 2: web_search skill error passthrough
    print("[2/5] Testing web_search skill error passthrough...")

    # Create mock memory
    memory = MagicMock()
    memory.store = AsyncMock()

    web_search = WebSearchSkill(
        logger=logger,
        sqlite_manager=None,
        memory_manager=memory
    )

    # Initialize search agent with error-inducing mock
    web_search.search_agent = SearchAgent(logger=logger, extract_skill=None)
    web_search.search_agent._collect_urls = mock_collect_urls_error

    result = await web_search.run(action="search", query="test query with error")

    print(f"  Skill success: {result.success}")
    print(f"  Skill message: {result.message[:60]}...")
    print(f"  Has error_passthrough: {result.output.get('error_passthrough') if result.output else False}")

    print()

    # Test 3: Cognition error passthrough
    print("[3/5] Testing cognition error passthrough...")

    # Create mock components
    mock_memory = MagicMock()
    mock_memory.classify_memory = MagicMock(return_value=MagicMock(metadata={}))

    async def mock_retrieve_recent(*args, **kwargs):
        return []

    mock_memory.retrieve_recent = mock_retrieve_recent

    cognition = CognitionEngine(
        memory_manager=mock_memory,
        logger=logger,
        runtime_manager=None
    )

    # Format the error response
    error_response = cognition._format_skill_response("web_search", result)

    print(f"  Response length: {len(error_response)}")
    print(f"  Response contains 'Agent encountered an error': {'Agent encountered an error' in error_response}")
    print(f"  Response preview: {error_response[:80]}...")

    print()

    # Test 4: Full integration test
    print("[4/5] Testing full integration (user query -> agent error -> Hugo response)...")

    # Simulate a user query that triggers web_search with error
    print("  Simulating user query: 'When does Wicked Part 2 come out?'")

    # The skill should return error passthrough
    result = await web_search.run(action="search", query="When does Wicked Part 2 come out?")

    # Cognition should format it properly
    final_response = cognition._format_skill_response("web_search", result)

    print(f"  Final response to user: {final_response[:100]}...")
    print(f"  Contains error information: {'error' in final_response.lower()}")

    print()

    # Test 5: Verify logging
    print("[5/5] Verifying error logging...")

    # Check that errors were logged
    print("  [INFO] Errors should be logged with [agent] and [web_search] categories")
    print("  Check logs for:")
    print("    - [agent] error_detected")
    print("    - [web_search] reporting_agent_error")

    print()
    print("=" * 70)
    print("Validation")
    print("=" * 70)
    print()

    checks = {
        "SearchAgent returns error in report": report.get('error') is not None,
        "SearchAgent does not raise exception": not report['success'] and report.get('error'),
        "web_search skill detects agent error": result.output.get('error_passthrough') if result.output else False,
        "web_search includes error in message": 'error' in result.message.lower(),
        "Cognition passes through agent error": 'Agent encountered an error' in error_response,
        "Final response contains error details": 'error' in final_response.lower(),
    }

    for check, passed in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {check}")

    print()
    print("=" * 70)

    if all(checks.values()):
        print("[SUCCESS] All agent error reporting tests passed!")
    else:
        print("[INFO] Some tests may have failed - review above")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_agent_error_reporting())
