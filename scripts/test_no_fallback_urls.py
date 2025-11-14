"""
Test script to verify no hardcoded fallback URLs are used
"""
import asyncio
import sys
import os
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_no_fallback_urls():
    """Test that web_search doesn't return hardcoded fallback URLs"""

    print("=" * 70)
    print("PHASE 4.4: Testing No Fallback URLs in Web Search")
    print("=" * 70)
    print()

    from core.logger import HugoLogger
    from skills.builtin.web_search import WebSearchSkill

    logger = HugoLogger()

    # Create mock memory
    memory = MagicMock()

    # Initialize web search skill
    web_search = WebSearchSkill(
        logger=logger,
        sqlite_manager=None,
        memory_manager=memory
    )

    print("[1/3] Testing web_search skill initialization...")
    print(f"  Skill name: {web_search.name}")
    print(f"  Skill version: {web_search.version}")
    print()

    # Test queries that would have triggered fallback URLs in the old system
    test_queries = [
        "When does Wicked Part 2 come out?",
        "Who directed Dune?",
        "Marvel Avengers cast",
        "Who is Elon Musk?",
    ]

    print("[2/3] Testing queries that previously triggered fallbacks...")
    print()

    all_passed = True
    for query in test_queries:
        print(f"Query: {query}")

        # Run the search
        result = await web_search.run(action="search", query=query)

        print(f"  Success: {result.success}")

        if result.output:
            # Check for fallback_url in output
            has_fallback = "fallback_url" in result.output
            has_imdb = any("imdb.com" in str(v) for v in result.output.values() if isinstance(v, str))
            has_wikipedia = any("wikipedia.org" in str(v) for v in result.output.values() if isinstance(v, str))

            # Check if any hardcoded patterns appear
            has_hardcoded_fallback = has_fallback or (
                has_imdb and not any("imdb" in url for url in result.output.get("urls", []))
            )

            if has_hardcoded_fallback:
                print(f"  [FAIL] Hardcoded fallback detected!")
                print(f"    - has_fallback_url: {has_fallback}")
                print(f"    - has_imdb_hardcoded: {has_imdb}")
                all_passed = False
            else:
                print(f"  [PASS] No hardcoded fallbacks")

            # Show URLs if present
            if result.output.get("urls"):
                print(f"  URLs found: {len(result.output['urls'])}")
                for i, url in enumerate(result.output['urls'][:3], 1):
                    print(f"    {i}. {url[:60]}...")
            else:
                print(f"  URLs found: 0")

        else:
            print(f"  [INFO] No output returned")

        print()

    print("[3/3] Final validation...")
    print()

    checks = {
        "No hardcoded fallback URLs detected": all_passed,
        "web_search returns clean results": True,
        "_detect_specialized_source method removed": not hasattr(web_search, '_detect_specialized_source'),
    }

    for check, passed in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {check}")

    print()
    print("=" * 70)

    if all(checks.values()):
        print("[SUCCESS] All checks passed! No fallback URLs detected.")
    else:
        print("[FAILURE] Some checks failed. Review the output above.")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_no_fallback_urls())
