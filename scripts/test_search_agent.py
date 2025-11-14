"""
Test script for SearchAgent autonomous multi-source investigation
"""
import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_search_agent():
    """Test SearchAgent multi-source investigation capabilities"""

    print("=" * 70)
    print("PHASE 4.4: Testing Autonomous SearchAgent")
    print("=" * 70)
    print()

    from core.logger import HugoLogger
    from core.agents import SearchAgent
    from skills.builtin.extract_and_answer import ExtractAndAnswerSkill

    logger = HugoLogger()

    # Initialize extract_and_answer skill for content extraction
    extract_skill = ExtractAndAnswerSkill(
        logger=logger,
        sqlite_manager=None,
        memory_manager=None
    )

    # Initialize SearchAgent
    search_agent = SearchAgent(
        logger=logger,
        extract_skill=extract_skill
    )

    print("[1/6] Testing SearchAgent initialization...")
    print(f"  Agent created: {search_agent is not None}")
    print(f"  Logger attached: {search_agent.logger is not None}")
    print(f"  Extract skill attached: {search_agent.extract_skill is not None}")
    print(f"  Max URLs per source: {search_agent.max_urls_per_source}")
    print(f"  Max total URLs: {search_agent.max_total_urls}")
    print()

    # Test 1: Entertainment query detection
    print("[2/6] Testing entertainment query detection...")
    test_queries = {
        "When does Wicked Part 2 come out?": True,
        "Who directed Dune?": True,
        "Marvel Avengers cast": True,
        "Python programming tutorial": False,
        "Star Wars release date": True,
        "Machine learning basics": False,
    }

    detection_passed = True
    for query, expected in test_queries.items():
        result = search_agent._is_entertainment_query(query)
        status = "[PASS]" if result == expected else "[FAIL]"
        if result != expected:
            detection_passed = False
        print(f"  {status} '{query}' -> {result} (expected {expected})")

    print()

    # Test 2: DuckDuckGo search
    print("[3/6] Testing DuckDuckGo HTML search...")
    try:
        ddg_urls = await search_agent._search_duckduckgo("Python asyncio tutorial")
        print(f"  [PASS] DuckDuckGo search completed")
        print(f"  URLs found: {len(ddg_urls)}")
        if ddg_urls:
            print(f"  Sample URL: {ddg_urls[0]['url'][:60]}...")
            print(f"  Source: {ddg_urls[0]['source']}")
    except Exception as e:
        print(f"  [FAIL] DuckDuckGo search failed: {str(e)[:60]}")

    print()

    # Test 3: Wikipedia search
    print("[4/6] Testing Wikipedia search...")
    try:
        wiki_urls = await search_agent._search_wikipedia("Python programming language")
        print(f"  [PASS] Wikipedia search completed")
        print(f"  URLs found: {len(wiki_urls)}")
        if wiki_urls:
            print(f"  Sample URL: {wiki_urls[0]['url'][:60]}...")
            print(f"  Source: {wiki_urls[0]['source']}")
            print(f"  Title: {wiki_urls[0]['title'][:40]}...")
    except Exception as e:
        print(f"  [FAIL] Wikipedia search failed: {str(e)[:60]}")

    print()

    # Test 4: IMDb search (entertainment query)
    print("[5/6] Testing IMDb search...")
    try:
        imdb_urls = await search_agent._search_imdb("Wicked movie")
        print(f"  [PASS] IMDb search completed")
        print(f"  URLs found: {len(imdb_urls)}")
        if imdb_urls:
            print(f"  Sample URL: {imdb_urls[0]['url'][:60]}...")
            print(f"  Source: {imdb_urls[0]['source']}")
    except Exception as e:
        print(f"  [INFO] IMDb search issue (may be normal): {str(e)[:60]}")

    print()

    # Test 5: Full investigation
    print("[6/6] Testing full autonomous investigation...")
    test_investigations = [
        "Python asyncio tutorial",
        "When does Wicked Part 2 come out?",
    ]

    investigation_results = []
    for query in test_investigations:
        print(f"\nQuery: {query}")
        try:
            report = await search_agent.investigate(query)

            investigation_results.append({
                "query": query,
                "success": report["success"],
                "urls_count": len(report["urls_checked"]),
                "passages_count": len(report["extracted_passages"]),
                "sources_used": report["sources_used"],
                "has_evidence": len(report["combined_evidence"]) > 0
            })

            print(f"  Success: {report['success']}")
            print(f"  URLs checked: {len(report['urls_checked'])}")
            print(f"  Passages extracted: {len(report['extracted_passages'])}")
            print(f"  Sources used: {', '.join(report['sources_used'])}")
            print(f"  Evidence length: {len(report['combined_evidence'])} chars")

            if report["extracted_passages"]:
                print(f"  Sample passage source: {report['extracted_passages'][0]['source']}")

        except Exception as e:
            print(f"  [FAIL] Investigation failed: {str(e)[:100]}")
            investigation_results.append({
                "query": query,
                "success": False,
                "error": str(e)
            })

    print()
    print("=" * 70)
    print("Final Validation")
    print("=" * 70)
    print()

    # Validation checks
    checks = {
        "SearchAgent initialized successfully": search_agent is not None,
        "Entertainment query detection working": detection_passed,
        "DuckDuckGo search functional": len(ddg_urls) > 0 if 'ddg_urls' in locals() else False,
        "Wikipedia search functional": len(wiki_urls) > 0 if 'wiki_urls' in locals() else False,
        "At least one investigation successful": any(r.get("success", False) for r in investigation_results),
        "Multi-source investigation working": any(len(r.get("sources_used", [])) > 1 for r in investigation_results),
        "Content extraction functional": any(r.get("passages_count", 0) > 0 for r in investigation_results),
        "Evidence synthesis functional": any(r.get("has_evidence", False) for r in investigation_results),
    }

    for check, passed in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {check}")

    print()
    print("=" * 70)

    if all(checks.values()):
        print("[SUCCESS] All SearchAgent tests passed!")
    else:
        print("[INFO] Some tests failed - this may be due to network issues")
        print("        Review the output above for details")

    print("=" * 70)


async def test_web_search_skill_integration():
    """Test web_search skill integration with SearchAgent"""

    print()
    print("=" * 70)
    print("Testing web_search Skill Integration with SearchAgent")
    print("=" * 70)
    print()

    from core.logger import HugoLogger
    from skills.builtin.web_search import WebSearchSkill

    logger = HugoLogger()

    # Create mock memory
    memory = MagicMock()
    memory.store = AsyncMock()

    # Initialize web search skill
    web_search = WebSearchSkill(
        logger=logger,
        sqlite_manager=None,
        memory_manager=memory
    )

    print("[1/3] Testing web_search skill initialization...")
    print(f"  Skill name: {web_search.name}")
    print(f"  Skill version: {web_search.version}")
    print(f"  Skill description: {web_search.description}")
    print()

    # Test search action
    print("[2/3] Testing search action with agent deployment...")
    test_query = "Python asyncio tutorial"
    print(f"Query: {test_query}")

    result = await web_search.run(action="search", query=test_query)

    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")

    if result.output:
        print(f"  URLs found: {len(result.output.get('urls', []))}")
        print(f"  Sources used: {', '.join(result.output.get('sources_used', []))}")
        print(f"  Passages count: {result.output.get('passages_count', 0)}")
        print(f"  Evidence length: {len(result.output.get('combined_evidence', ''))} chars")

    print()

    # Test help action
    print("[3/3] Testing help action...")
    help_result = await web_search.run(action="help")
    print(f"  Success: {help_result.success}")
    print(f"  Help text length: {len(help_result.output) if help_result.output else 0} chars")

    print()
    print("=" * 70)

    checks = {
        "web_search skill version 2.0.0": web_search.version == "2.0.0",
        "web_search delegates to SearchAgent": web_search.search_agent is not None,
        "Search action returns results": result.success and result.output is not None,
        "Agent deployment permission required": "agent_deployment" in web_search.requires_permissions(),
        "Help action functional": help_result.success,
    }

    for check, passed in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {check}")

    print()
    print("=" * 70)

    if all(checks.values()):
        print("[SUCCESS] web_search skill integration tests passed!")
    else:
        print("[INFO] Some integration tests failed")

    print("=" * 70)


async def main():
    """Run all tests"""
    await test_search_agent()
    await test_web_search_skill_integration()


if __name__ == "__main__":
    asyncio.run(main())
