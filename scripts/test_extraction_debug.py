"""
Debug script to trace extraction failure
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_direct_extraction():
    """Test direct URL extraction to isolate failure point"""

    from core.logger import HugoLogger
    from skills.builtin.extract_and_answer import ExtractAndAnswerSkill

    logger = HugoLogger()
    extract_skill = ExtractAndAnswerSkill(
        logger=logger,
        sqlite_manager=None,
        memory_manager=None,
        cognition=None
    )

    # Test URL (simple, should work)
    test_url = "https://realpython.com/async-io-python/"

    print("=" * 70)
    print("EXTRACTION DEBUG TEST")
    print("=" * 70)
    print(f"\nTest URL: {test_url}")
    print()

    print("[1/3] Testing _fetch_and_extract_direct()...")
    try:
        result = await extract_skill._fetch_and_extract_direct(test_url)

        if result is None:
            print("  [FAIL] Result is None")
        else:
            print(f"  [PASS] Result received")
            print(f"  Title: {result.get('title', 'N/A')[:60]}")
            print(f"  Content length: {len(result.get('content', ''))}")
            print(f"  Content preview: {result.get('content', '')[:100]}")

            if not result.get('content'):
                print("  [WARN] Content is empty!")
            elif len(result.get('content', '')) < 100:
                print("  [WARN] Content is very short (<100 chars)")
            else:
                print("  [PASS] Content extracted successfully")

    except Exception as e:
        print(f"  [ERROR] Exception occurred: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("[2/3] Testing with Wikipedia (should be simpler)...")
    wiki_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

    try:
        result = await extract_skill._fetch_and_extract_direct(wiki_url)

        if result is None:
            print("  [FAIL] Result is None")
        else:
            print(f"  [PASS] Result received")
            print(f"  Title: {result.get('title', 'N/A')[:60]}")
            print(f"  Content length: {len(result.get('content', ''))}")

            if len(result.get('content', '')) > 100:
                print("  [PASS] Content extracted successfully")
            else:
                print("  [FAIL] Content too short or empty")

    except Exception as e:
        print(f"  [ERROR] Exception occurred: {e}")

    print()
    print("[3/3] Testing with DuckDuckGo redirect URL...")
    ddg_redirect_url = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Frealpython.com%2Fasync-io-python%2F"

    try:
        result = await extract_skill._fetch_and_extract_direct(ddg_redirect_url)

        if result is None:
            print("  [FAIL] Result is None - This is likely the root cause!")
            print("  [INFO] DuckDuckGo redirect URLs may not resolve correctly")
        else:
            print(f"  [PASS] Result received")
            print(f"  Title: {result.get('title', 'N/A')[:60]}")
            print(f"  Content length: {len(result.get('content', ''))}")

            if len(result.get('content', '')) > 100:
                print("  [PASS] Content extracted from redirect URL")
            else:
                print("  [WARN] Content too short after redirect")

    except Exception as e:
        print(f"  [ERROR] Exception occurred: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_direct_extraction())
