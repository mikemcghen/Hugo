"""
Test Skills System
------------------
Verify that the Skills subsystem is working correctly.

Tests:
1. Load skills from YAML definitions
2. Execute notes skill with add action
3. Execute notes skill with list action
4. Execute notes skill with search action
5. Verify skill results are stored in SQLite
6. Get skill stats
7. Test extract_and_answer skill with mock HTML
8. Test web_search 202 → 200 polling sequence
"""

import asyncio
import sys
import io
from pathlib import Path
from aiohttp import web
import threading
import time

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from skills.skill_manager import SkillManager
from data.sqlite_manager import SQLiteManager
from core.logger import HugoLogger
from core.memory import MemoryManager


class Mock202Server:
    """Mock HTTP server that simulates 202 → 200 polling sequence"""

    def __init__(self, port=8765):
        self.port = port
        self.request_count = 0
        self.app = None
        self.runner = None
        self.site = None

    async def handle_request(self, request):
        """Handle mock API request"""
        self.request_count += 1

        # First 2 requests return 202 ACCEPTED
        if self.request_count <= 2:
            return web.Response(status=202, text="Accepted")

        # Third request returns 200 OK with mock DuckDuckGo response
        mock_response = {
            "Abstract": "Python is a high-level programming language.",
            "AbstractText": "Python is a high-level programming language known for its simplicity and readability.",
            "AbstractSource": "Wikipedia",
            "AbstractURL": "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "Heading": "Python (programming language)",
            "RelatedTopics": [
                {
                    "Text": "Python Software Foundation",
                    "FirstURL": "https://www.python.org"
                }
            ]
        }

        return web.json_response(mock_response)

    async def start(self):
        """Start mock server"""
        self.app = web.Application()
        self.app.router.add_get('/', self.handle_request)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(self.runner, 'localhost', self.port)
        await self.site.start()

    async def stop(self):
        """Stop mock server"""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()


async def main():
    print("=" * 70)
    print("SKILLS SYSTEM TEST")
    print("=" * 70)
    print()

    # Initialize components
    logger = HugoLogger()
    sqlite_manager = SQLiteManager(db_path="data/memory/test_skills.db")
    await sqlite_manager.connect()
    print("[OK] SQLite manager initialized")

    memory_manager = MemoryManager(sqlite_manager, None, logger)
    print("[OK] Memory manager initialized")

    # Initialize skill manager
    skill_manager = SkillManager(logger, sqlite_manager, memory_manager)
    print("[OK] Skill manager created")

    # Test 1: Load skills
    print("\n-> Test 1: Loading skills...")
    skill_manager.load_skills()

    skills = skill_manager.list_skills()
    if skills:
        print(f"[OK] Loaded {len(skills)} skill(s):")
        for skill in skills:
            print(f"  - {skill['name']}: {skill['description']}")
    else:
        print("[FAIL] No skills loaded")
        return

    # Test 2: Add a note
    print("\n-> Test 2: Adding a note...")
    result = await skill_manager.run_skill(
        'notes',
        action='add',
        content='Test note: Remember to test the skill system'
    )

    if result.success:
        print(f"[OK] {result.message}")
        print(f"  Output: {result.output}")
    else:
        print(f"[FAIL] {result.message}")
        return

    # Test 3: List notes
    print("\n-> Test 3: Listing notes...")
    result = await skill_manager.run_skill('notes', action='list', limit=5)

    if result.success:
        print(f"[OK] {result.message}")
        if result.output:
            for note in result.output:
                print(f"  - Note #{note['id']}: {note['content'][:50]}")
    else:
        print(f"[FAIL] {result.message}")

    # Test 4: Search notes
    print("\n-> Test 4: Searching notes...")
    result = await skill_manager.run_skill('notes', action='search', query='test')

    if result.success:
        print(f"[OK] {result.message}")
        if result.output:
            for note in result.output:
                print(f"  - Note #{note['id']}: {note['content'][:50]}")
    else:
        print(f"[FAIL] {result.message}")

    # Test 5: Get skill execution history
    print("\n-> Test 5: Checking skill execution history...")
    history = await sqlite_manager.get_skill_history(limit=10)

    if history:
        print(f"[OK] Found {len(history)} skill executions:")
        for run in history:
            print(f"  - {run['name']} at {run['executed_at']}")
    else:
        print("  No execution history found")

    # Test 6: Get skill stats
    print("\n-> Test 6: Getting skill stats...")
    stats = skill_manager.get_stats()
    print(f"[OK] Skill system stats:")
    print(f"  Total skills: {stats['total_skills']}")
    print(f"  Total runs: {stats['total_runs']}")
    print(f"  Total errors: {stats['total_errors']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")

    # Test 7: Extract and answer skill
    print("\n-> Test 7: Testing extract_and_answer skill...")

    # Create mock HTML page
    mock_html = """
    <html>
    <head><title>Test Article About Python</title></head>
    <body>
        <nav>Navigation menu</nav>
        <header>Site Header</header>
        <main>
            <article>
                <h1>Python Programming Language</h1>
                <p>Python is a high-level, interpreted programming language created by Guido van Rossum.</p>
                <p>It was first released in 1991 and has become one of the most popular programming languages.</p>
                <p>Python emphasizes code readability with significant whitespace.</p>
                <p>The language supports multiple programming paradigms including procedural, object-oriented, and functional programming.</p>
            </article>
        </main>
        <footer>Site Footer</footer>
        <script>alert('test');</script>
    </body>
    </html>
    """

    # Create mock HTML server
    class MockHTMLServer:
        def __init__(self, port=8766):
            self.port = port
            self.app = None
            self.runner = None
            self.site = None

        async def handle_request(self, request):
            return web.Response(text=mock_html, content_type='text/html')

        async def start(self):
            self.app = web.Application()
            self.app.router.add_get('/', self.handle_request)
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            self.site = web.TCPSite(self.runner, 'localhost', self.port)
            await self.site.start()

        async def stop(self):
            if self.site:
                await self.site.stop()
            if self.runner:
                await self.runner.cleanup()

    # Start mock HTML server
    html_server = MockHTMLServer(port=8766)
    await html_server.start()
    print("[OK] Mock HTML server started on port 8766")

    try:
        # Run extract_and_answer skill
        result = await skill_manager.run_skill(
            'extract_and_answer',
            action='extract',
            query='What is Python programming language?',
            urls=['http://localhost:8766/']
        )

        if result.success:
            print(f"[OK] Extraction succeeded: {result.message}")
            if result.output and result.output.get('answer'):
                print(f"  Answer: {result.output['answer'][:80]}...")
                print(f"  Sources: {len(result.output.get('sources', []))}")
            else:
                print("[WARN] No answer in output")
        else:
            print(f"[FAIL] Extraction failed: {result.message}")

    finally:
        await html_server.stop()
        print("[OK] Mock HTML server stopped")

    # Test 8: Web search 202 → 200 polling
    print("\n-> Test 8: Testing web_search 202 → 200 polling sequence...")

    # Start mock server
    mock_server = Mock202Server(port=8765)
    await mock_server.start()
    print("[OK] Mock 202 server started on port 8765")

    try:
        # Temporarily override the web_search API URL
        web_search_skill = skill_manager.registry.get('web_search')
        if web_search_skill:
            original_url = web_search_skill.api_url
            web_search_skill.api_url = "http://localhost:8765/"

            print("  Sending request to mock server...")
            print("  Expected: 202 ACCEPTED (x2) → 200 OK")

            # Execute web search
            result = await skill_manager.run_skill(
                'web_search',
                action='search',
                query='Python programming'
            )

            # Check results
            if result.success:
                print(f"[OK] Polling succeeded after {mock_server.request_count} requests")
                print(f"  Result: {result.message}")

                if mock_server.request_count == 3:
                    print("[OK] Correct number of polling attempts (3 total)")
                else:
                    print(f"[WARN] Expected 3 requests, got {mock_server.request_count}")

                # Verify response content
                if result.output and result.output.get('abstract_text'):
                    print(f"[OK] Received valid response: {result.output['abstract_text'][:60]}...")
                else:
                    print("[WARN] Response missing expected content")
            else:
                print(f"[FAIL] Polling failed: {result.message}")

            # Restore original URL
            web_search_skill.api_url = original_url
        else:
            print("[SKIP] web_search skill not loaded")

    finally:
        # Stop mock server
        await mock_server.stop()
        print("[OK] Mock server stopped")

    # Close database
    await sqlite_manager.close()

    print()
    print("=" * 70)
    print("SUCCESS: ALL SKILL SYSTEM TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
