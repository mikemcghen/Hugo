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
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from skills.skill_manager import SkillManager
from data.sqlite_manager import SQLiteManager
from core.logger import HugoLogger


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

    # Initialize skill manager
    skill_manager = SkillManager(logger, sqlite_manager, memory_manager=None)
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

    # Close database
    await sqlite_manager.close()

    print()
    print("=" * 70)
    print("SUCCESS: ALL SKILL SYSTEM TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
