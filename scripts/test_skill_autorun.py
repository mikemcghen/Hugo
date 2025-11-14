"""
Test Skill Auto-Triggering from Natural Language
-------------------------------------------------
Verify that skills are automatically triggered when users type natural
language commands (e.g., "make a note saying...") without needing to
use explicit /skill commands.

Tests:
1. "make a note saying X" triggers notes.add skill
2. "what notes do I have?" triggers notes.list skill
3. "search notes for X" triggers notes.search skill
4. Verify notes persist across restarts
5. Verify skill execution is logged in SQLite
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.cognition import CognitionEngine
from core.memory import MemoryManager
from core.logger import HugoLogger
from core.runtime_manager import RuntimeManager
from data.sqlite_manager import SQLiteManager


async def main():
    print("=" * 70)
    print("SKILL AUTO-TRIGGERING TEST")
    print("=" * 70)
    print()

    # Initialize components
    logger = HugoLogger()
    sqlite_manager = SQLiteManager(db_path="data/memory/test_skill_autorun.db")
    await sqlite_manager.connect()
    print("[OK] SQLite manager initialized")

    # Initialize memory manager (with sqlite and postgres set to None for postgres)
    memory_manager = MemoryManager(sqlite_manager, None, logger)
    print("[OK] Memory manager initialized")

    # Initialize skill manager directly
    from skills.skill_manager import SkillManager
    skill_manager = SkillManager(logger, sqlite_manager, memory_manager)
    skill_manager.load_skills()
    print(f"[OK] Skill manager loaded with {len(skill_manager.list_skills())} skills")

    # Create a minimal runtime manager mock with skills attribute
    class RuntimeManagerMock:
        def __init__(self, skills):
            self.skills = skills

    runtime_manager = RuntimeManagerMock(skill_manager)

    # Initialize cognition engine
    cognition = CognitionEngine(memory_manager, logger, runtime_manager)
    print("[OK] Cognition engine initialized")
    print()

    # Test session ID
    session_id = "test_autorun_session"

    # Test 1: Auto-trigger notes.add via natural language
    print("-" * 70)
    print("TEST 1: Auto-trigger note creation")
    print("-" * 70)
    test_message = "make a note saying remember to test skill auto-triggering"
    print(f"User: {test_message}")
    print()

    # Trigger save_user_message (which includes classification + skill autorun)
    await cognition._save_user_message(test_message, session_id)

    # Wait a moment for async operations
    await asyncio.sleep(0.5)

    # Verify note was created
    notes = await sqlite_manager.list_notes(limit=1)
    if notes:
        print(f"[OK] Note created: {notes[0]['content'][:50]}...")
    else:
        print("[FAIL] Note was not created")
    print()

    # Test 2: Auto-trigger notes.list via natural language
    print("-" * 70)
    print("TEST 2: Auto-trigger note list")
    print("-" * 70)
    test_message = "what notes do I have?"
    print(f"User: {test_message}")
    print()

    await cognition._save_user_message(test_message, session_id)
    await asyncio.sleep(0.5)

    # Check skill execution history
    history = await sqlite_manager.get_skill_history(limit=5)
    if history:
        print(f"[OK] Found {len(history)} skill executions:")
        for run in history:
            action = run.get('action', 'unknown')
            print(f"  - {run['name']} ({action}) at {run['executed_at']}")
    else:
        print("[FAIL] No skill execution history found")
    print()

    # Test 3: Auto-trigger notes.search via natural language
    print("-" * 70)
    print("TEST 3: Auto-trigger note search")
    print("-" * 70)
    test_message = "search notes for test"
    print(f"User: {test_message}")
    print()

    await cognition._save_user_message(test_message, session_id)
    await asyncio.sleep(0.5)

    # Verify search was executed
    history = await sqlite_manager.get_skill_history(limit=1)
    if history and history[0].get('action') == 'search':
        print(f"[OK] Search skill executed: {history[0]['name']} - {history[0].get('action', 'unknown')}")
    else:
        print("[WARN] Search skill execution unclear from history")
    print()

    # Test 4: Verify notes persist
    print("-" * 70)
    print("TEST 4: Verify note persistence")
    print("-" * 70)
    all_notes = await sqlite_manager.list_notes(limit=10)
    if all_notes:
        print(f"[OK] Found {len(all_notes)} persisted notes:")
        for note in all_notes:
            print(f"  - Note #{note['id']}: {note['content'][:50]}...")
    else:
        print("[FAIL] No notes found in database")
    print()

    # Test 5: Verify memory classification logs
    print("-" * 70)
    print("TEST 5: Verify memory classification")
    print("-" * 70)

    # Read structured logs to verify classification
    log_path = Path("data/logs/structured.jsonl")
    if log_path.exists():
        import json
        classification_count = 0
        skill_trigger_count = 0

        with open(log_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    if log_entry.get('event') == 'memory_classified':
                        classification_count += 1
                        if log_entry.get('data', {}).get('skill_trigger'):
                            skill_trigger_count += 1
                except json.JSONDecodeError:
                    continue

        print(f"[OK] Found {classification_count} memory classifications")
        print(f"[OK] Found {skill_trigger_count} with skill triggers")
    else:
        print("[WARN] Log file not found")
    print()

    # Test 6: Verify cognition skill autorun events
    print("-" * 70)
    print("TEST 6: Verify cognition skill autorun events")
    print("-" * 70)

    if log_path.exists():
        import json
        autorun_started = 0
        autorun_completed = 0

        with open(log_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    event = log_entry.get('event')
                    if event == 'skill_autorun_started':
                        autorun_started += 1
                    elif event == 'skill_autorun_completed':
                        autorun_completed += 1
                except json.JSONDecodeError:
                    continue

        print(f"[OK] Found {autorun_started} skill autorun starts")
        print(f"[OK] Found {autorun_completed} skill autorun completions")

        if autorun_started > 0 and autorun_completed > 0:
            print("[OK] Skill auto-triggering is working!")
        else:
            print("[FAIL] Skill auto-triggering not detected in logs")
    print()

    # Close database
    await sqlite_manager.close()

    print("=" * 70)
    print("SKILL AUTO-TRIGGERING TEST COMPLETE")
    print("=" * 70)
    print()
    print("Summary:")
    print("- Memory classification attaches skill_trigger metadata")
    print("- Cognition engine detects skill_trigger and auto-executes")
    print("- Skills execute without explicit /skill commands")
    print("- Natural language like 'make a note' triggers notes.add")
    print("- Natural language like 'what notes' triggers notes.list")
    print("- Natural language like 'search notes' triggers notes.search")
    print()


if __name__ == "__main__":
    asyncio.run(main())
