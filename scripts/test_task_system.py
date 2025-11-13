"""
Test Task System
================

Tests the task management system:
- Task creation
- Task listing
- Task retrieval
- Status updates
- Task assignment
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.tasks import TaskManager
from core.logger import HugoLogger
from data.sqlite_manager import SQLiteManager

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


async def test_task_system():
    """Test task management functionality"""
    print("=" * 70)
    print("TASK SYSTEM TEST")
    print("=" * 70)

    # Use test database
    test_db = "data/memory/test_task_system.db"
    Path(test_db).unlink(missing_ok=True)  # Clean slate

    logger = HugoLogger(log_dir="logs/tests")
    sqlite_manager = SQLiteManager(db_path=test_db)
    await sqlite_manager.connect()

    task_manager = TaskManager(sqlite_manager, logger)

    # Test 1: Create tasks
    print("\n📝 Test 1: Creating tasks...\n")

    task1 = await task_manager.create_task(
        title="Wire up Hugo TaskManager",
        description="Add tasks table, TaskManager, and REPL commands.",
        owner="michael",
        priority="high",
        tags=["hugo", "core", "tasks"]
    )
    print(f"✓ Created task #{task1.id}: {task1.title}")

    task2 = await task_manager.create_task(
        title="Investigate Ollama timeout handling",
        description="Look into why Ollama sometimes times out.",
        owner="hugo",
        priority="medium"
    )
    print(f"✓ Created task #{task2.id}: {task2.title}")

    task3 = await task_manager.create_task(
        title="Add reflection statistics dashboard",
        description="Create a view to see reflection trends over time.",
        owner="michael",
        priority="low"
    )
    print(f"✓ Created task #{task3.id}: {task3.title}")

    # Test 2: List all tasks
    print("\n📋 Test 2: Listing all tasks...\n")

    all_tasks = await task_manager.list_tasks()
    print(f"Found {len(all_tasks)} tasks:")
    for task in all_tasks:
        print(f"  {task.id}. [{task.status}] {task.title} (owner: {task.owner})")

    if len(all_tasks) != 3:
        print(f"\n✗ FAIL: Expected 3 tasks, got {len(all_tasks)}")
        return False
    else:
        print("\n✓ PASS: Correct number of tasks")

    # Test 3: List tasks by owner
    print("\n📋 Test 3: Listing michael's tasks...\n")

    michael_tasks = await task_manager.list_tasks(owner="michael")
    print(f"Found {len(michael_tasks)} tasks for michael:")
    for task in michael_tasks:
        print(f"  {task.id}. [{task.status}] {task.title}")

    if len(michael_tasks) != 2:
        print(f"\n✗ FAIL: Expected 2 tasks for michael, got {len(michael_tasks)}")
        return False
    else:
        print("\n✓ PASS: Correct number of tasks for michael")

    # Test 4: Get specific task
    print("\n🔍 Test 4: Getting task details...\n")

    task = await task_manager.get_task(task1.id)
    if task:
        print(f"✓ Found task {task.id}:")
        print(f"  Title: {task.title}")
        print(f"  Description: {task.description}")
        print(f"  Status: {task.status}")
        print(f"  Owner: {task.owner}")
        print(f"  Priority: {task.priority}")
        print(f"  Tags: {', '.join(task.tags)}")
    else:
        print(f"✗ FAIL: Could not find task {task1.id}")
        return False

    # Test 5: Update task status
    print("\n✏️ Test 5: Updating task status...\n")

    success = await task_manager.update_task_status(task1.id, "in_progress")
    if success:
        print(f"✓ Task {task1.id} status updated to in_progress")

        # Verify update
        updated_task = await task_manager.get_task(task1.id)
        if updated_task.status == "in_progress":
            print("✓ Status change verified")
        else:
            print(f"✗ FAIL: Status is {updated_task.status}, expected in_progress")
            return False
    else:
        print(f"✗ FAIL: Failed to update task status")
        return False

    # Test 6: Assign task
    print("\n👤 Test 6: Assigning task...\n")

    success = await task_manager.assign_task(task2.id, "claude-agent")
    if success:
        print(f"✓ Task {task2.id} assigned to claude-agent")

        # Verify assignment
        assigned_task = await task_manager.get_task(task2.id)
        if assigned_task.owner == "claude-agent":
            print("✓ Assignment verified")
        else:
            print(f"✗ FAIL: Owner is {assigned_task.owner}, expected claude-agent")
            return False
    else:
        print(f"✗ FAIL: Failed to assign task")
        return False

    # Test 7: Mark task as done
    print("\n✅ Test 7: Marking task as done...\n")

    success = await task_manager.update_task_status(task3.id, "done")
    if success:
        print(f"✓ Task {task3.id} marked as done")

        # Verify status
        done_task = await task_manager.get_task(task3.id)
        if done_task.status == "done":
            print("✓ Done status verified")
        else:
            print(f"✗ FAIL: Status is {done_task.status}, expected done")
            return False
    else:
        print(f"✗ FAIL: Failed to mark task as done")
        return False

    # Test 8: List tasks by status
    print("\n📊 Test 8: Listing done tasks...\n")

    done_tasks = await task_manager.list_tasks(status="done")
    print(f"Found {len(done_tasks)} done tasks:")
    for task in done_tasks:
        print(f"  {task.id}. {task.title}")

    if len(done_tasks) != 1:
        print(f"\n✗ FAIL: Expected 1 done task, got {len(done_tasks)}")
        return False
    else:
        print("\n✓ PASS: Correct number of done tasks")

    print("\n" + "=" * 70)
    print("✨ ALL TASK SYSTEM TESTS PASSED")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = asyncio.run(test_task_system())
    sys.exit(0 if success else 1)
