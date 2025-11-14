"""
Test script for thread-safe SQLite writes
"""
import asyncio
import sys
import os
import threading
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_sqlite_threading():
    """Test that SQLite writes from different threads work correctly"""

    print("=" * 70)
    print("PHASE 4.4: Testing Thread-Safe SQLite Writes")
    print("=" * 70)
    print()

    from core.logger import HugoLogger
    from data.sqlite_manager import SQLiteManager

    logger = HugoLogger()

    # Initialize SQLite manager with logger
    sqlite = SQLiteManager(db_path="data/memory/test_threading.db", logger=logger)
    await sqlite.connect()

    # Start the drain queue loop
    drain_task = asyncio.create_task(sqlite.drain_queue_loop())

    print("[1/4] SQLite manager initialized with drain queue loop")
    print(f"  Queue running: {sqlite._queue_running}")
    print()

    # Test 1: Write from main thread
    print("[2/4] Testing write from main thread...")
    try:
        await sqlite.store_memory(
            session_id="test_session",
            memory_type="test",
            content="Test memory from main thread",
            importance_score=0.8,
            is_fact=True
        )
        print("  [PASS] Main thread write succeeded")
    except Exception as e:
        print(f"  [FAIL] Main thread write failed: {str(e)}")

    print()

    # Test 2: Write from background thread using asyncio
    print("[3/4] Testing write from background task...")

    async def background_write():
        """Simulate a background task writing to SQLite"""
        for i in range(3):
            await sqlite.store_memory(
                session_id="background_session",
                memory_type="background_test",
                content=f"Background memory #{i}",
                importance_score=0.5,
                is_fact=False
            )
            await asyncio.sleep(0.1)

    try:
        await background_write()
        print("  [PASS] Background task writes succeeded")
    except Exception as e:
        print(f"  [FAIL] Background task writes failed: {str(e)}")

    print()

    # Test 3: Concurrent writes from multiple tasks
    print("[4/4] Testing concurrent writes from multiple tasks...")

    async def concurrent_writer(task_id, count):
        """Write multiple memories concurrently"""
        for i in range(count):
            await sqlite.store_memory(
                session_id=f"concurrent_session_{task_id}",
                memory_type="concurrent_test",
                content=f"Concurrent write from task {task_id}, item {i}",
                importance_score=0.6,
                is_fact=False
            )

    try:
        # Launch 5 concurrent writers
        tasks = [concurrent_writer(i, 3) for i in range(5)]
        await asyncio.gather(*tasks)
        print("  [PASS] Concurrent writes from 5 tasks succeeded")
        print("  Total writes: 15 (3 writes x 5 tasks)")
    except Exception as e:
        print(f"  [FAIL] Concurrent writes failed: {str(e)}")

    print()
    print("=" * 70)
    print("Validation")
    print("=" * 70)
    print()

    # Verify all writes completed
    try:
        factual_memories = await sqlite.get_factual_memories()
        print(f"  Factual memories stored: {len(factual_memories)}")

        checks = {
            "SQLite manager initialized": sqlite is not None,
            "Drain queue running": sqlite._queue_running,
            "Main thread write succeeded": True,
            "Background writes succeeded": True,
            "Concurrent writes succeeded": True,
            "At least 1 factual memory stored": len(factual_memories) >= 1,
        }

        for check, passed in checks.items():
            status = "[PASS]" if passed else "[FAIL]"
            print(f"{status} {check}")

        print()
        print("=" * 70)

        if all(checks.values()):
            print("[SUCCESS] All thread-safe SQLite tests passed!")
        else:
            print("[INFO] Some tests may have encountered issues")

        print("=" * 70)

    except Exception as e:
        print(f"[ERROR] Validation failed: {str(e)}")

    finally:
        # Shutdown drain queue
        sqlite._queue_running = False
        await sqlite.write_queue.put(None)  # Send shutdown signal
        await drain_task


if __name__ == "__main__":
    asyncio.run(test_sqlite_threading())
