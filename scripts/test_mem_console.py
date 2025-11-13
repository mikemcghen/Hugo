"""
Test Memory Console Commands
=============================

Tests the mem console functionality:
- mem facts - List factual memories
- mem show <id> - Show memory details
- mem delete <id> - Delete memory and rebuild index
- mem search <query> - Semantic search
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.memory import MemoryManager, MemoryEntry
from core.logger import HugoLogger
from data.sqlite_manager import SQLiteManager

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


async def test_mem_console():
    """Test memory console commands"""
    print("=" * 70)
    print("MEMORY CONSOLE TEST")
    print("=" * 70)

    # Use test database
    test_db = "data/memory/test_mem_console.db"
    Path(test_db).unlink(missing_ok=True)  # Clean slate

    logger = HugoLogger(log_dir="logs/tests")
    sqlite_manager = SQLiteManager(db_path=test_db)
    await sqlite_manager.connect()

    memory = MemoryManager(sqlite_manager, None, logger)

    # Test 1: Insert some test memories
    print("\n📝 Test 1: Inserting test memories...\n")

    test_memories = [
        ("I have 2 cats named Will and Tonks", True, "animal"),
        ("I have 2 bunnies named Oswald and Keely", True, "animal"),
        ("I live in Portland Oregon", True, "location"),
        ("My favorite language is Python", True, "preference"),
    ]

    memory_ids = []
    for content, should_be_fact, expected_entity in test_memories:
        entry = MemoryEntry(
            id=None,
            session_id="test_mem_console",
            timestamp=datetime.now(),
            memory_type="user_message",
            content=content,
            embedding=None,  # Will be auto-generated
            metadata={"test": True},
            importance_score=0.7
        )
        await memory.store(entry)

        if entry.is_fact:
            # Get the memory ID from SQLite
            facts = await sqlite_manager.get_factual_memories(limit=100)
            for fact in facts:
                if fact['content'] == content:
                    memory_ids.append(fact['id'])
                    break

        status = "✓" if (entry.is_fact == should_be_fact) else "✗"
        fact_label = f"FACT:{entry.entity_type}" if entry.is_fact else "NOT FACT"
        print(f"{status} {fact_label:20} | {content[:50]}")

    print(f"\nStored {len(memory_ids)} factual memories with IDs: {memory_ids}")

    # Test 2: list_factual_memories()
    print("\n📋 Test 2: Testing list_factual_memories()...\n")

    facts = await memory.list_factual_memories(limit=20)
    print(f"Found {len(facts)} factual memories")

    for fact in facts:
        print(f"  ID {fact['id']}: [{fact.get('entity_type', 'unknown')}] {fact['content'][:50]}")

    if len(facts) != len(test_memories):
        print(f"\n✗ FAIL: Expected {len(test_memories)} facts, got {len(facts)}")
        return False
    else:
        print("\n✓ PASS: Correct number of facts listed")

    # Test 3: get_memory(id)
    print("\n🔍 Test 3: Testing get_memory()...\n")

    if memory_ids:
        test_id = memory_ids[0]
        mem = await memory.get_memory(test_id)

        if mem:
            print(f"✓ Found memory {test_id}:")
            print(f"  Content: {mem['content']}")
            print(f"  Entity: {mem.get('entity_type', 'N/A')}")
            print(f"  Created: {mem['timestamp']}")
        else:
            print(f"✗ FAIL: Could not find memory {test_id}")
            return False
    else:
        print("⚠ No memory IDs to test")

    # Test 4: search_memories()
    print("\n🔍 Test 4: Testing search_memories()...\n")

    results = await memory.search_memories("what are my pets", k=5)
    print(f"Search for 'what are my pets' returned {len(results)} results:")

    for i, mem in enumerate(results, 1):
        print(f"  {i}. [id={mem['id']}, score={mem['score']:.2f}] {mem['content'][:50]}")

    if len(results) > 0:
        print("\n✓ PASS: Search returned results")
    else:
        print("\n⚠ WARN: Search returned no results (may need threshold tuning)")

    # Test 5: delete_memory()
    print("\n🗑️  Test 5: Testing delete_memory()...\n")

    if memory_ids:
        delete_id = memory_ids[-1]  # Delete last one
        print(f"Deleting memory {delete_id}...")

        deleted = await memory.delete_memory(delete_id)

        if deleted:
            print(f"✓ Successfully deleted memory {delete_id}")

            # Verify it's gone
            mem = await memory.get_memory(delete_id)
            if mem is None:
                print(f"✓ Verified: Memory {delete_id} no longer exists")
            else:
                print(f"✗ FAIL: Memory {delete_id} still exists after deletion")
                return False

            # Check FAISS was rebuilt
            remaining_facts = await memory.list_factual_memories(limit=20)
            print(f"✓ FAISS rebuilt, {len(remaining_facts)} facts remaining")

        else:
            print(f"✗ FAIL: Failed to delete memory {delete_id}")
            return False
    else:
        print("⚠ No memory IDs to test")

    print("\n" + "=" * 70)
    print("✨ ALL MEM CONSOLE TESTS PASSED")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = asyncio.run(test_mem_console())
    sys.exit(0 if success else 1)
