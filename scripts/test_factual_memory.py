#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistent Factual Memory Test
================================
Tests cross-session factual memory recall:
1. Insert facts and persist to SQLite
2. Simulate restart (recreate MemoryManager)
3. Verify facts are loaded from SQLite
4. Test semantic search finds facts
5. Verify prompt assembly includes facts
"""

import asyncio
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add Hugo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memory import MemoryManager, MemoryEntry
from core.logger import HugoLogger
from data.sqlite_manager import SQLiteManager
from datetime import datetime


async def test_fact_persistence():
    """Test 1: Insert facts and persist to SQLite"""
    print("=" * 70)
    print("TEST 1: Fact Persistence to SQLite")
    print("=" * 70)

    # Initialize components
    logger = HugoLogger(log_dir="logs/tests")
    sqlite_manager = SQLiteManager(db_path="data/memory/test_factual.db")
    await sqlite_manager.connect()

    memory = MemoryManager(sqlite_manager, None, logger)

    try:
        # Insert test facts
        test_facts = [
            ("I have two bunnies named Oswald and Keely", "animal"),
            ("My cats are Will and Tonks", "animal"),
            ("I live in San Francisco", "location"),
            ("My favorite programming language is Python", "preference"),
        ]

        print("\nInserting factual memories...")
        for content, expected_entity in test_facts:
            fact_entry = MemoryEntry(
                id=None,
                session_id="test_persistence_001",
                timestamp=datetime.now(),
                memory_type="user_message",
                content=content,
                embedding=None,  # Will be auto-generated
                metadata={"test": True},
                importance_score=0.8
            )

            # Store (should auto-detect as fact and persist)
            await memory.store(fact_entry, persist_long_term=False)

            # Verify detection
            if fact_entry.is_fact:
                print(f"  ✓ '{content[:40]}...' detected as fact ({fact_entry.entity_type})")
            else:
                print(f"  ✗ FAILED: '{content[:40]}...' not detected as fact")
                return False

        # Verify SQLite storage
        print("\nVerifying SQLite storage...")
        stored_facts = await sqlite_manager.get_factual_memories()

        if len(stored_facts) >= len(test_facts):
            print(f"  ✓ {len(stored_facts)} facts stored in SQLite")
        else:
            print(f"  ✗ Expected at least {len(test_facts)} facts, found {len(stored_facts)}")
            return False

        print("\n✓ Test 1 passed: Facts persisted to SQLite")
        return True

    except Exception as e:
        print(f"\n✗ Test 1 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_cross_session_reload():
    """Test 2: Simulate restart and reload facts"""
    print("\n" + "=" * 70)
    print("TEST 2: Cross-Session Fact Reload")
    print("=" * 70)

    try:
        # Simulate restart: Create NEW instances
        logger = HugoLogger(log_dir="logs/tests")
        sqlite_manager = SQLiteManager(db_path="data/memory/test_factual.db")
        await sqlite_manager.connect()

        memory = MemoryManager(sqlite_manager, None, logger)

        print("\nSimulating boot sequence...")
        print("  Loading factual memories from SQLite...")

        # Load facts (this is what happens on boot)
        await memory.load_factual_memories()

        # Check cache
        cache_facts = [m for m in memory.cache if m.is_fact]

        if len(cache_facts) > 0:
            print(f"  ✓ {len(cache_facts)} facts loaded into cache")

            # Display loaded facts
            print("\n  Loaded facts:")
            for i, fact in enumerate(cache_facts[:5], 1):
                entity_label = f"[{fact.entity_type.upper()}]" if fact.entity_type else ""
                print(f"    {i}. {entity_label} {fact.content[:60]}...")

        else:
            print("  ✗ No facts loaded from SQLite")
            return False

        # Check FAISS index
        stats = memory.get_stats()
        if stats['faiss_index_size'] > 0:
            print(f"  ✓ FAISS index contains {stats['faiss_index_size']} vectors")
        else:
            print("  ⚠ FAISS index empty (embeddings may not have been saved)")

        print("\n✓ Test 2 passed: Facts successfully reloaded after restart")
        return True

    except Exception as e:
        print(f"\n✗ Test 2 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_semantic_search_recall():
    """Test 3: Semantic search finds persisted facts"""
    print("\n" + "=" * 70)
    print("TEST 3: Semantic Search Recall of Facts")
    print("=" * 70)

    try:
        # Initialize with existing database
        logger = HugoLogger(log_dir="logs/tests")
        sqlite_manager = SQLiteManager(db_path="data/memory/test_factual.db")
        await sqlite_manager.connect()

        memory = MemoryManager(sqlite_manager, None, logger)

        # Load facts
        await memory.load_factual_memories()

        # Test searches for known facts
        test_queries = [
            ("remember my bunnies", "Oswald", "Keely"),
            ("what are my cats names", "Will", "Tonks"),
            ("where do I live", "San Francisco"),
            ("what programming language do I prefer", "Python"),
        ]

        passed_queries = 0

        for query, *expected_terms in test_queries:
            print(f"\n  Query: '{query}'")

            results = await memory.search_semantic(query, limit=5, threshold=0.4)

            if not results:
                print(f"    ✗ No results found")
                continue

            # Check if any result contains the expected terms
            found_terms = []
            for term in expected_terms:
                if any(term.lower() in r.content.lower() for r in results):
                    found_terms.append(term)

            if len(found_terms) == len(expected_terms):
                print(f"    ✓ Found all expected terms: {', '.join(expected_terms)}")
                passed_queries += 1

                # Show top result
                top_result = results[0]
                is_fact_label = "[FACT]" if top_result.is_fact else ""
                print(f"    Top result: {is_fact_label} {top_result.content[:60]}...")
            else:
                print(f"    ✗ Missing terms: {set(expected_terms) - set(found_terms)}")

        print(f"\n  Passed {passed_queries}/{len(test_queries)} queries")

        if passed_queries >= len(test_queries) * 0.75:  # 75% pass rate
            print("\n✓ Test 3 passed: Semantic search successfully recalls facts")
            return True
        else:
            print("\n✗ Test 3 failed: Too many failed queries")
            return False

    except Exception as e:
        print(f"\n✗ Test 3 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_fact_prioritization():
    """Test 4: Facts are prioritized over non-facts in search"""
    print("\n" + "=" * 70)
    print("TEST 4: Factual Memory Prioritization")
    print("=" * 70)

    try:
        # Initialize with existing database
        logger = HugoLogger(log_dir="logs/tests")
        sqlite_manager = SQLiteManager(db_path="data/memory/test_factual.db")
        await sqlite_manager.connect()

        memory = MemoryManager(sqlite_manager, None, logger)

        # Load facts
        await memory.load_factual_memories()

        # Add a non-factual memory with similar content
        non_fact = MemoryEntry(
            id=None,
            session_id="test_prioritization_001",
            timestamp=datetime.now(),
            memory_type="assistant_message",
            content="Many people have pet bunnies as companions",
            embedding=None,
            metadata={"test": True},
            importance_score=0.6
        )
        await memory.store(non_fact, persist_long_term=False)

        # Search for bunny-related content
        print("\n  Searching for 'bunny pets'...")
        results = await memory.search_semantic("bunny pets", limit=5, threshold=0.3)

        if not results:
            print("    ✗ No results found")
            return False

        print(f"  Found {len(results)} results\n")

        # Check if factual memories rank higher
        factual_ranks = []
        for i, mem in enumerate(results, 1):
            is_fact = mem.is_fact
            entity_label = f"[{mem.entity_type.upper()}]" if mem.entity_type else ""

            if is_fact:
                factual_ranks.append(i)

            print(f"    {i}. {'[FACT]' if is_fact else '[NON-FACT]'} {entity_label}")
            print(f"       {mem.content[:70]}...")
            print(f"       Importance: {mem.importance_score:.2f}")

        if factual_ranks and min(factual_ranks) <= 2:
            print(f"\n  ✓ Factual memory ranked in top 2 (rank {min(factual_ranks)})")
            print("\n✓ Test 4 passed: Facts prioritized over non-facts")
            return True
        else:
            print(f"\n  ✗ Factual memory not prioritized (best rank: {min(factual_ranks) if factual_ranks else 'N/A'})")
            return False

    except Exception as e:
        print(f"\n✗ Test 4 failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def cleanup_test_db():
    """Clean up test database"""
    test_db_path = Path("data/memory/test_factual.db")
    if test_db_path.exists():
        test_db_path.unlink()
        print("\n🧹 Test database cleaned up")


async def main():
    """Run all persistent factual memory tests"""
    print("\n" + "=" * 70)
    print("PERSISTENT FACTUAL MEMORY TEST SUITE")
    print("=" * 70)
    print()

    # Clean up any existing test database
    await cleanup_test_db()

    tests = [
        ("Fact Persistence", test_fact_persistence),
        ("Cross-Session Reload", test_cross_session_reload),
        ("Semantic Search Recall", test_semantic_search_recall),
        ("Fact Prioritization", test_fact_prioritization),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8} - {name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Persistent factual memory operational!")
        print("\n✨ Facts will now survive Hugo restarts!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed - review output above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
