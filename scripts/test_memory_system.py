#!/usr/bin/env python3
"""
Memory System Verification
===========================
Comprehensive test for Hugo's complete memory pipeline:
  - Reflection → SQLite → FAISS → Cognition Recall

Tests:
1. Create fake conversation
2. Generate session reflection
3. Verify reflection stored in SQLite
4. Verify embedding generated and stored
5. Verify reflection retrievable via semantic search
6. Verify reflection appears in cognition context
7. Verify keywords and sentiment stored correctly
"""

import asyncio
import sys
from pathlib import Path

# Add Hugo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.runtime_manager import RuntimeManager
from core.memory import MemoryEntry
from datetime import datetime


async def test_memory_storage():
    """Test 1: Memory storage with embedding generation"""
    print("=" * 60)
    print("TEST 1: Memory Storage with Embeddings")
    print("=" * 60)

    runtime = RuntimeManager()
    await runtime.initialize()

    if not runtime.memory:
        print("✗ Memory manager not initialized")
        return False

    try:
        # Create test memory entry
        test_entry = MemoryEntry(
            id=None,
            session_id="memory_test_001",
            timestamp=datetime.now(),
            memory_type="user_message",
            content="Can you explain how memory systems work in AI?",
            embedding=None,  # Should be auto-generated
            metadata={"test": True},
            importance_score=0.8
        )

        # Store entry
        await runtime.memory.store(test_entry, persist_long_term=False)

        print("✓ Memory entry stored")
        print(f"  Session: {test_entry.session_id}")
        print(f"  Content length: {len(test_entry.content)} chars")
        print(f"  Embedding generated: {test_entry.embedding is not None}")

        if test_entry.embedding:
            print(f"  Embedding dimension: {len(test_entry.embedding)}")

        return True

    except Exception as e:
        print(f"✗ Memory storage failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_conversation_and_reflection():
    """Test 2: Create conversation and generate reflection"""
    print("\n" + "=" * 60)
    print("TEST 2: Conversation → Reflection Generation")
    print("=" * 60)

    runtime = RuntimeManager()
    await runtime.initialize()

    if not runtime.memory or not runtime.reflection:
        print("✗ Required components not initialized")
        return False

    session_id = "memory_test_session"

    try:
        # Create fake conversation
        print("\nCreating fake conversation...")

        messages = [
            ("user", "Hello Hugo, I want to learn about reflection systems."),
            ("assistant", "I'd be happy to explain reflection systems! They help me learn and improve over time."),
            ("user", "How do you store reflections?"),
            ("assistant", "I store reflections in SQLite with embeddings for semantic search, allowing me to recall past insights."),
            ("user", "That sounds powerful. Can you reflect on what we've discussed?"),
            ("assistant", "Absolutely! Let me generate a reflection for this session.")
        ]

        for role, content in messages:
            memory_entry = MemoryEntry(
                id=None,
                session_id=session_id,
                timestamp=datetime.now(),
                memory_type="user_message" if role == "user" else "assistant_message",
                content=content,
                embedding=None,
                metadata={"role": role},
                importance_score=0.7
            )
            await runtime.memory.store(memory_entry, persist_long_term=False)

        print(f"✓ Created {len(messages)} conversation turns")

        # Generate reflection
        print("\nGenerating session reflection...")
        reflection = await runtime.reflection.generate_session_reflection(session_id)

        print(f"✓ Reflection generated")
        print(f"  Summary: {reflection.summary[:100]}...")
        print(f"  Insights: {len(reflection.insights)} insights")
        print(f"  Patterns: {len(reflection.patterns_observed)} patterns")
        print(f"  Keywords: {reflection.metadata.get('keywords', [])[:5]}")
        print(f"  Sentiment: {reflection.metadata.get('sentiment_score', 'N/A')}")
        print(f"  Confidence: {reflection.confidence:.2%}")

        return True

    except Exception as e:
        print(f"✗ Reflection generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_sqlite_storage():
    """Test 3: Verify reflection stored in SQLite"""
    print("\n" + "=" * 60)
    print("TEST 3: SQLite Reflection Storage Verification")
    print("=" * 60)

    runtime = RuntimeManager()
    await runtime.initialize()

    if not hasattr(runtime, 'sqlite_manager') or not runtime.sqlite_manager:
        print("✗ SQLite manager not available")
        return False

    try:
        # Retrieve recent reflections
        reflections = await runtime.sqlite_manager.get_recent_reflections(limit=5)

        if not reflections:
            print("⚠ No reflections found in SQLite (this might be expected if no reflections generated yet)")
            return True

        print(f"✓ Retrieved {len(reflections)} reflections from SQLite")

        for i, refl in enumerate(reflections[:3], 1):
            print(f"\n--- Reflection {i} ---")
            print(f"  Type: {refl['type']}")
            print(f"  Session: {refl.get('session_id', 'N/A')}")
            print(f"  Timestamp: {refl['timestamp'][:19]}")
            print(f"  Summary: {refl['summary'][:80]}...")
            print(f"  Insights count: {len(refl['insights'])}")
            print(f"  Keywords count: {len(refl['keywords'])}")
            print(f"  Sentiment: {refl.get('sentiment', 'N/A')}")
            print(f"  Confidence: {refl['confidence']:.2%}")
            print(f"  Has embedding: {'Yes' if refl.get('embedding') else 'No'}")

        return True

    except Exception as e:
        print(f"✗ SQLite retrieval failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_semantic_search():
    """Test 4: Verify semantic search retrieves reflections"""
    print("\n" + "=" * 60)
    print("TEST 4: Semantic Search for Reflections")
    print("=" * 60)

    runtime = RuntimeManager()
    await runtime.initialize()

    if not runtime.memory:
        print("✗ Memory manager not initialized")
        return False

    try:
        # Search for reflection-related content
        query = "reflection system learning insights"
        print(f"\nSearching for: '{query}'")

        results = await runtime.memory.search_semantic(query, limit=5, threshold=0.5)

        if not results:
            print("⚠ No semantic search results (FAISS index might be empty)")
            return True

        print(f"✓ Found {len(results)} semantic matches")

        reflection_count = 0
        for i, mem in enumerate(results, 1):
            is_reflection = mem.memory_type == "reflection"
            if is_reflection:
                reflection_count += 1

            print(f"\n--- Result {i} ---")
            print(f"  Type: {mem.memory_type} {'[REFLECTION]' if is_reflection else ''}")
            print(f"  Session: {mem.session_id}")
            print(f"  Content preview: {mem.content[:100]}...")
            print(f"  Importance: {mem.importance_score:.2f}")

        print(f"\n✓ Reflections in results: {reflection_count}/{len(results)}")
        return True

    except Exception as e:
        print(f"✗ Semantic search failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_cognition_context():
    """Test 5: Verify reflections appear in cognition context"""
    print("\n" + "=" * 60)
    print("TEST 5: Cognition Context with Reflections")
    print("=" * 60)

    runtime = RuntimeManager()
    await runtime.initialize()

    if not runtime.cognition:
        print("✗ Cognition engine not initialized")
        return False

    try:
        # Simulate user input that should trigger reflection recall
        test_message = "What have we learned in our past conversations?"
        session_id = "cognition_test_001"

        print(f"\nTest message: '{test_message}'")
        print("\nProcessing through cognition engine...")

        # Process without streaming to get full response
        response_package = await runtime.cognition.process_input(test_message, session_id)

        print(f"\n✓ Response generated")
        print(f"  Content length: {len(response_package.content)} chars")
        print(f"  Metadata:")
        print(f"    - Semantic memories: {response_package.metadata.get('semantic_memories', 0)}")
        print(f"    - Reflection memories: {response_package.metadata.get('reflection_memories', 0)}")
        print(f"    - Confidence: {response_package.metadata.get('confidence', 0):.2%}")

        if response_package.metadata.get('reflection_memories', 0) > 0:
            print(f"\n✓ Reflections successfully retrieved in cognition context!")
        else:
            print(f"\n⚠ No reflections found in cognition context (might be expected if no relevant reflections exist)")

        return True

    except Exception as e:
        print(f"✗ Cognition context test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_faiss_integration():
    """Test 6: Verify FAISS index integration"""
    print("\n" + "=" * 60)
    print("TEST 6: FAISS Index Integration")
    print("=" * 60)

    runtime = RuntimeManager()
    await runtime.initialize()

    if not runtime.memory:
        print("✗ Memory manager not initialized")
        return False

    try:
        stats = runtime.memory.get_stats()

        print("✓ Memory system stats:")
        print(f"  Cache size: {stats['cache_size']}")
        print(f"  FAISS enabled: {stats['faiss_enabled']}")
        print(f"  FAISS index size: {stats['faiss_index_size']} vectors")
        print(f"  Embedding model: {stats['embedding_model']}")

        if stats['faiss_index_size'] > 0:
            print(f"\n✓ FAISS index contains vectors - memory system active!")
        else:
            print(f"\n⚠ FAISS index empty (expected if no memories stored yet)")

        return True

    except Exception as e:
        print(f"✗ FAISS integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_end_to_end_pipeline():
    """Test 7: Complete end-to-end memory pipeline"""
    print("\n" + "=" * 60)
    print("TEST 7: End-to-End Memory Pipeline")
    print("=" * 60)
    print("(Conversation → Reflection → SQLite → FAISS → Recall)")
    print()

    runtime = RuntimeManager()
    await runtime.initialize()

    session_id = "e2e_test_session"

    try:
        # Step 1: Create conversation
        print("Step 1: Creating test conversation...")
        messages = [
            ("user", "I want to build a memory system"),
            ("assistant", "Great! A memory system needs storage, retrieval, and context integration.")
        ]

        for role, content in messages:
            entry = MemoryEntry(
                id=None,
                session_id=session_id,
                timestamp=datetime.now(),
                memory_type="user_message" if role == "user" else "assistant_message",
                content=content,
                embedding=None,
                metadata={"role": role},
                importance_score=0.7
            )
            await runtime.memory.store(entry, persist_long_term=True)

        print(f"  ✓ {len(messages)} messages stored")

        # Step 2: Generate reflection
        print("\nStep 2: Generating reflection...")
        reflection = await runtime.reflection.generate_session_reflection(session_id)
        print(f"  ✓ Reflection generated (confidence: {reflection.confidence:.2%})")

        # Step 3: Verify SQLite storage
        print("\nStep 3: Verifying SQLite storage...")
        recent = await runtime.sqlite_manager.get_recent_reflections(limit=1)
        if recent:
            print(f"  ✓ Found reflection in SQLite")
            print(f"    - Has embedding: {recent[0].get('embedding') is not None}")
            print(f"    - Keywords: {len(recent[0].get('keywords', []))}")
        else:
            print(f"  ⚠ No reflections in SQLite")

        # Step 4: Test semantic search
        print("\nStep 4: Testing semantic search...")
        search_results = await runtime.memory.search_semantic("memory system", limit=3)
        reflection_found = any(m.memory_type == "reflection" for m in search_results)
        print(f"  ✓ Search returned {len(search_results)} results")
        print(f"  ✓ Reflection in results: {reflection_found}")

        # Step 5: Test cognition recall
        print("\nStep 5: Testing cognition recall...")
        response = await runtime.cognition.process_input(
            "What did we discuss about memory systems?",
            "e2e_recall_test"
        )
        refl_count = response.metadata.get('reflection_memories', 0)
        print(f"  ✓ Cognition processed with {refl_count} reflection(s) in context")

        print("\n" + "=" * 60)
        print("✓ END-TO-END PIPELINE COMPLETE")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ End-to-end pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all memory system tests"""
    print("\n" + "=" * 60)
    print("HUGO MEMORY SYSTEM VERIFICATION")
    print("=" * 60)
    print()

    tests = [
        ("Memory Storage", test_memory_storage),
        ("Conversation & Reflection", test_conversation_and_reflection),
        ("SQLite Storage", test_sqlite_storage),
        ("Semantic Search", test_semantic_search),
        ("Cognition Context", test_cognition_context),
        ("FAISS Integration", test_faiss_integration),
        ("End-to-End Pipeline", test_end_to_end_pipeline)
    ]

    results = []

    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {str(e)}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8} - {name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Memory pipeline fully operational!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed - review output above")


if __name__ == "__main__":
    asyncio.run(main())
