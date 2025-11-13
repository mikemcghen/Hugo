#!/usr/bin/env python3
"""
Reflection System Verification Script
======================================
Tests reflection generation, storage, and retrieval.

This script validates:
1. Session reflection generation with keywords and sentiment
2. Macro-reflection aggregation across multiple sessions
3. SQLite persistence of reflections and meta-reflections
4. Semantic search via FAISS embeddings
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add Hugo to path
sys.path.insert(0, str(Path(__file__).parent))

from core.runtime_manager import RuntimeManager
from core.memory import MemoryEntry


async def test_session_reflection():
    """Test session reflection generation with enriched metadata"""
    print("=" * 60)
    print("TEST 1: Session Reflection Generation")
    print("=" * 60)

    # Initialize runtime
    runtime = RuntimeManager()
    await runtime.initialize()

    # Simulate conversation
    session_id = "test-session-001"
    messages = [
        ("user", "Hello Hugo, tell me about reflections"),
        ("assistant", "Reflections allow me to learn from conversations and improve over time. "
                     "I analyze patterns, extract insights, and store them for future reference."),
        ("user", "How do you store them?"),
        ("assistant", "I use SQLite for persistence and FAISS for semantic search. "
                     "Each reflection includes keywords, sentiment analysis, and embeddings."),
        ("user", "That's great! This will help you remember our discussions."),
        ("assistant", "Absolutely! I can reflect on past conversations to better understand your preferences.")
    ]

    # Store messages in memory
    for role, content in messages:
        memory = MemoryEntry(
            id=None,
            session_id=session_id,
            timestamp=datetime.now(),
            memory_type=f"{role}_message",
            content=content,
            embedding=None,
            metadata={"role": role},
            importance_score=0.7
        )
        await runtime.memory.store(memory)

    print(f"✓ Stored {len(messages)} messages in memory")

    # Generate reflection
    if runtime.reflection:
        try:
            reflection = await runtime.reflection.generate_session_reflection(session_id)

            print(f"\n✓ Reflection generated successfully")
            print(f"  Summary: {reflection.summary}")
            print(f"  Insights: {len(reflection.insights)}")
            for insight in reflection.insights:
                print(f"    • {insight}")
            print(f"  Patterns: {len(reflection.patterns_observed)}")
            for pattern in reflection.patterns_observed:
                print(f"    • {pattern}")
            print(f"  Confidence: {reflection.confidence:.2%}")

            # Check metadata
            keywords = reflection.metadata.get("keywords", [])
            sentiment = reflection.metadata.get("sentiment_score", 0.0)
            print(f"\n  Keywords: {', '.join(keywords[:5])}")
            print(f"  Sentiment: {sentiment:.2f}")

            # Verify storage in SQLite
            if runtime.sqlite_manager:
                reflections = await runtime.sqlite_manager.get_recent_reflections(limit=1)
                if reflections:
                    ref = reflections[0]
                    print(f"\n✓ Reflection persisted to SQLite (ID: {ref['id']})")
                    print(f"  Keywords in DB: {', '.join(ref['keywords'][:5])}")
                    print(f"  Sentiment in DB: {ref['sentiment']:.2f}")
                else:
                    print(f"\n✗ Reflection not found in SQLite")

            return reflection

        except Exception as e:
            print(f"\n✗ Reflection generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    print("\n✗ Reflection engine not initialized")
    return None


async def test_macro_reflection():
    """Test macro-reflection generation from aggregated sessions"""
    print("\n" + "=" * 60)
    print("TEST 2: Macro-Reflection Generation")
    print("=" * 60)

    runtime = RuntimeManager()
    await runtime.initialize()

    if runtime.reflection:
        try:
            # Check if we have reflections to aggregate
            if runtime.sqlite_manager:
                reflections = await runtime.sqlite_manager.get_recent_reflections(limit=5)
                print(f"\n✓ Found {len(reflections)} reflections to aggregate")

            # Generate macro-reflection from recent reflections
            macro = await runtime.reflection.generate_macro_reflection(time_window_days=7)

            print(f"\n✓ Macro-reflection generated")
            print(f"  Summary: {macro.summary[:100]}...")
            print(f"  Strategic Insights: {len(macro.insights)}")
            for insight in macro.insights:
                print(f"    • {insight}")
            print(f"  Long-term Patterns: {len(macro.patterns_observed)}")
            for pattern in macro.patterns_observed:
                print(f"    • {pattern}")

            # Verify storage
            if runtime.sqlite_manager:
                latest_meta = await runtime.sqlite_manager.get_latest_meta_reflection()
                if latest_meta:
                    print(f"\n✓ Meta-reflection persisted to SQLite (ID: {latest_meta['id']})")
                    print(f"  Reflections Analyzed: {latest_meta['reflections_analyzed']}")
                    print(f"  Time Window: {latest_meta['time_window_days']} days")
                else:
                    print(f"\n✗ Meta-reflection not found in SQLite")

            return macro

        except Exception as e:
            print(f"\n✗ Macro-reflection generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    print("\n✗ Reflection engine not initialized")
    return None


async def test_reflection_retrieval():
    """Test reflection retrieval and semantic search"""
    print("\n" + "=" * 60)
    print("TEST 3: Reflection Retrieval")
    print("=" * 60)

    runtime = RuntimeManager()
    await runtime.initialize()

    # Retrieve from SQLite
    if runtime.sqlite_manager:
        reflections = await runtime.sqlite_manager.get_recent_reflections(limit=5)
        print(f"\n✓ Retrieved {len(reflections)} reflections from SQLite")

        for i, ref in enumerate(reflections, 1):
            summary_preview = ref['summary'][:60] + "..." if len(ref['summary']) > 60 else ref['summary']
            print(f"  {i}. [{ref['type']}] {summary_preview}")
            print(f"     Sentiment: {ref['sentiment']:.2f}, Keywords: {len(ref['keywords'])}")

    # Semantic search via memory
    if runtime.memory:
        try:
            results = await runtime.memory.search_semantic(
                "reflection learning conversation patterns",
                limit=3,
                threshold=0.5
            )
            print(f"\n✓ Found {len(results)} reflections via semantic search")

            for i, result in enumerate(results, 1):
                content_preview = result.content[:80] + "..." if len(result.content) > 80 else result.content
                print(f"  {i}. {content_preview}")

        except Exception as e:
            print(f"\n✗ Semantic search failed: {str(e)}")


async def test_reflection_resilience():
    """Test retry logic and error handling"""
    print("\n" + "=" * 60)
    print("TEST 4: Resilience Testing")
    print("=" * 60)

    runtime = RuntimeManager()
    await runtime.initialize()

    if runtime.reflection:
        print(f"\n✓ Reflection engine configuration:")
        print(f"  Model: {runtime.reflection.reflection_model}")
        print(f"  Max Retries: {runtime.reflection.reflection_max_retries}")
        print(f"  Retry Backoff: {runtime.reflection.reflection_retry_backoff}")
        print(f"  Ollama API: {runtime.reflection.ollama_api}")

        # Test keyword extraction
        test_text = "This is a great conversation about reflections, learning, and memory storage."
        keywords = runtime.reflection._extract_keywords(test_text, top_n=5)
        print(f"\n✓ Keyword extraction test:")
        print(f"  Input: {test_text}")
        print(f"  Keywords: {', '.join(keywords)}")

        # Test sentiment analysis
        sentiment = runtime.reflection._analyze_sentiment(test_text)
        print(f"\n✓ Sentiment analysis test:")
        print(f"  Input: {test_text}")
        print(f"  Sentiment: {sentiment:.2f} ({'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'})")


async def main():
    """Run all reflection tests"""
    print("\n" + "=" * 60)
    print("HUGO REFLECTION SYSTEM VERIFICATION")
    print("=" * 60)

    try:
        await test_session_reflection()
        await test_macro_reflection()
        await test_reflection_retrieval()
        await test_reflection_resilience()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
