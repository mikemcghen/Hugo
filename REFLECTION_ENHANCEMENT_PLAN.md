# Hugo Reflection System Enhancement Plan

**Date:** 2025-11-12
**Status:** 🚧 In Progress

---

## Overview

Enhancing Hugo's reflection system with:
- ✅ SQLite reflection storage with embeddings
- ✅ Configuration parameters in .env
- 🚧 Retry logic with exponential backoff
- 🚧 Keyword extraction using TF-IDF
- 🚧 Sentiment analysis
- 🚧 Macro-reflection aggregation
- 🚧 REPL integration for session lifecycle
- 🚧 Verification and testing

---

## Completed Work

### 1. SQLite Schema Enhancement
Added two new tables to `data/sqlite_manager.py`:

**reflections table:**
```sql
CREATE TABLE IF NOT EXISTS reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    summary TEXT NOT NULL,
    insights TEXT,
    patterns TEXT,
    improvements TEXT,
    sentiment REAL,
    keywords TEXT,
    confidence REAL DEFAULT 0.75,
    embedding BLOB,
    metadata TEXT
)
```

**meta_reflections table:**
```sql
CREATE TABLE IF NOT EXISTS meta_reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    time_window_days INTEGER DEFAULT 7,
    summary TEXT NOT NULL,
    insights TEXT,
    patterns TEXT,
    improvements TEXT,
    reflections_analyzed INTEGER,
    confidence REAL DEFAULT 0.7,
    embedding BLOB,
    metadata TEXT
)
```

### 2. SQLite Manager Methods
Added methods to `SQLiteManager` class:
- `store_reflection()` - Store session/macro reflections
- `get_recent_reflections()` - Retrieve recent reflections
- `store_meta_reflection()` - Store weekly meta-reflections
- `get_latest_meta_reflection()` - Get most recent meta-reflection

### 3. Configuration
Added to `.env`:
```env
REFLECTION_MODEL=llama3:8b
REFLECTION_INTERVAL_DAYS=7
REFLECTION_SUMMARY_LENGTH=200
REFLECTION_MAX_RETRIES=2
REFLECTION_RETRY_BACKOFF=2
```

---

## Remaining Work

### 1. Enhance reflection.py

**Add Retry Logic:**
```python
async def _infer_with_retry(self, prompt: str, max_retries: int = 2) -> str:
    """Call Ollama with retry logic"""
    attempt = 0
    last_error = None

    while attempt < max_retries:
        attempt += 1
        try:
            response = requests.post(...)
            if response.status_code == 200:
                return response.json().get("response", "")
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    raise Exception(f"Ollama inference failed after {max_retries} attempts")
```

**Add Keyword Extraction:**
```python
def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
    """Extract keywords using TF-IDF"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from collections import Counter
    import re

    # Simple keyword extraction (frequency-based)
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', ...}
    filtered = [w for w in words if w not in stopwords and len(w) > 3]

    # Get top N most frequent
    counter = Counter(filtered)
    return [word for word, count in counter.most_common(top_n)]
```

**Add Sentiment Analysis:**
```python
def _analyze_sentiment(self, text: str) -> float:
    """Simple sentiment analysis (-1.0 to 1.0)"""
    positive_words = {'good', 'great', 'excellent', 'helpful', ...}
    negative_words = {'bad', 'poor', 'frustrated', 'error', ...}

    words = text.lower().split()
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    return (pos_count - neg_count) / total
```

**Update generate_session_reflection():**
- Add retry logic to Ollama calls
- Extract keywords from conversation
- Calculate sentiment from conversation
- Store in SQLite via new methods
- Generate and store embedding

**Update generate_macro_reflection():**
- Retrieve from SQLite instead of memory search
- Cluster reflections by similarity (FAISS)
- Add retry logic
- Store meta-reflection in SQLite

### 2. Integrate with REPL

**Update runtime/repl.py:**

```python
async def _handle_exit(self):
    """Handle session exit with reflection"""
    print("\n🪞 Generating session reflection...")

    if self.runtime.reflection:
        try:
            reflection = await self.runtime.reflection.generate_session_reflection(
                self.session_id
            )

            # Display brief summary
            print(f"\nSession Summary: {reflection.summary[:150]}...")
            if reflection.insights:
                print(f"\nKey Insights:")
                for insight in reflection.insights[:3]:
                    print(f"  • {insight}")

        except Exception as e:
            self.logger.log_error(e, {"phase": "exit_reflection"})

    print("\nGoodbye! 👋")
```

**Add session start greeting:**

```python
async def run(self):
    """Start REPL with context from last meta-reflection"""
    # ... existing setup ...

    # Check for recent meta-reflection
    if self.runtime.sqlite_manager:
        try:
            meta = await self.runtime.sqlite_manager.get_latest_meta_reflection()
            if meta:
                print(f"\n🪞 Last week you reflected on:")
                print(f"   {meta['summary'][:120]}...")
                print()
        except Exception as e:
            pass  # Silently fail if no meta-reflections

    # ... existing REPL loop ...
```

### 3. Extend MemoryManager

**Add to core/memory.py:**

```python
async def store_reflection(self, reflection: 'Reflection'):
    """
    Store reflection with embedding in both SQLite and FAISS.

    Args:
        reflection: Reflection object to store
    """
    # Format reflection content
    content = f"""[{reflection.type.value.upper()} REFLECTION]
Summary: {reflection.summary}
Insights: {', '.join(reflection.insights)}
Patterns: {', '.join(reflection.patterns_observed)}
"""

    # Generate embedding
    embedding = self.embedding_model.encode(content).tolist()

    # Store in SQLite
    if hasattr(self, 'sqlite_manager'):
        keywords = self._extract_keywords(content)
        sentiment = self._analyze_sentiment(content)

        await self.sqlite_manager.store_reflection(
            session_id=reflection.session_id,
            reflection_type=reflection.type.value,
            summary=reflection.summary,
            insights=reflection.insights,
            patterns=reflection.patterns_observed,
            improvements=reflection.areas_for_improvement,
            sentiment=sentiment,
            keywords=keywords,
            confidence=reflection.confidence,
            embedding=self._serialize_embedding(embedding),
            metadata=reflection.metadata
        )

    # Add to FAISS index
    self.faiss_index.add(np.array([embedding], dtype=np.float32))

    # Store in cache
    memory_entry = MemoryEntry(
        id=None,
        session_id=reflection.session_id or "system",
        timestamp=reflection.timestamp,
        memory_type="reflection",
        content=content,
        embedding=embedding,
        metadata={
            "reflection_type": reflection.type.value,
            "confidence": reflection.confidence
        },
        importance_score=0.9
    )

    self.cache.append(memory_entry)
```

### 4. Create Verification Script

**Create verify_reflection.py:**

```python
#!/usr/bin/env python3
"""
Reflection System Verification Script
Tests reflection generation, storage, and retrieval
"""

import asyncio
import sys
from pathlib import Path

# Add Hugo to path
sys.path.insert(0, str(Path(__file__).parent))

from core.runtime_manager import RuntimeManager
from core.memory import MemoryEntry
from datetime import datetime


async def test_session_reflection():
    """Test session reflection generation"""
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
        ("assistant", "Reflections allow me to learn from conversations..."),
        ("user", "How do you store them?"),
        ("assistant", "I use SQLite for persistence and FAISS for semantic search...")
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

    print(f"✓ Stored {len(messages)} messages")

    # Generate reflection
    if runtime.reflection:
        reflection = await runtime.reflection.generate_session_reflection(session_id)

        print(f"\n✓ Reflection generated successfully")
        print(f"  Summary: {reflection.summary}")
        print(f"  Insights: {len(reflection.insights)}")
        print(f"  Patterns: {len(reflection.patterns_observed)}")
        print(f"  Confidence: {reflection.confidence:.2%}")

        # Verify storage
        if runtime.sqlite_manager:
            reflections = await runtime.sqlite_manager.get_recent_reflections(limit=1)
            if reflections:
                print(f"\n✓ Reflection persisted to SQLite (ID: {reflections[0]['id']})")
            else:
                print(f"\n✗ Reflection not found in SQLite")

        return reflection

    print("\n✗ Reflection engine not initialized")
    return None


async def test_macro_reflection():
    """Test macro-reflection generation"""
    print("\n" + "=" * 60)
    print("TEST 2: Macro-Reflection Generation")
    print("=" * 60)

    runtime = RuntimeManager()
    await runtime.initialize()

    if runtime.reflection:
        # Generate macro-reflection from recent reflections
        macro = await runtime.reflection.generate_macro_reflection(time_window_days=7)

        print(f"\n✓ Macro-reflection generated")
        print(f"  Summary: {macro.summary[:100]}...")
        print(f"  Strategic Insights: {len(macro.insights)}")
        print(f"  Long-term Patterns: {len(macro.patterns_observed)}")

        # Verify storage
        if runtime.sqlite_manager:
            latest_meta = await runtime.sqlite_manager.get_latest_meta_reflection()
            if latest_meta:
                print(f"\n✓ Meta-reflection persisted to SQLite (ID: {latest_meta['id']})")
            else:
                print(f"\n✗ Meta-reflection not found in SQLite")

        return macro

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
            print(f"  {i}. [{ref['type']}] {ref['summary'][:60]}...")

    # Semantic search via memory
    if runtime.memory:
        results = await runtime.memory.search_semantic(
            "reflection learning conversation",
            limit=3,
            threshold=0.5
        )
        print(f"\n✓ Found {len(results)} reflections via semantic search")

        for result in results:
            print(f"  • {result.content[:80]}...")


async def main():
    """Run all reflection tests"""
    print("\n" + "=" * 60)
    print("HUGO REFLECTION SYSTEM VERIFICATION")
    print("=" * 60)

    try:
        await test_session_reflection()
        await test_macro_reflection()
        await test_reflection_retrieval()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Implementation Priority

1. ✅ SQLite schema and methods (COMPLETED)
2. ✅ Configuration in .env (COMPLETED)
3. 🚧 Enhance reflection.py with retry/keywords/sentiment
4. 🚧 Integrate with REPL lifecycle
5. 🚧 Update MemoryManager
6. 🚧 Create verification script
7. 🚧 Test and validate

---

## Next Steps

Continue with enhancing `core/reflection.py` to add:
1. `_infer_with_retry()` method for resilient Ollama calls
2. `_extract_keywords()` for keyword extraction
3. `_analyze_sentiment()` for sentiment scoring
4. Update `generate_session_reflection()` to use SQLite storage
5. Update `generate_macro_reflection()` for weekly aggregation

---

**Status:** Schema and configuration complete. Implementation of core logic in progress.
