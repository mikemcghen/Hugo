"""
SQLite Manager
--------------
Manages short-term session memory using SQLite.

Tables:
- recent_messages: Recent conversation messages
- session_summary: Session metadata and summaries
- pending_tasks: Active tasks and reminders
- context_embeddings: Cached embeddings for quick lookup
"""

import sqlite3
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime


class SQLiteManager:
    """
    Manages Hugo's short-term memory in SQLite.

    Fast, ephemeral storage for current session context.
    """

    def __init__(self, db_path: str = "data/memory/hugo_session.db"):
        """
        Initialize SQLite manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None

    async def connect(self):
        """Connect to SQLite database and create tables"""
        # Run in executor since sqlite3 is blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._connect_sync)

    def _connect_sync(self):
        """Synchronous connection (runs in executor)"""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create schema if it doesn't exist"""
        cursor = self.conn.cursor()

        # Recent messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recent_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                importance REAL DEFAULT 0.5
            )
        """)

        # Session summary table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_summary (
                session_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                message_count INTEGER DEFAULT 0,
                context_summary TEXT,
                mood TEXT,
                metadata TEXT
            )
        """)

        # Pending tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                task_description TEXT NOT NULL,
                priority INTEGER DEFAULT 3,
                status TEXT DEFAULT 'pending',
                due_date TEXT,
                metadata TEXT
            )
        """)

        # Context embeddings cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS context_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT UNIQUE NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        """)

        # Reflections table
        cursor.execute("""
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
        """)

        # Meta-reflections table (aggregated weekly insights)
        cursor.execute("""
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
        """)

        # Long-term memories table (for persistent factual memories)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                importance_score REAL DEFAULT 0.5,
                is_fact INTEGER DEFAULT 0,
                entity_type TEXT
            )
        """)

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON recent_messages(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON recent_messages(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_session ON pending_tasks(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reflections_session ON reflections(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reflections_timestamp ON reflections(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reflections_type ON reflections(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_meta_reflections_timestamp ON meta_reflections(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_is_fact ON memories(is_fact)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_entity_type ON memories(entity_type)")

        self.conn.commit()

    async def store_message(self, session_id: str, role: str, content: str,
                          embedding: Optional[bytes] = None,
                          metadata: Optional[Dict] = None,
                          importance: float = 0.5):
        """
        Store a message in short-term memory.

        Args:
            session_id: Session identifier
            role: Message role (user, assistant, system)
            content: Message content
            embedding: Optional embedding vector (as bytes)
            metadata: Optional metadata dictionary
            importance: Importance score (0.0 to 1.0)
        """
        import json

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._store_message_sync,
                                  session_id, role, content, embedding,
                                  json.dumps(metadata) if metadata else None,
                                  importance)

    def _store_message_sync(self, session_id, role, content, embedding, metadata_json, importance):
        """Synchronous message storage"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO recent_messages
            (session_id, timestamp, role, content, embedding, metadata, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, datetime.now().isoformat(), role, content,
              embedding, metadata_json, importance))

        self.conn.commit()

    async def get_recent_messages(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve recent messages for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages

        Returns:
            List of message dictionaries
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_messages_sync, session_id, limit)

    def _get_messages_sync(self, session_id, limit):
        """Synchronous message retrieval"""
        import json

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, session_id, timestamp, role, content, metadata, importance
            FROM recent_messages
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))

        messages = []
        for row in cursor.fetchall():
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
            messages.append({
                'id': row['id'],
                'session_id': row['session_id'],
                'timestamp': row['timestamp'],
                'role': row['role'],
                'content': row['content'],
                'metadata': metadata,
                'importance': row['importance']
            })

        return messages

    async def create_session(self, session_id: str, metadata: Optional[Dict] = None):
        """Create a new session entry"""
        import json

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._create_session_sync,
                                  session_id, json.dumps(metadata) if metadata else None)

    def _create_session_sync(self, session_id, metadata_json):
        """Synchronous session creation"""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute("""
            INSERT OR REPLACE INTO session_summary
            (session_id, started_at, last_activity, message_count, metadata)
            VALUES (?, ?, ?, 0, ?)
        """, (session_id, now, now, metadata_json))

        self.conn.commit()

    async def update_session(self, session_id: str, context_summary: Optional[str] = None,
                           mood: Optional[str] = None):
        """Update session summary"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._update_session_sync,
                                  session_id, context_summary, mood)

    def _update_session_sync(self, session_id, context_summary, mood):
        """Synchronous session update"""
        cursor = self.conn.cursor()

        updates = ["last_activity = ?"]
        params = [datetime.now().isoformat()]

        if context_summary:
            updates.append("context_summary = ?")
            params.append(context_summary)

        if mood:
            updates.append("mood = ?")
            params.append(mood)

        params.append(session_id)

        cursor.execute(f"""
            UPDATE session_summary
            SET {', '.join(updates)}
            WHERE session_id = ?
        """, params)

        self.conn.commit()

    async def clear_old_messages(self, days: int = 7):
        """
        Clear messages older than specified days.

        Args:
            days: Age threshold in days
        """
        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._clear_old_messages_sync, cutoff)

    def _clear_old_messages_sync(self, cutoff):
        """Synchronous message clearing"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM recent_messages WHERE timestamp < ?", (cutoff,))
        self.conn.commit()

    async def store_reflection(self, session_id: Optional[str], reflection_type: str,
                              summary: str, insights: List[str], patterns: List[str],
                              improvements: List[str], sentiment: Optional[float] = None,
                              keywords: Optional[List[str]] = None, confidence: float = 0.75,
                              embedding: Optional[bytes] = None, metadata: Optional[Dict] = None) -> int:
        """
        Store a reflection in the database.

        Args:
            session_id: Session identifier (None for macro reflections)
            reflection_type: Type of reflection (session, macro, performance, etc.)
            summary: Reflection summary text
            insights: List of key insights
            patterns: List of observed patterns
            improvements: List of improvement areas
            sentiment: Optional sentiment score
            keywords: Optional list of extracted keywords
            confidence: Confidence score (0.0 to 1.0)
            embedding: Optional embedding vector (as bytes)
            metadata: Optional metadata dictionary

        Returns:
            ID of the stored reflection
        """
        import json

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._store_reflection_sync,
            session_id, reflection_type, summary,
            json.dumps(insights) if insights else None,
            json.dumps(patterns) if patterns else None,
            json.dumps(improvements) if improvements else None,
            sentiment,
            json.dumps(keywords) if keywords else None,
            confidence, embedding,
            json.dumps(metadata) if metadata else None
        )

    def _store_reflection_sync(self, session_id, reflection_type, summary,
                               insights_json, patterns_json, improvements_json,
                               sentiment, keywords_json, confidence, embedding, metadata_json):
        """Synchronous reflection storage"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO reflections
            (session_id, type, timestamp, summary, insights, patterns, improvements,
             sentiment, keywords, confidence, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, reflection_type, datetime.now().isoformat(),
              summary, insights_json, patterns_json, improvements_json,
              sentiment, keywords_json, confidence, embedding, metadata_json))

        self.conn.commit()
        return cursor.lastrowid

    async def get_recent_reflections(self, reflection_type: Optional[str] = None,
                                     limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent reflections.

        Args:
            reflection_type: Optional filter by reflection type
            limit: Maximum number of reflections

        Returns:
            List of reflection dictionaries
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_reflections_sync,
                                         reflection_type, limit)

    def _get_reflections_sync(self, reflection_type, limit):
        """Synchronous reflection retrieval"""
        import json

        cursor = self.conn.cursor()

        if reflection_type:
            cursor.execute("""
                SELECT id, session_id, type, timestamp, summary, insights, patterns,
                       improvements, sentiment, keywords, confidence, metadata
                FROM reflections
                WHERE type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (reflection_type, limit))
        else:
            cursor.execute("""
                SELECT id, session_id, type, timestamp, summary, insights, patterns,
                       improvements, sentiment, keywords, confidence, metadata
                FROM reflections
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

        reflections = []
        for row in cursor.fetchall():
            reflections.append({
                'id': row['id'],
                'session_id': row['session_id'],
                'type': row['type'],
                'timestamp': row['timestamp'],
                'summary': row['summary'],
                'insights': json.loads(row['insights']) if row['insights'] else [],
                'patterns': json.loads(row['patterns']) if row['patterns'] else [],
                'improvements': json.loads(row['improvements']) if row['improvements'] else [],
                'sentiment': row['sentiment'],
                'keywords': json.loads(row['keywords']) if row['keywords'] else [],
                'confidence': row['confidence'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else {}
            })

        return reflections

    async def store_meta_reflection(self, summary: str, insights: List[str],
                                    patterns: List[str], improvements: List[str],
                                    reflections_analyzed: int, time_window_days: int = 7,
                                    confidence: float = 0.7, embedding: Optional[bytes] = None,
                                    metadata: Optional[Dict] = None) -> int:
        """
        Store a meta-reflection (aggregated from multiple reflections).

        Args:
            summary: Meta-reflection summary
            insights: Strategic insights
            patterns: Long-term patterns
            improvements: Strategic improvement areas
            reflections_analyzed: Number of reflections analyzed
            time_window_days: Time window in days
            confidence: Confidence score
            embedding: Optional embedding vector
            metadata: Optional metadata

        Returns:
            ID of the stored meta-reflection
        """
        import json

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._store_meta_reflection_sync,
            summary,
            json.dumps(insights) if insights else None,
            json.dumps(patterns) if patterns else None,
            json.dumps(improvements) if improvements else None,
            reflections_analyzed, time_window_days, confidence, embedding,
            json.dumps(metadata) if metadata else None
        )

    def _store_meta_reflection_sync(self, summary, insights_json, patterns_json,
                                    improvements_json, reflections_analyzed,
                                    time_window_days, confidence, embedding, metadata_json):
        """Synchronous meta-reflection storage"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO meta_reflections
            (created_at, time_window_days, summary, insights, patterns, improvements,
             reflections_analyzed, confidence, embedding, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), time_window_days, summary,
              insights_json, patterns_json, improvements_json,
              reflections_analyzed, confidence, embedding, metadata_json))

        self.conn.commit()
        return cursor.lastrowid

    async def get_latest_meta_reflection(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent meta-reflection.

        Returns:
            Meta-reflection dictionary or None
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_latest_meta_reflection_sync)

    def _get_latest_meta_reflection_sync(self):
        """Synchronous latest meta-reflection retrieval"""
        import json

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, created_at, time_window_days, summary, insights, patterns,
                   improvements, reflections_analyzed, confidence, metadata
            FROM meta_reflections
            ORDER BY created_at DESC
            LIMIT 1
        """)

        row = cursor.fetchone()
        if not row:
            return None

        return {
            'id': row['id'],
            'created_at': row['created_at'],
            'time_window_days': row['time_window_days'],
            'summary': row['summary'],
            'insights': json.loads(row['insights']) if row['insights'] else [],
            'patterns': json.loads(row['patterns']) if row['patterns'] else [],
            'improvements': json.loads(row['improvements']) if row['improvements'] else [],
            'reflections_analyzed': row['reflections_analyzed'],
            'confidence': row['confidence'],
            'metadata': json.loads(row['metadata']) if row['metadata'] else {}
        }

    async def close(self):
        """Close database connection"""
        if self.conn:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.conn.close)

    async def store_memory(self, session_id: str, memory_type: str, content: str,
                          embedding: Optional[bytes] = None, metadata: Optional[Dict] = None,
                          importance_score: float = 0.5, is_fact: bool = False,
                          entity_type: Optional[str] = None) -> int:
        """
        Store a memory entry in long-term storage.

        Args:
            session_id: Session identifier
            memory_type: Type of memory (user_message, assistant_message, reflection, etc.)
            content: Memory content
            embedding: Optional embedding vector (as bytes)
            metadata: Optional metadata dictionary
            importance_score: Importance score (0.0 to 1.0)
            is_fact: Whether this is a factual memory
            entity_type: Optional entity type for facts (animal, person, etc.)

        Returns:
            ID of the stored memory
        """
        import json

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._store_memory_sync,
            session_id, memory_type, content, embedding,
            json.dumps(metadata) if metadata else None,
            importance_score, 1 if is_fact else 0, entity_type
        )

    def _store_memory_sync(self, session_id, memory_type, content, embedding,
                           metadata_json, importance_score, is_fact, entity_type):
        """Synchronous memory storage"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO memories
            (session_id, timestamp, memory_type, content, embedding, metadata,
             importance_score, is_fact, entity_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, datetime.now().isoformat(), memory_type, content,
              embedding, metadata_json, importance_score, is_fact, entity_type))

        self.conn.commit()
        return cursor.lastrowid

    async def get_factual_memories(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all factual memories from storage.

        Args:
            limit: Optional maximum number of memories to return

        Returns:
            List of memory dictionaries with id, content, entity_type, embedding, timestamp
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_factual_memories_sync, limit)

    def _get_factual_memories_sync(self, limit):
        """Synchronous factual memory retrieval"""
        import json

        cursor = self.conn.cursor()

        if limit:
            cursor.execute("""
                SELECT id, session_id, timestamp, memory_type, content, embedding,
                       metadata, importance_score, entity_type
                FROM memories
                WHERE is_fact = 1
                ORDER BY importance_score DESC, timestamp DESC
                LIMIT ?
            """, (limit,))
        else:
            cursor.execute("""
                SELECT id, session_id, timestamp, memory_type, content, embedding,
                       metadata, importance_score, entity_type
                FROM memories
                WHERE is_fact = 1
                ORDER BY importance_score DESC, timestamp DESC
            """)

        memories = []
        for row in cursor.fetchall():
            memories.append({
                'id': row['id'],
                'session_id': row['session_id'],
                'timestamp': row['timestamp'],
                'memory_type': row['memory_type'],
                'content': row['content'],
                'embedding': row['embedding'],  # Bytes
                'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                'importance_score': row['importance_score'],
                'entity_type': row['entity_type']
            })

        return memories

    async def get_all_memories_with_embeddings(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all memories that have embeddings (for FAISS rebuild).

        Args:
            limit: Optional maximum number of memories to return

        Returns:
            List of memory dictionaries
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_all_memories_with_embeddings_sync, limit)

    def _get_all_memories_with_embeddings_sync(self, limit):
        """Synchronous retrieval of memories with embeddings"""
        import json

        cursor = self.conn.cursor()

        query = """
            SELECT id, session_id, timestamp, memory_type, content, embedding,
                   metadata, importance_score, is_fact, entity_type
            FROM memories
            WHERE embedding IS NOT NULL
            ORDER BY timestamp DESC
        """

        if limit:
            query += " LIMIT ?"
            cursor.execute(query, (limit,))
        else:
            cursor.execute(query)

        memories = []
        for row in cursor.fetchall():
            memories.append({
                'id': row['id'],
                'session_id': row['session_id'],
                'timestamp': row['timestamp'],
                'memory_type': row['memory_type'],
                'content': row['content'],
                'embedding': row['embedding'],  # Bytes
                'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                'importance_score': row['importance_score'],
                'is_fact': bool(row['is_fact']),
                'entity_type': row['entity_type']
            })

        return memories
