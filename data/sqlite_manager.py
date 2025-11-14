"""
SQLite Manager
--------------
Manages short-term session memory using SQLite.

Tables:
- recent_messages: Recent conversation messages
- session_summary: Session metadata and summaries
- pending_tasks: Active tasks and reminders
- context_embeddings: Cached embeddings for quick lookup
- reflections: Session and macro reflections
- meta_reflections: Aggregated weekly insights
- memories: Long-term persistent memories
- tasks: Structured task tracking system
- skills: Skill execution tracking
- notes: Personal notes storage
"""

import sqlite3
import asyncio
import threading
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime


class SQLiteManager:
    """
    Manages Hugo's short-term memory in SQLite.

    Fast, ephemeral storage for current session context.
    """

    def __init__(self, db_path: str = "data/memory/hugo_session.db", logger=None):
        """
        Initialize SQLite manager.

        Args:
            db_path: Path to SQLite database file
            logger: Optional HugoLogger instance for structured logging
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger

        # Per-thread connection pool
        self._connections: Dict[int, sqlite3.Connection] = {}
        self._conn_lock = threading.Lock()

        # Write serialization lock
        self.write_lock = threading.RLock()

        # Thread-safe write queue
        self.write_queue = asyncio.Queue()
        self._queue_running = False

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get or create a connection for the current thread.

        Returns:
            SQLite connection for this thread
        """
        tid = threading.get_ident()

        with self._conn_lock:
            if tid not in self._connections:
                conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                conn.row_factory = sqlite3.Row
                self._connections[tid] = conn

                if self.logger:
                    self.logger.log_event("sqlite", "new_thread_connection_created", {
                        "thread_id": tid,
                        "total_connections": len(self._connections)
                    })

        return self._connections[tid]

    async def connect(self):
        """Connect to SQLite database and create tables"""
        # Run in executor since sqlite3 is blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._connect_sync)

    def _connect_sync(self):
        """Synchronous connection (runs in executor)"""
        # Get connection for this thread
        conn = self._get_connection()
        self._create_tables()

    def _create_tables(self):
        """Create schema if it doesn't exist"""
        conn = self._get_connection()
        cursor = conn.cursor()

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

        # Tasks table (structured task tracking system)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL,
                owner TEXT,
                priority TEXT,
                tags TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Skills table (skill execution tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                args TEXT,
                result TEXT,
                executed_at TEXT NOT NULL
            )
        """)

        # Notes table (for notes skill)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
        """)

        # Agent actions table (autonomous agent tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                details TEXT,
                executed_at TEXT NOT NULL,
                success INTEGER NOT NULL
            )
        """)

        # Web monitor rules table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS web_monitor_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target TEXT NOT NULL,
                condition TEXT NOT NULL,
                threshold REAL NOT NULL,
                created_at TEXT NOT NULL
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_owner ON tasks(owner)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status_owner ON tasks(status, owner)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_skills_executed_at ON skills(executed_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_notes_created_at ON notes(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_actions_executed_at ON agent_actions(executed_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_actions_type ON agent_actions(action_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_web_monitor_rules_target ON web_monitor_rules(target)")

        conn.commit()

    async def drain_queue_loop(self):
        """
        Main thread drain loop for thread-safe SQLite writes.

        This loop processes all write operations from the write_queue,
        ensuring that all SQLite writes happen in a single thread.
        """
        self._queue_running = True

        if self.logger:
            self.logger.log_event("sqlite", "drain_queue_started", {
                "thread_safe": True
            })

        try:
            while self._queue_running:
                # Get next write job from queue
                job = await self.write_queue.get()

                if job is None:  # Shutdown signal
                    break

                operation_type = job.get("type")
                payload = job.get("payload")
                future = job.get("future")

                if self.logger:
                    self.logger.log_event("sqlite", "write_queued", {
                        "operation": operation_type
                    })

                try:
                    # Execute the appropriate synchronous write operation
                    # Uses per-thread connections with write lock for serialization
                    loop = asyncio.get_event_loop()

                    def execute_write():
                        if operation_type == "store_message":
                            self._store_message_sync(**payload)
                            return None
                        elif operation_type == "store_reflection":
                            return self._store_reflection_sync(**payload)
                        elif operation_type == "store_meta_reflection":
                            return self._store_meta_reflection_sync(**payload)
                        elif operation_type == "store_memory":
                            return self._store_memory_sync(**payload)
                        elif operation_type == "store_skill_run":
                            return self._store_skill_run_sync(**payload)
                        elif operation_type == "store_note":
                            return self._store_note_sync(**payload)
                        elif operation_type == "store_task":
                            return self._store_task_sync(**payload)
                        elif operation_type == "update_task":
                            self._update_task_sync(**payload)
                            return None
                        else:
                            raise ValueError(f"Unknown operation type: {operation_type}")

                    result = await loop.run_in_executor(None, execute_write)

                    if self.logger:
                        self.logger.log_event("sqlite", "write_completed", {
                            "operation": operation_type
                        })

                    # Resolve future with success
                    if future:
                        future.set_result(result)

                except Exception as e:
                    if self.logger:
                        self.logger.log_error(e, {
                            "phase": "drain_queue_loop",
                            "operation": operation_type
                        })

                    # Resolve future with exception
                    if future:
                        future.set_exception(e)

                finally:
                    self.write_queue.task_done()

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "drain_queue_loop_crashed"})
        finally:
            self._queue_running = False

    async def _queue_write(self, operation_type: str, payload: Dict[str, Any]) -> Any:
        """
        Queue a write operation and wait for it to complete.

        Args:
            operation_type: Type of write operation
            payload: Operation parameters

        Returns:
            Result from the write operation
        """
        # Create a future to wait for completion
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        # Package the job
        job = {
            "type": operation_type,
            "payload": payload,
            "future": future
        }

        # Queue the job
        await self.write_queue.put(job)

        # Wait for completion
        return await future

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

        await self._queue_write("store_message", {
            "session_id": session_id,
            "role": role,
            "content": content,
            "embedding": embedding,
            "metadata_json": json.dumps(metadata) if metadata else None,
            "importance": importance
        })

    def _store_message_sync(self, session_id, role, content, embedding, metadata_json, importance):
        """Synchronous message storage"""
        conn = self._get_connection()

        with self.write_lock:
            if self.logger:
                self.logger.log_event("sqlite", "write_locked", {"operation": "store_message"})

            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO recent_messages
                (session_id, timestamp, role, content, embedding, metadata, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (session_id, datetime.now().isoformat(), role, content,
                  embedding, metadata_json, importance))

            conn.commit()

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

        conn = self._get_connection()
        cursor = conn.cursor()
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
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute("""
            INSERT OR REPLACE INTO session_summary
            (session_id, started_at, last_activity, message_count, metadata)
            VALUES (?, ?, ?, 0, ?)
        """, (session_id, now, now, metadata_json))

        conn.commit()

    async def update_session(self, session_id: str, context_summary: Optional[str] = None,
                           mood: Optional[str] = None):
        """Update session summary"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._update_session_sync,
                                  session_id, context_summary, mood)

    def _update_session_sync(self, session_id, context_summary, mood):
        """Synchronous session update"""
        conn = self._get_connection()

        with self.write_lock:
            cursor = conn.cursor()

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

            conn.commit()

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
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM recent_messages WHERE timestamp < ?", (cutoff,))
        conn.commit()

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

        return await self._queue_write("store_reflection", {
            "session_id": session_id,
            "reflection_type": reflection_type,
            "summary": summary,
            "insights_json": json.dumps(insights) if insights else None,
            "patterns_json": json.dumps(patterns) if patterns else None,
            "improvements_json": json.dumps(improvements) if improvements else None,
            "sentiment": sentiment,
            "keywords_json": json.dumps(keywords) if keywords else None,
            "confidence": confidence,
            "embedding": embedding,
            "metadata_json": json.dumps(metadata) if metadata else None
        })

    def _store_reflection_sync(self, session_id, reflection_type, summary,
                               insights_json, patterns_json, improvements_json,
                               sentiment, keywords_json, confidence, embedding, metadata_json):
        """Synchronous reflection storage"""
        conn = self._get_connection()

        with self.write_lock:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO reflections
                (session_id, type, timestamp, summary, insights, patterns, improvements,
                 sentiment, keywords, confidence, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, reflection_type, datetime.now().isoformat(),
                  summary, insights_json, patterns_json, improvements_json,
                  sentiment, keywords_json, confidence, embedding, metadata_json))

            conn.commit()
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

        conn = self._get_connection()
        cursor = conn.cursor()

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

        return await self._queue_write("store_meta_reflection", {
            "summary": summary,
            "insights_json": json.dumps(insights) if insights else None,
            "patterns_json": json.dumps(patterns) if patterns else None,
            "improvements_json": json.dumps(improvements) if improvements else None,
            "reflections_analyzed": reflections_analyzed,
            "time_window_days": time_window_days,
            "confidence": confidence,
            "embedding": embedding,
            "metadata_json": json.dumps(metadata) if metadata else None
        })

    def _store_meta_reflection_sync(self, summary, insights_json, patterns_json,
                                    improvements_json, reflections_analyzed,
                                    time_window_days, confidence, embedding, metadata_json):
        """Synchronous meta-reflection storage"""
        conn = self._get_connection()

        with self.write_lock:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO meta_reflections
                (created_at, time_window_days, summary, insights, patterns, improvements,
                 reflections_analyzed, confidence, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(), time_window_days, summary,
                  insights_json, patterns_json, improvements_json,
                  reflections_analyzed, confidence, embedding, metadata_json))

            conn.commit()
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

        conn = self._get_connection()
        cursor = conn.cursor()
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

    # ==============================================
    # AGENT ACTION METHODS
    # ==============================================

    async def save_agent_action(self, action_type: str, details: str, executed_at: str, success: int) -> int:
        """
        Save an autonomous agent action.

        Args:
            action_type: Type of action
            details: Action details (JSON string or text)
            executed_at: ISO format timestamp
            success: 1 for success, 0 for failure

        Returns:
            ID of the stored action
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._save_agent_action_sync,
            action_type, details, executed_at, success
        )

    def _save_agent_action_sync(self, action_type, details, executed_at, success):
        """Synchronous agent action storage"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO agent_actions
            (action_type, details, executed_at, success)
            VALUES (?, ?, ?, ?)
        """, (action_type, details, executed_at, success))

        conn.commit()
        return cursor.lastrowid

    async def get_agent_actions(self, action_type: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve agent action history.

        Args:
            action_type: Optional filter by action type
            limit: Maximum number of records

        Returns:
            List of agent action dictionaries
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_agent_actions_sync, action_type, limit)

    def _get_agent_actions_sync(self, action_type, limit):
        """Synchronous agent action retrieval"""
        conn = self._get_connection()
        cursor = conn.cursor()

        if action_type:
            cursor.execute("""
                SELECT id, action_type, details, executed_at, success
                FROM agent_actions
                WHERE action_type = ?
                ORDER BY executed_at DESC
                LIMIT ?
            """, (action_type, limit))
        else:
            cursor.execute("""
                SELECT id, action_type, details, executed_at, success
                FROM agent_actions
                ORDER BY executed_at DESC
                LIMIT ?
            """, (limit,))

        actions = []
        for row in cursor.fetchall():
            actions.append({
                'id': row['id'],
                'action_type': row['action_type'],
                'details': row['details'],
                'executed_at': row['executed_at'],
                'success': bool(row['success'])
            })

        return actions

    # ==============================================
    # WEB MONITOR METHODS
    # ==============================================

    async def add_monitor_rule(self, target: str, condition: str, threshold: float, created_at: str) -> int:
        """
        Add a web monitor rule.

        Args:
            target: URL or endpoint to monitor
            condition: Condition (above, below, equals)
            threshold: Threshold value
            created_at: ISO format timestamp

        Returns:
            ID of the created rule
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._add_monitor_rule_sync,
            target, condition, threshold, created_at
        )

    def _add_monitor_rule_sync(self, target, condition, threshold, created_at):
        """Synchronous monitor rule creation"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO web_monitor_rules
            (target, condition, threshold, created_at)
            VALUES (?, ?, ?, ?)
        """, (target, condition, threshold, created_at))

        conn.commit()
        return cursor.lastrowid

    async def get_monitor_rules(self) -> List[Dict[str, Any]]:
        """
        Get all web monitor rules.

        Returns:
            List of monitor rule dictionaries
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_monitor_rules_sync)

    def _get_monitor_rules_sync(self):
        """Synchronous monitor rules retrieval"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, target, condition, threshold, created_at
            FROM web_monitor_rules
            ORDER BY created_at DESC
        """)

        rules = []
        for row in cursor.fetchall():
            rules.append({
                'id': row['id'],
                'target': row['target'],
                'condition': row['condition'],
                'threshold': float(row['threshold']),
                'created_at': row['created_at']
            })

        return rules

    async def remove_monitor_rule(self, rule_id: int):
        """
        Remove a web monitor rule.

        Args:
            rule_id: Rule ID to remove
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._remove_monitor_rule_sync, rule_id)

    def _remove_monitor_rule_sync(self, rule_id):
        """Synchronous monitor rule removal"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM web_monitor_rules
            WHERE id = ?
        """, (rule_id,))

        conn.commit()

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

        return await self._queue_write("store_memory", {
            "session_id": session_id,
            "memory_type": memory_type,
            "content": content,
            "embedding": embedding,
            "metadata_json": json.dumps(metadata) if metadata else None,
            "importance_score": importance_score,
            "is_fact": 1 if is_fact else 0,
            "entity_type": entity_type
        })

    def _store_memory_sync(self, session_id, memory_type, content, embedding,
                           metadata_json, importance_score, is_fact, entity_type):
        """Synchronous memory storage"""
        conn = self._get_connection()

        with self.write_lock:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO memories
                (session_id, timestamp, memory_type, content, embedding, metadata,
                 importance_score, is_fact, entity_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, datetime.now().isoformat(), memory_type, content,
                  embedding, metadata_json, importance_score, is_fact, entity_type))

            conn.commit()
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

        conn = self._get_connection()
        cursor = conn.cursor()

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

    async def get_memory_by_id(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch a single memory by ID.

        Args:
            memory_id: Memory ID to fetch

        Returns:
            Dictionary with memory data, or None if not found
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_memory_by_id_sync, memory_id)

    def _get_memory_by_id_sync(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Synchronous memory-by-id retrieval"""
        import json

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, session_id, timestamp, memory_type, content, embedding,
                   metadata, importance_score, is_fact, entity_type
            FROM memories
            WHERE id = ?
        """, (memory_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return {
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
        }

    async def delete_memory(self, memory_id: int) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if a row was deleted, False otherwise
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._delete_memory_sync, memory_id)

    def _delete_memory_sync(self, memory_id: int) -> bool:
        """Synchronous memory deletion"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        return cursor.rowcount > 0

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

        conn = self._get_connection()
        cursor = conn.cursor()

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

    # ==============================================
    # SKILL SYSTEM METHODS
    # ==============================================

    async def store_skill_run(self, name: str, args: str, result: str, executed_at: str) -> int:
        """
        Store a skill execution in the database.

        Args:
            name: Skill name
            args: JSON string of arguments
            result: JSON string of execution result
            executed_at: ISO format timestamp

        Returns:
            ID of the stored skill run
        """
        return await self._queue_write("store_skill_run", {
            "name": name,
            "args": args,
            "result": result,
            "executed_at": executed_at
        })

    def _store_skill_run_sync(self, name, args, result, executed_at):
        """Synchronous skill run storage"""
        conn = self._get_connection()

        with self.write_lock:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO skills
                (name, args, result, executed_at)
                VALUES (?, ?, ?, ?)
            """, (name, args, result, executed_at))

            conn.commit()
            return cursor.lastrowid

    async def get_skill_history(self, skill_name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve skill execution history.

        Args:
            skill_name: Optional filter by skill name
            limit: Maximum number of records

        Returns:
            List of skill execution dictionaries
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_skill_history_sync, skill_name, limit)

    def _get_skill_history_sync(self, skill_name, limit):
        """Synchronous skill history retrieval"""
        import json

        conn = self._get_connection()
        cursor = conn.cursor()

        if skill_name:
            cursor.execute("""
                SELECT id, name, args, result, executed_at
                FROM skills
                WHERE name = ?
                ORDER BY executed_at DESC
                LIMIT ?
            """, (skill_name, limit))
        else:
            cursor.execute("""
                SELECT id, name, args, result, executed_at
                FROM skills
                ORDER BY executed_at DESC
                LIMIT ?
            """, (limit,))

        history = []
        for row in cursor.fetchall():
            history.append({
                'id': row['id'],
                'name': row['name'],
                'args': json.loads(row['args']) if row['args'] else {},
                'result': json.loads(row['result']) if row['result'] else {},
                'executed_at': row['executed_at']
            })

        return history

    # ==============================================
    # NOTES SKILL METHODS
    # ==============================================

    async def store_note(self, content: str, created_at: str, metadata: Optional[str] = None) -> int:
        """
        Store a note in the database.

        Args:
            content: Note content
            created_at: ISO format timestamp
            metadata: Optional JSON string of metadata

        Returns:
            ID of the stored note
        """
        return await self._queue_write("store_note", {
            "content": content,
            "created_at": created_at,
            "metadata": metadata
        })

    def _store_note_sync(self, content, created_at, metadata):
        """Synchronous note storage"""
        conn = self._get_connection()

        with self.write_lock:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO notes
                (content, created_at, metadata)
                VALUES (?, ?, ?)
            """, (content, created_at, metadata))

            conn.commit()
            return cursor.lastrowid

    async def list_notes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List recent notes.

        Args:
            limit: Maximum number of notes to return

        Returns:
            List of note dictionaries
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._list_notes_sync, limit)

    def _list_notes_sync(self, limit):
        """Synchronous note listing"""
        import json

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, content, created_at, metadata
            FROM notes
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        notes = []
        for row in cursor.fetchall():
            notes.append({
                'id': row['id'],
                'content': row['content'],
                'created_at': row['created_at'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else {}
            })

        return notes

    async def search_notes(self, query: str) -> List[Dict[str, Any]]:
        """
        Search notes by keyword using SQLite LIKE.

        Args:
            query: Search query string

        Returns:
            List of matching note dictionaries
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._search_notes_sync, query)

    def _search_notes_sync(self, query):
        """Synchronous note search"""
        import json

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, content, created_at, metadata
            FROM notes
            WHERE content LIKE ?
            ORDER BY created_at DESC
        """, (f"%{query}%",))

        notes = []
        for row in cursor.fetchall():
            notes.append({
                'id': row['id'],
                'content': row['content'],
                'created_at': row['created_at'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else {}
            })

        return notes

    async def get_note(self, note_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific note by ID.

        Args:
            note_id: Note ID to retrieve

        Returns:
            Note dictionary or None if not found
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_note_sync, note_id)

    def _get_note_sync(self, note_id):
        """Synchronous note retrieval"""
        import json

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, content, created_at, metadata
            FROM notes
            WHERE id = ?
        """, (note_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return {
            'id': row['id'],
            'content': row['content'],
            'created_at': row['created_at'],
            'metadata': json.loads(row['metadata']) if row['metadata'] else {}
        }
