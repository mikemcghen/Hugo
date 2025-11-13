"""
Task Manager
------------
Manages Hugo's task tracking system with SQLite persistence.

Provides structured task management for tracking work items,
assignments, and progress across sessions.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class Task:
    """
    Represents a single task in Hugo's task system.

    Attributes:
        id: Unique task identifier (database primary key)
        title: Short task title
        description: Detailed task description
        status: Current task status (todo, in_progress, blocked, done)
        owner: Task owner (michael, hugo, claude-agent, etc.)
        priority: Task priority (low, medium, high)
        tags: List of tags for categorization
        created_at: Task creation timestamp
        updated_at: Last update timestamp
    """
    id: int
    title: str
    description: str
    status: str
    owner: str
    priority: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def tags_to_string(tags: List[str]) -> str:
        """Convert tags list to comma-separated string"""
        return ",".join(tags) if tags else ""

    @staticmethod
    def tags_from_string(tags_str: str) -> List[str]:
        """Convert comma-separated string to tags list"""
        if not tags_str or not tags_str.strip():
            return []
        return [tag.strip() for tag in tags_str.split(",") if tag.strip()]


class TaskManager:
    """
    Manages Hugo's task tracking system.

    Provides methods for creating, listing, updating, and managing tasks
    with SQLite persistence.
    """

    def __init__(self, sqlite_manager, logger):
        """
        Initialize TaskManager.

        Args:
            sqlite_manager: SQLiteManager instance for database operations
            logger: HugoLogger instance for event logging
        """
        self.sqlite = sqlite_manager
        self.logger = logger

    async def create_task(
        self,
        title: str,
        description: str,
        owner: str = "michael",
        priority: str = "medium",
        tags: Optional[List[str]] = None
    ) -> Task:
        """
        Create a new task.

        Args:
            title: Task title
            description: Task description
            owner: Task owner (default: michael)
            priority: Task priority (default: medium)
            tags: Optional list of tags

        Returns:
            Created Task object
        """
        if tags is None:
            tags = []

        now = datetime.now()
        tags_str = Task.tags_to_string(tags)

        loop = asyncio.get_event_loop()
        task_id = await loop.run_in_executor(
            None,
            self._create_task_sync,
            title,
            description,
            owner,
            priority,
            tags_str,
            now
        )

        task = Task(
            id=task_id,
            title=title,
            description=description,
            status="todo",
            owner=owner,
            priority=priority,
            tags=tags,
            created_at=now,
            updated_at=now
        )

        self.logger.log_event("tasks", "task_created", {
            "task_id": task_id,
            "title": title,
            "owner": owner,
            "priority": priority
        })

        return task

    def _create_task_sync(
        self,
        title: str,
        description: str,
        owner: str,
        priority: str,
        tags_str: str,
        now: datetime
    ) -> int:
        """Synchronous task creation (runs in executor)"""
        cursor = self.sqlite.conn.cursor()
        cursor.execute("""
            INSERT INTO tasks (title, description, status, owner, priority, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            title,
            description,
            "todo",
            owner,
            priority,
            tags_str,
            now.isoformat(),
            now.isoformat()
        ))
        self.sqlite.conn.commit()
        return cursor.lastrowid

    async def list_tasks(
        self,
        status: Optional[str] = None,
        owner: Optional[str] = None
    ) -> List[Task]:
        """
        List tasks with optional filtering.

        Args:
            status: Filter by status (optional)
            owner: Filter by owner (optional)

        Returns:
            List of Task objects
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._list_tasks_sync,
            status,
            owner
        )

    def _list_tasks_sync(
        self,
        status: Optional[str],
        owner: Optional[str]
    ) -> List[Task]:
        """Synchronous task listing (runs in executor)"""
        cursor = self.sqlite.conn.cursor()

        query = "SELECT * FROM tasks WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if owner:
            query += " AND owner = ?"
            params.append(owner)

        query += " ORDER BY created_at DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        tasks = []
        for row in rows:
            task = Task(
                id=row['id'],
                title=row['title'],
                description=row['description'],
                status=row['status'],
                owner=row['owner'],
                priority=row['priority'],
                tags=Task.tags_from_string(row['tags']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at'])
            )
            tasks.append(task)

        return tasks

    async def get_task(self, task_id: int) -> Optional[Task]:
        """
        Get a specific task by ID.

        Args:
            task_id: Task ID to fetch

        Returns:
            Task object if found, None otherwise
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_task_sync, task_id)

    def _get_task_sync(self, task_id: int) -> Optional[Task]:
        """Synchronous task retrieval (runs in executor)"""
        cursor = self.sqlite.conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return Task(
            id=row['id'],
            title=row['title'],
            description=row['description'],
            status=row['status'],
            owner=row['owner'],
            priority=row['priority'],
            tags=Task.tags_from_string(row['tags']),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at'])
        )

    async def update_task_status(self, task_id: int, status: str) -> bool:
        """
        Update task status.

        Args:
            task_id: Task ID to update
            status: New status

        Returns:
            True if updated successfully, False otherwise
        """
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            self._update_task_status_sync,
            task_id,
            status
        )

        if success:
            self.logger.log_event("tasks", "task_status_updated", {
                "task_id": task_id,
                "status": status
            })

        return success

    def _update_task_status_sync(self, task_id: int, status: str) -> bool:
        """Synchronous status update (runs in executor)"""
        cursor = self.sqlite.conn.cursor()
        cursor.execute("""
            UPDATE tasks
            SET status = ?, updated_at = ?
            WHERE id = ?
        """, (status, datetime.now().isoformat(), task_id))
        self.sqlite.conn.commit()
        return cursor.rowcount > 0

    async def assign_task(self, task_id: int, owner: str) -> bool:
        """
        Assign task to a new owner.

        Args:
            task_id: Task ID to reassign
            owner: New owner

        Returns:
            True if assigned successfully, False otherwise
        """
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            self._assign_task_sync,
            task_id,
            owner
        )

        if success:
            self.logger.log_event("tasks", "task_assigned", {
                "task_id": task_id,
                "owner": owner
            })

        return success

    def _assign_task_sync(self, task_id: int, owner: str) -> bool:
        """Synchronous task assignment (runs in executor)"""
        cursor = self.sqlite.conn.cursor()
        cursor.execute("""
            UPDATE tasks
            SET owner = ?, updated_at = ?
            WHERE id = ?
        """, (owner, datetime.now().isoformat(), task_id))
        self.sqlite.conn.commit()
        return cursor.rowcount > 0

    async def update_task(
        self,
        task_id: int,
        **fields
    ) -> bool:
        """
        Update arbitrary task fields.

        Args:
            task_id: Task ID to update
            **fields: Fields to update (title, description, priority, tags, etc.)

        Returns:
            True if updated successfully, False otherwise
        """
        if not fields:
            return False

        # Convert tags list to string if present
        if 'tags' in fields and isinstance(fields['tags'], list):
            fields['tags'] = Task.tags_to_string(fields['tags'])

        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            self._update_task_sync,
            task_id,
            fields
        )

        if success:
            self.logger.log_event("tasks", "task_updated", {
                "task_id": task_id,
                "fields": list(fields.keys())
            })

        return success

    def _update_task_sync(self, task_id: int, fields: Dict[str, Any]) -> bool:
        """Synchronous task update (runs in executor)"""
        # Build UPDATE query dynamically
        set_clauses = []
        params = []

        for key, value in fields.items():
            set_clauses.append(f"{key} = ?")
            params.append(value)

        set_clauses.append("updated_at = ?")
        params.append(datetime.now().isoformat())

        params.append(task_id)

        query = f"UPDATE tasks SET {', '.join(set_clauses)} WHERE id = ?"

        cursor = self.sqlite.conn.cursor()
        cursor.execute(query, params)
        self.sqlite.conn.commit()
        return cursor.rowcount > 0
