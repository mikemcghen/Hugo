"""
Notes Skill
-----------
Personal note-taking skill for Hugo.

Capabilities:
- Add new notes
- List all notes
- Search notes by keyword
- Get specific note by ID
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from skills.base_skill import BaseSkill, SkillResult


class NotesSkill(BaseSkill):
    """
    Note-taking and management skill.

    Stores notes in SQLite and provides search/retrieval functionality.
    """

    def __init__(self, logger=None, sqlite_manager=None, memory_manager=None):
        super().__init__(logger, sqlite_manager, memory_manager)

        self.name = "notes"
        self.description = "Create, search, and list personal notes"
        self.version = "1.0.0"

    async def run(self, action: str = "help", **kwargs) -> SkillResult:
        """
        Execute notes skill with specified action.

        Args:
            action: Action to perform (add, list, search, get, help)
            **kwargs: Action-specific arguments

        Returns:
            SkillResult with action outcome
        """
        # Route to appropriate action handler
        if action == "add":
            return await self._add_note(**kwargs)
        elif action == "list":
            return await self._list_notes(**kwargs)
        elif action == "search":
            return await self._search_notes(**kwargs)
        elif action == "get":
            return await self._get_note(**kwargs)
        elif action == "help":
            return self._help()
        else:
            return SkillResult(
                success=False,
                output=None,
                message=f"Unknown action: {action}. Use 'help' to see available actions."
            )

    async def _add_note(self, content: str = None, **kwargs) -> SkillResult:
        """
        Add a new note.

        Args:
            content: Note content

        Returns:
            SkillResult with note ID
        """
        if not content:
            return SkillResult(
                success=False,
                output=None,
                message="Missing required argument: content"
            )

        if not self.sqlite:
            return SkillResult(
                success=False,
                output=None,
                message="SQLite manager not available"
            )

        try:
            # Store note in SQLite
            note_id = await self.sqlite.store_note(
                content=content,
                created_at=datetime.now().isoformat(),
                metadata=json.dumps(kwargs)
            )

            return SkillResult(
                success=True,
                output={"note_id": note_id, "content": content},
                message=f"Note {note_id} created successfully",
                metadata={"save_to_memory": True}
            )

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "add_note"})

            return SkillResult(
                success=False,
                output=None,
                message=f"Failed to add note: {str(e)}"
            )

    async def _list_notes(self, limit: int = 10, **kwargs) -> SkillResult:
        """
        List recent notes.

        Args:
            limit: Maximum number of notes to return

        Returns:
            SkillResult with list of notes
        """
        if not self.sqlite:
            return SkillResult(
                success=False,
                output=None,
                message="SQLite manager not available"
            )

        try:
            # Get notes from SQLite
            notes = await self.sqlite.list_notes(limit=limit)

            if not notes:
                return SkillResult(
                    success=True,
                    output=[],
                    message="No notes found"
                )

            return SkillResult(
                success=True,
                output=notes,
                message=f"Found {len(notes)} notes",
                metadata={"count": len(notes)}
            )

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "list_notes"})

            return SkillResult(
                success=False,
                output=None,
                message=f"Failed to list notes: {str(e)}"
            )

    async def _search_notes(self, query: str = None, **kwargs) -> SkillResult:
        """
        Search notes by keyword.

        Args:
            query: Search query

        Returns:
            SkillResult with matching notes
        """
        if not query:
            return SkillResult(
                success=False,
                output=None,
                message="Missing required argument: query"
            )

        if not self.sqlite:
            return SkillResult(
                success=False,
                output=None,
                message="SQLite manager not available"
            )

        try:
            # Search notes in SQLite
            notes = await self.sqlite.search_notes(query=query)

            if not notes:
                return SkillResult(
                    success=True,
                    output=[],
                    message=f"No notes found matching '{query}'"
                )

            return SkillResult(
                success=True,
                output=notes,
                message=f"Found {len(notes)} notes matching '{query}'",
                metadata={"count": len(notes), "query": query}
            )

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "search_notes"})

            return SkillResult(
                success=False,
                output=None,
                message=f"Failed to search notes: {str(e)}"
            )

    async def _get_note(self, note_id: int = None, **kwargs) -> SkillResult:
        """
        Get a specific note by ID.

        Args:
            note_id: Note ID to retrieve

        Returns:
            SkillResult with note details
        """
        if note_id is None:
            return SkillResult(
                success=False,
                output=None,
                message="Missing required argument: note_id"
            )

        if not self.sqlite:
            return SkillResult(
                success=False,
                output=None,
                message="SQLite manager not available"
            )

        try:
            # Get note from SQLite
            note = await self.sqlite.get_note(note_id=note_id)

            if not note:
                return SkillResult(
                    success=False,
                    output=None,
                    message=f"Note {note_id} not found"
                )

            return SkillResult(
                success=True,
                output=note,
                message=f"Retrieved note {note_id}",
                metadata={"note_id": note_id}
            )

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "get_note"})

            return SkillResult(
                success=False,
                output=None,
                message=f"Failed to get note: {str(e)}"
            )

    def _help(self) -> SkillResult:
        """
        Return help information for notes skill.

        Returns:
            SkillResult with usage information
        """
        help_text = """
Notes Skill Usage:
==================

Actions:
  add       Add a new note
            Args: content (required)
            Example: /skill run notes add content="Remember to buy milk"

  list      List recent notes
            Args: limit (optional, default=10)
            Example: /skill run notes list limit=20

  search    Search notes by keyword
            Args: query (required)
            Example: /skill run notes search query="milk"

  get       Get a specific note by ID
            Args: note_id (required)
            Example: /skill run notes get note_id=5

  help      Show this help message
            Example: /skill run notes help
"""

        return SkillResult(
            success=True,
            output=help_text.strip(),
            message="Notes skill help"
        )

    def requires_permissions(self) -> List[str]:
        """Return required permissions"""
        return ["filesystem_read", "filesystem_write", "database_write"]
