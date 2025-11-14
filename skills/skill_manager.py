"""
Skill Manager
-------------
Manages loading, execution, and lifecycle of skills.

Responsibilities:
- Scan for YAML skill definitions
- Dynamically load Python skill classes
- Execute skills with argument validation
- Track skill runs in SQLite
- Integrate with memory system
"""

import os
import yaml
import importlib
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from .base_skill import BaseSkill, SkillResult
from .skill_registry import SkillRegistry


class SkillManager:
    """
    Central manager for Hugo's skill system.

    Loads skills from YAML definitions and manages their execution.
    """

    def __init__(self, logger, sqlite_manager=None, memory_manager=None):
        """
        Initialize the skill manager.

        Args:
            logger: HugoLogger instance
            sqlite_manager: SQLiteManager for persistence
            memory_manager: MemoryManager for storing skill results
        """
        self.logger = logger
        self.sqlite = sqlite_manager
        self.memory = memory_manager
        self.registry = SkillRegistry()

        # Skill directory
        self.skills_dir = Path(__file__).parent

        # Track loaded skills
        self.loaded_skills: Dict[str, Dict[str, Any]] = {}

    def load_skills(self):
        """
        Scan skills directory and load all YAML-defined skills.

        This method:
        1. Finds all .yaml files in skills/
        2. Parses skill definitions
        3. Dynamically imports Python classes
        4. Instantiates and registers skills
        """
        self.logger.log_event("skill_manager", "loading_skills", {
            "skills_dir": str(self.skills_dir)
        })

        loaded_count = 0
        error_count = 0

        # Find all YAML files recursively
        for yaml_path in self.skills_dir.rglob("*.yaml"):
            try:
                # Skip if in __pycache__ or hidden directories
                if any(part.startswith('.') or part == '__pycache__' for part in yaml_path.parts):
                    continue

                # Load YAML definition
                with open(yaml_path, 'r') as f:
                    skill_def = yaml.safe_load(f)

                if not skill_def:
                    continue

                skill_name = skill_def.get('name')
                if not skill_name:
                    self.logger.log_event("skill_manager", "invalid_definition", {
                        "path": str(yaml_path),
                        "error": "missing_name"
                    })
                    continue

                # Load the skill
                success = self._load_skill_from_definition(skill_def, yaml_path)

                if success:
                    loaded_count += 1
                else:
                    error_count += 1

            except Exception as e:
                error_count += 1
                self.logger.log_error(e, {
                    "phase": "skill_loading",
                    "path": str(yaml_path)
                })

        self.logger.log_event("skill_manager", "skills_loaded", {
            "loaded": loaded_count,
            "errors": error_count,
            "total_skills": self.registry.count()
        })

    def _load_skill_from_definition(self, skill_def: Dict[str, Any], yaml_path: Path) -> bool:
        """
        Load a single skill from its YAML definition.

        Args:
            skill_def: Parsed YAML definition
            yaml_path: Path to the YAML file

        Returns:
            True if loaded successfully, False otherwise
        """
        skill_name = skill_def.get('name')
        entrypoint = skill_def.get('entrypoint')

        if not entrypoint:
            self.logger.log_event("skill_manager", "invalid_definition", {
                "skill": skill_name,
                "error": "missing_entrypoint"
            })
            return False

        try:
            # Parse entrypoint (e.g., "skills.example.notes_skill:NotesSkill")
            module_path, class_name = entrypoint.split(':')

            # Dynamically import the module
            module = importlib.import_module(module_path)

            # Get the skill class
            skill_class = getattr(module, class_name)

            # Instantiate the skill
            skill_instance = skill_class(
                logger=self.logger,
                sqlite_manager=self.sqlite,
                memory_manager=self.memory
            )

            # Set metadata from YAML
            skill_instance.name = skill_name
            skill_instance.description = skill_def.get('description', 'No description')
            skill_instance.version = skill_def.get('version', '1.0.0')
            skill_instance.trigger = skill_def.get('trigger', 'manual')
            skill_instance.permissions = skill_def.get('permissions', [])

            # Register the skill
            self.registry.register(skill_name, skill_instance)

            # Store definition
            self.loaded_skills[skill_name] = {
                'definition': skill_def,
                'yaml_path': str(yaml_path),
                'loaded_at': datetime.now().isoformat()
            }

            self.logger.log_event("skill_manager", "skill_loaded", {
                "skill": skill_name,
                "entrypoint": entrypoint
            })

            return True

        except Exception as e:
            self.logger.log_error(e, {
                "phase": "skill_instantiation",
                "skill": skill_name,
                "entrypoint": entrypoint
            })
            return False

    def reload_skills(self):
        """
        Reload all skills from disk.

        This clears the registry and reloads all skill definitions.
        """
        self.logger.log_event("skill_manager", "reloading_skills", {})

        # Clear registry
        self.registry.clear()
        self.loaded_skills.clear()

        # Reload
        self.load_skills()

    def list_skills(self) -> List[Dict[str, Any]]:
        """
        Get list of all loaded skills with metadata.

        Returns:
            List of skill info dictionaries
        """
        skills = []

        for name in self.registry.list_names():
            skill = self.registry.get(name)
            info = self.loaded_skills.get(name, {})

            skills.append({
                'name': name,
                'description': skill.description if skill else 'N/A',
                'version': skill.version if skill else 'N/A',
                'trigger': getattr(skill, 'trigger', 'manual'),
                'permissions': getattr(skill, 'permissions', []),
                'loaded_at': info.get('loaded_at'),
                'stats': skill.get_stats() if skill else {}
            })

        return skills

    async def run_skill(self, name: str, **kwargs) -> SkillResult:
        """
        Execute a skill by name with given arguments.

        Args:
            name: Skill identifier
            **kwargs: Arguments to pass to the skill

        Returns:
            SkillResult with execution outcome
        """
        # Get skill from registry
        skill = self.registry.get(name)

        if not skill:
            return SkillResult(
                success=False,
                output=None,
                message=f"Skill '{name}' not found"
            )

        self.logger.log_event("skill_manager", "run_skill", {
            "skill": name,
            "args": list(kwargs.keys())
        })

        # Execute the skill
        result = await skill.execute(**kwargs)

        # Store result in SQLite if available
        if self.sqlite:
            await self._store_skill_run(name, kwargs, result)

        # Store result in memory if configured
        if result.success and result.metadata.get('save_to_memory'):
            await self._save_result_to_memory(name, result)

        return result

    async def _store_skill_run(self, skill_name: str, args: Dict[str, Any], result: SkillResult):
        """
        Store skill execution in SQLite for tracking.

        Args:
            skill_name: Name of the skill
            args: Arguments passed to skill
            result: Execution result
        """
        try:
            import json

            await self.sqlite.store_skill_run(
                name=skill_name,
                args=json.dumps(args),
                result=json.dumps(result.to_dict()),
                executed_at=datetime.now().isoformat()
            )

            self.logger.log_event("skill_manager", "run_stored", {
                "skill": skill_name,
                "success": result.success
            })

        except Exception as e:
            self.logger.log_error(e, {"phase": "store_skill_run"})

    async def _save_result_to_memory(self, skill_name: str, result: SkillResult):
        """
        Save skill result to memory system.

        Args:
            skill_name: Name of the skill
            result: Execution result
        """
        try:
            from core.memory import MemoryEntry

            # Create memory entry for skill result
            memory_entry = MemoryEntry(
                id=None,
                session_id="system",
                timestamp=datetime.now(),
                memory_type="note",  # Skill results stored as notes
                content=f"[Skill: {skill_name}] {result.message}",
                embedding=None,
                metadata={
                    "skill_name": skill_name,
                    "skill_output": result.output,
                    "skill_metadata": result.metadata
                },
                importance_score=0.7,
                is_fact=False
            )

            await self.memory.store(memory_entry, persist_long_term=True)

            self.logger.log_event("skill_manager", "result_saved_to_memory", {
                "skill": skill_name
            })

        except Exception as e:
            self.logger.log_error(e, {"phase": "save_result_to_memory"})

    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """
        Get a skill instance by name.

        Args:
            name: Skill identifier

        Returns:
            BaseSkill instance or None
        """
        return self.registry.get(name)

    async def auto_trigger(self, memory_entry) -> Optional[SkillResult]:
        """
        Auto-trigger skills based on memory classification.

        Checks if a memory entry should trigger a skill and executes it.

        Args:
            memory_entry: MemoryEntry that may trigger a skill

        Returns:
            SkillResult if a skill was triggered, None otherwise
        """
        # Check if memory has skill trigger metadata
        skill_trigger = memory_entry.metadata.get('skill_trigger')

        if not skill_trigger:
            return None

        # Get the skill
        skill = self.registry.get(skill_trigger)

        if not skill:
            self.logger.log_event("skill_manager", "auto_trigger_failed", {
                "skill": skill_trigger,
                "reason": "not_found"
            })
            return None

        # Check if skill supports auto-triggering
        if getattr(skill, 'trigger', 'manual') != 'auto':
            return None

        self.logger.log_event("skill_manager", "auto_trigger", {
            "skill": skill_trigger,
            "memory_type": memory_entry.memory_type
        })

        # Execute skill with memory content as input
        result = await self.run_skill(skill_trigger, content=memory_entry.content)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the skill system.

        Returns:
            Dictionary with skill system stats
        """
        skills = self.registry.all()

        total_runs = sum(skill._runs for skill in skills.values())
        total_errors = sum(skill._errors for skill in skills.values())

        return {
            "total_skills": len(skills),
            "loaded_skills": list(skills.keys()),
            "total_runs": total_runs,
            "total_errors": total_errors,
            "success_rate": (total_runs - total_errors) / total_runs if total_runs > 0 else 0.0
        }
