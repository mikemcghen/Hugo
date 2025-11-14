"""
Base Skill Class
----------------
Abstract base class for all Hugo skills.

All skills must inherit from BaseSkill and implement the run() method.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod


@dataclass
class SkillResult:
    """
    Result of a skill execution.
    
    Contains the output, metadata, and status of a skill run.
    """
    success: bool
    output: Any
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for storage"""
        return {
            "success": self.success,
            "output": self.output,
            "message": self.message,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "execution_time_ms": self.execution_time_ms
        }


class BaseSkill(ABC):
    """
    Abstract base class for all skills.
    
    Skills are dynamically loaded capabilities that extend Hugo's functionality.
    Each skill must define:
    - name: Unique identifier
    - description: Human-readable description
    - run(): Async method that executes the skill
    """
    
    def __init__(self, logger=None, sqlite_manager=None, memory_manager=None):
        """
        Initialize the skill with optional dependencies.
        
        Args:
            logger: HugoLogger instance
            sqlite_manager: SQLiteManager instance
            memory_manager: MemoryManager instance
        """
        self.logger = logger
        self.sqlite = sqlite_manager
        self.memory = memory_manager
        
        # Skill metadata
        self.name = self.__class__.__name__.replace("Skill", "").lower()
        self.description = "No description provided"
        self.version = "1.0.0"
        
        # Execution tracking
        self._runs = 0
        self._errors = 0
        self._last_run = None
    
    @abstractmethod
    async def run(self, **kwargs) -> SkillResult:
        """
        Execute the skill with given arguments.
        
        This method must be implemented by all skill subclasses.
        
        Args:
            **kwargs: Skill-specific arguments
            
        Returns:
            SkillResult containing execution outcome
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement run()")
    
    def requires_permissions(self) -> List[str]:
        """
        Define permissions required by this skill.
        
        Returns:
            List of permission identifiers (e.g., 'filesystem_read')
        """
        return []
    
    def validate_args(self, required_args: List[str], kwargs: Dict[str, Any]) -> bool:
        """
        Validate that required arguments are present.
        
        Args:
            required_args: List of required argument names
            kwargs: Arguments provided to the skill
            
        Returns:
            True if all required args present, False otherwise
        """
        missing = [arg for arg in required_args if arg not in kwargs]
        if missing:
            if self.logger:
                self.logger.log_event("skill", "validation_failed", {
                    "skill": self.name,
                    "missing_args": missing
                })
            return False
        return True
    
    async def before_run(self, **kwargs):
        """
        Lifecycle hook called before run().
        
        Override to add pre-execution logic.
        """
        self._runs += 1
        self._last_run = datetime.now()
        
        if self.logger:
            self.logger.log_event("skill", "started", {
                "skill": self.name,
                "args": list(kwargs.keys())
            })
    
    async def after_run(self, result: SkillResult):
        """
        Lifecycle hook called after run().
        
        Override to add post-execution logic.
        
        Args:
            result: The SkillResult from run()
        """
        if not result.success:
            self._errors += 1
        
        if self.logger:
            self.logger.log_event("skill", "completed", {
                "skill": self.name,
                "success": result.success,
                "execution_time_ms": result.execution_time_ms
            })
    
    async def on_error(self, error: Exception) -> SkillResult:
        """
        Lifecycle hook called when run() raises an exception.
        
        Args:
            error: The exception that was raised
            
        Returns:
            SkillResult indicating failure
        """
        self._errors += 1
        
        if self.logger:
            self.logger.log_error(error, {
                "skill": self.name,
                "phase": "skill_execution"
            })
        
        return SkillResult(
            success=False,
            output=None,
            message=f"Skill error: {str(error)}",
            metadata={"error_type": type(error).__name__}
        )
    
    async def execute(self, **kwargs) -> SkillResult:
        """
        Execute the skill with full lifecycle hooks.
        
        This is the main entry point for running a skill.
        Handles before/after hooks and error handling.
        
        Args:
            **kwargs: Skill arguments
            
        Returns:
            SkillResult with execution outcome
        """
        start_time = datetime.now()
        
        try:
            # Before hook
            await self.before_run(**kwargs)
            
            # Execute skill
            result = await self.run(**kwargs)
            
            # Calculate execution time
            end_time = datetime.now()
            result.execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # After hook
            await self.after_run(result)
            
            return result
            
        except Exception as e:
            # Error hook
            result = await self.on_error(e)
            
            # Calculate execution time
            end_time = datetime.now()
            result.execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics for this skill.
        
        Returns:
            Dictionary with runs, errors, and last run time
        """
        return {
            "name": self.name,
            "runs": self._runs,
            "errors": self._errors,
            "success_rate": (self._runs - self._errors) / self._runs if self._runs > 0 else 0.0,
            "last_run": self._last_run.isoformat() if self._last_run else None
        }
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', version='{self.version}')>"
