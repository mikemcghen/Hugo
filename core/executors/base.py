"""
Base Executor
-------------
All executors inherit from this. Provides safety checks, error handling,
and standardized result format.

Ported from Server-Files executors/base.py with async compatibility added.
"""

import time
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class ExecutorResult:
    """Standardized result from executor actions"""

    def __init__(self, success: bool, data: Any = None,
                 error: str = None, execution_time: float = 0.0):
        self.success = success
        self.data = data
        self.error = error
        self.execution_time = execution_time
        self.timestamp = int(time.time())

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        if self.success:
            return f"ExecutorResult(success=True, data={self.data!r})"
        return f"ExecutorResult(success=False, error={self.error!r})"


class BaseExecutor(ABC):
    """
    Base class for all Hugo executors.

    Executors are single-purpose workers that do ONE thing well.
    They don't make decisions — they execute and report.
    """

    name = "base"
    description = "Base executor"

    # Dangerous shell patterns blocked unconditionally
    FORBIDDEN_PATTERNS = [
        "rm -rf /",
        "rm -rf /*",
        "mkfs.",
        "dd if=",
        "> /dev/sd",
        "> /dev/nvme",
        "chmod -R 777 /",
        ":(){ :|:& };:",   # Fork bomb
        "mv / ",
        "wget | sh",
        "curl | sh",
        "curl -s | bash",
        "wget -O- | bash",
        "| sh",            # Pipe to sh (any form)
        "| bash",          # Pipe to bash (any form)
        "|sh",
        "|bash",
    ]

    def __init__(self):
        self.last_execution: Optional[dict] = None
        self.execution_count: int = 0

    def is_safe(self, command: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a command is safe to execute.

        Args:
            command: Shell command string to check

        Returns:
            (is_safe, reason_if_unsafe)
        """
        command_lower = command.lower()
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern.lower() in command_lower:
                return False, f"Blocked dangerous pattern: {pattern}"
        return True, None

    def execute(self, action: str, **params) -> ExecutorResult:
        """
        Execute an action synchronously.

        Subclasses should implement _execute() instead.
        """
        start_time = time.time()
        try:
            result = self._execute(action, **params)
            self.execution_count += 1
            self.last_execution = {
                "action": action,
                "params": params,
                "result": result.to_dict(),
                "timestamp": int(time.time()),
            }
            result.execution_time = time.time() - start_time
            return result
        except Exception as e:
            return ExecutorResult(
                success=False,
                error=f"Executor error: {str(e)}",
                execution_time=time.time() - start_time,
            )

    async def execute_async(self, action: str, **params) -> ExecutorResult:
        """
        Execute an action asynchronously via thread pool.

        Hugo's cognition pipeline is async, so executors that do I/O
        (SSH, Docker) are wrapped here to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.execute(action, **params))

    @abstractmethod
    def _execute(self, action: str, **params) -> ExecutorResult:
        """Implement the actual execution logic in subclasses."""
        pass

    def get_capabilities(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "actions": self._get_actions(),
        }

    def _get_actions(self) -> list:
        return []
