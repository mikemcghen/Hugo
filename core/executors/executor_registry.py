"""
Executor Registry
-----------------
Dynamic registration and lookup of executor types.

Pattern adapted from Server-Files agents/registry.py.

Executors register themselves with @ExecutorRegistry.register on module load.
ActionRouter uses this to find the right executor for a domain.
"""

from typing import Dict, Type, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseExecutor


class ExecutorRegistry:
    """Registry for executor types."""

    _executors: Dict[str, Type] = {}

    @classmethod
    def register(cls, executor_class: Type) -> Type:
        """
        Register an executor class.

        Usage as decorator:
            @ExecutorRegistry.register
            class MyExecutor(BaseExecutor):
                name = "my_executor"
                ...
        """
        cls._executors[executor_class.name] = executor_class
        return executor_class

    @classmethod
    def get_executor(cls, domain: str) -> Optional["BaseExecutor"]:
        """
        Get a fresh executor instance for the given domain name.

        Args:
            domain: Executor name (e.g., 'ssh', 'docker', 'monitor')

        Returns:
            Instantiated executor or None if not found
        """
        executor_class = cls._executors.get(domain)
        if executor_class:
            return executor_class()
        return None

    @classmethod
    def list_executors(cls) -> List[dict]:
        """List all registered executor types."""
        return [
            {"name": ec.name, "description": ec.description}
            for ec in cls._executors.values()
        ]

    @classmethod
    def is_registered(cls, domain: str) -> bool:
        return domain in cls._executors

    @classmethod
    def clear(cls):
        """Clear all registrations (for testing)."""
        cls._executors.clear()
