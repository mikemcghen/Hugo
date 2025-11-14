"""
Hugo Core Modules
-----------------
Central reasoning, memory, and reflection systems for Hugo AI assistant.
"""

__version__ = "0.1.0"
__codename__ = "The Right Hand"

from .cognition import CognitionEngine
from .memory import MemoryManager
from .reflection import ReflectionEngine
from .scheduler import MaintenanceScheduler
from .logger import HugoLogger
from .runtime_manager import RuntimeManager

__all__ = [
    "CognitionEngine",
    "MemoryManager",
    "ReflectionEngine",
    "MaintenanceScheduler",
    "HugoLogger",
    "RuntimeManager",
]
