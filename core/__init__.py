"""
Hugo Core Modules
-----------------
Central reasoning, memory, and reflection systems for Hugo AI assistant.
"""

__version__ = "0.1.0"
__codename__ = "The Right Hand"

# Lazy imports to avoid forcing all dependencies
__all__ = [
    "CognitionEngine",
    "MemoryManager",
    "ReflectionEngine",
    "MaintenanceScheduler",
    "HugoLogger",
    "RuntimeManager",
]

def __getattr__(name):
    """Lazy import of core modules"""
    if name == "CognitionEngine":
        from .cognition import CognitionEngine
        return CognitionEngine
    elif name == "MemoryManager":
        from .memory import MemoryManager
        return MemoryManager
    elif name == "ReflectionEngine":
        from .reflection import ReflectionEngine
        return ReflectionEngine
    elif name == "MaintenanceScheduler":
        from .scheduler import MaintenanceScheduler
        return MaintenanceScheduler
    elif name == "HugoLogger":
        from .logger import HugoLogger
        return HugoLogger
    elif name == "RuntimeManager":
        from .runtime_manager import RuntimeManager
        return RuntimeManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
