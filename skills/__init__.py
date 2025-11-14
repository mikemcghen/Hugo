"""
Skills Subsystem
----------------
Dynamic skill loading and execution system for Hugo.

Skills are YAML-defined capabilities that extend Hugo's functionality:
- File operations
- Note-taking and search
- System commands
- Custom tools and integrations
"""

__version__ = "0.1.0"

from .base_skill import BaseSkill, SkillResult
from .skill_manager import SkillManager
from .skill_registry import SkillRegistry

__all__ = [
    "BaseSkill",
    "SkillResult",
    "SkillManager",
    "SkillRegistry",
]
