"""
Action Result
-------------
Standardized envelope for all action outputs in Hugo.

Every executor, handler, and helper returns an ActionResult.
This replaces the inconsistent SkillResult shapes from the old skill handlers.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ActionResult:
    """
    Standardized result from any action in Hugo's action layer.

    Compatible with the existing inject_skill_block() mechanism via to_skill_block().
    """
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: int = field(default_factory=lambda: int(time.time()))
    domain: str = "unknown"
    action: str = "unknown"
    permission_level: str = "auto_execute"

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time": round(self.execution_time, 3),
            "timestamp": self.timestamp,
            "domain": self.domain,
            "action": self.action,
            "permission_level": self.permission_level,
        }

    def to_skill_block(self) -> str:
        """
        Produce a [SkillResult] or [SkillError] block compatible with
        the existing inject_skill_block() mechanism in core/skills/prompt_injection.py.
        """
        if self.success:
            output = str(self.data) if self.data is not None else ""
            return (
                f"[SkillResult]\n"
                f"skill: {self.domain}\n"
                f"action: {self.action}\n"
                f"output: \"{output}\""
            )
        else:
            return (
                f"[SkillError]\n"
                f"skill: {self.domain}\n"
                f"action: {self.action}\n"
                f"error: \"{self.error or 'Unknown error'}\""
            )

    def __repr__(self) -> str:
        if self.success:
            return f"ActionResult(success=True, domain={self.domain!r}, action={self.action!r}, data={self.data!r})"
        return f"ActionResult(success=False, domain={self.domain!r}, action={self.action!r}, error={self.error!r})"
