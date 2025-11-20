"""
Core Skills Module
------------------
Minimal skill routing system for CORE mode.

Components:
- trigger_detector: Detects skill commands at start of text
- router: Routes skills to appropriate handlers
- handlers: Execute skill logic (deterministic, no I/O)
- schemas: Format skill results and errors
- prompt_injection: Inject skill blocks into prompts
"""

from core.skills.trigger_detector import detect_skill, SkillTrigger
from core.skills.router import route_skill, validate_skill_name, is_allowed_skill, SkillBlock
from core.skills.handlers import (
    handle_search,
    handle_note,
    handle_fetch,
    handle_scrape,
    SkillResult
)
from core.skills.schemas import format_skill_result, format_skill_error
from core.skills.prompt_injection import inject_skill_block

__all__ = [
    'detect_skill',
    'SkillTrigger',
    'route_skill',
    'validate_skill_name',
    'is_allowed_skill',
    'SkillBlock',
    'handle_search',
    'handle_note',
    'handle_fetch',
    'handle_scrape',
    'SkillResult',
    'format_skill_result',
    'format_skill_error',
    'inject_skill_block',
]
