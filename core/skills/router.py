"""
Skill Router
------------
Routes skill triggers to appropriate handlers.

Requirements:
- validate_skill_name: Only lowercase alphabetic a-z
- is_allowed_skill: Whitelist check
- route_skill: Route to handler and return SkillBlock
- Must NOT call Ollama directly
- Must NOT do any network or file I/O
"""

import re
from dataclasses import dataclass
from typing import Optional

from core.skills.trigger_detector import SkillTrigger
from core.skills.handlers import (
    handle_search, handle_note, handle_fetch, handle_scrape,
    handle_ssh, handle_docker, handle_monitor,
)
from core.skills.schemas import format_skill_result, format_skill_error


# Allowed skills whitelist — original four + new infrastructure skills
ALLOWED_SKILLS = {'search', 'note', 'fetch', 'scrape', 'ssh', 'docker', 'monitor'}


@dataclass
class SkillBlock:
    """Wraps a formatted skill result or error block"""
    block: str


def validate_skill_name(skill_name: str) -> bool:
    """
    Validate that skill name contains only lowercase alphabetic characters.

    Args:
        skill_name: Skill name to validate

    Returns:
        True if valid, False otherwise

    Rules:
        - Only lowercase a-z allowed
        - No uppercase
        - No numbers
        - No underscores, hyphens, or special characters
        - No unicode
        - No empty strings
    """
    if not skill_name:
        return False

    # Must be only lowercase alphabetic characters
    pattern = r'^[a-z]+$'
    return bool(re.match(pattern, skill_name))


def is_allowed_skill(skill_name: str) -> bool:
    """
    Check if skill is in the allowed whitelist.

    Args:
        skill_name: Skill name to check

    Returns:
        True if skill is allowed, False otherwise

    Allowed skills:
        - search
        - note
        - fetch
        - scrape
    """
    return skill_name in ALLOWED_SKILLS


async def route_skill(trigger: SkillTrigger) -> SkillBlock:
    """
    Route skill trigger to appropriate handler.

    Args:
        trigger: Detected skill trigger

    Returns:
        SkillBlock containing formatted result or error

    Routing logic:
        1. Validate skill name format
        2. Check if skill is allowed
        3. Route to handler
        4. Format result or error
    """
    skill_name = trigger.skill_name
    args = trigger.args

    # Validate skill name format
    if not validate_skill_name(skill_name):
        error_msg = f"Invalid skill name format: {skill_name}"
        formatted = format_skill_error(skill_name, error_msg)
        return SkillBlock(block=formatted)

    # Check if skill is allowed
    if not is_allowed_skill(skill_name):
        error_msg = f"Unknown skill: {skill_name}"
        formatted = format_skill_error(skill_name, error_msg)
        return SkillBlock(block=formatted)

    # Route to appropriate handler
    handler = None
    if skill_name == 'search':
        handler = handle_search
    elif skill_name == 'note':
        handler = handle_note
    elif skill_name == 'fetch':
        handler = handle_fetch
    elif skill_name == 'scrape':
        handler = handle_scrape
    elif skill_name == 'ssh':
        handler = handle_ssh
    elif skill_name == 'docker':
        handler = handle_docker
    elif skill_name == 'monitor':
        handler = handle_monitor

    if handler is None:
        error_msg = f"No handler found for skill: {skill_name}"
        formatted = format_skill_error(skill_name, error_msg)
        return SkillBlock(block=formatted)

    # Execute handler
    try:
        result = await handler(args)

        if result is None:
            error_msg = f"Handler returned None for skill: {skill_name}"
            formatted = format_skill_error(skill_name, error_msg)
            return SkillBlock(block=formatted)

        # Format successful result
        formatted = format_skill_result(skill_name, args, result)
        return SkillBlock(block=formatted)

    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        formatted = format_skill_error(skill_name, error_msg)
        return SkillBlock(block=formatted)
