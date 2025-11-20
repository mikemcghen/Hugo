"""
Skill Trigger Detector
----------------------
Detects skill commands at the start of user input.

Format: /skillname arguments...

Requirements:
- Must detect commands ONLY at start of text
- Returns SkillTrigger with skill_name, args, raw_input
- Returns None when not matched
- Must NOT detect commands mid-string
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class SkillTrigger:
    """Represents a detected skill trigger"""
    skill_name: str
    args: str
    raw_input: str


def detect_skill(text: str) -> Optional[SkillTrigger]:
    """
    Detect skill command at start of text.

    Args:
        text: User input text

    Returns:
        SkillTrigger if command detected, None otherwise

    Examples:
        >>> detect_skill("/search python")
        SkillTrigger(skill_name='search', args='python', raw_input='/search python')

        >>> detect_skill("Can you /search?")
        None
    """
    if not text or not isinstance(text, str):
        return None

    # Strip for detection, but preserve raw input
    stripped = text.strip()

    if not stripped:
        return None

    # Must start with /
    if not stripped.startswith('/'):
        return None

    # Pattern: /skillname <optional args>
    # Skill name must be alphabetic characters only
    pattern = r'^/([a-zA-Z]+)(?:\s+(.*))?$'
    match = re.match(pattern, stripped)

    if not match:
        return None

    skill_name = match.group(1)
    args = match.group(2) if match.group(2) else ""

    # Normalize args: strip leading/trailing whitespace but preserve internal spacing
    args = args.strip()

    return SkillTrigger(
        skill_name=skill_name,
        args=args,
        raw_input=text
    )
