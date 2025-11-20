"""
Skill Schemas
-------------
Format skill results and errors into machine-readable blocks.

Block formats:

Success:
[SkillResult]
skill: <skill>
args: "<args>"
output: "<output>"

Error:
[SkillError]
skill: <skill>
error: "<message>"
"""

from core.skills.handlers import SkillResult


def format_skill_result(skill_name: str, args: str, result: SkillResult) -> str:
    """
    Format successful skill result into machine-readable block.

    Args:
        skill_name: Name of executed skill
        args: Arguments passed to skill
        result: SkillResult from handler

    Returns:
        Formatted block string

    Example:
        [SkillResult]
        skill: search
        args: "python asyncio"
        output: "[Placeholder: Search results]"
    """
    lines = [
        "[SkillResult]",
        f"skill: {skill_name}",
        f"args: \"{args}\"",
        f"output: \"{result.output}\""
    ]

    return "\n".join(lines)


def format_skill_error(skill_name: str, error_message: str) -> str:
    """
    Format skill error into machine-readable block.

    Args:
        skill_name: Name of skill that failed
        error_message: Error description

    Returns:
        Formatted error block string

    Example:
        [SkillError]
        skill: unknown
        error: "Unknown skill: unknown"
    """
    lines = [
        "[SkillError]",
        f"skill: {skill_name}",
        f"error: \"{error_message}\""
    ]

    return "\n".join(lines)
