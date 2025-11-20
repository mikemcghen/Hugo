"""
Prompt Injection
----------------
Inject skill blocks into prompts before user text.

Requirements:
- Place skill block BEFORE user text
- Preserve both pieces
- No extra whitespace at start or end
- Separate with appropriate spacing
"""


def inject_skill_block(skill_block: str, user_text: str) -> str:
    """
    Inject skill block into prompt before user text.

    Args:
        skill_block: Formatted skill result or error block
        user_text: User's original text

    Returns:
        Combined prompt with skill block first

    Example:
        >>> block = "[SkillResult]\\nskill: search\\noutput: \\"results\\""
        >>> text = "What did you find?"
        >>> inject_skill_block(block, text)
        '[SkillResult]\\nskill: search\\noutput: "results"\\n\\nWhat did you find?'

    Rules:
        - Skill block comes first
        - User text comes second
        - Separated by double newline
        - No extra whitespace at start or end
    """
    # Strip any trailing whitespace from skill block
    skill_block = skill_block.rstrip()

    # Strip any leading whitespace from user text
    user_text = user_text.lstrip()

    # Combine with double newline separator
    if user_text:
        combined = f"{skill_block}\n\n{user_text}"
    else:
        combined = skill_block

    return combined
