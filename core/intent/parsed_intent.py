"""
Parsed Intent
-------------
Structured representation of user intent parsed from natural language.

Adapted from Server-Files llm_intent_parser.py ParsedIntent dataclass,
with an added `needs_confirmation` field for the permission framework.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class ParsedIntent:
    """
    Structured representation of what a user wants to do.

    Produced by HugoIntentParser from natural language input.
    Consumed by ActionRouter to dispatch to the right executor.
    """
    requires_action: bool
    domain: Optional[str] = None       # 'docker', 'ssh', 'monitor', or None
    action: Optional[str] = None       # specific action within the domain
    target: Optional[str] = None       # what to act on (container name, host, etc.)
    parameters: Optional[Dict] = field(default_factory=dict)
    reasoning: str = ""                # LLM's explanation of why it parsed this way
    confidence: float = 0.0            # 0.0 - 1.0
    original_message: str = ""         # the raw user message
    needs_confirmation: bool = False   # whether PermissionGate flagged this for confirmation
