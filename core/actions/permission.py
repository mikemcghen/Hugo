"""
Permission Framework
--------------------
Three-tier permission system for Hugo's action layer.

Permission levels:
- AUTO_EXECUTE: Safe read-only or low-risk operations. Execute immediately.
- EXECUTE_AND_REPORT: Meaningful but reversible changes. Execute and tell the user.
- ASK_FIRST: Destructive, hard-to-reverse, or high-impact actions. Ask before executing.

Adapted from Server-Files helpers/digital.py permission model.
"""

from enum import Enum
from typing import Tuple, Optional


class PermissionLevel(Enum):
    AUTO_EXECUTE = "auto_execute"
    EXECUTE_AND_REPORT = "execute_and_report"
    ASK_FIRST = "ask_first"


# Permission table: (domain, action) -> PermissionLevel
# Keys are lowercase. Missing entries default to EXECUTE_AND_REPORT.
_PERMISSION_TABLE: dict[Tuple[str, str], PermissionLevel] = {
    # SSH - read actions are safe; write/exec actions need confirmation
    ("ssh", "list_hosts"):      PermissionLevel.AUTO_EXECUTE,
    ("ssh", "test"):            PermissionLevel.AUTO_EXECUTE,
    ("ssh", "run"):             PermissionLevel.ASK_FIRST,      # arbitrary remote commands
    ("ssh", "add_host"):        PermissionLevel.EXECUTE_AND_REPORT,
    ("ssh", "remove_host"):     PermissionLevel.ASK_FIRST,

    # Docker - listing/status is safe; mutations need reporting
    ("docker", "list"):         PermissionLevel.AUTO_EXECUTE,
    ("docker", "status"):       PermissionLevel.AUTO_EXECUTE,
    ("docker", "logs"):         PermissionLevel.AUTO_EXECUTE,
    ("docker", "start"):        PermissionLevel.EXECUTE_AND_REPORT,
    ("docker", "restart"):      PermissionLevel.EXECUTE_AND_REPORT,
    ("docker", "stop"):         PermissionLevel.ASK_FIRST,
    ("docker", "start_group"):  PermissionLevel.EXECUTE_AND_REPORT,
    ("docker", "stop_group"):   PermissionLevel.ASK_FIRST,

    # Monitor - all read-only
    ("monitor", "status"):      PermissionLevel.AUTO_EXECUTE,
    ("monitor", "full_status"): PermissionLevel.AUTO_EXECUTE,
    ("monitor", "check"):       PermissionLevel.AUTO_EXECUTE,
    ("monitor", "alerts"):      PermissionLevel.AUTO_EXECUTE,
    ("monitor", "uptime"):      PermissionLevel.AUTO_EXECUTE,
    ("monitor", "hosts"):       PermissionLevel.AUTO_EXECUTE,
    ("monitor", "scan"):        PermissionLevel.AUTO_EXECUTE,
    ("monitor", "services"):    PermissionLevel.AUTO_EXECUTE,
    ("monitor", "all_devices"): PermissionLevel.AUTO_EXECUTE,
    ("monitor", "add_host"):    PermissionLevel.EXECUTE_AND_REPORT,
    ("monitor", "remove_host"): PermissionLevel.ASK_FIRST,
    ("monitor", "add_service"): PermissionLevel.EXECUTE_AND_REPORT,
}

# Actions that are always dangerous regardless of domain
_ALWAYS_ASK = {"delete", "destroy", "wipe", "format", "drop", "purge", "rm"}


class PermissionGate:
    """
    Checks the permission level for a (domain, action) pair.

    Used by ActionRouter before dispatching to any executor.
    """

    def check(self, domain: str, action: str) -> PermissionLevel:
        """
        Return the permission level for this domain/action pair.

        Args:
            domain: Executor domain (e.g., 'docker', 'ssh', 'monitor')
            action: Action name (e.g., 'list', 'restart', 'run')

        Returns:
            PermissionLevel enum value
        """
        domain = domain.lower()
        action = action.lower()

        # Hardcoded dangerous actions always require confirmation
        if action in _ALWAYS_ASK:
            return PermissionLevel.ASK_FIRST

        key = (domain, action)
        return _PERMISSION_TABLE.get(key, PermissionLevel.EXECUTE_AND_REPORT)

    def requires_confirmation(self, level: PermissionLevel) -> bool:
        """Return True if this level requires user confirmation before executing."""
        return level == PermissionLevel.ASK_FIRST

    def confirmation_prompt(self, domain: str, action: str, target: Optional[str] = None) -> str:
        """
        Generate a natural language confirmation prompt for ASK_FIRST actions.

        Args:
            domain: Executor domain
            action: Action name
            target: Optional target (container name, host, etc.)

        Returns:
            Human-readable confirmation question
        """
        target_str = f" on {target}" if target else ""

        prompts = {
            ("ssh", "run"):         f"Run a command{target_str} via SSH?",
            ("ssh", "remove_host"): f"Remove host {target or 'unknown'} from SSH config?",
            ("docker", "stop"):     f"Stop container {target or 'unknown'}? It will be unavailable until restarted.",
            ("docker", "stop_group"): f"Stop container group {target or 'all'}?",
        }

        key = (domain.lower(), action.lower())
        if key in prompts:
            return prompts[key]

        return f"Proceed with {action}{target_str} on {domain}? (y/n)"
