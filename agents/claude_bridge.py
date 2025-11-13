"""
Claude Bridge
-------------
Interface for requesting external assistance from Claude Code.

This bridge provides a communication layer between Hugo's worker agent
and Claude Code for tasks that require file modifications or external tools.

Current Mode: SANDBOX - All requests auto-approved (no actual Claude Code integration)
Future Mode: PRODUCTION - Real integration with Claude Code via API/CLI
"""

from typing import Dict, Any, Optional, List
from datetime import datetime


class ClaudeBridge:
    """
    Bridge for communicating with Claude Code.

    In sandbox mode, this bridge:
    - Logs all file change requests
    - Auto-approves all requests
    - Tracks request history for auditing

    In production mode (future), this bridge will:
    - Send requests to Claude Code via API
    - Wait for human approval
    - Execute approved changes
    - Rollback on failures
    """

    def __init__(self, logger=None, sandbox_mode: bool = True):
        """
        Initialize Claude Bridge.

        Args:
            logger: Optional HugoLogger instance for logging
            sandbox_mode: If True, auto-approve all requests without external calls
        """
        self.logger = logger
        self.sandbox_mode = sandbox_mode
        self.request_history: List[Dict[str, Any]] = []

        if self.logger:
            self.logger.log_event("claude_bridge", "initialized", {
                "mode": "sandbox" if sandbox_mode else "production",
                "auto_approve": sandbox_mode
            })

    async def request_file_change(self, file_path: str, diff: str,
                                  explanation: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Request a file change through Claude Code.

        In sandbox mode: Auto-approves all requests without making changes
        In production mode: Sends request to Claude Code and waits for approval

        Args:
            file_path: Path to file to modify
            diff: Unified diff or change description
            explanation: Human-readable explanation of the change
            context: Optional context dictionary

        Returns:
            Dictionary with:
              - approved: bool - Whether change was approved
              - reason: str - Approval/rejection reason
              - request_id: str - Unique request identifier
        """
        request_id = f"claude_req_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        request_data = {
            "request_id": request_id,
            "file_path": file_path,
            "diff": diff,
            "explanation": explanation,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
            "mode": "sandbox" if self.sandbox_mode else "production"
        }

        if self.logger:
            self.logger.log_event("claude_bridge", "file_change_requested", {
                "request_id": request_id,
                "file_path": file_path,
                "explanation": explanation[:100]
            })

        if self.sandbox_mode:
            # Auto-approve in sandbox mode
            result = {
                "approved": True,
                "reason": "sandbox mode - auto-approved (no changes actually made)",
                "request_id": request_id,
                "executed": False  # Sandbox mode doesn't execute changes
            }

            if self.logger:
                self.logger.log_event("claude_bridge", "request_auto_approved", {
                    "request_id": request_id
                })

        else:
            # Future: Call Claude Code API
            result = {
                "approved": False,
                "reason": "production mode not yet implemented",
                "request_id": request_id,
                "executed": False
            }

            if self.logger:
                self.logger.log_event("claude_bridge", "request_pending_production", {
                    "request_id": request_id
                })

        # Store in history
        request_data["result"] = result
        self.request_history.append(request_data)

        return result

    async def request_code_execution(self, code: str, language: str,
                                     explanation: str) -> Dict[str, Any]:
        """
        Request code execution through Claude Code.

        In sandbox mode: Logs request but doesn't execute
        In production mode: Sends to Claude Code for execution

        Args:
            code: Code to execute
            language: Programming language (python, bash, etc.)
            explanation: Explanation of what the code does

        Returns:
            Dictionary with execution result or approval status
        """
        request_id = f"claude_exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if self.logger:
            self.logger.log_event("claude_bridge", "code_execution_requested", {
                "request_id": request_id,
                "language": language,
                "explanation": explanation[:100]
            })

        if self.sandbox_mode:
            result = {
                "approved": True,
                "executed": False,
                "reason": "sandbox mode - code logged but not executed",
                "request_id": request_id
            }
        else:
            result = {
                "approved": False,
                "executed": False,
                "reason": "production mode not yet implemented",
                "request_id": request_id
            }

        # Store in history
        self.request_history.append({
            "request_id": request_id,
            "type": "code_execution",
            "code": code,
            "language": language,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat(),
            "result": result
        })

        return result

    async def request_web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Request web search through Claude Code.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            Dictionary with search results or approval status
        """
        request_id = f"claude_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if self.logger:
            self.logger.log_event("claude_bridge", "web_search_requested", {
                "request_id": request_id,
                "query": query
            })

        if self.sandbox_mode:
            result = {
                "approved": True,
                "results": [],
                "reason": "sandbox mode - no actual web search performed",
                "request_id": request_id
            }
        else:
            result = {
                "approved": False,
                "results": [],
                "reason": "production mode not yet implemented",
                "request_id": request_id
            }

        # Store in history
        self.request_history.append({
            "request_id": request_id,
            "type": "web_search",
            "query": query,
            "max_results": max_results,
            "timestamp": datetime.now().isoformat(),
            "result": result
        })

        return result

    def get_request_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent request history.

        Args:
            limit: Maximum number of requests to return

        Returns:
            List of request dictionaries
        """
        return self.request_history[-limit:]

    def clear_history(self):
        """Clear request history"""
        self.request_history = []
        if self.logger:
            self.logger.log_event("claude_bridge", "history_cleared", {})

    def enable_production_mode(self):
        """
        Enable production mode (requires Claude Code integration).

        WARNING: This will enable actual file modifications and code execution.
        Only use when Claude Code integration is fully tested and secured.
        """
        if self.logger:
            self.logger.log_event("claude_bridge", "production_mode_requested", {
                "current_mode": "sandbox" if self.sandbox_mode else "production"
            })

        # For now, just log the request - don't actually enable
        return {
            "enabled": False,
            "reason": "production mode not yet implemented - requires Claude Code integration"
        }

    def disable_sandbox_mode(self):
        """
        Disable sandbox mode (alias for enable_production_mode).

        WARNING: This will enable actual file modifications and code execution.
        """
        return self.enable_production_mode()
