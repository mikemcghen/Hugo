"""
Action Router
-------------
Routes ParsedIntent objects to the appropriate executor and wraps the
result in a standardized ActionResult.

This is the bridge between the intent layer (HugoIntentParser) and the
executor layer (SSH/Docker/MonitorExecutor).

Flow:
    ParsedIntent → PermissionGate → Executor → ActionResult
"""

import time
from typing import Optional

from .action_result import ActionResult
from .permission import PermissionGate, PermissionLevel
from ..intent.parsed_intent import ParsedIntent
from ..executors.executor_registry import ExecutorRegistry

# Trigger executor imports so they self-register with ExecutorRegistry
import core.executors.ssh      # noqa: F401
import core.executors.docker   # noqa: F401
import core.executors.monitor  # noqa: F401


class ActionRouter:
    """
    Routes intent to executor and returns a standardized ActionResult.

    Used by CognitionEngine when natural language intent parsing detects
    an action request with sufficient confidence.
    """

    def __init__(self, logger=None):
        self.gate = PermissionGate()
        self.logger = logger

    async def route(self, intent: ParsedIntent) -> ActionResult:
        """
        Route a ParsedIntent to the appropriate executor.

        Args:
            intent: Parsed intent from HugoIntentParser

        Returns:
            ActionResult wrapping the executor result
        """
        start = time.time()
        domain = intent.domain or "unknown"
        action = intent.action or "unknown"
        target = intent.target
        params = intent.parameters or {}

        # Check permission level
        level = self.gate.check(domain, action)
        if self.gate.requires_confirmation(level):
            prompt = self.gate.confirmation_prompt(domain, action, target)
            self._log("permission_gate", {
                "domain": domain, "action": action,
                "level": level.value, "blocked": True
            })
            return ActionResult(
                success=False,
                error="confirmation_required",
                data={"prompt": prompt, "intent": intent.__dict__},
                domain=domain,
                action=action,
                permission_level=level.value,
                execution_time=time.time() - start,
            )

        # Find executor
        executor = ExecutorRegistry.get_executor(domain)
        if not executor:
            return ActionResult(
                success=False,
                error=f"No executor registered for domain: {domain!r}. "
                      f"Available: {[e['name'] for e in ExecutorRegistry.list_executors()]}",
                domain=domain,
                action=action,
                permission_level=level.value,
                execution_time=time.time() - start,
            )

        # Map target to the right param name based on domain
        enriched_params = {**params}
        if target:
            if domain == "docker":
                enriched_params.setdefault("container", target)
            elif domain == "ssh":
                enriched_params.setdefault("host", target)
            elif domain == "monitor":
                enriched_params.setdefault("host", target)

        self._log("action_dispatch", {
            "domain": domain, "action": action,
            "target": target, "level": level.value
        })

        # Execute asynchronously
        exec_result = await executor.execute_async(action, **enriched_params)

        result = ActionResult(
            success=exec_result.success,
            data=exec_result.data,
            error=exec_result.error,
            execution_time=exec_result.execution_time,
            timestamp=exec_result.timestamp,
            domain=domain,
            action=action,
            permission_level=level.value,
        )

        self._log("action_complete", {
            "domain": domain, "action": action,
            "success": result.success,
            "execution_time": result.execution_time,
        })

        return result

    def format_for_cognition(self, result: ActionResult) -> str:
        """
        Format an ActionResult as a human-readable string for injection
        into CognitionEngine's response pipeline.

        The result is also compatible with inject_skill_block() via to_skill_block().

        Args:
            result: ActionResult from route()

        Returns:
            Human-readable summary string
        """
        if not result.success:
            if result.error == "confirmation_required":
                prompt = result.data.get("prompt", "Confirm?") if result.data else "Confirm?"
                return f"{prompt} (Reply 'yes' to proceed or 'no' to cancel.)"
            return f"Action failed [{result.domain}/{result.action}]: {result.error}"

        data = result.data
        domain = result.domain
        action = result.action

        # Domain-specific formatting
        if domain == "docker":
            if action == "list" and isinstance(data, list):
                if not data:
                    return "No containers running."
                lines = [f"- {c['name']}: {c['status']}" for c in data]
                return "Running containers:\n" + "\n".join(lines)
            elif action in ("start", "stop", "restart") and isinstance(data, str):
                return data
            elif action == "logs" and isinstance(data, dict):
                logs = data.get("logs", "")
                container = data.get("container", "")
                return f"Logs for {container}:\n{logs[-2000:]}" if logs else f"No logs for {container}."
            elif action == "status" and isinstance(data, dict):
                return f"{data.get('container', '?')}: {data.get('status', '?')}"

        elif domain == "ssh":
            if action == "run" and isinstance(data, dict):
                stdout = data.get("stdout", "")
                return stdout if stdout else "(Command executed, no output)"
            elif action == "list_hosts" and isinstance(data, dict):
                if not data:
                    return "No SSH hosts configured."
                lines = [f"- {n}: {v.get('ip', '?')} ({v.get('description', '')})"
                         for n, v in data.items()]
                return "SSH hosts:\n" + "\n".join(lines)
            elif action == "test":
                return "SSH connection successful." if result.success else f"SSH test failed: {result.error}"

        elif domain == "monitor":
            if action in ("status", "full_status") and isinstance(data, dict):
                up = data.get("hosts_up", 0)
                down = data.get("hosts_down", 0)
                total = data.get("total_hosts", 0)
                lines = [f"Network status: {up}/{total} hosts up"]
                for name, info in data.get("hosts", {}).items():
                    status_str = "UP" if info.get("is_up") else "DOWN"
                    lines.append(f"  {name}: {status_str} ({info.get('uptime_percent', 0):.1f}% uptime)")
                return "\n".join(lines)
            elif action == "check" and isinstance(data, dict):
                is_up = "UP" if data.get("is_up") else "DOWN"
                return f"{data.get('name', '?')} is {is_up} (latency: {data.get('latency_ms', 0):.1f}ms)"
            elif action == "alerts" and isinstance(data, list):
                if not data:
                    return "No active alerts."
                return "Active alerts:\n" + "\n".join(f"- {a['host']}: {a['message']}" for a in data)
            elif action == "hosts" and isinstance(data, list):
                if not data:
                    return "No monitored hosts."
                lines = [f"- {h['name']} ({h['ip']}): {'UP' if h['is_up'] else 'DOWN'}" for h in data]
                return "Monitored hosts:\n" + "\n".join(lines)

        # Generic fallback
        if isinstance(data, str):
            return data
        if isinstance(data, (dict, list)):
            import json
            return json.dumps(data, indent=2, default=str)
        return str(data) if data else f"{domain}/{action} completed."

    def _log(self, event: str, details: dict):
        if self.logger:
            try:
                self.logger.log_event("action_router", event, details)
            except Exception:
                pass
