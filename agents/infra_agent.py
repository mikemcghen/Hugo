"""
Infrastructure Agent
====================
Wraps ActionRouter so that DelegationAgent can delegate infrastructure
subtasks (SSH, Docker, Monitor) to this agent.

Registered with AgentRegistry for task types: ssh, docker, monitor, execute.

task.context can contain:
    - domain:     "ssh" | "docker" | "monitor"
    - action:     action name (e.g. "list", "restart", "run")
    - target:     host or container name (optional)
    - parameters: dict of additional params (optional)

Alternatively, task.description can be natural language — it will be parsed
by HugoIntentParser to extract domain/action/target.
"""

import asyncio
import concurrent.futures
import time

from .base_agent import BaseAgent, AgentTask, AgentResult, ProjectContext
from .registry import AgentRegistry


@AgentRegistry.register
class InfraAgent(BaseAgent):
    """
    Agent that handles infrastructure subtasks via ActionRouter.

    DelegationAgent calls this when it needs SSH/Docker/Monitor actions
    as part of a larger coordinated task.
    """

    name = "infra_agent"
    agent_type = "infra"
    capabilities = ["ssh", "docker", "monitor", "execute"]

    def __init__(
        self,
        ollama_url: str = None,
        model: str = None,
        context: ProjectContext = None,
    ):
        super().__init__(ollama_url=ollama_url, model=model, context=context)
        self._action_router = None
        self._intent_parser = None

    # ── Lazy accessors ─────────────────────────────────────────────────────

    def _get_router(self):
        if self._action_router is None:
            from core.actions.action_router import ActionRouter
            self._action_router = ActionRouter()
        return self._action_router

    def _get_parser(self):
        if self._intent_parser is None:
            from core.intent.intent_parser import HugoIntentParser
            self._intent_parser = HugoIntentParser(
                ollama_url=self.ollama_url,
                model_name=self.model,
            )
        return self._intent_parser

    # ── Async bridge ───────────────────────────────────────────────────────

    def _run_async(self, coro):
        """Run a coroutine from a synchronous context, handling nested loops."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context — use a thread pool to avoid deadlock
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result(timeout=60)
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop in current thread
            return asyncio.run(coro)

    # ── Core execution ─────────────────────────────────────────────────────

    def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute an infrastructure task.

        Structured context takes priority over natural language parsing.
        """
        start = time.time()

        # 1. Try structured context first (explicit domain/action from DelegationAgent)
        domain = task.context.get("domain")
        action = task.context.get("action")

        if domain and action:
            from core.intent.parsed_intent import ParsedIntent
            intent = ParsedIntent(
                requires_action=True,
                domain=domain,
                action=action,
                target=task.context.get("target"),
                parameters=task.context.get("parameters", {}),
                confidence=1.0,
                reasoning=f"Structured subtask from DelegationAgent: {task.description}",
                original_message=task.description,
            )
        else:
            # 2. Parse natural language task description
            parser = self._get_parser()
            intent = parser.parse(task.description)
            if not intent.requires_action:
                return AgentResult(
                    task_id=task.id,
                    success=False,
                    errors=[
                        f"Could not parse infrastructure intent from: {task.description!r}"
                    ],
                    execution_time=time.time() - start,
                    agent_name=self.name,
                )

        # 3. Route through ActionRouter
        router = self._get_router()
        action_result = self._run_async(router.route(intent))

        elapsed = time.time() - start
        response_text = router.format_for_cognition(action_result)

        return AgentResult(
            task_id=task.id,
            success=action_result.success,
            output=response_text,
            artifacts=(
                [action_result.to_dict()]
                if hasattr(action_result, "to_dict")
                else []
            ),
            errors=(
                [action_result.error]
                if not action_result.success and action_result.error
                else []
            ),
            execution_time=elapsed,
            agent_name=self.name,
        )

    def get_system_prompt(self) -> str:
        return (
            "You are an infrastructure execution agent. "
            "You execute SSH commands, manage Docker containers, and monitor network hosts. "
            "Report results clearly and concisely."
        )
