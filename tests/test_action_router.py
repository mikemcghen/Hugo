"""
Tests for ActionRouter — routes ParsedIntent to executors, checks permissions.
All executors are mocked; no real SSH/Docker/network calls are made.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from core.actions.action_router import ActionRouter
from core.actions.action_result import ActionResult
from core.actions.permission import PermissionLevel
from core.intent.parsed_intent import ParsedIntent
from core.executors.base import ExecutorResult


def make_intent(domain, action, target=None, confidence=0.9, params=None):
    return ParsedIntent(
        requires_action=True,
        domain=domain,
        action=action,
        target=target,
        parameters=params or {},
        confidence=confidence,
        reasoning="test",
        original_message=f"{action} {target or ''}",
    )


@pytest.fixture
def router():
    return ActionRouter(logger=None)


class TestPermissionGating:
    @pytest.mark.asyncio
    async def test_ask_first_returns_confirmation_required(self, router):
        intent = make_intent("docker", "stop", target="ollama")
        result = await router.route(intent)
        assert result.success is False
        assert result.error == "confirmation_required"
        assert "prompt" in (result.data or {})

    @pytest.mark.asyncio
    async def test_ssh_run_returns_confirmation_required(self, router):
        intent = make_intent("ssh", "run", target="proxmox", params={"command": "df -h"})
        result = await router.route(intent)
        assert result.success is False
        assert result.error == "confirmation_required"


class TestExecutorDispatch:
    @pytest.mark.asyncio
    async def test_docker_list_dispatches_to_executor(self, router):
        intent = make_intent("docker", "list")
        mock_exec_result = ExecutorResult(success=True, data=[{"name": "ollama", "status": "running", "image": "ollama"}])

        with patch("core.executors.executor_registry.ExecutorRegistry.get_executor") as mock_get:
            mock_executor = MagicMock()
            mock_executor.execute_async = AsyncMock(return_value=mock_exec_result)
            mock_get.return_value = mock_executor

            result = await router.route(intent)

        assert result.success is True
        assert result.domain == "docker"
        assert result.action == "list"

    @pytest.mark.asyncio
    async def test_monitor_status_dispatches_to_executor(self, router):
        intent = make_intent("monitor", "status")
        mock_exec_result = ExecutorResult(success=True, data={"hosts_up": 3, "hosts_down": 0, "total_hosts": 3, "hosts": {}})

        with patch("core.executors.executor_registry.ExecutorRegistry.get_executor") as mock_get:
            mock_executor = MagicMock()
            mock_executor.execute_async = AsyncMock(return_value=mock_exec_result)
            mock_get.return_value = mock_executor

            result = await router.route(intent)

        assert result.success is True
        assert result.permission_level == PermissionLevel.AUTO_EXECUTE.value

    @pytest.mark.asyncio
    async def test_unknown_domain_returns_error(self, router):
        intent = make_intent("unknown_domain", "some_action")
        with patch("core.executors.executor_registry.ExecutorRegistry.get_executor", return_value=None):
            result = await router.route(intent)
        assert result.success is False
        assert "No executor" in (result.error or "")

    @pytest.mark.asyncio
    async def test_target_mapped_to_container_for_docker(self, router):
        intent = make_intent("docker", "restart", target="ollama")
        mock_exec_result = ExecutorResult(success=True, data="Container 'ollama' restarted")
        captured_kwargs = {}

        async def capture_execute(action, **kwargs):
            captured_kwargs.update(kwargs)
            return mock_exec_result

        with patch("core.executors.executor_registry.ExecutorRegistry.get_executor") as mock_get:
            mock_executor = MagicMock()
            mock_executor.execute_async = capture_execute
            mock_get.return_value = mock_executor

            await router.route(intent)

        assert captured_kwargs.get("container") == "ollama"


class TestFormatForCognition:
    def test_docker_list_formatting(self, router):
        result = ActionResult(
            success=True,
            data=[{"name": "ollama", "status": "Up 2 hours", "image": "ollama/ollama"}],
            domain="docker",
            action="list",
        )
        text = router.format_for_cognition(result)
        assert "ollama" in text
        assert "Up 2 hours" in text

    def test_docker_list_empty(self, router):
        result = ActionResult(success=True, data=[], domain="docker", action="list")
        text = router.format_for_cognition(result)
        assert "No containers" in text

    def test_monitor_status_formatting(self, router):
        result = ActionResult(
            success=True,
            data={"hosts_up": 2, "hosts_down": 1, "total_hosts": 3, "hosts": {
                "proxmox": {"ip": "192.168.0.1", "is_up": True, "uptime_percent": 99.5},
                "truenas": {"ip": "192.168.0.2", "is_up": False, "uptime_percent": 85.0},
            }},
            domain="monitor",
            action="status",
        )
        text = router.format_for_cognition(result)
        assert "2/3" in text or "2" in text
        assert "proxmox" in text.lower() or "UP" in text

    def test_failure_formatting(self, router):
        result = ActionResult(success=False, error="Connection refused", domain="ssh", action="run")
        text = router.format_for_cognition(result)
        assert "failed" in text.lower() or "Connection refused" in text

    def test_confirmation_required_formatting(self, router):
        result = ActionResult(
            success=False,
            error="confirmation_required",
            data={"prompt": "Stop container ollama? It will be unavailable until restarted."},
            domain="docker",
            action="stop",
        )
        text = router.format_for_cognition(result)
        assert "ollama" in text.lower() or "stop" in text.lower()
        assert "yes" in text.lower() or "confirm" in text.lower()


class TestActionResultToSkillBlock:
    def test_success_produces_skill_result_block(self):
        result = ActionResult(success=True, data="Container restarted", domain="docker", action="restart")
        block = result.to_skill_block()
        assert block.startswith("[SkillResult]")
        assert "docker" in block
        assert "restart" in block

    def test_failure_produces_skill_error_block(self):
        result = ActionResult(success=False, error="Host not found", domain="ssh", action="run")
        block = result.to_skill_block()
        assert block.startswith("[SkillError]")
        assert "Host not found" in block
