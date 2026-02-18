"""
Tests for DockerExecutor.
SSHExecutor is mocked — no real SSH or Docker calls made.
"""

import pytest
from unittest.mock import patch, MagicMock
from core.executors.docker import DockerExecutor
from core.executors.base import ExecutorResult


def ssh_ok(data):
    """Helper: create a successful SSH ExecutorResult."""
    return ExecutorResult(success=True, data={"stdout": data, "stderr": "", "exit_code": 0})


def ssh_fail(error="SSH failed"):
    return ExecutorResult(success=False, error=error)


@pytest.fixture
def executor():
    with patch("core.executors.docker.DockerExecutor.__init__", lambda self: None):
        e = DockerExecutor.__new__(DockerExecutor)
        e.last_execution = None
        e.execution_count = 0
        e.docker_host = "testhost"
        e.container_groups = {"mygroup": ["c1", "c2", "c3"]}

        mock_ssh = MagicMock()
        mock_ssh.execute = MagicMock(return_value=ssh_ok(""))
        e.ssh = mock_ssh
        return e


class TestListContainers:
    def test_list_running_containers(self, executor):
        executor.ssh.execute.return_value = ssh_ok(
            "ollama|Up 3 hours|ollama/ollama\nimmich_server|Up 1 hour|ghcr.io/immich"
        )
        result = executor.execute("list")
        assert result.success is True
        assert len(result.data) == 2
        assert result.data[0]["name"] == "ollama"

    def test_list_empty(self, executor):
        executor.ssh.execute.return_value = ssh_ok("")
        result = executor.execute("list")
        assert result.success is True
        assert result.data == []


class TestContainerLifecycle:
    def test_start_container(self, executor):
        executor.ssh.execute.return_value = ssh_ok("ollama")
        result = executor.execute("start", container="ollama")
        assert result.success is True
        assert "ollama" in str(result.data)

    def test_stop_container(self, executor):
        executor.ssh.execute.return_value = ssh_ok("ollama")
        result = executor.execute("stop", container="ollama")
        assert result.success is True

    def test_restart_container(self, executor):
        executor.ssh.execute.return_value = ssh_ok("ollama")
        result = executor.execute("restart", container="ollama")
        assert result.success is True

    def test_start_without_container_returns_error(self, executor):
        result = executor.execute("start")
        assert result.success is False
        assert "Container name required" in result.error

    def test_stop_without_container_returns_error(self, executor):
        result = executor.execute("stop")
        assert result.success is False


class TestContainerLogs:
    def test_get_logs(self, executor):
        executor.ssh.execute.return_value = ssh_ok("2024-01-01 INFO Starting server")
        result = executor.execute("logs", container="ollama", lines=10)
        assert result.success is True
        assert "logs" in result.data

    def test_logs_without_container_returns_error(self, executor):
        result = executor.execute("logs")
        assert result.success is False


class TestContainerStatus:
    def test_get_status(self, executor):
        executor.ssh.execute.return_value = ssh_ok("running")
        result = executor.execute("status", container="ollama")
        assert result.success is True
        assert result.data["container"] == "ollama"
        assert result.data["status"] == "running"


class TestContainerGroups:
    def test_start_group(self, executor):
        executor.ssh.execute.return_value = ssh_ok("c1")
        result = executor.execute("start_group", group="mygroup")
        assert result.success is True
        assert result.data["group"] == "mygroup"
        assert len(result.data["results"]) == 3

    def test_stop_group_unknown_raises_error(self, executor):
        result = executor.execute("stop_group", group="unknowngroup")
        assert result.success is False

    def test_start_group_unknown_raises_error(self, executor):
        result = executor.execute("start_group", group="unknowngroup")
        assert result.success is False


class TestSSHFailure:
    def test_start_propagates_ssh_error(self, executor):
        executor.ssh.execute.return_value = ssh_fail("Connection refused")
        result = executor.execute("start", container="ollama")
        assert result.success is False


class TestNoHostConfigured:
    def test_no_docker_host_returns_error(self):
        with patch("core.executors.docker.DockerExecutor.__init__", lambda self: None):
            e = DockerExecutor.__new__(DockerExecutor)
            e.last_execution = None
            e.execution_count = 0
            e.docker_host = ""
            e.container_groups = {}
            mock_ssh = MagicMock()
            mock_ssh.hosts = {}
            e.ssh = mock_ssh

        result = e.execute("list")
        assert result.success is False
        assert "No Docker host" in result.error
