"""
Tests for PermissionGate — the three-tier permission system.

Every (domain, action) pair in the permission table is verified.
Hardcoded dangerous patterns always return ASK_FIRST regardless of domain.
"""

import pytest
from core.actions.permission import PermissionGate, PermissionLevel


@pytest.fixture
def gate():
    return PermissionGate()


class TestPermissionLevels:
    def test_monitor_status_auto(self, gate):
        assert gate.check("monitor", "status") == PermissionLevel.AUTO_EXECUTE

    def test_monitor_check_auto(self, gate):
        assert gate.check("monitor", "check") == PermissionLevel.AUTO_EXECUTE

    def test_monitor_alerts_auto(self, gate):
        assert gate.check("monitor", "alerts") == PermissionLevel.AUTO_EXECUTE

    def test_monitor_hosts_auto(self, gate):
        assert gate.check("monitor", "hosts") == PermissionLevel.AUTO_EXECUTE

    def test_monitor_scan_auto(self, gate):
        assert gate.check("monitor", "scan") == PermissionLevel.AUTO_EXECUTE

    def test_monitor_add_host_report(self, gate):
        assert gate.check("monitor", "add_host") == PermissionLevel.EXECUTE_AND_REPORT

    def test_monitor_remove_host_ask(self, gate):
        assert gate.check("monitor", "remove_host") == PermissionLevel.ASK_FIRST

    def test_docker_list_auto(self, gate):
        assert gate.check("docker", "list") == PermissionLevel.AUTO_EXECUTE

    def test_docker_status_auto(self, gate):
        assert gate.check("docker", "status") == PermissionLevel.AUTO_EXECUTE

    def test_docker_logs_auto(self, gate):
        assert gate.check("docker", "logs") == PermissionLevel.AUTO_EXECUTE

    def test_docker_start_report(self, gate):
        assert gate.check("docker", "start") == PermissionLevel.EXECUTE_AND_REPORT

    def test_docker_restart_report(self, gate):
        assert gate.check("docker", "restart") == PermissionLevel.EXECUTE_AND_REPORT

    def test_docker_stop_ask(self, gate):
        assert gate.check("docker", "stop") == PermissionLevel.ASK_FIRST

    def test_docker_stop_group_ask(self, gate):
        assert gate.check("docker", "stop_group") == PermissionLevel.ASK_FIRST

    def test_ssh_list_hosts_auto(self, gate):
        assert gate.check("ssh", "list_hosts") == PermissionLevel.AUTO_EXECUTE

    def test_ssh_test_auto(self, gate):
        assert gate.check("ssh", "test") == PermissionLevel.AUTO_EXECUTE

    def test_ssh_run_ask(self, gate):
        assert gate.check("ssh", "run") == PermissionLevel.ASK_FIRST

    def test_ssh_remove_host_ask(self, gate):
        assert gate.check("ssh", "remove_host") == PermissionLevel.ASK_FIRST

    def test_ssh_add_host_report(self, gate):
        assert gate.check("ssh", "add_host") == PermissionLevel.EXECUTE_AND_REPORT


class TestAlwaysAskActions:
    def test_delete_always_ask(self, gate):
        assert gate.check("docker", "delete") == PermissionLevel.ASK_FIRST
        assert gate.check("ssh", "delete") == PermissionLevel.ASK_FIRST

    def test_destroy_always_ask(self, gate):
        assert gate.check("monitor", "destroy") == PermissionLevel.ASK_FIRST

    def test_wipe_always_ask(self, gate):
        assert gate.check("docker", "wipe") == PermissionLevel.ASK_FIRST


class TestUnknownCombinations:
    def test_unknown_domain_defaults_to_report(self, gate):
        assert gate.check("unknown_domain", "some_action") == PermissionLevel.EXECUTE_AND_REPORT

    def test_unknown_action_defaults_to_report(self, gate):
        assert gate.check("docker", "unknown_action") == PermissionLevel.EXECUTE_AND_REPORT


class TestRequiresConfirmation:
    def test_ask_first_requires_confirmation(self, gate):
        assert gate.requires_confirmation(PermissionLevel.ASK_FIRST) is True

    def test_auto_does_not_require(self, gate):
        assert gate.requires_confirmation(PermissionLevel.AUTO_EXECUTE) is False

    def test_report_does_not_require(self, gate):
        assert gate.requires_confirmation(PermissionLevel.EXECUTE_AND_REPORT) is False


class TestConfirmationPrompt:
    def test_docker_stop_prompt(self, gate):
        prompt = gate.confirmation_prompt("docker", "stop", "ollama")
        assert "ollama" in prompt.lower() or "stop" in prompt.lower()

    def test_ssh_run_prompt(self, gate):
        prompt = gate.confirmation_prompt("ssh", "run", "proxmox")
        assert len(prompt) > 5

    def test_generic_prompt_fallback(self, gate):
        prompt = gate.confirmation_prompt("monitor", "remove_host", "proxmox")
        assert "proxmox" in prompt or "monitor" in prompt or "remove_host" in prompt
