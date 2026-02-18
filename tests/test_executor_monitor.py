"""
Tests for MonitorExecutor and NetworkMonitorAdapter.
No actual network calls are made.
"""

import pytest
from unittest.mock import patch, MagicMock
from core.executors.monitor import MonitorExecutor
from core.executors.network_monitor_adapter import NetworkMonitorAdapter, HostStatus, get_monitor


@pytest.fixture
def adapter():
    """Fresh adapter instance with test hosts, ping mocked to success."""
    a = NetworkMonitorAdapter()
    a.hosts = {
        "proxmox": HostStatus(name="proxmox", ip="192.168.0.1", is_up=True, total_checks=10, total_up=10),
        "truenas": HostStatus(name="truenas", ip="192.168.0.2", is_up=False, total_checks=10, total_up=7),
    }
    return a


@pytest.fixture
def executor(adapter):
    e = MonitorExecutor()
    e.monitor = adapter
    return e


class TestStatusActions:
    def test_status_returns_summary(self, executor):
        result = executor.execute("status")
        assert result.success is True
        assert "hosts_up" in result.data
        assert "hosts_down" in result.data
        assert result.data["total_hosts"] == 2

    def test_hosts_returns_list(self, executor):
        result = executor.execute("hosts")
        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) == 2

    def test_alerts_returns_list(self, executor):
        result = executor.execute("alerts")
        assert result.success is True
        assert isinstance(result.data, list)

    def test_uptime_specific_host(self, executor):
        result = executor.execute("uptime", host="proxmox")
        assert result.success is True
        assert result.data["host"] == "proxmox"
        assert result.data["uptime_percent"] == 100.0

    def test_uptime_unknown_host_returns_error(self, executor):
        result = executor.execute("uptime", host="unknownhost")
        assert result.success is False

    def test_all_devices(self, executor):
        result = executor.execute("all_devices")
        assert result.success is True
        assert isinstance(result.data, list)


class TestHostManagement:
    def test_add_host(self, executor):
        result = executor.execute("add_host", name="newhost", ip="10.0.0.1")
        assert result.success is True
        assert "newhost" in executor.monitor.hosts

    def test_add_host_missing_params(self, executor):
        result = executor.execute("add_host")
        assert result.success is False
        assert "required" in result.error.lower()

    def test_remove_host(self, executor):
        result = executor.execute("remove_host", name="truenas")
        assert result.success is True
        assert "truenas" not in executor.monitor.hosts

    def test_remove_host_missing_name(self, executor):
        result = executor.execute("remove_host")
        assert result.success is False


class TestCheckHost:
    def test_check_known_host(self, executor, adapter):
        with patch.object(adapter, "_ping", return_value=(True, 1.5)):
            result = executor.execute("check", host="proxmox")
        assert result.success is True
        assert result.data["name"] == "proxmox"
        assert result.data["is_up"] is True

    def test_check_unknown_host_returns_error(self, executor):
        result = executor.execute("check", host="unknownhost")
        assert result.success is False

    def test_check_all_no_host(self, executor, adapter):
        with patch.object(adapter, "_ping", return_value=(True, 2.0)):
            result = executor.execute("check")
        assert result.success is True


class TestUnknownAction:
    def test_unknown_action_returns_error(self, executor):
        result = executor.execute("fly")
        assert result.success is False
        assert "Unknown monitor action" in result.error


class TestNetworkMonitorAdapter:
    def test_uptime_zero_checks(self):
        h = HostStatus(name="x", ip="1.1.1.1", total_checks=0, total_up=0)
        assert h.uptime_percent == 0.0

    def test_uptime_calculation(self):
        h = HostStatus(name="x", ip="1.1.1.1", total_checks=100, total_up=95)
        assert h.uptime_percent == 95.0

    def test_status_summary_structure(self, adapter):
        summary = adapter.get_status_summary()
        assert "hosts_up" in summary
        assert "hosts_down" in summary
        assert "total_hosts" in summary
        assert "hosts" in summary
        assert summary["hosts_up"] == 1
        assert summary["hosts_down"] == 1
