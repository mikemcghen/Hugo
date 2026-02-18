"""
Monitor Executor
----------------
Provides access to network monitoring data via the lightweight adapter.

Ported from Server-Files executors/monitor.py, replacing the
network_monitor dependency with network_monitor_adapter.
"""

from .base import BaseExecutor, ExecutorResult
from .network_monitor_adapter import get_monitor
from .executor_registry import ExecutorRegistry


@ExecutorRegistry.register
class MonitorExecutor(BaseExecutor):
    """Network monitoring and alerting."""

    name = "monitor"
    description = "Network monitoring — check host status, alerts, uptime"

    def __init__(self):
        super().__init__()
        self.monitor = get_monitor()

    def _execute(self, action: str, **params) -> ExecutorResult:
        if action == "status":
            return self._get_status()
        elif action == "full_status":
            return self._get_full_status()
        elif action == "check":
            host = params.get("host")
            return self._check_host(host) if host else self._check_all()
        elif action == "alerts":
            return self._get_alerts()
        elif action == "uptime":
            return self._get_uptime(params.get("host"))
        elif action == "hosts":
            return self._list_hosts()
        elif action == "add_host":
            name, ip = params.get("name"), params.get("ip")
            if not name or not ip:
                return ExecutorResult(success=False, error="Name and IP required")
            return self._add_host(name, ip)
        elif action == "remove_host":
            name = params.get("name")
            if not name:
                return ExecutorResult(success=False, error="Host name required")
            return self._remove_host(name)
        elif action == "scan":
            return self._scan_network()
        elif action == "services":
            return self._check_services(params.get("host"))
        elif action == "all_devices":
            return self._get_all_devices()
        elif action == "add_service":
            host, port, name = params.get("host"), params.get("port"), params.get("name")
            if not host or not port:
                return ExecutorResult(success=False, error="Host and port required")
            return self._add_service(host, int(port), name)
        else:
            return ExecutorResult(success=False, error=f"Unknown monitor action: {action}")

    def _get_status(self) -> ExecutorResult:
        summary = self.monitor.get_status_summary()
        return ExecutorResult(success=True, data=summary)

    def _get_full_status(self) -> ExecutorResult:
        status = self.monitor.get_full_network_status()
        return ExecutorResult(success=True, data=status)

    def _check_host(self, host: str) -> ExecutorResult:
        try:
            status = self.monitor.check_host(host)
            return ExecutorResult(success=True, data={
                "name": status.name,
                "ip": status.ip,
                "is_up": status.is_up,
                "latency_ms": status.latency_ms,
                "uptime_percent": round(status.uptime_percent, 1),
                "consecutive_failures": status.consecutive_failures,
            })
        except ValueError as e:
            return ExecutorResult(success=False, error=str(e))

    def _check_all(self) -> ExecutorResult:
        self.monitor.check_all()
        return ExecutorResult(success=True, data=self.monitor.get_status_summary())

    def _get_alerts(self) -> ExecutorResult:
        alerts = self.monitor.get_alerts(unacknowledged_only=True)
        return ExecutorResult(success=True, data=[{
            "host": a.host,
            "type": a.alert_type,
            "message": a.message,
            "timestamp": a.timestamp,
        } for a in alerts[-10:]])

    def _get_uptime(self, host: str = None) -> ExecutorResult:
        if host:
            status = self.monitor.hosts.get(host)
            if not status:
                return ExecutorResult(success=False, error=f"Unknown host: {host}")
            return ExecutorResult(success=True, data={
                "host": host,
                "uptime_percent": round(status.uptime_percent, 1),
                "total_checks": status.total_checks,
                "total_up": status.total_up,
            })
        uptime_data = {
            name: {"uptime_percent": round(h.uptime_percent, 1), "is_up": h.is_up}
            for name, h in self.monitor.hosts.items()
        }
        return ExecutorResult(success=True, data=uptime_data)

    def _list_hosts(self) -> ExecutorResult:
        hosts = [{
            "name": h.name, "ip": h.ip,
            "is_up": h.is_up,
            "uptime_percent": round(h.uptime_percent, 1),
        } for h in self.monitor.hosts.values()]
        return ExecutorResult(success=True, data=hosts)

    def _add_host(self, name: str, ip: str) -> ExecutorResult:
        self.monitor.add_host(name, ip)
        return ExecutorResult(success=True, data={"message": f"Added {name} ({ip}) to monitoring", "name": name, "ip": ip})

    def _remove_host(self, name: str) -> ExecutorResult:
        self.monitor.remove_host(name)
        return ExecutorResult(success=True, data={"message": f"Removed {name} from monitoring"})

    def _scan_network(self) -> ExecutorResult:
        try:
            devices = self.monitor.scan_network()
            return ExecutorResult(success=True, data={"devices_found": len(devices), "devices": devices})
        except Exception as e:
            return ExecutorResult(success=False, error=str(e))

    def _check_services(self, host: str = None) -> ExecutorResult:
        results = self.monitor.check_services(host)
        return ExecutorResult(success=True, data=results)

    def _get_all_devices(self) -> ExecutorResult:
        devices = self.monitor.get_all_devices()
        return ExecutorResult(success=True, data=devices)

    def _add_service(self, host: str, port: int, name: str = None) -> ExecutorResult:
        return ExecutorResult(success=True, data={
            "message": f"Service monitoring not fully implemented in lightweight adapter",
            "host": host, "port": port,
        })

    def _get_actions(self) -> list:
        return [
            {"name": "status", "params": []},
            {"name": "full_status", "params": []},
            {"name": "check", "params": ["host?"]},
            {"name": "alerts", "params": []},
            {"name": "uptime", "params": ["host?"]},
            {"name": "hosts", "params": []},
            {"name": "add_host", "params": ["name", "ip"]},
            {"name": "remove_host", "params": ["name"]},
            {"name": "scan", "params": []},
            {"name": "services", "params": ["host?"]},
            {"name": "all_devices", "params": []},
            {"name": "add_service", "params": ["host", "port", "name?"]},
        ]
