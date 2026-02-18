"""
Network Monitor Adapter
-----------------------
Lightweight replacement for Server-Files network_monitor.py.

Instead of pulling in the full Server-Files dependency chain, this adapter
provides the interface MonitorExecutor needs using subprocess ping and
a simple JSON state file.

State persisted at: data/network_monitor_state.json
"""

import json
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


_STATE_FILE = Path(__file__).parent.parent.parent / "data" / "network_monitor_state.json"
_SINGLETON: Optional["NetworkMonitorAdapter"] = None


@dataclass
class HostStatus:
    name: str
    ip: str
    is_up: bool = False
    latency_ms: float = 0.0
    total_checks: int = 0
    total_up: int = 0
    consecutive_failures: int = 0
    last_checked: float = 0.0

    @property
    def uptime_percent(self) -> float:
        if self.total_checks == 0:
            return 0.0
        return (self.total_up / self.total_checks) * 100.0


@dataclass
class Alert:
    host: str
    alert_type: str
    message: str
    timestamp: float = field(default_factory=time.time)


class NetworkMonitorAdapter:
    """
    Minimal network monitor backed by ping + JSON state.

    Provides the same interface as MonitorExecutor expects.
    """

    def __init__(self):
        self.hosts: Dict[str, HostStatus] = {}
        self._alerts: List[Alert] = []
        self._load_state()

    def _load_state(self):
        try:
            if _STATE_FILE.exists():
                data = json.loads(_STATE_FILE.read_text())
                for name, info in data.get("hosts", {}).items():
                    self.hosts[name] = HostStatus(**info)
        except Exception:
            pass

    def _save_state(self):
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            state = {"hosts": {n: asdict(h) for n, h in self.hosts.items()}}
            _STATE_FILE.write_text(json.dumps(state, indent=2))
        except Exception:
            pass

    def _ping(self, ip: str, count: int = 1, timeout: int = 2) -> tuple[bool, float]:
        """Return (is_up, latency_ms)."""
        try:
            result = subprocess.run(
                ["ping", "-n" if _is_windows() else "-c", str(count),
                 "-w" if _is_windows() else "-W", str(timeout), ip],
                capture_output=True, text=True, timeout=timeout + 2
            )
            if result.returncode == 0:
                # Try to extract latency from output
                for line in result.stdout.splitlines():
                    if "time=" in line.lower() or "Average" in line:
                        import re
                        m = re.search(r"time[<=](\d+\.?\d*)", line, re.IGNORECASE)
                        if m:
                            return True, float(m.group(1))
                return True, 0.0
            return False, 0.0
        except Exception:
            return False, 0.0

    def add_host(self, name: str, ip: str):
        self.hosts[name] = HostStatus(name=name, ip=ip)
        self._save_state()

    def remove_host(self, name: str):
        self.hosts.pop(name, None)
        self._save_state()

    def check_host(self, name: str) -> HostStatus:
        if name not in self.hosts:
            raise ValueError(f"Unknown host: {name}")
        status = self.hosts[name]
        is_up, latency = self._ping(status.ip)
        status.total_checks += 1
        status.last_checked = time.time()
        if is_up:
            status.is_up = True
            status.latency_ms = latency
            status.total_up += 1
            status.consecutive_failures = 0
        else:
            status.is_up = False
            status.consecutive_failures += 1
            if status.consecutive_failures >= 3:
                self._alerts.append(Alert(
                    host=name,
                    alert_type="host_down",
                    message=f"{name} ({status.ip}) is unreachable"
                ))
        self._save_state()
        return status

    def check_all(self):
        for name in list(self.hosts.keys()):
            self.check_host(name)

    def get_status_summary(self) -> dict:
        up = sum(1 for h in self.hosts.values() if h.is_up)
        total = len(self.hosts)
        return {
            "hosts_up": up,
            "hosts_down": total - up,
            "total_hosts": total,
            "hosts": {
                n: {"ip": h.ip, "is_up": h.is_up, "latency_ms": h.latency_ms,
                    "uptime_percent": round(h.uptime_percent, 1)}
                for n, h in self.hosts.items()
            }
        }

    def get_full_network_status(self) -> dict:
        self.check_all()
        return self.get_status_summary()

    def get_alerts(self, unacknowledged_only: bool = False) -> List[Alert]:
        return self._alerts

    def scan_network(self) -> list:
        """Basic network scan — not implemented in lightweight adapter."""
        return []

    def check_services(self, host: str = None) -> dict:
        return {"message": "Service port checking not implemented in lightweight adapter"}

    def get_all_devices(self) -> list:
        return [{"name": n, "ip": h.ip, "is_up": h.is_up} for n, h in self.hosts.items()]


def _is_windows() -> bool:
    import sys
    return sys.platform.startswith("win")


def get_monitor() -> NetworkMonitorAdapter:
    """Get or create the singleton monitor instance."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = NetworkMonitorAdapter()
    return _SINGLETON
