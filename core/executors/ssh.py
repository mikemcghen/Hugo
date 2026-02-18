"""
SSH Executor
------------
Runs commands on remote hosts via SSH.

Hosts are configured via configs/ssh_hosts.yaml rather than hardcoded IPs,
making this portable across environments.

Ported from Server-Files executors/ssh.py.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

from .base import BaseExecutor, ExecutorResult
from .executor_registry import ExecutorRegistry


def _load_hosts_from_yaml() -> Dict:
    """Load SSH hosts from configs/ssh_hosts.yaml if it exists.

    Supports two YAML formats:
      List (preferred):  hosts: [{name: foo, ip: 1.2.3.4, user: root, ...}]
      Dict (legacy):     hosts: {foo: {ip: 1.2.3.4, user: root, ...}}
    """
    try:
        import yaml
        config_path = Path(__file__).parent.parent.parent / "configs" / "ssh_hosts.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            if not data:
                return {}
            hosts_raw = data.get("hosts", {})
            # Convert list-of-dicts [{name: foo, ip: ...}] → {foo: {ip: ...}}
            if isinstance(hosts_raw, list):
                result = {}
                for entry in hosts_raw:
                    if isinstance(entry, dict) and "name" in entry:
                        entry = dict(entry)  # copy so we don't mutate yaml parse
                        name = entry.pop("name")
                        result[name] = entry
                return result
            return hosts_raw if isinstance(hosts_raw, dict) else {}
    except Exception:
        pass
    return {}


@ExecutorRegistry.register
class SSHExecutor(BaseExecutor):
    """Execute commands on remote hosts via SSH."""

    name = "ssh"
    description = "Execute commands on remote hosts via SSH"

    def __init__(self):
        super().__init__()

        # Load hosts from config, fall back to env-defined defaults
        self.hosts = _load_hosts_from_yaml()

        # Allow single host override via env vars for simple setups
        if not self.hosts:
            default_host = os.getenv("SSH_DEFAULT_HOST")
            default_ip = os.getenv("SSH_DEFAULT_IP")
            if default_host and default_ip:
                self.hosts[default_host] = {
                    "ip": default_ip,
                    "user": os.getenv("SSH_DEFAULT_USER", "root"),
                    "key": os.getenv("SSH_DEFAULT_KEY", "~/.ssh/id_rsa"),
                    "description": "Default SSH host from env",
                }

    def add_host(self, name: str, ip: str, user: str = "root",
                 key: str = "~/.ssh/id_rsa", description: str = "") -> bool:
        """Register a new host."""
        self.hosts[name] = {
            "ip": ip, "user": user, "key": key, "description": description
        }
        return True

    def remove_host(self, name: str) -> bool:
        if name in self.hosts:
            del self.hosts[name]
            return True
        return False

    def get_host(self, name: str) -> Optional[Dict]:
        """Get host info by name or IP."""
        if name in self.hosts:
            return self.hosts[name]
        for host_name, info in self.hosts.items():
            if info.get("ip") == name:
                return info
        return None

    def _execute(self, action: str, **params) -> ExecutorResult:
        if action == "run":
            return self._run_command(
                host=params.get("host"),
                command=params.get("command"),
                timeout=params.get("timeout", 30),
            )
        elif action == "test":
            return self._test_connection(host=params.get("host"))
        elif action == "list_hosts":
            return ExecutorResult(success=True, data=self.hosts)
        elif action == "add_host":
            success = self.add_host(
                name=params.get("name", ""),
                ip=params.get("ip", ""),
                user=params.get("user", "root"),
                key=params.get("key", "~/.ssh/id_rsa"),
                description=params.get("description", ""),
            )
            return ExecutorResult(success=success, data=f"Added host {params.get('name')}")
        elif action == "remove_host":
            success = self.remove_host(params.get("name", ""))
            return ExecutorResult(success=success, data=f"Removed host {params.get('name')}")
        else:
            return ExecutorResult(success=False, error=f"Unknown SSH action: {action}")

    def _run_command(self, host: str, command: str, timeout: int = 30) -> ExecutorResult:
        """Run a command on a remote host."""
        if not host:
            return ExecutorResult(success=False, error="No host specified")
        if not command:
            return ExecutorResult(success=False, error="No command specified")

        host_info = self.get_host(host)
        if not host_info:
            known = list(self.hosts.keys())
            return ExecutorResult(
                success=False,
                error=f"Unknown host: {host!r}. Known hosts: {known}"
            )

        safe, reason = self.is_safe(command)
        if not safe:
            return ExecutorResult(success=False, error=reason)

        ssh_cmd = [
            "ssh",
            "-i", host_info["key"],
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            f"{host_info['user']}@{host_info['ip']}",
            command,
        ]

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return ExecutorResult(
                    success=True,
                    data={
                        "stdout": result.stdout.strip(),
                        "stderr": result.stderr.strip(),
                        "exit_code": result.returncode,
                    },
                )
            else:
                return ExecutorResult(
                    success=False,
                    error=f"Command failed (exit {result.returncode}): {result.stderr.strip()}",
                    data={
                        "stdout": result.stdout.strip(),
                        "stderr": result.stderr.strip(),
                        "exit_code": result.returncode,
                    },
                )
        except subprocess.TimeoutExpired:
            return ExecutorResult(success=False, error=f"Command timed out after {timeout}s")
        except Exception as e:
            return ExecutorResult(success=False, error=f"SSH error: {str(e)}")

    def _test_connection(self, host: str) -> ExecutorResult:
        return self._run_command(host, "echo 'connection_ok'", timeout=10)

    def _get_actions(self) -> list:
        return [
            {"name": "run", "params": ["host", "command", "timeout?"]},
            {"name": "test", "params": ["host"]},
            {"name": "list_hosts", "params": []},
            {"name": "add_host", "params": ["name", "ip", "user?", "key?", "description?"]},
            {"name": "remove_host", "params": ["name"]},
        ]
