"""
Docker Executor
---------------
Manages Docker containers on a remote host via SSH.

Docker host is configurable via DOCKER_HOST_NAME env var (default: first SSH host).

Ported from Server-Files executors/docker.py.
"""

import os
from typing import Dict, List

from .base import BaseExecutor, ExecutorResult
from .executor_registry import ExecutorRegistry


@ExecutorRegistry.register
class DockerExecutor(BaseExecutor):
    """Manage Docker containers via SSH."""

    name = "docker"
    description = "Manage Docker containers on a remote host"

    def __init__(self):
        super().__init__()
        # Import here to avoid circular import at module level
        from .ssh import SSHExecutor
        self.ssh = SSHExecutor()
        self.docker_host = os.getenv("DOCKER_HOST_NAME", "")

        # If no host configured, use first available SSH host
        if not self.docker_host and self.ssh.hosts:
            self.docker_host = next(iter(self.ssh.hosts))

        # Known container groups (can be extended at runtime)
        self.container_groups: Dict[str, List[str]] = {}

    def _execute(self, action: str, **params) -> ExecutorResult:
        container = params.get("container")
        host = params.get("host", self.docker_host)

        if not host:
            return ExecutorResult(
                success=False,
                error="No Docker host configured. Set DOCKER_HOST_NAME env var or add SSH hosts."
            )

        if action == "list":
            return self._list_containers(host, all_containers=params.get("all", False))
        elif action == "status":
            return self._get_status(host, container)
        elif action == "start":
            if not container:
                return ExecutorResult(success=False, error="Container name required")
            return self._start_container(host, container)
        elif action == "stop":
            if not container:
                return ExecutorResult(success=False, error="Container name required")
            return self._stop_container(host, container)
        elif action == "restart":
            if not container:
                return ExecutorResult(success=False, error="Container name required")
            return self._restart_container(host, container)
        elif action == "logs":
            if not container:
                return ExecutorResult(success=False, error="Container name required")
            lines = params.get("lines", 50)
            return self._get_logs(host, container, lines)
        elif action == "start_group":
            group = params.get("group")
            if not group or group not in self.container_groups:
                return ExecutorResult(success=False, error=f"Unknown container group: {group!r}")
            return self._start_group(host, group)
        elif action == "stop_group":
            group = params.get("group")
            if not group or group not in self.container_groups:
                return ExecutorResult(success=False, error=f"Unknown container group: {group!r}")
            return self._stop_group(host, group)
        else:
            return ExecutorResult(success=False, error=f"Unknown Docker action: {action}")

    def _list_containers(self, host: str, all_containers: bool = False) -> ExecutorResult:
        flag = "-a" if all_containers else ""
        cmd = f"docker ps {flag} --format '{{{{.Names}}}}|{{{{.Status}}}}|{{{{.Image}}}}'"
        result = self.ssh.execute("run", host=host, command=cmd)
        if result.success:
            containers = []
            for line in result.data["stdout"].split("\n"):
                if "|" in line:
                    parts = line.split("|")
                    containers.append({
                        "name": parts[0],
                        "status": parts[1] if len(parts) > 1 else "",
                        "image": parts[2] if len(parts) > 2 else "",
                    })
            return ExecutorResult(success=True, data=containers)
        return result

    def _get_status(self, host: str, container: str) -> ExecutorResult:
        if not container:
            return self._list_containers(host)
        cmd = f"docker inspect --format '{{{{.State.Status}}}}' {container}"
        result = self.ssh.execute("run", host=host, command=cmd)
        if result.success:
            status = result.data["stdout"].strip()
            return ExecutorResult(success=True, data={"container": container, "status": status})
        return result

    def _start_container(self, host: str, container: str) -> ExecutorResult:
        result = self.ssh.execute("run", host=host, command=f"docker start {container}")
        if result.success:
            return ExecutorResult(success=True, data=f"Container {container!r} started")
        return result

    def _stop_container(self, host: str, container: str) -> ExecutorResult:
        result = self.ssh.execute("run", host=host, command=f"docker stop {container}")
        if result.success:
            return ExecutorResult(success=True, data=f"Container {container!r} stopped")
        return result

    def _restart_container(self, host: str, container: str) -> ExecutorResult:
        result = self.ssh.execute("run", host=host, command=f"docker restart {container}")
        if result.success:
            return ExecutorResult(success=True, data=f"Container {container!r} restarted")
        return result

    def _get_logs(self, host: str, container: str, lines: int = 50) -> ExecutorResult:
        result = self.ssh.execute(
            "run", host=host, command=f"docker logs --tail {lines} {container}"
        )
        if result.success:
            return ExecutorResult(success=True, data={
                "container": container,
                "logs": result.data.get("stdout", "") + result.data.get("stderr", ""),
            })
        return result

    def _start_group(self, host: str, group: str) -> ExecutorResult:
        containers = self.container_groups[group]
        results = []
        for c in containers:
            r = self._start_container(host, c)
            results.append({"container": c, "success": r.success})
        return ExecutorResult(
            success=all(r["success"] for r in results),
            data={"group": group, "results": results},
        )

    def _stop_group(self, host: str, group: str) -> ExecutorResult:
        containers = list(reversed(self.container_groups[group]))
        results = []
        for c in containers:
            r = self._stop_container(host, c)
            results.append({"container": c, "success": r.success})
        return ExecutorResult(
            success=all(r["success"] for r in results),
            data={"group": group, "results": results},
        )

    def _get_actions(self) -> list:
        return [
            {"name": "list", "params": ["host?", "all?"]},
            {"name": "status", "params": ["container?", "host?"]},
            {"name": "start", "params": ["container", "host?"]},
            {"name": "stop", "params": ["container", "host?"]},
            {"name": "restart", "params": ["container", "host?"]},
            {"name": "logs", "params": ["container", "lines?", "host?"]},
            {"name": "start_group", "params": ["group", "host?"]},
            {"name": "stop_group", "params": ["group", "host?"]},
        ]
