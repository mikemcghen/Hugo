"""
Skill Handlers
--------------
Execute skill logic for all supported skills.

Original handlers (search, note, fetch, scrape) retain placeholder
implementations — real integrations can be wired in later.

New infrastructure handlers (ssh, docker, monitor) delegate to the
executor layer and respect the PermissionGate.

Args format for infrastructure skills:
  /ssh <host> <command>       — e.g. /ssh proxmox df -h
  /docker <action> [target]   — e.g. /docker list  /docker restart ollama
  /monitor <action> [host]    — e.g. /monitor status  /monitor check proxmox
"""

from dataclasses import dataclass


@dataclass
class SkillResult:
    """Result from a skill handler"""
    success: bool
    output: str


async def handle_search(args: str) -> SkillResult:
    """
    Handle search skill.

    Args:
        args: Search query

    Returns:
        SkillResult with placeholder output

    Note:
        This is a placeholder implementation.
        Real implementation would perform actual search.
    """
    return SkillResult(
        success=True,
        output="[Placeholder: Search results]"
    )


async def handle_note(args: str) -> SkillResult:
    """
    Handle note skill.

    Args:
        args: Note content

    Returns:
        SkillResult with placeholder output

    Note:
        This is a placeholder implementation.
        Real implementation would save note to storage.
    """
    return SkillResult(
        success=True,
        output="[Placeholder: Note saved]"
    )


async def handle_fetch(args: str) -> SkillResult:
    """
    Handle fetch skill.

    Args:
        args: URL to fetch

    Returns:
        SkillResult with placeholder output

    Note:
        This is a placeholder implementation.
        Real implementation would fetch URL content.
    """
    return SkillResult(
        success=True,
        output="[Placeholder: URL content fetched]"
    )


async def handle_scrape(args: str) -> SkillResult:
    """
    Handle scrape skill.

    Args:
        args: URL to scrape

    Returns:
        SkillResult with placeholder output

    Note:
        This is a placeholder implementation.
        Real implementation would scrape webpage.
    """
    return SkillResult(
        success=True,
        output="[Placeholder: Page scraped]"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Infrastructure handlers — delegate to the executor + permission layer
# ──────────────────────────────────────────────────────────────────────────────

async def handle_ssh(args: str) -> SkillResult:
    """
    Handle /ssh skill.

    Args format: "<host> <command>"
    Example: /ssh proxmox df -h

    Checks PermissionGate before executing — SSH run requires confirmation.
    """
    from core.executors.ssh import SSHExecutor
    from core.actions.permission import PermissionGate

    parts = args.strip().split(" ", 1)
    host = parts[0] if parts else ""
    command = parts[1] if len(parts) > 1 else ""

    if not host:
        return SkillResult(success=False, output="Usage: /ssh <host> <command>")

    gate = PermissionGate()
    level = gate.check("ssh", "run")
    if gate.requires_confirmation(level):
        prompt = gate.confirmation_prompt("ssh", "run", host)
        return SkillResult(
            success=False,
            output=f"Confirmation required: {prompt} Reply 'yes' to proceed."
        )

    executor = SSHExecutor()
    result = executor.execute("run", host=host, command=command)
    if result.success:
        stdout = result.data.get("stdout", "") if isinstance(result.data, dict) else str(result.data)
        return SkillResult(success=True, output=stdout or "(No output)")
    return SkillResult(success=False, output=f"SSH error: {result.error}")


async def handle_docker(args: str) -> SkillResult:
    """
    Handle /docker skill.

    Args format: "<action> [container]"
    Examples:
        /docker list
        /docker restart ollama
        /docker logs ollama
        /docker status
    """
    from core.executors.docker import DockerExecutor
    from core.actions.permission import PermissionGate
    from core.actions.action_router import ActionRouter

    parts = args.strip().split(" ", 1)
    action = parts[0].lower() if parts else "list"
    container = parts[1].strip() if len(parts) > 1 else None

    gate = PermissionGate()
    level = gate.check("docker", action)
    if gate.requires_confirmation(level):
        prompt = gate.confirmation_prompt("docker", action, container)
        return SkillResult(
            success=False,
            output=f"Confirmation required: {prompt} Reply 'yes' to proceed."
        )

    executor = DockerExecutor()
    kwargs = {}
    if container:
        kwargs["container"] = container

    result = executor.execute(action, **kwargs)

    if result.success:
        # Format nicely using ActionRouter's formatter
        from core.actions.action_result import ActionResult
        ar = ActionResult(success=True, data=result.data, domain="docker", action=action)
        router = ActionRouter()
        return SkillResult(success=True, output=router.format_for_cognition(ar))

    return SkillResult(success=False, output=f"Docker error: {result.error}")


async def handle_monitor(args: str) -> SkillResult:
    """
    Handle /monitor skill.

    Args format: "<action> [host]"
    Examples:
        /monitor status
        /monitor check proxmox
        /monitor alerts
        /monitor hosts
    """
    from core.executors.monitor import MonitorExecutor
    from core.actions.action_result import ActionResult
    from core.actions.action_router import ActionRouter

    parts = args.strip().split(" ", 1)
    action = parts[0].lower() if parts and parts[0] else "status"
    host = parts[1].strip() if len(parts) > 1 else None

    executor = MonitorExecutor()
    kwargs = {}
    if host:
        kwargs["host"] = host

    result = executor.execute(action, **kwargs)

    if result.success:
        ar = ActionResult(success=True, data=result.data, domain="monitor", action=action)
        router = ActionRouter()
        return SkillResult(success=True, output=router.format_for_cognition(ar))

    return SkillResult(success=False, output=f"Monitor error: {result.error}")
