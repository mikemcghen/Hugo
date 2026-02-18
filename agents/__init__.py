"""
Agent System
------------
Sub-agent system for delegating complex tasks.

Components (original):
- WorkerAgent: Handles delegated technical tasks
- ClaudeBridge: Interface for requesting external assistance

Components (from Server-Files integration):
- BaseAgent: Base class for all agents
- AgentTask, AgentResult: Task/result dataclasses
- TaskStatus, TaskType: Enums for task lifecycle
- ProjectContext: Context for agent work
- AgentRegistry: Dynamic agent registration (@AgentRegistry.register)
- DelegationAgent: Breaks complex tasks into parallel subtasks
"""

from .worker_agent import WorkerAgent
from .claude_bridge import ClaudeBridge

# New agent system (Server-Files integration)
from .base_agent import BaseAgent, AgentTask, AgentResult, TaskStatus, TaskType, ProjectContext
from .registry import AgentRegistry
from .delegation_agent import DelegationAgent  # auto-registers with AgentRegistry on import

__all__ = [
    # Original
    "WorkerAgent", "ClaudeBridge",
    # New
    "BaseAgent", "AgentTask", "AgentResult", "TaskStatus", "TaskType", "ProjectContext",
    "AgentRegistry", "DelegationAgent",
]
