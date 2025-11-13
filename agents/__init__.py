"""
Agent Sandbox
-------------
Sub-agent system for delegating complex tasks.

This module provides a framework for Hugo to delegate tasks to specialized
worker agents while maintaining oversight and control.

Components:
- WorkerAgent: Handles delegated technical tasks
- ClaudeBridge: Interface for requesting external assistance (future: Claude Code integration)

Status: Sandbox mode - no self-modification capabilities enabled
"""

from .worker_agent import WorkerAgent
from .claude_bridge import ClaudeBridge

__all__ = ["WorkerAgent", "ClaudeBridge"]
