"""
Agent Registry
==============
Dynamic registration and lookup of agent types.

Ported from Server-Files agents/registry.py — direct port, no changes needed.

Usage:
    @AgentRegistry.register
    class MyAgent(BaseAgent):
        agent_type = "my_agent"
        capabilities = ["research", "analyze"]
"""

from typing import Dict, Type, List, Optional
from .base_agent import BaseAgent, AgentTask, ProjectContext
import os


class AgentRegistry:
    """Registry for agent types."""

    _agents: Dict[str, Type[BaseAgent]] = {}

    @classmethod
    def register(cls, agent_class: Type[BaseAgent]) -> Type[BaseAgent]:
        """
        Register an agent class. Can be used as a decorator.

        @AgentRegistry.register
        class MyAgent(BaseAgent):
            ...
        """
        cls._agents[agent_class.agent_type] = agent_class
        return agent_class

    @classmethod
    def unregister(cls, agent_type: str) -> bool:
        if agent_type in cls._agents:
            del cls._agents[agent_type]
            return True
        return False

    @classmethod
    def get_agent_class(cls, agent_type: str) -> Optional[Type[BaseAgent]]:
        return cls._agents.get(agent_type)

    @classmethod
    def get_agent_for_task(
        cls,
        task: AgentTask,
        context: ProjectContext = None,
        ollama_url: str = None,
        model: str = None,
    ) -> Optional[BaseAgent]:
        """Get an agent instance that can handle the given task type."""
        ollama_url = ollama_url or os.getenv("OLLAMA_API", "http://localhost:11434/api/generate")
        model = model or os.getenv("MODEL_NAME", "llama3:8b")

        for agent_class in cls._agents.values():
            if task.type in agent_class.capabilities:
                return agent_class(
                    ollama_url=ollama_url,
                    model=model,
                    context=context,
                )
        return None

    @classmethod
    def get_agents_for_capability(cls, capability: str) -> List[Type[BaseAgent]]:
        return [ac for ac in cls._agents.values() if capability in ac.capabilities]

    @classmethod
    def list_agents(cls) -> List[Dict]:
        return [
            {
                "type": ac.agent_type,
                "name": ac.name,
                "capabilities": ac.capabilities,
            }
            for ac in cls._agents.values()
        ]

    @classmethod
    def list_capabilities(cls) -> List[str]:
        caps = set()
        for ac in cls._agents.values():
            caps.update(ac.capabilities)
        return sorted(caps)

    @classmethod
    def clear(cls):
        """Clear all registered agents (for testing)."""
        cls._agents.clear()
