"""
Agent Base Classes
==================
Core infrastructure for the multi-agent system.

Ported from Server-Files agents/base.py with one key adaptation:
query_llm() uses requests directly (same as original) but is compatible
with Hugo's OllamaStabilityManager for future integration.

Original agents from this codebase (WorkerAgent, ClaudeBridge) are unchanged.
"""

import uuid
import time
import json
import os
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class TaskStatus(Enum):
    """Status of an agent task"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Types of tasks agents can handle"""
    RESEARCH = "research"
    EXPLORE = "explore"
    ANALYZE = "analyze"
    CODE = "code"
    MODIFY = "modify"
    CREATE = "create"
    REFACTOR = "refactor"
    TEST = "test"
    VALIDATE = "validate"
    VERIFY = "verify"
    WEB_SEARCH = "web_search"
    WEB_FETCH = "web_fetch"
    API_CALL = "api_call"
    DEEP_RESEARCH = "deep_research"
    DELEGATE = "delegate"
    SYNTHESIZE = "synthesize"


@dataclass
class AgentTask:
    """
    A task to be executed by an agent.

    Tasks can have dependencies — they won't start until all
    dependent tasks have completed.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str = "research"
    description: str = ""
    context: Dict = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    assigned_agent: Optional[str] = None
    parent_task_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "context": self.context,
            "dependencies": self.dependencies,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at,
            "assigned_agent": self.assigned_agent,
        }


@dataclass
class AgentResult:
    """Result from an agent's task execution."""
    task_id: str
    success: bool
    output: Any = None
    artifacts: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    follow_up_tasks: List[AgentTask] = field(default_factory=list)
    execution_time: float = 0.0
    agent_name: str = ""
    messages: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "output": self.output,
            "artifacts": self.artifacts,
            "errors": self.errors,
            "execution_time": self.execution_time,
            "agent_name": self.agent_name,
            "follow_up_count": len(self.follow_up_tasks),
        }


@dataclass
class ProjectContext:
    """Context for a project that agents are working on."""
    name: str = "unnamed"
    type: str = "external_code"
    base_path: str = ""
    language: str = "python"
    constraints: Dict = field(default_factory=dict)
    memory: Dict = field(default_factory=dict)
    files_modified: List[str] = field(default_factory=list)
    backups: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "base_path": self.base_path,
            "language": self.language,
            "constraints": self.constraints,
        }


@dataclass
class AgentMessage:
    """A message in an agent's conversation history"""
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Base class for all agents in the Hugo agent system.

    Agents use the local Ollama LLM to reason about tasks and generate outputs.
    Each agent type specializes in different capabilities.
    """

    name: str = "base_agent"
    agent_type: str = "base"
    capabilities: List[str] = []

    def __init__(
        self,
        ollama_url: str = None,
        model: str = None,
        context: ProjectContext = None,
    ):
        self.ollama_url = ollama_url or os.getenv("OLLAMA_API", "http://localhost:11434/api/generate")
        # Strip /api/generate to get base URL for /api/chat endpoint
        if self.ollama_url.endswith("/api/generate"):
            self.ollama_url = self.ollama_url[:-len("/api/generate")]
        self.model = model or os.getenv("MODEL_NAME", "llama3:8b")
        self.context = context or ProjectContext()
        self.messages: List[AgentMessage] = []
        self.is_running = False
        self.current_task: Optional[AgentTask] = None

    def can_handle(self, task: AgentTask) -> bool:
        return task.type in self.capabilities

    def query_llm(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> str:
        """Query the local Ollama LLM."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for msg in self.messages[-10:]:
                messages.append({"role": msg.role, "content": msg.content})
            messages.append({"role": "user", "content": prompt})

            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=120,
            )

            if response.status_code == 200:
                content = response.json().get("message", {}).get("content", "")
                self.messages.append(AgentMessage(role="user", content=prompt))
                self.messages.append(AgentMessage(role="assistant", content=content))
                return content
            return f"LLM Error: {response.status_code}"
        except Exception as e:
            return f"LLM Query Failed: {str(e)}"

    def parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        try:
            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                text = "\n".join(lines)
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    @abstractmethod
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute a task and return the result."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent type."""
        pass

    def log(self, message: str, level: str = "info"):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{self.name}] [{level.upper()}] {message}")

    def create_artifact(self, type: str, name: str, content: Any, metadata: Dict = None) -> Dict:
        return {
            "type": type,
            "name": name,
            "content": content,
            "metadata": metadata or {},
            "created_by": self.name,
            "created_at": time.time(),
        }
