"""
Delegation Agent
================
The project manager agent that coordinates helper agents for complex tasks.

Ported from Server-Files agents/delegation.py with one adaptation:
_execute_parallel() uses asyncio-compatible ThreadPoolExecutor wrapping
since Hugo's runtime is fully async.

Register with @AgentRegistry.register at module bottom (same as original).
"""

import time
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .base_agent import (
    BaseAgent, AgentTask, AgentResult, ProjectContext,
    TaskStatus, TaskType,
)
from .registry import AgentRegistry


@dataclass
class TaskNode:
    """A node in the task dependency graph"""
    task: AgentTask
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    result: Optional[AgentResult] = None


@AgentRegistry.register
class DelegationAgent(BaseAgent):
    """
    The coordinator agent that manages complex multi-step tasks.

    Given a high-level task description, it:
    1. Analyses and breaks the task into subtasks
    2. Builds a dependency graph
    3. Executes independent subtasks in parallel
    4. Passes outputs from completed tasks to dependent tasks
    5. Synthesizes all results into a final report
    """

    name = "delegation_agent"
    agent_type = "delegation"
    capabilities = ["delegate", "synthesize", "coordinate"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_graph: Dict[str, TaskNode] = {}
        self.completed_tasks: Dict[str, AgentResult] = {}
        self.running_tasks: Dict[str, AgentTask] = {}
        self.max_parallel = 3

    def get_system_prompt(self) -> str:
        project_info = ""
        if self.context:
            project_info = f"""
Working on project: {self.context.name}
Type: {self.context.type}
Path: {self.context.base_path}
"""
        return f"""You are a Delegation Agent — a project manager for complex tasks.
{project_info}
Your role is to:
1. Analyze complex tasks and break them into subtasks
2. Identify dependencies between subtasks
3. Coordinate multiple helper agents
4. Ensure work is done in the right order
5. Synthesize results into a coherent output

Available helper agents:
- ResearchAgent: Explores codebases, finds patterns, analyzes structure
- WebAgent: Internet research, API calls, documentation lookup
- CoderAgent: Writes and modifies code
- TesterAgent: Validates code, runs tests

Output structured plans in JSON format."""

    def execute(self, task: AgentTask) -> AgentResult:
        start_time = time.time()
        self.current_task = task
        self.log(f"Starting delegation for: {task.description}")

        try:
            plan = self._create_plan(task)
            if not plan["success"]:
                return AgentResult(
                    task_id=task.id,
                    success=False,
                    errors=plan.get("errors", ["Planning failed"]),
                    execution_time=time.time() - start_time,
                    agent_name=self.name,
                )

            self.log(f"Created plan with {len(plan['tasks'])} subtasks")
            self._build_task_graph(plan["tasks"])
            results = self._execute_plan()
            synthesis = self._synthesize_results(task, results)

            return AgentResult(
                task_id=task.id,
                success=synthesis["success"],
                output=synthesis,
                artifacts=synthesis.get("artifacts", []),
                errors=synthesis.get("errors", []),
                execution_time=time.time() - start_time,
                agent_name=self.name,
            )

        except Exception as e:
            self.log(f"Delegation failed: {e}", "error")
            return AgentResult(
                task_id=task.id,
                success=False,
                errors=[str(e)],
                execution_time=time.time() - start_time,
                agent_name=self.name,
            )

    def _create_plan(self, task: AgentTask) -> Dict:
        context_info = ""
        if self.context.base_path:
            context_info = f"\nProject path: {self.context.base_path}\nProject type: {self.context.type}"

        prompt = f"""Create an execution plan for this task:

TASK: {task.description}
{context_info}

Break this into subtasks, each assigned to a helper agent.
Consider what can run in parallel.

Available task types:
- research: Explore codebase, find patterns (ResearchAgent)
- explore: Discover project structure (ResearchAgent)
- analyze: Analyze code for patterns (ResearchAgent)
- web_search: Search internet (WebAgent)
- deep_research: Comprehensive web research (WebAgent)
- code: Generate new code (CoderAgent)
- modify: Modify existing code (CoderAgent)
- create: Create new files (CoderAgent)
- refactor: Refactor code (CoderAgent)
- validate: Check code syntax (TesterAgent)
- test: Run tests (TesterAgent)
- verify: Verify changes work (TesterAgent)

Output a plan in JSON format:
{{
    "success": true,
    "summary": "Brief plan summary",
    "tasks": [
        {{
            "id": "task1",
            "type": "research",
            "description": "What this task does",
            "context": {{}},
            "dependencies": [],
            "priority": 1
        }}
    ]
}}"""

        response = self.query_llm(prompt, self.get_system_prompt())
        plan = self.parse_json_response(response)

        if not plan:
            return {"success": False, "errors": ["Could not parse plan from LLM response"], "raw_response": response}

        if "tasks" not in plan or not plan["tasks"]:
            return {"success": False, "errors": ["No tasks in plan"]}

        normalized_tasks = []
        for task_data in plan["tasks"]:
            agent_task = AgentTask(
                id=task_data.get("id", str(uuid.uuid4())[:8]),
                type=task_data.get("type", "research"),
                description=task_data.get("description", ""),
                context=task_data.get("context", {}),
                dependencies=task_data.get("dependencies", []),
                priority=task_data.get("priority", 1),
                parent_task_id=self.current_task.id if self.current_task else None,
            )
            normalized_tasks.append(agent_task)

        return {"success": True, "summary": plan.get("summary", ""), "tasks": normalized_tasks}

    def _build_task_graph(self, tasks: List[AgentTask]):
        self.task_graph.clear()
        for task in tasks:
            self.task_graph[task.id] = TaskNode(task=task, dependencies=task.dependencies.copy())
        for task_id, node in self.task_graph.items():
            for dep_id in node.dependencies:
                if dep_id in self.task_graph:
                    self.task_graph[dep_id].dependents.append(task_id)

    def _get_ready_tasks(self) -> List[AgentTask]:
        ready = []
        for task_id, node in self.task_graph.items():
            if node.task.status != TaskStatus.PENDING:
                continue
            deps_satisfied = all(
                self.task_graph[dep_id].task.status == TaskStatus.COMPLETED
                for dep_id in node.dependencies
                if dep_id in self.task_graph
            )
            if deps_satisfied:
                ready.append(node.task)
        ready.sort(key=lambda t: t.priority)
        return ready

    def _execute_plan(self) -> Dict[str, AgentResult]:
        all_results: Dict[str, AgentResult] = {}
        while True:
            ready_tasks = self._get_ready_tasks()
            if not ready_tasks:
                pending = [n for n in self.task_graph.values() if n.task.status == TaskStatus.PENDING]
                if not pending:
                    break
                else:
                    self.log("Warning: Tasks pending but none ready — possible dependency issue", "warning")
                    break

            batch = ready_tasks[:self.max_parallel]
            batch_results = self._execute_parallel(batch)
            all_results.update(batch_results)

            for task_id, result in batch_results.items():
                if task_id in self.task_graph:
                    node = self.task_graph[task_id]
                    node.result = result
                    node.task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED

                    if result.success and result.output:
                        for dep_id in node.dependents:
                            if dep_id in self.task_graph:
                                dep_node = self.task_graph[dep_id]
                                if "predecessor_outputs" not in dep_node.task.context:
                                    dep_node.task.context["predecessor_outputs"] = {}
                                dep_node.task.context["predecessor_outputs"][task_id] = result.output

        return all_results

    def _execute_parallel(self, tasks: List[AgentTask]) -> Dict[str, AgentResult]:
        results = {}
        if len(tasks) == 1:
            task = tasks[0]
            results[task.id] = self._execute_single(task)
        else:
            with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
                future_to_task = {executor.submit(self._execute_single, task): task for task in tasks}
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        results[task.id] = future.result()
                    except Exception as e:
                        results[task.id] = AgentResult(task_id=task.id, success=False, errors=[str(e)])
        return results

    def _execute_single(self, task: AgentTask) -> AgentResult:
        self.log(f"Executing task {task.id}: {task.description[:50]}...")
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        agent = AgentRegistry.get_agent_for_task(
            task, context=self.context, ollama_url=self.ollama_url, model=self.model
        )

        if not agent:
            return AgentResult(
                task_id=task.id,
                success=False,
                errors=[f"No agent found for task type: {task.type}"]
            )

        result = agent.execute(task)
        task.completed_at = time.time()

        if result.success:
            self.log(f"Task {task.id} completed successfully")
        else:
            self.log(f"Task {task.id} failed: {result.errors}", "error")

        return result

    def _synthesize_results(self, original_task: AgentTask, results: Dict[str, AgentResult]) -> Dict:
        all_outputs = {}
        all_artifacts = []
        all_errors = []
        success_count = fail_count = 0

        for task_id, result in results.items():
            if result.success:
                success_count += 1
                if result.output:
                    all_outputs[task_id] = result.output
                if result.artifacts:
                    all_artifacts.extend(result.artifacts)
            else:
                fail_count += 1
                all_errors.extend(result.errors)

        overall_success = fail_count == 0

        summary = ""
        if all_outputs:
            prompt = f"""Summarize the results of this task execution:

ORIGINAL TASK: {original_task.description}

TASK RESULTS:
{json.dumps(all_outputs, indent=2, default=str)[:4000]}

Create a clear summary of:
1. What was accomplished
2. Key findings or outputs
3. Any issues encountered

Keep it concise but informative."""
            summary = self.query_llm(prompt, self.get_system_prompt())

        return {
            "success": overall_success,
            "summary": summary,
            "tasks_completed": success_count,
            "tasks_failed": fail_count,
            "outputs": all_outputs,
            "artifacts": all_artifacts,
            "errors": all_errors if all_errors else None,
        }

    def get_progress(self) -> Dict:
        total = len(self.task_graph)
        completed = sum(1 for n in self.task_graph.values() if n.task.status == TaskStatus.COMPLETED)
        running = sum(1 for n in self.task_graph.values() if n.task.status == TaskStatus.RUNNING)
        failed = sum(1 for n in self.task_graph.values() if n.task.status == TaskStatus.FAILED)
        pending = total - completed - running - failed
        return {
            "total": total,
            "completed": completed,
            "running": running,
            "failed": failed,
            "pending": pending,
            "percent_complete": round(completed / total * 100, 1) if total > 0 else 0,
        }
