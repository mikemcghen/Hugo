"""
Worker Agent
------------
Specialized agent for handling delegated technical tasks.

The WorkerAgent receives task instructions from Hugo's cognition engine
and executes them with enhanced focus and tooling.

Current Capabilities:
- Task planning and decomposition
- Technical analysis and research
- Code generation (sandbox mode - no file writes)
- Documentation generation

Future Capabilities:
- Tool access (web search, code execution, file operations)
- Multi-step planning with checkpoints
- Collaboration with Claude Code via ClaudeBridge
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime


class WorkerAgent:
    """
    Specialized worker agent for complex technical tasks.

    The WorkerAgent operates in sandbox mode, meaning it can:
    - Generate code and documentation
    - Perform analysis and research
    - Create detailed plans

    But it CANNOT:
    - Modify files directly
    - Execute arbitrary code
    - Access external resources without approval
    """

    def __init__(self, hugo_runtime):
        """
        Initialize worker agent.

        Args:
            hugo_runtime: Reference to Hugo's runtime manager for access
                         to cognition, memory, and logging systems
        """
        self.hugo = hugo_runtime
        self.logger = hugo_runtime.logger if hasattr(hugo_runtime, 'logger') else None
        self.cognition = hugo_runtime.cognition if hasattr(hugo_runtime, 'cognition') else None

        # Task history
        self.task_history: List[Dict[str, Any]] = []

        # Log agent initialization
        if self.logger:
            self.logger.log_event("worker_agent", "initialized", {
                "mode": "sandbox",
                "capabilities": ["planning", "analysis", "code_generation"]
            })

    async def run_task(self, instruction: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute a delegated task with enhanced focus.

        This method decomposes the task, plans execution, and generates
        a detailed response using Hugo's cognition engine.

        Args:
            instruction: Task instruction from user or Hugo
            context: Optional context dictionary with additional information

        Returns:
            str: Task completion response
        """
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if self.logger:
            self.logger.log_event("worker_agent", "task_started", {
                "task_id": task_id,
                "instruction_length": len(instruction),
                "has_context": context is not None
            })

        try:
            # Step 1: Decompose task
            task_plan = await self._plan_task(instruction, context)

            # Step 2: Execute task using cognition engine
            # For now, we simply use Hugo's cognition with a specialized prompt
            if self.cognition:
                enhanced_prompt = self._build_task_prompt(instruction, task_plan, context)

                # Use non-streaming inference for worker tasks
                # (Worker tasks are background operations)
                response_package = await self.cognition.process_input(
                    enhanced_prompt,
                    session_id=task_id
                )

                result = response_package.content
            else:
                result = self._fallback_task_execution(instruction, task_plan)

            # Step 3: Log completion
            if self.logger:
                self.logger.log_event("worker_agent", "task_completed", {
                    "task_id": task_id,
                    "result_length": len(result),
                    "success": True
                })

            # Add to task history
            self.task_history.append({
                "task_id": task_id,
                "instruction": instruction,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "success": True
            })

            return result

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {
                    "phase": "worker_agent_task_execution",
                    "task_id": task_id
                })

            error_message = f"Worker agent encountered an error: {str(e)}"

            # Add failed task to history
            self.task_history.append({
                "task_id": task_id,
                "instruction": instruction,
                "result": error_message,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })

            return error_message

    async def _plan_task(self, instruction: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Decompose task into a structured plan.

        Args:
            instruction: Task instruction
            context: Optional context

        Returns:
            Dictionary containing task plan with steps, requirements, and constraints
        """
        # Simple task analysis for now
        # Future: Use LLM for task decomposition

        steps = []
        requirements = []
        constraints = ["sandbox_mode", "no_file_writes", "no_code_execution"]

        # Detect task type
        instruction_lower = instruction.lower()

        if any(keyword in instruction_lower for keyword in ["implement", "create", "build", "develop"]):
            task_type = "implementation"
            steps = [
                "Analyze requirements",
                "Design architecture/approach",
                "Generate implementation code",
                "Document usage and examples"
            ]
            requirements = ["technical_knowledge", "code_generation"]

        elif any(keyword in instruction_lower for keyword in ["refactor", "improve", "optimize"]):
            task_type = "refactoring"
            steps = [
                "Analyze existing code",
                "Identify improvement opportunities",
                "Propose refactored solution",
                "Document changes"
            ]
            requirements = ["code_analysis", "best_practices"]

        elif any(keyword in instruction_lower for keyword in ["explain", "describe", "how", "what", "why"]):
            task_type = "explanation"
            steps = [
                "Understand the question",
                "Gather relevant knowledge",
                "Structure explanation",
                "Provide examples"
            ]
            requirements = ["technical_knowledge", "clear_communication"]

        else:
            task_type = "general"
            steps = [
                "Understand request",
                "Research and analyze",
                "Generate response"
            ]
            requirements = ["reasoning", "communication"]

        return {
            "task_type": task_type,
            "steps": steps,
            "requirements": requirements,
            "constraints": constraints,
            "estimated_complexity": len(steps)
        }

    def _build_task_prompt(self, instruction: str, task_plan: Dict[str, Any],
                          context: Optional[Dict[str, Any]]) -> str:
        """
        Build an enhanced prompt for the worker agent's task.

        Args:
            instruction: Original task instruction
            task_plan: Decomposed task plan
            context: Optional context

        Returns:
            Enhanced prompt string
        """
        prompt_parts = [
            "[WORKER AGENT MODE - Technical Task]",
            f"[Task Type: {task_plan['task_type'].title()}]",
            f"[Steps: {', '.join(task_plan['steps'])}]",
            "",
            "You are a specialized worker agent handling a delegated technical task.",
            "Provide a detailed, focused response that addresses all aspects of the request.",
            ""
        ]

        # Add context if available
        if context:
            prompt_parts.append("[Context]")
            for key, value in context.items():
                prompt_parts.append(f"  {key}: {value}")
            prompt_parts.append("")

        # Add task instruction
        prompt_parts.append("[Task]")
        prompt_parts.append(instruction)
        prompt_parts.append("")

        prompt_parts.append("[Requirements]")
        prompt_parts.append(f"- Task type: {task_plan['task_type']}")
        prompt_parts.append(f"- Follow steps: {' → '.join(task_plan['steps'])}")
        prompt_parts.append("- Provide code examples where applicable")
        prompt_parts.append("- Include detailed explanations")
        prompt_parts.append("- Maintain Hugo's tone: professional, clear, and helpful")

        return "\n".join(prompt_parts)

    def _fallback_task_execution(self, instruction: str, task_plan: Dict[str, Any]) -> str:
        """
        Fallback task execution when cognition engine is unavailable.

        Args:
            instruction: Task instruction
            task_plan: Task plan

        Returns:
            Basic task response
        """
        return f"""Worker Agent Task Response:

Task: {instruction}

Task Type: {task_plan['task_type'].title()}

Planned Steps:
{chr(10).join(f"  {i+1}. {step}" for i, step in enumerate(task_plan['steps']))}

Status: Worker agent is operating in fallback mode.
Hugo's cognition engine is currently unavailable.

The worker agent can execute this task once the cognition engine is restored.
"""

    def get_task_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent task history.

        Args:
            limit: Maximum number of tasks to return

        Returns:
            List of task history dictionaries
        """
        return self.task_history[-limit:]

    def clear_history(self):
        """Clear task history"""
        self.task_history = []
        if self.logger:
            self.logger.log_event("worker_agent", "history_cleared", {})
