"""
Autonomous Agent Engine
-----------------------
Implements Hugo's autonomous agent loop for proactive actions.

Features:
- Condition-based triggers
- Scheduled task execution
- Web monitoring
- Autonomous skill execution
- Agent-level reflection
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any


class HugoAgent:
    """
    Autonomous agent for proactive task execution.

    Runs on a scheduled tick cycle, evaluating triggers and executing
    skills autonomously based on conditions, schedules, and patterns.
    """

    def __init__(self, memory_manager, skill_manager, sqlite_manager, reflection_engine, logger):
        """
        Initialize the autonomous agent.

        Args:
            memory_manager: MemoryManager instance
            skill_manager: SkillManager instance
            sqlite_manager: SQLiteManager instance
            reflection_engine: ReflectionEngine instance
            logger: HugoLogger instance
        """
        self.memory = memory_manager
        self.skills = skill_manager
        self.sqlite = sqlite_manager
        self.reflection = reflection_engine
        self.logger = logger

        self.enabled = True
        self.tick_count = 0
        self.last_tick = None

        # Track action history for reflection
        self.action_history = []

    async def tick(self):
        """
        Main agent tick cycle.

        Evaluates triggers, executes autonomous actions, and reflects.
        Called every 5 seconds by the scheduler.
        """
        if not self.enabled:
            return

        self.tick_count += 1
        self.last_tick = datetime.now()

        self.logger.log_event("agent", "agent_tick", {
            "tick_count": self.tick_count,
            "timestamp": self.last_tick.isoformat()
        })

        try:
            # Evaluate all active triggers
            await self.evaluate_triggers()

            # Execute autonomous actions based on triggers
            await self.execute_autonomous_actions()

            # Periodic reflection (every 100 ticks = ~8 minutes)
            if self.tick_count % 100 == 0:
                await self.reflect_on_actions()

        except Exception as e:
            self.logger.log_error(e, {"phase": "agent_tick"})

    async def evaluate_triggers(self):
        """
        Evaluate all active triggers.

        Checks for:
        - Scheduled tasks that are due
        - Web monitor rules that need checking
        - Reminder patterns
        - Habit tracking
        """
        try:
            # Check for due tasks
            tasks = await self.sqlite.get_tasks_by_status("todo")
            for task in tasks:
                # Check if task has a due date
                if task.get("due_date"):
                    due_date = datetime.fromisoformat(task["due_date"])
                    if due_date <= datetime.now():
                        self.logger.log_event("agent", "agent_trigger_detected", {
                            "trigger_type": "task_due",
                            "task_id": task["id"],
                            "title": task["title"]
                        })

                        # Queue autonomous action
                        await self._queue_action({
                            "type": "task_reminder",
                            "task_id": task["id"],
                            "task_title": task["title"]
                        })

            # Check web monitor rules
            monitor_rules = await self.sqlite.get_monitor_rules()
            for rule in monitor_rules:
                await self._check_monitor_rule(rule)

        except Exception as e:
            self.logger.log_error(e, {"phase": "evaluate_triggers"})

    async def _check_monitor_rule(self, rule: Dict[str, Any]):
        """
        Check a single web monitor rule.

        Args:
            rule: Monitor rule dictionary with target, condition, threshold
        """
        try:
            # Run web_monitor skill to check condition
            if self.skills.get("web_monitor"):
                result = await self.skills.run_skill(
                    "web_monitor",
                    action="check",
                    rule_id=rule["id"]
                )

                if result.success and result.output:
                    if result.output.get("triggered"):
                        self.logger.log_event("agent", "agent_trigger_detected", {
                            "trigger_type": "web_monitor",
                            "rule_id": rule["id"],
                            "target": rule["target"],
                            "condition": rule["condition"],
                            "threshold": rule["threshold"],
                            "current_value": result.output.get("current_value")
                        })

                        # Queue autonomous action
                        await self._queue_action({
                            "type": "web_monitor_alert",
                            "rule": rule,
                            "current_value": result.output.get("current_value")
                        })

        except Exception as e:
            self.logger.log_error(e, {
                "phase": "check_monitor_rule",
                "rule_id": rule.get("id")
            })

    async def _queue_action(self, action: Dict[str, Any]):
        """
        Queue an autonomous action for execution.

        Args:
            action: Action dictionary with type and details
        """
        self.action_history.append({
            "action": action,
            "timestamp": datetime.now(),
            "executed": False
        })

    async def execute_autonomous_actions(self):
        """
        Execute all queued autonomous actions.

        Actions can include:
        - Skill execution
        - Memory storage
        - Alert creation
        - Task updates
        """
        try:
            # Process all unexecuted actions
            for action_entry in self.action_history:
                if action_entry["executed"]:
                    continue

                action = action_entry["action"]
                action_type = action.get("type")

                self.logger.log_event("agent", "agent_autonomous_action", {
                    "action_type": action_type,
                    "details": action
                })

                if action_type == "task_reminder":
                    await self._execute_task_reminder(action)
                elif action_type == "web_monitor_alert":
                    await self._execute_monitor_alert(action)

                # Mark as executed
                action_entry["executed"] = True

                # Save to database
                await self.sqlite.save_agent_action(
                    action_type=action_type,
                    details=str(action),
                    executed_at=datetime.now().isoformat(),
                    success=1
                )

        except Exception as e:
            self.logger.log_error(e, {"phase": "execute_autonomous_actions"})

    async def _execute_task_reminder(self, action: Dict[str, Any]):
        """
        Execute a task reminder action.

        Args:
            action: Task reminder action details
        """
        try:
            from core.memory import MemoryEntry

            # Create memory entry for the reminder
            reminder_entry = MemoryEntry(
                id=None,
                session_id="agent_autonomous",
                timestamp=datetime.now(),
                memory_type="task",
                content=f"Task reminder: {action['task_title']} is now due",
                embedding=None,
                metadata={
                    "task_id": action["task_id"],
                    "autonomous_action": True,
                    "action_type": "task_reminder"
                },
                importance_score=0.8,
                is_fact=False
            )

            await self.memory.store(reminder_entry, persist_long_term=True)

            self.logger.log_event("agent", "agent_autonomous_skill_run", {
                "skill": "task_reminder",
                "task_id": action["task_id"]
            })

        except Exception as e:
            self.logger.log_error(e, {
                "phase": "execute_task_reminder",
                "task_id": action.get("task_id")
            })

    async def _execute_monitor_alert(self, action: Dict[str, Any]):
        """
        Execute a web monitor alert action.

        Args:
            action: Monitor alert action details
        """
        try:
            from core.memory import MemoryEntry

            rule = action["rule"]
            current_value = action.get("current_value")

            # Create memory entry for the alert
            alert_entry = MemoryEntry(
                id=None,
                session_id="agent_autonomous",
                timestamp=datetime.now(),
                memory_type="knowledge",
                content=f"Web monitor alert: {rule['target']} is {rule['condition']} {rule['threshold']} (current: {current_value})",
                embedding=None,
                metadata={
                    "rule_id": rule["id"],
                    "target": rule["target"],
                    "condition": rule["condition"],
                    "threshold": rule["threshold"],
                    "current_value": current_value,
                    "autonomous_action": True,
                    "action_type": "web_monitor_alert"
                },
                importance_score=0.85,
                is_fact=True
            )

            await self.memory.store(alert_entry, persist_long_term=True)

            self.logger.log_event("agent", "agent_autonomous_skill_run", {
                "skill": "web_monitor",
                "rule_id": rule["id"],
                "triggered": True
            })

        except Exception as e:
            self.logger.log_error(e, {
                "phase": "execute_monitor_alert",
                "rule_id": action.get("rule", {}).get("id")
            })

    async def reflect_on_actions(self):
        """
        Perform agent-level reflection on recent actions.

        Creates a reflection summary of autonomous behavior patterns,
        effectiveness, and learning opportunities.
        """
        try:
            # Count executed actions
            executed_count = sum(1 for a in self.action_history if a["executed"])

            # Get recent agent actions from database
            recent_actions = await self.sqlite.get_agent_actions(limit=50)

            # Analyze patterns
            action_types = {}
            for action in recent_actions:
                action_type = action.get("action_type", "unknown")
                action_types[action_type] = action_types.get(action_type, 0) + 1

            # Create reflection summary
            reflection_summary = f"Agent reflection: Executed {executed_count} actions in last {self.tick_count} ticks. Action distribution: {action_types}"

            # Store reflection
            from core.memory import MemoryEntry

            reflection_entry = MemoryEntry(
                id=None,
                session_id="agent_autonomous",
                timestamp=datetime.now(),
                memory_type="reflection",
                content=reflection_summary,
                embedding=None,
                metadata={
                    "tick_count": self.tick_count,
                    "executed_actions": executed_count,
                    "action_types": action_types,
                    "reflection_type": "agent_autonomous"
                },
                importance_score=0.7,
                is_fact=False
            )

            await self.memory.store(reflection_entry, persist_long_term=True)

            self.logger.log_event("agent", "agent_reflection_saved", {
                "tick_count": self.tick_count,
                "executed_actions": executed_count,
                "action_types": action_types
            })

            # Clear old action history
            self.action_history = [a for a in self.action_history if not a["executed"]]

        except Exception as e:
            self.logger.log_error(e, {"phase": "reflect_on_actions"})

    def enable(self):
        """Enable the autonomous agent."""
        self.enabled = True
        self.logger.log_event("agent", "agent_enabled", {})

    def disable(self):
        """Disable the autonomous agent."""
        self.enabled = False
        self.logger.log_event("agent", "agent_disabled", {})

    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status.

        Returns:
            Dictionary with agent status information
        """
        return {
            "enabled": self.enabled,
            "tick_count": self.tick_count,
            "last_tick": self.last_tick.isoformat() if self.last_tick else None,
            "queued_actions": len([a for a in self.action_history if not a["executed"]]),
            "executed_actions": len([a for a in self.action_history if a["executed"]])
        }
