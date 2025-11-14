"""
Web Monitor Skill
-----------------
Monitors web conditions and triggers alerts.

Actions:
- check: Check a specific monitor rule
- add: Add a new monitor rule
- list: List all monitor rules
- remove: Remove a monitor rule
"""

import aiohttp
import asyncio
import json
import re
from typing import Dict, Any, List
from datetime import datetime

from skills.base_skill import BaseSkill, SkillResult


class WebMonitorSkill(BaseSkill):
    """
    Web monitoring skill for condition-based alerts.

    Monitors URLs, APIs, or values and triggers when conditions are met.
    """

    def __init__(self, logger=None, sqlite_manager=None, memory_manager=None):
        super().__init__(logger, sqlite_manager, memory_manager)

        self.name = "web_monitor"
        self.description = "Monitors web conditions and triggers alerts"
        self.version = "1.0.0"

    async def run(self, action: str = "check", **kwargs) -> SkillResult:
        """
        Execute web monitor skill.

        Args:
            action: Action to perform (check, add, list, remove, help)
            **kwargs: Action-specific arguments

        Returns:
            SkillResult with monitoring results
        """
        if action == "check":
            return await self._check(**kwargs)
        elif action == "add":
            return await self._add(**kwargs)
        elif action == "list":
            return await self._list(**kwargs)
        elif action == "remove":
            return await self._remove(**kwargs)
        elif action == "help":
            return self._help()
        else:
            return SkillResult(
                success=False,
                output=None,
                message=f"Unknown action: {action}"
            )

    async def _check(self, rule_id: int = None, **kwargs) -> SkillResult:
        """
        Check a specific monitor rule.

        Args:
            rule_id: Rule ID to check

        Returns:
            SkillResult with check results
        """
        if rule_id is None:
            return SkillResult(
                success=False,
                output=None,
                message="Missing required argument: rule_id"
            )

        if not self.sqlite:
            return SkillResult(
                success=False,
                output=None,
                message="SQLite manager not available"
            )

        try:
            # Get rule from database
            rules = await self.sqlite.get_monitor_rules()
            rule = next((r for r in rules if r["id"] == rule_id), None)

            if not rule:
                return SkillResult(
                    success=False,
                    output=None,
                    message=f"Rule {rule_id} not found"
                )

            # Parse target (could be URL, API endpoint, etc.)
            target = rule["target"]
            condition = rule["condition"]
            threshold = float(rule["threshold"])

            # Fetch current value
            current_value = await self._fetch_value(target)

            if current_value is None:
                return SkillResult(
                    success=False,
                    output=None,
                    message=f"Failed to fetch value from {target}"
                )

            # Check condition
            triggered = False
            if condition == "above" and current_value > threshold:
                triggered = True
            elif condition == "below" and current_value < threshold:
                triggered = True
            elif condition == "equals" and abs(current_value - threshold) < 0.01:
                triggered = True

            results = {
                "rule_id": rule_id,
                "target": target,
                "condition": condition,
                "threshold": threshold,
                "current_value": current_value,
                "triggered": triggered,
                "timestamp": datetime.now().isoformat()
            }

            return SkillResult(
                success=True,
                output=results,
                message=f"Rule {rule_id} check: {'TRIGGERED' if triggered else 'OK'}",
                metadata={"rule_id": rule_id, "triggered": triggered}
            )

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "check_monitor_rule", "rule_id": rule_id})

            return SkillResult(
                success=False,
                output=None,
                message=f"Check failed: {str(e)}"
            )

    async def _fetch_value(self, target: str) -> float:
        """
        Fetch numeric value from target.

        Args:
            target: URL or API endpoint

        Returns:
            Numeric value or None if failed
        """
        try:
            # Check if target is a URL
            if not target.startswith("http"):
                return None

            timeout = aiohttp.ClientTimeout(total=10)
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; HugoBot/1.0)"
            }

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(target, headers=headers) as response:
                    if response.status != 200:
                        return None

                    # Try to parse as JSON
                    try:
                        data = await response.json()
                        # Look for numeric values in common fields
                        for key in ["value", "price", "result", "data", "number"]:
                            if key in data and isinstance(data[key], (int, float)):
                                return float(data[key])
                        # If data itself is numeric
                        if isinstance(data, (int, float)):
                            return float(data)
                    except:
                        pass

                    # Try to extract number from text
                    text = await response.text()
                    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
                    if numbers:
                        return float(numbers[0])

            return None

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "fetch_value", "target": target})
            return None

    async def _add(self, target: str = None, condition: str = None, threshold: float = None, **kwargs) -> SkillResult:
        """
        Add a new monitor rule.

        Args:
            target: URL or endpoint to monitor
            condition: Condition (above, below, equals)
            threshold: Threshold value

        Returns:
            SkillResult with rule ID
        """
        if not target or not condition or threshold is None:
            return SkillResult(
                success=False,
                output=None,
                message="Missing required arguments: target, condition, threshold"
            )

        if condition not in ["above", "below", "equals"]:
            return SkillResult(
                success=False,
                output=None,
                message="Invalid condition. Must be: above, below, or equals"
            )

        if not self.sqlite:
            return SkillResult(
                success=False,
                output=None,
                message="SQLite manager not available"
            )

        try:
            # Add rule to database
            rule_id = await self.sqlite.add_monitor_rule(
                target=target,
                condition=condition,
                threshold=float(threshold),
                created_at=datetime.now().isoformat()
            )

            return SkillResult(
                success=True,
                output={"rule_id": rule_id, "target": target, "condition": condition, "threshold": threshold},
                message=f"Monitor rule {rule_id} created",
                metadata={"rule_id": rule_id}
            )

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "add_monitor_rule"})

            return SkillResult(
                success=False,
                output=None,
                message=f"Failed to add rule: {str(e)}"
            )

    async def _list(self, **kwargs) -> SkillResult:
        """
        List all monitor rules.

        Returns:
            SkillResult with list of rules
        """
        if not self.sqlite:
            return SkillResult(
                success=False,
                output=None,
                message="SQLite manager not available"
            )

        try:
            rules = await self.sqlite.get_monitor_rules()

            if not rules:
                return SkillResult(
                    success=True,
                    output=[],
                    message="No monitor rules found"
                )

            return SkillResult(
                success=True,
                output=rules,
                message=f"Found {len(rules)} monitor rules",
                metadata={"count": len(rules)}
            )

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "list_monitor_rules"})

            return SkillResult(
                success=False,
                output=None,
                message=f"Failed to list rules: {str(e)}"
            )

    async def _remove(self, rule_id: int = None, **kwargs) -> SkillResult:
        """
        Remove a monitor rule.

        Args:
            rule_id: Rule ID to remove

        Returns:
            SkillResult with removal confirmation
        """
        if rule_id is None:
            return SkillResult(
                success=False,
                output=None,
                message="Missing required argument: rule_id"
            )

        if not self.sqlite:
            return SkillResult(
                success=False,
                output=None,
                message="SQLite manager not available"
            )

        try:
            await self.sqlite.remove_monitor_rule(rule_id)

            return SkillResult(
                success=True,
                output={"rule_id": rule_id},
                message=f"Monitor rule {rule_id} removed",
                metadata={"rule_id": rule_id}
            )

        except Exception as e:
            if self.logger:
                self.logger.log_error(e, {"phase": "remove_monitor_rule", "rule_id": rule_id})

            return SkillResult(
                success=False,
                output=None,
                message=f"Failed to remove rule: {str(e)}"
            )

    def _help(self) -> SkillResult:
        """
        Return help information.

        Returns:
            SkillResult with usage information
        """
        help_text = """
Web Monitor Skill Usage:
=========================

Actions:
  check     Check a specific monitor rule
            Args: rule_id (required)
            Example: /skill run web_monitor check rule_id=1

  add       Add a new monitor rule
            Args: target, condition, threshold (required)
            Example: /net monitor add "https://api.example.com/price" above 100

  list      List all monitor rules
            Example: /net monitor list

  remove    Remove a monitor rule
            Args: rule_id (required)
            Example: /net monitor remove 1

  help      Show this help message
            Example: /skill run web_monitor help

Conditions:
  above     Trigger when value is above threshold
  below     Trigger when value is below threshold
  equals    Trigger when value equals threshold (±0.01)
"""

        return SkillResult(
            success=True,
            output=help_text.strip(),
            message="Web monitor skill help"
        )

    def requires_permissions(self) -> List[str]:
        """Return required permissions"""
        return ["external_http", "internet_access", "database_write"]
