"""
Hugo Intent Parser
------------------
Uses the local Ollama model to understand natural language intent and determine
if the user wants an action performed (SSH, Docker, Monitor) or just conversation.

Adapted from Server-Files llm_intent_parser.py with:
- Capabilities trimmed to SSH, Docker, Monitor only
- Hugo-specific domain terminology
- Confidence threshold gating (INTENT_CONFIDENCE_THRESHOLD env var, default 0.75)
- Uses Hugo's existing OllamaStabilityManager for resilient LLM calls
- Graceful fallback: any parse failure returns requires_action=False
"""

import json
import os
import requests
from typing import Dict, List, Optional

from .parsed_intent import ParsedIntent


class HugoIntentParser:
    """
    Parses natural language messages to detect infrastructure action intent.

    Returns ParsedIntent with:
    - requires_action=True if confidence >= threshold AND an executable domain/action is detected
    - requires_action=False for conversation, ambiguous input, or LLM failures
    """

    def __init__(
        self,
        ollama_url: str = None,
        model_name: str = None,
        logger=None,
        confidence_threshold: float = None,
    ):
        self.ollama_url = (ollama_url or os.getenv("OLLAMA_API", "http://localhost:11434/api/generate"))
        # Strip /api/generate suffix if present — we need the base URL
        if self.ollama_url.endswith("/api/generate"):
            self.ollama_url = self.ollama_url[:-len("/api/generate")]
        self.model_name = model_name or os.getenv("INTENT_MODEL", os.getenv("MODEL_NAME", "llama3:8b"))
        self.logger = logger
        self.confidence_threshold = confidence_threshold or float(
            os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.75")
        )

        # Available capabilities — trimmed to the executors we've ported
        self.capabilities = {
            "ssh": {
                "description": "Run commands on remote hosts via SSH",
                "actions": [
                    {"name": "run", "description": "Run a command on a remote host", "params": ["host", "command"]},
                    {"name": "test", "description": "Test SSH connection to a host", "params": ["host"]},
                    {"name": "list_hosts", "description": "List all known SSH hosts"},
                    {"name": "add_host", "description": "Add a new SSH host", "params": ["name", "ip"]},
                ],
            },
            "docker": {
                "description": "Manage Docker containers",
                "actions": [
                    {"name": "list", "description": "List all running containers"},
                    {"name": "status", "description": "Check status of a container", "params": ["container"]},
                    {"name": "start", "description": "Start a container", "params": ["container"]},
                    {"name": "stop", "description": "Stop a container", "params": ["container"]},
                    {"name": "restart", "description": "Restart a container", "params": ["container"]},
                    {"name": "logs", "description": "View container logs", "params": ["container", "lines"]},
                ],
            },
            "monitor": {
                "description": "Network monitoring — check host connectivity and uptime",
                "actions": [
                    {"name": "status", "description": "Get overall network status"},
                    {"name": "check", "description": "Check if a specific host is up", "params": ["host"]},
                    {"name": "alerts", "description": "Get recent network alerts"},
                    {"name": "uptime", "description": "Get uptime statistics", "params": ["host"]},
                    {"name": "hosts", "description": "List all monitored hosts"},
                    {"name": "add_host", "description": "Add a host to monitoring", "params": ["name", "ip"]},
                    {"name": "scan", "description": "Scan network for devices"},
                ],
            },
            "memory": {
                "description": "Recall or search Hugo's persistent memory",
                "actions": [
                    {"name": "search", "description": "Search memory for relevant information", "params": ["query"]},
                    {"name": "recall", "description": "Recall facts about a topic", "params": ["topic"]},
                    {"name": "list_recent", "description": "List recently stored memories"},
                    {"name": "forget", "description": "Remove a specific memory", "params": ["memory_id"]},
                ],
            },
        }

    def _build_capabilities_description(self) -> str:
        lines = ["Available capabilities (ONLY these domains are executable):"]
        for domain, info in self.capabilities.items():
            lines.append(f"\n## {domain.upper()}: {info['description']}")
            for action in info["actions"]:
                params = (f" (params: {', '.join(action.get('params', []))})"
                          if action.get("params") else "")
                lines.append(f"  - {action['name']}: {action['description']}{params}")
        return "\n".join(lines)

    def _query_llm(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Query Ollama with low temperature for consistent JSON parsing.
        Falls back to empty string on any error — callers handle this gracefully.
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": 400,
                    },
                },
                timeout=30,
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            self._log("llm_error", {"status_code": response.status_code})
            return ""
        except Exception as e:
            self._log("llm_exception", {"error": str(e)})
            return ""

    def parse(self, message: str, conversation_context: List[Dict] = None) -> ParsedIntent:
        """
        Parse a user message to determine if it's an action request.

        Args:
            message: User's natural language input
            conversation_context: Recent conversation turns for context (optional)

        Returns:
            ParsedIntent — requires_action=False if confidence < threshold or parse fails
        """
        capabilities_desc = self._build_capabilities_description()

        context_str = ""
        if conversation_context:
            recent = conversation_context[-3:]
            context_str = "\nRecent conversation:\n" + "\n".join(
                f"- {m.get('role', 'user')}: {str(m.get('content', ''))[:100]}"
                for m in recent
            )

        prompt = f"""You are an intent parser for Hugo, an AI assistant with infrastructure control.
Determine if the user wants to perform an infrastructure action or is just having a conversation.

{capabilities_desc}
{context_str}

User message: "{message}"

Respond with ONLY valid JSON in this exact format:
{{
    "requires_action": true/false,
    "domain": "ssh|docker|monitor|memory|null",
    "action": "action_name or null",
    "target": "specific target (container name, host name) or null",
    "parameters": {{}},
    "reasoning": "brief explanation",
    "confidence": 0.0-1.0
}}

Rules:
- requires_action=true ONLY if user wants infrastructure work done (check, start, stop, restart, list, run command, etc.) OR wants Hugo to recall/search memory
- General conversation, questions about topics, coding help = requires_action: false
- Only use domains from the capabilities list above (ssh, docker, monitor, memory)
- Set confidence lower if the intent is ambiguous
- Extract container names, host names into target field
- For docker containers, put the container name in target
- For SSH commands, put the host in target and command in parameters.command

Respond with ONLY the JSON, no other text:"""

        raw = ""
        try:
            raw = self._query_llm(prompt)
            # Strip markdown code blocks if present
            text = raw.strip()
            if text.startswith("```"):
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else text
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            data = json.loads(text)

            confidence = float(data.get("confidence", 0.0))
            requires_action = bool(data.get("requires_action", False))

            # Gate on confidence threshold
            if requires_action and confidence < self.confidence_threshold:
                self._log("below_threshold", {
                    "message": message[:80],
                    "confidence": confidence,
                    "threshold": self.confidence_threshold,
                })
                return ParsedIntent(
                    requires_action=False,
                    reasoning=f"Confidence {confidence:.2f} below threshold {self.confidence_threshold:.2f}",
                    confidence=confidence,
                    original_message=message,
                )

            return ParsedIntent(
                requires_action=requires_action,
                domain=data.get("domain") or None,
                action=data.get("action") or None,
                target=data.get("target") or None,
                parameters=data.get("parameters") or {},
                reasoning=data.get("reasoning", ""),
                confidence=confidence,
                original_message=message,
            )

        except Exception as e:
            self._log("parse_error", {"error": str(e), "raw": raw[:200]})
            return ParsedIntent(
                requires_action=False,
                reasoning=f"Failed to parse LLM response: {str(e)}",
                confidence=0.0,
                original_message=message,
            )

    def map_to_executor(self, intent: ParsedIntent) -> Optional[Dict]:
        """
        Map a ParsedIntent to an executor call dict.

        Returns:
            {"executor": str, "action": str, "params": dict, "confidence": float}
            or None if intent doesn't require action
        """
        if not intent.requires_action or not intent.domain or not intent.action:
            return None

        params = dict(intent.parameters or {})

        # Enrich params based on domain/target conventions
        if intent.target:
            if intent.domain == "docker":
                params.setdefault("container", intent.target)
            elif intent.domain == "ssh":
                params.setdefault("host", intent.target)
            elif intent.domain == "monitor":
                params.setdefault("host", intent.target)

        return {
            "executor": intent.domain,
            "action": intent.action,
            "params": params,
            "confidence": intent.confidence,
            "reasoning": intent.reasoning,
        }

    def _log(self, event: str, details: dict):
        if self.logger:
            try:
                self.logger.log_event("intent_parser", event, details)
            except Exception:
                pass
