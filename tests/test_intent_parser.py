"""
Tests for HugoIntentParser.
All Ollama LLM calls are mocked — no network calls made.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from core.intent.intent_parser import HugoIntentParser
from core.intent.parsed_intent import ParsedIntent


@pytest.fixture
def parser():
    return HugoIntentParser(
        ollama_url="http://localhost:11434",
        model_name="test-model",
        confidence_threshold=0.75,
    )


def mock_llm_response(data: dict):
    """Create a mock LLM response returning the given JSON dict."""
    return json.dumps(data)


class TestConversationalMessages:
    def test_greeting_returns_no_action(self, parser):
        with patch.object(parser, "_query_llm", return_value=mock_llm_response({
            "requires_action": False, "domain": None, "action": None,
            "target": None, "parameters": {}, "reasoning": "Just a greeting", "confidence": 0.95
        })):
            result = parser.parse("Hey there!")
        assert result.requires_action is False

    def test_general_question_returns_no_action(self, parser):
        with patch.object(parser, "_query_llm", return_value=mock_llm_response({
            "requires_action": False, "domain": None, "action": None,
            "target": None, "parameters": {}, "reasoning": "General question", "confidence": 0.9
        })):
            result = parser.parse("What is the capital of France?")
        assert result.requires_action is False

    def test_coding_question_returns_no_action(self, parser):
        with patch.object(parser, "_query_llm", return_value=mock_llm_response({
            "requires_action": False, "domain": None, "action": None,
            "target": None, "parameters": {}, "reasoning": "Code question", "confidence": 0.88
        })):
            result = parser.parse("How do I write a Python decorator?")
        assert result.requires_action is False


class TestDockerIntents:
    def test_list_containers_detected(self, parser):
        with patch.object(parser, "_query_llm", return_value=mock_llm_response({
            "requires_action": True, "domain": "docker", "action": "list",
            "target": None, "parameters": {}, "reasoning": "User wants to list containers", "confidence": 0.92
        })):
            result = parser.parse("list all docker containers")
        assert result.requires_action is True
        assert result.domain == "docker"
        assert result.action == "list"

    def test_restart_container_with_target(self, parser):
        with patch.object(parser, "_query_llm", return_value=mock_llm_response({
            "requires_action": True, "domain": "docker", "action": "restart",
            "target": "ollama", "parameters": {}, "reasoning": "Restart ollama container", "confidence": 0.95
        })):
            result = parser.parse("restart the ollama container")
        assert result.requires_action is True
        assert result.domain == "docker"
        assert result.action == "restart"
        assert result.target == "ollama"

    def test_stop_container_detected(self, parser):
        with patch.object(parser, "_query_llm", return_value=mock_llm_response({
            "requires_action": True, "domain": "docker", "action": "stop",
            "target": "immich", "parameters": {}, "reasoning": "Stop immich", "confidence": 0.90
        })):
            result = parser.parse("stop immich")
        assert result.domain == "docker"
        assert result.action == "stop"


class TestMonitorIntents:
    def test_network_status_detected(self, parser):
        with patch.object(parser, "_query_llm", return_value=mock_llm_response({
            "requires_action": True, "domain": "monitor", "action": "status",
            "target": None, "parameters": {}, "reasoning": "Network status check", "confidence": 0.88
        })):
            result = parser.parse("how's the network looking?")
        assert result.domain == "monitor"
        assert result.action == "status"

    def test_check_host_with_target(self, parser):
        with patch.object(parser, "_query_llm", return_value=mock_llm_response({
            "requires_action": True, "domain": "monitor", "action": "check",
            "target": "proxmox", "parameters": {}, "reasoning": "Check proxmox", "confidence": 0.91
        })):
            result = parser.parse("is proxmox up?")
        assert result.target == "proxmox"


class TestConfidenceThreshold:
    def test_below_threshold_returns_no_action(self, parser):
        with patch.object(parser, "_query_llm", return_value=mock_llm_response({
            "requires_action": True, "domain": "docker", "action": "list",
            "target": None, "parameters": {}, "reasoning": "Ambiguous", "confidence": 0.5
        })):
            result = parser.parse("can you look into that docker thing")
        assert result.requires_action is False
        assert result.confidence == 0.5

    def test_exactly_at_threshold_allowed(self, parser):
        with patch.object(parser, "_query_llm", return_value=mock_llm_response({
            "requires_action": True, "domain": "monitor", "action": "status",
            "target": None, "parameters": {}, "reasoning": "Just above threshold", "confidence": 0.75
        })):
            result = parser.parse("network status")
        assert result.requires_action is True


class TestMalformedLLMResponse:
    def test_empty_response_returns_no_action(self, parser):
        with patch.object(parser, "_query_llm", return_value=""):
            result = parser.parse("restart docker")
        assert result.requires_action is False

    def test_invalid_json_returns_no_action(self, parser):
        with patch.object(parser, "_query_llm", return_value="not json at all {broken"):
            result = parser.parse("restart docker")
        assert result.requires_action is False
        assert result.confidence == 0.0

    def test_markdown_wrapped_json_parsed(self, parser):
        wrapped = "```json\n" + json.dumps({
            "requires_action": True, "domain": "docker", "action": "list",
            "target": None, "parameters": {}, "reasoning": "test", "confidence": 0.85
        }) + "\n```"
        with patch.object(parser, "_query_llm", return_value=wrapped):
            result = parser.parse("list containers")
        assert result.requires_action is True
        assert result.domain == "docker"

    def test_llm_exception_returns_no_action(self, parser):
        with patch.object(parser, "_query_llm", side_effect=Exception("Network error")):
            result = parser.parse("restart ollama")
        assert result.requires_action is False


class TestMapToExecutor:
    def test_docker_intent_mapped_correctly(self, parser):
        intent = ParsedIntent(
            requires_action=True, domain="docker", action="restart",
            target="ollama", parameters={}, confidence=0.9, original_message="restart ollama"
        )
        mapping = parser.map_to_executor(intent)
        assert mapping is not None
        assert mapping["executor"] == "docker"
        assert mapping["action"] == "restart"
        assert mapping["params"].get("container") == "ollama"

    def test_no_action_returns_none(self, parser):
        intent = ParsedIntent(requires_action=False, confidence=0.0, original_message="hi")
        assert parser.map_to_executor(intent) is None
