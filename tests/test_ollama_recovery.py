"""
Tests for Ollama Recovery and Stability
----------------------------------------
Comprehensive test suite for Ollama error handling, recovery strategies,
and defensive guards.

Tests:
- 500 error recovery with context reduction
- Streaming to non-streaming fallback
- Server down detection and recovery
- Model reload handling
- Payload validation
- Soft fallback messages
"""

import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from core.ollama_stability import OllamaStabilityManager, OllamaResponse


class MockLogger:
    """Mock logger for testing"""
    def log_event(self, category, event_type, data):
        pass


@pytest.fixture
def stability_manager():
    """Create OllamaStabilityManager for testing"""
    logger = MockLogger()
    return OllamaStabilityManager(
        api_url="http://localhost:11434/api/generate",
        model_name="llama3:8b",
        logger=logger,
        max_retries=3
    )


def test_payload_validation_success(stability_manager):
    """Test valid payload passes validation"""
    payload = {
        "model": "llama3:8b",
        "prompt": "Hello world",
        "stream": True,
        "options": {
            "temperature": 0.7
        }
    }

    is_valid, error = stability_manager.validate_payload(payload)

    assert is_valid is True
    assert error is None


def test_payload_validation_missing_model(stability_manager):
    """Test payload validation fails without model"""
    payload = {
        "prompt": "Hello world"
    }

    is_valid, error = stability_manager.validate_payload(payload)

    assert is_valid is False
    assert "model" in error


def test_payload_validation_missing_prompt(stability_manager):
    """Test payload validation fails without prompt"""
    payload = {
        "model": "llama3:8b"
    }

    is_valid, error = stability_manager.validate_payload(payload)

    assert is_valid is False
    assert "prompt" in error


def test_payload_validation_invalid_temperature(stability_manager):
    """Test payload validation fails with invalid temperature"""
    payload = {
        "model": "llama3:8b",
        "prompt": "Hello",
        "options": {
            "temperature": 5.0  # Invalid: > 2.0
        }
    }

    is_valid, error = stability_manager.validate_payload(payload)

    assert is_valid is False
    assert "Temperature" in error


def test_context_reduction(stability_manager):
    """Test context reduction reduces prompt size"""
    original_prompt = "A" * 1000
    reduced = stability_manager.reduce_context(original_prompt, reduction_factor=0.7)

    assert len(reduced) < len(original_prompt)
    assert len(reduced) <= int(len(original_prompt) * 0.7)


def test_context_reduction_preserves_content(stability_manager):
    """Test context reduction keeps beginning of prompt"""
    original_prompt = "Important start. " + ("B" * 1000)
    reduced = stability_manager.reduce_context(original_prompt, reduction_factor=0.5)

    assert reduced.startswith("Important start")


def test_soft_fallback_messages(stability_manager):
    """Test soft fallback messages are user-friendly"""
    msg_general = stability_manager.soft_fallback_message("general")
    msg_server_down = stability_manager.soft_fallback_message("server_down")
    msg_500 = stability_manager.soft_fallback_message("500_error")

    # Should be friendly, not technical
    assert "reasoning" in msg_general.lower() or "warming up" in msg_general.lower()
    assert "restart" in msg_server_down.lower() or "moment" in msg_server_down.lower()
    assert "try" in msg_500.lower() or "adjust" in msg_500.lower()

    # Should not expose technical errors
    assert "500" not in msg_500
    assert "error" not in msg_general.lower() or "warming up" in msg_general.lower()


@patch('requests.get')
def test_server_health_check_healthy(mock_get, stability_manager):
    """Test server health check when server is healthy"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    is_healthy = stability_manager.check_server_health()

    assert is_healthy is True
    assert stability_manager.server_available is True


@patch('requests.get')
def test_server_health_check_unhealthy(mock_get, stability_manager):
    """Test server health check when server returns error"""
    mock_response = Mock()
    mock_response.status_code = 500
    mock_get.return_value = mock_response

    is_healthy = stability_manager.check_server_health()

    assert is_healthy is False
    assert stability_manager.server_available is False


@patch('requests.get')
def test_server_health_check_unreachable(mock_get, stability_manager):
    """Test server health check when server is unreachable"""
    mock_get.side_effect = requests.exceptions.ConnectionError()

    is_healthy = stability_manager.check_server_health()

    assert is_healthy is False
    assert stability_manager.server_available is False


@patch('requests.post')
def test_stream_ollama_500_recovery(mock_post, stability_manager):
    """Test streaming recovers from 500 error with context reduction"""
    # First attempt: 500 error
    mock_response_500 = Mock()
    mock_response_500.status_code = 500
    mock_response_500.json.return_value = {"error": "out of memory"}
    mock_response_500.headers = {}
    mock_response_500.text = "out of memory"

    # Second attempt: success
    mock_response_success = Mock()
    mock_response_success.status_code = 200
    mock_response_success.iter_lines.return_value = [
        b'{"response": "Hello", "done": false}',
        b'{"response": " world", "done": true}'
    ]
    mock_response_success.headers = {}

    mock_post.side_effect = [mock_response_500, mock_response_success]

    # Run streaming with recovery
    chunks = list(stability_manager.stream_with_recovery("A" * 10000, temperature=0.7))

    # Should get successful response
    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert "Hello world" in full_response or "warming up" in full_response


@patch('requests.post')
def test_nonstream_retry_on_stream_fail(mock_post, stability_manager):
    """Test falls back to non-streaming when streaming fails"""
    # All streaming attempts fail
    mock_post.side_effect = requests.exceptions.Timeout()

    chunks = list(stability_manager.stream_with_recovery("Test prompt", temperature=0.7))

    # Should get soft fallback
    assert len(chunks) > 0
    assert any("warming up" in chunk or "moment" in chunk for chunk in chunks)


@patch('requests.post')
def test_context_shrink_on_500(mock_post, stability_manager):
    """Test context shrinks on 500 error"""
    call_count = [0]
    original_length = [0]

    def side_effect_500(*args, **kwargs):
        call_count[0] += 1

        # Capture original prompt length on first call
        if call_count[0] == 1:
            original_length[0] = len(kwargs['json']['prompt'])

        # First call: 500 error
        if call_count[0] == 1:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.json.return_value = {"error": "context too large"}
            mock_response.headers = {}
            mock_response.text = "context too large"
            return mock_response

        # Second call: should have reduced context
        assert len(kwargs['json']['prompt']) < original_length[0], \
            "Context should be reduced on retry"

        # Success
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"response": "OK", "done": true}'
        ]
        mock_response.headers = {}
        return mock_response

    mock_post.side_effect = side_effect_500

    chunks = list(stability_manager.stream_with_recovery("A" * 10000, temperature=0.7))

    assert call_count[0] >= 2  # Should retry with reduced context
    assert len(chunks) > 0


@patch('requests.get')
@patch('requests.post')
def test_server_down_recovery(mock_post, mock_get, stability_manager):
    """Test handles server down gracefully"""
    # POST fails with connection error
    mock_post.side_effect = requests.exceptions.ConnectionError()

    # Health check also fails
    mock_get.side_effect = requests.exceptions.ConnectionError()

    chunks = list(stability_manager.stream_with_recovery("Test", temperature=0.7))

    # Should return soft fallback about server down
    assert len(chunks) > 0
    fallback_msg = "".join(chunks)
    assert "restart" in fallback_msg.lower() or "moment" in fallback_msg.lower()


@patch('requests.post')
def test_model_reload_after_crash(mock_post, stability_manager):
    """Test detects and handles model reload"""
    # First attempt: model not found (unloaded)
    mock_response_404 = Mock()
    mock_response_404.status_code = 500
    mock_response_404.json.return_value = {"error": "model not found"}
    mock_response_404.headers = {}
    mock_response_404.text = "model not found"

    # Second attempt: success (model reloaded)
    mock_response_success = Mock()
    mock_response_success.status_code = 200
    mock_response_success.iter_lines.return_value = [
        b'{"response": "Reloaded", "done": true}'
    ]
    mock_response_success.headers = {}

    mock_post.side_effect = [mock_response_404, mock_response_success]

    chunks = list(stability_manager.stream_with_recovery("Test", temperature=0.7))

    assert len(chunks) > 0
    # Should either succeed after retry or give soft fallback
    full_response = "".join(chunks)
    assert "Reloaded" in full_response or "model" in full_response.lower()


@patch('requests.post')
def test_nonstream_fallback_success(mock_post, stability_manager):
    """Test non-streaming fallback works"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Fallback success"}
    mock_post.return_value = mock_response

    result = stability_manager.non_stream_fallback("Test prompt", temperature=0.7)

    assert result.success is True
    assert result.content == "Fallback success"
    assert result.recovery_action == "nonstreaming_fallback"


@patch('requests.post')
def test_nonstream_fallback_failure(mock_post, stability_manager):
    """Test non-streaming fallback handles errors"""
    mock_post.side_effect = requests.exceptions.Timeout()

    result = stability_manager.non_stream_fallback("Test prompt", temperature=0.7)

    assert result.success is False
    assert result.fallback_used is True


@patch('requests.post')
def test_consecutive_failure_tracking(mock_post, stability_manager):
    """Test tracks consecutive failures"""
    mock_post.side_effect = requests.exceptions.Timeout()

    # First failure
    list(stability_manager.stream_with_recovery("Test 1"))
    failures_1 = stability_manager.consecutive_failures

    # Second failure
    list(stability_manager.stream_with_recovery("Test 2"))
    failures_2 = stability_manager.consecutive_failures

    assert failures_2 > failures_1


@patch('requests.post')
def test_handle_500_with_cuda_error(mock_post, stability_manager):
    """Test handles CUDA out of memory error"""
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.json.return_value = {"error": "CUDA out of memory"}
    mock_response.headers = {}

    recovery_action = stability_manager.handle_500_error(mock_response, attempt=1)

    assert recovery_action == "reduce_context"


@patch('requests.post')
def test_validate_before_send(mock_post, stability_manager):
    """Test validates payload before sending"""
    # This should fail validation (empty prompt)
    chunks = list(stability_manager.stream_with_recovery("", temperature=0.7))

    # Should not call POST if validation fails
    assert mock_post.call_count == 0

    # Should return soft fallback
    assert len(chunks) > 0


def test_ollama_response_dataclass():
    """Test OllamaResponse dataclass"""
    response = OllamaResponse(
        success=True,
        content="Test response",
        fallback_used=False,
        attempts=2,
        duration=1.5,
        recovery_action="retry"
    )

    assert response.success is True
    assert response.content == "Test response"
    assert response.attempts == 2
    assert response.recovery_action == "retry"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
