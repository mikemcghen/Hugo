# Phase 5.2: Ollama Stability Testing Guide

## Running the Tests

### Prerequisites

```bash
pip install pytest requests
```

### Run All Ollama Recovery Tests

```bash
# From project root
pytest tests/test_ollama_recovery.py -v
```

### Run Specific Test Categories

```bash
# Payload validation tests
pytest tests/test_ollama_recovery.py -k "validation" -v

# Recovery strategy tests
pytest tests/test_ollama_recovery.py -k "recovery" -v

# Fallback tests
pytest tests/test_ollama_recovery.py -k "fallback" -v

# Health check tests
pytest tests/test_ollama_recovery.py -k "health" -v
```

### Run Individual Tests

```bash
# Test 500 error recovery
pytest tests/test_ollama_recovery.py::test_stream_ollama_500_recovery -v

# Test context reduction
pytest tests/test_ollama_recovery.py::test_context_shrink_on_500 -v

# Test soft fallback messages
pytest tests/test_ollama_recovery.py::test_soft_fallback_messages -v

# Test server down handling
pytest tests/test_ollama_recovery.py::test_server_down_recovery -v
```

---

## Test Coverage

### Payload Validation Tests

✅ **test_payload_validation_success**
- Validates correct payloads pass

✅ **test_payload_validation_missing_model**
- Rejects payloads without model field

✅ **test_payload_validation_missing_prompt**
- Rejects payloads without prompt field

✅ **test_payload_validation_invalid_temperature**
- Rejects temperature outside [0.0, 2.0] range

✅ **test_validate_before_send**
- Ensures validation happens before HTTP request

### Context Reduction Tests

✅ **test_context_reduction**
- Reduces prompt size by specified factor

✅ **test_context_reduction_preserves_content**
- Keeps beginning of prompt intact

✅ **test_context_shrink_on_500**
- Automatically reduces context on 500 error

### Recovery Strategy Tests

✅ **test_stream_ollama_500_recovery**
- Recovers from 500 error with context reduction + retry

✅ **test_nonstream_retry_on_stream_fail**
- Falls back to non-streaming when streaming fails

✅ **test_model_reload_after_crash**
- Detects and handles model unload/reload

✅ **test_server_down_recovery**
- Handles server down with immediate soft fallback

✅ **test_handle_500_with_cuda_error**
- CUDA memory errors trigger context reduction

### Fallback Tests

✅ **test_soft_fallback_messages**
- Generates user-friendly fallback messages

✅ **test_nonstream_fallback_success**
- Non-streaming fallback succeeds

✅ **test_nonstream_fallback_failure**
- Handles non-streaming fallback errors

### Server Health Tests

✅ **test_server_health_check_healthy**
- Detects healthy Ollama server

✅ **test_server_health_check_unhealthy**
- Detects unhealthy server (500 response)

✅ **test_server_health_check_unreachable**
- Detects unreachable server (connection error)

### Monitoring Tests

✅ **test_consecutive_failure_tracking**
- Tracks consecutive failures for monitoring

✅ **test_ollama_response_dataclass**
- OllamaResponse dataclass works correctly

---

## Expected Output

```bash
$ pytest tests/test_ollama_recovery.py -v

tests/test_ollama_recovery.py::test_payload_validation_success PASSED          [  5%]
tests/test_ollama_recovery.py::test_payload_validation_missing_model PASSED    [ 11%]
tests/test_ollama_recovery.py::test_payload_validation_missing_prompt PASSED   [ 16%]
tests/test_ollama_recovery.py::test_payload_validation_invalid_temperature PASSED [ 22%]
tests/test_ollama_recovery.py::test_context_reduction PASSED                   [ 27%]
tests/test_ollama_recovery.py::test_context_reduction_preserves_content PASSED [ 33%]
tests/test_ollama_recovery.py::test_soft_fallback_messages PASSED              [ 38%]
tests/test_ollama_recovery.py::test_server_health_check_healthy PASSED         [ 44%]
tests/test_ollama_recovery.py::test_server_health_check_unhealthy PASSED       [ 50%]
tests/test_ollama_recovery.py::test_server_health_check_unreachable PASSED     [ 55%]
tests/test_ollama_recovery.py::test_stream_ollama_500_recovery PASSED          [ 61%]
tests/test_ollama_recovery.py::test_nonstream_retry_on_stream_fail PASSED      [ 66%]
tests/test_ollama_recovery.py::test_context_shrink_on_500 PASSED               [ 72%]
tests/test_ollama_recovery.py::test_server_down_recovery PASSED                [ 77%]
tests/test_ollama_recovery.py::test_model_reload_after_crash PASSED            [ 83%]
tests/test_ollama_recovery.py::test_nonstream_fallback_success PASSED          [ 88%]
tests/test_ollama_recovery.py::test_nonstream_fallback_failure PASSED          [ 94%]
tests/test_ollama_recovery.py::test_consecutive_failure_tracking PASSED        [100%]

==================== 18 passed in 1.24s ====================
```

---

## Manual Integration Tests

### Test 1: Normal Conversation (Baseline)

```bash
python main.py

> Hello Hugo, how are you today?
# Expected: Normal response, no errors
```

### Test 2: Very Long Context (500 Error Recovery)

```bash
> [Paste a very long text, 50+ paragraphs]
# Expected: Hugo responds (may see slight pause for context reduction)
# Check logs for: "context_reduction" event
```

### Test 3: Ollama Server Restart (Connection Error Recovery)

```bash
# In terminal 1: Start Hugo
python main.py

# In terminal 2: Stop Ollama
docker stop ollama  # or: ollama stop

# In terminal 1: Try to chat
> What's the weather like?
# Expected: "My reasoning core is restarting… one moment please."

# In terminal 2: Start Ollama
docker start ollama  # or: ollama serve

# In terminal 1: Try again
> Try again
# Expected: Normal response resumes
```

### Test 4: Rapid-Fire Messages (Stress Test)

```bash
# Send messages quickly
> Hello
> How are you?
> Tell me a joke
> What's 2+2?
> Goodbye

# Expected: All messages handled gracefully
# No crashes or hangs
```

### Test 5: Mid-Stream Model Unload

```bash
# Start long generation
> Write a 500-word essay on AI

# While generating, unload model (in another terminal)
ollama unload llama3:8b

# Expected: Generation stops gracefully with soft fallback
# Retry automatically when model reloads
```

---

## Debugging Failed Tests

### Verbose Output

```bash
# Show full output including print statements
pytest tests/test_ollama_recovery.py -vv -s
```

### Debug on Failure

```bash
# Drop into debugger on first failure
pytest tests/test_ollama_recovery.py --pdb
```

### Run Single Failing Test

```bash
# Isolate the failing test
pytest tests/test_ollama_recovery.py::test_stream_ollama_500_recovery -vv
```

### Check Test Mocks

```python
# If tests are failing due to mock issues, verify:
from unittest.mock import Mock, patch
import requests

# Ensure mock_post is being called
with patch('requests.post') as mock_post:
    mock_post.return_value = Mock(status_code=200)
    # ... test code ...
    assert mock_post.called
```

---

## Performance Testing

### Measure Recovery Time

```python
import time
from core.ollama_stability import OllamaStabilityManager

manager = OllamaStabilityManager(...)

start = time.time()
for chunk in manager.stream_with_recovery("Test prompt"):
    pass
duration = time.time() - start

print(f"Recovery time: {duration:.2f}s")
# Expected: < 10s for 3 retries
```

### Measure Context Reduction Impact

```python
original = "A" * 10000
reduced = manager.reduce_context(original, 0.7)

print(f"Original: {len(original)} chars")
print(f"Reduced: {len(reduced)} chars")
print(f"Reduction: {(1 - len(reduced)/len(original)) * 100:.1f}%")
# Expected: ~30% reduction
```

---

## Continuous Integration

### GitHub Actions

```yaml
name: Ollama Recovery Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install pytest requests
      - name: Run Ollama recovery tests
        run: |
          pytest tests/test_ollama_recovery.py -v --junitxml=test-results.xml
      - name: Upload test results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: test-results.xml
```

---

## Test Maintenance

### Adding New Tests

```python
# Template for new recovery test
@patch('requests.post')
def test_new_recovery_scenario(mock_post, stability_manager):
    """Test description"""
    # Arrange
    mock_post.side_effect = [
        # First attempt: error
        requests.exceptions.SomeError(),
        # Second attempt: success
        Mock(status_code=200, iter_lines=...)
    ]

    # Act
    chunks = list(stability_manager.stream_with_recovery("Test"))

    # Assert
    assert len(chunks) > 0
    assert "expected_content" in "".join(chunks)
```

### Updating Mocks for New Ollama Versions

```python
# If Ollama API changes, update mock responses:
mock_response.json.return_value = {
    "response": "...",
    "done": True,
    "context": [...],  # New field?
    "model": "llama3:8b"
}
```

---

## Logs to Check

After running tests or manual integration, check logs for:

```bash
# Successful recovery
grep "ollama_server_recovering" data/logs/structured.jsonl

# Context reductions
grep "context_reduction" data/logs/structured.jsonl

# Fallback activations
grep "fallback" data/logs/structured.jsonl

# Health checks
grep "server_health_check" data/logs/structured.jsonl

# Payload validations
grep "ollama_request_payload" data/logs/structured.jsonl
```

---

## Success Criteria

✅ All 18 tests pass
✅ No unhandled exceptions in manual tests
✅ Soft fallback messages appear (not technical errors)
✅ Context reduction triggers on large prompts with 500 errors
✅ Recovery completes within 10 seconds
✅ Hugo remains responsive after Ollama restart
✅ Logs show recovery events clearly

---

**If all criteria met: Phase 5.2 is production-ready!** ✅
