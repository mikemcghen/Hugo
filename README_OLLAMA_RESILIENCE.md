# Hugo Ollama Resilience Enhancement

> **Enterprise-grade fault tolerance for Hugo's Ollama LLM integration**

**Version:** 1.0
**Date:** 2025-11-12
**Status:** ✅ Production Ready

---

## Quick Start

### 1. Install Dependencies
```bash
pip install aiohttp==3.9.1
```

### 2. Configure (Optional)
Edit `.env`:
```env
OLLAMA_TIMEOUT=60          # Seconds to wait per request
OLLAMA_MAX_RETRIES=3       # Maximum retry attempts
OLLAMA_RETRY_BACKOFF=2     # Exponential backoff multiplier
OLLAMA_ASYNC_MODE=true     # Enable non-blocking inference
```

### 3. Test
```bash
python -m runtime.cli shell
You: Hello Hugo!
```

---

## What's New

### ✨ Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Configurable Timeouts** | Set request timeout via `OLLAMA_TIMEOUT` | Prevent infinite hangs |
| **Retry with Backoff** | Exponential retry on failures (2s, 4s, 8s) | Handle transient errors |
| **Async Inference** | Non-blocking HTTP via aiohttp | Keep REPL responsive |
| **Graceful Fallback** | User-friendly messages when Ollama unavailable | Never crash |
| **Enhanced Logging** | Track all attempts with duration and status | Debug easily |
| **Memory Continuity** | Store messages even during failures | Preserve history |

---

## How It Works

### Before Enhancement ❌
```
User: "Tell me about AI"
→ Ollama timeout (no retry)
→ Hugo crashes or hangs indefinitely
❌ Bad user experience
```

### After Enhancement ✅
```
User: "Tell me about AI"
→ Attempt 1: Timeout (60s) → Log: status=timeout
→ Wait 2 seconds (backoff)
→ Attempt 2: Timeout (60s) → Log: status=timeout
→ Wait 4 seconds (backoff)
→ Attempt 3: Success! (3.2s) → Log: status=success
Hugo: "Artificial Intelligence is..."
✅ Resilient and transparent
```

---

## Retry Logic

### Exponential Backoff Formula
```
wait_time = OLLAMA_RETRY_BACKOFF ^ attempt_number

Example with OLLAMA_RETRY_BACKOFF=2:
  Attempt 1 fails → Wait 2^1 = 2 seconds
  Attempt 2 fails → Wait 2^2 = 4 seconds
  Attempt 3 fails → Wait 2^3 = 8 seconds
  All failed → Fallback mode
```

### Error Handling Matrix

| Error Type | Example | Retry? | Fallback? |
|------------|---------|--------|-----------|
| `ReadTimeout` | Request exceeds `OLLAMA_TIMEOUT` | ✅ Yes | After MAX_RETRIES |
| `ConnectionError` | Ollama not running | ✅ Yes | After MAX_RETRIES |
| `RequestException` | HTTP 500, 503 | ✅ Yes | After MAX_RETRIES |
| `HTTPError 404` | Model not found | ❌ No (future) | Immediate |
| `General Exception` | Unexpected error | ✅ Yes | After MAX_RETRIES |

---

## Async Mode

### Why Async?

**Problem:** Long Ollama responses block the REPL
```
User types message → Wait 30 seconds → Response appears
(User cannot type during this time)
```

**Solution:** Async inference keeps REPL responsive
```
User types message → Can immediately type next message
(Response appears when ready, doesn't block)
```

### Configuration
```env
OLLAMA_ASYNC_MODE=true   # Enable async (recommended)
OLLAMA_ASYNC_MODE=false  # Use sync (for debugging)
```

### Fallback Behavior
If `aiohttp` not installed:
- Async mode uses `asyncio.run_in_executor()` with sync version
- Still non-blocking, slightly less efficient

---

## Fallback Mode

### When Triggered
- All `OLLAMA_MAX_RETRIES` attempts exhausted
- Ollama server unreachable
- Network connectivity issues

### Behavior
1. Generate graceful message (4 rotating options)
2. Store user message in memory
3. Store fallback message as assistant response
4. Mark `ollama_available = False`
5. Continue conversation

### Fallback Messages
```
1. "I'm having trouble connecting to my reasoning core. Let's pause for a moment."
2. "My reasoning system seems to be taking a break. Could you try again in a moment?"
3. "I'm experiencing some difficulty accessing my core processes right now."
4. "Connection to my inference engine is temporarily unavailable. Please give me a moment."
```

**Note:** Messages rotate based on timestamp for variety

---

## Logging

### Log Events

All inference attempts are logged with:
- **attempt** - Current attempt number
- **duration** - Time taken (seconds)
- **status** - success, timeout, connection_error, request_error
- **error** - Error message (if failed)

### Example Success Log
```json
{
  "event": "ollama_inference",
  "attempt": 1,
  "duration": 2.8,
  "status": "success",
  "response_length": 234
}
```

### Example Failure Log
```json
{
  "event": "ollama_inference",
  "attempt": 2,
  "duration": 60.02,
  "status": "timeout",
  "error": "ReadTimeout: timed out"
}
```

### Example Retry Log
```json
{
  "event": "ollama_retry",
  "attempt": 2,
  "backoff_seconds": 4
}
```

---

## Configuration Guide

### Presets

#### Fast Response (Recommended)
```env
OLLAMA_TIMEOUT=30
OLLAMA_MAX_RETRIES=2
OLLAMA_RETRY_BACKOFF=1.5
OLLAMA_ASYNC_MODE=true
```
- Quick failure detection
- Less waiting time
- Good for development

#### High Reliability (Production)
```env
OLLAMA_TIMEOUT=120
OLLAMA_MAX_RETRIES=5
OLLAMA_RETRY_BACKOFF=2
OLLAMA_ASYNC_MODE=true
```
- Tolerates slow responses
- Maximum resilience
- Good for production

#### Debug Mode
```env
OLLAMA_TIMEOUT=10
OLLAMA_MAX_RETRIES=1
OLLAMA_RETRY_BACKOFF=0
OLLAMA_ASYNC_MODE=false
```
- Fast failure
- No retries (see errors quickly)
- Synchronous for easier debugging

---

## Testing Scenarios

### Scenario 1: Normal Operation
```bash
# Start Ollama
ollama serve

# Start Hugo
python -m runtime.cli shell

You: Hello Hugo!
Hugo: Hello! I'm Hugo, your local AI assistant...

✅ Expected: Successful response in 2-3 seconds
```

### Scenario 2: Slow Response
```bash
# Ollama running but slow network

You: Write a detailed essay
→ Attempt 1: Success after 45 seconds

Hugo: [Essay appears]

✅ Expected: Single attempt succeeds (within timeout)
```

### Scenario 3: Temporary Network Issue
```bash
# Network flaky, intermittent drops

You: What's the weather?
→ Attempt 1: Timeout
→ Wait 2 seconds
→ Attempt 2: Success

Hugo: [Response]

✅ Expected: Retry succeeds, transparent to user
```

### Scenario 4: Ollama Not Running
```bash
# Stop Ollama first
# pkill ollama

You: Calculate 2+2
→ Attempt 1: ConnectionError
→ Wait 2 seconds
→ Attempt 2: ConnectionError
→ Wait 4 seconds
→ Attempt 3: ConnectionError
→ Fallback mode

Hugo: I'm having trouble connecting to my reasoning core...

✅ Expected: Graceful fallback message
```

### Scenario 5: Ollama Recovery
```bash
# After Scenario 4, start Ollama
ollama serve

You: Try again?
→ Attempt 1: Success

Hugo: [Normal response]

✅ Expected: Automatic recovery, no restart needed
```

---

## Memory and Reflection

### Behavior During Failures

**User messages:** Always stored
**Fallback responses:** Stored as assistant messages
**Reflection:** Analyzes entire conversation including fallbacks

### Example Reflection After Failure
```
============================================================
SESSION REFLECTION
============================================================

The conversation included a greeting and a technical question.
Hugo experienced a temporary connection issue but maintained
conversation continuity through fallback responses.

Key Insights:
  • User remained patient during connection disruption
  • Fallback mode preserved conversation flow
  • Automatic recovery demonstrated resilience

Patterns:
  • User prefers conversational interaction
  • Willing to retry after errors

============================================================
```

---

## Troubleshooting

### Issue: Constant "connection refused"
**Symptoms:** Every request fails with ConnectionError

**Diagnosis:**
```bash
curl http://localhost:11434/api/version
```

**Solution:**
```bash
ollama serve
ollama pull llama3:8b
```

---

### Issue: Frequent timeouts
**Symptoms:** Requests timeout even when Ollama running

**Diagnosis:** Timeout too short for model/hardware

**Solution:** Increase timeout
```env
OLLAMA_TIMEOUT=120  # Give more time
```

---

### Issue: Long wait times during failures
**Symptoms:** Takes 3+ minutes to get fallback

**Diagnosis:** Too many retries or high timeout

**Solution:** Reduce retries or timeout
```env
OLLAMA_MAX_RETRIES=2    # Fewer attempts
OLLAMA_RETRY_BACKOFF=1.5  # Shorter backoff
```

---

### Issue: Async mode not working
**Symptoms:** Log shows "async_fallback_sync"

**Diagnosis:** aiohttp not installed

**Solution:**
```bash
pip install aiohttp==3.9.1
```

---

### Issue: Hugo stops responding
**Symptoms:** No response after message

**Diagnosis:** Check logs for errors

**Solution:**
```bash
# Check Hugo logs
tail -f data/logs/hugo.log

# Look for:
# - ollama_inference events
# - Error stack traces
# - Connection refused messages
```

---

## Performance

### Latency Breakdown

**Successful Inference (No Retries):**
- Perception: ~50ms
- Context assembly: ~5ms
- Ollama inference: 2-5s (varies by prompt)
- Output construction: ~50ms
- Memory storage: ~100ms
- **Total: ~2-5 seconds**

**Failed Inference (3 Timeouts, 60s each):**
- Attempt 1: 60s + log
- Backoff: 2s
- Attempt 2: 60s + log
- Backoff: 4s
- Attempt 3: 60s + log
- Fallback: <10ms
- **Total: ~186 seconds (3 minutes)**

**Connection Errors (Fast Fail):**
- Attempt 1: 0.1s + log
- Backoff: 2s
- Attempt 2: 0.1s + log
- Backoff: 4s
- Attempt 3: 0.1s + log
- Fallback: <10ms
- **Total: ~6.3 seconds**

---

## Code Reference

### Key Files
- **[.env](.env)** - Configuration
- **[core/cognition.py](core/cognition.py)** - Resilience logic
- **[requirements.txt](requirements.txt)** - Dependencies

### Key Methods
- `_local_infer()` - Synchronous inference with retry (Line 226)
- `_local_infer_async()` - Async inference with retry (Line 392)
- `_fallback_response()` - Graceful fallback messages (Line 364)

### Key Configuration
- `OLLAMA_TIMEOUT` - Request timeout in seconds
- `OLLAMA_MAX_RETRIES` - Maximum retry attempts
- `OLLAMA_RETRY_BACKOFF` - Exponential backoff multiplier
- `OLLAMA_ASYNC_MODE` - Enable async inference

---

## Documentation

| Document | Purpose |
|----------|---------|
| **[OLLAMA_RESILIENCE.md](OLLAMA_RESILIENCE.md)** | Comprehensive guide (600+ lines) |
| **[RESILIENCE_SUMMARY.md](RESILIENCE_SUMMARY.md)** | Executive summary |
| **[OLLAMA_FLOW_DIAGRAM.md](OLLAMA_FLOW_DIAGRAM.md)** | Visual flow diagrams |
| **[README_OLLAMA_RESILIENCE.md](README_OLLAMA_RESILIENCE.md)** | This quick reference |

---

## FAQ

**Q: Do I need to install aiohttp?**
A: Optional. If not installed, async mode falls back to sync in executor.

**Q: What happens if Ollama crashes during a conversation?**
A: Hugo enters fallback mode, stores the exchange, and auto-recovers when Ollama restarts.

**Q: Can I disable retries?**
A: Yes, set `OLLAMA_MAX_RETRIES=1` to disable.

**Q: Will fallback responses show in reflection?**
A: Yes, reflections analyze all conversation turns including fallbacks.

**Q: Can I customize fallback messages?**
A: Yes, edit `_fallback_response()` in [core/cognition.py](core/cognition.py#L364).

**Q: Does async mode work on Windows?**
A: Yes, aiohttp is cross-platform.

**Q: What if my model is slow?**
A: Increase `OLLAMA_TIMEOUT` to 120 or higher.

**Q: How do I monitor retry attempts?**
A: Check logs for `ollama_retry` events with backoff_seconds.

---

## Compatibility

| Component | Version | Required |
|-----------|---------|----------|
| Python | ≥3.11 | ✅ |
| Ollama | Latest | ✅ |
| requests | ≥2.31 | ✅ |
| aiohttp | ≥3.9.1 | ⚠️ Optional |
| Hugo | 2025-11-12+ | ✅ |

---

## Support

### Reporting Issues
If you encounter problems:
1. Check logs: `data/logs/hugo.log`
2. Verify Ollama: `curl http://localhost:11434/api/version`
3. Check configuration: `.env` file
4. Review documentation: [OLLAMA_RESILIENCE.md](OLLAMA_RESILIENCE.md)

### Example Issue Report
```
Hugo Version: 2025-11-12
Python Version: 3.11
Ollama Status: Running
aiohttp Installed: Yes

Issue: Constant timeouts even with Ollama running
Config:
  OLLAMA_TIMEOUT=60
  OLLAMA_MAX_RETRIES=3

Logs:
  [paste relevant log lines]
```

---

## Summary

Hugo's Ollama integration now features:

✅ **Never crashes** on connection failures
✅ **Automatically retries** with intelligent backoff
✅ **Non-blocking async mode** keeps REPL responsive
✅ **Graceful fallback** when Ollama unavailable
✅ **Comprehensive logging** for debugging
✅ **Memory continuity** preserved during failures
✅ **Production-ready** reliability

**Ready for deployment!**

---

**Version:** 1.0
**Status:** ✅ Production Ready
**Date:** 2025-11-12

_Hugo Ollama Resilience Enhancement_
