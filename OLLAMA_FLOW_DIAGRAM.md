# Hugo Ollama Resilience - Flow Diagrams

## 1. Enhanced Inference Flow (Synchronous)

```
┌─────────────────────────────────────────────────────────────┐
│                    User Input Received                       │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              _perceive() - Typo Correction                   │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│         _assemble_context() - Retrieve History               │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  _synthesize() - Build Prompt                │
└────────────────────────┬────────────────────────────────────┘
                         ↓
                   Is Async Mode?
                    ↙        ↘
                YES          NO
                 ↓            ↓
    _local_infer_async()   _local_infer()
                 │            │
                 └────┬───────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│                  RETRY LOOP (Max 3 Attempts)                 │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  Attempt 1:                                             │ │
│ │    ├─ Log: ollama_inference_attempt                     │ │
│ │    ├─ POST to Ollama (timeout: 60s)                     │ │
│ │    └─ Result?                                           │ │
│ │         ├─ SUCCESS → Return Response                    │ │
│ │         ├─ TIMEOUT → Wait 2s, goto Attempt 2            │ │
│ │         ├─ CONNECTION_ERROR → Wait 2s, goto Attempt 2   │ │
│ │         └─ REQUEST_ERROR → Wait 2s, goto Attempt 2      │ │
│ │                                                           │ │
│ │  Attempt 2:                                             │ │
│ │    ├─ Log: ollama_retry (backoff=2s)                    │ │
│ │    ├─ POST to Ollama (timeout: 60s)                     │ │
│ │    └─ Result?                                           │ │
│ │         ├─ SUCCESS → Return Response                    │ │
│ │         └─ FAILURE → Wait 4s, goto Attempt 3            │ │
│ │                                                           │ │
│ │  Attempt 3:                                             │ │
│ │    ├─ Log: ollama_retry (backoff=4s)                    │ │
│ │    ├─ POST to Ollama (timeout: 60s)                     │ │
│ │    └─ Result?                                           │ │
│ │         ├─ SUCCESS → Return Response                    │ │
│ │         └─ FAILURE → Enter Fallback Mode                │ │
│ └─────────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         ↓
                  Response Ready?
                    ↙        ↘
                 YES          NO
                  ↓            ↓
         Generated Text   _fallback_response()
                  │            │
                  └────┬───────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│           _construct_output() - Package Response             │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│             Store in Memory (User + Assistant)               │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  Display to User in REPL                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Async vs Sync Comparison

### Synchronous Mode (OLLAMA_ASYNC_MODE=false)
```
User types message
    ↓
REPL BLOCKED while waiting for Ollama
    ↓ (2-60 seconds)
Response arrives
    ↓
REPL UNBLOCKED - displays response
    ↓
User can type again
```

### Asynchronous Mode (OLLAMA_ASYNC_MODE=true)
```
User types message
    ↓
Async inference starts
    ↓
REPL IMMEDIATELY AVAILABLE for next input
    ↓ (user can type while waiting)
Response arrives in background
    ↓
Response displayed when ready
    ↓
REPL still responsive
```

---

## 3. Error Handling Decision Tree

```
Ollama Request Sent
    ↓
┌───────────────────────────────────────┐
│         Response Received?            │
└───────┬───────────────────────────────┘
        ↓
     Success?
    ↙     ↘
  YES      NO
   ↓        ↓
Return   Error Type?
Result      │
            ├─ ReadTimeout
            │    ↓
            │  Duration > OLLAMA_TIMEOUT?
            │    ↓ YES
            │  Log: status=timeout
            │    ↓
            │  Retry < MAX_RETRIES?
            │    ↓ YES
            │  Wait: backoff^attempt
            │    ↓
            │  Retry Request
            │
            ├─ ConnectionError
            │    ↓
            │  Log: status=connection_error
            │    ↓
            │  Ollama not running?
            │    ↓
            │  Retry < MAX_RETRIES?
            │    ↓ YES
            │  Wait: backoff^attempt
            │    ↓
            │  Retry Request
            │
            ├─ RequestException
            │    ↓
            │  Log: status=request_error
            │    ↓
            │  HTTP error (500, 503, etc)?
            │    ↓
            │  Retry < MAX_RETRIES?
            │    ↓ YES
            │  Wait: backoff^attempt
            │    ↓
            │  Retry Request
            │
            └─ All Retries Exhausted
                 ↓
               Enter Fallback Mode
                 ↓
               Generate Graceful Message
                 ↓
               Store in Memory
                 ↓
               Display to User
```

---

## 4. Retry Backoff Timeline

### Scenario: 3 Timeouts (Worst Case)

```
Time (seconds)
0     10    20    30    40    50    60    70    80    90   100   110   120   130   140   150   160   170   180   190
│─────│─────│─────│─────│─────│─────│─────│─────│─────│─────│─────│─────│─────│─────│─────│─────│─────│─────│─────│
│                                                                                                                     │
├─────────────────────────────────────────────────────────┤ Attempt 1: Timeout (60s)
│                                                          └─ Log: status=timeout
│
├──┤ Backoff: 2^1 = 2s
│  └─ Log: retry, backoff_seconds=2
│
├─────────────────────────────────────────────────────────┤ Attempt 2: Timeout (60s)
│                                                          └─ Log: status=timeout
│
├────────┤ Backoff: 2^2 = 4s
│        └─ Log: retry, backoff_seconds=4
│
├─────────────────────────────────────────────────────────┤ Attempt 3: Timeout (60s)
│                                                          └─ Log: status=timeout
│
└─ Enter Fallback Mode
   └─ Log: ollama_fallback_mode
   └─ Return: "I'm having trouble connecting to my reasoning core..."

Total time: 60 + 2 + 60 + 4 + 60 = 186 seconds (3 minutes 6 seconds)
```

### Scenario: 2 Connection Errors + Success

```
Time (seconds)
0     1     2     3     4     5     6     7     8     9    10
│─────│─────│─────│─────│─────│─────│─────│─────│─────│─────│
│                                                              │
├┤ Attempt 1: ConnectionError (~0.1s)
│└─ Log: status=connection_error
│
├──┤ Backoff: 2^1 = 2s
│  └─ Log: retry, backoff_seconds=2
│
├┤ Attempt 2: ConnectionError (~0.1s)
│└─ Log: status=connection_error
│
├────────┤ Backoff: 2^2 = 4s
│        └─ Log: retry, backoff_seconds=4
│
├─────────┤ Attempt 3: SUCCESS (3.2s)
│         └─ Log: status=success, duration=3.2
│
└─ Return Response

Total time: 0.1 + 2 + 0.1 + 4 + 3.2 = 9.4 seconds
```

---

## 5. Memory Continuity Flow

### Normal Operation
```
User: "Hello"
    ↓
Store in memory (type: user_message)
    ↓
Ollama inference → SUCCESS
    ↓
Store in memory (type: assistant_message, content: "Hello! How can I help?")
    ↓
On exit → Reflection analyzes conversation
```

### Fallback Operation
```
User: "What's 2+2?"
    ↓
Store in memory (type: user_message)
    ↓
Ollama inference → ALL RETRIES FAILED
    ↓
Generate fallback: "I'm having trouble connecting..."
    ↓
Store in memory (type: assistant_message, content: "I'm having trouble...")
    ↓
On exit → Reflection analyzes conversation (including fallback)
```

**Result:** Conversation continuity maintained even during failures

---

## 6. Logging Flow

### Successful Inference (1 Attempt)
```
[2025-11-12 10:30:15] ollama_inference_attempt
  {attempt: 1, max_retries: 3, timeout: 60}

[2025-11-12 10:30:18] ollama_inference
  {attempt: 1, duration: 2.8, status: "success", response_length: 234}

[2025-11-12 10:30:18] ollama_inference_complete
  {response_length: 234, async_mode: true}
```

### Failed Inference with Retry
```
[2025-11-12 10:31:00] ollama_inference_attempt
  {attempt: 1, max_retries: 3, timeout: 60}

[2025-11-12 10:32:00] ollama_inference
  {attempt: 1, duration: 60.02, status: "timeout", error: "ReadTimeout"}

[2025-11-12 10:32:00] ollama_retry
  {attempt: 1, backoff_seconds: 2}

[2025-11-12 10:32:02] ollama_inference_attempt
  {attempt: 2, max_retries: 3, timeout: 60}

[2025-11-12 10:32:08] ollama_inference
  {attempt: 2, duration: 5.7, status: "success", response_length: 456}

[2025-11-12 10:32:08] ollama_inference_complete
  {response_length: 456, async_mode: true}
```

### Complete Failure
```
[2025-11-12 10:33:00] ollama_inference_attempt
  {attempt: 1, max_retries: 3, timeout: 60}

[2025-11-12 10:33:00] ollama_inference
  {attempt: 1, duration: 0.05, status: "connection_error"}

[2025-11-12 10:33:00] ollama_retry
  {attempt: 1, backoff_seconds: 2}

[2025-11-12 10:33:02] ollama_inference_attempt
  {attempt: 2, max_retries: 3, timeout: 60}

[2025-11-12 10:33:02] ollama_inference
  {attempt: 2, duration: 0.05, status: "connection_error"}

[2025-11-12 10:33:02] ollama_retry
  {attempt: 2, backoff_seconds: 4}

[2025-11-12 10:33:06] ollama_inference_attempt
  {attempt: 3, max_retries: 3, timeout: 60}

[2025-11-12 10:33:06] ollama_inference
  {attempt: 3, duration: 0.05, status: "connection_error"}

[2025-11-12 10:33:06] ollama_fallback_mode
  {total_attempts: 3, last_error: "ConnectionError: ..."}

[2025-11-12 10:33:06] ollama_inference_complete
  {response_length: 67, async_mode: true}
```

---

## 7. State Tracking

```
CognitionEngine State Variables:

ollama_available: bool
    ├─ True: Ollama last known to be working
    └─ False: Ollama unavailable (fallback mode)

last_connection_attempt: float | None
    ├─ timestamp of last attempt
    └─ Used for health check cooldown (future feature)

Usage:
    if self.ollama_available:
        # Try normal inference
    else:
        # Consider skipping retry (circuit breaker pattern)
```

---

## 8. Configuration Impact

### Low Timeout (Fast Fail)
```env
OLLAMA_TIMEOUT=10
OLLAMA_MAX_RETRIES=2
OLLAMA_RETRY_BACKOFF=1.5
```
**Effect:** Quick failure detection, less waiting, more frequent fallbacks

### High Reliability (Patient)
```env
OLLAMA_TIMEOUT=120
OLLAMA_MAX_RETRIES=5
OLLAMA_RETRY_BACKOFF=2
```
**Effect:** Tolerates slow responses, fewer fallbacks, longer waits

---

_Visual flow diagrams for Hugo's enhanced Ollama resilience_
