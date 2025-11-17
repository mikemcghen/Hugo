# Phase 5.1: Streaming Stability Testing

## Running the Tests

### Prerequisites

```bash
pip install pytest pytest-asyncio
```

### Run All Streaming Tests

```bash
# From project root
pytest tests/test_repl_streaming.py tests/test_cognition_streaming.py -v
```

### Run Individual Test Suites

```bash
# REPL streaming tests only
pytest tests/test_repl_streaming.py -v

# Cognition streaming tests only
pytest tests/test_cognition_streaming.py -v
```

### Run Specific Tests

```bash
# Test the critical fix (no TypeError on async for)
pytest tests/test_repl_streaming.py::test_repl_no_crash_on_responsepackage -v

# Test skill bypass wrapping
pytest tests/test_cognition_streaming.py::test_cognition_skill_bypass_wrapped -v

# Test unified interface
pytest tests/test_repl_streaming.py::test_repl_unified_interface -v
```

## Test Coverage

### REPL Streaming Tests (`test_repl_streaming.py`)

✅ **test_repl_streaming_normal** - Normal streaming conversation
✅ **test_repl_skill_bypass_single_shot** - Skill bypass (web_search) responses
✅ **test_repl_no_crash_on_responsepackage** - Critical: no TypeError
✅ **test_repl_unified_interface** - Consistent interface for both modes
✅ **test_async_helpers_stream_single** - stream_single utility
✅ **test_async_helpers_ensure_async_iterator** - ensure_async_iterator utility
✅ **test_async_helpers_is_async_iterator** - is_async_iterator utility

### Cognition Streaming Tests (`test_cognition_streaming.py`)

✅ **test_cognition_force_streaming_interface** - Always returns async iterator
✅ **test_cognition_streaming_normal_conversation** - Normal streaming
✅ **test_cognition_skill_bypass_wrapped** - Skill bypass wrapped
✅ **test_cognition_non_streaming_wrapped** - Non-streaming wrapped
✅ **test_cognition_extraction_synthesis_wrapped** - Extraction mode wrapped
✅ **test_no_type_error_on_iteration** - Critical: no TypeError on any type
✅ **test_ensure_async_iterator_passthrough** - Passthrough for existing iterators
✅ **test_ensure_async_iterator_wrapping** - Wrapping for non-iterators
✅ **test_streaming_vs_non_streaming_behavior** - Consistent behavior

## Expected Results

All tests should **PASS**. If any tests fail:

1. Check that `runtime/utils/async_helpers.py` exists
2. Verify `cognition.py` changes are applied
3. Verify `repl.py` changes are applied

## Manual Integration Test

After running automated tests, verify with Hugo directly:

```bash
# Start Hugo
python main.py

# Test 1: Normal conversation (should stream if enabled)
> Hello Hugo

# Test 2: Skill bypass (should NOT crash with TypeError)
> When is Pittsburgh Light Up Night?

# Test 3: Short message (non-streaming, should NOT crash)
> Thanks

# Test 4: Jarvis mode template (if Phase 5 applied)
> Continue
```

**Expected behavior:**
- No `TypeError: 'async for' requires an object with __aiter__` errors
- All responses display correctly
- Both streaming and non-streaming work seamlessly

## Debugging

If you encounter issues:

```bash
# Run with verbose output
pytest tests/test_repl_streaming.py -vv

# Run with stdout capture disabled (see print statements)
pytest tests/test_repl_streaming.py -v -s

# Run with pdb on failure
pytest tests/test_repl_streaming.py --pdb
```

## CI/CD Integration

Add to your test suite:

```bash
# In your CI pipeline
pytest tests/test_*_streaming.py --junitxml=test-results.xml
```
