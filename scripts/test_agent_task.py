#!/usr/bin/env python3
"""
Agent Task Test Script
======================
Tests Hugo's worker agent delegation system.

Verifies:
- Worker agent is initialized correctly
- Task delegation triggers on heavy work keywords
- Claude bridge auto-approves in sandbox mode
- Agent task history is tracked
"""

import asyncio
import sys
from pathlib import Path

# Add Hugo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.runtime_manager import RuntimeManager
from agents.worker_agent import WorkerAgent
from agents.claude_bridge import ClaudeBridge


async def test_worker_agent_initialization():
    """Test worker agent initialization"""
    print("=" * 60)
    print("TEST 1: Worker Agent Initialization")
    print("=" * 60)

    runtime = RuntimeManager()
    await runtime.initialize()

    try:
        worker = WorkerAgent(runtime)
        print(f"✓ Worker agent initialized")
        print(f"  Has logger: {worker.logger is not None}")
        print(f"  Has cognition: {worker.cognition is not None}")
        print(f"  Task history: {len(worker.task_history)} tasks")
        return True

    except Exception as e:
        print(f"✗ Worker agent initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_worker_agent_task():
    """Test worker agent task execution"""
    print("\n" + "=" * 60)
    print("TEST 2: Worker Agent Task Execution")
    print("=" * 60)

    runtime = RuntimeManager()
    await runtime.initialize()

    worker = WorkerAgent(runtime)

    # Test task
    task = "Implement a retry mechanism with exponential backoff for API calls"

    print(f"Task: {task}")
    print("\nExecuting...\n")

    try:
        result = await worker.run_task(task, context={"test_mode": True})

        print(f"\n✓ Task completed")
        print(f"  Result length: {len(result)} chars")
        print(f"  Task history: {len(worker.task_history)} tasks")

        # Show first 200 chars of result
        print(f"\nResult preview:")
        print(f"  {result[:200]}...")

        return True

    except Exception as e:
        print(f"\n✗ Task execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_delegation():
    """Test that cognition engine delegates to worker agent"""
    print("\n" + "=" * 60)
    print("TEST 3: Agent Delegation from Cognition")
    print("=" * 60)

    runtime = RuntimeManager()
    await runtime.initialize()

    if not runtime.cognition:
        print("✗ Cognition engine not initialized")
        return False

    # Prompts that should trigger delegation
    delegation_prompts = [
        "implement a function to calculate fibonacci numbers",
        "refactor this code to use async/await",
        "create a new feature for user authentication"
    ]

    for i, prompt in enumerate(delegation_prompts, 1):
        print(f"\n--- Test {i}/{len(delegation_prompts)} ---")
        print(f"Prompt: {prompt}")

        # Check if heavy work detected
        is_heavy_work = runtime.cognition._detect_heavy_work(prompt)
        print(f"Heavy work detected: {is_heavy_work}")

        if is_heavy_work:
            print("✓ Delegation would be triggered")
        else:
            print("✗ Delegation would NOT be triggered (unexpected)")
            return False

    print("\n✓ All delegation prompts correctly detected")
    return True


async def test_claude_bridge():
    """Test Claude Bridge auto-approval in sandbox mode"""
    print("\n" + "=" * 60)
    print("TEST 4: Claude Bridge (Sandbox Mode)")
    print("=" * 60)

    runtime = RuntimeManager()
    await runtime.initialize()

    bridge = ClaudeBridge(logger=runtime.logger if hasattr(runtime, 'logger') else None, sandbox_mode=True)

    print(f"Claude Bridge initialized")
    print(f"  Sandbox mode: True")
    print(f"  Auto-approve: True")

    # Test file change request
    print("\n--- File Change Request ---")
    result = await bridge.request_file_change(
        file_path="test/example.py",
        diff="+ def new_function():\n+     pass",
        explanation="Adding a new function for testing"
    )

    print(f"Approved: {result['approved']}")
    print(f"Reason: {result['reason']}")
    print(f"Executed: {result['executed']}")

    if result['approved'] and not result['executed']:
        print("✓ Sandbox mode: approved but not executed (expected)")
    else:
        print("✗ Unexpected behavior in sandbox mode")
        return False

    # Test code execution request
    print("\n--- Code Execution Request ---")
    result = await bridge.request_code_execution(
        code="print('Hello from worker agent')",
        language="python",
        explanation="Testing code execution request"
    )

    print(f"Approved: {result['approved']}")
    print(f"Executed: {result['executed']}")
    print(f"Reason: {result['reason']}")

    if result['approved'] and not result['executed']:
        print("✓ Sandbox mode: approved but not executed (expected)")
    else:
        print("✗ Unexpected behavior in sandbox mode")
        return False

    # Check request history
    print(f"\n✓ Request history: {len(bridge.get_request_history())} requests")

    return True


async def test_agent_configuration():
    """Test agent delegation configuration"""
    print("\n" + "=" * 60)
    print("TEST 5: Agent Configuration")
    print("=" * 60)

    import os

    agent_delegation = os.getenv("AGENT_DELEGATION_ENABLED", "false").lower()
    sandbox_mode = os.getenv("SANDBOX_APPROVAL_MODE", "manual").lower()

    print(f"AGENT_DELEGATION_ENABLED: {agent_delegation}")
    print(f"SANDBOX_APPROVAL_MODE: {sandbox_mode}")

    if agent_delegation == "true":
        print("✓ Agent delegation is enabled")
    else:
        print("⚠ Agent delegation is disabled")

    if sandbox_mode == "auto":
        print("✓ Sandbox auto-approval is enabled")
    else:
        print("⚠ Sandbox requires manual approval")

    return True


async def main():
    """Run all agent tests"""
    print("\n" + "=" * 60)
    print("HUGO AGENT SYSTEM VERIFICATION")
    print("=" * 60)

    try:
        # Test 1: Worker agent initialization
        test1 = await test_worker_agent_initialization()

        # Test 2: Worker agent task execution
        test2 = await test_worker_agent_task()

        # Test 3: Agent delegation
        test3 = await test_agent_delegation()

        # Test 4: Claude bridge
        test4 = await test_claude_bridge()

        # Test 5: Configuration
        test5 = await test_agent_configuration()

        if all([test1, test2, test3, test4, test5]):
            print("\n" + "=" * 60)
            print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("✗ SOME TESTS FAILED")
            print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
