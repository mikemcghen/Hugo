"""
Demo Skill
==========
A demonstration skill showing Hugo's skill system architecture.

This skill demonstrates:
- Parameter handling
- Execution logging
- Result formatting
- Error handling
"""

import asyncio
from typing import Dict, Any


async def execute(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the demo skill.

    Args:
        context: Execution context containing:
            - parameters: Dict of skill parameters
            - session_id: Current session ID
            - logger: Logger instance
            - user: User information

    Returns:
        Result dictionary with:
            - success: bool indicating execution success
            - result: Execution result data
            - message: Human-readable message
            - metadata: Additional execution metadata
    """
    # Extract parameters
    parameters = context.get("parameters", {})
    message = parameters.get("message", "Hello from Hugo!")
    repeat = parameters.get("repeat", 1)

    logger = context.get("logger")
    session_id = context.get("session_id")

    # Log execution start
    if logger:
        logger.log_event("skill", "demo_skill_started", {
            "session_id": session_id,
            "parameters": parameters
        })

    try:
        # Execute skill logic
        output_lines = []

        output_lines.append("=" * 50)
        output_lines.append("  DEMO SKILL EXECUTION")
        output_lines.append("=" * 50)
        output_lines.append("")

        for i in range(repeat):
            output_lines.append(f"[{i + 1}/{repeat}] {message}")

        output_lines.append("")
        output_lines.append("=" * 50)
        output_lines.append("  Skill Parameters:")
        output_lines.append(f"    message: {message}")
        output_lines.append(f"    repeat: {repeat}")
        output_lines.append("=" * 50)

        result_text = "\n".join(output_lines)

        # Log success
        if logger:
            logger.log_event("skill", "demo_skill_completed", {
                "session_id": session_id,
                "success": True
            })

        return {
            "success": True,
            "result": result_text,
            "message": f"Demo skill executed successfully ({repeat} iteration(s))",
            "metadata": {
                "iterations": repeat,
                "output_length": len(result_text)
            }
        }

    except Exception as e:
        # Log error
        if logger:
            logger.log_error(e, {
                "skill": "demo_skill",
                "session_id": session_id
            })

        return {
            "success": False,
            "result": None,
            "message": f"Demo skill failed: {str(e)}",
            "metadata": {
                "error": str(e)
            }
        }


async def validate() -> Dict[str, Any]:
    """
    Validate the skill (called during skill validation/testing).

    Returns:
        Validation result dictionary
    """
    validation_results = {
        "passed": True,
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "errors": []
    }

    # Test 1: Basic execution
    try:
        context = {
            "parameters": {"message": "Test", "repeat": 1},
            "session_id": "validation",
            "logger": None
        }
        result = await execute(context)

        validation_results["tests_run"] += 1

        if result["success"]:
            validation_results["tests_passed"] += 1
        else:
            validation_results["tests_failed"] += 1
            validation_results["errors"].append("Basic execution failed")
            validation_results["passed"] = False

    except Exception as e:
        validation_results["tests_run"] += 1
        validation_results["tests_failed"] += 1
        validation_results["errors"].append(f"Basic execution exception: {str(e)}")
        validation_results["passed"] = False

    # Test 2: Multiple iterations
    try:
        context = {
            "parameters": {"message": "Test", "repeat": 3},
            "session_id": "validation",
            "logger": None
        }
        result = await execute(context)

        validation_results["tests_run"] += 1

        if result["success"] and result["metadata"]["iterations"] == 3:
            validation_results["tests_passed"] += 1
        else:
            validation_results["tests_failed"] += 1
            validation_results["errors"].append("Multiple iterations test failed")
            validation_results["passed"] = False

    except Exception as e:
        validation_results["tests_run"] += 1
        validation_results["tests_failed"] += 1
        validation_results["errors"].append(f"Multiple iterations exception: {str(e)}")
        validation_results["passed"] = False

    return validation_results


# Example of how to test the skill directly
if __name__ == "__main__":
    async def test():
        print("Testing demo skill...\n")

        context = {
            "parameters": {
                "message": "Hello, Hugo!",
                "repeat": 2
            },
            "session_id": "test_session",
            "logger": None
        }

        result = await execute(context)

        print(result["result"])
        print(f"\nStatus: {result['message']}")

    asyncio.run(test())
