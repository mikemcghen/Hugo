"""
Tests for demo_skill
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from main import execute, validate


@pytest.mark.asyncio
async def test_basic_execution():
    """Test basic skill execution"""
    context = {
        "parameters": {
            "message": "Test message",
            "repeat": 1
        },
        "session_id": "test",
        "logger": None
    }

    result = await execute(context)

    assert result["success"] is True
    assert "Test message" in result["result"]
    assert result["metadata"]["iterations"] == 1


@pytest.mark.asyncio
async def test_multiple_iterations():
    """Test skill with multiple iterations"""
    context = {
        "parameters": {
            "message": "Iteration test",
            "repeat": 3
        },
        "session_id": "test",
        "logger": None
    }

    result = await execute(context)

    assert result["success"] is True
    assert result["metadata"]["iterations"] == 3


@pytest.mark.asyncio
async def test_default_parameters():
    """Test skill with default parameters"""
    context = {
        "parameters": {},
        "session_id": "test",
        "logger": None
    }

    result = await execute(context)

    assert result["success"] is True
    assert "Hello from Hugo!" in result["result"]


@pytest.mark.asyncio
async def test_validation():
    """Test skill validation"""
    result = await validate()

    assert result["passed"] is True
    assert result["tests_failed"] == 0
    assert len(result["errors"]) == 0
