"""
Pytest Configuration and Fixtures
----------------------------------
Shared fixtures for Hugo test suite.
"""

import pytest
import asyncio
from pathlib import Path


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_logger():
    """Mock HugoLogger for testing"""
    from unittest.mock import Mock

    logger = Mock()
    logger.log_event = Mock()
    logger.log_error = Mock()
    logger.log_reflection = Mock()

    return logger


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "log_level": "DEBUG",
        "session_retention_days": 7,
        "embedding_dimension": 384,
    }


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure"""
    data_dir = tmp_path / "data"
    (data_dir / "memory").mkdir(parents=True)
    (data_dir / "reflections").mkdir(parents=True)
    (data_dir / "logs").mkdir(parents=True)
    (data_dir / "backups").mkdir(parents=True)
    (data_dir / "vault").mkdir(parents=True)

    return data_dir
