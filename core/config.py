"""
Hugo Configuration Helper
--------------------------
Centralized configuration management for Hugo's operational modes and settings.

Modes:
- core: Minimal, stable, clean cognition pipeline (default)
- full: Advanced features (agent delegation, reflection, etc.)
"""

import os
from enum import Enum
from typing import Optional


class HugoMode(Enum):
    """Hugo operational modes"""
    CORE = "core"  # Minimal, stable pipeline
    FULL = "full"  # Full features (agent, reflection, etc.)


def get_hugo_mode() -> HugoMode:
    """
    Get current Hugo operational mode from environment.

    Returns:
        HugoMode enum value

    Example:
        >>> mode = get_hugo_mode()
        >>> if mode == HugoMode.CORE:
        ...     # Use core pipeline
    """
    mode_str = os.getenv("HUGO_MODE", "core").lower().strip()

    if mode_str == "full":
        return HugoMode.FULL
    else:
        # Default to core mode
        return HugoMode.CORE


def is_core_mode() -> bool:
    """
    Check if Hugo is in core mode.

    Returns:
        True if in core mode, False otherwise

    Example:
        >>> if is_core_mode():
        ...     # Use simplified pipeline
    """
    return get_hugo_mode() == HugoMode.CORE


def is_full_mode() -> bool:
    """
    Check if Hugo is in full mode.

    Returns:
        True if in full mode, False otherwise
    """
    return get_hugo_mode() == HugoMode.FULL


def get_config_summary() -> dict:
    """
    Get summary of current configuration.

    Returns:
        Dictionary with config details
    """
    return {
        "hugo_mode": get_hugo_mode().value,
        "is_core": is_core_mode(),
        "is_full": is_full_mode(),
        "model_engine": os.getenv("MODEL_ENGINE", "ollama"),
        "model_name": os.getenv("MODEL_NAME", "llama3:8b"),
        "ollama_api": os.getenv("OLLAMA_API", "http://localhost:11434/api/generate"),
    }
