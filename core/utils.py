"""
Utility Functions
-----------------
Helper functions and utilities for Hugo core systems.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid


def generate_session_id() -> str:
    """Generate a unique session identifier"""
    return f"session_{uuid.uuid4().hex[:12]}"


def generate_message_id() -> str:
    """Generate a unique message identifier"""
    return f"msg_{uuid.uuid4().hex[:8]}"


def hash_text(text: str) -> str:
    """Generate SHA-256 hash of text"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length with suffix"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_timestamp(dt: Optional[datetime] = None, format: str = "iso") -> str:
    """
    Format datetime as string.

    Args:
        dt: DateTime object (defaults to now)
        format: Format type (iso, human, short)
    """
    if dt is None:
        dt = datetime.now()

    if format == "iso":
        return dt.isoformat()
    elif format == "human":
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    elif format == "short":
        return dt.strftime("%m/%d %H:%M")
    else:
        return dt.isoformat()


def parse_cron(expression: str) -> Dict[str, Any]:
    """
    Parse cron expression into components.

    Args:
        expression: Cron expression (e.g., "0 2 * * *")

    Returns:
        Dictionary with parsed components

    TODO: Implement proper cron parsing
    """
    parts = expression.split()

    return {
        "minute": parts[0] if len(parts) > 0 else "*",
        "hour": parts[1] if len(parts) > 1 else "*",
        "day": parts[2] if len(parts) > 2 else "*",
        "month": parts[3] if len(parts) > 3 else "*",
        "weekday": parts[4] if len(parts) > 4 else "*"
    }


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe filesystem use"""
    import re
    # Remove or replace unsafe characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return safe.strip()


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge (takes precedence)

    Returns:
        Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate basic text similarity (0.0 to 1.0).

    TODO: Implement proper similarity metric (cosine, jaccard, etc.)

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score
    """
    # Placeholder: simple character overlap
    set1 = set(text1.lower())
    set2 = set(text2.lower())

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Extract top keywords from text.

    TODO: Implement proper keyword extraction (TF-IDF, etc.)

    Args:
        text: Input text
        top_n: Number of keywords to extract

    Returns:
        List of keywords
    """
    # Placeholder: simple word frequency
    import re
    from collections import Counter

    words = re.findall(r'\b\w+\b', text.lower())

    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    words = [w for w in words if w not in stop_words and len(w) > 3]

    counter = Counter(words)
    return [word for word, count in counter.most_common(top_n)]


def validate_json(json_string: str) -> tuple[bool, Optional[Dict]]:
    """
    Validate JSON string.

    Args:
        json_string: JSON string to validate

    Returns:
        Tuple of (is_valid, parsed_dict or None)
    """
    try:
        parsed = json.loads(json_string)
        return True, parsed
    except json.JSONDecodeError:
        return False, None


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2h 30m", "45s")
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.0f}m"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.0f}h {minutes:.0f}m"


def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed configuration dictionary
    """
    import yaml

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml_config(data: Dict[str, Any], path: str):
    """
    Save dictionary to YAML file.

    Args:
        data: Dictionary to save
        path: Output path
    """
    import yaml

    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
