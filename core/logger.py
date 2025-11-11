"""
Hugo Logger
-----------
Structured event logging for Hugo's operations.

Log Categories:
- Events: System events, state changes
- Reflections: Self-reflection entries
- Performance: Metrics and diagnostics
- Errors: Exceptions and failures
- Security: Directive violations, access attempts
- User: User interactions and sessions

Log Levels:
- DEBUG: Detailed debugging information
- INFO: General informational messages
- WARNING: Warning messages
- ERROR: Error messages
- CRITICAL: Critical failures
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from enum import Enum


class LogCategory(Enum):
    """Categories of log entries"""
    EVENT = "event"
    REFLECTION = "reflection"
    PERFORMANCE = "performance"
    ERROR = "error"
    SECURITY = "security"
    USER = "user"


class HugoLogger:
    """
    Structured logging system for Hugo with categorization and persistence.

    Features:
    - Structured JSON logging
    - Category-based filtering
    - Automatic rotation
    - Real-time and batch logging
    - Performance metrics tracking
    """

    def __init__(self, log_dir: str = "data/logs"):
        """
        Initialize Hugo logger.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup Python logger
        self.logger = logging.getLogger("hugo")
        self.logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(
            self.log_dir / "hugo.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Structured log file for JSON entries
        self.structured_log_path = self.log_dir / "structured.jsonl"

    def log_event(self, category: str, event_type: str, data: Dict[str, Any],
                  level: str = "INFO"):
        """
        Log a structured event.

        Args:
            category: Event category (cognition, memory, etc.)
            event_type: Specific event type
            data: Event data dictionary
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "event_type": event_type,
            "level": level,
            "data": data
        }

        # Write to structured log
        with open(self.structured_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')

        # Also log to standard logger
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(f"[{category}] {event_type}: {json.dumps(data)}")

    async def log_reflection(self, reflection_data: Dict[str, Any]):
        """
        Log a reflection entry.

        Args:
            reflection_data: Reflection metadata and content
        """
        self.log_event(
            category="reflection",
            event_type="generated",
            data=reflection_data,
            level="INFO"
        )

    def log_performance(self, metric_name: str, value: float,
                       context: Optional[Dict[str, Any]] = None):
        """
        Log a performance metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
            context: Optional context data
        """
        data = {
            "metric": metric_name,
            "value": value,
            "context": context or {}
        }

        self.log_event(
            category="performance",
            event_type="metric",
            data=data,
            level="DEBUG"
        )

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Log an error with context.

        Args:
            error: Exception object
            context: Optional context data
        """
        data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }

        self.log_event(
            category="error",
            event_type="exception",
            data=data,
            level="ERROR"
        )

        # Also log full traceback to standard logger
        self.logger.exception("Exception occurred", exc_info=error)

    def log_security(self, event_type: str, data: Dict[str, Any]):
        """
        Log a security-related event.

        Args:
            event_type: Type of security event
            data: Event data
        """
        self.log_event(
            category="security",
            event_type=event_type,
            data=data,
            level="WARNING"
        )

    def log_user_interaction(self, session_id: str, interaction_type: str,
                           data: Dict[str, Any]):
        """
        Log a user interaction.

        Args:
            session_id: Session identifier
            interaction_type: Type of interaction
            data: Interaction data
        """
        data["session_id"] = session_id

        self.log_event(
            category="user",
            event_type=interaction_type,
            data=data,
            level="INFO"
        )

    def query_logs(self, category: Optional[str] = None,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  limit: int = 100) -> list:
        """
        Query structured logs with filters.

        Args:
            category: Filter by category
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum results

        Returns:
            List of log entries matching filters

        TODO: Implement efficient log querying (maybe use SQLite for logs?)
        """
        results = []

        if not self.structured_log_path.exists():
            return results

        with open(self.structured_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(results) >= limit:
                    break

                try:
                    entry = json.loads(line)

                    # Apply filters
                    if category and entry.get("category") != category:
                        continue

                    # TODO: Add time filtering

                    results.append(entry)
                except json.JSONDecodeError:
                    continue

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get logging statistics.

        Returns:
            Dictionary with log statistics
        """
        # TODO: Implement log statistics (count by category, errors, etc.)
        return {
            "total_entries": 0,
            "by_category": {},
            "by_level": {},
            "log_file_size": self.structured_log_path.stat().st_size if self.structured_log_path.exists() else 0
        }

    async def rotate_logs(self, max_size_mb: int = 100):
        """
        Rotate log files if they exceed size threshold.

        Args:
            max_size_mb: Maximum log file size in MB

        TODO: Implement log rotation with compression
        """
        self.log_event("logger", "rotation_triggered", {"max_size_mb": max_size_mb})
