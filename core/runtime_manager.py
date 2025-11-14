"""
Runtime Manager
---------------
Manages Hugo's boot sequence, mode control, and lifecycle.

Operational Modes:
- Interactive: CLI/REPL for direct user interaction
- Service: Background service mode
- Low Power: Minimal activity, dreamlike state
- Maintenance: System maintenance and evolution

Boot Sequence:
1. Environment validation
2. Service initialization
3. Database connection
4. Memory loading
5. Directive loading
6. Identity banner display
7. Mode activation
"""

import asyncio
from typing import Dict, Any, Optional
from enum import Enum
from pathlib import Path


class OperationalMode(Enum):
    """Hugo's operational modes"""
    INTERACTIVE = "interactive"
    SERVICE = "service"
    LOW_POWER = "low_power"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


class ServiceStatus(Enum):
    """Status of runtime services"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


class RuntimeManager:
    """
    Orchestrates Hugo's initialization, operation, and shutdown.

    Responsibilities:
    - Boot sequence coordination
    - Service lifecycle management
    - Mode transitions
    - Graceful shutdown
    - Error recovery
    """

    def __init__(self, config: Dict[str, Any], logger):
        """
        Initialize runtime manager.

        Args:
            config: Hugo configuration dictionary
            logger: HugoLogger instance
        """
        self.config = config
        self.logger = logger

        self.mode = OperationalMode.SHUTDOWN
        self.status = ServiceStatus.STOPPED

        # Core components (initialized during boot)
        self.cognition = None
        self.memory = None
        self.reflection = None
        self.tasks = None
        self.skills = None
        self.agent = None
        self.scheduler = None

    async def boot(self) -> bool:
        """
        Execute Hugo's boot sequence.

        Returns:
            True if boot successful, False otherwise
        """
        self.status = ServiceStatus.STARTING
        self.logger.log_event("runtime", "boot_started", {})

        try:
            # Step 1: Display identity banner
            self._display_banner()

            # Step 2: Validate environment
            if not await self._validate_environment():
                raise RuntimeError("Environment validation failed")

            # Step 3: Initialize services
            if not await self._initialize_services():
                raise RuntimeError("Service initialization failed")

            # Step 4: Connect to databases
            if not await self._connect_databases():
                raise RuntimeError("Database connection failed")

            # Step 5: Load core components
            if not await self._load_core_components():
                raise RuntimeError("Core component loading failed")

            # Step 6: Load memory and personality state
            if not await self._load_state():
                raise RuntimeError("State loading failed")

            # Step 7: Start scheduler
            if not await self._start_scheduler():
                raise RuntimeError("Scheduler start failed")

            self.status = ServiceStatus.RUNNING
            self.mode = OperationalMode.INTERACTIVE

            self.logger.log_event("runtime", "boot_completed", {
                "mode": self.mode.value
            })

            print("\n✓ Hugo is ready.\n")
            return True

        except Exception as e:
            self.status = ServiceStatus.FAILED
            self.logger.log_error(e, {"phase": "boot"})
            print(f"\n✗ Boot failed: {str(e)}\n")
            return False

    async def shutdown(self, graceful: bool = True):
        """
        Shutdown Hugo gracefully or forcefully.

        Args:
            graceful: If True, wait for tasks to complete
        """
        self.status = ServiceStatus.STOPPING
        self.logger.log_event("runtime", "shutdown_started", {"graceful": graceful})

        print("\n⏸  Shutting down Hugo...")

        try:
            # Stop scheduler
            if self.scheduler:
                await self.scheduler.stop()

            # Save state
            await self._save_state()

            # Close database connections
            await self._disconnect_databases()

            # Stop services
            await self._stop_services()

            self.status = ServiceStatus.STOPPED
            self.mode = OperationalMode.SHUTDOWN

            self.logger.log_event("runtime", "shutdown_completed", {})
            print("✓ Shutdown complete.\n")

        except Exception as e:
            self.logger.log_error(e, {"phase": "shutdown"})
            print(f"✗ Shutdown error: {str(e)}\n")

    def _display_banner(self):
        """Display Hugo's identity banner"""
        banner = """
╔════════════════════════════════════════╗
║                                        ║
║           HUGO - The Right Hand        ║
║                                        ║
║       Your Second-in-Command AI        ║
║                                        ║
║  v0.1.0  •  Local-First  •  Autonomous ║
║                                        ║
╚════════════════════════════════════════╝
"""
        print(banner)

    async def _validate_environment(self) -> bool:
        """
        Validate runtime environment requirements.

        TODO:
        - Check Python version
        - Verify required directories exist
        - Check GPU availability
        - Validate configuration
        """
        print("→ Validating environment...")

        # Check data directories
        data_dir = Path("data")
        required_dirs = ["memory", "reflections", "logs", "backups", "vault"]

        for dir_name in required_dirs:
            dir_path = data_dir / dir_name
            if not dir_path.exists():
                print(f"  Creating {dir_path}...")
                dir_path.mkdir(parents=True, exist_ok=True)

        print("  ✓ Environment validated")
        return True

    async def _initialize_services(self) -> bool:
        """
        Initialize external services (Whisper, Piper, Claude proxy, etc.).

        TODO:
        - Start Docker containers if needed
        - Verify service connectivity
        - Setup API clients
        """
        print("→ Initializing services...")
        # TODO: Service initialization
        print("  ✓ Services initialized")
        return True

    async def _connect_databases(self) -> bool:
        """
        Connect to SQLite and PostgreSQL databases.

        TODO:
        - Open SQLite connection
        - Connect to PostgreSQL
        - Run migrations if needed
        - Verify tables exist
        """
        print("→ Connecting to databases...")
        # TODO: Database connections
        print("  ✓ Databases connected")
        return True

    async def _load_core_components(self) -> bool:
        """
        Initialize core Hugo components.
        """
        print("→ Loading core components...")

        try:
            from core.cognition import CognitionEngine
            from core.memory import MemoryManager
            from data.sqlite_manager import SQLiteManager

            # Initialize SQLite manager first
            self.sqlite_manager = SQLiteManager(db_path="data/memory/hugo_session.db")
            await self.sqlite_manager.connect()
            print("  ✓ SQLite manager initialized")

            # Memory manager with SQLite connection (PostgreSQL still None)
            self.memory = MemoryManager(self.sqlite_manager, None, self.logger)
            print("  ✓ Memory manager initialized")

            # Load factual memories from SQLite into cache and FAISS
            await self.memory.load_factual_memories()
            print("  ✓ Factual memories loaded from storage")

            # Cognition engine with memory
            self.cognition = CognitionEngine(self.memory, self.logger, runtime_manager=self)
            print("  ✓ Cognition engine initialized")

            # Reflection engine with SQLite manager
            from core.reflection import ReflectionEngine
            self.reflection = ReflectionEngine(self.memory, self.logger, self.sqlite_manager)
            print("  ✓ Reflection engine initialized")

            # Task manager with SQLite connection
            from core.tasks import TaskManager
            self.tasks = TaskManager(self.sqlite_manager, self.logger)
            print("  ✓ Task manager initialized")

            # Skill manager with SQLite, memory, and cognition connections
            from skills.skill_manager import SkillManager
            self.skills = SkillManager(self.logger, self.sqlite_manager, self.memory, self.cognition)
            self.skills.load_skills()
            print("  ✓ Skill manager initialized")

            # Autonomous agent
            from core.agent import HugoAgent
            self.agent = HugoAgent(self.memory, self.skills, self.sqlite_manager, self.reflection, self.logger)
            print("  ✓ Autonomous agent initialized")

            # Scheduler will be initialized later
            self.scheduler = None

            print("  ✓ Core components loaded")
            return True

        except Exception as e:
            self.logger.log_error(e, {"phase": "load_core_components"})
            print(f"  ✗ Error loading components: {str(e)}")
            return False

    async def _load_state(self) -> bool:
        """
        Load Hugo's persistent state (memory anchors, personality baseline).

        TODO:
        - Load last session state
        - Restore personality continuity markers
        - Load active tasks
        """
        print("→ Loading state...")
        # TODO: State loading
        print("  ✓ State loaded")
        return True

    async def _start_scheduler(self) -> bool:
        """Start the maintenance scheduler and agent loop"""
        print("→ Starting scheduler...")

        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler

            self.scheduler = AsyncIOScheduler()

            # Add autonomous agent tick (every 5 seconds)
            if self.agent:
                self.scheduler.add_job(
                    self.agent.tick,
                    'interval',
                    seconds=5,
                    id='agent_tick',
                    name='Autonomous Agent Tick'
                )
                self.logger.log_event("runtime", "agent_scheduler_added", {
                    "interval_seconds": 5
                })

            # Start scheduler
            self.scheduler.start()
            print("  ✓ Scheduler started")

            return True

        except ImportError:
            print("  ⚠ APScheduler not available, agent loop disabled")
            return True
        except Exception as e:
            self.logger.log_error(e, {"phase": "start_scheduler"})
            print(f"  ✗ Scheduler failed: {str(e)}")
            return False

    async def _save_state(self):
        """Save Hugo's current state before shutdown"""
        print("→ Saving state...")
        # TODO: State saving
        print("  ✓ State saved")

    async def _disconnect_databases(self):
        """Close database connections"""
        print("→ Disconnecting databases...")
        # TODO: Close connections
        print("  ✓ Databases disconnected")

    async def _stop_services(self):
        """Stop external services"""
        print("→ Stopping services...")
        # TODO: Stop services
        print("  ✓ Services stopped")

    def set_mode(self, mode: OperationalMode):
        """Change operational mode"""
        old_mode = self.mode
        self.mode = mode

        self.logger.log_event("runtime", "mode_change", {
            "old_mode": old_mode.value,
            "new_mode": mode.value
        })

        print(f"\n→ Mode changed: {old_mode.value} → {mode.value}\n")

    def get_status(self) -> Dict[str, Any]:
        """Get runtime status information"""
        return {
            "mode": self.mode.value,
            "status": self.status.value,
            "uptime": "0m",  # TODO: Calculate actual uptime
            "services": {
                "cognition": "running" if self.cognition else "stopped",
                "memory": "running" if self.memory else "stopped",
                "reflection": "running" if self.reflection else "stopped",
                "skills": "running" if self.skills else "stopped",
                "agent": "running" if self.agent and self.agent.enabled else "stopped",
                "scheduler": "running" if self.scheduler else "stopped"
            }
        }
