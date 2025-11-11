"""
Startup Script
--------------
Main entry point for Hugo when run as a service or daemon.
"""

import asyncio
import signal
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import HugoLogger
from core.runtime_manager import RuntimeManager
from runtime.service_manager import ServiceManager


class HugoService:
    """
    Hugo service daemon.

    Runs Hugo as a background service with proper signal handling
    and graceful shutdown.
    """

    def __init__(self):
        """Initialize service"""
        self.logger = HugoLogger()
        self.service_manager = ServiceManager(self.logger)
        self.runtime_manager = None
        self.running = False

    async def start(self):
        """Start Hugo service"""
        self.running = True

        self.logger.log_event("startup", "service_starting", {})

        # Start external services
        await self.service_manager.start_services()

        # Initialize runtime
        config = {}  # TODO: Load config from file
        self.runtime_manager = RuntimeManager(config, self.logger)

        success = await self.runtime_manager.boot()

        if not success:
            self.logger.log_event("startup", "boot_failed", {})
            return False

        self.logger.log_event("startup", "service_started", {})

        # Main service loop
        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

        return True

    async def stop(self):
        """Stop Hugo service"""
        self.running = False

        self.logger.log_event("startup", "service_stopping", {})

        if self.runtime_manager:
            await self.runtime_manager.shutdown(graceful=True)

        await self.service_manager.stop_services()

        self.logger.log_event("startup", "service_stopped", {})

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(sig, frame):
            print("\n\nReceived signal, shutting down gracefully...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main service entry point"""
    service = HugoService()
    service.setup_signal_handlers()

    try:
        await service.start()
    except Exception as e:
        service.logger.log_error(e)
        print(f"Service error: {str(e)}")
        sys.exit(1)
    finally:
        await service.stop()


if __name__ == '__main__':
    asyncio.run(main())
