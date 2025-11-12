#!/usr/bin/env python3
"""
Test Boot Sequence
==================
Verify that RuntimeManager properly initializes CognitionEngine
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def test_boot():
    """Test the boot sequence"""
    print("=" * 60)
    print("Testing Hugo Boot Sequence")
    print("=" * 60)
    print()

    try:
        from core.logger import HugoLogger
        from core.runtime_manager import RuntimeManager

        # Initialize logger
        logger = HugoLogger()
        print("✓ Logger initialized")

        # Initialize runtime manager
        config = {}
        runtime = RuntimeManager(config, logger)
        print("✓ RuntimeManager created")

        # Boot Hugo
        print()
        print("Starting boot sequence...")
        print()
        success = await runtime.boot()

        if success:
            print()
            print("=" * 60)
            print("Boot Status Check:")
            print("=" * 60)
            print(f"  Runtime status: {runtime.status.value}")
            print(f"  Operational mode: {runtime.mode.value}")
            print(f"  Memory manager: {'✓ Initialized' if runtime.memory else '✗ Not initialized'}")
            print(f"  Directive filter: {'✓ Initialized' if runtime.directives else '✗ Not initialized'}")
            print(f"  Cognition engine: {'✓ Initialized' if runtime.cognition else '✗ Not initialized'}")
            print()

            if runtime.cognition:
                print("=" * 60)
                print("Testing Cognition Engine:")
                print("=" * 60)
                print(f"  Model engine: {runtime.cognition.model_engine}")
                print(f"  Model name: {runtime.cognition.model_name}")
                print(f"  Ollama API: {runtime.cognition.ollama_api}")
                print()

                # Test a simple input
                print("Processing test input...")
                response = await runtime.cognition.process_input(
                    "Hello, this is a test",
                    "test-session"
                )
                print(f"✓ Response received: {len(response.content)} chars")
                print(f"  Preview: {response.content[:100]}...")
                print()

                return True
            else:
                print("✗ Cognition engine not initialized")
                return False
        else:
            print("✗ Boot failed")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the test"""
    result = asyncio.run(test_boot())

    print("=" * 60)
    print("Result:", "✓ PASS" if result else "✗ FAIL")
    print("=" * 60)
    print()

    return 0 if result else 1


if __name__ == '__main__':
    sys.exit(main())
