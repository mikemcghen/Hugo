"""
Hugo CLI
--------
Command-line interface for Hugo AI assistant.

Commands:
- hugo up: Start Hugo services
- hugo down: Stop Hugo services
- hugo rebuild: Rebuild and restart services
- hugo reflect: Generate reflection report
- hugo skill: Manage skills (--new, --list, --validate)
- hugo status: Show system status
- hugo log: View logs
- hugo config: Manage configuration
- hugo shell: Enter interactive REPL
"""

import asyncio
import argparse
import sys
from typing import Optional
from pathlib import Path

# Import core components
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import HugoLogger
from core.runtime_manager import RuntimeManager, OperationalMode
from runtime.service_manager import ServiceManager


class HugoCLI:
    """
    Hugo's command-line interface.

    Provides utility commands for managing Hugo and a conversational
    REPL mode for direct interaction.
    """

    def __init__(self):
        """Initialize CLI"""
        self.logger = HugoLogger()
        self.runtime_manager = None
        self.service_manager = ServiceManager(self.logger)

    def run(self, argv: Optional[list] = None):
        """
        Main CLI entry point.

        Args:
            argv: Command-line arguments (defaults to sys.argv)
        """
        parser = argparse.ArgumentParser(
            prog='hugo',
            description='Hugo - Your Local-First AI Assistant',
            epilog='For more information, visit: https://github.com/yourusername/hugo'
        )

        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # hugo up
        up_parser = subparsers.add_parser('up', help='Start Hugo services')
        up_parser.add_argument('--detach', '-d', action='store_true',
                              help='Run in detached mode')

        # hugo down
        down_parser = subparsers.add_parser('down', help='Stop Hugo services')
        down_parser.add_argument('--force', '-f', action='store_true',
                                help='Force immediate shutdown')

        # hugo rebuild
        rebuild_parser = subparsers.add_parser('rebuild', help='Rebuild and restart')

        # hugo reflect
        reflect_parser = subparsers.add_parser('reflect', help='Generate reflection')
        reflect_parser.add_argument('--days', type=int, default=1,
                                   help='Days to reflect on (default: 1)')
        reflect_parser.add_argument('--type', choices=['session', 'macro', 'performance'],
                                   default='macro', help='Reflection type')

        # hugo skill
        skill_parser = subparsers.add_parser('skill', help='Manage skills')
        skill_group = skill_parser.add_mutually_exclusive_group()
        skill_group.add_argument('--new', metavar='NAME', help='Create new skill')
        skill_group.add_argument('--list', action='store_true', help='List all skills')
        skill_group.add_argument('--validate', metavar='NAME', help='Validate skill')
        skill_group.add_argument('--remove', metavar='NAME', help='Remove skill')

        # hugo status
        status_parser = subparsers.add_parser('status', help='Show system status')
        status_parser.add_argument('--verbose', '-v', action='store_true',
                                  help='Show detailed status')

        # hugo log
        log_parser = subparsers.add_parser('log', help='View logs')
        log_parser.add_argument('--category', choices=['event', 'reflection', 'performance', 'error', 'security', 'user'],
                               help='Filter by category')
        log_parser.add_argument('--tail', type=int, default=50,
                               help='Number of lines to show (default: 50)')
        log_parser.add_argument('--follow', '-f', action='store_true',
                               help='Follow log output')

        # hugo config
        config_parser = subparsers.add_parser('config', help='Manage configuration')
        config_parser.add_argument('--show', action='store_true', help='Show current config')
        config_parser.add_argument('--edit', action='store_true', help='Edit config file')

        # hugo shell
        shell_parser = subparsers.add_parser('shell', help='Interactive REPL')

        # Parse arguments
        args = parser.parse_args(argv)

        if not args.command:
            parser.print_help()
            return

        # Execute command
        try:
            asyncio.run(self._execute_command(args))
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
        except Exception as e:
            self.logger.log_error(e)
            print(f"\nError: {str(e)}")
            sys.exit(1)

    async def _execute_command(self, args):
        """Execute the parsed command"""

        if args.command == 'up':
            await self.cmd_up(detach=args.detach)

        elif args.command == 'down':
            await self.cmd_down(force=args.force)

        elif args.command == 'rebuild':
            await self.cmd_rebuild()

        elif args.command == 'reflect':
            await self.cmd_reflect(days=args.days, reflect_type=args.type)

        elif args.command == 'skill':
            await self.cmd_skill(args)

        elif args.command == 'status':
            await self.cmd_status(verbose=args.verbose)

        elif args.command == 'log':
            await self.cmd_log(category=args.category, tail=args.tail, follow=args.follow)

        elif args.command == 'config':
            await self.cmd_config(show=args.show, edit=args.edit)

        elif args.command == 'shell':
            await self.cmd_shell()

    # Command implementations

    async def cmd_up(self, detach: bool = False):
        """Start Hugo services"""
        print("Starting Hugo services...\n")

        # Start Docker services
        await self.service_manager.start_services()

        # Initialize runtime
        config = {}  # TODO: Load config from file
        self.runtime_manager = RuntimeManager(config, self.logger)

        success = await self.runtime_manager.boot()

        if not success:
            print("\n✗ Failed to start Hugo")
            return

        if detach:
            print("Hugo is running in the background.")
            print("Use 'hugo shell' to interact or 'hugo down' to stop.")
        else:
            # Stay in foreground
            print("Hugo is running. Press Ctrl+C to stop.")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                await self.cmd_down()

    async def cmd_down(self, force: bool = False):
        """Stop Hugo services"""
        print("\nStopping Hugo services...\n")

        if self.runtime_manager:
            await self.runtime_manager.shutdown(graceful=not force)

        await self.service_manager.stop_services()

        print("✓ Hugo stopped successfully")

    async def cmd_rebuild(self):
        """Rebuild and restart Hugo"""
        print("Rebuilding Hugo...\n")

        await self.cmd_down()
        await self.service_manager.rebuild_services()
        await self.cmd_up()

    async def cmd_reflect(self, days: int = 1, reflect_type: str = 'macro'):
        """Generate reflection report"""
        print(f"Generating {reflect_type} reflection for {days} day(s)...\n")

        # TODO: Initialize runtime if not running
        # TODO: Call reflection engine
        # TODO: Format and display reflection

        print("Reflection generation not yet implemented")

    async def cmd_skill(self, args):
        """Manage skills"""

        if args.new:
            await self._skill_new(args.new)
        elif args.list:
            await self._skill_list()
        elif args.validate:
            await self._skill_validate(args.validate)
        elif args.remove:
            await self._skill_remove(args.remove)

    async def _skill_new(self, name: str):
        """Create new skill scaffold"""
        print(f"Creating new skill: {name}\n")

        from core.utils import sanitize_filename

        safe_name = sanitize_filename(name)
        skill_dir = Path("skills") / safe_name

        if skill_dir.exists():
            print(f"✗ Skill '{name}' already exists")
            return

        # Create skill directory structure
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "tests").mkdir(exist_ok=True)

        # Create skill.yaml
        skill_yaml = f"""name: {safe_name}
version: "0.1.0"
description: "A new Hugo skill"
author: "User"
created: "2025-11-11"

triggers:
  - type: manual
    command: "{safe_name}"

parameters: []

validation:
  tests: true
  sandboxed: true
"""
        (skill_dir / "skill.yaml").write_text(skill_yaml, encoding='utf-8')

        # Create main.py
        main_py = f'''"""
{name} Skill
{'=' * (len(name) + 6)}
Skill implementation.
"""

async def execute(context: dict):
    """
    Execute the skill.

    Args:
        context: Execution context with parameters

    Returns:
        Result dictionary
    """
    print(f"Executing {safe_name} skill...")

    # TODO: Implement skill logic

    return {{
        "success": True,
        "result": "Skill executed successfully"
    }}
'''
        (skill_dir / "main.py").write_text(main_py, encoding='utf-8')

        # Create test file
        test_py = f'''"""
Tests for {name} skill
"""

import pytest


@pytest.mark.asyncio
async def test_execute():
    """Test skill execution"""
    from ..main import execute

    context = {{}}
    result = await execute(context)

    assert result["success"] is True
'''
        (skill_dir / "tests" / "test_main.py").write_text(test_py, encoding='utf-8')

        print(f"✓ Skill '{name}' created at {skill_dir}")
        print(f"\nNext steps:")
        print(f"  1. Edit {skill_dir / 'skill.yaml'} to configure the skill")
        print(f"  2. Implement logic in {skill_dir / 'main.py'}")
        print(f"  3. Add tests in {skill_dir / 'tests'}")
        print(f"  4. Validate with: hugo skill --validate {safe_name}")

    async def _skill_list(self):
        """List all skills"""
        print("Installed skills:\n")

        skills_dir = Path("skills")
        if not skills_dir.exists():
            print("No skills directory found")
            return

        skills = [d for d in skills_dir.iterdir() if d.is_dir() and (d / "skill.yaml").exists()]

        if not skills:
            print("No skills installed")
            return

        for skill_dir in sorted(skills):
            skill_yaml_path = skill_dir / "skill.yaml"
            # TODO: Parse skill.yaml and show details
            print(f"  • {skill_dir.name}")

        print(f"\nTotal: {len(skills)} skill(s)")

    async def _skill_validate(self, name: str):
        """Validate a skill"""
        print(f"Validating skill: {name}\n")

        # TODO: Run skill tests
        # TODO: Check for security issues
        # TODO: Verify directive compliance

        print("Skill validation not yet implemented")

    async def _skill_remove(self, name: str):
        """Remove a skill"""
        print(f"Removing skill: {name}")

        # TODO: Confirm removal
        # TODO: Remove skill directory
        # TODO: Update registry

        print("Skill removal not yet implemented")

    async def cmd_status(self, verbose: bool = False):
        """Show system status"""
        print("Hugo Status\n" + "=" * 50 + "\n")

        # Service status
        service_status = await self.service_manager.get_status()

        print("Services:")
        for service, status in service_status.items():
            status_icon = "✓" if status == "running" else "✗"
            print(f"  {status_icon} {service}: {status}")

        # Runtime status
        if self.runtime_manager:
            runtime_status = self.runtime_manager.get_status()
            print(f"\nRuntime:")
            print(f"  Mode: {runtime_status['mode']}")
            print(f"  Status: {runtime_status['status']}")
            print(f"  Uptime: {runtime_status['uptime']}")

            if verbose:
                print(f"\nCore Components:")
                for component, status in runtime_status['services'].items():
                    status_icon = "✓" if status == "running" else "✗"
                    print(f"  {status_icon} {component}: {status}")

        print()

    async def cmd_log(self, category: Optional[str] = None, tail: int = 50, follow: bool = False):
        """View logs"""
        print(f"Showing logs (last {tail} entries)\n" + "=" * 50 + "\n")

        # Query logs
        logs = self.logger.query_logs(category=category, limit=tail)

        if not logs:
            print("No log entries found")
            return

        for entry in logs:
            timestamp = entry.get('timestamp', 'N/A')
            category = entry.get('category', 'N/A')
            event_type = entry.get('event_type', 'N/A')
            level = entry.get('level', 'INFO')

            print(f"[{timestamp}] [{level}] {category}.{event_type}")

            if 'data' in entry and entry['data']:
                for key, value in entry['data'].items():
                    print(f"  {key}: {value}")
            print()

        # TODO: Implement follow mode
        if follow:
            print("\n(Follow mode not yet implemented)")

    async def cmd_config(self, show: bool = False, edit: bool = False):
        """Manage configuration"""

        if show:
            print("Current Configuration\n" + "=" * 50 + "\n")
            # TODO: Load and display config
            print("Config display not yet implemented")

        elif edit:
            print("Opening config editor...")
            # TODO: Open config file in editor
            print("Config editing not yet implemented")

        else:
            print("Use --show to view config or --edit to modify it")

    async def cmd_shell(self):
        """Enter interactive REPL"""
        from runtime.repl import HugoREPL

        # Initialize runtime if needed
        if not self.runtime_manager:
            config = {}  # TODO: Load config
            self.runtime_manager = RuntimeManager(config, self.logger)
            await self.runtime_manager.boot()

        # Start REPL
        repl = HugoREPL(self.runtime_manager, self.logger)
        await repl.run()


def main():
    """CLI entry point"""
    cli = HugoCLI()
    cli.run()


if __name__ == '__main__':
    main()
