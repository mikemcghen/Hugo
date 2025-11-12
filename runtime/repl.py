"""
Hugo REPL
---------
Interactive Read-Eval-Print Loop for conversational interaction with Hugo.

Features:
- Conversational interface
- Command history
- Tab completion
- Multi-line input
- Rich formatting
"""

import asyncio
import sys
from typing import Optional
from datetime import datetime


class HugoREPL:
    """
    Interactive REPL for Hugo conversations.

    Provides a conversational interface with command history,
    tab completion, and rich formatting.
    """

    def __init__(self, runtime_manager, logger):
        """
        Initialize REPL.

        Args:
            runtime_manager: RuntimeManager instance
            logger: HugoLogger instance
        """
        self.runtime = runtime_manager
        self.logger = logger
        self.session_id = f"shell_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.running = False
        self.history = []

    async def run(self):
        """Start the REPL loop"""
        self.running = True

        self._display_welcome()

        try:
            while self.running:
                # Get user input
                try:
                    user_input = await self._get_input()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print("\n(Use 'exit' or Ctrl+D to quit)")
                    continue

                if not user_input.strip():
                    continue

                # Add to history
                self.history.append(user_input)

                # Check for special commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    await self._handle_exit()
                    break
                elif user_input.lower() == 'help':
                    self._display_help()
                    continue
                elif user_input.lower() == 'clear':
                    self._clear_screen()
                    continue
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue

                # Process through cognition engine
                await self._process_message(user_input)

        except Exception as e:
            self.logger.log_error(e)
            print(f"\nREPL error: {str(e)}")

        finally:
            self.running = False
            print("\nGoodbye!\n")

    def _display_welcome(self):
        """Display REPL welcome message"""
        welcome = """
╔════════════════════════════════════════╗
║        Hugo Interactive Shell          ║
╚════════════════════════════════════════╝

Type 'help' for available commands, or just start chatting.
Type 'exit' to quit.

"""
        print(welcome)

    def _display_help(self):
        """Display help message"""
        help_text = """
Hugo Shell Commands:
--------------------
  help      Show this help message
  clear     Clear the screen
  history   Show command history
  exit      Exit the shell (also: quit, bye, Ctrl+D)

Just type naturally to chat with Hugo!
"""
        print(help_text)

    async def _get_input(self) -> str:
        """
        Get user input with prompt.

        Returns:
            User input string

        TODO: Implement proper async input with readline support
        """
        # For now, use simple input
        # TODO: Use aioconsole or prompt_toolkit for better experience
        loop = asyncio.get_event_loop()
        prompt = "\nYou: "
        return await loop.run_in_executor(None, input, prompt)

    async def _process_message(self, message: str):
        """
        Process user message through Hugo's cognition engine.

        Args:
            message: User message to process
        """
        print("\nHugo: ", end="", flush=True)

        try:
            # Log user message
            self.logger.log_user_interaction(
                session_id=self.session_id,
                interaction_type="message",
                data={"content": message}
            )

            # Store user message in memory
            if self.runtime.memory:
                from core.memory import MemoryEntry
                from datetime import datetime

                user_memory = MemoryEntry(
                    id=None,
                    session_id=self.session_id,
                    timestamp=datetime.now(),
                    memory_type="user_message",
                    content=message,
                    embedding=None,  # Will be auto-generated
                    metadata={"role": "user"},
                    importance_score=0.7
                )
                await self.runtime.memory.store(user_memory)

            # Process through cognition engine
            if self.runtime.cognition:
                response_package = await self.runtime.cognition.process_input(
                    message,
                    self.session_id
                )
                response_text = response_package.content
            else:
                response_text = "(Cognition engine not initialized. Please check boot sequence.)"

            # Display response
            print(response_text)

            # Log Hugo's response
            self.logger.log_user_interaction(
                session_id=self.session_id,
                interaction_type="response",
                data={"content": response_text}
            )

            # Store assistant response in memory with enriched metadata
            if self.runtime.memory:
                # Extract persona-driven metadata from response_package
                response_metadata = {
                    "role": "assistant",
                    "persona_name": response_package.metadata.get("persona_name", "Hugo"),
                    "mood": response_package.metadata.get("mood", "conversational"),
                    "user_sentiment": response_package.metadata.get("user_sentiment", "neutral"),
                    "tone_adjustment": response_package.metadata.get("tone_adjustment", "conversational"),
                    "conversation_turns": response_package.metadata.get("conversation_turns", 0),
                    "semantic_memories": response_package.metadata.get("semantic_memories", 0),
                    "confidence": response_package.metadata.get("confidence", 0.0)
                }

                assistant_memory = MemoryEntry(
                    id=None,
                    session_id=self.session_id,
                    timestamp=datetime.now(),
                    memory_type="assistant_message",
                    content=response_text,
                    embedding=None,  # Will be auto-generated
                    metadata=response_metadata,
                    importance_score=0.7
                )
                await self.runtime.memory.store(assistant_memory)

        except Exception as e:
            self.logger.log_error(e)
            print(f"Sorry, I encountered an error: {str(e)}")

    async def _handle_exit(self):
        """Handle graceful exit with reflection"""
        print("\nGenerating session reflection...")

        # Generate session reflection if available
        if self.runtime.reflection:
            try:
                reflection = await self.runtime.reflection.generate_session_reflection(
                    self.session_id
                )

                # Display reflection summary
                print("\n" + "=" * 60)
                print("SESSION REFLECTION")
                print("=" * 60)
                print(f"\n{reflection.summary}\n")

                if reflection.insights:
                    print("Key Insights:")
                    for insight in reflection.insights:
                        print(f"  • {insight}")
                    print()

                if reflection.patterns_observed:
                    print("Patterns:")
                    for pattern in reflection.patterns_observed:
                        print(f"  • {pattern}")
                    print()

                print("=" * 60)

            except Exception as e:
                self.logger.log_error(e, {"phase": "session_reflection"})
                print(f"(Reflection generation encountered an issue: {str(e)})")
        else:
            print("(Reflection engine not initialized)")

        self.logger.log_event("repl", "session_ended", {
            "session_id": self.session_id,
            "message_count": len(self.history)
        })

    def _show_history(self):
        """Display command history"""
        if not self.history:
            print("\nNo history yet.")
            return

        print("\nCommand History:")
        for i, cmd in enumerate(self.history, 1):
            print(f"  {i}. {cmd}")
        print()

    def _clear_screen(self):
        """Clear the terminal screen"""
        import os
        os.system('cls' if sys.platform == 'win32' else 'clear')
        self._display_welcome()
