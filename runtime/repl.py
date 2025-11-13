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

        # Display latest meta-reflection if available
        await self._display_latest_meta_reflection()

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
                elif user_input.lower().startswith('/reflect'):
                    await self._handle_reflect_command(user_input)
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

    async def _display_latest_meta_reflection(self):
        """Display latest meta-reflection insight and recent reflections as startup context"""
        # Check if sqlite_manager exists on runtime
        if not hasattr(self.runtime, 'sqlite_manager') or not self.runtime.sqlite_manager:
            return

        try:
            # Load latest meta-reflection
            meta = await self.runtime.sqlite_manager.get_latest_meta_reflection()
            if meta:
                print("🪞 Last Meta-Reflection:")
                # Truncate summary to 120 characters for conciseness
                summary = meta['summary']
                if len(summary) > 120:
                    summary = summary[:117] + "..."
                print(f"   {summary}")

            # Load recent high-confidence session reflections
            recent_reflections = await self.runtime.sqlite_manager.get_recent_reflections(
                reflection_type="session",
                limit=3
            )

            high_confidence_reflections = [
                r for r in recent_reflections
                if r.get('confidence', 0) >= 0.7
            ]

            if high_confidence_reflections:
                print(f"\n🧠 Recent Insights ({len(high_confidence_reflections)} memories):")
                for refl in high_confidence_reflections[:2]:  # Show top 2
                    summary = refl['summary']
                    if len(summary) > 100:
                        summary = summary[:97] + "..."
                    print(f"   • {summary}")

            print()  # Extra newline for spacing

        except Exception as e:
            # Silently fail - don't disrupt startup
            self.logger.log_event("repl", "boot_context_load_failed", {
                "error": str(e)
            })

    def _display_help(self):
        """Display help message"""
        help_text = """
Hugo Shell Commands:
--------------------
  help           Show this help message
  clear          Clear the screen
  history        Show command history
  /reflect recent   Show recent reflections
  /reflect meta     Show latest meta-reflection
  exit           Exit the shell (also: quit, bye, Ctrl+D)

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

    def _should_use_streaming(self, message: str) -> bool:
        """
        Determine if streaming should be used for this message.

        Uses streaming when:
        - Message suggests a detailed/technical response
        - Message contains implementation keywords
        - STREAMING_ENABLED is true in environment

        Args:
            message: User message

        Returns:
            bool: True if streaming should be used
        """
        import os

        # Check if streaming is enabled
        streaming_enabled = os.getenv("STREAMING_ENABLED", "true").lower() == "true"
        if not streaming_enabled:
            return False

        # Keywords that suggest detailed responses
        streaming_keywords = [
            "implement", "explain", "describe", "how", "what", "why",
            "refactor", "design", "architecture", "detail", "write",
            "create", "build", "develop", "plan", "guide", "tutorial"
        ]

        message_lower = message.lower()
        for keyword in streaming_keywords:
            if keyword in message_lower:
                return True

        # Use streaming for longer messages (>50 chars)
        if len(message) > 50:
            return True

        return False

    async def _process_message(self, message: str):
        """
        Process user message through Hugo's cognition engine.

        Supports both streaming and non-streaming modes based on message context.

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

            # Determine if streaming should be used
            use_streaming = self._should_use_streaming(message)

            response_text = ""
            response_package = None

            # Process through cognition engine
            if self.runtime.cognition:
                if use_streaming:
                    # Stream response token by token
                    async for chunk in self.runtime.cognition.process_input_streaming(
                        message,
                        self.session_id
                    ):
                        # Check if it's the final ResponsePackage or a string chunk
                        if isinstance(chunk, str):
                            print(chunk, end="", flush=True)
                            response_text += chunk
                        else:
                            # Final response package
                            response_package = chunk

                    print()  # Newline after streaming
                else:
                    # Non-streaming response
                    response_package = await self.runtime.cognition.process_input(
                        message,
                        self.session_id
                    )
                    response_text = response_package.content
                    print(response_text)
            else:
                response_text = "(Cognition engine not initialized. Please check boot sequence.)"
                print(response_text)

            # Log Hugo's response
            self.logger.log_user_interaction(
                session_id=self.session_id,
                interaction_type="response",
                data={"content": response_text, "streaming": use_streaming}
            )

            # Store assistant response in memory with enriched metadata
            if self.runtime.memory and response_package:
                # Extract persona-driven metadata from response_package
                response_metadata = {
                    "role": "assistant",
                    "persona_name": response_package.metadata.get("persona_name", "Hugo"),
                    "mood": response_package.metadata.get("mood", "conversational"),
                    "user_sentiment": response_package.metadata.get("user_sentiment", "neutral"),
                    "tone_adjustment": response_package.metadata.get("tone_adjustment", "conversational"),
                    "conversation_turns": response_package.metadata.get("conversation_turns", 0),
                    "semantic_memories": response_package.metadata.get("semantic_memories", 0),
                    "confidence": response_package.metadata.get("confidence", 0.0),
                    "streaming": use_streaming
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
            print(f"\nSorry, I encountered an error: {str(e)}")

    async def _handle_exit(self):
        """Handle graceful exit with reflection"""
        print("\n🪞 Generating session reflection...")

        # Generate session reflection if available
        if self.runtime.reflection:
            try:
                reflection = await self.runtime.reflection.generate_session_reflection(
                    self.session_id
                )

                # Display brief summary
                print(f"\nSession Summary: {reflection.summary[:150]}..." if len(reflection.summary) > 150 else f"\nSession Summary: {reflection.summary}")

                if reflection.insights:
                    print(f"\nKey Insights:")
                    for insight in reflection.insights[:3]:  # Show first 3 insights
                        print(f"  • {insight}")

                # Get metadata for confirmation
                keywords_count = len(reflection.metadata.get("keywords", []))
                sentiment = reflection.metadata.get("sentiment_score", 0.0)

                print(f"\n✨ Session reflection stored (length: {len(reflection.summary)} chars, "
                      f"keywords: {keywords_count}, sentiment: {sentiment:.2f})")

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

    async def _handle_reflect_command(self, command: str):
        """Handle /reflect commands"""
        parts = command.lower().split()

        if len(parts) < 2:
            print("\nUsage:")
            print("  /reflect recent   - Show recent reflections")
            print("  /reflect meta     - Show latest meta-reflection")
            return

        subcommand = parts[1]

        if subcommand == 'recent':
            await self._show_recent_reflections()
        elif subcommand == 'meta':
            await self._show_meta_reflection()
        else:
            print(f"\nUnknown reflect command: {subcommand}")
            print("Available: recent, meta")

    async def _show_recent_reflections(self):
        """Display recent reflections"""
        if not self.runtime.sqlite_manager:
            print("\n(SQLite manager not initialized)")
            return

        try:
            reflections = await self.runtime.sqlite_manager.get_recent_reflections(limit=3)

            if not reflections:
                print("\nNo reflections found yet.")
                return

            print("\n" + "=" * 60)
            print("RECENT REFLECTIONS")
            print("=" * 60)

            for i, ref in enumerate(reflections, 1):
                print(f"\n{i}. [{ref['type'].upper()}] - {ref['timestamp'][:10]}")
                print(f"   Summary: {ref['summary']}")

                if ref['insights']:
                    print(f"   Insights:")
                    for insight in ref['insights'][:2]:  # Show first 2
                        print(f"     • {insight}")

                if ref['keywords']:
                    print(f"   Keywords: {', '.join(ref['keywords'][:5])}")

                if ref['sentiment'] is not None:
                    sentiment_label = "positive" if ref['sentiment'] > 0.3 else "negative" if ref['sentiment'] < -0.3 else "neutral"
                    print(f"   Sentiment: {ref['sentiment']:.2f} ({sentiment_label})")

            print("\n" + "=" * 60)

        except Exception as e:
            self.logger.log_error(e, {"phase": "show_recent_reflections"})
            print(f"\nError retrieving reflections: {str(e)}")

    async def _show_meta_reflection(self):
        """Display latest meta-reflection"""
        if not self.runtime.sqlite_manager:
            print("\n(SQLite manager not initialized)")
            return

        try:
            meta = await self.runtime.sqlite_manager.get_latest_meta_reflection()

            if not meta:
                print("\nNo meta-reflections found yet.")
                return

            print("\n" + "=" * 60)
            print("LATEST META-REFLECTION")
            print("=" * 60)
            print(f"\nCreated: {meta['created_at'][:10]}")
            print(f"Time Window: {meta['time_window_days']} days")
            print(f"Reflections Analyzed: {meta['reflections_analyzed']}")
            print(f"\nSummary:")
            print(f"  {meta['summary']}")

            if meta['insights']:
                print(f"\nStrategic Insights:")
                for insight in meta['insights']:
                    print(f"  • {insight}")

            if meta['patterns']:
                print(f"\nLong-term Patterns:")
                for pattern in meta['patterns']:
                    print(f"  • {pattern}")

            if meta['improvements']:
                print(f"\nImprovement Areas:")
                for improvement in meta['improvements']:
                    print(f"  • {improvement}")

            print(f"\nConfidence: {meta['confidence']:.2%}")
            print("=" * 60)

        except Exception as e:
            self.logger.log_error(e, {"phase": "show_meta_reflection"})
            print(f"\nError retrieving meta-reflection: {str(e)}")
