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
                elif user_input.lower() == 'facts':
                    await self._show_all_factual_memories()
                    continue
                elif user_input.lower().startswith('mem ') or user_input.lower() == 'mem':
                    await self._handle_mem_command(user_input)
                    continue
                elif user_input.lower().startswith('task ') or user_input.lower() == 'task':
                    await self._handle_task_command(user_input)
                    continue
                elif user_input.lower().startswith('skill ') or user_input.lower() == 'skill':
                    await self._handle_skill_command(user_input)
                    continue
                elif user_input.lower().startswith('/skill'):
                    await self._handle_skill_command(user_input)
                    continue
                elif user_input.lower().startswith('/reflect'):
                    await self._handle_reflect_command(user_input)
                    continue
                elif user_input.lower().startswith('/memory'):
                    await self._handle_memory_command(user_input)
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
  help                  Show this help message
  clear                 Clear the screen
  history               Show command history
  facts                 Show all factual memories (from persistent storage)
  exit                  Exit the shell (also: quit, bye, Ctrl+D)

Memory commands:
  mem facts             List recent factual memories
  mem show <id>         Show full details for a memory
  mem delete <id>       Delete a memory and update index
  mem search <query>    Semantic search over stored memories

Task commands:
  task new              Create a new task (interactive)
  task list             List all tasks
  task list mine        List tasks owned by michael
  task show <id>        Show a specific task
  task done <id>        Mark a task as done
  task assign <id> <owner>  Assign a task to an owner

Skill commands:
  /skill list           List all loaded skills
  /skill run <name> <args>  Execute a skill
  /skill reload         Reload all skills from disk

Reflection commands:
  /reflect recent       Show recent reflections
  /reflect meta         Show latest meta-reflection

Debug commands:
  /memory show facts    Show all factual memories
  /memory search <query> Debug semantic search with ranking scores

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
                    # Stream response token by token using generate_reply
                    async for chunk in await self.runtime.cognition.generate_reply(
                        message,
                        self.session_id,
                        streaming=True
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
                    # Non-streaming response using generate_reply (includes skill bypass)
                    response_package = await self.runtime.cognition.generate_reply(
                        message,
                        self.session_id,
                        streaming=False
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

    async def _handle_memory_command(self, command: str):
        """Handle /memory debug commands"""
        parts = command.split(maxsplit=2)

        if len(parts) < 2:
            print("\nUsage:")
            print("  /memory show facts         - Display all factual memories")
            print("  /memory search <query>     - Debug semantic search with scores")
            return

        subcommand = parts[1].lower()

        if subcommand == 'show' and len(parts) >= 3 and parts[2].lower() == 'facts':
            await self._show_factual_memories()
        elif subcommand == 'search' and len(parts) >= 3:
            query = parts[2]
            await self._debug_semantic_search(query)
        else:
            print(f"\nUnknown memory command")
            print("Available: show facts, search <query>")

    async def _show_factual_memories(self):
        """Display all factual memories with details"""
        if not self.runtime.memory:
            print("\n(Memory manager not initialized)")
            return

        try:
            facts = await self.runtime.memory.get_factual_memories(limit=20)

            if not facts:
                print("\nNo factual memories found yet.")
                return

            print("\n" + "=" * 70)
            print("FACTUAL MEMORIES")
            print("=" * 70)

            for i, fact in enumerate(facts, 1):
                entity_label = f"[{fact.entity_type.upper()}]" if fact.entity_type else "[UNKNOWN]"
                print(f"\n{i}. {entity_label} (Importance: {fact.importance_score:.2f})")
                print(f"   Content: {fact.content}")
                print(f"   Session: {fact.session_id}")
                print(f"   Timestamp: {fact.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(fact.timestamp, 'strftime') else str(fact.timestamp)}")
                print(f"   Has embedding: {'Yes' if fact.embedding else 'No'}")

            print("\n" + "=" * 70)
            print(f"Total: {len(facts)} factual memories")
            print("=" * 70)

        except Exception as e:
            self.logger.log_error(e, {"phase": "show_factual_memories"})
            print(f"\nError retrieving factual memories: {str(e)}")

    async def _show_all_factual_memories(self):
        """Display all factual memories from persistent storage"""
        if not self.runtime.memory:
            print("\n(Memory manager not initialized)")
            return

        try:
            facts = await self.runtime.memory.get_all_factual_memories(limit=20)

            if not facts:
                print("\nNo factual memories found yet.")
                return

            print("\n" + "=" * 70)
            print("FACTUAL MEMORIES (most recent first)")
            print("=" * 70)

            for i, fact in enumerate(facts, 1):
                entity_label = f"[{fact.entity_type.upper()}]" if fact.entity_type else "[UNKNOWN]"
                print(f"\n{i}. {entity_label} {fact.content}")

            print("\n" + "=" * 70)
            print(f"Total: {len(facts)} factual memories")
            print("=" * 70)

        except Exception as e:
            self.logger.log_error(e, {"phase": "show_all_factual_memories"})
            print(f"\nError retrieving factual memories: {str(e)}")

    async def _debug_semantic_search(self, query: str):
        """Debug semantic search with detailed scoring"""
        if not self.runtime.memory:
            print("\n(Memory manager not initialized)")
            return

        try:
            print(f"\n🔍 Semantic Search Debug")
            print(f"Query: '{query}'")
            print("=" * 70)

            # Perform semantic search
            results = await self.runtime.memory.search_semantic(query, limit=10, threshold=0.5)

            if not results:
                print("\nNo results found (threshold: 0.5)")
                return

            print(f"\nFound {len(results)} results:\n")

            for i, mem in enumerate(results, 1):
                # Display result details
                is_fact = mem.is_fact
                is_reflection = mem.memory_type == "reflection"

                type_label = []
                if is_fact:
                    type_label.append(f"FACT:{mem.entity_type.upper()}" if mem.entity_type else "FACT")
                if is_reflection:
                    type_label.append("REFLECTION")
                if not type_label:
                    type_label.append(mem.memory_type.upper())

                print(f"{i}. [{', '.join(type_label)}]")
                print(f"   Importance: {mem.importance_score:.2f}")
                print(f"   Content: {mem.content[:100]}..." if len(mem.content) > 100 else f"   Content: {mem.content}")
                print(f"   Session: {mem.session_id}")
                print(f"   Timestamp: {mem.timestamp.strftime('%Y-%m-%d') if hasattr(mem.timestamp, 'strftime') else str(mem.timestamp)}")

                # Calculate recency
                from datetime import datetime
                if hasattr(mem.timestamp, 'strftime'):
                    age_days = (datetime.now() - mem.timestamp).days
                    print(f"   Age: {age_days} days")

                print()

            print("=" * 70)
            print(f"Ranking factors applied:")
            print(f"  - Base: FAISS cosine similarity")
            print(f"  - Factual boost: +0.15")
            print(f"  - Recency boost: +0.10 (if <30 days)")
            print(f"  - Reflection boost: +0.05")
            print("=" * 70)

        except Exception as e:
            self.logger.log_error(e, {"phase": "debug_semantic_search"})
            print(f"\nError during semantic search: {str(e)}")

    async def _handle_mem_command(self, command: str):
        """Handle mem console commands"""
        parts = command.split(maxsplit=2)

        if len(parts) < 2:
            print("\nUsage:")
            print("  mem facts           - List recent factual memories")
            print("  mem show <id>       - Show full details for a memory")
            print("  mem delete <id>     - Delete a memory and update index")
            print("  mem search <query>  - Semantic search over stored memories")
            return

        subcommand = parts[1].lower()

        if subcommand == 'facts':
            await self._mem_facts()
        elif subcommand == 'show' and len(parts) >= 3:
            await self._mem_show(parts[2])
        elif subcommand == 'delete' and len(parts) >= 3:
            await self._mem_delete(parts[2])
        elif subcommand == 'search' and len(parts) >= 3:
            query = parts[2].strip('"\'')  # Remove quotes if present
            await self._mem_search(query)
        else:
            print(f"\nUnknown mem command or missing argument")
            print("Available: facts, show <id>, delete <id>, search <query>")

    async def _mem_facts(self):
        """List factual memories (mem facts)"""
        if not self.runtime.memory:
            print("\n(Memory manager not initialized)")
            return

        try:
            facts = await self.runtime.memory.list_factual_memories(limit=20)

            if not facts:
                print("\nNo factual memories found yet.")
                self.logger.log_event("mem_console", "facts_listed", {"count": 0})
                return

            print("\n" + "=" * 70)
            print(f"{'ID':<6} {'Type':<8} {'Entity':<12} {'Created':<20} Preview")
            print("=" * 70)

            for fact in facts:
                fact_id = fact.get('id', 'N/A')
                memory_type = 'fact'
                entity_type = fact.get('entity_type', 'unknown')[:11]
                timestamp = fact.get('timestamp', '')[:19]
                content = fact.get('content', '')
                preview = content[:35] + '...' if len(content) > 35 else content

                print(f"{fact_id:<6} {memory_type:<8} {entity_type:<12} {timestamp:<20} {preview}")

            print("=" * 70)
            print(f"Total: {len(facts)} factual memories")
            print()

            self.logger.log_event("mem_console", "facts_listed", {"count": len(facts)})

        except Exception as e:
            self.logger.log_error(e, {"phase": "mem_facts"})
            print(f"\nError listing facts: {str(e)}")

    async def _mem_show(self, memory_id_str: str):
        """Show full memory details (mem show <id>)"""
        if not self.runtime.memory:
            print("\n(Memory manager not initialized)")
            return

        try:
            # Parse ID
            try:
                memory_id = int(memory_id_str)
            except ValueError:
                print(f"\nInvalid memory ID: '{memory_id_str}' (must be an integer)")
                return

            # Fetch memory
            memory = await self.runtime.memory.get_memory(memory_id)

            if not memory:
                print(f"\nNo memory found with id {memory_id}")
                self.logger.log_event("mem_console", "show_memory", {
                    "id": memory_id,
                    "found": False
                })
                return

            # Display memory details
            print("\n" + "=" * 70)
            print(f"Memory ID: {memory.get('id', 'N/A')}")
            print(f"Type: {'fact' if memory.get('is_fact') else memory.get('memory_type', 'N/A')}")

            if memory.get('entity_type'):
                print(f"Entity: {memory.get('entity_type')}")

            print(f"Created: {memory.get('timestamp', 'N/A')}")
            print(f"Session: {memory.get('session_id', 'N/A')}")
            print(f"Importance: {memory.get('importance_score', 'N/A')}")

            metadata = memory.get('metadata', {})
            if metadata:
                # Show relevant metadata fields
                if 'keywords' in metadata:
                    keywords = metadata['keywords']
                    if isinstance(keywords, list):
                        print(f"Keywords: {', '.join(keywords[:10])}")

                if 'sentiment_score' in metadata:
                    sentiment = metadata['sentiment_score']
                    sentiment_label = "positive" if sentiment > 0.3 else "negative" if sentiment < -0.3 else "neutral"
                    print(f"Sentiment: {sentiment:.2f} ({sentiment_label})")

            print(f"\nContent:")
            print(f"{memory.get('content', '(no content)')}")
            print("=" * 70)

            self.logger.log_event("mem_console", "show_memory", {
                "id": memory_id,
                "found": True
            })

        except Exception as e:
            self.logger.log_error(e, {"phase": "mem_show", "memory_id": memory_id_str})
            print(f"\nError showing memory: {str(e)}")

    async def _mem_delete(self, memory_id_str: str):
        """Delete a memory (mem delete <id>)"""
        if not self.runtime.memory:
            print("\n(Memory manager not initialized)")
            return

        try:
            # Parse ID
            try:
                memory_id = int(memory_id_str)
            except ValueError:
                print(f"\nInvalid memory ID: '{memory_id_str}' (must be an integer)")
                return

            # Delete memory
            deleted = await self.runtime.memory.delete_memory(memory_id)

            if deleted:
                print(f"\n✓ Deleted memory {memory_id} and rebuilt index")
                self.logger.log_event("mem_console", "delete_memory", {
                    "id": memory_id,
                    "deleted": True
                })
            else:
                print(f"\nNo memory found with id {memory_id}")
                self.logger.log_event("mem_console", "delete_memory", {
                    "id": memory_id,
                    "deleted": False
                })

        except Exception as e:
            self.logger.log_error(e, {"phase": "mem_delete", "memory_id": memory_id_str})
            print(f"\nError deleting memory: {str(e)}")

    async def _mem_search(self, query: str):
        """Search memories semantically (mem search <query>)"""
        if not self.runtime.memory:
            print("\n(Memory manager not initialized)")
            return

        try:
            print(f"\nSearch results for: {query}\n")

            results = await self.runtime.memory.search_memories(query, k=5)

            if not results:
                print("No results found")
                self.logger.log_event("mem_console", "search_memories", {
                    "query": query,
                    "results": 0
                })
                return

            for i, mem in enumerate(results, 1):
                mem_id = mem.get('id', 'N/A')
                score = mem.get('score', 0.0)
                mem_type = 'fact' if mem.get('is_fact') else mem.get('memory_type', 'unknown')
                entity_type = mem.get('entity_type', '')

                type_label = f"{mem_type}"
                if entity_type:
                    type_label = f"{mem_type}, entity={entity_type}"

                content = mem.get('content', '')
                preview = content[:60] + '...' if len(content) > 60 else content

                print(f"{i}) [id={mem_id}, score={score:.2f}, type={type_label}]")
                print(f"   {preview}\n")

            self.logger.log_event("mem_console", "search_memories", {
                "query": query,
                "results": len(results)
            })

        except Exception as e:
            self.logger.log_error(e, {"phase": "mem_search", "query": query})
            print(f"\nError searching memories: {str(e)}")

    async def _handle_task_command(self, command: str):
        """Handle task console commands"""
        parts = command.split(maxsplit=2)

        if len(parts) < 2 or parts[1].lower() == 'help':
            print("\nTask commands:")
            print("  task new                 - Create a new task (interactive)")
            print("  task list                - List all tasks")
            print("  task list mine           - List tasks owned by michael")
            print("  task show <id>           - Show a specific task")
            print("  task done <id>           - Mark a task as done")
            print("  task assign <id> <owner> - Assign a task to an owner")
            return

        subcommand = parts[1].lower()

        if subcommand == 'new':
            await self._task_new()
        elif subcommand == 'list':
            if len(parts) >= 3 and parts[2].lower() == 'mine':
                await self._task_list(owner="michael")
            else:
                await self._task_list()
        elif subcommand == 'show' and len(parts) >= 3:
            await self._task_show(parts[2])
        elif subcommand == 'done' and len(parts) >= 3:
            await self._task_done(parts[2])
        elif subcommand == 'assign' and len(parts) >= 3:
            task_id_and_owner = parts[2].split(maxsplit=1)
            if len(task_id_and_owner) >= 2:
                await self._task_assign(task_id_and_owner[0], task_id_and_owner[1])
            else:
                print("\nUsage: task assign <id> <owner>")
        else:
            print(f"\nUnknown task command or missing argument")
            print("Type 'task help' for available commands")

    async def _task_new(self):
        """Create a new task interactively"""
        if not self.runtime.tasks:
            print("\n(Task manager not initialized)")
            return

        try:
            print("\n--- Create New Task ---")

            # Prompt for title
            loop = asyncio.get_event_loop()
            title = await loop.run_in_executor(None, input, "Title: ")
            if not title.strip():
                print("Title cannot be empty. Task creation cancelled.")
                return

            # Prompt for description
            description = await loop.run_in_executor(None, input, "Description: ")

            # Prompt for owner
            owner = await loop.run_in_executor(None, input, "Owner [michael]: ")
            owner = owner.strip() if owner.strip() else "michael"

            # Prompt for priority
            priority = await loop.run_in_executor(None, input, "Priority [medium]: ")
            priority = priority.strip() if priority.strip() else "medium"

            # Create task
            task = await self.runtime.tasks.create_task(
                title=title.strip(),
                description=description.strip(),
                owner=owner,
                priority=priority
            )

            print(f"\n✓ Created task #{task.id}: {task.title} (owner: {task.owner}, status: {task.status})")

        except Exception as e:
            self.logger.log_error(e, {"phase": "task_new"})
            print(f"\nError creating task: {str(e)}")

    async def _task_list(self, owner: Optional[str] = None):
        """List tasks"""
        if not self.runtime.tasks:
            print("\n(Task manager not initialized)")
            return

        try:
            tasks = await self.runtime.tasks.list_tasks(owner=owner)

            if not tasks:
                if owner:
                    print(f"\nNo tasks found for owner: {owner}")
                else:
                    print("\nNo tasks found yet.")
                return

            print("\n" + "=" * 80)
            print(f"{'ID':<6} {'Status':<13} {'Owner':<12} {'Priority':<10} Title")
            print("=" * 80)

            for task in tasks:
                task_id = task.id
                status = task.status
                task_owner = task.owner or 'N/A'
                task_priority = task.priority or 'N/A'
                title = task.title[:40] + '...' if len(task.title) > 40 else task.title

                print(f"{task_id:<6} {status:<13} {task_owner:<12} {task_priority:<10} {title}")

            print("=" * 80)
            print(f"Total: {len(tasks)} task(s)")
            print()

        except Exception as e:
            self.logger.log_error(e, {"phase": "task_list"})
            print(f"\nError listing tasks: {str(e)}")

    async def _task_show(self, task_id_str: str):
        """Show task details"""
        if not self.runtime.tasks:
            print("\n(Task manager not initialized)")
            return

        try:
            # Parse ID
            try:
                task_id = int(task_id_str)
            except ValueError:
                print(f"\nInvalid task ID: '{task_id_str}' (must be an integer)")
                return

            # Fetch task
            task = await self.runtime.tasks.get_task(task_id)

            if not task:
                print(f"\nNo task found with id {task_id}")
                return

            # Display task details
            print("\n" + "=" * 70)
            print(f"Task #{task.id}")
            print("=" * 70)
            print(f"  Title: {task.title}")
            print(f"  Description: {task.description}")
            print(f"  Status: {task.status}")
            print(f"  Owner: {task.owner}")
            print(f"  Priority: {task.priority}")

            if task.tags:
                print(f"  Tags: {', '.join(task.tags)}")

            print(f"  Created: {task.created_at.strftime('%Y-%m-%d %H:%M:%S') if hasattr(task.created_at, 'strftime') else str(task.created_at)}")
            print(f"  Updated: {task.updated_at.strftime('%Y-%m-%d %H:%M:%S') if hasattr(task.updated_at, 'strftime') else str(task.updated_at)}")
            print("=" * 70)

        except Exception as e:
            self.logger.log_error(e, {"phase": "task_show", "task_id": task_id_str})
            print(f"\nError showing task: {str(e)}")

    async def _task_done(self, task_id_str: str):
        """Mark task as done"""
        if not self.runtime.tasks:
            print("\n(Task manager not initialized)")
            return

        try:
            # Parse ID
            try:
                task_id = int(task_id_str)
            except ValueError:
                print(f"\nInvalid task ID: '{task_id_str}' (must be an integer)")
                return

            # Update status
            success = await self.runtime.tasks.update_task_status(task_id, "done")

            if success:
                print(f"\n✓ Task {task_id} marked as done")
            else:
                print(f"\nNo task found with id {task_id}")

        except Exception as e:
            self.logger.log_error(e, {"phase": "task_done", "task_id": task_id_str})
            print(f"\nError marking task as done: {str(e)}")

    async def _task_assign(self, task_id_str: str, owner: str):
        """Assign task to owner"""
        if not self.runtime.tasks:
            print("\n(Task manager not initialized)")
            return

        try:
            # Parse ID
            try:
                task_id = int(task_id_str)
            except ValueError:
                print(f"\nInvalid task ID: '{task_id_str}' (must be an integer)")
                return

            # Assign task
            success = await self.runtime.tasks.assign_task(task_id, owner)

            if success:
                print(f"\n✓ Task {task_id} assigned to {owner}")
            else:
                print(f"\nNo task found with id {task_id}")

        except Exception as e:
            self.logger.log_error(e, {"phase": "task_assign", "task_id": task_id_str})
            print(f"\nError assigning task: {str(e)}")

    # ==============================================
    # SKILL CONSOLE COMMANDS
    # ==============================================

    async def _handle_skill_command(self, command: str):
        """Handle skill console commands (skill and /skill)"""
        # Strip leading slash if present
        command = command.lstrip('/')

        parts = command.split(maxsplit=2)

        if len(parts) < 2 or parts[1].lower() == 'help':
            print("\nSkill commands:")
            print("  skill list               - List all loaded skills")
            print("  /skill list              - List all loaded skills")
            print("  skill run <name> <args>  - Execute a skill")
            print("  /skill run <name> <args> - Execute a skill")
            print("  skill reload             - Reload all skills from disk")
            print("  /skill reload            - Reload all skills from disk")
            return

        subcommand = parts[1].lower()

        if subcommand == 'list':
            await self._skill_list()
        elif subcommand == 'run' and len(parts) >= 3:
            await self._skill_run(parts[2])
        elif subcommand == 'reload':
            await self._skill_reload()
        else:
            print(f"\nUnknown skill command or missing argument")
            print("Type 'skill help' for available commands")

    async def _skill_list(self):
        """List all loaded skills"""
        if not self.runtime.skills:
            print("\n(Skill manager not initialized)")
            return

        try:
            skills = self.runtime.skills.list_skills()

            if not skills:
                print("\nNo skills loaded yet.")
                return

            print("\n" + "=" * 90)
            print(f"{'Name':<15} {'Version':<10} {'Trigger':<10} {'Description'}")
            print("=" * 90)

            for skill in skills:
                name = skill['name'][:14]
                version = skill.get('version', 'N/A')[:9]
                trigger = skill.get('trigger', 'manual')[:9]
                description = skill.get('description', 'No description')[:50]
                description = description + '...' if len(skill.get('description', '')) > 50 else description

                print(f"{name:<15} {version:<10} {trigger:<10} {description}")

            print("=" * 90)
            print(f"Total: {len(skills)} skill(s)")
            print()

        except Exception as e:
            self.logger.log_error(e, {"phase": "skill_list"})
            print(f"\nError listing skills: {str(e)}")

    async def _skill_run(self, args_str: str):
        """Run a skill with arguments"""
        if not self.runtime.skills:
            print("\n(Skill manager not initialized)")
            return

        try:
            # Parse skill name and arguments
            # Format: skill run <skill_name> action=<action> arg1=value1 arg2=value2
            parts = args_str.split()
            if not parts:
                print("\nUsage: skill run <name> [action=<action>] [arg1=value1] [arg2=value2]")
                return

            skill_name = parts[0]
            kwargs = {}

            # Parse key=value arguments
            for arg in parts[1:]:
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    kwargs[key] = value
                else:
                    # If no '=' found, treat it as action (for backward compatibility)
                    if 'action' not in kwargs:
                        kwargs['action'] = arg

            # Execute skill
            print(f"\n→ Running skill '{skill_name}'...")
            result = await self.runtime.skills.run_skill(skill_name, **kwargs)

            # Display result
            if result.success:
                print(f"✓ {result.message}")
                if result.output:
                    if isinstance(result.output, str):
                        print(f"\n{result.output}")
                    elif isinstance(result.output, list):
                        if result.output:
                            print()
                            for item in result.output:
                                print(f"  - {item}")
                    elif isinstance(result.output, dict):
                        print()
                        for key, value in result.output.items():
                            print(f"  {key}: {value}")
            else:
                print(f"✗ {result.message}")

            print()

        except Exception as e:
            self.logger.log_error(e, {"phase": "skill_run", "args": args_str})
            print(f"\nError running skill: {str(e)}")

    async def _skill_reload(self):
        """Reload all skills from disk"""
        if not self.runtime.skills:
            print("\n(Skill manager not initialized)")
            return

        try:
            print("\n→ Reloading skills...")
            self.runtime.skills.reload_skills()

            skills_count = self.runtime.skills.registry.count()
            print(f"✓ Reloaded {skills_count} skill(s)")

        except Exception as e:
            self.logger.log_error(e, {"phase": "skill_reload"})
            print(f"\nError reloading skills: {str(e)}")
