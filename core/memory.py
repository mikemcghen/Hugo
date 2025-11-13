"""
Memory Manager
--------------
Implements Hugo's hybrid memory architecture:
- Short-term: SQLite-based session memory (fast, ephemeral)
- Long-term: PostgreSQL with pgvector (persistent, searchable)
- Vector index: FAISS local cache + pgvector for semantic search

Memory Types:
- Episodic: Conversation history, events
- Semantic: Knowledge, learned patterns
- Procedural: Skills, capabilities
"""

import asyncio
import os
import numpy as np
import faiss
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class MemoryEntry:
    """Single memory record"""
    id: Optional[int]
    session_id: str
    timestamp: datetime
    memory_type: str  # episodic, semantic, procedural, reflection
    content: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    importance_score: float
    is_fact: bool = False  # True if contains factual user information
    entity_type: Optional[str] = None  # animal, person, location, preference, etc.


class MemoryManager:
    """
    Manages Hugo's layered memory system with semantic search capabilities.

    Architecture:
    - Hot cache: Recent memories in RAM
    - Short-term: SQLite for current session
    - Long-term: PostgreSQL for persistent storage
    - Vector search: FAISS + pgvector for semantic retrieval
    """

    def __init__(self, sqlite_conn, postgres_conn, logger):
        """
        Initialize memory manager with dual persistence layers.

        Args:
            sqlite_conn: SQLite connection for short-term memory
            postgres_conn: PostgreSQL connection for long-term memory
            logger: HugoLogger instance
        """
        self.sqlite = sqlite_conn
        self.postgres = postgres_conn
        self.logger = logger

        # In-memory cache for hot access
        self.cache = []
        self.cache_size = int(os.getenv("MEMORY_CACHE_SIZE", "100"))

        # Embedding configuration
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))
        self.enable_faiss = os.getenv("ENABLE_FAISS", "true").lower() == "true"

        # Initialize embedding model
        self.embedding_model = None
        if self.enable_faiss:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.logger.log_event("memory", "embedding_model_loaded",
                                    {"model": self.embedding_model_name})
            except Exception as e:
                self.logger.log_error(e)
                self.enable_faiss = False

        # Initialize FAISS index for local vector search
        self.faiss_index = None
        self.faiss_index_path = Path(os.getenv("FAISS_INDEX_PATH", "data/memory/faiss_index.bin"))
        self.memory_id_map = []  # Maps FAISS index positions to memory IDs

        if self.enable_faiss:
            self._initialize_faiss_index()

    def _initialize_faiss_index(self):
        """Initialize or load FAISS index"""
        try:
            # Try to load existing index
            if self.faiss_index_path.exists():
                self.faiss_index = faiss.read_index(str(self.faiss_index_path))
                self.logger.log_event("memory", "faiss_index_loaded",
                                    {"path": str(self.faiss_index_path)})
            else:
                # Create new index (using Flat for exact search)
                index_type = os.getenv("FAISS_INDEX_TYPE", "Flat")
                if index_type == "Flat":
                    self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension)
                else:
                    # Default to Flat if unknown type
                    self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension)

                self.logger.log_event("memory", "faiss_index_created",
                                    {"dimension": self.embedding_dimension, "type": index_type})

                # Ensure directory exists
                self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            self.logger.log_error(e)
            self.faiss_index = None
            self.enable_faiss = False

    def _save_faiss_index(self):
        """Save FAISS index to disk"""
        if self.faiss_index and self.enable_faiss:
            try:
                faiss.write_index(self.faiss_index, str(self.faiss_index_path))
                self.logger.log_event("memory", "faiss_index_saved",
                                    {"path": str(self.faiss_index_path)})
            except Exception as e:
                self.logger.log_error(e)

    def _detect_facts(self, content: str) -> tuple[bool, Optional[str]]:
        """
        Detect if content contains factual user information.

        Questions are NEVER stored as facts, even if they mention entities.
        Only declarative sentences are considered factual.

        Args:
            content: Message content to analyze

        Returns:
            Tuple of (is_fact, entity_type)
        """
        content_lower = content.lower().strip()

        # Guard: Never treat questions as facts
        # Check if ends with '?'
        if content.strip().endswith('?'):
            return False, None

        # Check if starts with question phrases (case-insensitive)
        question_starters = ["do you", "can you", "will you", "would you", "could you",
                            "are you", "is there", "did you", "have you", "what",
                            "when", "where", "why", "how", "who"]
        for starter in question_starters:
            if content_lower.startswith(starter):
                return False, None

        # Fact indicators with entity types (only for declarative sentences)
        fact_patterns = {
            "animal": ["cat", "dog", "pet", "bird", "fish", "hamster", "rabbit", "parrot", "bunny", "bunnies"],
            "person": ["name is", "called", "my wife", "my husband", "my friend", "my child", "my son", "my daughter"],
            "location": ["live in", "from", "city", "country", "state", "address"],
            "preference": ["favorite", "prefer", "like", "love", "enjoy", "interested in", "hobby"],
            "possession": ["own", "have", "got", "bought", "car", "house", "computer"],
            "occupation": ["work as", "job", "career", "profession", "engineer", "teacher", "doctor"],
            "contact": ["email", "phone", "number", "address", "@"]
        }

        for entity_type, patterns in fact_patterns.items():
            for pattern in patterns:
                if pattern in content_lower:
                    return True, entity_type

        return False, None

    async def store(self, entry: MemoryEntry, persist_long_term: bool = False):
        """
        Store a memory entry in appropriate layer(s).

        Automatically detects and tags factual information.
        Facts are ALWAYS persisted to SQLite for cross-session recall.

        Args:
            entry: MemoryEntry to store
            persist_long_term: If True, also write to PostgreSQL
        """
        # Detect facts if this is a user message
        if entry.memory_type == "user_message" and not entry.is_fact:
            is_fact, entity_type = self._detect_facts(entry.content)
            if is_fact:
                entry.is_fact = True
                entry.entity_type = entity_type
                entry.importance_score = max(entry.importance_score, 0.85)  # Boost factual memories

                # Log fact extraction
                self.logger.log_event("memory", "fact_extracted", {
                    "summary": entry.content[:80],
                    "entity_type": entity_type
                })

        # Force facts to be persisted long-term
        if entry.is_fact:
            persist_long_term = True

        # Generate embedding if not present
        if entry.embedding is None and self.enable_faiss:
            entry.embedding = await self._generate_embedding(entry.content)

        # Add to cache
        self.cache.append(entry)
        if len(self.cache) > self.cache_size:
            self.cache.pop(0)

        # Add to FAISS index if enabled
        if self.enable_faiss and self.faiss_index and entry.embedding:
            try:
                embedding_vector = np.array([entry.embedding], dtype=np.float32)
                self.faiss_index.add(embedding_vector)
                self.memory_id_map.append(entry.id)

                # Periodically save index
                if self.faiss_index.ntotal % 100 == 0:
                    self._save_faiss_index()

            except Exception as e:
                self.logger.log_error(e)

        # Store in SQLite (short-term)
        await self._store_sqlite(entry)

        # Persist facts and long-term memories to SQLite memories table
        if persist_long_term and self.sqlite:
            try:
                import pickle
                # Serialize embedding to bytes
                embedding_bytes = pickle.dumps(entry.embedding) if entry.embedding else None

                # Store in SQLite memories table
                memory_id = await self.sqlite.store_memory(
                    session_id=entry.session_id,
                    memory_type=entry.memory_type,
                    content=entry.content,
                    embedding=embedding_bytes,
                    metadata=entry.metadata,
                    importance_score=entry.importance_score,
                    is_fact=entry.is_fact,
                    entity_type=entry.entity_type
                )

                # Update entry ID if not set
                if entry.id is None:
                    entry.id = memory_id

                self.logger.log_event("memory", "sqlite_persisted", {
                    "memory_id": memory_id,
                    "is_fact": entry.is_fact,
                    "entity_type": entry.entity_type
                })

            except Exception as e:
                self.logger.log_error(e, {"phase": "sqlite_persistence"})

        # Optionally persist to PostgreSQL
        if persist_long_term:
            await self._store_postgres(entry)

        self.logger.log_event("memory", "stored", {
            "type": entry.memory_type,
            "long_term": persist_long_term,
            "has_embedding": entry.embedding is not None,
            "is_fact": entry.is_fact,
            "entity_type": entry.entity_type if entry.is_fact else None
        })

    async def retrieve_recent(self, session_id: str, limit: int = 20) -> List[MemoryEntry]:
        """
        Retrieve recent memories from current session.

        Args:
            session_id: Session identifier
            limit: Maximum number of entries to return

        Returns:
            List of recent MemoryEntry objects
        """
        # TODO: Query SQLite for recent session memories
        # Placeholder implementation
        return [e for e in self.cache if e.session_id == session_id][:limit]

    async def search_semantic(self, query: str, limit: int = 10,
                            threshold: float = 0.6) -> List[MemoryEntry]:
        """
        Search memories using hybrid semantic + factual ranking.

        Ranking formula:
        - Base: FAISS cosine similarity
        - Boost: +0.15 if is_fact=True
        - Boost: +0.10 if memory within last 30 days
        - Boost: +0.05 if memory_type='reflection'

        Args:
            query: Search query text
            limit: Maximum results to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of semantically similar MemoryEntry objects, ranked by relevance
        """
        if not self.enable_faiss or not self.faiss_index or not self.embedding_model:
            self.logger.log_event("memory", "semantic_search_unavailable", {})
            return []

        try:
            # Generate embedding for query
            query_embedding = await self._generate_embedding(query)
            query_vector = np.array([query_embedding], dtype=np.float32)

            # Search FAISS index (get more results for ranking)
            k = min(limit * 3, self.faiss_index.ntotal)
            if k == 0:
                return []

            distances, indices = self.faiss_index.search(query_vector, k)

            # Convert L2 distances to similarity scores
            similarities = 1 - (distances[0] ** 2 / 2)

            # Hybrid ranking with boosts
            scored_results = []
            from datetime import timedelta

            for idx, base_similarity in zip(indices[0], similarities):
                if idx == -1:
                    continue

                if base_similarity < threshold:
                    continue

                # Retrieve memory from cache
                if idx >= len(self.cache):
                    continue

                memory = self.cache[idx]
                score = base_similarity

                # Factual memory boost (+0.15)
                if memory.is_fact:
                    score += 0.15

                # Recency boost (+0.10 for last 30 days)
                age_days = (datetime.now() - memory.timestamp).days
                if age_days <= 30:
                    score += 0.10

                # Reflection boost (+0.05)
                if memory.memory_type == "reflection":
                    score += 0.05

                scored_results.append((score, memory))

            # Sort by score descending
            scored_results.sort(key=lambda x: x[0], reverse=True)

            # Extract top memories
            results = [memory for score, memory in scored_results[:limit]]

            self.logger.log_event("memory", "semantic_search_completed", {
                "query_length": len(query),
                "results": len(results),
                "factual_results": sum(1 for m in results if m.is_fact),
                "reflection_results": sum(1 for m in results if m.memory_type == "reflection")
            })

            return results

        except Exception as e:
            self.logger.log_error(e)
            return []

    async def search_temporal(self, session_id: str, start_time: datetime,
                            end_time: datetime) -> List[MemoryEntry]:
        """
        Search memories by time range.

        Args:
            session_id: Session identifier
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of MemoryEntry objects in time range
        """
        # TODO: Query SQLite and PostgreSQL by timestamp
        return []

    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Generate a summary of a session's memories.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary containing session summary statistics

        TODO:
        - Count messages, events, reflections
        - Calculate average importance
        - Extract key topics
        - Identify mood patterns
        """
        return {
            "session_id": session_id,
            "message_count": 0,
            "duration": "0m",
            "key_topics": [],
            "avg_importance": 0.0
        }

    async def consolidate_session(self, session_id: str):
        """
        Move session memories from SQLite to PostgreSQL for long-term storage.

        Args:
            session_id: Session to consolidate

        TODO:
        - Extract all session memories from SQLite
        - Filter by importance threshold
        - Batch insert to PostgreSQL
        - Update vector index
        - Clear SQLite records
        """
        self.logger.log_event("memory", "consolidation_started", {"session_id": session_id})
        # Placeholder for consolidation logic
        pass

    async def _store_sqlite(self, entry: MemoryEntry):
        """Store entry in SQLite (short-term memory)"""
        # TODO: Implement SQLite insertion
        pass

    async def _store_postgres(self, entry: MemoryEntry):
        """Store entry in PostgreSQL (long-term memory)"""
        # TODO: Implement PostgreSQL insertion with vector embedding
        pass

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate vector embedding for text using SentenceTransformers.

        Args:
            text: Text to encode

        Returns:
            List of floats representing the embedding vector
        """
        if not self.embedding_model:
            # Return zero vector if embedding model not available
            return [0.0] * self.embedding_dimension

        try:
            # Generate embedding using sentence-transformers
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)

            # Normalize for cosine similarity (optional but recommended)
            embedding = embedding / np.linalg.norm(embedding)

            return embedding.tolist()

        except Exception as e:
            self.logger.log_error(e)
            return [0.0] * self.embedding_dimension

    async def prune_old_memories(self, days_threshold: int = 90):
        """
        Remove low-importance memories older than threshold.

        Args:
            days_threshold: Age in days after which to consider pruning

        TODO:
        - Identify old, low-importance memories
        - Archive to backup before deletion
        - Update indices
        - Log pruning statistics
        """
        self.logger.log_event("memory", "pruning_started", {"threshold_days": days_threshold})
        pass

    async def get_factual_memories(self, limit: int = 10) -> List[MemoryEntry]:
        """
        Retrieve factual memories (user-specific information) from cache.

        Args:
            limit: Maximum number of facts to return

        Returns:
            List of factual MemoryEntry objects, sorted by importance
        """
        facts = [m for m in self.cache if m.is_fact]
        facts.sort(key=lambda m: m.importance_score, reverse=True)
        return facts[:limit]

    async def get_all_factual_memories(self, limit: int = 20) -> List[MemoryEntry]:
        """
        Retrieve factual memories from persistent SQLite storage.

        This reflects the actual persisted state, not just in-memory cache.
        Returns most recent facts first.

        Args:
            limit: Maximum number of facts to return

        Returns:
            List of factual MemoryEntry objects from SQLite, most recent first
        """
        if not self.sqlite:
            self.logger.log_event("memory", "factual_retrieval_skipped", {
                "reason": "no_sqlite_connection"
            })
            return []

        try:
            import pickle

            # Retrieve all factual memories from SQLite
            factual_memories = await self.sqlite.get_factual_memories(limit=limit)

            if not factual_memories:
                return []

            result = []
            for mem_dict in factual_memories:
                # Deserialize embedding if present
                embedding = None
                if mem_dict.get('embedding'):
                    try:
                        embedding = pickle.loads(mem_dict['embedding'])
                    except Exception as e:
                        self.logger.log_error(e, {"phase": "embedding_deserialization", "memory_id": mem_dict['id']})

                # Parse timestamp
                try:
                    timestamp = datetime.fromisoformat(mem_dict['timestamp'])
                except:
                    timestamp = datetime.now()

                # Create MemoryEntry
                entry = MemoryEntry(
                    id=mem_dict['id'],
                    session_id=mem_dict['session_id'],
                    timestamp=timestamp,
                    memory_type=mem_dict['memory_type'],
                    content=mem_dict['content'],
                    embedding=embedding,
                    metadata=mem_dict.get('metadata', {}),
                    importance_score=mem_dict.get('importance_score', 0.5),
                    is_fact=True,
                    entity_type=mem_dict.get('entity_type')
                )
                result.append(entry)

            return result

        except Exception as e:
            self.logger.log_error(e, {"phase": "get_all_factual_memories"})
            return []

    async def update_fact(self, entity_type: str, old_content: str, new_content: str,
                         session_id: str) -> bool:
        """
        Update a factual memory by removing old information and adding corrected information.

        Args:
            entity_type: Type of entity being updated (animal, person, etc.)
            old_content: Old/incorrect information (for deletion)
            new_content: New/correct information
            session_id: Current session ID

        Returns:
            True if update successful
        """
        try:
            # Remove old conflicting memories from cache
            self.cache = [m for m in self.cache if old_content.lower() not in m.content.lower()]

            # Create new corrected fact
            corrected_fact = MemoryEntry(
                id=None,
                session_id=session_id,
                timestamp=datetime.now(),
                memory_type="user_message",
                content=new_content,
                embedding=None,  # Will be generated
                metadata={
                    "reason": "corrected",
                    "entity_type": entity_type,
                    "corrected_from": old_content[:100]  # Track what was corrected
                },
                importance_score=0.95,  # High importance for corrections
                is_fact=True,
                entity_type=entity_type
            )

            # Store corrected fact
            await self.store(corrected_fact, persist_long_term=True)

            self.logger.log_event("memory", "fact_updated", {
                "entity_type": entity_type,
                "old_content_preview": old_content[:50],
                "new_content_preview": new_content[:50]
            })

            return True

        except Exception as e:
            self.logger.log_error(e, {"phase": "fact_update"})
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        faiss_size = self.faiss_index.ntotal if self.faiss_index else 0
        fact_count = sum(1 for m in self.cache if m.is_fact)

        return {
            "cache_size": len(self.cache),
            "factual_memories": fact_count,
            "sqlite_records": 0,  # TODO: Query actual count
            "postgres_records": 0,  # TODO: Query actual count
            "faiss_index_size": faiss_size,
            "faiss_enabled": self.enable_faiss,
            "embedding_model": self.embedding_model_name if self.enable_faiss else None
        }

    async def load_factual_memories(self):
        """
        Load all factual memories from SQLite and hydrate cache + FAISS index.

        This should be called on boot to restore cross-session facts.
        """
        if not self.sqlite:
            self.logger.log_event("memory", "factual_load_skipped", {
                "reason": "no_sqlite_connection"
            })
            return

        try:
            # Retrieve all factual memories from SQLite
            factual_memories = await self.sqlite.get_factual_memories()

            if not factual_memories:
                self.logger.log_event("memory", "factual_memories_loaded", {
                    "count": 0,
                    "faiss_added": 0
                })
                return

            import pickle
            loaded_count = 0
            faiss_added = 0

            for mem_dict in factual_memories:
                # Deserialize embedding
                embedding = None
                if mem_dict['embedding']:
                    try:
                        embedding = pickle.loads(mem_dict['embedding'])
                    except Exception as e:
                        self.logger.log_error(e, {"phase": "embedding_deserialization", "memory_id": mem_dict['id']})

                # Parse timestamp
                from datetime import datetime
                try:
                    timestamp = datetime.fromisoformat(mem_dict['timestamp'])
                except:
                    timestamp = datetime.now()

                # Create MemoryEntry
                entry = MemoryEntry(
                    id=mem_dict['id'],
                    session_id=mem_dict['session_id'],
                    timestamp=timestamp,
                    memory_type=mem_dict['memory_type'],
                    content=mem_dict['content'],
                    embedding=embedding,
                    metadata=mem_dict['metadata'],
                    importance_score=mem_dict['importance_score'],
                    is_fact=True,
                    entity_type=mem_dict['entity_type']
                )

                # Add to cache
                self.cache.append(entry)
                loaded_count += 1

                # Add to FAISS index if embedding exists
                if self.enable_faiss and self.faiss_index and embedding:
                    try:
                        embedding_vector = np.array([embedding], dtype=np.float32)
                        self.faiss_index.add(embedding_vector)
                        self.memory_id_map.append(entry.id)
                        faiss_added += 1
                    except Exception as e:
                        self.logger.log_error(e, {"phase": "faiss_add_on_load", "memory_id": mem_dict['id']})

            # Save FAISS index after loading
            if faiss_added > 0:
                self._save_faiss_index()

            self.logger.log_event("memory", "factual_memories_loaded", {
                "count": loaded_count,
                "faiss_added": faiss_added
            })

        except Exception as e:
            self.logger.log_error(e, {"phase": "load_factual_memories"})

    async def list_factual_memories(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List factual memories from SQLite storage.

        Args:
            limit: Maximum number of memories to return

        Returns:
            List of memory dictionaries
        """
        if not self.sqlite:
            return []

        return await self.sqlite.get_factual_memories(limit=limit)

    async def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch a single memory by ID from SQLite.

        Args:
            memory_id: Memory ID to fetch

        Returns:
            Memory dictionary or None if not found
        """
        if not self.sqlite:
            return None

        return await self.sqlite.get_memory_by_id(memory_id)

    async def delete_memory(self, memory_id: int) -> bool:
        """
        Delete a memory by ID and rebuild FAISS index.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if deletion succeeded, False otherwise
        """
        if not self.sqlite:
            return False

        try:
            # Delete from SQLite
            deleted = await self.sqlite.delete_memory(memory_id)

            if deleted:
                # Rebuild FAISS index to maintain consistency
                await self._rebuild_faiss_index_from_sqlite()

                self.logger.log_event("memory", "memory_deleted", {
                    "memory_id": memory_id
                })

            return deleted

        except Exception as e:
            self.logger.log_error(e, {"phase": "delete_memory", "memory_id": memory_id})
            return False

    async def _rebuild_faiss_index_from_sqlite(self):
        """
        Rebuild FAISS index from all memories in SQLite.

        Called after deletion to ensure index consistency.
        """
        if not self.enable_faiss or not self.faiss_index or not self.sqlite:
            return

        try:
            import pickle

            # Clear existing index and mapping
            self.faiss_index.reset()
            self.memory_id_map.clear()
            self.cache.clear()

            # Retrieve all memories with embeddings
            all_memories = await self.sqlite.get_all_memories_with_embeddings()

            indexed_count = 0
            for mem_dict in all_memories:
                # Deserialize embedding
                embedding = None
                if mem_dict.get('embedding'):
                    try:
                        embedding = pickle.loads(mem_dict['embedding'])
                    except Exception as e:
                        self.logger.log_error(e, {"phase": "embedding_deserialization", "memory_id": mem_dict['id']})
                        continue

                if embedding is None:
                    continue

                # Parse timestamp
                try:
                    timestamp = datetime.fromisoformat(mem_dict['timestamp'])
                except:
                    timestamp = datetime.now()

                # Create MemoryEntry and add to cache
                entry = MemoryEntry(
                    id=mem_dict['id'],
                    session_id=mem_dict['session_id'],
                    timestamp=timestamp,
                    memory_type=mem_dict['memory_type'],
                    content=mem_dict['content'],
                    embedding=embedding,
                    metadata=mem_dict.get('metadata', {}),
                    importance_score=mem_dict.get('importance_score', 0.5),
                    is_fact=mem_dict.get('is_fact', False),
                    entity_type=mem_dict.get('entity_type')
                )
                self.cache.append(entry)

                # Add to FAISS index
                try:
                    embedding_vector = np.array([embedding], dtype=np.float32)
                    self.faiss_index.add(embedding_vector)
                    self.memory_id_map.append(entry.id)
                    indexed_count += 1
                except Exception as e:
                    self.logger.log_error(e, {"phase": "faiss_add_on_rebuild", "memory_id": mem_dict['id']})

            # Save FAISS index
            if indexed_count > 0:
                self._save_faiss_index()

            self.logger.log_event("memory", "faiss_index_rebuilt_after_delete", {
                "indexed_count": indexed_count
            })

        except Exception as e:
            self.logger.log_error(e, {"phase": "rebuild_faiss_index"})

    async def search_memories(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search over memories (wrapper for REPL console).

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of memory dictionaries with scores
        """
        try:
            # Use existing semantic search
            results = await self.search_semantic(query, limit=k, threshold=0.3)

            # Convert to dictionary format with scores
            formatted_results = []
            for mem in results:
                formatted_results.append({
                    'id': mem.id,
                    'score': mem.importance_score,  # Using importance as proxy for relevance
                    'content': mem.content,
                    'is_fact': mem.is_fact,
                    'entity_type': mem.entity_type,
                    'memory_type': mem.memory_type,
                    'created_at': mem.timestamp.isoformat() if hasattr(mem.timestamp, 'isoformat') else str(mem.timestamp)
                })

            return formatted_results

        except Exception as e:
            self.logger.log_error(e, {"phase": "search_memories"})
            return []
