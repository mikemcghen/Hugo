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
    memory_type: str  # episodic, semantic, procedural
    content: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    importance_score: float


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

    async def store(self, entry: MemoryEntry, persist_long_term: bool = False):
        """
        Store a memory entry in appropriate layer(s).

        Args:
            entry: MemoryEntry to store
            persist_long_term: If True, also write to PostgreSQL
        """
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

        # Optionally persist to PostgreSQL
        if persist_long_term:
            await self._store_postgres(entry)

        self.logger.log_event("memory", "stored", {
            "type": entry.memory_type,
            "long_term": persist_long_term,
            "has_embedding": entry.embedding is not None
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
                            threshold: float = 0.7) -> List[MemoryEntry]:
        """
        Search memories using semantic similarity via FAISS.

        Args:
            query: Search query text
            limit: Maximum results to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of semantically similar MemoryEntry objects
        """
        if not self.enable_faiss or not self.faiss_index or not self.embedding_model:
            self.logger.log_event("memory", "semantic_search_unavailable", {})
            return []

        try:
            # Generate embedding for query
            query_embedding = await self._generate_embedding(query)
            query_vector = np.array([query_embedding], dtype=np.float32)

            # Search FAISS index
            # Note: FAISS uses L2 distance, we'll need to convert to cosine similarity
            k = min(limit * 2, self.faiss_index.ntotal)  # Search more, filter later
            if k == 0:
                return []

            distances, indices = self.faiss_index.search(query_vector, k)

            # Convert L2 distances to similarity scores (approximate cosine similarity)
            # For normalized vectors: cosine_similarity ≈ 1 - (L2_distance^2 / 2)
            similarities = 1 - (distances[0] ** 2 / 2)

            # Filter by threshold and map to memory entries
            results = []
            for idx, similarity in zip(indices[0], similarities):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue

                if similarity >= threshold:
                    # Retrieve memory from cache or database
                    # For now, return placeholder entries
                    # TODO: Implement actual memory retrieval by ID
                    if idx < len(self.cache):
                        results.append(self.cache[idx])

                if len(results) >= limit:
                    break

            self.logger.log_event("memory", "semantic_search_completed",
                                {"query_length": len(query), "results": len(results)})

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

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        faiss_size = self.faiss_index.ntotal if self.faiss_index else 0

        return {
            "cache_size": len(self.cache),
            "sqlite_records": 0,  # TODO: Query actual count
            "postgres_records": 0,  # TODO: Query actual count
            "faiss_index_size": faiss_size,
            "faiss_enabled": self.enable_faiss,
            "embedding_model": self.embedding_model_name if self.enable_faiss else None
        }
