"""
Memory Manager
--------------
Implements Hugo's hybrid memory architecture with intelligent classification:
- Short-term: SQLite-based session memory (fast, ephemeral)
- Long-term: PostgreSQL with pgvector (persistent, searchable)
- Vector index: FAISS local cache + pgvector for semantic search

Memory Classification:
- factual: User-specific information (persisted, high priority)
- credential: Passwords, API keys, secrets (persisted, restricted retrieval)
- preference: User preferences and settings (persisted)
- identity: Self-identity, aspirations, goals (persisted, highest priority)
- task: Tasks, reminders, todos (persisted)
- knowledge: Definitions, explanations, facts about world (persisted)
- emotional: Emotional context and sentiment (persisted)
- note: Important notes and journal entries (persisted)
- conversation: General chitchat (not persisted)
- ignore: Noise, greetings, acknowledgments (not persisted)
"""

import asyncio
import os
import re
import requests
import json
import numpy as np
import faiss
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
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
    memory_type: str  # factual, credential, preference, identity, task, knowledge, emotional, note, conversation, reflection
    content: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    importance_score: float
    is_fact: bool = False  # Deprecated: Use memory_type == "factual"
    entity_type: Optional[str] = None  # animal, person, location, preference, etc.


@dataclass
class MemoryClassification:
    """Result of memory classification"""
    memory_type: str  # factual, credential, preference, identity, task, knowledge, emotional, note, conversation, ignore
    importance: float  # 0.0 to 1.0
    should_persist: bool
    embedding_allowed: bool = True
    entity_type: Optional[str] = None  # For factual memories
    reasoning: Optional[str] = None  # Why this classification was chosen
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryManager:
    """
    Manages Hugo's layered memory system with intelligent classification.

    Architecture:
    - Hot cache: Recent memories in RAM
    - Short-term: SQLite for current session
    - Long-term: PostgreSQL for persistent storage
    - Vector search: FAISS + pgvector for semantic retrieval
    """

    # Memory type priority for retrieval (higher = more important)
    MEMORY_PRIORITY = {
        "identity": 10,
        "factual": 9,
        "preference": 8,
        "emotional": 7,
        "knowledge": 6,
        "task": 5,
        "note": 4,
        "conversation": 3,
        "credential": 1,  # Low priority, only retrieved when explicitly requested
        "ignore": 0
    }

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

        # Ollama configuration for LLM-based classification
        self.ollama_api = os.getenv("OLLAMA_API", "http://localhost:11434/api/generate")
        self.classification_model = os.getenv("CLASSIFICATION_MODEL", "llama3:8b")
        self.enable_llm_classification = os.getenv("ENABLE_LLM_CLASSIFICATION", "false").lower() == "true"

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

    def classify_memory(self, message_text: str) -> MemoryClassification:
        """
        Classify a memory using rule-based detection with optional LLM fallback.

        Classification priority:
        1. Credentials (passwords, API keys, secrets)
        2. Identity (I am, I want to be, aspirations)
        3. Factual (user-specific declarative information)
        4. Preference (likes, dislikes, settings)
        5. Task (reminders, todos, obligations)
        6. Knowledge (definitions, explanations, facts about world)
        7. Emotional (feelings, sentiment, mood)
        8. Note (journal entries, important notes)
        9. Conversation (normal chitchat)
        10. Ignore (greetings, acknowledgments, noise)

        Args:
            message_text: Message content to classify

        Returns:
            MemoryClassification with type, importance, and persistence flag
        """
        text = message_text.strip()
        text_lower = text.lower()

        # Guard: Empty or very short messages are usually noise
        if len(text) < 3:
            return MemoryClassification(
                memory_type="ignore",
                importance=0.0,
                should_persist=False,
                embedding_allowed=False,
                reasoning="too_short"
            )

        # Guard: Questions are usually conversation unless they're tasks
        is_question = text.endswith('?')

        # Pattern 1: Credentials (HIGHEST PRIORITY for security)
        credential_patterns = [
            (r'password\s*(?:is|:|=)\s*[\w\d@#$%^&*()]+', "password"),
            (r'api[_\s]?key\s*(?:is|:|=)\s*[\w\d-]+', "api_key"),
            (r'secret\s*(?:is|:|=)\s*[\w\d]+', "secret"),
            (r'token\s*(?:is|:|=)\s*[\w\d]+', "token"),
            (r'access[_\s]?key\s*(?:is|:|=)\s*[\w\d]+', "access_key"),
        ]

        for pattern, cred_type in credential_patterns:
            if re.search(pattern, text_lower):
                return MemoryClassification(
                    memory_type="credential",
                    importance=0.95,
                    should_persist=True,
                    embedding_allowed=False,  # Don't embed sensitive data
                    entity_type=cred_type,
                    reasoning=f"credential:{cred_type}",
                    metadata={"credential_type": cred_type}
                )

        # Pattern 2: Identity (who I am, aspirations, goals)
        identity_patterns = [
            r'\bi am\b(?! (going|doing|working|thinking|feeling|happy|sad))',
            r'\bi want to be\b',
            r'\bi aspire to\b',
            r'\bmy goal is\b',
            r'\bmy purpose\b',
            r'\bmy mission\b',
            r'\bi define myself\b',
            r'\bi see myself as\b',
        ]

        for pattern in identity_patterns:
            if re.search(pattern, text_lower):
                return MemoryClassification(
                    memory_type="identity",
                    importance=0.95,
                    should_persist=True,
                    embedding_allowed=True,
                    entity_type="identity",
                    reasoning="identity_statement"
                )

        # Pattern 3: Factual information (declarative statements about user)
        factual_patterns = {
            "animal": [r'\b(cat|dog|pet|bird|fish|hamster|rabbit|parrot|bunny|bunnies)\b'],
            "person": [r'\bname is\b', r'\bcalled\b', r'\b(my wife|my husband|my friend|my child|my son|my daughter|my brother|my sister|my parent)\b'],
            "location": [r'\blive in\b', r'\bfrom\b', r'\bcity\b.*\bis\b', r'\bcountry\b.*\bis\b', r'\baddress\b'],
            "occupation": [r'\bwork as\b', r'\bjob\b.*\bis\b', r'\bcareer\b', r'\b(engineer|teacher|doctor|developer|designer|manager)\b'],
            "contact": [r'\bemail\b.*\bis\b', r'\bphone\b', r'\bnumber\b.*\bis\b', r'@\w+\.\w+'],
            "possession": [r'\b(own|have|got|bought)\b.*\b(car|house|computer|laptop|bike|watch)\b'],
            "birthday": [r'\bbirthday\b', r'\bborn on\b', r'\bborn in\b'],
            "routine": [r'\busually\b', r'\bevery (day|morning|evening|week)\b', r'\broutine\b'],
        }

        # Only classify as factual if NOT a question
        if not is_question:
            for entity_type, patterns in factual_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        # Additional check: Not a question starter
                        question_starters = ["do you", "can you", "will you", "would you", "could you",
                                            "are you", "is there", "did you", "have you", "what",
                                            "when", "where", "why", "how", "who"]
                        if any(text_lower.startswith(starter) for starter in question_starters):
                            continue

                        return MemoryClassification(
                            memory_type="factual",
                            importance=0.85,
                            should_persist=True,
                            embedding_allowed=True,
                            entity_type=entity_type,
                            reasoning=f"factual:{entity_type}"
                        )

        # Pattern 4: Preferences
        preference_patterns = [
            r'\b(favorite|prefer|like|love|enjoy|hate|dislike)\b',
            r'\b(interested in|hobby|hobbies)\b',
            r'\b(best|worst)\b.*\b(food|color|movie|song|book|game|music|genre)\b',
        ]

        if not is_question:
            for pattern in preference_patterns:
                if re.search(pattern, text_lower):
                    return MemoryClassification(
                        memory_type="preference",
                        importance=0.80,
                        should_persist=True,
                        embedding_allowed=True,
                        entity_type="preference",
                        reasoning="preference_detected"
                    )

        # Pattern 5: Tasks and reminders
        task_patterns = [
            r'\b(remind|remember)\b.*\bto\b',
            r'\bi need to\b',
            r'\bi have to\b',
            r'\bi must\b',
            r'\bdon\'t forget\b',
            r'\btodo\b',
            r'\btask\b',
        ]

        for pattern in task_patterns:
            if re.search(pattern, text_lower):
                return MemoryClassification(
                    memory_type="task",
                    importance=0.75,
                    should_persist=True,
                    embedding_allowed=True,
                    entity_type="task",
                    reasoning="task_or_reminder"
                )

        # Pattern 6: Knowledge (definitions, explanations, facts about the world)
        knowledge_patterns = [
            r'\bwhat is\b',
            r'\bwhat are\b',
            r'\bdefine\b',
            r'\bexplain\b',
            r'\bhow does\b',
            r'\bhow do\b',
            r'\btell me about\b',
        ]

        if is_question:
            for pattern in knowledge_patterns:
                if re.search(pattern, text_lower):
                    return MemoryClassification(
                        memory_type="knowledge",
                        importance=0.60,
                        should_persist=True,
                        embedding_allowed=True,
                        entity_type="knowledge",
                        reasoning="knowledge_query"
                    )

        # Pattern 7: Emotional content
        emotional_patterns = [
            r'\b(feel|feeling|felt|emotion)\b',
            r'\b(happy|sad|angry|frustrated|excited|anxious|worried|stressed|calm|peaceful|depressed|joyful)\b',
            r'\b(love|hate|fear|hope|wish)\b',
            r'\bi\'m (so|very)?\s*(happy|sad|angry|excited|worried|stressed)\b',
        ]

        for pattern in emotional_patterns:
            if re.search(pattern, text_lower):
                return MemoryClassification(
                    memory_type="emotional",
                    importance=0.70,
                    should_persist=True,
                    embedding_allowed=True,
                    entity_type="emotional",
                    reasoning="emotional_content"
                )

        # Pattern 8: Notes and journal entries (longer, reflective content)
        note_patterns = [
            r'\b(note|journal|diary)\b',
            r'\btoday (i|was)\b',
            r'\bthinking about\b',
            r'\brealized that\b',
            r'\blearned that\b',
        ]

        for pattern in note_patterns:
            if re.search(pattern, text_lower):
                return MemoryClassification(
                    memory_type="note",
                    importance=0.65,
                    should_persist=True,
                    embedding_allowed=True,
                    entity_type="note",
                    reasoning="note_or_journal"
                )

        # Pattern 9: Ignore (greetings, acknowledgments)
        ignore_patterns = [
            r'^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|sure|fine|good|great|awesome)[\s!.]*$',
            r'^(bye|goodbye|see you|later|cya)[\s!.]*$',
            r'^(um|uh|hmm|well)[\s.]*$',
        ]

        for pattern in ignore_patterns:
            if re.match(pattern, text_lower):
                return MemoryClassification(
                    memory_type="ignore",
                    importance=0.0,
                    should_persist=False,
                    embedding_allowed=False,
                    reasoning="greeting_or_acknowledgment"
                )

        # LLM Fallback: Use Ollama to classify ambiguous cases
        if self.enable_llm_classification:
            try:
                llm_classification = self._classify_with_llm(message_text)
                if llm_classification:
                    return llm_classification
            except Exception as e:
                self.logger.log_error(e, {"phase": "llm_classification_fallback"})

        # Default: Normal conversation (not persisted)
        return MemoryClassification(
            memory_type="conversation",
            importance=0.3,
            should_persist=False,
            embedding_allowed=False,
            reasoning="general_conversation"
        )

    def _classify_with_llm(self, message_text: str) -> Optional[MemoryClassification]:
        """
        Use Ollama LLM to classify ambiguous memories.

        Args:
            message_text: Message to classify

        Returns:
            MemoryClassification or None if classification fails
        """
        try:
            classification_prompt = f"""Classify this message into ONE of these memory types:
- factual: Personal facts about the user
- preference: User likes/dislikes
- identity: Who the user is or wants to be
- task: Something user needs to do
- knowledge: General knowledge or definition
- emotional: Emotional state or feelings
- note: Journal entry or important note
- conversation: General chitchat

Message: "{message_text}"

Return ONLY valid JSON in this format:
{{"type": "factual", "importance": 0.85, "reason": "brief explanation"}}"""

            payload = {
                "model": self.classification_model,
                "prompt": classification_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 100
                }
            }

            response = requests.post(self.ollama_api, json=payload, timeout=10)
            response.raise_for_status()

            result = response.json()
            llm_response = result.get("response", "").strip()

            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', llm_response)
            if json_match:
                classification_data = json.loads(json_match.group())
                memory_type = classification_data.get("type", "conversation")
                importance = float(classification_data.get("importance", 0.5))
                reason = classification_data.get("reason", "llm_classified")

                # Determine persistence
                should_persist = memory_type not in ["conversation", "ignore"]

                return MemoryClassification(
                    memory_type=memory_type,
                    importance=importance,
                    should_persist=should_persist,
                    embedding_allowed=should_persist,
                    reasoning=f"llm:{reason}"
                )

        except Exception as e:
            self.logger.log_error(e, {"phase": "llm_classification"})

        return None

    async def store(self, entry: MemoryEntry, persist_long_term: bool = False):
        """
        Store a memory entry using intelligent classification.

        Automatically classifies user messages and determines persistence.
        Only persistent memories are embedded and indexed in FAISS.

        Args:
            entry: MemoryEntry to store
            persist_long_term: If True, force long-term persistence
        """
        # Classify user messages
        if entry.memory_type == "user_message":
            classification = self.classify_memory(entry.content)

            # Update entry based on classification
            entry.memory_type = classification.memory_type
            entry.importance_score = max(entry.importance_score, classification.importance)

            # Merge metadata
            if classification.metadata:
                entry.metadata.update(classification.metadata)

            # Backward compatibility: Set is_fact for factual memories
            if classification.memory_type == "factual":
                entry.is_fact = True
                entry.entity_type = classification.entity_type

            # Update persistence flag
            if classification.should_persist:
                persist_long_term = True

            # Log classification
            self.logger.log_event("memory", "memory_classified", {
                "type": classification.memory_type,
                "importance": classification.importance,
                "should_persist": classification.should_persist,
                "embedding_allowed": classification.embedding_allowed,
                "entity_type": classification.entity_type,
                "reasoning": classification.reasoning,
                "content_preview": entry.content[:80]
            })

            # Don't store "ignore" type memories at all
            if classification.memory_type == "ignore":
                self.logger.log_event("memory", "memory_ignored", {
                    "reason": classification.reasoning,
                    "content_preview": entry.content[:80]
                })
                return

        # Force persistence for important memory types
        if entry.memory_type in ["identity", "factual", "credential", "preference", "task", "knowledge", "emotional", "note"]:
            persist_long_term = True

        # Generate embedding ONLY for persistent memories where allowed
        embedding_allowed = entry.memory_type not in ["credential", "ignore", "conversation"]
        if persist_long_term and embedding_allowed and entry.embedding is None and self.enable_faiss:
            entry.embedding = await self._generate_embedding(entry.content)

        # Add to cache
        self.cache.append(entry)
        if len(self.cache) > self.cache_size:
            self.cache.pop(0)

        # Add to FAISS index ONLY for persistent, embeddable memories
        if persist_long_term and embedding_allowed and self.enable_faiss and self.faiss_index and entry.embedding:
            try:
                embedding_vector = np.array([entry.embedding], dtype=np.float32)
                self.faiss_index.add(embedding_vector)
                self.memory_id_map.append(entry.id)

                # Periodically save index
                if self.faiss_index.ntotal % 100 == 0:
                    self._save_faiss_index()

            except Exception as e:
                self.logger.log_error(e)

        # Store in SQLite (short-term) - always store for session context
        await self._store_sqlite(entry)

        # Persist to SQLite memories table for long-term storage
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
                    "memory_type": entry.memory_type,
                    "importance": entry.importance_score,
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
            "importance": entry.importance_score,
            "entity_type": entry.entity_type
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

    async def retrieve_by_type(self, session_id: str, types: List[str], limit: int = 5) -> List[MemoryEntry]:
        """
        Retrieve memories filtered by memory types.

        Args:
            session_id: Session identifier
            types: List of memory types to retrieve
            limit: Maximum number of entries per type

        Returns:
            List of MemoryEntry objects matching the types
        """
        results = []
        for memory_type in types:
            type_memories = [
                e for e in self.cache
                if e.session_id == session_id and e.memory_type == memory_type
            ]
            # Sort by importance descending
            type_memories.sort(key=lambda m: m.importance_score, reverse=True)
            results.extend(type_memories[:limit])

        # Sort final results by type priority
        results.sort(key=lambda m: self.MEMORY_PRIORITY.get(m.memory_type, 0), reverse=True)

        return results[:limit * len(types)]

    async def search_semantic(self, query: str, limit: int = 10,
                            threshold: float = 0.6,
                            exclude_credentials: bool = True,
                            memory_types: Optional[List[str]] = None) -> List[MemoryEntry]:
        """
        Search memories using hybrid semantic + type-based ranking.

        Ranking formula:
        - Base: FAISS cosine similarity
        - Boost: +0.30 if memory_type='identity'
        - Boost: +0.20 if memory_type='factual'
        - Boost: +0.15 if memory_type='preference'
        - Boost: +0.10 if memory within last 30 days
        - Boost: +0.05 if memory_type='reflection'

        Credentials are excluded by default for security.

        Args:
            query: Search query text
            limit: Maximum results to return
            threshold: Minimum similarity score (0-1)
            exclude_credentials: If True, exclude credential memories (default: True)
            memory_types: Optional list of memory types to filter by

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

            # Hybrid ranking with type-based boosts
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

                # Exclude credentials unless explicitly requested
                if exclude_credentials and memory.memory_type == "credential":
                    continue

                # Filter by memory types if specified
                if memory_types and memory.memory_type not in memory_types:
                    continue

                score = base_similarity

                # Type-based boosts (using priority mapping)
                priority_boost = self.MEMORY_PRIORITY.get(memory.memory_type, 0) / 100.0
                score += priority_boost

                # Recency boost (+0.10 for last 30 days)
                age_days = (datetime.now() - memory.timestamp).days
                if age_days <= 30:
                    score += 0.10

                scored_results.append((score, memory))

            # Sort by score descending
            scored_results.sort(key=lambda x: x[0], reverse=True)

            # Extract top memories
            results = [memory for score, memory in scored_results[:limit]]

            self.logger.log_event("memory", "semantic_search_completed", {
                "query_length": len(query),
                "results": len(results),
                "memory_types": memory_types,
                "type_distribution": {
                    mem_type: sum(1 for m in results if m.memory_type == mem_type)
                    for mem_type in set(m.memory_type for m in results)
                }
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
        Retrieve factual memories from cache.

        Args:
            limit: Maximum number of facts to return

        Returns:
            List of factual MemoryEntry objects, sorted by importance
        """
        facts = [m for m in self.cache if m.memory_type == "factual" or m.is_fact]
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
                memory_type="factual",
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

        # Count by memory type
        type_counts = {}
        for mem in self.cache:
            mem_type = mem.memory_type
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1

        return {
            "cache_size": len(self.cache),
            "memory_types": type_counts,
            "factual_memories": type_counts.get("factual", 0),
            "credentials": type_counts.get("credential", 0),
            "preferences": type_counts.get("preference", 0),
            "identity": type_counts.get("identity", 0),
            "tasks": type_counts.get("task", 0),
            "sqlite_records": 0,  # TODO: Query actual count
            "postgres_records": 0,  # TODO: Query actual count
            "faiss_index_size": faiss_size,
            "faiss_enabled": self.enable_faiss,
            "embedding_model": self.embedding_model_name if self.enable_faiss else None
        }

    async def load_factual_memories(self):
        """
        Load all persistent memories from SQLite and hydrate cache + FAISS index.

        This should be called on boot to restore cross-session memories.
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
                    is_fact=mem_dict.get('is_fact', False),
                    entity_type=mem_dict.get('entity_type')
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
                    'memory_type': mem.memory_type,
                    'entity_type': mem.entity_type,
                    'is_fact': mem.is_fact,
                    'created_at': mem.timestamp.isoformat() if hasattr(mem.timestamp, 'isoformat') else str(mem.timestamp)
                })

            return formatted_results

        except Exception as e:
            self.logger.log_error(e, {"phase": "search_memories"})
            return []
