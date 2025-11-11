# Hugo Data Layer

## Overview

Hugo uses a hybrid memory architecture with two persistence layers:

### Short-Term Memory (SQLite)
- **Purpose**: Fast, ephemeral storage for current session
- **Location**: `data/memory/hugo_session.db`
- **Lifespan**: Session duration + configurable retention
- **Use Cases**:
  - Recent conversation messages
  - Active session context
  - Pending tasks
  - Cached embeddings

### Long-Term Memory (PostgreSQL with pgvector)
- **Purpose**: Persistent, searchable knowledge base
- **Location**: PostgreSQL database (local or remote)
- **Lifespan**: Permanent (with optional archival/pruning)
- **Use Cases**:
  - Historical conversations
  - Reflections and learnings
  - Skills registry
  - System events
  - Audit logs

## Data Flow

1. **Session Start**
   - New session created in SQLite
   - Recent context loaded from PostgreSQL

2. **During Session**
   - Messages stored in SQLite for fast access
   - High-importance items marked for long-term storage

3. **Session End**
   - Session reflection generated
   - Important memories consolidated to PostgreSQL
   - SQLite records archived or cleared

4. **Periodic Maintenance**
   - Old SQLite records pruned
   - PostgreSQL indices optimized
   - Vector embeddings updated

## Vector Search

Hugo uses vector embeddings for semantic search:

- **Embedding Model**: sentence-transformers (local)
- **Dimension**: 384 (default, configurable)
- **Index**: FAISS (local cache) + pgvector (persistent)
- **Use Cases**:
  - Finding relevant past conversations
  - Semantic skill matching
  - Context-aware memory retrieval

## Directory Structure

```
data/
├── memory/              # SQLite databases
│   └── hugo_session.db
├── reflections/         # Exported reflection files
├── logs/               # Structured logs
├── backups/            # Database backups
└── vault/              # Encrypted sensitive data
```

## Database Schema

See [database_schema.yaml](../database_schema.yaml) for complete schema definitions.

## Setup

### SQLite

SQLite databases are created automatically on first run. No setup required.

### PostgreSQL with pgvector

1. Install PostgreSQL 12+
2. Install pgvector extension:
   ```sql
   CREATE EXTENSION vector;
   ```
3. Update connection string in `configs/environment.env`
4. Run migrations (handled by Hugo on first boot)

## Development

### Using SQLiteManager

```python
from data.sqlite_manager import SQLiteManager

manager = SQLiteManager()
await manager.connect()

await manager.store_message(
    session_id="session_123",
    role="user",
    content="Hello Hugo!",
    importance=0.8
)

messages = await manager.get_recent_messages("session_123", limit=10)
```

### Using PostgresManager

```python
from data.postgres_manager import PostgresManager

manager = PostgresManager("postgresql://user:pass@localhost/hugo")
await manager.connect()

await manager.store_message(
    session_id="session_123",
    role="assistant",
    content="Hello! How can I help?",
    embedding=[0.1, 0.2, ...],  # 384-dim vector
    importance=0.7
)

similar = await manager.search_messages_semantic(
    query_embedding=[0.15, 0.18, ...],
    limit=5,
    threshold=0.75
)
```

## TODO

- [ ] Implement SQLAlchemy ORM models
- [ ] Add database migration system (Alembic)
- [ ] Implement FAISS local vector cache
- [ ] Add automatic backup scheduling
- [ ] Create data export utilities
- [ ] Implement privacy-preserving encryption for vault
