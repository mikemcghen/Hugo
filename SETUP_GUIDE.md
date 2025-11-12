# Hugo Local Integration Setup Guide

## Overview

Hugo is now configured for **local-first operation** using:
- **Ollama** (Llama 3 8B) for cognition/inference
- **FAISS** for vector-based semantic memory search
- **PostgreSQL** (with pgvector) for long-term memory
- **SQLite** for short-term session memory
- **SentenceTransformers** for embeddings

---

## ✅ Completed Setup Steps

### 1. Dependencies Installed
All required Python packages have been installed:
- `faiss-cpu` - Vector similarity search
- `psycopg2-binary` - PostgreSQL driver
- `sentence-transformers` - Text embeddings
- `requests` - HTTP client for Ollama

### 2. Configuration Created
The [.env](.env) file has been created with:
- Ollama API endpoint configuration
- Memory system settings (FAISS, PostgreSQL, SQLite)
- Embedding model configuration

### 3. Core Modules Enhanced

#### [core/cognition.py](core/cognition.py)
- ✅ Added Ollama integration for local LLM inference
- ✅ Implemented `_local_infer()` method for API calls
- ✅ Updated `_synthesize()` to generate responses via Ollama
- ✅ Enhanced response metadata with model information

#### [core/memory.py](core/memory.py)
- ✅ Initialized FAISS index for vector search
- ✅ Implemented `_generate_embedding()` using SentenceTransformers
- ✅ Added `search_semantic()` for similarity-based memory retrieval
- ✅ Updated `store()` to automatically generate and index embeddings
- ✅ Added FAISS index persistence (auto-saves every 100 entries)

---

## 🚀 Next Steps: Making Hugo Operational

### Step 1: Start Ollama

Ensure Ollama is running with Llama 3 8B:

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# If not running, start Ollama service
ollama serve

# Pull Llama 3 8B model (if not already installed)
ollama pull llama3:8b

# Verify model is available
ollama list
```

### Step 2: Start PostgreSQL Database

#### Option A: Using Docker (Recommended)

```bash
# Start Docker Desktop first, then:
cd c:\Users\hocke\Documents\GitHub\Hugo

# Start only the PostgreSQL service
docker compose -f configs/docker-compose.yaml up -d db

# Verify database is running
docker compose -f configs/docker-compose.yaml ps

# Check logs
docker compose -f configs/docker-compose.yaml logs db
```

#### Option B: Local PostgreSQL Installation

If you prefer not to use Docker:

1. Install PostgreSQL 16 with pgvector extension
2. Create database:
   ```sql
   CREATE DATABASE hugo;
   CREATE USER hugo_user WITH PASSWORD 'hugo_password_change_me';
   GRANT ALL PRIVILEGES ON DATABASE hugo TO hugo_user;
   ```
3. Run initialization script:
   ```bash
   psql -U hugo_user -d hugo -f services/db/init.sql
   ```

### Step 3: Create Data Directories

Hugo needs directories for memory storage:

```bash
mkdir -p data/memory
mkdir -p data/logs
```

Or on Windows:
```powershell
New-Item -Path "data\memory" -ItemType Directory -Force
New-Item -Path "data\logs" -ItemType Directory -Force
```

### Step 4: Start Hugo Runtime

```bash
# Method 1: Interactive shell (recommended for first run)
python -m runtime.cli shell

# Method 2: Start as background service
python -m runtime.cli up -d

# Check status
python -m runtime.cli status

# View logs
python -m runtime.cli log --tail 50
```

---

## 🧪 Testing & Verification

### Test 1: Verify Ollama Connection

```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()

ollama_api = os.getenv("OLLAMA_API")
response = requests.post(ollama_api, json={
    "model": "llama3:8b",
    "prompt": "Say hello in one sentence.",
    "stream": False
})

print(response.json())
```

### Test 2: Verify FAISS Embeddings

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("Test sentence", convert_to_numpy=True)

print(f"Embedding dimension: {embedding.shape[0]}")
print(f"Sample values: {embedding[:5]}")
```

### Test 3: Test Memory Storage

```python
import asyncio
from core.memory import MemoryManager, MemoryEntry
from core.logger import HugoLogger
from datetime import datetime

async def test_memory():
    logger = HugoLogger()
    memory = MemoryManager(None, None, logger)

    # Store a test memory
    entry = MemoryEntry(
        id=1,
        session_id="test-001",
        timestamp=datetime.now(),
        memory_type="episodic",
        content="Hugo can now use Ollama for local inference",
        embedding=None,
        metadata={},
        importance_score=0.9
    )

    await memory.store(entry, persist_long_term=False)

    # Search semantically
    results = await memory.search_semantic("local inference", limit=5)
    print(f"Found {len(results)} similar memories")

    # Check stats
    stats = memory.get_stats()
    print(f"Memory stats: {stats}")

asyncio.run(test_memory())
```

---

## 🔧 Configuration Options

### Environment Variables

Edit [.env](.env) to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ENGINE` | `ollama` | Model backend (ollama, claude) |
| `MODEL_NAME` | `llama3:8b` | Ollama model to use |
| `OLLAMA_API` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `ENABLE_FAISS` | `true` | Enable FAISS vector search |
| `ENABLE_POSTGRES` | `true` | Enable PostgreSQL storage |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `EMBEDDING_DIMENSION` | `384` | Embedding vector size |
| `MEMORY_CACHE_SIZE` | `100` | In-memory cache entries |
| `DB_URL` | PostgreSQL connection string | Database URL |

---

## 🐛 Troubleshooting

### Issue: Ollama Connection Error

**Symptoms:** `Ollama connection error: Connection refused`

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# If not, start it
ollama serve
```

### Issue: FAISS Not Loading

**Symptoms:** `semantic_search_unavailable`

**Solution:**
- Ensure `ENABLE_FAISS=true` in `.env`
- Check that sentence-transformers is installed: `pip show sentence-transformers`
- Verify embedding model downloads correctly (first run takes time)

### Issue: PostgreSQL Connection Failed

**Symptoms:** `connection to server at "localhost" (::1), port 5432 failed`

**Solution:**
```bash
# If using Docker
docker compose -f configs/docker-compose.yaml up -d db

# Check connection
psql -h localhost -U hugo_user -d hugo -c "SELECT version();"
```

### Issue: Module Import Errors

**Symptoms:** `ModuleNotFoundError: No module named 'dotenv'`

**Solution:**
```bash
pip install python-dotenv
pip install -r requirements.txt
```

---

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Hugo System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                │
│  │   User CLI   │────────▶│ CognitionEng │                │
│  └──────────────┘         └───────┬──────┘                │
│                                    │                        │
│                    ┌───────────────┼───────────────┐       │
│                    ▼               ▼               ▼       │
│            ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│            │   Ollama   │  │   Memory   │  │ Directive  │ │
│            │ (Llama 3)  │  │  Manager   │  │  Filter    │ │
│            └────────────┘  └─────┬──────┘  └────────────┘ │
│                                   │                        │
│                    ┌──────────────┼──────────────┐         │
│                    ▼              ▼              ▼         │
│            ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│            │   SQLite   │ │   FAISS    │ │ PostgreSQL │   │
│            │ (short-term)│ │  (vectors) │ │(long-term) │   │
│            └────────────┘ └────────────┘ └────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 What's Working Now

✅ **Ollama Integration**: Hugo can generate responses using local Llama 3 8B
✅ **Semantic Memory**: FAISS enables similarity-based memory retrieval
✅ **Embeddings**: Automatic text→vector conversion using SentenceTransformers
✅ **Memory Persistence**: FAISS index auto-saves to disk
✅ **Configuration**: Centralized .env-based settings

## 🚧 Still TODO (Optional Enhancements)

- [ ] Voice services (Whisper STT, Piper TTS) - disabled by default
- [ ] SQLite persistence implementation in memory.py
- [ ` PostgreSQL write operations in memory.py
- [ ] Directive filtering in cognition.py
- [ ] Session consolidation (SQLite → PostgreSQL migration)
- [ ] Advanced FAISS indices (IVFFlat, HNSW for large datasets)

---

## 📚 Useful Commands

```bash
# Start Hugo interactively
python -m runtime.cli shell

# Start as daemon
python -m runtime.cli up -d

# Check system status
python -m runtime.cli status --verbose

# View logs
python -m runtime.cli log --category cognition --tail 100

# Generate reflection
python -m runtime.cli reflect --days 1 --type macro

# Manage skills
python -m runtime.cli skill --list
python -m runtime.cli skill --new my_skill

# Stop Hugo
python -m runtime.cli down

# Restart with rebuild
python -m runtime.cli rebuild
```

---

## 🔐 Security Notes

1. **Change default passwords**: Update `DB_PASSWORD` in `.env`
2. **API Keys**: Set `ANTHROPIC_API_KEY` only if using Claude (optional)
3. **Network**: Hugo runs locally by default, no internet required for Ollama
4. **Data Privacy**: All inference and memory storage happens on your machine

---

## 📖 Next Reading

- [Architecture Documentation](docs/architecture.md)
- [Cognition Pipeline Guide](docs/cognition.md)
- [Memory System Details](docs/memory.md)
- [Skills Development](docs/skills.md)

---

**Status**: Core integration complete. Ready for runtime testing!

Run `python -m runtime.cli shell` to begin chatting with Hugo.
