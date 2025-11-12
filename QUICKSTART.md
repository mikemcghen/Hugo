# Hugo Quick Start Guide

Get Hugo up and running in 5 minutes with local Ollama!

---

## Prerequisites

- **Python 3.9+** installed
- **Ollama** installed ([ollama.com](https://ollama.com)) - for local LLM inference
- **Git** for cloning
- **Docker Desktop** (optional, for PostgreSQL long-term memory)

---

## Step 1: Install Hugo

```bash
# Clone repository
git clone https://github.com/yourusername/hugo.git
cd hugo

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Hugo
pip install -e .
```

---

## Step 2: Start Ollama

```bash
# Terminal 1: Start Ollama service
ollama serve

# Terminal 2: Pull Llama 3 8B model
ollama pull llama3:8b

# Verify it's running
curl http://localhost:11434/api/version
```

**Note**: The `.env` file is already configured for local Ollama operation. No API keys needed!

---

## Step 3: Start PostgreSQL (Optional)

PostgreSQL provides long-term memory persistence. If you want to skip this, Hugo will use in-memory storage.

### Option A: With Docker (Recommended)

```bash
# Start PostgreSQL with pgvector
docker compose -f configs/docker-compose.yaml up -d db

# Verify it's running
docker compose ps
```

### Option B: Without Docker

Hugo works perfectly without PostgreSQL using in-memory FAISS cache. Just skip to Step 4!

---

## Step 4: Start Hugo

**Quick Start Script (Easiest)**

Windows:
```cmd
start_hugo.bat
```

Unix/Mac:
```bash
./start_hugo.sh
```

**Manual Start**

```bash
# Interactive shell mode
python -m runtime.cli shell

# OR start as background service
python -m runtime.cli up
```

Hugo will:
- Load SentenceTransformers embedding model (first run downloads ~90MB)
- Initialize FAISS vector index
- Connect to Ollama
- Start interactive shell

---

## Step 5: Start Chatting!

Once in the shell, try:

```
You: Hello Hugo! Can you introduce yourself?

You: What's special about local-first AI?

You: Tell me about your memory system

You: exit
```

Hugo will respond using Ollama (Llama 3 8B) with full context awareness and semantic memory!

---

## Common Commands

```bash
# Interactive shell
python -m runtime.cli shell

# Check system status
python -m runtime.cli status --verbose

# View logs
python -m runtime.cli log --tail 50

# Manage skills
python -m runtime.cli skill --list
python -m runtime.cli skill --new my_skill

# Generate reflection
python -m runtime.cli reflect --days 1

# Stop Hugo
python -m runtime.cli down
```

---

## Troubleshooting

### "Ollama connection error"
```bash
# Start Ollama service
ollama serve

# Verify it's running
curl http://localhost:11434/api/version

# Pull the model
ollama pull llama3:8b
```

### "Module not found"
```bash
# Install all dependencies
pip install -r requirements.txt

# Specifically:
pip install faiss-cpu psycopg2-binary sentence-transformers requests
```

### "First run is slow"
- Normal! SentenceTransformers downloads the embedding model (~90MB) on first use
- Model is cached at `~/.cache/torch/sentence_transformers/`
- Subsequent runs are much faster

### PostgreSQL connection failed
```bash
# Start PostgreSQL (optional)
docker compose -f configs/docker-compose.yaml up -d db

# Or disable PostgreSQL in .env
ENABLE_POSTGRES=false
```

### Verify setup
```bash
# Run verification script
python verify_setup.py
```

---

## Next Steps

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Comprehensive setup and configuration
- **[INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md)** - Technical details and architecture
- Enable PostgreSQL for persistent long-term memory
- Explore the skills system: `python -m runtime.cli skill --list`
- Read architecture docs in `docs/`

---

## What's Working Now

✓ **Local LLM inference** via Ollama (Llama 3 8B)
✓ **Semantic memory search** using FAISS
✓ **Text embeddings** with SentenceTransformers
✓ **Conversation context** maintained across turns
✓ **Personality-aware responses** from Hugo
✓ **No API keys required** - fully local operation

---

## Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed setup and troubleshooting
- **[INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md)** - Technical implementation details
- **[verify_setup.py](verify_setup.py)** - Automated setup verification
- **[README.md](README.md)** - Full project documentation

---

**You're ready to go! Run `start_hugo.bat` or `./start_hugo.sh` to begin.**

For questions, see SETUP_GUIDE.md or run `python verify_setup.py` to diagnose issues.
