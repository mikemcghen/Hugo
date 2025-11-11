# Hugo - The Right Hand

**Your Local-First Autonomous AI Assistant**

> *A self-evolving, locally grounded AI companion that acts as your second-in-command.*

---

## 🎯 Overview

Hugo is a local-first AI assistant designed to be your strategic companion and system liaison. Built with privacy, autonomy, and transparency at its core, Hugo operates within clear ethical boundaries while continuously learning and evolving.

### Key Features

- **🧠 Self-Reflective Learning**: Hugo generates reflections on interactions and performance, continuously improving reasoning and capabilities
- **🔒 Privacy First**: All data stays local by default, with full control over what gets shared
- **⚡ Hybrid Memory**: Fast SQLite for sessions, PostgreSQL with vector search for long-term knowledge
- **🎙️ Voice Capable**: Whisper for speech-to-text, Piper for text-to-speech (GPU-accelerated)
- **🔧 Dynamic Skills**: Create and install new capabilities on the fly
- **🎭 Adaptive Personality**: Mood-based responses (focused, reflective, conversational, operational)
- **📊 Transparent Operations**: Full audit logs and directive compliance tracking

---

## 📋 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Hugo Core                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Cognition   │  │   Memory     │  │  Reflection  │     │
│  │   Engine     │  │   Manager    │  │    Engine    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Directives  │  │  Scheduler   │  │    Logger    │     │
│  │    Filter    │  │              │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                     Services Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐  │
│  │ Whisper  │  │  Piper   │  │  Claude   │  │ Postgres │  │
│  │  (STT)   │  │  (TTS)   │  │   Proxy   │  │ +pgvector│  │
│  └──────────┘  └──────────┘  └───────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                    Interfaces                               │
│          CLI / REPL / API / Future: Desktop App             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Docker Desktop** (with GPU support for voice features)
- **NVIDIA GPU** (optional, for Whisper/Piper acceleration)
- **Anthropic API Key** (for Claude integration)

### Installation

1. **Clone the repository**
   ```bash
   cd hugo
   ```

2. **Copy environment template**
   ```bash
   cp configs/environment.env .env
   ```

3. **Edit `.env` and add your API keys**
   ```bash
   nano .env  # or use your preferred editor
   ```

4. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Start services**
   ```bash
   hugo up
   ```

6. **Enter interactive shell**
   ```bash
   hugo shell
   ```

---

## 🎮 Usage

### CLI Commands

```bash
# Start Hugo services
hugo up

# Stop Hugo services
hugo down

# Rebuild services
hugo rebuild

# Generate reflection report
hugo reflect --days 7 --type macro

# Manage skills
hugo skill --list
hugo skill --new my_skill
hugo skill --validate my_skill

# View system status
hugo status --verbose

# View logs
hugo log --category reflection --tail 50

# Interactive shell
hugo shell
```

### Interactive Shell

```
You: Hello Hugo!

Hugo: Hello! I'm ready to help. What would you like to work on?

You: Tell me about your capabilities

Hugo: I'm your local-first AI assistant with several key capabilities:
- Conversational interaction with contextual memory
- Self-reflection and continuous learning
- Dynamic skill system for extending functionality
- Voice input/output (Whisper + Piper)
- Privacy-first operation with full local data control
...
```

---

## 📁 Project Structure

```
hugo/
├── core/                   # Core reasoning and memory systems
│   ├── cognition.py       # Multi-layer reasoning engine
│   ├── memory.py          # Hybrid memory management
│   ├── reflection.py      # Self-reflection engine
│   ├── directives.py      # Ethical guardrails
│   ├── scheduler.py       # Maintenance & evolution
│   └── ...
├── runtime/               # CLI and service orchestration
│   ├── cli.py            # Command-line interface
│   ├── repl.py           # Interactive shell
│   └── service_manager.py
├── skills/                # Dynamic skill system
│   ├── registry.json     # Skill registry
│   └── demo_skill/       # Example skill
├── data/                  # Persistent storage
│   ├── memory/           # SQLite session databases
│   ├── reflections/      # Reflection exports
│   ├── logs/             # Structured logs
│   └── vault/            # Encrypted secrets
├── services/              # Docker service definitions
│   ├── whisper/          # Speech-to-text
│   ├── piper/            # Text-to-speech
│   ├── claude_proxy/     # Claude API proxy
│   └── db/               # PostgreSQL + pgvector
├── configs/               # Configuration files
│   ├── docker-compose.yaml
│   ├── environment.env
│   └── hugo_manifest.yaml
└── docs/                  # Documentation
```

---

## 🧩 Skills System

Hugo's capabilities can be extended through a dynamic skill system.

### Creating a New Skill

```bash
hugo skill --new my_awesome_skill
```

This creates:
```
skills/my_awesome_skill/
├── skill.yaml          # Metadata and configuration
├── main.py             # Skill implementation
├── tests/              # Unit tests
│   └── test_main.py
└── README.md
```

### Skill Structure

**skill.yaml**:
```yaml
name: my_awesome_skill
version: "0.1.0"
description: "Does something awesome"
triggers:
  - type: manual
    command: "awesome"
parameters:
  - name: target
    type: string
    required: true
```

**main.py**:
```python
async def execute(context: dict) -> dict:
    """Execute the skill"""
    # Your logic here
    return {
        "success": True,
        "result": "Result data",
        "message": "Human-readable message"
    }
```

### Validating Skills

```bash
hugo skill --validate my_awesome_skill
```

---

## 🔐 Security & Privacy

### Core Principles

1. **Privacy First**: All data stored locally by default
2. **Consent Required**: Explicit approval for file writes, system commands, external API calls
3. **Transparent Operations**: Full audit logs of all actions
4. **Directive Compliance**: Every response checked against ethical guidelines
5. **Sandboxed Execution**: New skills tested in isolation

### Directives

Hugo operates under three layers of directives:

**Core Ethics**:
- Privacy First
- Truthfulness
- Transparency
- Loyalty
- Autonomy with Accountability

**Behavioral Conduct**:
- Non-Manipulation
- Empathic Precision
- Intellectual Honesty
- Constructive Conflict

**Autonomy Boundaries**:
- Sandbox Rule: Test changes in isolation
- Consent Rule: Ask before irreversible actions
- Duty Hierarchy: User > System > Self
- Self-Maintenance: Preserve core identity

---

## 🧠 Memory & Reflection

### Hybrid Memory Architecture

**Short-Term (SQLite)**:
- Current session messages
- Active tasks
- Cached embeddings
- Cleared after consolidation

**Long-Term (PostgreSQL + pgvector)**:
- Historical conversations
- Reflections and learnings
- Skills registry
- System events
- Permanent storage with vector search

### Reflection Types

1. **Session Reflections**: End-of-session learning summaries
2. **Performance Reflections**: Reasoning quality assessments
3. **Macro Reflections**: Periodic trend analysis (weekly, monthly)
4. **Skill Reflections**: Capability development insights

---

## ⚙️ Configuration

### Environment Variables

Key settings in `.env`:

```bash
# API Keys
ANTHROPIC_API_KEY=your_key_here

# Database
DB_PASSWORD=secure_password
POSTGRES_CONNECTION_STRING=postgresql://hugo_user:pass@localhost:5432/hugo

# Services
WHISPER_MODEL=base          # tiny, base, small, medium, large
PIPER_VOICE=en_US-lessac-medium

# Features
ENABLE_VOICE_INPUT=true
ENABLE_AUTONOMOUS_MAINTENANCE=true
REQUIRE_CONSENT_FILE_WRITE=true

# Performance
USE_GPU=true
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## 🛠️ Development

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/core/test_cognition.py

# Skills tests
pytest skills/demo_skill/tests/
```

### Database Migrations

```bash
# TODO: Add migration commands when Alembic is integrated
```

### Adding Core Modules

1. Create module in `core/`
2. Implement with proper docstrings
3. Add tests in `tests/core/`
4. Update `core/__init__.py`

---

## 📊 Monitoring & Logs

### View Logs

```bash
# All logs
hugo log

# Filtered by category
hugo log --category reflection

# Follow mode
hugo log --follow

# Specific count
hugo log --tail 100
```

### Log Categories

- **event**: System events and state changes
- **reflection**: Self-reflection entries
- **performance**: Metrics and diagnostics
- **error**: Exceptions and failures
- **security**: Directive violations, access attempts
- **user**: User interactions and sessions

---

## 🎯 Roadmap

### v0.1.0 (Current)
- ✅ Core reasoning engine
- ✅ Hybrid memory system
- ✅ CLI interface
- ✅ Skills system
- ✅ Docker services

### v0.2.0 (Planned)
- [ ] Full cognition pipeline implementation
- [ ] Vector search integration
- [ ] Autonomous skill creation
- [ ] Desktop companion app
- [ ] Voice interface polish

### v0.3.0 (Future)
- [ ] Multi-modal input (images, documents)
- [ ] Advanced reasoning chains
- [ ] Collaborative multi-agent system
- [ ] Browser extension
- [ ] Mobile companion

---

## 🤝 Contributing

Hugo is a personal project, but contributions and feedback are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

[To be determined - likely MIT or Apache 2.0]

---

## 🙏 Acknowledgments

Built with:
- **Anthropic Claude** - Core reasoning
- **OpenAI Whisper** - Speech recognition
- **Piper TTS** - Voice synthesis
- **PostgreSQL + pgvector** - Vector database
- **FastAPI** - Service APIs
- **Docker** - Containerization

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/hugo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/hugo/discussions)
- **Documentation**: [docs/](docs/)

---

**Hugo - Your Right Hand in the Digital World** 🤖✨
