# Hugo — The Right Hand

Your local-first autonomous AI assistant. Hugo runs entirely on your own hardware using a local Ollama model, with persistent memory, infrastructure control, voice I/O, and a pluggable skill system — no data leaves your network.

---

## What Hugo Can Do

- **Conversation with memory** — Hugo remembers facts, goals, and context across sessions using FAISS semantic search over SQLite/PostgreSQL
- **Infrastructure control** — Run SSH commands, manage Docker containers, and monitor network hosts via natural language or explicit `/skill` commands
- **Multi-step delegation** — Compound requests ("check all hosts and restart anything that's down") are broken into parallel subtasks automatically
- **Skill system** — Pluggable skills for web search, URL fetching, notes, and more
- **Voice I/O** — Whisper (STT) + Piper (TTS) for hands-free use
- **Adaptive persona** — Jarvis-style personality with mood-aware responses (focused, reflective, conversational, operational)
- **Self-reflection** — Generates session reflections for continuous improvement
- **CORE / FULL modes** — CORE is a lean, stable pipeline; FULL enables all infrastructure capabilities

---

## Architecture

```
User Input
    │
    ├─ CORE mode ──────────────────────► Minimal LLM pipeline (stable, no infra)
    │
    ├─ Pending confirmation + yes/no ──► Execute or cancel pending intent
    │
    ├─ /skill command ─────────────────► Skill router → handler → executor
    │       /ssh, /docker, /monitor route to real infrastructure executors
    │
    ├─ Natural language (FULL mode)
    │       └─ HugoIntentParser (local LLM, confidence >= 0.75)
    │               ├─ memory domain ──► direct memory lookup
    │               ├─ compound request ► DelegationAgent (parallel subtasks)
    │               │                         └─ InfraAgent → ActionRouter
    │               └─ single action ──► ActionRouter → PermissionGate → Executor
    │                       ASK_FIRST ──► returns confirmation prompt
    │
    └─ Standard cognitive pipeline
            _perceive() → _assemble_context() → _synthesize() → PersonaEngine
```

### Key Modules

| Path | Purpose |
|---|---|
| `core/cognition.py` | Main reasoning pipeline |
| `core/memory.py` | FAISS + SQLite/PostgreSQL memory manager |
| `core/persona_engine.py` | Jarvis persona, tone shaping, mood spectrum |
| `core/ollama_stability.py` | Resilient Ollama connection with retries |
| `core/intent/intent_parser.py` | Natural language → ParsedIntent (local LLM) |
| `core/actions/action_router.py` | Routes intents to executors with permission gating |
| `core/actions/permission.py` | AUTO / EXECUTE_AND_REPORT / ASK_FIRST levels |
| `core/executors/ssh.py` | SSH executor (hosts from `configs/ssh_hosts.yaml`) |
| `core/executors/docker.py` | Docker container management (via SSH) |
| `core/executors/monitor.py` | Network ping / uptime monitoring |
| `agents/delegation_agent.py` | Breaks compound tasks into parallel subtasks |
| `agents/infra_agent.py` | Wraps ActionRouter for DelegationAgent subtasks |
| `skills/` | Pluggable skill system |
| `runtime/cli.py` | CLI entry point (`hugo` command) |
| `runtime/repl.py` | Interactive REPL |

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) running locally with a model pulled (e.g. `ollama pull llama3`)
- PostgreSQL with pgvector (for long-term semantic memory) — optional; falls back to SQLite
- NVIDIA GPU recommended for embedding model (`sentence-transformers`) and Whisper voice features

---

## Setup

**1. Clone and create a virtual environment**

```bash
cd Hugo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure environment**

```bash
cp configs/environment.env .env
# Edit .env — key values:
#   HUGO_MODE=interactive          # interactive | service | low_power
#   MODEL_NAME=llama3:8b           # any Ollama model
#   DB_PASSWORD=...                # PostgreSQL password
#   ANTHROPIC_API_KEY=...          # optional, for Claude fallback
```

**3. Start Ollama**

```bash
ollama serve
ollama pull llama3
```

**4. Configure SSH hosts** *(optional — required for infrastructure control)*

Edit `configs/ssh_hosts.yaml`:

```yaml
hosts:
  - name: homeserver
    ip: 192.168.1.100
    user: ubuntu
    key: ~/.ssh/id_rsa
    description: Primary home server
```

**5. Run Hugo**

```bash
# Interactive REPL
python -m runtime.cli shell

# As a background service
python runtime/startup.py
```

---

## Skill Commands

Hugo understands both natural language and explicit `/skill` commands.

| Command | Description |
|---|---|
| `/search <query>` | Web search |
| `/fetch <url>` | Fetch and summarize a URL |
| `/note <text>` | Save a note to memory |
| `/ssh run host=<name> command=<cmd>` | Run a command on a remote host |
| `/ssh list_hosts` | List configured SSH hosts |
| `/docker list` | List running containers |
| `/docker restart container=<name>` | Restart a container |
| `/docker logs container=<name>` | View container logs |
| `/monitor status` | Network status overview |
| `/monitor check host=<name>` | Check if a host is up |

Natural language equivalents work in FULL mode:
- *"list all docker containers"*
- *"restart the nginx container"*
- *"is homeserver up?"*
- *"run df -h on homeserver"*
- *"what do you remember about my goals?"*
- *"check all hosts and restart anything that's down"*

---

## Permission Levels

Infrastructure actions are gated automatically:

| Level | Examples | Behavior |
|---|---|---|
| `AUTO_EXECUTE` | Monitor status, Docker list | Runs immediately |
| `EXECUTE_AND_REPORT` | Container restart | Runs and reports result |
| `ASK_FIRST` | SSH writes, destructive commands | Prompts for confirmation first |

---

## Operational Modes

Set `HUGO_MODE` in `.env`:

| Mode | Description |
|---|---|
| `interactive` | Full capabilities — NL intent parsing, executors, delegation |
| `service` | Same as interactive, runs as a background daemon |
| `low_power` / `core` | Lean pipeline — no infrastructure calls, minimal resource use |

---

## Memory

Hugo uses a two-tier memory system:

**Short-term (SQLite)** — Current session messages, active tasks, cached embeddings. Consolidated at session end.

**Long-term (PostgreSQL + pgvector)** — Facts, goals, relationships, reflections, and conversation history with semantic vector search. Persists forever by default.

You can query memory naturally: *"what do you remember about project Metix?"* or *"recall my goals"*.

---

## Privacy & Safety

- All data stays local — no external API calls unless you explicitly trigger a web skill
- Dangerous shell patterns are blocked unconditionally at the executor level (`rm -rf /`, fork bombs, pipe-to-shell, etc.)
- Consent flags in `.env` control which action categories require manual approval
- Every action is logged with full audit trail

---

## Project Structure

```
Hugo/
├── agents/              # Multi-agent system
│   ├── base_agent.py    # BaseAgent, AgentTask, AgentResult
│   ├── registry.py      # @AgentRegistry.register decorator
│   ├── delegation_agent.py  # Parallel multi-step task coordinator
│   └── infra_agent.py   # Infrastructure subtask executor
├── configs/
│   ├── environment.env  # Environment variable template
│   ├── ssh_hosts.yaml   # SSH host configuration
│   ├── hugo_manifest.yaml  # Persona definition
│   └── docker-compose.yaml
├── core/
│   ├── actions/         # ActionResult, PermissionGate, ActionRouter
│   ├── executors/       # SSH, Docker, Monitor executors + registry
│   ├── intent/          # ParsedIntent, HugoIntentParser
│   ├── skills/          # Skill trigger detection and routing
│   ├── cognition.py     # Main reasoning pipeline
│   ├── memory.py        # FAISS + SQLite/PostgreSQL memory
│   ├── ollama_stability.py
│   ├── persona_engine.py
│   └── reflection.py
├── runtime/             # CLI, REPL, startup, service manager
├── services/            # Whisper STT, Piper TTS, Claude proxy microservices
├── skills/              # Pluggable skill definitions
├── tests/               # Unit and integration tests (117 tests)
└── requirements.txt
```

---

## Testing

```bash
pytest tests/ -v
```

117 tests covering permission gating, intent parsing, all three executors, action routing, and the full cognition integration pipeline.

---

## Roadmap

### Done
- Core reasoning and cognitive pipeline
- Hybrid memory (SQLite + PostgreSQL + FAISS)
- Jarvis persona with mood spectrum and self-reflection
- Infrastructure executors (SSH, Docker, Monitor)
- NL intent parsing with permission gating
- Multi-step delegation via DelegationAgent
- Skill system (web search, fetch, notes)
- CLI and interactive REPL
- Docker service orchestration

### Planned
- Voice interface polish (Whisper + Piper integration)
- Autonomous skill creation
- Browser extension
- Desktop companion app

---

## Built With

- [Ollama](https://ollama.ai) — Local LLM inference
- [OpenAI Whisper](https://github.com/openai/whisper) — Speech-to-text
- [Piper TTS](https://github.com/rhasspy/piper) — Voice synthesis
- [PostgreSQL + pgvector](https://github.com/pgvector/pgvector) — Vector database
- [FAISS](https://github.com/facebookresearch/faiss) — Semantic memory search
- [FastAPI](https://fastapi.tiangolo.com) — Service APIs

---

MIT License
