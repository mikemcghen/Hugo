# Hugo Architecture

## Overview

Hugo is built on a **layered hybrid architecture** combining local-first operation with optional cloud integration. The system emphasizes privacy, transparency, and autonomous evolution while maintaining strict ethical boundaries.

---

## System Layers

### 1. Core Layer

The foundational reasoning and memory systems.

#### Cognition Engine (`core/cognition.py`)

Multi-stage reasoning pipeline:

```
User Input
    ↓
[Perception Layer]
    ↓ (Intent, Tone, Emotion)
[Context Assembly]
    ↓ (Memories, Directives, Tasks)
[Synthesis Layer]
    ↓ (Reasoning Chain, Personality)
[Output Construction]
    ↓ (Response + Directive Checks)
[Post Reflection]
    ↓
Heuristic Updates
```

**Key Components**:
- Intent recognition
- Emotional context mapping
- Personality tone injection
- Multi-step reasoning
- Performance self-assessment

#### Memory Manager (`core/memory.py`)

Hybrid memory architecture:

- **Hot Cache**: Recent messages in RAM (100 entries)
- **Short-Term**: SQLite session database (fast, ephemeral)
- **Long-Term**: PostgreSQL with pgvector (persistent, searchable)
- **Vector Index**: FAISS local cache + pgvector for semantic search

**Memory Types**:
- **Episodic**: Conversations, events, interactions
- **Semantic**: Knowledge, concepts, learned patterns
- **Procedural**: Skills, capabilities, how-to knowledge

#### Reflection Engine (`core/reflection.py`)

Generates structured self-reflections:

- **Session Reflections**: End-of-conversation summaries
- **Performance Reflections**: Quality assessments
- **Macro Reflections**: Long-term trend analysis
- **Skill Reflections**: Capability evolution

Reflections serve as:
- Memory anchors for continuity
- Learning logs for improvement
- Narrative artifacts for transparency

#### Directive Filter (`core/directives.py`)

Ethical and behavioral guardrails:

**Three Layers**:
1. **Core Ethics**: Privacy, Truthfulness, Transparency, Loyalty, Autonomy
2. **Behavioral Conduct**: Non-manipulation, Empathy, Honesty, Constructive Conflict
3. **Autonomy Boundaries**: Sandbox, Consent, Duty Hierarchy, Self-Maintenance

Every response and action passes through directive checks before execution.

---

### 2. Runtime Layer

Service orchestration and user interfaces.

#### CLI (`runtime/cli.py`)

Command-line interface with subcommands:
- `hugo up/down` - Service management
- `hugo skill` - Skill operations
- `hugo reflect` - Generate reflections
- `hugo status/log` - System monitoring
- `hugo shell` - Interactive REPL

#### REPL (`runtime/repl.py`)

Interactive conversational interface:
- Command history
- Tab completion (future)
- Multi-line input support
- Rich formatting
- Session management

#### Service Manager (`runtime/service_manager.py`)

Docker container orchestration:
- Start/stop services
- Health checks
- Service connectivity verification
- Automatic restarts

---

### 3. Data Layer

Persistent storage with vector search.

#### SQLite Manager (`data/sqlite_manager.py`)

Short-term session memory:
- `recent_messages`: Conversation history
- `session_summary`: Session metadata
- `pending_tasks`: Active tasks
- `context_embeddings`: Cached vectors

Fast, file-based, cleared after consolidation.

#### PostgreSQL Manager (`data/postgres_manager.py`)

Long-term persistent memory:
- `messages`: Archived conversations with embeddings
- `reflections`: Self-reflection records
- `skills`: Installed capabilities
- `events`: System event log
- `audit_log`: Security audit trail

Uses pgvector extension for semantic similarity search.

#### Vector Search

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **Local Cache**: FAISS index for fast lookups
- **Persistent Store**: pgvector with cosine similarity
- **Use Cases**: Context retrieval, semantic skill matching, reflection analysis

---

### 4. Skills Layer

Dynamic capability extension system.

#### Structure

```
skills/
├── registry.json          # Validation states
└── <skill_name>/
    ├── skill.yaml        # Metadata
    ├── main.py           # Implementation
    ├── tests/            # Unit tests
    └── README.md
```

#### Skill Lifecycle

1. **Creation**: User or Hugo generates skill scaffold
2. **Implementation**: Logic written in `main.py`
3. **Testing**: Unit tests in `tests/`
4. **Validation**: Sandboxed execution, directive audit
5. **Registration**: Added to registry.json
6. **Execution**: Invoked via triggers or commands

#### Triggers

- **Manual**: Direct command invocation
- **Keyword**: Semantic keyword matching
- **Scheduled**: Time-based execution
- **Event**: System event-driven

---

### 5. Services Layer

External containerized services.

#### Whisper Service

- **Purpose**: Speech-to-text transcription
- **Model**: OpenAI Whisper (base/medium/large)
- **GPU**: CUDA acceleration via NVIDIA runtime
- **API**: FastAPI on port 8001
- **Endpoint**: POST /transcribe

#### Piper Service

- **Purpose**: Text-to-speech synthesis
- **Engine**: Piper TTS
- **Voices**: Multiple voice models (en_US-lessac-medium default)
- **GPU**: CUDA acceleration
- **API**: FastAPI on port 8002
- **Endpoint**: POST /synthesize

#### Claude Proxy

- **Purpose**: Local proxy for Claude API
- **Features**: Request caching, rate limiting, monitoring
- **API**: FastAPI on port 8003
- **Endpoint**: POST /v1/messages

#### PostgreSQL + pgvector

- **Purpose**: Long-term memory with vector search
- **Version**: PostgreSQL 16 + pgvector extension
- **Port**: 5432
- **Features**: Full-text search, vector similarity, JSONB support

---

## Data Flow

### User Message Processing

```
1. User Input (CLI/REPL)
       ↓
2. Cognition Engine
   - Perception: Extract intent, tone, emotion
   - Context: Retrieve relevant memories
   - Synthesis: Build reasoning chain
   - Output: Generate response
       ↓
3. Directive Filter
   - Check privacy compliance
   - Verify truthfulness
   - Detect manipulation
       ↓
4. Memory Storage
   - Cache in RAM
   - Store in SQLite (short-term)
   - Mark for PostgreSQL (if important)
       ↓
5. Response to User
       ↓
6. Post Reflection
   - Log performance metrics
   - Update heuristics
   - Trigger macro reflection if needed
```

### Reflection Generation

```
1. Trigger (session end, scheduled, manual)
       ↓
2. Retrieve Context
   - Session memories from SQLite
   - Historical patterns from PostgreSQL
   - Recent reflections
       ↓
3. Analysis
   - Identify patterns
   - Extract insights
   - Assess performance
       ↓
4. Generate Narrative
   - Structured reflection format
   - Human-readable summary
       ↓
5. Store
   - PostgreSQL reflections table
   - Export to docs/reflections/
       ↓
6. Update Continuity Markers
   - Personality baseline
   - Memory anchors
```

### Skill Execution

```
1. Trigger Match
   - Command, keyword, schedule, or event
       ↓
2. Load Skill
   - Read skill.yaml
   - Import main.py
       ↓
3. Validate Context
   - Check consent requirements
   - Verify sandboxing if needed
       ↓
4. Execute
   - Call skill.execute(context)
   - Capture output
       ↓
5. Log & Reflect
   - Record execution
   - Update success rate
   - Generate skill reflection if needed
       ↓
6. Return Result
```

---

## Operational Modes

### Interactive Mode
- CLI/REPL active
- User-driven interactions
- Real-time responses
- Full service availability

### Service Mode
- Background daemon
- Scheduled tasks only
- Reduced resource usage
- API endpoints active

### Low Power Mode
- Minimal activity
- Dreamlike processing
- Reflection-focused
- Reduced service load

### Maintenance Mode
- System updates
- Database optimization
- Skill validation
- Performance analysis

---

## Security Architecture

### Directive-Based Security

Every operation passes through directive filters:
- **Pre-execution**: Check consent requirements
- **Execution**: Monitor for violations
- **Post-execution**: Audit and log

### Consent System

Actions requiring consent:
- File system writes
- System commands
- External API calls
- Skill installation
- Configuration changes

### Audit Trail

All operations logged:
- Who/what triggered
- What was attempted
- What was approved/denied
- Results and side effects

### Data Privacy

- **Local First**: All data local by default
- **Vault System**: Encrypted secrets storage
- **Session Tokens**: Time-limited access
- **Zero Trust**: Verify every action

---

## Performance Considerations

### Memory Optimization

- **Hot Cache**: 100 recent messages in RAM
- **Lazy Loading**: Load old sessions on demand
- **Batch Operations**: Consolidate writes
- **Index Optimization**: Regular VACUUM and ANALYZE

### Vector Search Optimization

- **FAISS Cache**: Fast local lookups
- **Index Selection**: IVFFlat or HNSW based on data size
- **Dimension Reduction**: Consider PCA for large datasets
- **Batch Embedding**: Generate embeddings in batches

### GPU Utilization

- **Dynamic Allocation**: Share GPU between services
- **Model Loading**: Load on demand, cache in VRAM
- **Batch Processing**: Group requests for efficiency

---

## Extensibility

### Adding Core Modules

1. Create module in `core/`
2. Implement with async/await
3. Add to `core/__init__.py`
4. Integrate with runtime manager

### Adding Services

1. Create Dockerfile in `services/<name>/`
2. Add to docker-compose.yaml
3. Update service_manager.py
4. Add health check endpoint

### Custom Interfaces

1. Implement interface in `runtime/`
2. Use RuntimeManager for core access
3. Follow directive compliance patterns
4. Add to startup options

---

## Future Enhancements

### Near-Term
- Full cognition pipeline implementation
- Advanced vector search with FAISS
- Autonomous skill creation
- Desktop companion app

### Mid-Term
- Multi-modal input (images, PDFs)
- Advanced reasoning chains (tree-of-thought, etc.)
- Collaborative multi-agent workflows
- Browser extension

### Long-Term
- Federated learning across instances
- Compressed personality serialization
- Mobile companion apps
- IoT/smart home integration

---

## References

- [Memory Architecture](MEMORY.md)
- [Skills System](SKILLS.md)
- [Directives](DIRECTIVES.md)
- [API Documentation](API.md)
