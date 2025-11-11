-- Hugo PostgreSQL Database Initialization
-- Creates schema and enables pgvector extension

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create hugo database (if running this manually)
-- Database is created by Docker, so this is optional
-- CREATE DATABASE hugo;

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    message_count INTEGER DEFAULT 0,
    context_summary TEXT,
    context_vector vector(384),
    mood TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding_vector vector(384),
    importance REAL DEFAULT 0.5,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Reflections table
CREATE TABLE IF NOT EXISTS reflections (
    id SERIAL PRIMARY KEY,
    type TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    session_id TEXT,
    summary TEXT NOT NULL,
    insights JSONB,
    patterns_observed JSONB,
    areas_for_improvement JSONB,
    confidence REAL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Skills table
CREATE TABLE IF NOT EXISTS skills (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    version TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'active',
    validation_status TEXT DEFAULT 'pending',
    installed_at TIMESTAMPTZ NOT NULL,
    last_executed TIMESTAMPTZ,
    execution_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Events table
CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    category TEXT NOT NULL,
    event_type TEXT NOT NULL,
    level TEXT NOT NULL,
    data JSONB,
    session_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Changes table
CREATE TABLE IF NOT EXISTS changes (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    change_type TEXT NOT NULL,
    component TEXT NOT NULL,
    description TEXT,
    old_value TEXT,
    new_value TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    action TEXT NOT NULL,
    resource TEXT NOT NULL,
    user_id TEXT,
    result TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indices
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_reflections_type ON reflections(type);
CREATE INDEX IF NOT EXISTS idx_reflections_timestamp ON reflections(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_category ON events(category);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_skills_status ON skills(status);

-- Create vector indices for similarity search
-- Note: ivfflat requires training data, so these are created as placeholder
-- They should be rebuilt with actual data after some records are inserted

-- For now, use simpler brute-force search (works for smaller datasets)
-- Will be optimized later with IVFFlat or HNSW indices

-- CREATE INDEX IF NOT EXISTS idx_messages_embedding ON messages
--     USING ivfflat (embedding_vector vector_cosine_ops) WITH (lists = 100);

-- CREATE INDEX IF NOT EXISTS idx_sessions_context ON sessions
--     USING ivfflat (context_vector vector_cosine_ops) WITH (lists = 100);

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO hugo_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO hugo_user;

-- Insert initial data or configuration if needed
-- (none for now)

-- Success message
SELECT 'Hugo database initialized successfully' AS status;
