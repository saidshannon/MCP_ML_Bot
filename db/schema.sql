-- Phase 1 schema for Autonomous ML Operations Agent Platform

CREATE TABLE IF NOT EXISTS runs (
    run_id SERIAL PRIMARY KEY,
    task_description TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    finished_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS agent_calls (
    call_id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES runs(run_id) ON DELETE CASCADE,
    agent_name TEXT NOT NULL,
    input JSONB,
    output JSONB,
    duration_ms INTEGER,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);