# Autonomous ML Operations Agent Platform

A multi-agent system for AI engineering workflows using LangGraph and MCP integration.

## Goal

Build agents that orchestrate ML workflows for data analysis, modeling, and reporting while comparing multiple LLM providers and offering observability.
## Repository Structure

- `mcp_servers/`
  - `fraud_model_mcp/`
  - `eda_mcp/`
- `agents/`
  - `orchestrator.py`
  - `data_analyst_agent.py`
  - `decision_agent.py`
  - `modelling_agent.py`
  - `report_writer_agent.py`
- `core/`
  - `state.py`
  - `db.py`
  - `approval.py`
- `dashboard/app.py`
- `db/schema.sql`
- `docker-compose.yml`
- `requirements.txt`

## Deployment checklist

1. Copy `.env.example` to `.env` and fill in your runtime values.
2. Start Postgres with Docker Compose:
   - `docker compose up -d postgres`
3. Run the orchestrator:
   - `python -m agents.orchestrator --task "Detect fraud in payment data" --dataset sample_dataset.csv`
4. Start the dashboard:
   - `streamlit run dashboard/app.py`

## Current status

The workflow is wired end to end with LangGraph, MCP tools, Postgres-backed run logging, and a runnable orchestrator. The remaining deployment work is environment setup and service startup rather than architecture changes.
