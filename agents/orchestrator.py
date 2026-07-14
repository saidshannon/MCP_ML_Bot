"""LangGraph-based multi-agent orchestrator for the ML ops workflow."""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, StateGraph

from agents.data_analyst_agent import run_data_analyst
from agents.decision_agent import run_decision_agent
from agents.modelling_agent import run_modelling_agent
from agents.report_writer_agent import run_report_writer
from core.db import create_schema, get_pg_connection
from core.state import AgentState


MCP_SERVERS = {
    "fraud_model": {
        "command": "python",
        "args": ["-m", "mcp_servers.fraud_model_mcp.server"],
        "transport": "stdio",
    },
    "eda": {
        "command": "python",
        "args": ["-m", "mcp_servers.eda_mcp.server"],
        "transport": "stdio",
    },
}


def route_after_eda(state: AgentState) -> str:
    eda_findings = state.eda_findings or {}
    if not eda_findings or "quality_ok" not in eda_findings:
        return "end"

    issues = eda_findings.get("issues", [])
    if eda_findings.get("quality_ok") is False and any(issue.get("severity") == "high" for issue in issues):
        state.errors.append("High-severity data quality issue detected")
        return "end"

    return "decision"


async def approval_gate(state: AgentState) -> AgentState:
    print("\n" + "=" * 60)
    print("APPROVAL GATE")
    print("=" * 60)
    print(f"Dataset path: {state.dataset_path}")
    print(f"Chosen approach: {state.chosen_approach}")
    print(f"EDA issue count: {len((state.eda_findings or {}).get('issues', []))}")
    print(f"Run ID: {state.run_id}")

    auto_approve = os.getenv("AUTO_APPROVE", "").strip().lower() in {"1", "true", "yes", "y"}
    if auto_approve:
        return state

    answer = input("Approve training run? (yes/no): ").strip().lower()
    if answer == "yes":
        return state

    state.errors.append("Training rejected by human reviewer")
    raise ValueError("Pipeline halted by human reviewer")


async def log_run_start(state: AgentState) -> None:
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (task_description, status, started_at)
                VALUES (%s, %s, NOW())
                RETURNING run_id
                """,
                (state.task_description, "started"),
            )
            row = cur.fetchone()
            if row:
                state.run_id = row[0]
            conn.commit()


async def log_run_end(state: AgentState, success: bool) -> None:
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE runs
                SET status = %s, finished_at = NOW()
                WHERE run_id = %s
                """,
                ("completed" if success else "failed", state.run_id),
            )
            conn.commit()


async def build_and_run(task_description: str, dataset_path: str) -> AgentState:
    initial_state = AgentState(
        task_description=task_description,
        dataset_path=dataset_path,
        eda_findings={},
        chosen_approach="",
        model_results={},
        report="",
        errors=[],
        retry_count=0,
        run_id=None,
    )

    client=MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    print(f"Tools discovered: {[t.name for t in tools]}")

    async def eda_node(state: AgentState) -> AgentState:
        return await run_data_analyst(state, tools)

    async def decision_node(state: AgentState) -> AgentState:
        return await run_decision_agent(state)

    async def gate_node(state: AgentState) -> AgentState:
        return await approval_gate(state)

    async def modelling_node(state: AgentState) -> AgentState:
        return await run_modelling_agent(state, tools)

    async def report_node(state: AgentState) -> AgentState:
        return await run_report_writer(state)

    graph = StateGraph(AgentState)
    graph.add_node("data_analyst", eda_node)
    graph.add_node("decision", decision_node)
    graph.add_node("approval_gate", gate_node)
    graph.add_node("modelling", modelling_node)
    graph.add_node("report_writer", report_node)

    graph.set_entry_point("data_analyst")
    graph.add_conditional_edges("data_analyst", route_after_eda, {"decision": "decision", "end": END})
    graph.add_edge("decision", "approval_gate")
    graph.add_edge("approval_gate", "modelling")
    graph.add_edge("modelling", "report_writer")
    graph.add_edge("report_writer", END)

    pipeline = graph.compile()

    #Initialising DB Tables
    from pathlib import Path
    create_schema(Path(__file__).parent.parent / "db" / "schema.sql")
    await log_run_start(initial_state)

    try:
        final_state = await pipeline.ainvoke(initial_state)
        if isinstance(final_state, dict):
            final_state = AgentState.model_validate(final_state)
        await log_run_end(final_state, success=True)
        return final_state
    except Exception as exc:
        initial_state.errors.append(str(exc))
        await log_run_end(initial_state, success=False)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Task description in plain English")
    parser.add_argument("--dataset", required=True, help="Path to dataset CSV")
    args = parser.parse_args()

    result = asyncio.run(build_and_run(args.task, args.dataset))
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(result.report)
    if result.errors:
        print("\nErrors encountered:")
        for err in result.errors:
            print(f"  - {err}")
