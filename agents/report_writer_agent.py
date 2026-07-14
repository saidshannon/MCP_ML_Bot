from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from core.db import get_pg_connection
from core.state import AgentState


def _log_agent_call(state: AgentState, agent_name: str, payload: Dict[str, Any], output: Dict[str, Any], duration_ms: int, success: bool, error_message: str | None = None) -> None:
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_calls (run_id, agent_name, input, output, duration_ms, success, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    state.run_id,
                    agent_name,
                    json.dumps(payload),
                    json.dumps(output),
                    duration_ms,
                    success,
                    error_message,
                ),
            )
            conn.commit()


async def run_report_writer(state: AgentState) -> AgentState:
    system_prompt = (
        "You are a report writer. Write a polished markdown report based on the shared agent state. "
        "Cover four sections: data quality summary, chosen approach with rationale, model metrics, and estimated business impact. "
        "Derive the business impact from the actual model metrics in the provided state rather than inventing fixed numbers. "
        "Be concise and production-ready."
    )

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Export it before running the orchestrator.")

    agent = create_agent(
    model=ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=groq_api_key,
    ),
    system_prompt=system_prompt
    )

    start_time = time.time()
    try:
        result = await agent.ainvoke(
            {
                "messages":[
                HumanMessage(
                    content=f"{system_prompt}\n\nState:\n{json.dumps({'eda_findings': state.eda_findings, 'chosen_approach': state.chosen_approach, 'model_results': state.model_results}, default=str)}"
                )
            ]
            }
            
        )

        print("#########REPORT WRITER AGENT########")
        print(result)
        messages = result.get("messages", []) if isinstance(result, dict) else []
        content = messages[-1].content if messages else ""
        state.report = content

        _log_agent_call(
            state,
            "report_writer",
            {"task": state.task_description},
            {"report": content},
            int((time.time() - start_time) * 1000),
            True,
            None,
        )

    except Exception as exc:
        state.report = "# Report generation failed"
        state.errors.append(f"Report writing failed: {exc}")
        _log_agent_call(
            state,
            "report_writer",
            {"task": state.task_description},
            {"error": str(exc)},
            int((time.time() - start_time) * 1000),
            False,
            str(exc),
        )

    return state