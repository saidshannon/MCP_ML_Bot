from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from core.db import get_pg_connection
from core.state import AgentState


class DecisionRecommendation(BaseModel):
    approach: str = Field(..., description="One of: prompt_engineering, trained_model, or fine_tuning")
    rationale: str = Field(..., description="Reasoning for the selected approach")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")


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


async def run_decision_agent(state: AgentState) -> AgentState:
    system_prompt = (
        "Given this task and EDA findings, recommend one of: "
        "'prompt_engineering', 'trained_model', or 'fine_tuning'. "
        "Return only JSON with keys: approach, rationale, confidence (0-1), no other formatting."
    )

    task_context = {
        "task": state.task_description,
        "eda_findings": state.eda_findings or {},
    }

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

    print("#TASK")
    print(task_context['task'])
    print("EDA FINDINGS")
    print(task_context['eda_findings'])

    try:
        result = await agent.ainvoke(
            {
                "messages":[
                HumanMessage(
                    content=f"{system_prompt}\n\nTask: {task_context['task']}\n\nEDA findings: {json.dumps(task_context['eda_findings'], default=str)}"
                )
            ]
            }

        )
        print("#########DECISION AGENT########")
        print(result)
        messages = result.get("messages", []) if isinstance(result, dict) else []
        content = messages[-1].content if messages else ""
        recommendation = DecisionRecommendation.model_validate(json.loads(content))

        latency_ms = int((time.time() - start_time) * 1000)

        state.chosen_approach = recommendation.approach
        output_payload = {
            "chosen_approach": state.chosen_approach,
            "rationale": recommendation.rationale,
            "confidence": recommendation.confidence,
        }

        _log_agent_call(
            state,
            "decision_agent",
            {"task": state.task_description, "eda_findings": state.eda_findings or {}},
            output_payload,
            latency_ms,
            True,
            None,
        )
    except Exception as exc:
        latency_ms = int((time.time() - start_time) * 1000)
        state.chosen_approach = "prompt_engineering"
        state.errors.append(f"Decision agent failed: {exc}")
        output_payload = {"chosen_approach": state.chosen_approach, "error": str(exc)}
        _log_agent_call(
            state,
            "decision_agent",
            {"task": state.task_description, "eda_findings": state.eda_findings or {}},
            output_payload,
            latency_ms,
            False,
            str(exc),
        )

    return state
