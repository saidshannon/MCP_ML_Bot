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


def _log_agent_call(state: AgentState, agent_name: str, payload: Dict[str, Any], output: Dict[str, Any], duration_ms: int, success: bool, retry_count: int, error_message: str | None = None) -> None:
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


async def run_modelling_agent(state: AgentState, tools: list) -> AgentState:
    system_prompt = (
        "You are a modelling agent. Use the provided MCP tools to train and evaluate a model. "
        "Use the dataset context and EDA findings to decide the best hyperparameters and training strategy. "
        "You must autonomously decide when to call train_model. "
        # "When a call fails, explain what you changed and why in the next attempt. "
        f"Current EDA findings: {json.dumps(state.eda_findings or {}, default=str)}. "
        f"Current chosen approach: {state.chosen_approach}."
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
    system_prompt=system_prompt,
    tools=tools
    )

    for attempt in range(2):
        start_time = time.time()
        try:
            user_prompt = (
                f"Train a model for the task: {state.task_description}. "
                "Use the MCP tool to train the model and choose hyperparameters autonomously."
                if attempt == 0
                else (
                    "The previous attempt failed. Explain what you changed and why, "
                    "then call the training tool again with a revised strategy."
                )
            )

            response = await agent.ainvoke(
                {
                    "messages": [HumanMessage(content=user_prompt)],
                }
            )

            content = response["messages"][-1].content if response.get("messages") else ""
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)
            print("########MODEL_AGENT###############")
            print(response)
            print(response["messages"][-1].content)
            state.model_results = {
                "attempt": attempt + 1,
                "raw_response": content,
            }
            state.retry_count = attempt

            _log_agent_call(
                state,
                "modelling_agent",
                {"task": state.task_description, "attempt": attempt + 1},
                state.model_results,
                int((time.time() - start_time) * 1000),
                True,
                attempt,
                None,
            )
            return state

        except Exception as exc:
            state.retry_count = attempt + 1
            state.errors.append(f"Modelling attempt {attempt + 1} failed: {exc}")

            _log_agent_call(
                state,
                "modelling_agent",
                {"task": state.task_description, "attempt": attempt + 1},
                {"error": str(exc)},
                int((time.time() - start_time) * 1000),
                False,
                attempt + 1,
                str(exc),
            )

    return state