"""Data analyst agent for EDA summarization and findings extraction."""

import json
import os
import time
from typing import Any, Dict, List

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ValidationError

from core.db import get_pg_connection
from core.state import AgentState

class EDAFindings(BaseModel):
    quality_ok: bool = Field(..., description = "Whether Dataset passes quality checks")
    issues: List[Dict[str,Any]] = Field(default_factory = list)
    stats: Dict[str,Any] = Field(default_factory = dict)

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


async def run_data_analyst(state: AgentState, tools: list) -> AgentState:
        system_prompt = (
        "You are a data analyst. Use the provided MCP tools to inspect the dataset only. "
        "Do not perform modelling or reporting. "
        "You must autonomously decide when to call profile_dataset and check_data_quality. "
        "Only return strict JSON matching the below schema with no other formatting: "
        """{"quality_ok": bool, "issues": [{"issue": str, "severity": str, "details": str}],
        "stats": {"row_count": int, "class_balance": dict, "missing_pct": float}}"""
    )
        user_prompt = (f"Analyze the dataset at {state.dataset_path} for the task: "
                       f"{state.task_description}. Use the MCP Tools to gather profiling and quality findings")
        
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

        attempts = 0
        last_error = "No validation yet"

        while attempts <= 0:
            start_time = time.time()
            try:
                response = await agent.ainvoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=user_prompt if attempts == 0 else f"{user_prompt}\n\nPrevious attempt failed with validation error: {last_error}. Fix the JSON and return the schema exactly."
                            )
                        ]
                    }
                )

                messages = response.get("messages", []) if isinstance(response, dict) else []
                content = messages[-1].content if messages else ""
                if isinstance(content, list):
                    content = "\n".join(str(item) for item in content)

                print("###############################")
                print(content)

                findings = EDAFindings.model_validate(json.loads(content))

                state.eda_findings = {
                    "quality_ok": findings.quality_ok,
                    "issues": findings.issues,
                    "stats": findings.stats,
                }

                output_payload = {
                    "findings": state.eda_findings,
                    "raw_response": content,
                }

                _log_agent_call(
                    state,
                    "data_analyst",
                    {"task": state.task_description, "dataset_path": state.dataset_path},
                    output_payload,
                    int((time.time() - start_time) * 1000),
                    True,
                    None,
                )

                return state

            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                last_error = str(exc)
                output_payload = {"error": last_error}
                _log_agent_call(
                    state,
                    "data_analyst",
                    {"task": state.task_description, "dataset_path": state.dataset_path},
                    output_payload,
                    int((time.time() - start_time) * 1000),
                    False,
                    last_error,
                )

                state.errors.append(f"EDA validation failed: {last_error}")
                attempts += 1

        state.eda_findings = {
            "quality_ok": False,
            "issues": [{"issue": "EDA validation failed", "severity": "high", "details": last_error or "Unknown validation error"}],
            "stats": {},
        }
        return state