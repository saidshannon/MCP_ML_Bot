from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    run_id: Optional[int] = Field(None, description="Primary key for the run")
    task_description: str = Field(..., description="The user-facing task or objective")
    dataset_path: Optional[str] = Field(None, description="Path to the dataset used for analysis")
    eda_findings: Optional[Dict[str, Any]] = Field(default_factory=dict)
    chosen_approach: Optional[str] = Field(None, description="Selected solution: prompt engineering, trained model, or fine-tuning")
    model_results: Optional[Dict[str, Any]] = Field(default_factory=dict)
    report: Optional[str] = Field(None, description="Final synthesized markdown report")
    errors: Optional[List[str]] = Field(default_factory=list)
    retry_count: int = Field(0, ge=0)

    class Config:
        extra = "forbid"
        validate_assignment = True
