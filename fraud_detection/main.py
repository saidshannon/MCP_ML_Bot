"""Stub fraud API — replace internals with real model later."""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
import subprocess

from contextlib import asynccontextmanager
from pathlib import Path
from core.db import create_schema

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs on startup
    try:
        create_schema(Path("db/schema.sql"))
        print("Schema initialised successfully.")
    except Exception as e:
        print(f"Schema init warning: {e}")
    yield
    # Runs on shutdown (nothing needed here)

app = FastAPI(lifespan=lifespan)

class TrainRequest(BaseModel):
    dataset_path: str
    model_type: str = "xgboost"
    hyperparams: Dict[str, Any] = {}

class PredictRequest(BaseModel):
    features: Dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
def train(request: TrainRequest):
    return {
        "status": "trained",
        "model_type": request.model_type,
        "metrics": {"accuracy": 0.92, "f1_score": 0.88}
    }

@app.post("/predict")
def predict(request: PredictRequest):
    return {"status": "ok", "fraud_score": 0.76}

@app.get("/feature-importance")
def feature_importance(model_id: str = "default"):
    return {
        "model_id": model_id,
        "feature_importance": [
            {"feature": "amount", "importance": 0.42},
            {"feature": "merchant_country", "importance": 0.21},
        ]
    }

# fraud_detection/main.py
import asyncio
from fastapi import BackgroundTasks

@app.post("/trigger-run")
async def trigger_run(task: str, background_tasks: BackgroundTasks, dataset: str = "data/sample.csv"):
    """Trigger an orchestrator run as a background task."""
    background_tasks.add_task(run_pipeline, task, dataset)
    print("Pipeline started")
    return {"status": "started", "task": task}

async def run_pipeline(task: str, dataset: str):
    """Actually runs the pipeline in the background."""
    try:
        print("Inside run pipeline")
        import sys
        import os
        sys.path.insert(0, "/app")  # ensure project root is on path
        from agents.orchestrator import build_and_run
        await build_and_run(task_description=task, dataset_path=dataset)
    except Exception as e:
        print(f"Pipeline run failed: {e}")