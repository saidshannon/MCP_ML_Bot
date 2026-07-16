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

import asyncio
from fastapi import BackgroundTasks

@app.post("/trigger-run")
async def trigger_run(task: str, background_tasks: BackgroundTasks, dataset: str = "data/sample.csv"):
    background_tasks.add_task(run_pipeline_sync, task, dataset)
    print(f"Pipeline triggered for task: {task}")
    return {"status": "started", "task": task}

def run_pipeline_sync(task: str, dataset: str):
    """
    Sync wrapper required because FastAPI BackgroundTasks runs functions
    in a thread pool — async functions don't work correctly there.
    We create a new event loop for this thread and run the async pipeline in it.
    """
    print(f"run_pipeline_sync called for: {task}")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_run_pipeline_async(task, dataset))
    except Exception as e:
        print(f"Pipeline sync wrapper failed: {e}")
    finally:
        loop.close()

async def _run_pipeline_async(task: str, dataset: str):
    """Actual async pipeline execution."""
    try:
        print(f"Starting pipeline: {task}")
        import sys
        sys.path.insert(0, "/app")
        from agents.orchestrator import build_and_run
        await build_and_run(task_description=task, dataset_path=dataset)
        print(f"Pipeline completed: {task}")
    except Exception as e:
        print(f"Pipeline failed: {e}", flush=True)
        import traceback
        traceback.print_exc()