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

@app.post("/trigger-run")
async def trigger_run(task: str, dataset: str = "data/sample.csv"):
    subprocess.Popen([
        "python", "-m", "agents.orchestrator",
        "--task", task,
        "--dataset", dataset
    ])
    return {"status": "started", "task": task}