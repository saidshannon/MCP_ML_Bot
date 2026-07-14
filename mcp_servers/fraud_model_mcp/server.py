from typing import Any, Dict

from mcp.server.fastmcp import FastMCP
from . import FraudModelMCPServer

mcp = FastMCP()
_impl = FraudModelMCPServer()


@mcp.tool(name="train_model")
def train_model(dataset_path: str, model_type: str, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """Train a fraud detection model and return training metrics."""
    return _impl.train_model(dataset_path, model_type, hyperparams)


@mcp.tool(name="predict")
def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    """Predict a fraud score from a feature set."""
    return _impl.predict(features)


@mcp.tool(name="get_feature_importance")
def get_feature_importance(model_id: str) -> Dict[str, Any]:
    """Return ranked feature importance for a trained model."""
    return _impl.get_feature_importance(model_id)


if __name__ == "__main__":
    mcp.run()
