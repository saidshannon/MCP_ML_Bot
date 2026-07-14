"""EDA MCP server package implementation."""

from typing import Any, Dict, List


class EdaMCPServer:
    def profile_dataset(self, path: str) -> Dict[str, Any]:
        return {
            "path": path,
            "missing_values": 0.12,
            "dtypes": {
                "amount": "float64",
                "merchant_country": "object",
                "is_fraud": "int64",
            },
            "class_balance": {
                "fraud": 0.03,
                "non_fraud": 0.97,
            },
            "basic_stats": {
                "rows": 10000,
                "columns": 5,
            },
        }

    def check_data_quality(self, path: str) -> List[Dict[str, Any]]:
        return [
            {
                "issue": "High class imbalance",
                "severity": "medium",
                "details": "Fraud transactions account for only 3% of rows.",
            },
            {
                "issue": "Missing values detected",
                "severity": "low",
                "details": "A small percentage of values are missing in one feature.",
            },
        ]
