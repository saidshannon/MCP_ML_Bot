from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP
from . import EdaMCPServer

mcp = FastMCP()
_impl = EdaMCPServer()


@mcp.tool(name="profile_dataset")
def profile_dataset(path: str) -> Dict[str, Any]:
    """Produce dataset profiling results including missing values, dtypes, class balance, and basic stats."""
    return _impl.profile_dataset(path)


@mcp.tool(name="check_data_quality")
def check_data_quality(path: str) -> List[Dict[str, Any]]:
    """Run data quality checks and return flagged issues."""
    return _impl.check_data_quality(path)


if __name__ == "__main__":
    mcp.run()
