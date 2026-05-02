import json
from pathlib import Path

from config import MCP_CONFIG_PATH
from custom_types import McpConfig


def load_mcp_config(config_path: Path = MCP_CONFIG_PATH) -> McpConfig:
    """Load MCP server configuration, defaulting to no external servers."""
    if not config_path.exists():
        return {"mcpServers": {}}

    with config_path.open() as f:
        data = json.load(f)

    data.setdefault("mcpServers", {})
    return data
