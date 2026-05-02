from enum import Enum
from typing import Any, NotRequired, TypedDict


class FunctionSpec(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class OllamaTool(TypedDict):
    type: str
    function: FunctionSpec


class CommandHistory(TypedDict):
    role: str
    content: str
    tool_call_id: NotRequired[str]


class Mode(str, Enum):
    NORMAL = "normal"
    INSERT = "insert"
    TOOLS = "tools"


class McpServiceConfig(TypedDict):
    command: str
    args: list[str]
    env: NotRequired[dict[str, str]]


class McpConfig(TypedDict):
    mcpServers: dict[str, McpServiceConfig]
    tinyclaw: NotRequired[dict[str, Any]]
