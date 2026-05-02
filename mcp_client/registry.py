import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from auth.oauth import resolve_credentials
from config import SERVER_SCRIPT
from custom_types import McpConfig, OllamaTool


@dataclass
class ToolRegistry:
    """Maps tool names to the MCP session that can execute them."""

    sessions_by_tool: dict[str, ClientSession] = field(default_factory=dict)
    tools: list[OllamaTool] = field(default_factory=list)

    def register(self, session: ClientSession, tools_response) -> None:
        for tool in tools_response.tools:
            if tool.name in self.sessions_by_tool:
                print(
                    f"[TinyClaw] Warning: tool '{tool.name}' already registered, overwriting."
                )
            self.sessions_by_tool[tool.name] = session
            self.tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                }
            )


async def connect_mcp_servers(config: McpConfig, stack: AsyncExitStack) -> ToolRegistry:
    """Connect to local plugins and configured external MCP servers."""
    registry = ToolRegistry()

    await _connect_local_server(stack, registry)

    for name, service in config.get("mcpServers", {}).items():
        try:
            resolved_env = resolve_credentials(name, service.get("env", {}))
            params = StdioServerParameters(
                command=service["command"],
                args=service["args"],
                env={**os.environ, **resolved_env},
            )
            await _connect_server(stack, params, registry)
        except Exception as exc:
            print(f"[TinyClaw] Warning: failed to connect to '{name}': {exc}")

    return registry


async def _connect_local_server(stack: AsyncExitStack, registry: ToolRegistry) -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER_SCRIPT)])
    await _connect_server(stack, params, registry)


async def _connect_server(
    stack: AsyncExitStack,
    params: StdioServerParameters,
    registry: ToolRegistry,
) -> None:
    read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
    session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
    await session.initialize()
    registry.register(session, await session.list_tools())
