import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from mcp import ClientSession

from config import MAX_STEPS
from custom_types import CommandHistory, OllamaTool
from llm.llm import LLMClient

LogCallback = Callable[[str, str], Awaitable[None]]


class Agent:
    """Runs the tool-using conversation loop, independent of the UI."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: dict[str, ClientSession],
        tools: list[OllamaTool],
        debug: bool = False,
    ):
        self.llm = llm_client
        self.tool_registry = tool_registry
        self.tools = tools
        self.debug = debug
        self.history: list[CommandHistory] = []

    async def turn(
        self,
        user_message: str,
        log_callback: LogCallback,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Process one user message through the full agent loop."""
        self.history.append({"role": "user", "content": user_message})
        if self.debug:
            await log_callback("system", "Starting agent turn…")

        for _ in range(MAX_STEPS):
            if stop_event is not None and stop_event.is_set():
                break

            msg = await self.llm.chat(self.history, self.tools)
            msg_dict = {"role": msg.role, "content": msg.content}
            self.history.append(msg_dict)

            if msg.content:
                await log_callback("assistant", msg.content)

            if not msg.tool_calls:
                break

            results = await asyncio.gather(
                *(self._execute_tool(call, log_callback) for call in msg.tool_calls)
            )
            self.history.extend(results)
            await log_callback("system", "All tool calls completed")

    def clear_history(self) -> None:
        self.history.clear()

    async def _execute_tool(
        self, call: dict[str, Any], log_callback: LogCallback
    ) -> CommandHistory:
        """Execute a tool call and return the corresponding tool-result message."""
        tool_call_id = call["id"]
        name = call["function"]["name"]
        args = self._parse_tool_arguments(call["function"].get("arguments", {}))

        await log_callback("system", f"Using tool: {name} ({json.dumps(args)})")

        session = self.tool_registry.get(name)
        if session is None:
            await log_callback("system", f"{name} → unknown tool, skipping")
            return self._tool_result(tool_call_id, f"Error: unknown tool '{name}'")

        try:
            result = await session.call_tool(name, args)
            result_text = (
                result.content[0].text
                if result.content and hasattr(result.content[0], "text")
                else str(result.content)
            )
            if self.debug:
                await log_callback("system", f"{name} → {result_text}")
            return self._tool_result(tool_call_id, result_text)
        except Exception as e:
            await log_callback("system", f"{name} failed: {e}")
            return self._tool_result(tool_call_id, f"Error: {e}")

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> Any:
        if not isinstance(arguments, str):
            return arguments
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments

    @staticmethod
    def _tool_result(tool_call_id: str, content: str) -> CommandHistory:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}