# agent.py
import asyncio
import json
from typing import Callable, Awaitable

from mcp import ClientSession
from llm.llm import LLMClient
from custom_types import CommandHistory, OllamaTool  # adjust imports as needed

from config import MAX_STEPS

class Agent:
    """Runs the tool‑using conversation loop, independent of the UI."""

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
        log_callback: Callable[[str, str], Awaitable[None]],
        stop_event: asyncio.Event | None = None,          # ← new parameter
    ) -> None:
        """
        Process one user message through the full agent loop.
        The loop checks `stop_event` before each new LLM call.
        """
        self.history.append({"role": "user", "content": user_message})
        if self.debug:
            await log_callback("system", "Starting agent turn…")

        for step in range(MAX_STEPS):
            # Co‑operative cancellation – exit the loop immediately
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
                *[self._execute_tool(call, log_callback) for call in msg.tool_calls]
            )
            self.history.extend(results)
            await log_callback("system", "All tool calls completed")



    async def _execute_tool(self, call: dict, log_callback) -> CommandHistory:
        """
        Execute a tool call and return a message dict that includes `tool_call_id`.
        
        Args:
            call: A tool call dict like:
                {
                    'type': 'function',
                    'function': {'name': 'calculate', 'arguments': '{"expression": "100 * 5"}'},
                    'id': 'chatcmpl-tool-a7ffe499f695f251'
                }
        """
        tool_call_id = call['id']          # extract the ID
        name = call['function']['name']
        args_str = call['function']['arguments']

        # Parse arguments (they may be a JSON string)
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = args_str

        await log_callback("system", f"Using tool: {name} ({json.dumps(args)})")

        session = self.tool_registry.get(name)
        if session is None:
            await log_callback("system", f"{name} → unknown tool, skipping")
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": f"Error: unknown tool '{name}'"
            }

        try:
            result = await session.call_tool(name, args)
            result_text = (
                result.content[0].text
                if result.content and hasattr(result.content[0], "text")
                else str(result.content)
            )
            if self.debug:
                await log_callback("system", f"{name} → {result_text}")
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_text
            }
        except Exception as e:
            await log_callback("system", f"{name} failed: {e}")
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": f"Error: {e}"
            }