# agent.py
import asyncio
import json
from typing import Callable, Awaitable

from mcp import ClientSession
from llm.llm import LLMClient
from custom_types import CommandHistory, OllamaTool  # adjust imports as needed

MAX_STEPS = 8

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
        log_callback: Callable[[str, str], Awaitable[None]],  # e.g., async def log(role, text)
    ) -> None:
        """
        Process one user message through the full agent loop.
        All output (assistant replies, tool calls, debug info) goes via `log_callback`.
        """
        self.history.append({"role": "user", "content": user_message})
        await log_callback("system", "Starting agent turn…" if self.debug else "")

        for step in range(MAX_STEPS):
            msg = await self.llm.chat(self.history, self.tools)
            # Convert LLMMessage to dict for history
            msg_dict = {"role": msg.role, "content": msg.content}
            self.history.append(msg_dict)

            if msg.content:
                await log_callback("assistant", msg.content)

            if not msg.tool_calls:
                break

            # Execute all tool calls concurrently
            results = await asyncio.gather(
                *[self._execute_tool(call, log_callback) for call in msg.tool_calls]
            )
            self.history.extend(results)
            await log_callback("system", "All tool calls completed")

            if self.debug and step == MAX_STEPS - 1:
                await log_callback("system", "Max steps reached. Stopping.")

    async def _execute_tool(self, call, log_callback) -> CommandHistory:
        name = call.function.name
        args = call.function.arguments

        await log_callback("system", f"Using tool: {name} ({json.dumps(args)})")

        session = self.tool_registry.get(name)
        if session is None:
            await log_callback("system", f"{name} → unknown tool, skipping")
            return {"role": "tool", "content": f"Error: unknown tool '{name}'"}

        try:
            result = await session.call_tool(name, args)
            result_text = (
                result.content[0].text
                if result.content and hasattr(result.content[0], "text")
                else str(result.content)
            )
            if self.debug:
                await log_callback("system", f"{name} → {result_text}")
            return {"role": "tool", "content": result_text}
        except Exception as e:
            await log_callback("system", f"{name} failed: {e}")
            return {"role": "tool", "content": f"Error: {e}"}