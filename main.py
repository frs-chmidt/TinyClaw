import asyncio
import json
import os
import sys
from argparse import ArgumentParser, Namespace
from contextlib import AsyncExitStack
from pathlib import Path
from typing import override

from llm.llm import LLMClient
from agent.agent import Agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Label
from typing_extensions import final

from auth.oauth import resolve_credentials
from custom_types import CommandHistory, McpConfig, OllamaTool, Mode
from chat_interface.helpers import write_user, write_system, write_assistant    

from config import (
    ASCII_LOGO,
    MODEL,
    MCP_CONFIG_PATH,
    PROVIDER,
    SERVER_SCRIPT,
    SYSTEM_PROMPT,
    MAX_STEPS
)


def load_mcp_config(config_path: Path = MCP_CONFIG_PATH) -> McpConfig:
    if not config_path.exists():
        return {"mcpServers": {}}
    with open(config_path) as f:
        data = json.load(f)
    if "mcpServers" not in data:
        data["mcpServers"] = {}
    return data




@final
class ChatApp(App):
    """Minimal Textual chat app."""

    CSS_PATH = "app.css"

    TITLE = "TinyClaw"
    SUB_TITLE = "Your personal assistant"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("i", "enter_insert", "Insert mode"),
        ("escape", "enter_normal", "Normal mode"),
        ("t", "show_tools", "Show tools"),
        ("c", "clear_chat", "Clear"),
    ]

    mode: Mode
    debug_active: bool
    SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(
        self,
        tool_registry: dict[str, ClientSession],
        tools: list[OllamaTool],
        args: Namespace,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_registry = tool_registry
        self.tools = tools
        self.llm_client = LLMClient(
            provider=PROVIDER,
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
        )
        self.agent = Agent(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            tools=self.tools,
            debug=args.debug,
        )
        self.history: list[CommandHistory] = []
        self.mode = Mode.NORMAL
        self.debug_active = args.debug  # pyright: ignore[reportAny]

        self.loading = False
        self.spinner_frame = 0
        self.spinner_task = None

    

    @override
    def compose(self) -> ComposeResult:
        """

        Sets up the TUI Layout and all available Widgets

        Yields: TUI Layout

        """
        yield Header(show_clock=False, icon="")
        with Vertical():
            yield RichLog(id="log", markup=True, wrap=True)
            yield RichLog(id="tools", markup=True, wrap=True)
            yield Label("", id="loadingStatus")
            yield Input(placeholder="Type a message and press Enter…")
        with Horizontal(id="footer-outer"):
            yield Label("", id="status")
            with Horizontal(id="footer-inner"):
                yield Footer(show_command_palette=False)

    def update_status(self):
        """
        Updates the status bar with the current MODE
        """

        status = self.query_one("#status", Label)

        if self.mode == Mode.NORMAL:
            status.update("[bold yellow]NORMAL[/]")
        elif self.mode == Mode.INSERT:
            status.update("[bold green]INSERT[/]")
        elif self.mode == Mode.TOOLS:
            status.update("[bold magenta]TOOLS[/]")

    def on_mount(self) -> None:
        """
        On mount print the list of tools loaded from the MCP
        """

        log = self.query_one("#log", RichLog)

        tool_names = [t["function"]["name"] for t in self.tools] if self.tools else []

        tools_view = self.query_one("#tools", RichLog)
        tools_view.display = False

        loading_label = self.query_one("#loadingStatus", Label)
        loading_label.display = False

        write_system(log, ASCII_LOGO)

        if self.debug_active:
            write_system(log, "Debug mode is active. Expect detailed logs.")

        if tool_names:
            write_system(log, f"Succesfully loaded tools: {', '.join(tool_names)}")
        else:
            write_system(log, "No tools loaded (add .py files to plugins/)")

        write_system(log, f"{self.TITLE} is ready for you! Press 'i' to interact.")

        self.update_status()

    def start_loading(self):
        self.loading = True
        self.spinner_frame = 0

        def tick():
            if not self.loading:
                return

            label = self.query_one("#loadingStatus", Label)
            label.display = True
            frame = self.SPINNER[self.spinner_frame % len(self.SPINNER)]
            label.update(f"[bold cyan]{frame} Thinking...[/]")

            self.spinner_frame += 1

        self.spinner_task = self.set_interval(0.1, tick)

    def stop_loading(self):
        self.loading = False

        if self.spinner_task:
            self.spinner_task.stop()
            label = self.query_one("#loadingStatus", Label)
            label.display = False
            self.spinner_task = None

        self.update_status()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """
        Input handler. Handles input field submission. Runs agent with request in thread.

        Args:
            event: Input event by Textual
        """

        # Only allow typing while in insert mode!
        if self.mode != Mode.INSERT:
            return

        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        log = self.query_one("#log", RichLog)
        write_user(log, text)
        self.history.append({"role": "user", "content": text})

        # Run the agentic loop in a worker so the UI stays responsive
        self.run_worker(
            self._agent_turn(log),
            exclusive=True,  # makes it so that the previous request gets cancelled upon a new request!
            thread=False,
        )

    def action_enter_insert(self):
        """
        Action which gets called when the "enter_insert" event is triggered.
        Updates mode. Enables / Disables the required fields / widgets.
        """

        self.mode = Mode.INSERT

        tools_view = self.query_one("#tools")
        log = self.query_one("#log")

        tools_view.display = False
        log.display = True

        input_field = self.query_one(Input)
        input_field.disabled = False
        input_field.focus()
        input_field.placeholder = "Type a message..."

        self.update_status()

    def action_enter_normal(self):
        """
        Action which gets called when the "enter_normal" event is triggered.
        Updates mode. Enables / Disables the required fields / widgets.
        """

        self.mode = Mode.NORMAL

        tools_view = self.query_one("#tools")
        log = self.query_one("#log")

        tools_view.display = False
        log.display = True

        input_field = self.query_one(Input)
        input_field.disabled = True
        input_field.blur()

        self.update_status()

    def action_show_tools(self):
        """
        Action which gets called when the "show_tools" event is triggered.
        Updates mode. Enables / Disables the required fields / widgets.
        """

        self.mode = Mode.TOOLS

        tools_view = self.query_one("#tools", RichLog)
        log = self.query_one("#log")

        tools_view.display = True
        log.display = False

        tools_view.clear()

        for t in self.tools:
            fn = t["function"]
            tools_view.write(f"[bold #bb9af7]{fn['name']}[/]")
            write_system(tools_view, fn["description"])
            write_system(tools_view, json.dumps(fn["parameters"], indent=2))

        self.update_status()

    def action_clear_chat(self):
        """
        Action which gets called when the "clear_chat" event is triggered.
        Clears the history and the log.
        """

        self.history.clear()
        self.query_one("#log", RichLog).clear()



    async def _agent_turn(self, log: RichLog) -> None:
        """Now just delegates to the agent, converting RichLog writes to the callback."""
        self.action_enter_normal()
        self.start_loading()

        # Async helper to call the correct RichLog method
        async def log_callback(role: str, text: str):
            if not text:
                return
            if role == "assistant":
                write_assistant(log, text)
            elif role == "user":
                write_user(log, text)
            else:  # system / tool
                write_system(log, text)

        # The agent expects the latest user message already in history
        # (in on_input_submitted we already append to self.history, let's pass it)
        last_user_msg = self.history[-1]["content"]
        await self.agent.turn(last_user_msg, log_callback)

        self.stop_loading()

async def run(args: Namespace) -> None:
    """
    Loads mcp.json, resolves credentials for each service, connects to all MCP
    servers (local + external), aggregates their tools, and starts the TUI.
    """
    config = load_mcp_config()

    async with AsyncExitStack() as stack:
        tool_registry: dict[str, ClientSession] = {}
        all_tools: list[OllamaTool] = []

        def _register_tools(session: ClientSession, tools_response) -> None:
            for t in tools_response.tools:
                if t.name in tool_registry:
                    print(
                        f"[TinyClaw] Warning: tool '{t.name}' already registered, overwriting."
                    )
                tool_registry[t.name] = session
                all_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description or "",
                            "parameters": t.inputSchema,
                        },
                    }
                )

        # Always connect to the local plugin server first
        local_params = StdioServerParameters(
            command=sys.executable,
            args=[str(SERVER_SCRIPT)],
        )
        r, w = await stack.enter_async_context(stdio_client(local_params))
        local_session = await stack.enter_async_context(ClientSession(r, w))
        await local_session.initialize()
        _register_tools(local_session, await local_session.list_tools())

        # Connect to each external service from mcp.json
        for name, service in config.get("mcpServers", {}).items():
            try:
                env_template = service.get("env", {})
                resolved_env = resolve_credentials(name, env_template)
                merged_env = {**os.environ, **resolved_env}

                ext_params = StdioServerParameters(
                    command=service["command"],
                    args=service["args"],
                    env=merged_env,
                )
                r, w = await stack.enter_async_context(stdio_client(ext_params))
                ext_session = await stack.enter_async_context(ClientSession(r, w))
                await ext_session.initialize()
                _register_tools(ext_session, await ext_session.list_tools())
            except Exception as e:
                print(f"[TinyClaw] Warning: failed to connect to '{name}': {e}")

        app = ChatApp(tool_registry=tool_registry, tools=all_tools, args=args)
        await app.run_async()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        dest="debug",
        default=False,
        help="Print additional information to the log.",
    )

    args = parser.parse_args()

    asyncio.run(run(args))
