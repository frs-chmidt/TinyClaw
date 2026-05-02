import json
from argparse import Namespace
from typing import override

from mcp import ClientSession
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Label, RichLog
from typing_extensions import final

from agent.agent import Agent
from config import ASCII_LOGO, MODEL, PROVIDER, SYSTEM_PROMPT
from custom_types import Mode, OllamaTool
from llm.llm import LLMClient
from ui.logging import write_assistant, write_system, write_user


@final
class ChatApp(App):
    """Textual chat UI for TinyClaw."""

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
    SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(
        self,
        tool_registry: dict[str, ClientSession],
        tools: list[OllamaTool],
        args: Namespace,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        llm_client = LLMClient(
            provider=PROVIDER,
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
        )
        self.agent = Agent(
            llm_client=llm_client,
            tool_registry=tool_registry,
            tools=tools,
            debug=args.debug,
        )
        self.tools = tools
        self.mode = Mode.NORMAL
        self.debug_active = args.debug
        self.loading = False
        self.spinner_frame = 0
        self.spinner_task = None

    @override
    def compose(self) -> ComposeResult:
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

    def on_mount(self) -> None:
        log = self.query_one("#log", RichLog)
        self.query_one("#tools", RichLog).display = False
        self.query_one("#loadingStatus", Label).display = False

        write_system(log, ASCII_LOGO)
        if self.debug_active:
            write_system(log, "Debug mode is active. Expect detailed logs.")

        tool_names = [tool["function"]["name"] for tool in self.tools]
        if tool_names:
            write_system(log, f"Successfully loaded tools: {', '.join(tool_names)}")
        else:
            write_system(log, "No tools loaded (add .py files to plugins/)")
        write_system(log, f"{self.TITLE} is ready for you! Press 'i' to interact.")
        self._update_status()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.mode != Mode.INSERT:
            return

        text = event.value.strip()
        if not text:
            return

        event.input.value = ""
        log = self.query_one("#log", RichLog)
        write_user(log, text)

        self.run_worker(
            self._agent_turn(text, log),
            exclusive=True,
            thread=False,
        )

    def action_enter_insert(self) -> None:
        self.mode = Mode.INSERT
        self._show_chat_log()

        input_field = self.query_one(Input)
        input_field.disabled = False
        input_field.focus()
        input_field.placeholder = "Type a message..."
        self._update_status()

    def action_enter_normal(self) -> None:
        self.mode = Mode.NORMAL
        self._show_chat_log()

        input_field = self.query_one(Input)
        input_field.disabled = True
        input_field.blur()
        self._update_status()

    def action_show_tools(self) -> None:
        self.mode = Mode.TOOLS
        tools_view = self.query_one("#tools", RichLog)
        self.query_one("#log", RichLog).display = False
        tools_view.display = True
        tools_view.clear()

        for tool in self.tools:
            function = tool["function"]
            tools_view.write(f"[bold #bb9af7]{function['name']}[/]")
            write_system(tools_view, function["description"])
            write_system(tools_view, json.dumps(function["parameters"], indent=2))

        self._update_status()

    def action_clear_chat(self) -> None:
        self.agent.clear_history()
        self.query_one("#log", RichLog).clear()

    async def _agent_turn(self, user_message: str, log: RichLog) -> None:
        self.action_enter_normal()
        self._start_loading()

        async def log_callback(role: str, text: str) -> None:
            if not text:
                return
            if role == "assistant":
                write_assistant(log, text)
            elif role == "user":
                write_user(log, text)
            else:
                write_system(log, text)

        await self.agent.turn(user_message, log_callback)
        self._stop_loading()

    def _show_chat_log(self) -> None:
        self.query_one("#tools", RichLog).display = False
        self.query_one("#log", RichLog).display = True

    def _update_status(self) -> None:
        labels = {
            Mode.NORMAL: "[bold yellow]NORMAL[/]",
            Mode.INSERT: "[bold green]INSERT[/]",
            Mode.TOOLS: "[bold magenta]TOOLS[/]",
        }
        self.query_one("#status", Label).update(labels[self.mode])

    def _start_loading(self) -> None:
        self.loading = True
        self.spinner_frame = 0

        def tick() -> None:
            if not self.loading:
                return
            label = self.query_one("#loadingStatus", Label)
            label.display = True
            frame = self.SPINNER[self.spinner_frame % len(self.SPINNER)]
            label.update(f"[bold cyan]{frame} Thinking...[/]")
            self.spinner_frame += 1

        self.spinner_task = self.set_interval(0.1, tick)

    def _stop_loading(self) -> None:
        self.loading = False
        if self.spinner_task:
            self.spinner_task.stop()
            self.query_one("#loadingStatus", Label).display = False
            self.spinner_task = None
        self._update_status()
