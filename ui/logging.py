from textual.widgets import RichLog


def write_user(log: RichLog, text: str) -> None:
    log.write("\n[bold #7aa2f7]You[/bold #7aa2f7]")
    log.write(f"[#c0caf5]{text}[/]\n")
    log.scroll_end(animate=False)


def write_system(log: RichLog, text: str) -> None:
    log.write(f"[dim]{text}[/dim]\n")
    log.scroll_end(animate=False)


def write_assistant(log: RichLog, text: str) -> None:
    log.write("\n[bold #9ece6a]TINYCLAW[/bold #9ece6a]")
    log.write(f"[#c0caf5]{text}[/]\n")
    log.scroll_end(animate=False)
