def write_user(log: RichLog, text: str) -> None:
        """
        Helper function for messages by the user

        Args:
            log: log object to print to
            text: text to print
        """

        log.write("\n[bold #7aa2f7]You[/bold #7aa2f7]")
        log.write(f"[#c0caf5]{text}[/]\n")
        log.scroll_end(animate=False)

def write_system(log: RichLog, text: str):
    """
    Helper function for system messages

    Args:
        log: log object to print to
        text: text to print
    """

    log.write(f"[dim]{text}[/dim]\n")
    log.scroll_end(animate=False)

def write_assistant(log: RichLog, text: str):
    """
    Helper function for messages by the llm ("assistant")

    Args:
        log: log object to print to
        text: text to print
    """

    log.write(f"\n[bold #9ece6a]{self.TITLE}[/bold #9ece6a]")
    log.write(f"[#c0caf5]{text}[/]\n")
    log.scroll_end(animate=False)