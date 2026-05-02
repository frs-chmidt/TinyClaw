import importlib.util
import logging
import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("agent-tools", log_level="WARNING")

PLUGINS_DIR = Path(__file__).parent / "plugins"

logger = logging.getLogger("mcp-server")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("[mcp-server] %(levelname)s: %(message)s"))

logger.handlers.clear()
logger.addHandler(handler)
logger.propagate = False


def load_plugins() -> None:
    """Dynamically load and register all plugins from the plugins directory."""

    if not PLUGINS_DIR.exists():
        logger.info("Plugins directory not found, creating it...")
        PLUGINS_DIR.mkdir()
        return

    for plugin_file in sorted(PLUGINS_DIR.glob("*.py")):
        if plugin_file.name.startswith("_"):
            continue  # skip __init__.py etc.

        module_name = f"plugins.{plugin_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, plugin_file)

        if spec is None:
            logger.info("Could not load spec from file. Skipping plugin: %s", plugin_file)
            continue

        module = importlib.util.module_from_spec(spec)

        # Inject the mcp instance so plugins can register tools
        module.mcp = mcp  # pyright: ignore[reportAttributeAccessIssue]

        try:
            if spec.loader is None:
                logger.info("Loader of spec is null. Skipping plugin: %s", plugin_file)
                continue

            spec.loader.exec_module(module)
            logger.info("Loaded plugin: %s", plugin_file.name)
        except Exception as exc:
            logger.info("Failed to load %s: %s", plugin_file.name, exc)


load_plugins()

if __name__ == "__main__":
    mcp.run()
