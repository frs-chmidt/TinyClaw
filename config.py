
from pathlib import Path


SERVER_SCRIPT = Path(__file__).parent / "mcp_server.py"
MCP_CONFIG_PATH = Path(__file__).parent / "mcp.json"

MAX_STEPS = 8
SYSTEM_PROMPT = "You are a helpful assistant. Use tools when they help."
PROVIDER = "openrouter"  # or "ollama" or "google"
MODEL = "openai/gpt-oss-120b:free"  #"qwen3.5:4b"


ASCII_LOGO = """
████████╗██╗███╗   ██╗██╗   ██╗ ██████╗██╗      █████╗ ██╗    ██╗
╚══██╔══╝██║████╗  ██║╚██╗ ██╔╝██╔════╝██║     ██╔══██╗██║    ██║
   ██║   ██║██╔██╗ ██║ ╚████╔╝ ██║     ██║     ███████║██║ █╗ ██║
   ██║   ██║██║╚██╗██║  ╚██╔╝  ██║     ██║     ██╔══██║██║███╗██║
   ██║   ██║██║ ╚████║   ██║   ╚██████╗███████╗██║  ██║╚███╔███╔╝
   ╚═╝   ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ 
"""
