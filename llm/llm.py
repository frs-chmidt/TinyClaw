# llm.py
import ollama
from typing import Any, Optional

class LLMClient:
    """Wraps the LLM backend (Ollama) so you can swap models or providers later."""

    def __init__(self, model: str = "qwen4b:4b", system_prompt: str = ""):
        self.model = model
        self.system_prompt = system_prompt
        self._client = ollama.AsyncClient()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict]] = None,
    ) -> ollama.Message:
        """Send a chat request and return the assistant message."""
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages
        response = await self._client.chat(
            model=self.model,
            messages=full_messages,
            tools=tools or None,
        )
        return response.message