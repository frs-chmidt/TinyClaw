# llm.py
import os
from typing import Any, Dict, List, Optional

import ollama
import openai
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present

class LLMMessage:
    def __init__(
        self,
        role: str,
        content: str = "",
        tool_calls: Optional[List[Dict]] = None,
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []

    def __repr__(self) -> str:
        return f"LLMMessage(role={self.role}, content={self.content[:50]}..., tool_calls={self.tool_calls})"





class LLMClient:
    """
    Unified async client for Ollama and OpenRouter (via OpenAI SDK).
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        system_prompt: str = "",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,  # only for OpenRouter (optional)
    ):
        """
        Args:
            provider: "ollama" or "openrouter"
            model: Model name.
                - Ollama: e.g., "qwen4b:4b"
                - OpenRouter: e.g., "anthropic/claude-3.5-sonnet"
            system_prompt: System instruction.
            api_key: API key (env var if not provided).
            base_url: Custom endpoint for OpenRouter (defaults to OpenRouter's API).
        """
        self.provider = provider
        self.system_prompt = system_prompt
        self.model = model or self._default_model(provider)

        # Configure the appropriate client
        if provider == "ollama":
            self._ollama_client = ollama.AsyncClient()
        elif provider == "openrouter":
            api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("Missing OpenRouter API key. Set OPENROUTER_API_KEY env var or pass api_key.")
            self._openai_client = openai.AsyncOpenAI(
                base_url=base_url or "https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @staticmethod
    def _default_model(provider: str) -> str:
        if provider == "ollama":
            return "qwen4b:4b"
        elif provider == "openrouter":
            return "anthropic/claude-3.5-sonnet"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
    ) -> LLMMessage:
        """
        Send a chat request and return the assistant message.

        Args:
            messages: List of message dicts with "role" and "content".
            tools: List of OpenAI‑style tool definitions (supported by openrouter).
        """
        if self.provider == "ollama":
            return await self._ollama_chat(messages, tools)
        elif self.provider == "openrouter":
            return await self._openrouter_chat(messages, tools)
        else:
            raise RuntimeError(f"Invalid provider: {self.provider}")



    async def _ollama_chat(self, messages: List[Dict], tools: Optional[List[Dict]]) -> LLMMessage:
        # Insert system prompt as a system message if present
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        response = await self._ollama_client.chat(
            model=self.model,
            messages=messages,
            tools=tools or None,
        )
        return LLMMessage(
            role=response.message.role,
            content=response.message.content,
            tool_calls=getattr(response.message, "tool_calls", None),
        )



    async def _openrouter_chat(self, messages: List[Dict], tools: Optional[List[Dict]]) -> LLMMessage:
        # Prepend system message if needed
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        kwargs = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            # Some OpenRouter models require tool_choice to be set when tools are provided
            kwargs["tool_choice"] = "auto"

        response = await self._openai_client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                    "id": tc.id,
                }
                for tc in msg.tool_calls
            ]

        return LLMMessage(
            role=msg.role,
            content=msg.content or "",
            tool_calls=tool_calls,
        )