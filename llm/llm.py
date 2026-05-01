# llm.py
import os
from typing import Any, Dict, List, Optional, Union

import ollama
import openai
import google.generativeai as genai

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



def _convert_to_gemini_tools(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI tool definitions to Gemini function declarations."""
    declarations = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            declarations.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {"type": "object", "properties": {}}),
            })
    return declarations


def _extract_gemini_tool_calls(response) -> List[Dict]:
    """Extract tool calls from a Gemini response (Content with parts)."""
    tool_calls = []
    for part in response.parts:
        if fn := part.function_call:
            tool_calls.append({
                "type": "function",
                "function": {
                    "name": fn.name,
                    "arguments": str(fn.args),  # already a dict, but OpenAI expects JSON string
                },
                "id": fn.name,  # Gemini doesn't provide an id, use function name
            })
    return tool_calls



class LLMClient:
    """
    Unified async client for Ollama, OpenRouter (via OpenAI SDK), and Google Gemini.
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
            provider: "ollama", "openrouter", "google"
            model: Model name.
                - Ollama: e.g., "qwen4b:4b"
                - OpenRouter: e.g., "anthropic/claude-3.5-sonnet" (prefix with openrouter/ not needed)
                - Google: e.g., "gemini-1.5-pro"
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
        elif provider == "google":
            api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Missing Google API key. Set GOOGLE_API_KEY env var or pass api_key.")
            genai.configure(api_key=api_key)
            self._genai_model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt if system_prompt else None,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @staticmethod
    def _default_model(provider: str) -> str:
        if provider == "ollama":
            return "qwen4b:4b"
        elif provider == "openrouter":
            return "anthropic/claude-3.5-sonnet"
        elif provider == "google":
            return "gemini-1.5-pro"
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
            tools: List of OpenAI‑style tool definitions (supported by openrouter and google).
        """
        if self.provider == "ollama":
            return await self._ollama_chat(messages, tools)
        elif self.provider == "openrouter":
            return await self._openrouter_chat(messages, tools)
        elif self.provider == "google":
            return await self._google_chat(messages, tools)
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


    async def _google_chat(self, messages: List[Dict], tools: Optional[List[Dict]]) -> LLMMessage:
        # Convert messages to Gemini's format
        # Gemini expects a chat history: user/model alternating.
        # System prompt is already set in the model's system_instruction.
        gemini_messages = []
        for m in messages:
            role = m["role"]
            if role == "system":
                continue  # already handled via system_instruction
            google_role = "user" if role == "user" else "model"
            gemini_messages.append({
                "role": google_role,
                "parts": [m["content"]],
            })

        # Start a chat session
        chat = self._genai_model.start_chat(history=gemini_messages)

        # The last message is from the user – we need to send it separately
        # However, start_chat already consumes the history. The actual generation
        # will use the full conversation.
        # For simplicity, we extract the last user message and pass it as prompt,
        # while the rest is history.

        if not gemini_messages:
            raise ValueError("No user message provided")

        # Find the last user message
        last_user_msg = None
        history = []
        for m in gemini_messages:
            if m["role"] == "user":
                last_user_msg = m
            else:
                history.append(m)
        # Rebuild chat with history only (excluding last user message)
        chat = self._genai_model.start_chat(history=history)

        # Prepare generation config
        generation_config = None
        if tools:
            gemini_tools = _convert_to_gemini_tools(tools)
            if gemini_tools:
                # Gemini uses tool_config to enable function calling
                generation_config = genai.types.GenerationConfig(
                    temperature=0.7,
                    # Note: Gemini's function calling is enabled by passing tools to the model
                )
                # Actually need to pass tools to the chat.send_message
                # We'll do that below

        # Send the last user message with optional tools
        response = await chat.send_message_async(
            last_user_msg["parts"][0],
            tools=gemini_tools if tools else None,
        )

        # Extract content and tool calls
        content_parts = []
        tool_calls = []
        for part in response.parts:
            if text := part.text:
                content_parts.append(text)
            if part.function_call:
                tool_calls.extend(_extract_gemini_tool_calls(response))

        content = " ".join(content_parts) if content_parts else ""
        return LLMMessage(role="assistant", content=content, tool_calls=tool_calls or None)