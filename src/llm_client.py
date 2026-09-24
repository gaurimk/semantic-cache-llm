"""
llm_client.py
-------------
Forwards a cache MISS to Ollama — a free, open-source tool that runs
LLMs (like Llama 3.2) entirely on your own computer. No API key, no
per-token bill, and it works offline once a model is downloaded.

Ollama exposes an OpenAI-compatible endpoint at /v1/chat/completions,
so this client's job is small: just forward the request and pass the
answer back.
"""

from typing import List, Dict, Any, Optional
import httpx

from .config import settings


class LLMClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Calls Ollama's OpenAI-compatible chat endpoint and returns the
        raw JSON response.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions", json=payload
                )
                response.raise_for_status()
                return response.json()
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    "Could not reach Ollama at "
                    f"{self.base_url}. Is Ollama running? "
                    "Start it with: ollama serve  (or just open the Ollama app)"
                ) from exc


def get_llm_client() -> LLMClient:
    return LLMClient(settings.ollama_base_url)
