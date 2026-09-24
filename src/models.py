"""
models.py
---------
Defines the "shape" of requests and responses using Pydantic.

We deliberately mirror OpenAI's /v1/chat/completions request format.
That is what makes this a "drop-in" proxy: any app, script, or SDK that
already knows how to talk to OpenAI (or Ollama, which copies the same
format) can talk to our cache instead, just by changing the base URL.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str  # "system", "user", or "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model name, e.g. 'llama3.2:1b'")
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)
    # Custom field (not part of the official OpenAI schema) so callers can
    # tell, programmatically, whether this reply came from the cache.
    cache_status: str = "MISS"  # "HIT" or "MISS"
    cache_similarity: Optional[float] = None
