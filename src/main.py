"""
main.py
-------
This is the FastAPI application — the "front door" of the project.

WHAT IT DOES, IN PLAIN ENGLISH:
Your app (or a script, or curl) sends a chat message to this server
instead of sending it directly to an LLM. This server:

  1. Turns the message into an embedding (a "meaning fingerprint").
  2. Checks whether we've already answered a very similar question
     recently (a "cache hit"). If so, it replies instantly, for free,
     using the saved answer.
  3. If not, it forwards the question to Ollama (a free, local LLM),
     saves the answer for next time, and returns it.

Every response includes an `X-Cache-Status: HIT` or `MISS` header and
a `cache_status` field in the JSON body, so you can always see what
happened.
"""

import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response

from .config import settings
from .models import ChatCompletionRequest, ChatCompletionResponse, Usage
from .embeddings import get_embedding_service
from .cache_store import get_cache_store, build_namespace
from .llm_client import get_llm_client
from . import metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("semantic-cache")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the embedding model and cache store once at startup so the
    # first real request isn't slowed down by a cold start.
    logger.info("Loading embedding model (first run downloads it once, then it's cached)...")
    get_embedding_service()
    get_cache_store()
    logger.info("Semantic cache ready.")
    yield
    logger.info("Shutting down semantic cache.")


app = FastAPI(
    title="Semantic Caching Layer for Local LLMs",
    description="A free, local, drop-in caching proxy for LLM chat requests.",
    version="1.0.0",
    lifespan=lifespan,
)


def _last_user_message(messages) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    # Fallback: concatenate everything if no user-role message is found
    return " ".join(m.content for m in messages)


def _system_prompt(messages) -> str:
    for msg in messages:
        if msg.role == "system":
            return msg.content
    return ""


def _estimate_tokens(text: str) -> int:
    # Rough, free, offline approximation (~4 characters per token).
    # Good enough for illustrative cost-savings numbers, not for billing.
    return max(1, len(text) // 4)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest, response: Response):
    start_time = time.time()
    metrics.CACHE_REQUESTS_TOTAL.inc()

    embedder = get_embedding_service()
    cache = get_cache_store()

    user_prompt = _last_user_message(req.messages)
    system_prompt = _system_prompt(req.messages)
    namespace = build_namespace(req.model, system_prompt, req.temperature or 0.7)

    embedding = embedder.embed(user_prompt)
    lookup = cache.lookup(user_prompt, embedding, namespace)

    if lookup.hit:
        elapsed = time.time() - start_time
        metrics.CACHE_HITS_TOTAL.inc()
        metrics.REQUEST_LATENCY_SECONDS.labels(cache_status="HIT").observe(elapsed)
        metrics.ESTIMATED_COST_SAVED_USD.inc(
            _estimate_tokens(user_prompt) / 1000 * settings.estimated_cost_per_1k_tokens_usd
        )
        response.headers["X-Cache-Status"] = "HIT"
        return ChatCompletionResponse(
            id=f"cache-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=req.model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": lookup.response_text},
                "finish_reason": "stop",
            }],
            usage=Usage(),
            cache_status="HIT",
            cache_similarity=round(lookup.similarity, 4),
        )

    # --- Cache miss: forward to the free local LLM (Ollama) ---------
    llm = get_llm_client()
    try:
        raw = await llm.chat_completion(
            model=req.model,
            messages=[m.model_dump() for m in req.messages],
            temperature=req.temperature or 0.7,
            max_tokens=req.max_tokens,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        choice = raw["choices"][0]
        answer_text = choice["message"]["content"]
        usage_raw = raw.get("usage", {})
    except (KeyError, IndexError) as exc:
        raise HTTPException(
            status_code=502, detail=f"Unexpected response from LLM provider: {raw}"
        ) from exc

    cache.store(
        prompt=user_prompt,
        embedding=embedding,
        response_text=answer_text,
        model=req.model,
        namespace=namespace,
        prompt_tokens=usage_raw.get("prompt_tokens", _estimate_tokens(user_prompt)),
        completion_tokens=usage_raw.get("completion_tokens", _estimate_tokens(answer_text)),
    )

    elapsed = time.time() - start_time
    metrics.CACHE_MISSES_TOTAL.inc()
    metrics.REQUEST_LATENCY_SECONDS.labels(cache_status="MISS").observe(elapsed)
    response.headers["X-Cache-Status"] = "MISS"

    return ChatCompletionResponse(
        id=raw.get("id", f"miss-{uuid.uuid4().hex[:12]}"),
        created=int(time.time()),
        model=req.model,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": answer_text},
            "finish_reason": choice.get("finish_reason", "stop"),
        }],
        usage=Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        ),
        cache_status="MISS",
        cache_similarity=round(lookup.similarity, 4) if lookup.similarity else None,
    )


@app.get("/stats")
async def stats():
    """Human-readable snapshot of cache health — no dashboard required."""
    cache = get_cache_store()
    total_requests = metrics.CACHE_REQUESTS_TOTAL._value.get()
    total_hits = metrics.CACHE_HITS_TOTAL._value.get()
    hit_rate = (total_hits / total_requests) if total_requests else 0.0
    cache_snapshot = cache.stats()
    metrics.CACHE_SIZE.set(cache_snapshot["total_entries"])
    return {
        "total_requests": int(total_requests),
        "cache_hits": int(total_hits),
        "cache_misses": int(metrics.CACHE_MISSES_TOTAL._value.get()),
        "hit_rate": round(hit_rate, 4),
        "estimated_cost_saved_usd": round(
            metrics.ESTIMATED_COST_SAVED_USD._value.get(), 6
        ),
        **cache_snapshot,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/admin/purge-expired")
async def purge_expired():
    cache = get_cache_store()
    deleted = cache.purge_expired()
    return {"deleted": deleted}


@app.post("/admin/invalidate")
async def invalidate(model: str, system_prompt: str = "", temperature: float = 0.7):
    """
    Manually clear cached answers for a given model/system-prompt/temperature
    combination — useful right after you change a prompt or upgrade a model.
    """
    cache = get_cache_store()
    namespace = build_namespace(model, system_prompt, temperature)
    deleted = cache.invalidate_by_namespace(namespace)
    return {"namespace": namespace, "deleted": deleted}


@app.get("/metrics")
async def metrics_endpoint():
    from fastapi.responses import Response as FastAPIResponse
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    if not settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics are disabled")
    return FastAPIResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    return {
        "name": "Semantic Caching Layer for Local LLMs",
        "docs": "/docs",
        "stats": "/stats",
        "health": "/health",
    }
