"""
cache_store.py
--------------
This is the "brain" of the semantic cache. It talks to ChromaDB (a
free, open-source vector database that just saves its data as files on
your disk — no server, no account, no cost) to:

  1. Store a prompt's embedding + the LLM's answer + some metadata.
  2. Given a NEW prompt, find the most similar thing we've stored
     before and decide whether it's similar enough to reuse.

KEY DESIGN DECISIONS (explained):

- Cache "namespaces": Two identical user messages should NOT share an
  answer if they used a different system prompt, model, or temperature
  (e.g. "creative writing mode" vs. "strict factual mode"). So every
  entry is tagged with a `namespace` string built from those three
  things, and lookups are filtered to the same namespace first.

- TTL (time-to-live): Some answers stay true forever ("What is the
  capital of France?"). Others go stale fast ("What's today's date?").
  Every entry stores an `expires_at` timestamp, and expired entries are
  ignored (and lazily cleaned up) rather than served.

- Near-miss logging: If a prompt was CLOSE to the threshold but didn't
  quite make it, we log that so you can later analyze whether the
  threshold should be adjusted (see scripts/analyze_near_misses.py).
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import chromadb

from .config import settings

TIME_SENSITIVE_KEYWORDS = [
    "today", "now", "current", "latest", "currently",
    "this week", "this month", "this year", "recent", "breaking",
]


def classify_ttl(prompt: str) -> int:
    """
    A very simple TTL classifier: if the prompt smells time-sensitive,
    give it a short lifetime; otherwise treat it as stable knowledge.
    This is intentionally simple so it's easy to understand and extend.
    """
    lowered = prompt.lower()
    if any(keyword in lowered for keyword in TIME_SENSITIVE_KEYWORDS):
        return settings.short_ttl_seconds
    return settings.default_ttl_seconds


def build_namespace(model: str, system_prompt: str, temperature: float) -> str:
    """
    Combines model + system prompt + temperature into one short hash so
    that requests with different "context" never share cache entries,
    even if the user's question is identical.
    """
    raw = f"{model}::{system_prompt}::{round(temperature, 3)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class CacheLookupResult:
    hit: bool
    similarity: float = 0.0
    response_text: Optional[str] = None
    model: Optional[str] = None
    entry_id: Optional[str] = None


class CacheStore:
    def __init__(self):
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.near_miss_log_path = Path(settings.log_dir) / "near_misses.jsonl"

    # ------------------------------------------------------------------
    def lookup(
        self,
        prompt: str,
        embedding: List[float],
        namespace: str,
    ) -> CacheLookupResult:
        """
        Looks for the closest previously-cached prompt within the same
        namespace. Returns a hit only if similarity clears the
        configured threshold AND the entry hasn't expired.
        """
        count = self.collection.count()
        if count == 0:
            return CacheLookupResult(hit=False)

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(5, count),
            where={"namespace": namespace},
        )

        ids = results.get("ids", [[]])[0]
        if not ids:
            return CacheLookupResult(hit=False)

        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]

        # Chroma returns cosine *distance*; similarity = 1 - distance
        best_similarity = 1 - distances[0]
        best_meta = metadatas[0]
        best_id = ids[0]
        best_doc = documents[0]

        now = time.time()
        expired = best_meta.get("expires_at", 0) < now

        if not expired and best_similarity >= settings.similarity_threshold:
            self._bump_hit_count(best_id, best_meta)
            return CacheLookupResult(
                hit=True,
                similarity=best_similarity,
                response_text=best_meta.get("response_text") or best_doc,
                model=best_meta.get("model"),
                entry_id=best_id,
            )

        # Not a hit — but log it if it was a "near miss" for later tuning
        lower_bound = settings.similarity_threshold - settings.near_miss_margin
        if not expired and lower_bound <= best_similarity < settings.similarity_threshold:
            self._log_near_miss(prompt, best_similarity, best_meta)

        return CacheLookupResult(hit=False, similarity=best_similarity)

    # ------------------------------------------------------------------
    def store(
        self,
        prompt: str,
        embedding: List[float],
        response_text: str,
        model: str,
        namespace: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> str:
        """Saves a fresh (prompt -> response) pair for future reuse."""
        entry_id = str(uuid.uuid4())
        now = time.time()
        ttl = classify_ttl(prompt)

        self.collection.add(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[response_text],
            metadatas=[{
                "namespace": namespace,
                "model": model,
                "prompt_text": prompt,
                "response_text": response_text,
                "created_at": now,
                "expires_at": now + ttl,
                "ttl_seconds": ttl,
                "hit_count": 0,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }],
        )
        return entry_id

    # ------------------------------------------------------------------
    def _bump_hit_count(self, entry_id: str, meta: Dict[str, Any]) -> None:
        """Keeps a running count of how many times each entry was reused."""
        new_meta = dict(meta)
        new_meta["hit_count"] = int(meta.get("hit_count", 0)) + 1
        new_meta["last_hit_at"] = time.time()
        try:
            self.collection.update(ids=[entry_id], metadatas=[new_meta])
        except Exception:
            # Non-fatal: hit-count bookkeeping should never break a cache hit
            pass

    def _log_near_miss(self, prompt: str, similarity: float, meta: Dict[str, Any]) -> None:
        record = {
            "timestamp": time.time(),
            "incoming_prompt": prompt,
            "similarity": similarity,
            "closest_cached_prompt": meta.get("prompt_text"),
            "threshold": settings.similarity_threshold,
        }
        with open(self.near_miss_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Basic counts used by the /stats endpoint."""
        return {
            "total_entries": self.collection.count(),
            "similarity_threshold": settings.similarity_threshold,
        }

    def purge_expired(self) -> int:
        """Deletes entries whose TTL has passed. Safe to call periodically."""
        now = time.time()
        all_entries = self.collection.get(include=["metadatas"])
        ids_to_delete = [
            entry_id
            for entry_id, meta in zip(all_entries["ids"], all_entries["metadatas"])
            if meta.get("expires_at", 0) < now
        ]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def invalidate_by_namespace(self, namespace: str) -> int:
        """Manual invalidation — e.g. after you change a system prompt."""
        matches = self.collection.get(where={"namespace": namespace}, include=[])
        ids = matches.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)


_cache_store_instance: Optional[CacheStore] = None


def get_cache_store() -> CacheStore:
    global _cache_store_instance
    if _cache_store_instance is None:
        _cache_store_instance = CacheStore()
    return _cache_store_instance
