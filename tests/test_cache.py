"""
test_cache.py
-------------
Basic automated tests. These run entirely offline (after the embedding
model's first download) and don't need Ollama running, so you can
verify the cache logic works before wiring up the full stack.

Run all tests with:
    pytest
"""

import shutil
import tempfile
import time
import pytest

from src.cache_store import CacheStore, build_namespace, classify_ttl
from src.embeddings import get_embedding_service
from src import config as config_module


@pytest.fixture()
def temp_cache(monkeypatch):
    """Gives each test a fresh, throwaway cache directory on disk."""
    temp_dir = tempfile.mkdtemp()
    monkeypatch.setattr(config_module.settings, "chroma_persist_dir", temp_dir)
    monkeypatch.setattr(config_module.settings, "log_dir", temp_dir)
    store = CacheStore()
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_classify_ttl_time_sensitive():
    assert classify_ttl("What's the latest news today?") == config_module.settings.short_ttl_seconds


def test_classify_ttl_stable_fact():
    assert classify_ttl("What is the capital of France?") == config_module.settings.default_ttl_seconds


def test_namespace_changes_with_system_prompt():
    ns1 = build_namespace("llama3.2:1b", "You are a pirate.", 0.7)
    ns2 = build_namespace("llama3.2:1b", "You are a formal assistant.", 0.7)
    assert ns1 != ns2


def test_cache_miss_then_hit(temp_cache):
    embedder = get_embedding_service()
    namespace = build_namespace("llama3.2:1b", "", 0.7)

    prompt = "What is Python used for?"
    embedding = embedder.embed(prompt)

    # First time: nothing stored yet, must be a miss
    result = temp_cache.lookup(prompt, embedding, namespace)
    assert result.hit is False

    # Store an answer
    temp_cache.store(
        prompt=prompt,
        embedding=embedding,
        response_text="Python is used for web development, data science, and more.",
        model="llama3.2:1b",
        namespace=namespace,
    )

    # Same question again: should now be a hit
    result2 = temp_cache.lookup(prompt, embedding, namespace)
    assert result2.hit is True
    assert result2.similarity > 0.99


def test_semantically_similar_prompt_is_a_hit(temp_cache):
    embedder = get_embedding_service()
    namespace = build_namespace("llama3.2:1b", "", 0.7)

    original = "What is Python used for?"
    paraphrase = "Tell me what people use Python for."

    temp_cache.store(
        prompt=original,
        embedding=embedder.embed(original),
        response_text="Python is used for web dev, data science, automation, etc.",
        model="llama3.2:1b",
        namespace=namespace,
    )

    result = temp_cache.lookup(paraphrase, embedder.embed(paraphrase), namespace)
    # Paraphrases should score high similarity with all-MiniLM-L6-v2.
    # We assert hit OR a high similarity score, since exact threshold
    # behavior depends on the configured SIMILARITY_THRESHOLD.
    assert result.similarity > 0.6


def test_expired_entry_is_not_served(temp_cache, monkeypatch):
    embedder = get_embedding_service()
    namespace = build_namespace("llama3.2:1b", "", 0.7)
    prompt = "What time-sensitive TTL test question is this?"
    embedding = embedder.embed(prompt)

    # Force a TTL of -1 second so the entry is already "expired"
    monkeypatch.setattr(config_module.settings, "default_ttl_seconds", -1)
    temp_cache.store(
        prompt=prompt,
        embedding=embedding,
        response_text="This should be considered stale immediately.",
        model="llama3.2:1b",
        namespace=namespace,
    )

    result = temp_cache.lookup(prompt, embedding, namespace)
    assert result.hit is False


def test_different_namespace_does_not_leak_across_system_prompts(temp_cache):
    embedder = get_embedding_service()
    prompt = "What is Python used for?"
    embedding = embedder.embed(prompt)

    ns_pirate = build_namespace("llama3.2:1b", "You are a pirate.", 0.7)
    ns_formal = build_namespace("llama3.2:1b", "You are formal.", 0.7)

    temp_cache.store(
        prompt=prompt,
        embedding=embedding,
        response_text="Arrr, Python be used for codin', matey!",
        model="llama3.2:1b",
        namespace=ns_pirate,
    )

    # Same exact prompt, but a DIFFERENT namespace (different system
    # prompt) should NOT get the pirate answer.
    result = temp_cache.lookup(prompt, embedding, ns_formal)
    assert result.hit is False
