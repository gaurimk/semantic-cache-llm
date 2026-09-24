"""
embeddings.py
-------------
Turns text into a list of numbers (a "vector" / "embedding") that
captures its *meaning*. Two sentences that mean the same thing end up
with similar vectors, even if they use completely different words.

WHY THIS MATTERS FOR CACHING (non-technical explanation):
"What is Python?" and "Explain Python to me" are different strings of
text, so a normal cache (which only checks for an exact match) would
treat them as two different questions and call the LLM twice. By
comparing meanings instead of exact words, we can recognize that these
are "the same question" and answer the second one instantly from cache.

This uses `sentence-transformers`, a free, open-source library. The
model file (~80MB) downloads once from Hugging Face the first time you
run the app, is cached locally, and then works completely offline with
no API key and no per-call cost — unlike OpenAI's embedding API.
"""

from functools import lru_cache
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings


class EmbeddingService:
    def __init__(self, model_name: str):
        # Loading the model is the slow part (a few seconds), so we do
        # it once when the app starts, not on every request.
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        """Return a normalized embedding vector for a single string."""
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """
        How similar two vectors are, from -1 (opposite) to 1 (identical).
        Since our vectors are already normalized, this is just a dot
        product, which is cheap to compute.
        """
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr))


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """
    Returns a single shared instance of the embedding model so it is
    only loaded into memory once per process.
    """
    return EmbeddingService(settings.embedding_model_name)
