"""
demo.py
-------
A short, friendly walkthrough you can run (or record a screen capture
of) to show the cache working:

  1. Ask a brand-new question -> should be a MISS (goes to the LLM).
  2. Ask the exact same question again -> should be a HIT (instant).
  3. Ask a differently-worded version of the same question -> should
     ALSO be a HIT, because the cache compares meaning, not exact text.

Run it with:
    python scripts/demo.py
"""

import time
import httpx

BASE_URL = "http://localhost:8000"
MODEL = "llama3.2:1b"


def ask(question: str, client: httpx.Client):
    print(f"\n> Asking: {question!r}")
    start = time.time()
    resp = client.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0.7,
        },
    )
    elapsed = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    status = data.get("cache_status")
    similarity = data.get("cache_similarity")
    answer = data["choices"][0]["message"]["content"]

    print(f"  Cache status: {status}"
          + (f" (similarity={similarity})" if similarity is not None else ""))
    print(f"  Took:         {elapsed*1000:.0f} ms")
    print(f"  Answer:       {answer[:200]}{'...' if len(answer) > 200 else ''}")


def main():
    with httpx.Client(timeout=120.0) as client:
        ask("What is Python used for?", client)                  # expect MISS
        ask("What is Python used for?", client)                  # expect HIT
        ask("Tell me what people use Python for.", client)       # expect HIT (paraphrase)

    print("\nCheck overall stats with:  curl http://localhost:8000/stats")


if __name__ == "__main__":
    main()
