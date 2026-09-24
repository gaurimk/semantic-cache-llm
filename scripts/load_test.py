"""
load_test.py
------------
Sends a batch of requests through the cache proxy to see how it
performs under realistic, mixed traffic (some brand-new questions,
some repeats, some "same meaning, different words").

Run it with:
    python scripts/load_test.py --num-requests 200

This is free to run as many times as you like — everything happens on
your own machine with a local LLM, so there is no bill to worry about.
"""

import argparse
import json
import random
import time
import statistics
import httpx

BASE_QUESTIONS = [
    "What is Python used for?",
    "Explain what a REST API is.",
    "What's the difference between a list and a tuple in Python?",
    "How does garbage collection work?",
    "What is Docker and why do people use it?",
    "Explain the concept of recursion with an example.",
    "What is a vector database?",
    "How do neural networks learn?",
    "What is the difference between SQL and NoSQL databases?",
    "Explain what an API rate limit is.",
]

# Paraphrases that a good semantic cache SHOULD recognize as "the same
# question" even though the wording is different.
PARAPHRASES = {
    "What is Python used for?": "Tell me what people use Python for.",
    "Explain what a REST API is.": "Can you explain REST APIs to me?",
    "What is Docker and why do people use it?": "Why do developers use Docker?",
}


def build_traffic(num_requests: int) -> list[str]:
    traffic = []
    for _ in range(num_requests):
        roll = random.random()
        base = random.choice(BASE_QUESTIONS)
        if roll < 0.4 and base in PARAPHRASES:
            traffic.append(PARAPHRASES[base])
        else:
            traffic.append(base)
    return traffic


def main():
    parser = argparse.ArgumentParser(description="Load test the semantic cache proxy.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument("--num-requests", type=int, default=200)
    args = parser.parse_args()

    traffic = build_traffic(args.num_requests)

    hits, misses, errors = 0, 0, 0
    hit_latencies, miss_latencies = [], []

    print(f"Sending {len(traffic)} requests to {args.base_url} ...")
    with httpx.Client(timeout=120.0) as client:
        for i, question in enumerate(traffic, start=1):
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": question}],
                "temperature": 0.7,
            }
            start = time.time()
            try:
                resp = client.post(f"{args.base_url}/v1/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                errors += 1
                print(f"[{i}] ERROR: {exc}")
                continue
            elapsed = time.time() - start

            if data.get("cache_status") == "HIT":
                hits += 1
                hit_latencies.append(elapsed)
            else:
                misses += 1
                miss_latencies.append(elapsed)

            if i % 20 == 0:
                print(f"  ...{i}/{len(traffic)} done")

    total = hits + misses
    print("\n--- Load Test Results ---")
    print(f"Total requests:   {len(traffic)}")
    print(f"Errors:           {errors}")
    print(f"Cache hits:       {hits}")
    print(f"Cache misses:     {misses}")
    if total:
        print(f"Hit rate:         {hits / total:.1%}")
    if hit_latencies:
        print(f"Avg HIT latency:  {statistics.mean(hit_latencies)*1000:.1f} ms")
    if miss_latencies:
        print(f"Avg MISS latency: {statistics.mean(miss_latencies)*1000:.1f} ms")
    if hit_latencies and miss_latencies:
        speedup = statistics.mean(miss_latencies) / max(statistics.mean(hit_latencies), 1e-6)
        print(f"Speedup (MISS/HIT): {speedup:.1f}x faster on a cache hit")

    results = {
        "total_requests": len(traffic),
        "errors": errors,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / total if total else 0,
        "avg_hit_latency_ms": statistics.mean(hit_latencies) * 1000 if hit_latencies else None,
        "avg_miss_latency_ms": statistics.mean(miss_latencies) * 1000 if miss_latencies else None,
    }
    with open("load_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved detailed results to load_test_results.json")


if __name__ == "__main__":
    main()
