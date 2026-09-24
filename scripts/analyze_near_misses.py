"""
analyze_near_misses.py
-----------------------
Reads logs/near_misses.jsonl (created automatically by the running
app) and shows you questions that were CLOSE to a cache hit but just
missed the similarity threshold. Use this to decide whether to lower
the threshold, or to spot prompts that should be rephrased/normalized.

Run it with:
    python scripts/analyze_near_misses.py
"""

import json
from pathlib import Path
from collections import Counter

LOG_PATH = Path("logs/near_misses.jsonl")


def main():
    if not LOG_PATH.exists():
        print(f"No near-miss log found yet at {LOG_PATH}. "
              "Run the app and send some traffic first.")
        return

    records = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("Log file exists but is empty — nothing to analyze yet.")
        return

    records.sort(key=lambda r: r["similarity"], reverse=True)

    print(f"Found {len(records)} near-miss events.\n")
    print(f"{'Similarity':>10}  {'Incoming Prompt':<45}  Closest Cached Prompt")
    print("-" * 100)
    for r in records[:25]:
        print(
            f"{r['similarity']:>10.4f}  "
            f"{r['incoming_prompt'][:43]:<45}  "
            f"{(r.get('closest_cached_prompt') or '')[:45]}"
        )

    avg_sim = sum(r["similarity"] for r in records) / len(records)
    print(f"\nAverage near-miss similarity: {avg_sim:.4f}")
    print(f"Current threshold (from these logs): {records[0]['threshold']}")
    print(
        "\nIf many near-misses look like they SHOULD have matched, "
        "consider lowering SIMILARITY_THRESHOLD slightly in your .env file."
    )


if __name__ == "__main__":
    main()
