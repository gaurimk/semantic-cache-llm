"""
metrics.py
----------
Optional monitoring using `prometheus_client` — a free, open-source
Python library. If ENABLE_METRICS=true (the default), the app exposes
a /metrics endpoint that a free Prometheus server can scrape, and a
free Grafana dashboard can visualize (see monitoring/ and
docker-compose.yml). None of this costs anything, and the app works
fine even if you never touch Prometheus or Grafana at all.
"""

from prometheus_client import Counter, Histogram, Gauge

CACHE_REQUESTS_TOTAL = Counter(
    "cache_requests_total", "Total number of chat completion requests received"
)
CACHE_HITS_TOTAL = Counter(
    "cache_hits_total", "Number of requests served from the semantic cache"
)
CACHE_MISSES_TOTAL = Counter(
    "cache_misses_total", "Number of requests forwarded to the LLM"
)
REQUEST_LATENCY_SECONDS = Histogram(
    "request_latency_seconds",
    "End-to-end request latency in seconds",
    labelnames=("cache_status",),
)
CACHE_SIZE = Gauge(
    "cache_entries_total", "Number of entries currently stored in the cache"
)
ESTIMATED_COST_SAVED_USD = Counter(
    "estimated_cost_saved_usd_total",
    "Illustrative estimated dollars saved by serving from cache instead of the LLM",
)
