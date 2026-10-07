"""Prometheus metrics.

Label cardinality is kept bounded on purpose: aliases, providers, models and API
key *names* come from config, never from free-form client input.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)

REQUEST_LATENCY = Histogram(
    "gateway_request_duration_seconds",
    "End-to-end request latency as seen by the gateway.",
    ["endpoint", "model_alias", "provider", "status"],
    buckets=LATENCY_BUCKETS,
)
TIME_TO_FIRST_TOKEN = Histogram(
    "gateway_time_to_first_token_seconds",
    "Time from request start to the first streamed chunk.",
    ["model_alias", "provider"],
    buckets=LATENCY_BUCKETS,
)
PROVIDER_REQUESTS = Counter(
    "gateway_provider_requests_total",
    "Calls made to upstream providers (each retry counts).",
    ["provider", "model", "outcome"],
)
PROVIDER_ERRORS = Counter(
    "gateway_provider_errors_total",
    "Upstream errors by provider and error kind.",
    ["provider", "kind", "status"],
)
FALLBACKS = Counter(
    "gateway_fallbacks_total",
    "Times a request moved on to the next target in its fallback chain.",
    ["model_alias", "from_provider"],
)
TOKENS = Counter(
    "gateway_tokens_total",
    "Tokens processed, by direction.",
    ["provider", "model", "api_key", "type"],
)
COST_USD = Counter(
    "gateway_cost_usd_total",
    "Estimated spend in USD, from the configured price table.",
    ["provider", "model", "api_key"],
)
CACHE_LOOKUPS = Counter(
    "gateway_cache_lookups_total",
    "Response cache lookups.",
    ["cache", "result"],
)
RATE_LIMITED = Counter(
    "gateway_rate_limited_total",
    "Requests rejected by the per-key rate limiter.",
    ["api_key"],
)
