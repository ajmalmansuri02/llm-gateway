// k6 load test for llm-gateway.
//
// Defaults hit the "demo-echo" alias (in-process fake provider), which measures
// the gateway's own overhead: auth, rate limiting, Redis round-trips, caching,
// accounting and serialisation. Point MODEL at "chat-local" to load a real model.
//
//   docker run --rm -i --network host grafana/k6 run - < loadtest/k6-chat.js
//   k6 run -e BASE_URL=http://localhost:8080 -e MODEL=chat-local loadtest/k6-chat.js
//
// Env: BASE_URL, API_KEY, MODEL, CACHE_RATIO (0..1 share of repeated prompts),
//      QUICK=1 for a ~20s smoke run

import http from "k6/http";
import { check } from "k6";
import { Counter, Trend } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080";
const API_KEY = __ENV.API_KEY || "sk-gw-loadtest-local-only";
const MODEL = __ENV.MODEL || "demo-echo";
const CACHE_RATIO = parseFloat(__ENV.CACHE_RATIO || "0.3");
const QUICK = __ENV.QUICK === "1";
const d = (full, quick) => (QUICK ? quick : full);

const cacheHits = new Counter("gateway_cache_hits");
const ttfb = new Trend("gateway_ttfb_ms", true);
const streamed = new Trend("gateway_stream_total_ms", true);

export const options = {
  scenarios: {
    non_streaming: {
      executor: "ramping-arrival-rate",
      exec: "chat",
      startRate: 5,
      timeUnit: "1s",
      preAllocatedVUs: 20,
      maxVUs: 200,
      stages: [
        { target: 50, duration: d("30s", "5s") },
        { target: 50, duration: d("1m", "10s") },
        { target: 0, duration: d("15s", "5s") },
      ],
    },
    streaming: {
      executor: "constant-vus",
      exec: "chatStream",
      vus: 5,
      duration: d("1m45s", "20s"),
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    "http_req_duration{scenario:non_streaming}": ["p(95)<500"],
  },
};

const headers = {
  "Content-Type": "application/json",
  Authorization: `Bearer ${API_KEY}`,
};

function prompt() {
  // A slice of traffic repeats a small set of prompts so the cache gets exercised.
  if (Math.random() < CACHE_RATIO) {
    return `What is a token bucket? (variant ${Math.floor(Math.random() * 5)})`;
  }
  return `Summarise request ${__VU}-${__ITER}-${Math.random().toString(36).slice(2)}`;
}

export function chat() {
  const res = http.post(
    `${BASE_URL}/v1/chat/completions`,
    JSON.stringify({ model: MODEL, messages: [{ role: "user", content: prompt() }], temperature: 0 }),
    { headers, tags: { name: "chat" } },
  );
  check(res, {
    "status 200": (r) => r.status === 200,
    "has choices": (r) => r.status === 200 && r.json("choices.0.message.content") !== undefined,
  });
  ttfb.add(res.timings.waiting);
  if (res.headers["X-Gateway-Cache"] && res.headers["X-Gateway-Cache"].includes("hit")) {
    cacheHits.add(1);
  }
}

export function chatStream() {
  const res = http.post(
    `${BASE_URL}/v1/chat/completions`,
    JSON.stringify({
      model: MODEL,
      stream: true,
      messages: [{ role: "user", content: prompt() }],
    }),
    { headers, tags: { name: "chat_stream" } },
  );
  check(res, {
    "stream 200": (r) => r.status === 200,
    "stream ends with [DONE]": (r) => r.body && r.body.trim().endsWith("data: [DONE]"),
  });
  streamed.add(res.timings.duration);
}
