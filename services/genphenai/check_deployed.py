#!/usr/bin/env python3
"""Checks a deployed GenPhenAI against model-v4.0.0's staging fixture.

    python3 services/genphenai/check_deployed.py https://<host>/genphenai

- /healthz names the model and the v4-layer pinned in model.lock, with their
  sha256s, and the layer is enabled.
- Each of tests/data/v4_staging_fixture.json's 32 cases through /rank-genes:
  the network's top 10 without inputs, and v4-layer's top 10 with the case's
  sex, age and onset: the same genes in the same order, each probability
  within 2e-5 (responses round to 6 decimals, and float32 kernels drift
  across CPUs).
- The latency of those requests, p50 and p95.

Standard library only, so it runs on any machine with Python 3.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "tests" / "data" / "v4_staging_fixture.json"
LOCK = HERE / "model.lock"
TOLERANCE = 2e-5


def read_lock() -> dict[str, str]:
    pairs = {}
    for line in LOCK.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            pairs[key] = value
    return pairs


def call(url: str, body: dict | None = None) -> tuple[dict, float]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    return payload, time.perf_counter() - start


def compare(body: dict, expected: list) -> str | None:
    got = [(r["gene_symbol"], r["score"]) for r in body["results"]]
    want = [(gene.upper(), p) for gene, p in expected]
    if [g for g, _ in got] != [g for g, _ in want]:
        return f"top 10 {[g for g, _ in got]}, expected {[g for g, _ in want]}"
    worst = max(abs(s - p) for (_, s), (_, p) in zip(got, want))
    return None if worst <= TOLERANCE else f"probability off by {worst:.2e}"


def main(base: str) -> int:
    base = base.rstrip("/")
    lock = read_lock()
    failures = []

    health, _ = call(f"{base}/healthz")
    expected_model = {"release": lock["MODEL_TAG"], "sha256": lock["MODEL_SHA256"]}
    if health.get("model") != expected_model:
        failures.append(f"/healthz model {health.get('model')}, expected {expected_model}")
    layer = health.get("layer") or {}
    if (layer.get("release"), layer.get("sha256"), layer.get("enabled")) != (
        lock["V4_LAYER_TAG"], lock["V4_LAYER_SHA256"], True
    ):
        failures.append(f"/healthz layer {layer or None}, expected {lock['V4_LAYER_TAG']} enabled")
    print(f"healthz: model {health.get('model')}, layer {layer.get('name')} enabled={layer.get('enabled')}")

    timings = []
    cases = json.loads(FIXTURE.read_text())["cases"]
    for case in cases:
        inputs = case["inputs"]
        for name, body, expected in (
            ("network", {"hpo_ids": case["hpo_ids"], "top_k": 10}, case["network_top"]),
            (
                "v4-layer",
                {
                    "hpo_ids": case["hpo_ids"],
                    "top_k": 10,
                    "sex": inputs["sex"],
                    "age_years": inputs["age_years"],
                    "onset_years": inputs["onset_years"],
                },
                case["f3_top"],
            ),
        ):
            response, seconds = call(f"{base}/rank-genes", body)
            timings.append(seconds)
            problem = compare(response, expected)
            if problem:
                failures.append(f"{case['kind']} {case['case_id']} ({name}): {problem}")

    p95 = statistics.quantiles(timings, n=20)[-1]
    print(f"cases: {len(cases)}, requests: {len(timings)}, "
          f"latency p50 {statistics.median(timings) * 1000:.0f} ms, p95 {p95 * 1000:.0f} ms")
    for failure in failures:
        print(f"FAIL {failure}")
    print("all checks passed" if not failures else f"{len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: check_deployed.py https://<host>/genphenai")
    sys.exit(main(sys.argv[1]))
