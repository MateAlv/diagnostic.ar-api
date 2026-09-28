"""Measure what one extraction costs, to set the global extraction budget.

Runs inside the recognizer container, calling the pipeline directly, so the
budget it is measuring for does not stop it. Every `claude -p` call goes
through a wrapper that asks for JSON output and records its token usage, and
the recognizer's response cache is off, so each extraction pays in full.

    ssh nodo0-diagnostic.ar 'docker exec -i diagnosticar-staging-hpo-recognizer-1 python -' \
        < scripts/calibrate_extraction_budget.py

With a Claude subscription behind it, read the usage page before and after
(claude.ai → Settings → Usage: the current session and the week), with nothing
else using the account in between. docs/EXTRACTION_LIMITS.md turns the two
readings into GLOBAL_EXTRACTIONS_PER_5H and GLOBAL_EXTRACTIONS_PER_WEEK.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import time

RUNS = int(os.environ.get("CALIBRATION_RUNS", "5"))
WRAPPER = "/tmp/claude-usage"
LOG = "/tmp/claude-usage.jsonl"

# A typical history: eight findings, one negation, 267 characters.
TEXT = (
    "Varón de 7 años derivado por talla baja y retraso global del desarrollo. "
    "Presenta microcefalia, orejas de implantación baja, hipertelorismo y "
    "cardiopatía congénita operada al año. La madre refiere episodios de "
    "hipoglucemia y constipación crónica. No tiene hipoacusia."
)

WRAPPER_SOURCE = f'''#!/usr/bin/env python3
import json, subprocess, sys
p = subprocess.run(["claude"] + sys.argv[1:] + ["--output-format", "json"], capture_output=True, text=True)
if p.returncode != 0:
    sys.stderr.write(p.stderr); sys.exit(p.returncode)
d = json.loads(p.stdout)
with open("{LOG}", "a") as f:
    f.write(json.dumps({{"usage": d.get("usage"), "cost": d.get("total_cost_usd")}}) + "\\n")
sys.stdout.write(d.get("result", ""))
'''


def main() -> int:
    with open(WRAPPER, "w") as handle:
        handle.write(WRAPPER_SOURCE)
    os.chmod(WRAPPER, stat.S_IRWXU)
    if os.path.exists(LOG):
        os.remove(LOG)
    # Read by the provider and the response cache when the pipeline is built.
    os.environ["CLAUDE_BINARY"] = WRAPPER
    os.environ["LLM_CACHE_ENABLED"] = "false"

    from app.config import Settings
    from app.recognizer_pipeline import RecognizerPipeline

    pipeline = RecognizerPipeline(Settings())
    started = time.time()
    try:
        for run in range(1, RUNS + 1):
            t0 = time.time()
            response, _ = pipeline.extract_with_meta(TEXT, "es-AR", TEXT)
            print(f"run {run}: {time.time() - t0:.0f}s, {len(response['phenotypes'])} HPO terms", flush=True)
        calls = [json.loads(line) for line in open(LOG)]
    finally:
        for path in (WRAPPER, LOG):
            if os.path.exists(path):
                os.remove(path)

    cost = sum(c["cost"] or 0 for c in calls)
    tokens = sum(
        sum((c["usage"] or {}).get(k, 0) for k in (
            "input_tokens", "output_tokens",
            "cache_creation_input_tokens", "cache_read_input_tokens",
        ))
        for c in calls
    )
    print(json.dumps({
        "extractions": RUNS,
        "llm_calls": len(calls),
        "minutes": round((time.time() - started) / 60, 1),
        "tokens_per_extraction": round(tokens / RUNS),
        "api_equivalent_usd_per_extraction": round(cost / RUNS, 3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
