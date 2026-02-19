#!/usr/bin/env python3
"""Compute model confidence from a probability-like output distribution.

Confidence is defined as:
    confidence = 1 - normalized_shannon_entropy

Where normalized Shannon entropy is:
    H_norm = H / log(N)
    H = -sum(p_i * log(p_i))

This yields:
- close to 0 for near-uniform distributions (low confidence)
- close to 1 for very skewed distributions (high confidence)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping


def load_score_map(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at {path}, got: {type(raw).__name__}")

    score_map: dict[str, float] = {}
    for key, value in raw.items():
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric value for key '{key}': {value!r}") from exc

        if not math.isfinite(score):
            raise ValueError(f"Non-finite value for key '{key}': {value!r}")
        if score < 0:
            raise ValueError(f"Negative value for key '{key}': {value!r}")

        score_map[str(key)] = score

    if not score_map:
        raise ValueError(f"Input JSON object is empty: {path}")
    return score_map


def normalize_distribution(values: Mapping[str, float]) -> dict[str, float]:
    positives = {k: v for k, v in values.items() if v > 0}
    if not positives:
        raise ValueError("Distribution has no positive mass.")

    total = sum(positives.values())
    if total <= 0:
        raise ValueError("Sum of positive values must be > 0.")

    return {k: v / total for k, v in positives.items()}


def normalized_shannon_entropy(values: Mapping[str, float]) -> float:
    probs = list(normalize_distribution(values).values())
    n = len(probs)
    if n <= 1:
        return 0.0

    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(n)
    if max_entropy == 0:
        return 0.0

    h_norm = entropy / max_entropy
    return min(1.0, max(0.0, h_norm))


def confidence_from_distribution(values: Mapping[str, float]) -> float:
    """Return confidence in [0, 1], where 1 means highly concentrated."""
    return 1.0 - normalized_shannon_entropy(values)


def compute_confidence_from_json(path: Path) -> dict[str, float | int]:
    values = load_score_map(path)
    probs = list(normalize_distribution(values).values())
    n = len(probs)
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(n) if n > 1 else 0.0
    h_norm = normalized_shannon_entropy(values)
    confidence = 1.0 - h_norm

    return {
        "n_labels_total": len(values),
        "n_labels_positive": n,
        "entropy": entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": h_norm,
        "confidence": confidence,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute confidence from a distribution JSON via normalized Shannon entropy."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/genphenia_results.json"),
        help="Path to JSON object mapping labels to scores/probabilities.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON (default prints a compact text summary).",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    result = compute_confidence_from_json(args.input)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    print(f"input: {args.input}")
    print(f"labels_total: {result['n_labels_total']}")
    print(f"labels_positive: {result['n_labels_positive']}")
    print(f"normalized_entropy: {result['normalized_entropy']:.6f}")
    print(f"confidence: {result['confidence']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
