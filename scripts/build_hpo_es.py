from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_map(tsv_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with tsv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row.get("predicate_id") != "rdfs:label":
                continue
            hpo_id = (row.get("subject_id") or "").strip()
            label_es = (row.get("translation_value") or "").strip()
            if not hpo_id or not label_es:
                continue
            if hpo_id not in mapping:
                mapping[hpo_id] = label_es
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hp-es.json from hp-es.tsv")
    parser.add_argument("--input", default="data/hpo/hp-es.tsv")
    parser.add_argument("--output", default="data/hpo/hp-es.json")
    args = parser.parse_args()

    tsv_path = Path(args.input)
    out_path = Path(args.output)
    if not tsv_path.exists():
        raise SystemExit(f"Input file not found: {tsv_path}")

    mapping = build_map(tsv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(mapping)} labels to {out_path}")


if __name__ == "__main__":
    main()
