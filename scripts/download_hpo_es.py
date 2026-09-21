"""Download the official Spanish HPO labels.

Curated by the HPO Internationalisation Effort. These are what `/extract-hpo`
returns as `label_es`, so they are the Spanish symptom names a user reads in
the wizard.

`download_hpo.py` fetches `hp.obo` but nothing fetched this file, which is why
the committed copy drifted months behind upstream. Run both together:

    python scripts/download_hpo.py
    python scripts/download_hpo_es.py
    python scripts/build_hpo_es.py
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import requests

DEFAULT_URL = (
    "https://raw.githubusercontent.com/obophenotype/hpo-translations/"
    "main/babelon/hp-es.babelon.tsv"
)

# The file carries ~20k terms. Anything far below that is a truncated download
# or a moved file, and installing it would silently shrink symptom coverage.
MIN_EXPECTED_LABELS = 15_000


def count_labels(payload: bytes) -> int:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")), delimiter="\t")
    return sum(
        1
        for row in reader
        if row.get("predicate_id") == "rdfs:label"
        and (row.get("subject_id") or "").strip()
        and (row.get("translation_value") or "").strip()
    )


def download(url: str, dest: Path) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    labels = count_labels(response.content)
    if labels < MIN_EXPECTED_LABELS:
        raise SystemExit(
            f"Refusing to install: only {labels} labels found, expected at least "
            f"{MIN_EXPECTED_LABELS}. The source file may have moved or been truncated."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    print(f"Wrote {dest} ({labels} labels)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="data/hpo/hp-es.tsv")
    args = parser.parse_args()
    download(args.url, Path(args.out))


if __name__ == "__main__":
    main()
