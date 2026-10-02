"""Checks a package of HPO embeddings against the build about to use it.

    python check_embeddings.py <unpacked package> <embeddings.lock> <hpo_tsv_reader.py>

Run by Dockerfile.recognizer. The package comes from a diagnostic.ar-model
release (fetch_embeddings.sh); its manifest records what it was made from.
Exits non-zero, saying what to do, when:

- the encoder or the HPO labels it was made with are not the ones pinned in
  embeddings.lock (the extractor encodes text with that same encoder, so
  vectors from another one would be compared against the wrong space);
- the recognizer's generator script changed since (the embeddings would no
  longer be what that script produces);
- any file in the package does not match its sha256 in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lock(path: Path) -> dict[str, str]:
    pairs = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            pairs[key] = value
    return pairs


def problems(package: Path, lock: dict[str, str], generator: Path) -> list[str]:
    manifest = json.loads((package / "manifest.json").read_text())
    found = []
    for key, pin in (
        ("clinlinker_repo", "CLINLINKER_REPO"),
        ("clinlinker_revision", "CLINLINKER_REVISION"),
        ("hpo_translations_commit", "HPO_TRANSLATIONS_COMMIT"),
    ):
        if manifest.get(key) != lock.get(pin):
            found.append(
                f"{key}: the embeddings were made with {manifest.get(key)!r}, "
                f"this build pins {lock.get(pin)!r}"
            )
    if manifest.get("generator_sha256") != sha256(generator):
        found.append(f"{generator.name} changed since the embeddings were generated")
    for name, digest in manifest.get("files", {}).items():
        path = package / name
        if not path.is_file() or sha256(path) != digest:
            found.append(f"{name} does not match the manifest")
    return found


def main() -> int:
    package, lock_path, generator = (Path(arg) for arg in sys.argv[1:4])
    found = problems(package, read_lock(lock_path), generator)
    if found:
        print(
            "The HPO embeddings do not match this build. Publish new ones in "
            "diagnostic.ar-model (hpo-embeddings/embeddings.lock) and pin them in "
            "services/api-gateway/embeddings.lock:",
            *(f"  - {p}" for p in found),
            sep="\n",
            file=sys.stderr,
        )
        return 1
    manifest = json.loads((package / "manifest.json").read_text())
    print(f"HPO embeddings v{manifest['version']}: {manifest['terms']} terms, checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
