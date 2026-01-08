import argparse
from pathlib import Path

import requests

DEFAULT_URL = "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="data/hpo/hp.obo")
    args = parser.parse_args()
    download(args.url, Path(args.out))


if __name__ == "__main__":
    main()
