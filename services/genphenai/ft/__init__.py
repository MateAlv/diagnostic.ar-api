"""ft-v1.1: a residual over GenPhenia v3.0's gene scores that can only lower a gene.

Vendored from MateAlv/diagnostic.ar-model at ft-v1.1.0 (6bed09efe504f7150fb68b2734d28536cd5a6787),
src/diagnostic_ar_model/ft/__init__.py, without its default PACKAGE_DIR: the
service passes the directory it unpacked the release into. Do not edit here:
change it there, release, and copy it again.

The package is the release's ft-v1.1-confirmed.tar.gz: ft.json, ft_prod.py and
the bundle, each pinned by sha256 in ft.json. `load` checks both, imports
`ft_prod.py` from the package, and `apply` decides whether a request uses it.
It is off unless FT_V11_ENABLED says otherwise, and with no new input it
returns v3.0's distribution itself.
"""

import hashlib
import importlib.util
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

CONFIG = "ft.json"
FLAG = "FT_V11_ENABLED"
_TRUE = {"1", "true", "yes", "on"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(package_dir: Path) -> dict:
    return json.loads((Path(package_dir) / CONFIG).read_text())


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """FT_V11_ENABLED, off unless it is 1, true, yes or on."""
    environ = os.environ if environ is None else environ
    return environ.get(FLAG, "").strip().lower() in _TRUE


def _module_file(config: dict) -> str:
    (name,) = [n for n in config["files"] if n.endswith(".py")]
    return name


def _bundle_file(config: dict) -> str:
    (name,) = [n for n in config["files"] if n.endswith(".json.gz")]
    return name


def load_module(package_dir: Path) -> ModuleType:
    """`ft_prod.py` from the package, once its sha256 matches the pinned one."""
    package_dir = Path(package_dir)
    config = read_config(package_dir)
    name = _module_file(config)
    path = package_dir / name
    actual = sha256_file(path)
    if actual != config["files"][name]["sha256"]:
        raise ValueError(f"{path}: sha256 {actual}, expected {config['files'][name]['sha256']}")
    spec = importlib.util.spec_from_file_location("ft_prod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(package_dir: Path, genes: Sequence[str]) -> Any:
    """An `ft_prod.FtProd` over `genes`, v3.0's output order; refuses any file that is not the pinned one."""
    package_dir = Path(package_dir)
    config = read_config(package_dir)
    ft_prod = load_module(package_dir)
    name = _bundle_file(config)
    model = ft_prod.FtProd(ft_prod.load(package_dir / name, sha256=config["files"][name]["sha256"]))
    if model.genes != tuple(genes):
        raise ValueError(f"{name} does not follow v3.0's gene order ({len(model.genes)} vs {len(genes)} genes)")
    return model


def _years(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def case_inputs(
    sex: str | None = None,
    age_years: float | None = None,
    onset_years: float | None = None,
    excluded: Sequence[str] = (),
) -> dict:
    """The `case_inputs` ft_prod reads; a sex other than male or female, or a non-finite age, is unknown."""
    sex = (sex or "").strip().lower()
    return {
        "sex": sex if sex in ("male", "female") else None,
        "age_years": _years(age_years),
        "onset_years": _years(onset_years),
        "excluded": list(excluded or ()),
    }


def has_inputs(inputs: Mapping) -> bool:
    """Whether the case says anything the residual reads."""
    return bool(
        inputs.get("sex") in ("male", "female")
        or _years(inputs.get("age_years")) is not None
        or _years(inputs.get("onset_years")) is not None
        or inputs.get("excluded")
    )


def apply(model: Any, probs: Any, inputs: Mapping, on: bool) -> Any:
    """v3.0's probabilities with the residual, or `probs` itself when off or with nothing new."""
    if not on or model is None or not has_inputs(inputs):
        return probs
    return model.apply_probs(probs, inputs)
