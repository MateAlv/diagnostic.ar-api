"""v4-layer: model-v4.0.0's post-processing over the v4 network's logits.

The package is the release's v4-layer.tar.gz (MateAlv/diagnostic.ar-model,
model-v4.0.0): v4.json, v4_layer.py and the bundle, the last two pinned by
sha256 in v4.json. `load` checks both, imports `v4_layer.py` from the package,
and refuses to run over any network but the one v4.json names, whose scores
the add-on was fitted on: the pinned tarball's sha256 and the model.onnx the
engine loaded must both be that network's.

It is on unless V4_LAYER_ENABLED says otherwise. Sex, age and onset are the
only inputs it reads; with none of them the service serves the network's own
softmax, and excluded terms change nothing (their weights are 0).
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

CONFIG = "v4.json"
FLAG = "V4_LAYER_ENABLED"
_FALSE = {"0", "false", "no", "off"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{path}: sha256 {actual}, expected {expected}")


def read_config(package_dir: Path) -> dict:
    return json.loads((Path(package_dir) / CONFIG).read_text())


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """V4_LAYER_ENABLED, on unless it is 0, false, no or off."""
    environ = os.environ if environ is None else environ
    return environ.get(FLAG, "").strip().lower() not in _FALSE


def bundle_file(config: dict) -> str:
    (name,) = [n for n in config["files"] if n.endswith(".json.gz")]
    return name


def _module_file(config: dict) -> str:
    (name,) = [n for n in config["files"] if n.endswith(".py")]
    return name


def check_base_model(config: dict, model_sha256: str, onnx_path: Path) -> None:
    """Refuse a network other than the one the layer was fitted on."""
    base = config["base_model"]
    if model_sha256 != base["sha256"]:
        raise ValueError(
            f"v4-layer was fitted on {base['release']}'s {base['asset']} ({base['sha256']}), "
            f"not on the pinned model ({model_sha256 or 'none'})"
        )
    check_file(onnx_path, base["model_onnx_sha256"])


def load_module(package_dir: Path) -> ModuleType:
    """`v4_layer.py` from the package, once its sha256 matches the pinned one."""
    package_dir = Path(package_dir)
    config = read_config(package_dir)
    name = _module_file(config)
    path = package_dir / name
    check_file(path, config["files"][name]["sha256"])
    spec = importlib.util.spec_from_file_location("v4_layer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(package_dir: Path, genes: Sequence[str], model_sha256: str, onnx_path: Path) -> Any:
    """A `v4_layer.V4Layer` over `genes`, the network's output order; refuses any file that is not the pinned one."""
    package_dir = Path(package_dir)
    config = read_config(package_dir)
    check_base_model(config, model_sha256, onnx_path)
    module = load_module(package_dir)
    name = bundle_file(config)
    layer = module.V4Layer(module.load(package_dir / name, sha256=config["files"][name]["sha256"]))
    if layer.genes != tuple(genes):
        raise ValueError(f"{name} does not follow the network's gene order ({len(layer.genes)} vs {len(genes)} genes)")
    return layer


def _years(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def case_inputs(sex: str | None = None, age_years: float | None = None, onset_years: float | None = None) -> dict:
    """The `case_inputs` v4_layer reads; a sex other than male or female, or a non-finite age, is unknown."""
    sex = (sex or "").strip().lower()
    return {
        "sex": sex if sex in ("male", "female") else None,
        "age_years": _years(age_years),
        "onset_years": _years(onset_years),
    }


def has_inputs(inputs: Mapping) -> bool:
    """Whether the case says anything the layer reads."""
    return (
        inputs.get("sex") in ("male", "female")
        or inputs.get("age_years") is not None
        or inputs.get("onset_years") is not None
    )
