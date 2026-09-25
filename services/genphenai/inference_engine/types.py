from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, cast
import hashlib
import json
import re

import numpy as np

ModelVersion = Literal["2.0", "2.5", "3.0"]

_CHECKPOINT_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+$")
_MODEL_VERSIONS = {"2.0", "2.5", "3.0"}


def normalize_model_version(value: str) -> ModelVersion:
    normalized = str(value).strip().lower()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    if normalized not in _MODEL_VERSIONS:
        raise ValueError(
            f"Unsupported model_version: {value!r}. "
            "Expected one of: 2.0, 2.5, 3.0 (or prefixed forms v2.0/v2.5/v3.0)."
        )
    return cast(ModelVersion, normalized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


@dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_id: str
    checkpoint_path: str
    model_version: ModelVersion
    in_channels: int
    hidden_channels: int
    num_heads: int
    genes_path: str
    hp_json_path: str
    embeddings_data_path: str
    embeddings_index_path: str
    onnx_path: Optional[str] = None
    include_ancestors: bool = True
    include_reverse_edges: bool = False
    checkpoint_key: str = "model_state_dict"

    def validate(self) -> None:
        if self.model_version not in _MODEL_VERSIONS:
            raise ValueError(
                f"Unsupported model_version: {self.model_version!r}. "
                "Expected one of: 2.0, 2.5, 3.0."
            )
        if not _CHECKPOINT_ID_RE.match(self.checkpoint_id):
            raise ValueError(
                "checkpoint_id must match [a-zA-Z0-9._-], got "
                f"{self.checkpoint_id!r}"
            )
        _require_positive_int("in_channels", self.in_channels)
        _require_positive_int("hidden_channels", self.hidden_channels)
        _require_positive_int("num_heads", self.num_heads)

        if self.model_version != "2.5" and self.num_heads != 4:
            # Keep defaults explicit to avoid silent confusion.
            raise ValueError(
                "num_heads is only used for model_version 2.5 and must remain 4 "
                "for 2.0/3.0 specs"
            )

        if not self.checkpoint_key:
            raise ValueError("checkpoint_key cannot be empty")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CheckpointSpec":
        spec = cls(
            checkpoint_id=raw["checkpoint_id"],
            checkpoint_path=raw["checkpoint_path"],
            onnx_path=raw.get("onnx_path"),
            model_version=normalize_model_version(raw["model_version"]),
            in_channels=int(raw["in_channels"]),
            hidden_channels=int(raw["hidden_channels"]),
            num_heads=int(raw.get("num_heads", 4)),
            genes_path=raw["genes_path"],
            hp_json_path=raw["hp_json_path"],
            embeddings_data_path=raw["embeddings_data_path"],
            embeddings_index_path=raw["embeddings_index_path"],
            include_ancestors=bool(raw.get("include_ancestors", True)),
            include_reverse_edges=bool(raw.get("include_reverse_edges", False)),
            checkpoint_key=str(raw.get("checkpoint_key", "model_state_dict")),
        )
        spec.validate()
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_path": self.checkpoint_path,
            "onnx_path": self.onnx_path,
            "model_version": self.model_version,
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "num_heads": self.num_heads,
            "genes_path": self.genes_path,
            "hp_json_path": self.hp_json_path,
            "embeddings_data_path": self.embeddings_data_path,
            "embeddings_index_path": self.embeddings_index_path,
            "include_ancestors": self.include_ancestors,
            "include_reverse_edges": self.include_reverse_edges,
            "checkpoint_key": self.checkpoint_key,
        }


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_id: str
    checkpoint_path: str
    model_version: ModelVersion
    in_channels: int
    hidden_channels: int
    out_channels: int
    num_heads: int
    genes_path: str
    hp_json_path: str
    embeddings_data_path: str
    embeddings_index_path: str
    include_ancestors: bool
    include_reverse_edges: bool
    checkpoint_key: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    checkpoint_mtime_ns: int
    onnx_path: str
    hpo_compact_npz_path: str
    hpo_compact_index_path: str
    spec_path: str
    created_at_utc: str
    enabled: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CheckpointRecord":
        return cls(
            checkpoint_id=raw["checkpoint_id"],
            checkpoint_path=raw["checkpoint_path"],
            model_version=normalize_model_version(raw["model_version"]),
            in_channels=int(raw["in_channels"]),
            hidden_channels=int(raw["hidden_channels"]),
            out_channels=int(raw["out_channels"]),
            num_heads=int(raw["num_heads"]),
            genes_path=raw["genes_path"],
            hp_json_path=raw["hp_json_path"],
            embeddings_data_path=raw["embeddings_data_path"],
            embeddings_index_path=raw["embeddings_index_path"],
            include_ancestors=bool(raw["include_ancestors"]),
            include_reverse_edges=bool(raw["include_reverse_edges"]),
            checkpoint_key=raw["checkpoint_key"],
            checkpoint_sha256=raw["checkpoint_sha256"],
            checkpoint_size_bytes=int(raw.get("checkpoint_size_bytes", 0)),
            checkpoint_mtime_ns=int(raw.get("checkpoint_mtime_ns", 0)),
            onnx_path=raw["onnx_path"],
            hpo_compact_npz_path=raw["hpo_compact_npz_path"],
            hpo_compact_index_path=raw["hpo_compact_index_path"],
            spec_path=raw["spec_path"],
            created_at_utc=raw["created_at_utc"],
            enabled=bool(raw.get("enabled", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_path": self.checkpoint_path,
            "model_version": self.model_version,
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "out_channels": self.out_channels,
            "num_heads": self.num_heads,
            "genes_path": self.genes_path,
            "hp_json_path": self.hp_json_path,
            "embeddings_data_path": self.embeddings_data_path,
            "embeddings_index_path": self.embeddings_index_path,
            "include_ancestors": self.include_ancestors,
            "include_reverse_edges": self.include_reverse_edges,
            "checkpoint_key": self.checkpoint_key,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "checkpoint_mtime_ns": self.checkpoint_mtime_ns,
            "onnx_path": self.onnx_path,
            "hpo_compact_npz_path": self.hpo_compact_npz_path,
            "hpo_compact_index_path": self.hpo_compact_index_path,
            "spec_path": self.spec_path,
            "created_at_utc": self.created_at_utc,
            "enabled": self.enabled,
        }


@dataclass
class RegistryState:
    version: int
    active_checkpoint_id: Optional[str]
    checkpoints: dict[str, CheckpointRecord] = field(default_factory=dict)

    @classmethod
    def empty(cls, version: int) -> "RegistryState":
        return cls(version=version, active_checkpoint_id=None, checkpoints={})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RegistryState":
        checkpoints_raw = raw.get("checkpoints", {})
        checkpoints = {
            ckpt_id: CheckpointRecord.from_dict(record_raw)
            for ckpt_id, record_raw in checkpoints_raw.items()
        }
        return cls(
            version=int(raw["version"]),
            active_checkpoint_id=raw.get("active_checkpoint_id"),
            checkpoints=checkpoints,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "active_checkpoint_id": self.active_checkpoint_id,
            "checkpoints": {
                ckpt_id: record.to_dict()
                for ckpt_id, record in sorted(self.checkpoints.items())
            },
        }

    def dump_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)


@dataclass(frozen=True)
class PreprocessedGraph:
    node_ids: tuple[str, ...]
    x: np.ndarray
    edge_index: np.ndarray
    batch: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        if self.edge_index.shape[0] != 2:
            return 0
        return int(self.edge_index.shape[1])
