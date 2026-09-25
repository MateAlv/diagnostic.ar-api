from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import shutil

from inference_engine.config import (
    REGISTRY_VERSION,
    artifacts_dir,
    checkpoints_dir,
    registry_path,
    state_dir,
)
from inference_engine.export_onnx import export_record_to_onnx
from inference_engine.preprocess import build_compact_hpo_artifact
from inference_engine.types import (
    CheckpointRecord,
    CheckpointSpec,
    RegistryState,
    sha256_file,
)


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve(path_value: str, relative_to: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (relative_to / path).resolve()


def _artifact_paths_exist(record: CheckpointRecord) -> bool:
    return (
        Path(record.onnx_path).exists()
        and Path(record.hpo_compact_npz_path).exists()
        and Path(record.hpo_compact_index_path).exists()
    )


def _load_genes(genes_path: Path) -> list[str]:
    payload = _load_json(genes_path)
    if isinstance(payload, list):
        genes = [str(x) for x in payload]
    elif isinstance(payload, dict) and isinstance(payload.get("genes"), list):
        genes = [str(x) for x in payload["genes"]]
    else:
        raise ValueError(
            f"genes_path must point to a JSON list or dict with 'genes' list: {genes_path}"
        )
    if not genes:
        raise ValueError(f"Gene list is empty in {genes_path}")
    return genes


class CheckpointRegistryManager:
    def __init__(self, bundle_dir: Path):
        self.bundle_dir = bundle_dir.resolve()
        self.checkpoints_dir = checkpoints_dir(self.bundle_dir)
        self.artifacts_dir = artifacts_dir(self.bundle_dir)
        self.state_dir = state_dir(self.bundle_dir)
        self.registry_path = registry_path(self.bundle_dir)

        self.init_layout()
        self.state = self._load_state()

    def init_layout(self) -> None:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> RegistryState:
        if not self.registry_path.exists():
            return RegistryState.empty(version=REGISTRY_VERSION)
        raw = _load_json(self.registry_path)
        state = RegistryState.from_dict(raw)
        if state.version != REGISTRY_VERSION:
            raise RuntimeError(
                f"Unsupported registry version {state.version}. "
                f"Expected {REGISTRY_VERSION}."
            )
        return state

    def save_state(self) -> None:
        self.state.dump_json(self.registry_path)

    def set_active(self, checkpoint_id: str) -> None:
        if checkpoint_id not in self.state.checkpoints:
            raise KeyError(f"Unknown checkpoint_id: {checkpoint_id}")
        self.state.active_checkpoint_id = checkpoint_id
        self.save_state()

    def get_active_record(self) -> CheckpointRecord:
        active_id = self.state.active_checkpoint_id
        if not active_id:
            raise RuntimeError("No active checkpoint configured")
        record = self.state.checkpoints.get(active_id)
        if record is None:
            raise RuntimeError(f"Active checkpoint {active_id!r} not found in registry")
        return record

    def list_records(self) -> list[CheckpointRecord]:
        return [
            self.state.checkpoints[key]
            for key in sorted(self.state.checkpoints.keys())
        ]

    def load_spec_file(self, spec_path: Path) -> CheckpointSpec:
        spec_raw = _load_json(spec_path)
        return CheckpointSpec.from_dict(spec_raw)

    def write_spec_file(self, package_dir: Path, spec: CheckpointSpec) -> Path:
        package_dir.mkdir(parents=True, exist_ok=True)
        spec_path = package_dir / "spec.json"
        with spec_path.open("w", encoding="utf-8") as handle:
            json.dump(spec.to_dict(), handle, indent=2, sort_keys=True)
        return spec_path

    def install_checkpoint_package(
        self,
        spec: CheckpointSpec,
        *,
        copy_checkpoint: bool = True,
    ) -> Path:
        package_dir = self.checkpoints_dir / spec.checkpoint_id
        package_dir.mkdir(parents=True, exist_ok=True)

        source_checkpoint = Path(spec.checkpoint_path).resolve()
        if copy_checkpoint:
            if not source_checkpoint.exists():
                raise FileNotFoundError(
                    f"checkpoint_path does not exist and cannot be copied: {source_checkpoint}"
                )
            target_checkpoint = package_dir / "model.pt"
            shutil.copy2(source_checkpoint, target_checkpoint)
            normalized_spec = replace(spec, checkpoint_path="model.pt")
        else:
            normalized_spec = spec

        return self.write_spec_file(package_dir, normalized_spec)

    def register_spec_file(
        self,
        spec_path: Path,
        *,
        activate: bool = False,
    ) -> CheckpointRecord:
        spec = self.load_spec_file(spec_path)
        existing = self.state.checkpoints.get(spec.checkpoint_id)
        record = self._build_record_from_spec(
            spec,
            spec_path=spec_path,
            existing=existing,
        )
        self.state.checkpoints[record.checkpoint_id] = record
        if activate or self.state.active_checkpoint_id is None:
            self.state.active_checkpoint_id = record.checkpoint_id
        self.save_state()
        return record

    def _build_record_from_spec(
        self,
        spec: CheckpointSpec,
        *,
        spec_path: Path,
        existing: CheckpointRecord | None = None,
    ) -> CheckpointRecord:
        package_dir = spec_path.parent

        checkpoint_path = _resolve(spec.checkpoint_path, package_dir)
        genes_path = _resolve(spec.genes_path, package_dir)
        hp_json_path = _resolve(spec.hp_json_path, package_dir)
        embeddings_data_path = _resolve(spec.embeddings_data_path, package_dir)
        embeddings_index_path = _resolve(spec.embeddings_index_path, package_dir)

        required_files = [
            checkpoint_path,
            genes_path,
            hp_json_path,
            embeddings_data_path,
            embeddings_index_path,
        ]
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Missing required file: {file_path}")

        checkpoint_stat = checkpoint_path.stat()
        checkpoint_sha = sha256_file(checkpoint_path)
        genes = _load_genes(genes_path)
        out_channels = len(genes)

        artifact_dir = self.artifacts_dir / spec.checkpoint_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if spec.onnx_path:
            onnx_path = _resolve(spec.onnx_path, package_dir)
        else:
            onnx_path = artifact_dir / "model.onnx"
        hpo_compact_npz_path = artifact_dir / "hpo_graph_compact.npz"
        hpo_compact_index_path = artifact_dir / "hpo_graph_index.json"

        build_compact_hpo_artifact(
            hp_json_path=hp_json_path,
            npz_out_path=hpo_compact_npz_path,
            index_out_path=hpo_compact_index_path,
        )

        provisional = CheckpointRecord(
            checkpoint_id=spec.checkpoint_id,
            checkpoint_path=str(checkpoint_path),
            model_version=spec.model_version,
            in_channels=spec.in_channels,
            hidden_channels=spec.hidden_channels,
            out_channels=out_channels,
            num_heads=spec.num_heads,
            genes_path=str(genes_path),
            hp_json_path=str(hp_json_path),
            embeddings_data_path=str(embeddings_data_path),
            embeddings_index_path=str(embeddings_index_path),
            include_ancestors=spec.include_ancestors,
            include_reverse_edges=spec.include_reverse_edges,
            checkpoint_key=spec.checkpoint_key,
            checkpoint_sha256=checkpoint_sha,
            checkpoint_size_bytes=int(checkpoint_stat.st_size),
            checkpoint_mtime_ns=int(checkpoint_stat.st_mtime_ns),
            onnx_path=str(onnx_path),
            hpo_compact_npz_path=str(hpo_compact_npz_path),
            hpo_compact_index_path=str(hpo_compact_index_path),
            spec_path=str(spec_path.resolve()),
            created_at_utc=_utc_now_iso(),
            enabled=True,
        )

        checkpoint_changed = (
            existing is not None
            and existing.checkpoint_sha256 != checkpoint_sha
        )
        use_prebuilt_onnx = False
        if onnx_path.exists():
            if existing is None or not checkpoint_changed:
                use_prebuilt_onnx = True
            else:
                use_prebuilt_onnx = (
                    int(onnx_path.stat().st_mtime_ns)
                    >= int(checkpoint_stat.st_mtime_ns)
                )

        if not use_prebuilt_onnx:
            try:
                export_record_to_onnx(provisional, output_onnx_path=onnx_path)
            except RuntimeError as exc:
                if onnx_path.exists() and checkpoint_changed:
                    raise RuntimeError(
                        "Checkpoint is newer than the ONNX artifact, and ONNX export "
                        "failed. Refresh model.onnx for this checkpoint or run "
                        "'uv sync --extra export'."
                    ) from exc
                raise
        return provisional

    def sync_checkpoint_packages(
        self,
        *,
        auto_activate_latest: bool = False,
    ) -> list[str]:
        changed: list[str] = []
        state_changed = False
        latest_changed_stamp = -1
        latest_changed_id: str | None = None

        spec_files = sorted(self.checkpoints_dir.glob("*/spec.json"))
        for spec_path in spec_files:
            spec = self.load_spec_file(spec_path)
            package_dir = spec_path.parent
            checkpoint_path = _resolve(spec.checkpoint_path, package_dir)

            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint path from spec does not exist: {checkpoint_path}"
                )

            checkpoint_stat = checkpoint_path.stat()
            current_size = int(checkpoint_stat.st_size)
            current_mtime_ns = int(checkpoint_stat.st_mtime_ns)
            existing = self.state.checkpoints.get(spec.checkpoint_id)
            has_valid_artifacts = (
                existing is not None
                and _artifact_paths_exist(existing)
            )
            same_stat = (
                existing is not None
                and existing.checkpoint_size_bytes == current_size
                and existing.checkpoint_mtime_ns == current_mtime_ns
            )
            if same_stat and has_valid_artifacts:
                continue

            current_sha = sha256_file(checkpoint_path)
            same_sha = (
                existing is not None
                and existing.checkpoint_sha256 == current_sha
            )
            if same_sha and has_valid_artifacts:
                if (
                    existing.checkpoint_size_bytes != current_size
                    or existing.checkpoint_mtime_ns != current_mtime_ns
                ):
                    self.state.checkpoints[spec.checkpoint_id] = replace(
                        existing,
                        checkpoint_size_bytes=current_size,
                        checkpoint_mtime_ns=current_mtime_ns,
                    )
                    state_changed = True
                continue

            record = self.register_spec_file(spec_path, activate=False)
            changed.append(record.checkpoint_id)

            changed_stamp = max(
                int(spec_path.stat().st_mtime_ns),
                current_mtime_ns,
            )
            if changed_stamp > latest_changed_stamp:
                latest_changed_stamp = changed_stamp
                latest_changed_id = record.checkpoint_id

        if auto_activate_latest and latest_changed_id is not None:
            self.state.active_checkpoint_id = latest_changed_id
            state_changed = True

        if state_changed:
            self.save_state()

        return changed
