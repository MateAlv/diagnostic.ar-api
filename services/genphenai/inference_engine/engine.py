from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

import numpy as np

from inference_engine.checkpoint_registry import CheckpointRegistryManager
from inference_engine.preprocess import EmbeddingStore, GraphPreprocessor, HPODag
from inference_engine.runtime_onnx import ONNXRuntimeModel
from inference_engine.types import CheckpointRecord


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64, copy=False)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    denom = np.sum(exp)
    if denom <= 0:
        return np.zeros_like(logits, dtype=np.float64)
    return exp / denom


def _load_genes(genes_path: Path) -> list[str]:
    with genes_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        genes = [str(x) for x in payload]
    elif isinstance(payload, dict) and isinstance(payload.get("genes"), list):
        genes = [str(x) for x in payload["genes"]]
    else:
        raise ValueError(
            f"Invalid genes payload in {genes_path}. Expected list or dict with 'genes'."
        )
    if not genes:
        raise ValueError(f"Gene list is empty in {genes_path}")
    return genes


class GenpheniaEngine:
    def __init__(
        self,
        manager: CheckpointRegistryManager,
        record: CheckpointRecord,
        genes: list[str],
        preprocessor: GraphPreprocessor,
        runtime: ONNXRuntimeModel,
        *,
        auto_refresh_seconds: float = 5.0,
        auto_activate_latest: bool = True,
    ) -> None:
        self.manager = manager
        self.record = record
        self.genes = genes
        self.preprocessor = preprocessor
        self.runtime = runtime
        self.active_checkpoint_id = record.checkpoint_id
        self.auto_refresh_seconds = max(0.0, auto_refresh_seconds)
        self.auto_activate_latest = auto_activate_latest
        self._last_refresh_ts = 0.0

    @classmethod
    def load(
        cls,
        bundle_dir: str | Path,
        *,
        auto_refresh_seconds: float = 5.0,
        auto_activate_latest: bool = True,
    ) -> "GenpheniaEngine":
        manager = CheckpointRegistryManager(Path(bundle_dir))
        manager.sync_checkpoint_packages(auto_activate_latest=auto_activate_latest)
        record = manager.get_active_record()
        genes, preprocessor, runtime = cls._build_runtime_components(record)
        return cls(
            manager=manager,
            record=record,
            genes=genes,
            preprocessor=preprocessor,
            runtime=runtime,
            auto_refresh_seconds=auto_refresh_seconds,
            auto_activate_latest=auto_activate_latest,
        )

    @staticmethod
    def _build_runtime_components(
        record: CheckpointRecord,
    ) -> tuple[list[str], GraphPreprocessor, ONNXRuntimeModel]:
        genes = _load_genes(Path(record.genes_path))
        dag = HPODag.from_compact(
            npz_path=Path(record.hpo_compact_npz_path),
            index_json_path=Path(record.hpo_compact_index_path),
        )
        embeddings = EmbeddingStore(
            embeddings_data_path=Path(record.embeddings_data_path),
            embeddings_index_path=Path(record.embeddings_index_path),
            expected_dim=record.in_channels,
        )
        preprocessor = GraphPreprocessor(
            hpo_dag=dag,
            embeddings=embeddings,
            include_ancestors=record.include_ancestors,
            include_reverse_edges=record.include_reverse_edges,
            allow_unknown_ids=False,
        )
        runtime = ONNXRuntimeModel(Path(record.onnx_path))
        return genes, preprocessor, runtime

    def _maybe_refresh_registry_and_runtime(self) -> None:
        now = time.monotonic()
        if now - self._last_refresh_ts < self.auto_refresh_seconds:
            return
        self._last_refresh_ts = now

        self.manager.sync_checkpoint_packages(
            auto_activate_latest=self.auto_activate_latest
        )
        latest_record = self.manager.get_active_record()
        if latest_record.checkpoint_id == self.active_checkpoint_id:
            return

        genes, preprocessor, runtime = self._build_runtime_components(latest_record)
        self.record = latest_record
        self.genes = genes
        self.preprocessor = preprocessor
        self.runtime = runtime
        self.active_checkpoint_id = latest_record.checkpoint_id

    def predict(
        self,
        hpo_ids: list[str],
        top_k: int = 60,
        return_distribution: bool = False,
    ) -> dict[str, Any]:
        self._maybe_refresh_registry_and_runtime()

        graph = self.preprocessor.preprocess(hpo_ids)
        logits = self.runtime.predict_logits(graph)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        if logits.shape[0] != 1:
            raise RuntimeError(
                f"Expected single-graph output [1, C], got {tuple(logits.shape)}"
            )

        probabilities = _softmax(logits[0])
        k = max(1, min(int(top_k), len(self.genes)))
        top_indices = np.argsort(probabilities)[-k:][::-1]
        predictions = [
            {
                "rank": rank + 1,
                "gene_index": int(gene_idx),
                "gene": self.genes[int(gene_idx)],
                "probability": float(probabilities[int(gene_idx)]),
            }
            for rank, gene_idx in enumerate(top_indices.tolist())
        ]

        result: dict[str, Any] = {
            "active_checkpoint_id": self.active_checkpoint_id,
            "model_version": self.record.model_version,
            "input_hpo_ids": list(hpo_ids),
            "expanded_hpo_ids": list(graph.node_ids),
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "top_predictions": predictions,
        }
        if return_distribution:
            result["distribution"] = {
                gene: float(probabilities[i])
                for i, gene in enumerate(self.genes)
            }
        return result

    def health(self) -> dict[str, Any]:
        self._maybe_refresh_registry_and_runtime()
        return {
            "active_checkpoint_id": self.active_checkpoint_id,
            "model_version": self.record.model_version,
            "out_channels": self.record.out_channels,
            "in_channels": self.record.in_channels,
            "registry_path": str(self.manager.registry_path),
            "cache": self.preprocessor.cache_stats(),
            "runtime": self.runtime.health(),
        }
