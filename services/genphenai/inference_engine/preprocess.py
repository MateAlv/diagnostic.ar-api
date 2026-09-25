from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re

import numpy as np

from inference_engine.types import PreprocessedGraph

_HPO_ID_RE = re.compile(r"^HP:\d+$")


def normalize_hpo_id(hpo_id: str) -> str:
    value = hpo_id.strip()
    if "/" in value:
        value = value.split("/")[-1]
    if value.startswith("HP_"):
        value = value.replace("HP_", "HP:", 1)
    return value


def _parse_hpo_graph_json(hp_json_path: Path) -> dict[str, set[str]]:
    with hp_json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    parents_map: dict[str, set[str]] = {}

    if "graphs" in data:
        graph = data["graphs"][0]
        for node in graph.get("nodes", []):
            node_id = normalize_hpo_id(str(node.get("id", "")))
            if node_id.startswith("HP:"):
                parents_map.setdefault(node_id, set())

        for edge in graph.get("edges", []):
            pred = str(edge.get("pred", ""))
            pred_lower = pred.lower()
            if "subclassof" not in pred_lower and "is_a" not in pred_lower:
                continue
            child = normalize_hpo_id(str(edge.get("sub", "")))
            parent = normalize_hpo_id(str(edge.get("obj", "")))
            if not child.startswith("HP:") or not parent.startswith("HP:"):
                continue
            parents_map.setdefault(child, set()).add(parent)
            parents_map.setdefault(parent, set())
    elif "terms" in data:
        for term in data.get("terms", []):
            child = normalize_hpo_id(str(term.get("id", "")))
            if not child.startswith("HP:"):
                continue
            parents_map.setdefault(child, set())
            for parent_raw in term.get("is_a", []):
                if isinstance(parent_raw, dict):
                    parent_raw = parent_raw.get("id", "")
                parent = normalize_hpo_id(str(parent_raw).split(" ! ")[0])
                if parent.startswith("HP:"):
                    parents_map.setdefault(child, set()).add(parent)
                    parents_map.setdefault(parent, set())
    else:
        raise ValueError(
            f"Unsupported HPO JSON structure in {hp_json_path}. "
            "Expected keys: 'graphs' or 'terms'."
        )

    return parents_map


def build_compact_hpo_artifact(
    hp_json_path: Path,
    npz_out_path: Path,
    index_out_path: Path,
) -> None:
    parents_map = _parse_hpo_graph_json(hp_json_path)
    node_ids = sorted(parents_map.keys())
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    parent_offsets = [0]
    parent_indices: list[int] = []
    for node_id in node_ids:
        for parent_id in sorted(parents_map.get(node_id, set())):
            if parent_id in id_to_idx:
                parent_indices.append(id_to_idx[parent_id])
        parent_offsets.append(len(parent_indices))

    npz_out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_out_path,
        node_ids=np.asarray(node_ids, dtype="<U16"),
        parent_offsets=np.asarray(parent_offsets, dtype=np.int64),
        parent_indices=np.asarray(parent_indices, dtype=np.int64),
    )

    index_out_path.parent.mkdir(parents=True, exist_ok=True)
    with index_out_path.open("w", encoding="utf-8") as handle:
        json.dump(id_to_idx, handle, sort_keys=True)


@dataclass(frozen=True)
class HPODag:
    node_ids: tuple[str, ...]
    id_to_idx: dict[str, int]
    parent_offsets: np.ndarray
    parent_indices: np.ndarray

    @classmethod
    def from_compact(
        cls,
        npz_path: Path,
        index_json_path: Path | None = None,
    ) -> "HPODag":
        archive = np.load(npz_path, allow_pickle=False)
        node_ids = tuple(str(x) for x in archive["node_ids"].tolist())
        parent_offsets = archive["parent_offsets"].astype(np.int64, copy=False)
        parent_indices = archive["parent_indices"].astype(np.int64, copy=False)

        if index_json_path is not None and index_json_path.exists():
            with index_json_path.open("r", encoding="utf-8") as handle:
                id_to_idx = {str(k): int(v) for k, v in json.load(handle).items()}
        else:
            id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

        return cls(
            node_ids=node_ids,
            id_to_idx=id_to_idx,
            parent_offsets=parent_offsets,
            parent_indices=parent_indices,
        )

    @classmethod
    def from_parent_map(cls, parents_map: dict[str, set[str]]) -> "HPODag":
        node_ids = sorted(parents_map.keys())
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        parent_offsets = [0]
        parent_indices: list[int] = []
        for node_id in node_ids:
            for parent_id in sorted(parents_map.get(node_id, set())):
                if parent_id in id_to_idx:
                    parent_indices.append(id_to_idx[parent_id])
            parent_offsets.append(len(parent_indices))
        return cls(
            node_ids=tuple(node_ids),
            id_to_idx=id_to_idx,
            parent_offsets=np.asarray(parent_offsets, dtype=np.int64),
            parent_indices=np.asarray(parent_indices, dtype=np.int64),
        )

    def parents(self, hpo_id: str) -> tuple[str, ...]:
        node_id = normalize_hpo_id(hpo_id)
        idx = self.id_to_idx.get(node_id)
        if idx is None:
            return ()
        start = int(self.parent_offsets[idx])
        end = int(self.parent_offsets[idx + 1])
        parent_idx = self.parent_indices[start:end]
        return tuple(self.node_ids[int(i)] for i in parent_idx)

    def ancestors(self, hpo_id: str) -> set[str]:
        node_id = normalize_hpo_id(hpo_id)
        if node_id not in self.id_to_idx:
            return set()

        result: set[str] = set()
        queue: deque[str] = deque([node_id])
        seen = {node_id}

        while queue:
            current = queue.popleft()
            for parent_id in self.parents(current):
                if parent_id in seen:
                    continue
                seen.add(parent_id)
                result.add(parent_id)
                queue.append(parent_id)

        return result


class EmbeddingStore:
    def __init__(
        self,
        embeddings_data_path: Path,
        embeddings_index_path: Path,
        expected_dim: int | None = None,
        dtype: str = "float32",
    ) -> None:
        with embeddings_index_path.open("r", encoding="utf-8") as handle:
            raw_index = json.load(handle)
        self.index_map: dict[str, int] = {
            normalize_hpo_id(str(k)): int(v) for k, v in raw_index.items()
        }
        if not self.index_map:
            raise ValueError(f"Embedding index is empty: {embeddings_index_path}")
        if min(self.index_map.values()) < 0:
            raise ValueError(
                f"Embedding index contains negative row ids: {embeddings_index_path}"
            )

        raw_memmap = np.memmap(embeddings_data_path, dtype=dtype, mode="r")
        num_vectors = max(self.index_map.values()) + 1
        itemsize = int(np.dtype(dtype).itemsize)
        total_elems = int(raw_memmap.size)
        if total_elems == 0:
            raise ValueError(f"Embedding memmap is empty: {embeddings_data_path}")
        if total_elems % num_vectors != 0:
            raise ValueError(
                "Embedding memmap shape cannot be inferred from index cardinality. "
                f"elements={total_elems}, num_vectors={num_vectors}, "
                f"path={embeddings_data_path}"
            )
        inferred_dim = total_elems // num_vectors
        if inferred_dim <= 0:
            raise ValueError(
                f"Inferred non-positive embedding dim {inferred_dim} for {embeddings_data_path}"
            )
        if raw_memmap.nbytes != total_elems * itemsize:
            raise ValueError(
                f"Unexpected memmap byte size mismatch for {embeddings_data_path}"
            )

        if expected_dim is not None and expected_dim != inferred_dim:
            raise ValueError(
                f"Expected embedding dim {expected_dim}, but inferred {inferred_dim} "
                f"from {embeddings_data_path}"
            )

        self.dim = inferred_dim
        self.vectors = raw_memmap.reshape(num_vectors, self.dim)
        self.zero = np.zeros((self.dim,), dtype=np.float32)

    def lookup_many(self, hpo_ids: tuple[str, ...]) -> np.ndarray:
        out = np.zeros((len(hpo_ids), self.dim), dtype=np.float32)
        for i, hpo_id in enumerate(hpo_ids):
            idx = self.index_map.get(hpo_id)
            if idx is not None:
                out[i] = self.vectors[idx]
        return out


@dataclass(frozen=True)
class _CachedStructure:
    node_ids: tuple[str, ...]
    edge_index: np.ndarray


class GraphPreprocessor:
    def __init__(
        self,
        hpo_dag: HPODag,
        embeddings: EmbeddingStore,
        include_ancestors: bool = True,
        include_reverse_edges: bool = False,
        allow_unknown_ids: bool = False,
        cache_size: int = 4096,
    ) -> None:
        self.hpo_dag = hpo_dag
        self.embeddings = embeddings
        self.include_ancestors = include_ancestors
        self.include_reverse_edges = include_reverse_edges
        self.allow_unknown_ids = allow_unknown_ids
        self.cache_size = max(cache_size, 16)
        self._structure_cache: OrderedDict[
            tuple[str, ...], _CachedStructure
        ] = OrderedDict()

    def _canonicalize_hpo_ids(self, hpo_ids: list[str]) -> tuple[str, ...]:
        if not hpo_ids:
            raise ValueError("hpo_ids cannot be empty")

        seen: set[str] = set()
        ordered_ids: list[str] = []
        for raw_id in hpo_ids:
            normalized = normalize_hpo_id(raw_id)
            if not _HPO_ID_RE.match(normalized):
                raise ValueError(
                    f"Invalid HPO ID: {raw_id!r}. Expected format like 'HP:0001250'."
                )
            if not self.allow_unknown_ids and normalized not in self.hpo_dag.id_to_idx:
                raise ValueError(
                    f"Unknown HPO ID not found in local DAG: {normalized}. "
                    "Ensure IDs come from the same HPO release as hp.json."
                )
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered_ids.append(normalized)

        if not ordered_ids:
            raise ValueError("No valid HPO IDs were provided")

        return tuple(ordered_ids)

    def _build_structure(self, canonical_observed_ids: tuple[str, ...]) -> _CachedStructure:
        observed = list(canonical_observed_ids)
        observed_set = set(observed)

        expanded_set = set(observed)
        if self.include_ancestors:
            for hpo_id in observed:
                expanded_set.update(self.hpo_dag.ancestors(hpo_id))

        extra = sorted(expanded_set - observed_set)
        node_ids = tuple(observed + extra)
        id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

        edges: list[tuple[int, int]] = []
        for child_idx, child_id in enumerate(node_ids):
            for parent_id in self.hpo_dag.parents(child_id):
                parent_idx = id_to_idx.get(parent_id)
                if parent_idx is None:
                    continue
                edges.append((child_idx, parent_idx))
                if self.include_reverse_edges:
                    edges.append((parent_idx, child_idx))

        if edges:
            edge_index = np.asarray(edges, dtype=np.int64).T
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        return _CachedStructure(node_ids=node_ids, edge_index=edge_index)

    def _get_cached_structure(
        self,
        canonical_observed_ids: tuple[str, ...],
    ) -> _CachedStructure:
        cached = self._structure_cache.get(canonical_observed_ids)
        if cached is not None:
            self._structure_cache.move_to_end(canonical_observed_ids)
            return cached

        built = self._build_structure(canonical_observed_ids)
        self._structure_cache[canonical_observed_ids] = built
        self._structure_cache.move_to_end(canonical_observed_ids)
        if len(self._structure_cache) > self.cache_size:
            self._structure_cache.popitem(last=False)
        return built

    def preprocess(self, hpo_ids: list[str]) -> PreprocessedGraph:
        canonical_observed_ids = self._canonicalize_hpo_ids(hpo_ids)
        structure = self._get_cached_structure(canonical_observed_ids)

        x = self.embeddings.lookup_many(structure.node_ids)
        batch = np.zeros((len(structure.node_ids),), dtype=np.int64)
        return PreprocessedGraph(
            node_ids=structure.node_ids,
            x=x,
            edge_index=structure.edge_index.copy(),
            batch=batch,
        )

    def cache_stats(self) -> dict[str, int]:
        return {
            "size": len(self._structure_cache),
            "capacity": self.cache_size,
        }
