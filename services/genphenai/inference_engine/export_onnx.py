from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from inference_engine.types import CheckpointRecord


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for checkpoint export. "
            "Install with: uv sync --extra export"
        ) from exc
    return torch


def _load_raw_state_dict(
    checkpoint_path: Path,
    checkpoint_key: str,
) -> dict[str, Any]:
    torch = _require_torch()
    payload = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(payload, dict) and checkpoint_key in payload:
        state = payload[checkpoint_key]
    elif isinstance(payload, dict):
        maybe_tensor_values = [v for v in payload.values() if hasattr(v, "shape")]
        if maybe_tensor_values:
            state = payload
        else:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has no '{checkpoint_key}' entry and "
                "does not look like a raw state_dict."
            )
    else:
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)}")

    if not isinstance(state, dict):
        raise ValueError("Loaded checkpoint state is not a dictionary")
    return state


def export_record_to_onnx(
    record: CheckpointRecord,
    output_onnx_path: Path,
    opset_version: int = 17,
    validate_parity: bool = True,
    max_abs_tol: float = 1e-4,
    mean_abs_tol: float = 1e-5,
    parity_trials: int = 10,
) -> None:
    torch = _require_torch()
    from inference_engine.adapter_models import (
        build_adapter_model,
        convert_checkpoint_state_dict,
    )

    model = build_adapter_model(
        model_version=record.model_version,
        in_channels=record.in_channels,
        hidden_channels=record.hidden_channels,
        out_channels=record.out_channels,
        num_heads=record.num_heads,
    )
    model.eval()

    raw_state = _load_raw_state_dict(
        checkpoint_path=Path(record.checkpoint_path),
        checkpoint_key=record.checkpoint_key,
    )
    converted_state = convert_checkpoint_state_dict(raw_state)

    load_result = model.load_state_dict(converted_state, strict=False)
    if load_result.missing_keys:
        raise RuntimeError(
            "Missing keys when loading converted state_dict into adapter model: "
            f"{load_result.missing_keys}"
        )

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_nodes = 16
    dummy_edges = 32
    x = torch.randn(dummy_nodes, record.in_channels, dtype=torch.float32)
    edge_index = torch.randint(
        low=0, high=dummy_nodes, size=(2, dummy_edges), dtype=torch.long
    )
    batch = torch.zeros(dummy_nodes, dtype=torch.long)

    torch.onnx.export(
        model,
        (x, edge_index, batch),
        str(output_onnx_path),
        opset_version=opset_version,
        input_names=["x", "edge_index", "batch"],
        output_names=["logits"],
        dynamic_axes={
            "x": {0: "num_nodes"},
            "edge_index": {1: "num_edges"},
            "batch": {0: "num_nodes"},
            "logits": {0: "batch_size"},
        },
        do_constant_folding=True,
    )

    if validate_parity:
        _validate_onnx_parity(
            model=model,
            onnx_path=output_onnx_path,
            in_channels=record.in_channels,
            max_abs_tol=max_abs_tol,
            mean_abs_tol=mean_abs_tol,
            trials=parity_trials,
        )


def _validate_onnx_parity(
    model: Any,
    onnx_path: Path,
    in_channels: int,
    max_abs_tol: float,
    mean_abs_tol: float,
    trials: int,
) -> None:
    torch = _require_torch()
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required to validate export parity. "
            "Install with: uv sync --extra export"
        ) from exc

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_abs_seen = 0.0
    mean_abs_seen = 0.0

    rng = np.random.default_rng(seed=42)
    model.eval()

    for _ in range(trials):
        num_nodes = int(rng.integers(8, 64))
        num_edges = int(rng.integers(num_nodes, max(num_nodes * 3, num_nodes + 1)))
        x = torch.randn(num_nodes, in_channels, dtype=torch.float32)
        edge_index = torch.randint(
            low=0,
            high=num_nodes,
            size=(2, num_edges),
            dtype=torch.long,
        )
        batch = torch.zeros(num_nodes, dtype=torch.long)

        with torch.no_grad():
            torch_logits = model(x, edge_index, batch).cpu().numpy()
        ort_logits = session.run(
            None,
            {
                "x": x.cpu().numpy().astype(np.float32, copy=False),
                "edge_index": edge_index.cpu().numpy().astype(np.int64, copy=False),
                "batch": batch.cpu().numpy().astype(np.int64, copy=False),
            },
        )[0]

        diff = np.abs(torch_logits - ort_logits)
        max_abs_seen = max(max_abs_seen, float(np.max(diff)))
        mean_abs_seen = max(mean_abs_seen, float(np.mean(diff)))

    if max_abs_seen > max_abs_tol or mean_abs_seen > mean_abs_tol:
        raise RuntimeError(
            "ONNX parity check failed. "
            f"max_abs={max_abs_seen:.6g} (tol={max_abs_tol}), "
            f"mean_abs={mean_abs_seen:.6g} (tol={mean_abs_tol})"
        )
