from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from inference_engine.types import PreprocessedGraph


class ONNXRuntimeModel:
    def __init__(
        self,
        onnx_path: Path,
        providers: list[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for ONNX inference. "
                "Install with: uv sync --extra runtime"
            ) from exc

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self._ort = ort
        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_names = [entry.name for entry in self.session.get_inputs()]
        self.output_names = [entry.name for entry in self.session.get_outputs()]

    def predict_logits(self, graph: PreprocessedGraph) -> np.ndarray:
        inputs: dict[str, Any] = {}
        if "x" in self.input_names:
            inputs["x"] = graph.x.astype(np.float32, copy=False)
        if "edge_index" in self.input_names:
            inputs["edge_index"] = graph.edge_index.astype(np.int64, copy=False)
        if "batch" in self.input_names:
            inputs["batch"] = graph.batch.astype(np.int64, copy=False)

        outputs = self.session.run(None, inputs)
        if not outputs:
            raise RuntimeError("ONNX runtime returned no outputs")
        return np.asarray(outputs[0], dtype=np.float32)

    def health(self) -> dict[str, Any]:
        return {
            "onnx_path": str(self.onnx_path),
            "providers": self.session.get_providers(),
            "inputs": self.input_names,
            "outputs": self.output_names,
        }
