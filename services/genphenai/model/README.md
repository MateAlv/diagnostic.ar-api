# GenPhenia model bundle

`/rank-genes` serves the GenPhenia v3.0 network: a 3-layer GIN over the
patient's HPO subgraph, BioBERT node embeddings (768 dimensions), 5,229 output
genes. It is loaded in process by `app/main.py` and runs through its ONNX
export on the CPU.

The bundle is not in this repository. It is a release of
`MateAlv/diagnostic.ar-model`, pinned by tag and sha256 in `../model.lock`:

```bash
services/genphenai/fetch_model.sh   # → model-dist/*.tar.gz and model/genphenia-v3.0/
```

The Dockerfile copies the tarball from `model-dist/` and refuses it unless the
hash matches, so an image can only contain the pinned model.

| Path in the bundle | Used for |
|---|---|
| `checkpoints/best_model.pt` | The trained weights. Not read at run time, but the engine's registry requires it and records its hash |
| `checkpoints/graphens_best_model/spec.json` | Architecture and the paths below, relative to the spec |
| `checkpoints/graphens_best_model/genes.json` | Output index → gene symbol |
| `artifacts/graphens_best_model/model.onnx` | The weights exported to ONNX; what actually runs |
| `artifacts/graphens_best_model/hpo_graph_*` | The HPO DAG in compact form |
| `data/embeddings/*.mmap`, `*.index.json` | Node features (gsarti/biobert-nli, 2025-03-17) |
| `data/ontology/hp.json` | The HPO release the model was trained on |

`model-v3.0.0`'s `best_model.pt` is `ee10aa42…73955`, the checkpoint brainy's
inference server registered on 2026-02-20; its rankings were checked against
brainy's on 2026-09-25 (`tests/data/reference_rankings.json`).

`../inference_engine/` is vendored from
`diagnostic.ar-model/vendor/genphenia_inference_engine/`; only the ONNX path
runs here, and torch is imported only by the export code, which never runs.
`../data/panelapp_panels.csv` maps genes to specialties; it comes from
`LUCAI-Bioinformatics/diagnosticar_tools` at `e328ba3`.

## Serving a new model

Publish it from diagnostic.ar-model, then update `model.lock`, the checkpoint
hash pinned in `tests/test_rank_genes.py`, and the reference rankings. A test
that still passes against the old numbers means the model did not change.
