# GenPhenia model bundle

`/rank-genes` serves the GenPhenia v4 network (`genphenia-v4.0`, release
`model-v4.0.0`): E0's architecture — a 3-layer GIN over the patient's HPO
subgraph, BioBERT node embeddings (768 dimensions), 5,229 output genes —
retrained, with E0's exact input contract. It is loaded in process by
`app/main.py` and runs through its ONNX export on the CPU, followed by
v4-layer (below).

The bundles are not in this repository. They are releases of
`MateAlv/diagnostic.ar-model`, pinned by tag and sha256 in `../model.lock`:

```bash
services/genphenai/fetch_model.sh   # → model-dist/*.tar.gz, model/genphenia-v4.0/ and model/v4-layer/
```

The Dockerfile copies the tarballs from `model-dist/` and refuses them unless
the hashes match, so an image can only contain the pinned model.

| Path in the bundle | Used for |
|---|---|
| `checkpoints/best_model.pt` | The trained weights. Not read at run time, but the engine's registry requires it and records its hash |
| `checkpoints/graphens_best_model/spec.json` | Architecture and the paths below, relative to the spec |
| `checkpoints/graphens_best_model/genes.json` | Output index → gene symbol |
| `artifacts/graphens_best_model/model.onnx` | The weights exported to ONNX; what actually runs |
| `artifacts/graphens_best_model/hpo_graph_*` | The HPO DAG in compact form |
| `data/embeddings/*.mmap`, `*.index.json` | Node features (gsarti/biobert-nli, 2025-03-17) |
| `data/ontology/hp.json` | The HPO release the model was trained on |

**The engine reports `model_version` "3.0" for v4**: spec.json selects the
architecture, and the engine accepts only 2.0, 2.5 and 3.0. The tarball's
sha256 names the model, and `/healthz` reports it under `model`. Keep the
shipped `model.onnx` (`fafbecbe…`): the engine cannot rebuild it correctly,
for v4 or for v3.0. `best_model.pt` is `bff1a05b…`, the v4 network in E0's
checkpoint layout.

`../inference_engine/` is vendored from
`diagnostic.ar-model/vendor/genphenia_inference_engine/`; only the ONNX path
runs here, and torch is imported only by the export code, which never runs.
`../data/panelapp_panels.csv` maps genes to specialties; it comes from
`LUCAI-Bioinformatics/diagnosticar_tools` at `e328ba3`.

## v4-layer

`model.lock` also pins `v4-layer.tar.gz` of the same release: `v4.json`,
`v4_layer.py` and the bundle. On the network's logits, before the softmax, it
applies an add-on that can only lower a gene — a gene limited to the other
sex, an X-linked-recessive-only gene for a female, a gene whose earliest
onset is later than the patient's onset (or age) — and a hard rule that moves
the genes impossible for the patient's sex below every other gene: 101 for a
woman, 36 for a man.

- **Inputs.** The optional `sex`, `age_years` and `onset_years` of
  `/rank-genes`; the onset takes precedence over the age. `excluded_hpo_ids`
  is accepted and changes nothing (its weights are 0).
- **Loading.** Whenever it is pinned. `../v4layer/` refuses any file that does
  not match `v4.json`, a pinned model other than the one `v4.json` names, and a
  `model.onnx` that is not that network's; the service then does not start.
  The add-on was fitted on v4's scores: it must never run over E0.
- **The flag.** On unless `V4_LAYER_ENABLED=false`, which serves the network
  alone, with no redeploy. Without sex, age or onset the response is the
  network's own, bit for bit.
- **`/healthz`.** Reports it under `layer`, with the tarball's and the
  bundle's sha256 and whether it is enabled.
- **Responses.** `layer` names it when it was applied. `rerank` stays null:
  nothing uses the excluded terms, and the question agent only says its "no"
  answers moved the list when it is set.

What it does, its evidence and its limits are in diagnostic.ar-model's
`docs/model-v4.0.md`, and in diagnostic.ar-training's
`docs/HANDOFF-v4-prod.md`. `tests/data/v4_staging_fixture.json` is that
release's `staging-fixture.json` (`99d83c92…`): 32 cases with the network's
and v4-layer's expected top 10.

The re-ranker rr-v1 and ft-v1.1, which this service ran over v3.0, are gone:
both were fitted on v3.0's scores and lower v4 (ADR-0020 in
diagnostic.ar-training).

## Serving a new model

Publish it from diagnostic.ar-model, then update `model.lock`, the hashes
pinned in `tests/test_rank_genes.py`, and the fixture. A test that still
passes against the old numbers means the model did not change.

Going back to v3.0 is pinning `model-v3.0.0` again (tarball
`2201b7f2868afaf00439f76a6197bcf1d94eccc5ea171173719ca48c517a61c8`) without
`V4_LAYER_*`, or reverting the change that brought v4.
