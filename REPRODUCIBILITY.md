# Reproducibility

## Environment and assets

Use Python 3.10, the packages pinned in `requirements-frozen.txt`, one CUDA GPU,
and the official Matterport/DialNav assets. Clone RAINbow at commit
`4ef165fad77e675026958e16d9e516e523192a33`, then overlay the submitted files:

```bash
git clone https://github.com/happilee12/RAINbow.git external/RAINbow
git -C external/RAINbow checkout 4ef165fad77e675026958e16d9e516e523192a33
cp -a source_snapshot/external/RAINbow/. external/RAINbow/
```

Place the four checkpoints listed in `WEIGHTS_MANIFEST.json` under
`external/RAINbow/dataset/checkpoints/`. Install Matterport3DSimulator and set
`DIALNAV_MATTERSIM_BUILD` if its build directory is not
`external/Matterport3DSimulator/build`.

## Frozen execution

The run uses seed 0, batch size 8, at most 50 actions, WTA `ctc_0.7_2`, GTL
top-5, temporal weight 0.5, and answer-path limit 20. For one split:

```bash
DIALNAV_PYTHON=python \
  source_snapshot/scripts/run_causal_dialog_route_reranker.sh \
  causal_dialog_route_reranker_test 0 test /path/to/dialnav_test.json
```

Run `val_seen` and `val_unseen` identically with their official annotation
paths. `scripts/build_submission.py` validates IDs, fields, graph adjacency,
dialog indices, scans, and targets before combining the three split outputs.

## Checks

```bash
python source_snapshot/scripts/test_agent_boundary.py
python source_snapshot/scripts/test_submission_integrity.py
python source_snapshot/scripts/test_temporal_localization.py
python source_snapshot/scripts/test_temporal_gtl_rerank.py
bash -n source_snapshot/scripts/run_causal_dialog_route_reranker.sh
sha256sum -c CODE_CHECKSUMS.sha256
```

The supplied `submission.json` has 91/241/285 records and SHA256
`52c191922ff14423efe5b290c0a9e1aa0fa0e094c06227145db3a421fbbf5efc`.
Its Test slice is canonically identical to the saved single V6-A Test run.
