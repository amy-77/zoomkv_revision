# ZoomKV revision experiment snapshot

This repository is a source-only backup of the exact ZoomKV and ParisKV
runtime trees used for the VLDB revision experiments on `h20-9-57`, together
with the launchers, run parameters, and an auditable snapshot of the run-0
progress at the time the scheduler received SIGTERM.

The authoritative server source was copied from:

```text
/DATA/disk0/qyl/code/zoomkv_revision_migrated_20261008
```

The snapshot was refreshed and verified on 2026-10-12 at 14:47 China Standard
Time. Model weights, datasets, generated predictions, compiled extensions,
Python bytecode, caches, and machine-local environments are intentionally not
stored in Git.

## Layout

- `runtime/zoomkv_urania_runtime_20261011/`: ZoomKV implementation used by the
  current Urania-aligned experiment.
- `runtime/pariskv_urania_runtime_20261011/`: ParisKV implementation used by
  the same experiment.
- `orchestration/run_urania_correct_grid.py`: original 24-job, three-seed
  launcher.
- `orchestration/run_urania_run0_sharded.py`: two-shards-per-cell run-0
  launcher used on `h20-9-57`.
- `runtime/*/run/eval_math_aime.py`: exact evaluation entry points and CLI
  parameter definitions.
- `progress/`: original scheduler manifest/status/log and immutable progress
  snapshots.
- `scripts/snapshot_experiment_progress.py`: recomputes per-cell progress from
  base and shard JSONL files, deduplicating by qid.

## Experiment protocol

The launchers record the exact commands. The run-0 configuration used:

- models: Qwen3-4B-Thinking-2507 and Qwen3-8B;
- methods: ZoomKV and ParisKV;
- datasets: MATH-500 (`pass@1`) and AIME 2025 (`pass@8`);
- seed: 42 for run 0 (the full grid uses 42, 43, 44);
- temperature 0.6, top-p 0.95, top-k 20;
- maximum 38,912 new tokens;
- MATH: update interval 256 and full threshold 1,024;
- AIME: update interval 512, full threshold 2,048, stop after the first
  Urania-correct sample;
- Qwen3-8B YaRN factor 4; Qwen3-4B-Thinking-2507 YaRN factor 0;
- full predictions enabled.

The runner expects models below `<root>/models/` and the benchmark input below
`<root>/ZoomKV/data/math_aime/`. Absolute server paths in the archived
launchers reflect the original machine and should be adapted before reuse.

## Recompute a progress snapshot

Run this on a machine containing the result tree:

```bash
python scripts/snapshot_experiment_progress.py \
  /path/to/results/urania_correct_20261011 \
  --runs 0 \
  --scheduler-log /path/to/results/urania_correct_20261011/run0_sharded_scheduler.log \
  --output progress/run0_progress.json
```

The script visits the base JSONL first and then the shard JSONLs. It
deduplicates by `qid`, reports any conflicting overlaps, lists missing qid
ranges, and publishes an accuracy only for exact complete coverage. Accuracy
from an incomplete cell is labeled diagnostic.

See [progress/CURRENT_PROGRESS.md](progress/CURRENT_PROGRESS.md) for the
recorded state and [docs/SOURCE_PROVENANCE.md](docs/SOURCE_PROVENANCE.md) for
the backup provenance and exclusions.
