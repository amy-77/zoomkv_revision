# Source provenance

## Server source

- Host alias: `h20-9-57`
- Hostname observed in the progress snapshot:
  `bm-ksyun-qy-zonea-d-h20-141g-9-57`
- Source root:
  `/DATA/disk0/qyl/code/zoomkv_revision_migrated_20261008`
- Result root:
  `/DATA/disk0/qyl/code/zoomkv_revision_migrated_20261008/results/urania_correct_20261011`
- Runtime directories copied:
  `code/zoomkv_urania_runtime_20261011` and
  `code/pariskv_urania_runtime_20261011`
- Launchers copied from `code/` without modification.

The code and run metadata were copied from the server to the Mac on
2026-10-12. The corresponding local repository path at creation time was:

```text
/Users/qiyanlin/Desktop/ZoomKV/vldb_revision/VLDB_27_TokenTune/revision/zoomkv_revision_github
```

The previously existing Mac-side patch backup is:

```text
/Users/qiyanlin/Desktop/ZoomKV/vldb_revision/VLDB_27_TokenTune/revision/shard_accel_patch
```

The new progress collector and current JSON snapshot were also copied into
that existing backup directory.

## Deliberate exclusions

The source copy excludes `.git`, `__pycache__`, `*.pyc`, compiled `*.so` and
`*.o` files, Ninja build state, locks, model weights, datasets, generated
predictions, Triton/CUDA caches, and virtual environments. These are either
machine-specific artifacts, large data, or generated outputs. CUDA/C++ source
files, Python source, JSON model configs, codebooks, evaluation entry points,
and all run configuration scripts are retained.

The exact run commands as instantiated by the scheduler are preserved in
`progress/run0_sharded_manifest.json` and
`progress/run0_sharded_status_at_sigterm.json`.
