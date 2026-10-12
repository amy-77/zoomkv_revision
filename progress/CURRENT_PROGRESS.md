# h20-9-57 run-0 progress at interruption

Last authoritative verification: **2026-10-12 14:47 CST**.

The scheduler log contains `[signal] received 15; terminating workers` at
approximately 11:50 CST. At verification time there were no run-0 worker
processes and all eight GPUs were idle. The JSONL files had not changed since
the interruption.

| Model | Method | Dataset | Deduplicated qids | Status | Accuracy |
|---|---|---:|---:|---|---:|
| Qwen3-4B-Thinking-2507 | ZoomKV | MATH-500 | 500 / 500 | complete | **92.80% publishable** |
| Qwen3-4B-Thinking-2507 | ZoomKV | AIME 2025 | 30 / 30 | complete | **73.33% publishable** |
| Qwen3-4B-Thinking-2507 | ParisKV | MATH-500 | 337 / 500 | stopped incomplete | 92.58% diagnostic |
| Qwen3-4B-Thinking-2507 | ParisKV | AIME 2025 | 29 / 30 | stopped incomplete | 75.86% diagnostic |
| Qwen3-8B | ZoomKV | MATH-500 | 143 / 500 | stopped incomplete | 90.21% diagnostic |
| Qwen3-8B | ZoomKV | AIME 2025 | 27 / 30 | stopped incomplete | 59.26% diagnostic |
| Qwen3-8B | ParisKV | MATH-500 | 101 / 500 | stopped incomplete | 87.13% diagnostic |
| Qwen3-8B | ParisKV | AIME 2025 | 17 / 30 | stopped incomplete | 47.06% diagnostic |

All eight cells had zero conflicting qid overlaps. The exact source-file
sizes, mtimes, SHA-256 values, missing qid ranges, and scheduler tail are in
`h20_9_57_run0_progress_current.json`.

Only the two complete cells are suitable for a paper table. The six partial
accuracies are diagnostic prefixes and must not be reported as final results.
