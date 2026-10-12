#!/usr/bin/env python3
"""Create an auditable progress snapshot for the Urania ZoomKV/ParisKV grid.

The collector reads both the original JSONL and any ``.shardN-of-M.jsonl``
files.  Records are deduplicated by qid, with shard records taking precedence
over the pre-sharding prefix.  Partial scores are explicitly diagnostic; only
complete qid coverage produces a publishable score.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
from pathlib import Path
import socket
from typing import Any, Iterable


MODELS = (
    ("qwen3_4b_thinking2507", "Qwen3-4B-Thinking-2507", "Qwen3-4B"),
    ("qwen3_8b", "Qwen3-8B", "Qwen3-8B"),
)
METHODS = ("zoomkv", "pariskv")
DATASETS = (
    ("math500", 500, "pass@1"),
    ("aime2025", 30, "pass@8"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict) or "qid" not in value:
                    malformed += 1
                    continue
                rows.append(value)
            except (json.JSONDecodeError, TypeError, ValueError):
                # An active worker can be between writes at snapshot time.
                malformed += 1
    return rows, malformed


def source_record(root: Path, path: Path, row_count: int, malformed: int) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(root)),
        "bytes": stat.st_size,
        "sha256": sha256(path),
        "mtime_utc": dt.datetime.fromtimestamp(
            stat.st_mtime, dt.timezone.utc
        ).isoformat(),
        "json_rows": row_count,
        "malformed_lines_skipped": malformed,
    }


def result_signature(row: dict[str, Any]) -> str:
    samples = row.get("samples", [])
    selected = {
        "pass_at_k": row.get("pass_at_k"),
        "n_attempted": row.get("n_attempted"),
        "samples": [
            {
                "generation_seed": sample.get("generation_seed"),
                "correct": sample.get("correct"),
                "boxed": sample.get("boxed"),
            }
            for sample in samples
            if isinstance(sample, dict)
        ],
    }
    return json.dumps(selected, sort_keys=True, ensure_ascii=False)


def compress_ranges(values: Iterable[int]) -> list[str]:
    ordered = sorted(set(values))
    if not ordered:
        return []
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ranges


def scheduler_observation(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "path": str(path) if path else None,
            "exists": False,
            "sigterm_observed": False,
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    stat = path.stat()
    signal_lines = [line for line in text.splitlines() if "[signal]" in line]
    return {
        "path": str(path),
        "exists": True,
        "bytes": stat.st_size,
        "sha256": sha256(path),
        "mtime_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
        "sigterm_observed": any("received 15" in line for line in signal_lines),
        "signal_lines": signal_lines,
        "tail": text.splitlines()[-24:],
    }


def collect_cell(
    root: Path,
    model_dir: str,
    model_name: str,
    file_tag: str,
    method: str,
    dataset: str,
    expected: int,
    metric: str,
    run_id: int,
    stopped: bool,
) -> dict[str, Any]:
    directory = root / model_dir / method / dataset
    stem = f"{file_tag}_{dataset}_{method}_run{run_id}"
    paths = [directory / f"{stem}.jsonl"]
    paths.extend(
        Path(value)
        for value in sorted(glob.glob(str(directory / f"{stem}.shard*-of-*.jsonl")))
    )
    paths = [path for path in paths if path.is_file()]

    chosen: dict[int, dict[str, Any]] = {}
    signatures: dict[int, set[str]] = {}
    conflicts: set[int] = set()
    sources: list[dict[str, Any]] = []
    for path in paths:
        rows, malformed = read_jsonl(path)
        sources.append(source_record(root, path, len(rows), malformed))
        for row in rows:
            qid = int(row["qid"])
            signature = result_signature(row)
            if qid in signatures and signature not in signatures[qid]:
                conflicts.add(qid)
            signatures.setdefault(qid, set()).add(signature)
            # Base is visited first; a current shard intentionally supersedes
            # an overlapping prefix record.
            chosen[qid] = row

    expected_qids = set(range(expected))
    observed_qids = set(chosen)
    missing = expected_qids - observed_qids
    unexpected = observed_qids - expected_qids
    done = len(observed_qids & expected_qids)
    correct = sum(float(chosen[qid].get("pass_at_k", 0.0)) for qid in observed_qids & expected_qids)
    complete = done == expected and not unexpected and not conflicts
    diagnostic_accuracy = 100.0 * correct / done if done else None
    if complete:
        status = "complete"
    elif not paths:
        status = "not_started"
    elif stopped:
        status = "stopped_incomplete"
    else:
        status = "running"

    return {
        "model": model_name,
        "method": method,
        "dataset": dataset,
        "metric": metric,
        "run_id": run_id,
        "seed": 42 + run_id,
        "status": status,
        "unique_expected_qids": done,
        "expected_qids": expected,
        "coverage_percent": 100.0 * done / expected,
        "missing_qid_count": len(missing),
        "missing_qid_ranges": compress_ranges(missing),
        "unexpected_qids": sorted(unexpected),
        "correct_problem_count": correct,
        "diagnostic_accuracy_percent": diagnostic_accuracy,
        "publishable_accuracy_percent": diagnostic_accuracy if complete else None,
        "overlapping_qids_with_conflicting_records": sorted(conflicts),
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="urania_correct_YYYYMMDD result root")
    parser.add_argument("--runs", default="0", help="comma-separated run ids")
    parser.add_argument("--scheduler-log", type=Path)
    parser.add_argument("--output", type=Path, help="write atomically instead of stdout")
    args = parser.parse_args()

    root = args.root.resolve()
    scheduler = scheduler_observation(args.scheduler_log)
    stopped = bool(scheduler.get("sigterm_observed"))
    run_ids = [int(value) for value in args.runs.split(",") if value.strip()]
    cells = [
        collect_cell(
            root,
            model_dir,
            model_name,
            file_tag,
            method,
            dataset,
            expected,
            metric,
            run_id,
            stopped,
        )
        for model_dir, model_name, file_tag in MODELS
        for method in METHODS
        for dataset, expected, metric in DATASETS
        for run_id in run_ids
    ]
    status_counts: dict[str, int] = {}
    for cell in cells:
        status = str(cell["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    output = {
        "schema_version": 2,
        "snapshot_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_host": socket.gethostname(),
        "source_root": str(root),
        "selection_rule": (
            "Deduplicate by qid; shard records supersede overlapping base records. "
            "Only exact expected-qid coverage without conflicts is publishable; "
            "partial accuracy is diagnostic."
        ),
        "summary": {"cells": len(cells), "status_counts": status_counts},
        "scheduler": scheduler,
        "cells": cells,
    }
    rendered = json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
