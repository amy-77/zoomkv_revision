#!/usr/bin/env python3
"""Independent queued verification of the fully Urania-aligned grid."""
import datetime as dt
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results/urania_correct_20261011"
OUT = BASE / "verification"
sys.path.insert(0, str(ROOT / "code/zoomkv_urania_runtime_20261011/run"))
from urania_rescore_queue import (  # noqa: E402
    aime_extract, compare_answers, math_extract, math_verify_score, truncate_math,
)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def targets():
    result = []
    for model_label, model_tag in (("qwen3_4b_thinking2507", "Qwen3-4B"), ("qwen3_8b", "Qwen3-8B")):
        for method in ("zoomkv", "pariskv"):
            for dataset, expected in (("aime2025", 30), ("math500", 500)):
                for run in range(3):
                    path = BASE / model_label / method / dataset / f"{model_tag}_{dataset}_{method}_run{run}.jsonl"
                    result.append({"model": model_label, "method": method, "dataset": dataset,
                                   "run": run, "expected": expected, "path": path})
    return result


def score_sample(dataset, pred, gold):
    if dataset == "aime2025":
        extracted = aime_extract(pred)
        return extracted, compare_answers(extracted, gold)
    clean = truncate_math(pred)
    extracted = math_extract(clean)
    return extracted, math_verify_score(gold, clean, extracted)


def verify_target(target):
    source_rows = rows(target["path"])
    out_dir = OUT / target["model"] / target["method"] / target["dataset"]
    out_dir.mkdir(parents=True, exist_ok=True)
    diffs = []
    verified = []
    correct = 0
    for row in source_rows:
        samples = []
        for sample in row.get("samples", []):
            if "pred" not in sample:
                raise RuntimeError(f"missing pred: {target['path']} qid={row['qid']}")
            extracted, new_correct = score_sample(target["dataset"], sample["pred"], row["answer"])
            old_correct = bool(sample.get("correct"))
            rec = {"sample": sample.get("sample"), "old_correct": old_correct,
                   "verified_correct": bool(new_correct), "extracted_answer": extracted}
            samples.append(rec)
            if old_correct != bool(new_correct):
                diffs.append({"level": "sample", "qid": row["qid"], "answer": row["answer"], **rec})
        old_pass = bool(row.get("pass_at_k"))
        new_pass = any(x["verified_correct"] for x in samples)
        correct += new_pass
        if old_pass != new_pass:
            diffs.append({"level": "problem", "qid": row["qid"], "answer": row["answer"],
                          "old_pass_at_k": old_pass, "verified_pass_at_k": new_pass})
        verified.append({"qid": row["qid"], "answer": row["answer"],
                         "old_pass_at_k": old_pass, "verified_pass_at_k": new_pass, "samples": samples})
    stem = target["path"].stem
    with (out_dir / f"{stem}.verified.jsonl").open("w", encoding="utf-8") as f:
        for row in verified:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / f"{stem}.diff.jsonl").open("w", encoding="utf-8") as f:
        for row in diffs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {k: v for k, v in target.items() if k != "path"}
    summary.update(source=str(target["path"]), n_problems=len(source_rows), correct=correct,
                   accuracy=correct / len(source_rows), differences=len(diffs), publishable=not diffs)
    atomic_json(out_dir / f"{stem}.summary.json", summary)
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "queue.json", {"state": "waiting", "pid": os.getpid(),
                                      "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    while True:
        state = []
        all_ready = True
        for target in targets():
            count = len(rows(target["path"]))
            ready = count == target["expected"]
            all_ready &= ready
            state.append({"path": str(target["path"]), "rows": count,
                          "expected": target["expected"], "ready": ready})
        atomic_json(OUT / "readiness.json", {"updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                              "ready": all_ready, "targets": state})
        if all_ready:
            break
        time.sleep(60)
    atomic_json(OUT / "queue.json", {"state": "verifying", "pid": os.getpid(),
                                      "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    summaries = [verify_target(target) for target in targets()]
    grouped = {}
    for summary in summaries:
        grouped.setdefault((summary["model"], summary["method"], summary["dataset"]), []).append(summary)
    aggregate = []
    for (model, method, dataset), values in sorted(grouped.items()):
        accs = [x["accuracy"] * 100 for x in sorted(values, key=lambda x: x["run"])]
        aggregate.append({"model": model, "method": method, "dataset": dataset,
                          "run_percent": accs, "mean_percent": statistics.mean(accs),
                          "sample_sd_percentage_points": statistics.stdev(accs),
                          "publishable": all(x["publishable"] for x in values)})
    manifest = {"completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "summaries": summaries, "aggregates": aggregate,
                "all_publishable": all(x["publishable"] for x in summaries)}
    atomic_json(OUT / "manifest.json", manifest)
    atomic_json(OUT / "overleaf_authoritative_results.json", aggregate)
    atomic_json(OUT / "queue.json", {"state": "complete", "pid": os.getpid(),
                                      "completed_utc": manifest["completed_utc"],
                                      "all_publishable": manifest["all_publishable"]})


if __name__ == "__main__":
    main()
