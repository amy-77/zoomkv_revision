#!/usr/bin/env python3
"""Wait for the sparse benchmark grid, then rescore with Urania's scorers.

The source JSONL files are never modified.  Qwen3-8B rows produced before
--save_full_pred was enabled are losslessly hydrated from the legacy node
copies, with qid/sample-count/correctness checks before they are accepted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

ROOT = Path("/DATA/disk0/qyl/code/zoomkv_revision_migrated_20261008")
RESULTS = ROOT / "results"
LEGACY = RESULTS / "urania_rescore_legacy_q8_fullpred_20261011"
OUT = RESULTS / "urania_rescored_20261011"

URANIA_AIME_SHA256 = "8acad4e99d6b3e0abe14e4f1843339bac10f379444fe04d5c8471567544284c3"
URANIA_MATH_SHA256 = "ef530a64e0e60d49d232155c108b3d1d5200c2e9c958ff9101ed8f1d0c3d4ea0"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def jsonl_rows(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# Exact scoring logic copied from the Urania scripts identified by the hashes
# above, separated from their model-loading/evaluation code.
def aime_extract(response):
    markers = (
        "\nHuman:", "\nUser:", "\nAssistant:", "\nSystem:",
        "Human:", "User:", "Assistant:", "System:",
        "You are an expert mathematical problem solver", "<|im_start|>", "<|im_end|>",
    )
    cut = len(response)
    for marker in markers:
        pos = response.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    response = response[:cut]

    def placeholder(content):
        norm = re.sub(r"[\s{}\\]+", "", str(content).strip().lower())
        return (not norm or "your_answer" in norm or "youranswer" in norm or norm in {
            "answer", "finalanswer", "theanswer", "yourfinalanswer", "integeranswer", "finalinteger",
        })

    starts = [m.start() for m in re.finditer(r"\\boxed\{", response)]
    for start in reversed(starts):
        pos, depth, begin = start + 7, 1, start + 7
        while pos < len(response) and depth:
            depth += (response[pos] == "{") - (response[pos] == "}")
            pos += 1
        if depth == 0:
            content = re.sub(r"\s+", " ", response[begin:pos - 1]).strip()
            if placeholder(content):
                continue
            cleaned = re.sub(r"\\text[a-z]*\{([^}]*)\}", r"\1", content)
            cleaned = cleaned.replace("$", "").replace(",", "").strip().rstrip(".")
            if re.fullmatch(r"[+-]?\d{1,7}", cleaned):
                return str(int(cleaned))
            return content
    match = re.search(r"####\s*(.+)", response)
    if match:
        return match.group(1).strip()
    match = re.search(r"[Tt]he (?:answer|solution) is[:\s]+(.+?)(?:\.|$)", response, re.I)
    if match:
        return match.group(1).strip()
    matches = list(re.finditer(r"(?:S|sum|total|answer|result)\s*=\s*(-?\d+(?:\.\d+)?)", response, re.I))
    if matches:
        return matches[-1].group(1)
    matches = list(re.finditer(r"=\s*(-?\d+(?:\.\d+)?)\s*\.", response))
    if matches:
        return matches[-1].group(1)
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    if lines and not any(k in lines[-1].lower() for k in (
        "user:", "assistant:", "question:", "problem:", "let", "we have", "therefore",
    )):
        match = re.search(r"(-?\d+(?:\.\d+)?)", lines[-1])
        if match:
            return match.group(1)
    return None


def normalize_answer(answer):
    if not answer:
        return ""
    answer = re.sub(r"\s+", "", str(answer).strip())
    answer = answer.replace(r"\%", "").replace("%", "")
    answer = answer.replace(r"\circ", "").replace(r"^\circ", "")
    answer = re.sub(r"\\[dt]?frac", r"\\frac", answer)
    for old, new in ((r"\left(", "("), (r"\right)", ")"),
                     (r"\left[", "["), (r"\right]", "]"),
                     (r"\left\{", "{"), (r"\right\}", "}")):
        answer = answer.replace(old, new)
    answer = answer.replace(r"\,", "")
    return re.sub(r"\\text[a-z]*\{([^}]+)\}", r"\1", answer).lower()


def compare_answers(predicted, ground_truth):
    if not predicted or not ground_truth:
        return False
    pred_text = str(predicted).strip()
    candidates = [pred_text]
    match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)", pred_text)
    if match and match.group(1) != pred_text:
        candidates.append(match.group(1))
    gold_norm = normalize_answer(ground_truth)
    for candidate in candidates:
        if normalize_answer(candidate) == gold_norm:
            return True
        try:
            import sympy
            if sympy.simplify(sympy.sympify(str(candidate).replace("\\", "")) -
                              sympy.sympify(str(ground_truth).replace("\\", ""))) == 0:
                return True
        except Exception:
            pass
    return False


def truncate_math(response):
    cut = len(response)
    for pattern in (r"\n\s*(?:Human|User|Assistant):",
                    r"\n\s*You are an expert mathematical problem solver\.",
                    r"\n\s*Problem:\s+"):
        match = re.search(pattern, response)
        if match:
            cut = min(cut, match.start())
    return response[:cut].strip()


def math_extract(response):
    response = truncate_math(response)
    for match in re.finditer(r"\\boxed\{", response):
        pos, depth, begin = match.start() + 7, 1, match.start() + 7
        while pos < len(response) and depth:
            depth += (response[pos] == "{") - (response[pos] == "}")
            pos += 1
        if depth == 0:
            content = re.sub(r"\s+", " ", response[begin:pos - 1]).strip()
            if content not in ["your_answer", "answer", "result", "..."]:
                return content
    match = re.search(r"####\s*(.+)", response)
    if match:
        return match.group(1).strip()
    match = re.search(r"[Tt]he (?:answer|solution) is[:\s]+(.+?)(?:\.|$)", response, re.I)
    if match:
        return match.group(1).strip()
    matches = list(re.finditer(r"(?:S|sum|total|answer|result)\s*=\s*(-?\d+(?:\.\d+)?)", response, re.I))
    if matches:
        return matches[-1].group(1)
    matches = list(re.finditer(r"=\s*(-?\d+(?:\.\d+)?)\s*\.", response))
    if matches:
        return matches[-1].group(1)
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    if lines and not any(k in lines[-1].lower() for k in (
        "user:", "assistant:", "question:", "problem:", "let", "we have", "therefore",
    )):
        match = re.search(r"(-?\d+(?:\.\d+)?)", lines[-1])
        if match:
            return match.group(1)
    return None


def math_fallback_compare(predicted, ground_truth):
    if compare_answers(predicted, ground_truth):
        return True
    if not predicted or not ground_truth:
        return False
    try:
        import sympy
        def cvt(value):
            value = str(value).replace(r"\left", "").replace(r"\right", "")
            value = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", value)
            return value.replace("π", "pi").replace(r"\pi", "pi").replace("∞", "oo").replace("\\", "")
        return sympy.simplify(sympy.sympify(cvt(predicted)) - sympy.sympify(cvt(ground_truth))) == 0
    except Exception:
        return False


def math_verify_score(gold, generated, extracted):
    try:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
        configs = (LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig())
        table = {"π": r"\pi", "θ": r"\theta", "√": r"\sqrt{}", "∞": r"\infty", "±": r"\pm"}
        def uni(value):
            value = str(value)
            for old, new in table.items():
                value = value.replace(old, new)
            return value
        gold_expr = parse(uni(gold), extraction_config=configs)
        pred_expr = parse(uni(extracted if extracted else generated), extraction_config=configs)
        if verify(gold_expr, pred_expr):
            return True
    except Exception:
        pass
    return math_fallback_compare(extracted, gold)


def targets():
    groups = [
        ("q4_zoomkv", RESULTS / "zoomkv_qwen3_4b_thinking2507_20261011", "Qwen3-4B"),
        ("q4_pariskv", RESULTS / "pariskv_qwen3_4b_thinking2507_20261011", "Qwen3-4B"),
        ("q8_zoomkv", RESULTS / "zoomkv_aligned_20261007", "Qwen3-8B"),
        ("q8_pariskv", RESULTS / "pariskv", "Qwen3-8B"),
    ]
    result = []
    for group, base, model in groups:
        for dataset, expected in (("aime2025", 30), ("math500", 500)):
            for run in range(3):
                method = group.split("_")[-1]
                path = base / dataset / f"{model}_{dataset}_{method}_run{run}.jsonl"
                result.append({"group": group, "dataset": dataset, "run": run,
                               "expected": expected, "path": path})
    return result


def readiness():
    report, ready = [], True
    for target in targets():
        try:
            count = len(jsonl_rows(target["path"]))
        except Exception as exc:
            count = -1
            target["error"] = repr(exc)
        ok = count == target["expected"]
        ready &= ok
        report.append({**{k: str(v) if isinstance(v, Path) else v for k, v in target.items()},
                       "rows": count, "ready": ok})
    return ready, report


def legacy_path(target):
    if not target["group"].startswith("q8_"):
        return None
    method = target["group"].split("_")[-1]
    return LEGACY / method / target["dataset"] / target["path"].name


def hydrate_samples(target, row, legacy_by_qid):
    samples = row.get("samples", [])
    if all("pred" in sample for sample in samples):
        return samples, "current"
    old = legacy_by_qid.get(int(row["qid"]))
    if old is None:
        raise RuntimeError(f"missing legacy qid={row['qid']} for {target['path']}")
    legacy_samples = old.get("samples", [])
    if len(legacy_samples) != len(samples) or not all("pred" in sample for sample in legacy_samples):
        raise RuntimeError(f"legacy sample mismatch qid={row['qid']} for {target['path']}")
    if [bool(x.get("correct")) for x in legacy_samples] != [bool(x.get("correct")) for x in samples]:
        raise RuntimeError(f"legacy correctness mismatch qid={row['qid']} for {target['path']}")
    return legacy_samples, "legacy_full_pred"


def rescore_target(target):
    rows = jsonl_rows(target["path"])
    lp = legacy_path(target)
    legacy_by_qid = {}
    if lp and lp.exists():
        legacy_by_qid = {int(row["qid"]): row for row in jsonl_rows(lp)}
    out_dir = OUT / target["group"] / target["dataset"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = target["path"].stem
    scored_path = out_dir / f"{stem}.urania.jsonl"
    diff_path = out_dir / f"{stem}.diff.jsonl"
    scored_lines, diffs = [], []
    old_ok = new_ok = hydrated = changed_samples = unresolved_early_stop = 0
    for row in rows:
        samples, source = hydrate_samples(target, row, legacy_by_qid)
        hydrated += source == "legacy_full_pred"
        rescored_samples = []
        for sample in samples:
            pred = sample["pred"]
            if target["dataset"] == "aime2025":
                extracted = aime_extract(pred)
                correct = compare_answers(extracted, row["answer"])
            else:
                clean = truncate_math(pred)
                extracted = math_extract(clean)
                correct = math_verify_score(row["answer"], clean, extracted)
            old_correct = bool(sample.get("correct", False))
            rec = {"sample": sample.get("sample"), "generation_seed": sample.get("generation_seed"),
                   "old_correct": old_correct, "urania_correct": bool(correct),
                   "urania_extracted_answer": extracted, "pred_source": source}
            rescored_samples.append(rec)
            if old_correct != bool(correct):
                changed_samples += 1
                diffs.append({"level": "sample", "qid": row["qid"], "answer": row["answer"], **rec})
        old_pass = bool(row.get("pass_at_k", False))
        new_pass = any(x["urania_correct"] for x in rescored_samples)
        incomplete = bool(row.get("early_stopped")) and old_pass and not new_pass and len(samples) < int(row.get("n", 1))
        unresolved_early_stop += incomplete
        old_ok += old_pass
        new_ok += new_pass
        out_row = {"qid": row["qid"], "answer": row["answer"], "model": row.get("model"),
                   "method": row.get("method"), "dataset": row.get("dataset"), "run_id": row.get("run_id"),
                   "old_pass_at_k": old_pass, "urania_pass_at_k": new_pass,
                   "urania_unresolved_early_stop": incomplete, "n": row.get("n"),
                   "n_attempted": len(samples), "pred_source": source, "samples": rescored_samples}
        scored_lines.append(out_row)
        if old_pass != new_pass or incomplete:
            diffs.append({"level": "problem", **{k: v for k, v in out_row.items() if k != "samples"}})
    with scored_path.open("w", encoding="utf-8") as f:
        for row in scored_lines:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with diff_path.open("w", encoding="utf-8") as f:
        for row in diffs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "source": str(target["path"]), "source_sha256": hashlib.sha256(target["path"].read_bytes()).hexdigest(),
        "dataset": target["dataset"], "group": target["group"], "run": target["run"],
        "n_problems": len(rows), "old_correct": old_ok, "urania_correct": new_ok,
        "old_accuracy": old_ok / len(rows), "urania_accuracy": new_ok / len(rows),
        "problem_flips": sum(d.get("level") == "problem" and not d.get("urania_unresolved_early_stop") for d in diffs),
        "sample_flips": changed_samples, "legacy_hydrated_rows": hydrated,
        "unresolved_early_stop_rows": unresolved_early_stop,
        "publishable": unresolved_early_stop == 0,
        "scored_jsonl": str(scored_path), "diff_jsonl": str(diff_path),
    }
    atomic_json(out_dir / f"{stem}.summary.json", summary)
    return summary


def run_rescore():
    summaries = [rescore_target(target) for target in targets()]
    grouped = {}
    for summary in summaries:
        key = (summary["group"], summary["dataset"])
        grouped.setdefault(key, []).append(summary)
    aggregates = []
    for (group, dataset), values in sorted(grouped.items()):
        accs = [v["urania_accuracy"] * 100 for v in values]
        aggregates.append({
            "group": group, "dataset": dataset, "n_runs": len(values),
            "run_percent": accs, "mean_percent": statistics.mean(accs),
            "sample_sd_percentage_points": statistics.stdev(accs) if len(accs) > 1 else None,
            "publishable": all(v["publishable"] for v in values),
            "unresolved_early_stop_rows": sum(v["unresolved_early_stop_rows"] for v in values),
        })
    manifest = {
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scorer": "Urania AIME extractor/comparator + Urania MATH-500 Math-Verify",
        "urania_aime_source_sha256": URANIA_AIME_SHA256,
        "urania_math_source_sha256": URANIA_MATH_SHA256,
        "math_verify_available": True,
        "summaries": summaries, "aggregates": aggregates,
        "all_publishable": all(x["publishable"] for x in summaries),
    }
    atomic_json(OUT / "manifest.json", manifest)
    atomic_json(OUT / "overleaf_authoritative_results.json", aggregates)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true", help="Fail unless all inputs are ready, then rescore")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "queue.json", {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "pid": os.getpid(),
        "state": "waiting", "targets": len(targets()),
    })
    while True:
        ready, report = readiness()
        atomic_json(OUT / "readiness.json", {
            "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "ready": ready, "targets": report,
        })
        if ready:
            break
        if args.once:
            return 2
        time.sleep(args.poll_seconds)
    atomic_json(OUT / "queue.json", {
        "started_rescore_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "pid": os.getpid(),
        "state": "rescoring", "targets": len(targets()),
    })
    try:
        manifest = run_rescore()
    except Exception as exc:
        atomic_json(OUT / "queue.json", {
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "pid": os.getpid(),
            "state": "failed", "error": repr(exc),
        })
        raise
    atomic_json(OUT / "queue.json", {
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "pid": os.getpid(),
        "state": "complete", "targets": len(targets()),
        "all_publishable": manifest["all_publishable"],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
