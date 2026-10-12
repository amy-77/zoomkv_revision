#!/usr/bin/env python3
"""Run the eight run-0 jobs as two disjoint qid shards per GPU."""

import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent.parent
BASE_OUT = ROOT / "results/urania_correct_20261011"
CONTROL = BASE_OUT / "run0_sharded_control"
NUM_SHARDS = 2

MODELS = {
    "q4": {
        "path": ROOT / "models/Qwen3-4B-Thinking-2507",
        "tag": "Qwen3-4B",
        "label": "qwen3_4b_thinking2507",
        "yarn": 0,
    },
    "q8": {
        "path": ROOT / "models/Qwen3-8B",
        "tag": "Qwen3-8B",
        "label": "qwen3_8b",
        "yarn": 4,
    },
}

# Preserve the proven one-job-per-GPU placement, then add the second shard on
# the same GPU. This yields two resident model processes per card.
SPECS = [
    ("q8", "zoomkv", "math500", 0),
    ("q4", "zoomkv", "math500", 1),
    ("q4", "pariskv", "math500", 2),
    ("q8", "pariskv", "math500", 3),
    ("q4", "zoomkv", "aime2025", 4),
    ("q8", "zoomkv", "aime2025", 5),
    ("q8", "pariskv", "aime2025", 6),
    ("q4", "pariskv", "aime2025", 7),
]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def paths(job):
    stem = f"{job['model_tag']}_{job['dataset']}_{job['method']}_run0"
    out_dir = Path(job["out_dir"])
    base = out_dir / f"{stem}.jsonl"
    shard = out_dir / f"{stem}.shard{job['shard_id']}-of-{NUM_SHARDS}.jsonl"
    return base, shard, shard.with_suffix(".summary.json")


def shard_complete(job):
    _, shard, summary = paths(job)
    if not shard.exists() or not summary.exists():
        return False
    try:
        data = json.loads(summary.read_text())
        rows = read_rows(shard)
        return bool(
            data.get("complete")
            and data.get("n_problems") == data.get("n_expected")
            and len(rows) == data.get("n_expected")
            and len({int(row["qid"]) for row in rows}) == len(rows)
        )
    except Exception:
        return False


def build_jobs():
    jobs = []
    for model_key, method, dataset, gpu in SPECS:
        model = MODELS[model_key]
        k = 1 if dataset == "math500" else 8
        interval = 256 if dataset == "math500" else 512
        threshold = 1024 if dataset == "math500" else 2048
        expected = 500 if dataset == "math500" else 30
        runtime = ROOT / f"code/{method}_urania_runtime_20261011"
        out_dir = BASE_OUT / model["label"] / method / dataset
        for shard_id in range(NUM_SHARDS):
            command = [
                sys.executable,
                "-u",
                str(runtime / "run/eval_math_aime.py"),
                "--model_name", str(model["path"]),
                "--dataset", dataset,
                "--data_dir", str(ROOT / "ZoomKV/data/math_aime"),
                "--kv_method", method,
                "--run_id", "0",
                "--seed", "42",
                "--samples_per_problem", str(k),
                "--max_new_tokens", "38912",
                "--temperature", "0.6",
                "--top_k", "20",
                "--top_p", "0.95",
                "--update_interval", str(interval),
                "--full_threshold", str(threshold),
                "--yarn_factor", str(model["yarn"]),
                "--device", "cuda:0",
                "--out_dir", str(out_dir),
                "--num_shards", str(NUM_SHARDS),
                "--shard_id", str(shard_id),
                "--save_full_pred",
            ]
            if k == 8:
                command.append("--stop_on_first_correct")
            jobs.append({
                "name": f"{model['tag']}_{dataset}_{method}_run0_shard{shard_id}",
                "model_key": model_key,
                "model_tag": model["tag"],
                "method": method,
                "dataset": dataset,
                "expected": expected,
                "gpu": gpu,
                "shard_id": shard_id,
                "runtime": str(runtime),
                "out_dir": str(out_dir),
                "command": command,
                "state": "queued",
                "attempts": 0,
            })
    return jobs


def merge_job_pair(pair):
    first = pair[0]
    base, _, _ = paths(first)
    sources = [base] + [paths(job)[1] for job in pair]
    by_qid = {}
    for source in sources:
        for row in read_rows(source):
            qid = int(row["qid"])
            if qid in by_qid:
                old = by_qid[qid]
                old_sig = [(x.get("generation_seed"), x.get("correct"), x.get("boxed")) for x in old.get("samples", [])]
                new_sig = [(x.get("generation_seed"), x.get("correct"), x.get("boxed")) for x in row.get("samples", [])]
                if old_sig != new_sig:
                    raise RuntimeError(f"conflicting duplicate qid={qid} for {base}")
                continue
            by_qid[qid] = row

    expected = int(first["expected"])
    if len(by_qid) != expected or set(by_qid) != set(range(expected)):
        raise RuntimeError(f"incomplete merge for {base}: {len(by_qid)}/{expected}")

    tmp = base.with_name(base.name + ".merge.tmp")
    with tmp.open("w", encoding="utf-8") as fout:
        for qid in range(expected):
            fout.write(json.dumps(by_qid[qid], ensure_ascii=False) + "\n")
    tmp.replace(base)
    scores = [float(by_qid[qid]["pass_at_k"]) for qid in range(expected)]
    summary = {
        "model": first["model_tag"],
        "dataset": first["dataset"],
        "method": first["method"],
        "run_id": 0,
        "seed": 42,
        "n_problems": expected,
        "n_expected": expected,
        "complete": True,
        "k": 1 if first["dataset"] == "math500" else 8,
        "accuracy": sum(scores) / len(scores),
        "merged_from_shards": NUM_SHARDS,
    }
    atomic_json(base.with_suffix(".summary.json"), summary)


def main():
    CONTROL.mkdir(parents=True, exist_ok=True)
    lock = (CONTROL / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = build_jobs()
    atomic_json(CONTROL / "manifest.json", {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "jobs": jobs})
    active = {}
    stopping = False

    def request_stop(signum, _frame):
        nonlocal stopping
        stopping = True
        print(f"[signal] received {signum}; terminating workers", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    while True:
        for name, (proc, job, handle) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            handle.close()
            del active[name]
            if code == 0 and shard_complete(job):
                job["state"] = "complete"
            elif job["attempts"] < 2 and not stopping:
                job["state"] = "queued"
                job["exit_code"] = code
            else:
                job["state"] = "failed"
                job["exit_code"] = code
            print(f"[exit] {name} rc={code} state={job['state']}", flush=True)

        if stopping:
            for proc, _, _ in active.values():
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
            return 2

        for job in jobs:
            if job["state"] != "queued":
                continue
            if shard_complete(job):
                job["state"] = "complete"
                continue
            job["attempts"] += 1
            log_dir = Path(job["out_dir"]) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{job['name']}.attempt{job['attempts']}.log"
            env = os.environ.copy()
            for key in list(env):
                if key.startswith("POLAR_"):
                    env.pop(key)
            env.update(
                CUDA_VISIBLE_DEVICES=str(job["gpu"]),
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                CUDA_HOME="/usr/local/cuda-12.4",
                DISABLE_TRITON_FALLBACK="1",
                TMPDIR=str(ROOT / "code/tmp"),
                CUDA_CACHE_PATH=str(BASE_OUT / ".cuda_cache"),
                PYTHONUNBUFFERED="1",
                TOKENIZERS_PARALLELISM="false",
                TORCH_CUDA_ARCH_LIST="9.0",
                FLASHINFER_DISABLE_VERSION_CHECK="1",
                MAX_JOBS="4",
                OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="1",
                ZOOMKV_CPU_THREADS="4",
                PARISKV_CPU_THREADS="4",
                ZOOMKV_GEN_PROGRESS_EVERY="1024",
                TRITON_CACHE_DIR=str(BASE_OUT / f".triton_{job['model_key']}_gpu{job['gpu']}"),
                PATH=f"/DATA/disk0/qyl/envs/retroinfer/bin:{env.get('PATH', '')}",
            )
            handle = log_path.open("a")
            proc = subprocess.Popen(
                job["command"],
                cwd=job["runtime"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            job.update(
                state="running",
                pid=proc.pid,
                attempts=job["attempts"],
                log=str(log_path),
                started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            active[job["name"]] = (proc, job, handle)
            print(f"[launch] gpu={job['gpu']} pid={proc.pid} {job['name']}", flush=True)
            time.sleep(1)

        for job in jobs:
            base, shard, _ = paths(job)
            job["base_rows"] = len(read_rows(base))
            job["shard_rows"] = len(read_rows(shard))
        atomic_json(CONTROL / "status.json", {
            "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scheduler_pid": os.getpid(),
            "jobs": jobs,
        })

        if not active and all(job["state"] in ("complete", "failed") for job in jobs):
            break
        time.sleep(20)

    failed = [job["name"] for job in jobs if job["state"] == "failed"]
    if failed:
        print(f"[done] failed={failed}", flush=True)
        return 1

    grouped = {}
    for job in jobs:
        key = (job["model_key"], job["method"], job["dataset"])
        grouped.setdefault(key, []).append(job)
    for pair in grouped.values():
        merge_job_pair(pair)
    print("[done] all shards complete and merged", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
