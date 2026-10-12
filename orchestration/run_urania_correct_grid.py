#!/usr/bin/env python3
"""Run the fully Urania-aligned Qwen3 sparse-evaluation grid."""
import collections
import datetime
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
CONTROL = ROOT / "results/urania_correct_20261011/control"
BASE_OUT = ROOT / "results/urania_correct_20261011"
RUNTIMES = {
    "zoomkv": ROOT / "code/zoomkv_urania_runtime_20261011",
    "pariskv": ROOT / "code/pariskv_urania_runtime_20261011",
}
MODELS = {
    "q4": {
        "path": ROOT / "models/Qwen3-4B-Thinking-2507",
        "tag": "Qwen3-4B", "label": "qwen3_4b_thinking2507",
        "gpus": [1, 2, 4, 7], "capacity": 3, "slot_mib": 28000, "yarn": 0,
    },
    "q8": {
        "path": ROOT / "models/Qwen3-8B",
        "tag": "Qwen3-8B", "label": "qwen3_8b",
        "gpus": [0, 3, 5, 6], "capacity": 1, "slot_mib": 60000, "yarn": 4,
    },
}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def output_paths(job):
    dest = Path(job["out_dir"])
    name = f"{job['model_tag']}_{job['dataset']}_{job['method']}_run{job['run_id']}"
    return dest / f"{name}.jsonl", dest / f"{name}.summary.json"


def complete(job):
    output, summary = output_paths(job)
    if not output.exists() or not summary.exists():
        return False
    try:
        s = json.loads(summary.read_text())
        rows = [json.loads(x) for x in output.read_text().splitlines() if x.strip()]
        return bool(
            s.get("complete") and s.get("n_problems") == job["n_expected"]
            and len(rows) == job["n_expected"]
            and len({int(r["qid"]) for r in rows}) == job["n_expected"]
            and all(all("pred" in sample for sample in r.get("samples", [])) for r in rows)
        )
    except Exception:
        return False


def build_jobs(python):
    jobs = []
    for model_key, model in MODELS.items():
        for run_id in range(3):
            for dataset, k, n, interval, threshold in (
                ("math500", 1, 500, 256, 1024),
                ("aime2025", 8, 30, 512, 2048),
            ):
                for method in ("zoomkv", "pariskv"):
                    out = BASE_OUT / model["label"] / method / dataset
                    out.mkdir(parents=True, exist_ok=True)
                    command = [
                        python, "-u", str(RUNTIMES[method] / "run/eval_math_aime.py"),
                        "--model_name", str(model["path"]), "--dataset", dataset,
                        "--data_dir", str(ROOT / "ZoomKV/data/math_aime"),
                        "--kv_method", method, "--run_id", str(run_id), "--seed", str(42 + run_id),
                        "--samples_per_problem", str(k), "--max_new_tokens", "38912",
                        "--temperature", "0.6", "--top_k", "20", "--top_p", "0.95",
                        "--update_interval", str(interval), "--full_threshold", str(threshold),
                        "--yarn_factor", str(model["yarn"]), "--device", "cuda:0",
                        "--out_dir", str(out), "--save_full_pred",
                    ]
                    if k == 8:
                        command.append("--stop_on_first_correct")
                    jobs.append({
                        "name": f"{model['tag']}_{dataset}_{method}_run{run_id}",
                        "model_key": model_key, "model_tag": model["tag"], "dataset": dataset,
                        "method": method, "run_id": run_id, "seed": 42 + run_id,
                        "n_expected": n, "k": k, "out_dir": str(out), "command": command,
                        "state": "queued", "attempts": 0,
                    })
    return jobs


def main():
    CONTROL.mkdir(parents=True, exist_ok=True)
    lock = (CONTROL / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = build_jobs(sys.executable)
    atomic_json(CONTROL / "manifest.json", {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": {
            "prompt": "exact Urania dataset-specific AIME/MATH prompt",
            "scorer": "exact Urania AIME scorer; Urania MATH Math-Verify scorer",
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "max_new_tokens": 38912, "seeds": [42, 43, 44],
            "aime2025": {"pass_k": 8, "stop_on_first_urania_correct": True},
            "math500": {"pass_k": 1}, "eos_early_stop": True,
            "save_full_pred": True,
        },
        "models": {k: {kk: str(vv) if isinstance(vv, Path) else vv for kk, vv in v.items()}
                   for k, v in MODELS.items()},
        "jobs": jobs,
    })
    active = {}
    print(f"[queue] Urania-correct grid: {len(jobs)} jobs", flush=True)
    while True:
        for name, (proc, job, handle) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            handle.close()
            del active[name]
            job["exit_code"] = code
            job["state"] = "complete" if code == 0 and complete(job) else ("queued" if job["attempts"] < 2 else "failed")
            print(f"[exit] {name} rc={code} state={job['state']}", flush=True)

        report = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True
        )
        free = {int(a): int(b) for a, b in (line.split(",") for line in report.splitlines())}
        counts = collections.Counter(job["gpu"] for _, job, _ in active.values())
        for job in jobs:
            if job["state"] != "queued":
                continue
            if complete(job):
                job["state"] = "complete"
                continue
            model = MODELS[job["model_key"]]
            choices = [g for g in model["gpus"] if counts[g] < model["capacity"] and free[g] >= model["slot_mib"]]
            if not choices:
                continue
            gpu = min(choices, key=lambda g: (counts[g], g))
            job["attempts"] += 1
            logfile = Path(job["out_dir"]) / "logs" / f"{job['name']}.attempt{job['attempts']}.log"
            logfile.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            for key in list(env):
                if key.startswith("POLAR_"):
                    env.pop(key)
            env.update(
                CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                CUDA_HOME="/usr/local/cuda-12.4", DISABLE_TRITON_FALLBACK="1",
                TMPDIR=str(ROOT / "code/tmp"), CUDA_CACHE_PATH=str(BASE_OUT / ".cuda_cache"),
                PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", TORCH_CUDA_ARCH_LIST="9.0",
                FLASHINFER_DISABLE_VERSION_CHECK="1", MAX_JOBS="4", OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", ZOOMKV_CPU_THREADS="4",
                PARISKV_CPU_THREADS="4", ZOOMKV_GEN_PROGRESS_EVERY="1024",
                TRITON_CACHE_DIR=str(BASE_OUT / f".triton_{job['model_key']}_gpu{gpu}"),
                PATH=f"/DATA/disk0/qyl/envs/retroinfer/bin:{env.get('PATH', '')}",
            )
            handle = logfile.open("a")
            proc = subprocess.Popen(job["command"], cwd=RUNTIMES[job["method"]], env=env,
                                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            job.update(state="running", pid=proc.pid, gpu=gpu,
                       started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), log=str(logfile))
            active[job["name"]] = (proc, job, handle)
            counts[gpu] += 1
            free[gpu] -= model["slot_mib"]
            print(f"[launch] gpu={gpu} pid={proc.pid} {job['name']}", flush=True)
            time.sleep(2)

        atomic_json(CONTROL / "status.json", {
            "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "queue_pid": os.getpid(), "counts": dict(collections.Counter(j["state"] for j in jobs)),
            "jobs": jobs,
        })
        if not active and all(j["state"] in ("complete", "failed") for j in jobs):
            break
        time.sleep(20)

    groups = collections.defaultdict(list)
    for job in jobs:
        if job["state"] == "complete":
            _, summary = output_paths(job)
            groups[(job["model_key"], job["method"], job["dataset"])].append(json.loads(summary.read_text())["accuracy"] * 100)
    aggregate = [{
        "model": model, "method": method, "dataset": dataset, "n_runs": len(values),
        "run_percent": values, "mean_percent": statistics.mean(values),
        "sample_sd_percentage_points": statistics.stdev(values) if len(values) > 1 else None,
    } for (model, method, dataset), values in sorted(groups.items())]
    atomic_json(CONTROL / "aggregate.json", aggregate)
    failed = [j["name"] for j in jobs if j["state"] == "failed"]
    print(f"[queue] finished failed={failed}", flush=True)
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
