#!/usr/bin/env python3
"""MATH-500 (pass@1) and AIME 2025 (pass@8) for Full Attention and ZoomKV.

Three independent runs are launched by run/run_math_aime_repeats.sh with
different seeds. Each run writes per-problem records plus a summary so the
mean and standard deviation can be computed across runs.

Generation (both tasks): temperature=0.6, top_k=20, top_p=0.95,
max_new_tokens=38912 (~38.9K). Qwen models use YaRN with factor 4.0.
ZoomKV uses a single chunk level (no hierarchy).
"""

import argparse
import json
import os
import random
import re
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from urania_rescore_queue import (
    aime_extract,
    compare_answers as urania_compare_answers,
    math_extract,
    math_verify_score,
    truncate_math,
)

MAX_NEW_TOKENS = 38912


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model(model_name: str):
    if "llama" in model_name.lower():
        from model_hub.llama import LlamaModel
        return LlamaModel, "llama"
    from model_hub.qwen import QwenModel
    return QwenModel, "qwen"


def load_problems(dataset: str, data_dir: str):
    path = os.path.join(data_dir, f"{dataset}.jsonl")
    rows = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(tokenizer, problem: str, backend: str, dataset: str) -> str:
    del tokenizer, backend
    if dataset == "aime2025":
        return f"""You are an expert mathematical problem solver. Solve the following problem with complete step-by-step reasoning.

Problem: {problem.strip()}

Instructions:
1. Work through the problem step-by-step
2. Show your key reasoning and calculations
3. When you reach the final answer, put only the final integer inside a LaTeX boxed expression
4. STOP immediately after providing the boxed answer - do not verify or explore alternatives

Solve the problem:"""
    return f"""You are an expert mathematical problem solver. Solve the following problem with complete step-by-step reasoning.

Problem: {problem.strip()}

Instructions:
1. Show your complete reasoning process and all intermediate steps
2. Explain your thought process thoroughly
3. Verify your calculations and approach as needed
4. At the end, provide your final answer in this EXACT format: \\boxed{{your_answer}}
5. STOP immediately after the boxed answer. Do not start another problem or continue the conversation.

Take your time to work through this problem completely:"""


def extract_boxed(text: str):
    key = r"\boxed{"
    idx = text.rfind(key)
    if idx < 0:
        return None
    i = idx + len(key)
    depth = 1
    chars = []
    while i < len(text) and depth:
        ch = text[i]
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            chars.append(ch)
        else:
            chars.append(ch)
        i += 1
    return "".join(chars).strip() if depth == 0 else None


def normalize_answer(text: str) -> str:
    text = text.strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\,", "").replace("\\!", "").replace(" ", "")
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("\\%", "%")
    if text.endswith("."):
        text = text[:-1]
    return text


def _int_strings_equal(a, b):
    a = a.lstrip("0") or "0"
    b = b.lstrip("0") or "0"
    if a.startswith("-"):
        a = "-" + (a[1:].lstrip("0") or "0")
    if b.startswith("-"):
        b = "-" + (b[1:].lstrip("0") or "0")
    if len(a) > 20 or len(b) > 20:
        return a == b
    try:
        return int(a) == int(b)
    except ValueError:
        return False


def answers_match(pred: str, gold: str, dataset: str) -> bool:
    if pred is None:
        return False
    pred_n = normalize_answer(pred)
    gold_n = normalize_answer(str(gold))
    if dataset == "aime2025":
        pred_nums = re.findall(r"-?\d+", pred_n)
        gold_nums = re.findall(r"-?\d+", gold_n)
        if not pred_nums or not gold_nums:
            return False
        return _int_strings_equal(pred_nums[-1], gold_nums[-1])
    if pred_n == gold_n:
        return True
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
        transforms = standard_transformations + (implicit_multiplication_application,)
        p_expr = parse_expr(pred_n.replace("^", "**"), transformations=transforms)
        g_expr = parse_expr(gold_n.replace("^", "**"), transformations=transforms)
        return sympy.simplify(p_expr - g_expr) == 0
    except Exception:
        return False


def grade(text: str, gold: str, dataset: str) -> bool:
    if dataset == "aime2025":
        return urania_compare_answers(aime_extract(text), gold)
    clean = truncate_math(text)
    extracted = math_extract(clean)
    return math_verify_score(gold, clean, extracted)


def polar_config(args):
    is_pariskv = getattr(args, "kv_method", "zoomkv") == "pariskv"
    codebook = ""
    if is_pariskv:
        codebook = os.path.join(
            PROJECT_ROOT,
            "turboquant/codebooks/codebook_d128_m8_Kr1_Kw256_rabitq_sign.json",
        )
    return {
        "PolarANN": {
            "static_pattern_start": 4,
            "static_pattern_end": 512,
            "core": 22,
            "nprobe": 150,
            "cache_unit_size": 8,
            "cache_cluster_num": 450,
            "cache_module": (
                "cache_hub.polar_cache"
                if is_pariskv
                else "cache_hub.polar_cache_kivi_quest_density_seperate_kernel"
            ),
            "codebook_path": codebook,
            "collision_ratio": 0.3,
            "candidate_ratio": 0.5 if is_pariskv else 0.4,
            "budget_ratio": 0.2,
            "dynamic_update_interval": args.update_interval,
            "sink_size": 64,
            "local_size": 256,
            "final_topk": 100,
            "full_attention_threshold": args.full_threshold,
            "enable_offload": True,
        }
    }


def apply_zoomkv_env():
    os.environ["POLAR_CACHE_MODULE"] = "cache_hub.polar_cache_kivi_quest_density_seperate_kernel"
    os.environ["POLAR_QUEST_HIERARCHICAL"] = "0"
    os.environ["POLAR_QUEST_CHUNK"] = "8"
    # retrieval-zone length: <5k → 0.7, 5k–10k → 0.5, >10k → 0.4
    os.environ["POLAR_QUEST_CAND_SCHEDULE"] = "4999:0.7,10000:0.5,0.4"
    os.environ["POLAR_CDS_DENSE_RATIO"] = "0.3"
    os.environ["POLAR_CDS_DENSE_TOPK"] = "8"
    os.environ["POLAR_CDS_SPARSE_TOPK"] = "4"
    os.environ.pop("POLAR_FORCE_CAND_TOKENS", None)
    os.environ.pop("POLAR_ADAPTIVE_DYNAMIC_CAND_RATIOS", None)
    os.environ.pop("POLAR_ADAPTIVE_DYNAMIC_COLL_RATIOS", None)
    os.environ.setdefault("ZOOMKV_GEN_PROGRESS_EVERY", "256")


def apply_pariskv_env():
    """ParisKV paper/repo default candidate schedule.

    Official repo https://github.com/amy-77/ParisKV and arXiv:2602.07721:
    retrieval-zone <5k → cand 0.50 / coll 0.50
    5k–10k → 0.25 / 0.30
    10k–30k → 0.20 / 0.25
    ≥30k → 0.10 / 0.20
    MATH-500: local=256, update=256, full-thres=1K, topk=100.
    """
    os.environ["POLAR_CACHE_MODULE"] = "cache_hub.polar_cache"
    os.environ["POLAR_ADAPTIVE_DYNAMIC_CAND_THRESHOLDS"] = "5000,10000,30000"
    os.environ["POLAR_ADAPTIVE_DYNAMIC_CAND_RATIOS"] = "0.5,0.25,0.2,0.1"
    os.environ["POLAR_ADAPTIVE_DYNAMIC_COLL_RATIOS"] = "0.5,0.3,0.25,0.2"
    os.environ.pop("POLAR_ADAPTIVE_FORCE_CANDIDATE_RATIO", None)
    os.environ.pop("POLAR_ADAPTIVE_FORCE_COLLISION_RATIO", None)
    os.environ.pop("POLAR_QUEST_CAND_SCHEDULE", None)
    os.environ.pop("POLAR_QUEST_HIERARCHICAL", None)
    os.environ.pop("POLAR_FORCE_CAND_TOKENS", None)
    os.environ.setdefault("ZOOMKV_GEN_PROGRESS_EVERY", "256")


def load_done(path: str):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done.add(int(rec["qid"]))
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset", choices=["math500", "aime2025"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--attention_type", choices=["Full_Flash_Attn", "PolarANN"], default=None)
    parser.add_argument("--kv_method", choices=["full", "zoomkv", "pariskv"], default=None,
                        help="full / ZoomKV / ParisKV. Sets attention backend and retrieval knobs.")
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--samples_per_problem", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--update_interval", type=int, default=256)
    parser.add_argument("--full_threshold", type=int, default=1024)
    parser.add_argument("--yarn_factor", type=float, default=0.0)
    parser.add_argument("--yarn_orig_max", type=int, default=32768)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Debug cap on number of problems")
    parser.add_argument("--qid", type=int, default=-1, help="0-based problem index; -1 = all")
    parser.add_argument("--problem", type=int, default=0,
                        help="1-based problem number (AIME 第N题). Overrides --qid.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split the dataset into this many disjoint qid-modulo shards")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Zero-based shard index used with --num_shards")
    parser.add_argument("--stop_on_first_correct", action="store_true",
                        help="Stop remaining pass@k samples once one sample is correct")
    parser.add_argument("--save_full_pred", action="store_true",
                        help="Store full generation text (large). Default stores boxed only.")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("require num_shards >= 1 and 0 <= shard_id < num_shards")
    if args.kv_method is None:
        if args.attention_type == "PolarANN":
            args.kv_method = "zoomkv"
        elif args.attention_type == "Full_Flash_Attn":
            args.kv_method = "full"
        else:
            raise SystemExit("pass --kv_method or --attention_type")
    if args.attention_type is None:
        args.attention_type = "Full_Flash_Attn" if args.kv_method == "full" else "PolarANN"

    if args.kv_method != "pariskv":
        raise SystemExit("This frozen runtime is only for ParisKV")
    torch.set_num_threads(int(os.environ.get("PARISKV_CPU_THREADS", "4")))
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault("ZOOMKV_GEN_PROGRESS_EVERY", "256")
    if args.kv_method == "pariskv":
        apply_pariskv_env()
    elif args.kv_method == "zoomkv":
        apply_zoomkv_env()

    set_seed(args.seed)
    model_cls, backend = resolve_model(args.model_name)
    all_problems = load_problems(args.dataset, args.data_dir)
    selected_qid = None
    if args.problem > 0:
        selected_qid = args.problem - 1
    elif args.qid >= 0:
        selected_qid = args.qid
    if selected_qid is not None:
        if selected_qid < 0 or selected_qid >= len(all_problems):
            raise SystemExit(f"problem index {selected_qid} out of range 0..{len(all_problems)-1}")
        problems = [(selected_qid, all_problems[selected_qid])]
        print(f"[select] dataset={args.dataset} 1-based={selected_qid+1} qid={selected_qid}", flush=True)
    else:
        problems = [
            (qid, row) for qid, row in enumerate(all_problems)
            if qid % args.num_shards == args.shard_id
        ]
        if args.limit > 0:
            problems = problems[: args.limit]
        if args.num_shards > 1:
            print(
                f"[shard] {args.shard_id}/{args.num_shards} "
                f"assigned={len(problems)}/{len(all_problems)}",
                flush=True,
            )

    dtype = torch.bfloat16
    rope_scaling = None
    if backend == "qwen" and args.yarn_factor > 0:
        rope_scaling = {
            "rope_type": "yarn",
            "factor": args.yarn_factor,
            "original_max_position_embeddings": args.yarn_orig_max,
        }
    # Prompt is short; leave room for the full 38.9K generation.
    max_length = 4096 + args.max_new_tokens + 128
    print(
        f"[load] {args.model_name} backend={backend} kv={args.kv_method} "
        f"attn={args.attention_type} dataset={args.dataset} run={args.run_id} "
        f"seed={args.seed} k={args.samples_per_problem} yarn={rope_scaling}",
        flush=True,
    )
    llm = model_cls(
        model_name=args.model_name,
        max_length=max_length,
        dtype=dtype,
        device_map=args.device,
        rope_scaling=rope_scaling,
    )
    tokenizer = llm.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tag = args.kv_method
    model_tag = os.path.basename(args.model_name.rstrip("/"))
    if "Qwen3-4B" in model_tag:
        model_tag = "Qwen3-4B"
    base_out_path = os.path.join(args.out_dir, f"{model_tag}_{args.dataset}_{tag}_run{args.run_id}.jsonl")
    if args.num_shards > 1:
        out_path = base_out_path.replace(
            ".jsonl", f".shard{args.shard_id}-of-{args.num_shards}.jsonl"
        )
        base_done = load_done(base_out_path)
    else:
        out_path = base_out_path
        base_done = set()
    done = load_done(out_path) | base_done
    attn_config = polar_config(args) if args.attention_type == "PolarANN" else {"Full_Flash_Attn": {}}
    assigned_qids = {qid for qid, _ in problems}
    n_expected = len(assigned_qids - base_done)
    running_scores = []
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    running_scores.append(json.loads(line)["pass_at_k"])

    with open(out_path, "a", encoding="utf-8") as fout:
        for qid, row in problems:
            if qid in done:
                continue
            prompt = build_prompt(tokenizer, row["problem"], backend, args.dataset)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs.input_ids.to(llm.layers[0].device)
            attn_mask = inputs.attention_mask.to(llm.layers[0].device)
            gold = row["answer"]
            samples = []
            n_correct = 0
            k = args.samples_per_problem
            print(
                f"[problem] #{qid} gold={gold} pass@{k} early_stop={args.stop_on_first_correct}",
                flush=True,
            )
            for sample_idx in range(args.samples_per_problem):
                generation_seed = args.seed * 1_000_003 + int(qid) * 17 + sample_idx
                set_seed(generation_seed)
                print(
                    f"[pass] start #{qid} sample {sample_idx + 1}/{k}",
                    flush=True,
                )
                with torch.no_grad():
                    output_ids, timing = llm.generate(
                        attention_type=args.attention_type,
                        inputs_ids=input_ids,
                        attention_masks=attn_mask,
                        max_new_length=args.max_new_tokens,
                        attn_config=attn_config,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        disable_early_stop=False,
                    )
                gen_ids = output_ids[0]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                try:
                    ok = grade(text, gold, args.dataset)
                except Exception as exc:
                    print(f"[grade] unreadable prediction: {type(exc).__name__}: {exc}", flush=True)
                    ok = False
                n_correct += int(ok)
                boxed = extract_boxed(text)
                sample_rec = {
                    "sample": sample_idx,
                    "generation_seed": generation_seed,
                    "correct": ok,
                    "n_tokens": len(gen_ids),
                    "boxed": boxed,
                    "decode_time": timing.get("decode_time", 0),
                }
                if args.save_full_pred:
                    sample_rec["pred"] = text
                samples.append(sample_rec)
                decode_s = float(timing.get("decode_time", 0) or 0)
                print("=" * 72, flush=True)
                print(
                    f"[pass] done  #{qid} sample {sample_idx + 1}/{k}  "
                    f"{'CORRECT' if ok else 'WRONG'}  "
                    f"boxed={boxed}  gold={gold}  "
                    f"tokens={len(gen_ids)}  decode={decode_s:.1f}s  "
                    f"correct_so_far={n_correct}/{sample_idx + 1}  "
                    f"pass@{k}={1 if n_correct > 0 else 0}",
                    flush=True,
                )
                print("=" * 72, flush=True)
                if args.stop_on_first_correct and ok:
                    print(
                        f"[early-stop] #{qid} after sample {sample_idx + 1}/{k} "
                        f"(pass@{k}=1, skip remaining)",
                        flush=True,
                    )
                    break
            # n == k: pass@k is 1 iff at least one sample is correct.
            passed = 1.0 if n_correct > 0 else 0.0
            rec = {
                "qid": qid,
                "model": model_tag,
                "dataset": args.dataset,
                "method": tag,
                "run_id": args.run_id,
                "seed": args.seed,
                "answer": gold,
                "n": k,
                "c": n_correct,
                "n_attempted": len(samples),
                "early_stopped": bool(args.stop_on_first_correct and n_correct > 0 and len(samples) < k),
                "pass_at_k": passed,
                "input_len": int(input_ids.shape[1]),
                "samples": samples,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            running_scores.append(passed)
            run_acc = float(np.mean(running_scores))
            print(
                f"[progress] {model_tag} {args.dataset} {tag} run{args.run_id} "
                f"qid={qid} {len(running_scores)}/{n_expected} "
                f"last_pass={passed:.0f} running_acc={run_acc:.4f}",
                flush=True,
            )
            write_summary(out_path, args, model_tag, running_scores, n_expected, complete=False)

    write_summary(out_path, args, model_tag, running_scores, n_expected, complete=True)


def write_summary(path, args, model_tag, scores, n_expected, complete):
    if not scores:
        return
    acc = float(np.mean(scores))
    summary = {
        "model": model_tag,
        "dataset": args.dataset,
        "method": args.kv_method,
        "run_id": args.run_id,
        "seed": args.seed,
        "n_problems": len(scores),
        "n_expected": n_expected,
        "complete": bool(complete and len(scores) >= n_expected),
        "k": args.samples_per_problem,
        "accuracy": acc,
    }
    summary_path = path.replace(".jsonl", ".summary.json")
    with open(summary_path + ".tmp", "w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2)
    os.replace(summary_path + ".tmp", summary_path)
    if complete:
        print(f"[summary] {summary_path} accuracy={acc:.4f} n={len(scores)}", flush=True)


def summarize(path: str, args):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if not rows:
        return
    model_tag = rows[0].get("model") or os.path.basename(args.model_name.rstrip("/"))
    scores = [row["pass_at_k"] for row in rows]
    write_summary(path, args, model_tag, scores, len(rows), complete=True)


if __name__ == "__main__":
    main()
