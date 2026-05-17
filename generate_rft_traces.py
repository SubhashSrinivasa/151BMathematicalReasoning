"""
generate_rft_traces.py — rejection-sampling trace generation (STaR-style).

This is a DATA step, not a training step: nothing is trained here. It
manufactures the chain-of-thought solutions that data/public.jsonl lacks, so
that train_lora_cot.py has correct worked solutions to imitate.

Pipeline position:
    generate_rft_traces.py  ->  data/rft_sft.jsonl     (this script — no training)
    train_lora_cot.py       ->  CoT SFT LoRA adapter
    train_grpo.py           ->  GRPO LoRA adapter
    test_inference.py       ->  evaluate

What it does:
  - Loads the BASE Qwen3-4B-Thinking (no adapter) with vLLM.
  - For each problem, samples K full-reasoning rollouts.
  - Scores every rollout with judger.py — the same verifier used at eval time.
  - Keeps only CORRECT rollouts (<= KEEP shortest distinct ones per problem).
  - Writes one jsonl row per kept trace to data/rft_sft.jsonl.

Problems the base model never solves are simply dropped here — GRPO will still
attempt them later.

Setup: same environment as run_inference.py.

Run with defaults (full public set, K=6 samples each):
    .venv/bin/python generate_rft_traces.py

Common toggles:
    --data PATH        input jsonl   (default: data/public.jsonl)
    --output PATH      traces jsonl  (default: data/rft_sft.jsonl)
    --k N              rollouts sampled per problem      (default: 6)
    --keep N           max correct traces kept per problem (default: 2)
    --max-tokens N     generation length cap            (default: 3072)
    --n-samples N      only process the first N problems (default: all)
"""

import os
import sys
import json
import re
import time
import signal
import argparse
from pathlib import Path
from typing import Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_DATA_PATH   = "data/public.jsonl"
DEFAULT_OUTPUT_PATH = "data/rft_sft.jsonl"
JUDGE_TIMEOUT_S     = 5


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    print("=" * 70, flush=True)


def step(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Same prompts as train_lora.py / run_inference.py — keep the task description
# identical across trace generation, SFT, RL and inference.
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def is_correct(judger, text: str, item: dict) -> bool:
    """Score one rollout against the gold answer with the competition judger."""
    if item.get("options"):
        return extract_letter(text) == str(item["answer"]).strip().upper()

    gold = item["answer"]
    gold_list = gold if isinstance(gold, list) else [gold]

    def handler(signum, frame):
        raise TimeoutError("judge timeout")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(JUDGE_TIMEOUT_S)
    try:
        return bool(judger.auto_judge(
            pred=text, gold=gold_list, options=[[]] * len(gold_list),
        ))
    except Exception:
        return False
    finally:
        signal.alarm(0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rejection-sampling CoT trace generation.")
    p.add_argument("--model",       default=DEFAULT_MODEL_ID,    help="Base model id or path.")
    p.add_argument("--data",        default=DEFAULT_DATA_PATH,   help="Input jsonl.")
    p.add_argument("--output",      default=DEFAULT_OUTPUT_PATH, help="Output traces jsonl.")
    p.add_argument("--k",           type=int, default=6,         help="Rollouts sampled per problem.")
    p.add_argument("--keep",        type=int, default=2,         help="Max correct traces kept per problem.")
    p.add_argument("--max-tokens",  type=int, default=3072,      help="Generation length cap.")
    p.add_argument("--temperature", type=float, default=1.0,     help="Sampling temperature (diversity).")
    p.add_argument("--n-samples",   type=int, default=None,      help="Limit to first N problems (debug).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. CUDA ──────────────────────────────────────────────────────────────
    banner("STEP 1 / 6  CUDA sanity check")
    import torch
    step(f"torch {torch.__version__} cuda {torch.version.cuda}")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available.")
    step(f"GPU 0: {torch.cuda.get_device_name(0)}")

    # ── 2. Imports ───────────────────────────────────────────────────────────
    banner("STEP 2 / 6  Imports (transformers + vllm)")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)
    step("imports + judger ready.")

    # ── 3. Load dataset ──────────────────────────────────────────────────────
    banner("STEP 3 / 6  Load dataset")
    data = [json.loads(line) for line in open(args.data)]
    if args.n_samples:
        data = data[:args.n_samples]
    n_mcq = sum(bool(d.get("options")) for d in data)
    step(f"loaded {len(data)} problems  ({n_mcq} MCQ, {len(data) - n_mcq} free-form)")

    # ── 4. Load model + build prompts ────────────────────────────────────────
    banner("STEP 4 / 6  Load base model with vLLM")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    t0 = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=16384,
        max_num_seqs=32,
        max_num_batched_tokens=16384,
    )
    step(f"LLM ready in {time.time() - t0:.1f}s.")

    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False, add_generation_prompt=True,
        ))
    step(f"built {len(prompts)} chat-template prompts.")

    # n=K -> vLLM returns K rollouts per prompt in one batched call.
    sampling_params = SamplingParams(
        n=args.k,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
    )

    # ── 5. Generate + reject ─────────────────────────────────────────────────
    banner(f"STEP 5 / 6  Sample K={args.k} rollouts/problem and reject incorrect ones")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    step(f"generation done in {(time.time() - t0) / 60:.1f} min.")

    step("scoring rollouts with the judger ...")
    kept_rows  = []
    n_solved   = 0
    n_attempts = n_hits = 0
    for item, out in zip(data, outputs):
        completions = [o.text.strip() for o in out.outputs]
        n_attempts += len(completions)
        correct = []
        for text in completions:
            if is_correct(judger, text, item):
                n_hits += 1
                correct.append(text)
        if not correct:
            continue
        n_solved += 1
        # dedup, then keep the shortest few (concise correct traces are the
        # best SFT targets — less room for memorized noise).
        uniq = sorted(set(correct), key=len)[:args.keep]
        for text in uniq:
            kept_rows.append({
                "id":         item.get("id"),
                "question":   item["question"],
                "options":    item.get("options"),
                "answer":     item["answer"],
                "completion": text,
            })

    # ── 6. Save + stats ──────────────────────────────────────────────────────
    banner("STEP 6 / 6  Save traces")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in kept_rows:
            f.write(json.dumps(row) + "\n")

    def pct(c, t): return f"{c}/{t} ({100 * c / t:.1f}%)" if t else "N/A"
    print(f"\n  problems solved (>=1 correct rollout) : {pct(n_solved, len(data))}")
    print(f"  rollout hit rate                      : {pct(n_hits, n_attempts)}")
    print(f"  traces written                        : {len(kept_rows)}  -> {out_path}")
    if kept_rows:
        ex = kept_rows[0]
        step("first kept trace (truncated):")
        print("    Q:", ex["question"][:160].replace("\n", " "))
        print("    A:", ex["completion"][:400].replace("\n", "\\n"),
              "..." if len(ex["completion"]) > 400 else "")
    banner("DONE  —  next: train_lora_cot.py")


if __name__ == "__main__":
    main()
