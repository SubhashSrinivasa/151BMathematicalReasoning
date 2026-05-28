"""
Inference with a trained LoRA adapter on top of the base Qwen3-4B-Thinking.

Uses vLLM's runtime LoRA support (no merging — adapter stays as a separate
~50 MB file, base model is loaded once).

Usage:
    .venv/bin/python test_inference.py --adapter checkpoints/qwen3-4b-lora

Required:
    --adapter PATH    LoRA adapter directory produced by train_lora.py

Optional toggles (with defaults):
    --model PATH      base model id  (default: Qwen/Qwen3-4B-Thinking-2507)
    --data PATH       dataset jsonl  (default: data/public.jsonl)
    --output PATH     results jsonl  (default: results/lora_results.jsonl)
    --n-samples N     limit to first N items (default: all)
    --no-eval         write {id, is_mcq, response} only (private test set)
    --max-lora-rank   max rank vLLM should allocate slots for (default: 32)
"""

import os
import sys
import json
import re
import time
import argparse
from pathlib import Path
from typing import Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_DATA_PATH   = "data/public.jsonl"
DEFAULT_OUTPUT_PATH = "results/lora_results.jsonl"
MAX_TOKENS          = 16000


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    print("=" * 70, flush=True)


def step(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with a LoRA adapter.")
    p.add_argument("--adapter",        required=True,                help="LoRA adapter directory (REQUIRED).")
    p.add_argument("--model",          default=DEFAULT_MODEL_ID,     help="Base model id or path.")
    p.add_argument("--data",           default=DEFAULT_DATA_PATH,    help="Input jsonl.")
    p.add_argument("--output",         default=DEFAULT_OUTPUT_PATH,  help="Output jsonl.")
    p.add_argument("--n-samples",      type=int, default=None,       help="Limit to first N items (default: all).")
    p.add_argument("--no-eval",        action="store_true",          help="Skip scoring; private-test format.")
    p.add_argument("--max-lora-rank",  type=int, default=32,         help="Max LoRA rank slots (>= adapter rank).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.adapter).exists():
        sys.exit(f"Adapter path does not exist: {args.adapter}")

    # ── 1. CUDA ──────────────────────────────────────────────────────────────
    banner("STEP 1 / 7  CUDA sanity check")
    import torch
    step(f"torch {torch.__version__} cuda {torch.version.cuda}")
    step(f"cuda available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available.")
    step(f"GPU 0: {torch.cuda.get_device_name(0)}")

    # ── 2. Imports ───────────────────────────────────────────────────────────
    banner("STEP 2 / 7  Imports")
    step("importing transformers ...")
    from transformers import AutoTokenizer
    step("importing vllm ...")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    step("importing tqdm ...")
    from tqdm import tqdm
    step("imports done.")

    # ── 3. Load dataset ──────────────────────────────────────────────────────
    banner("STEP 3 / 7  Load dataset")
    step(f"reading {args.data} ...")
    data = [json.loads(line) for line in open(args.data)]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options")   for d in data)
    step(f"loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    # ── 4. Prompt construction (sanity preview) ─────────────────────────────
    banner("STEP 4 / 7  Prompt construction")
    mcq_sample  = next((d for d in data if d.get("options")), None)
    free_sample = next((d for d in data if not d.get("options")), None)
    for label, item in [("MCQ", mcq_sample), ("Free-form", free_sample)]:
        if item is None:
            continue
        sys_p, usr_p = build_prompt(item["question"], item.get("options"))
        step(f"{label} user prompt (first 200 chars): {usr_p[:200]!r}")

    # ── 5. Load base + adapter ───────────────────────────────────────────────
    banner("STEP 5 / 7  Load base model + LoRA adapter")
    step(f"base model: {args.model}")
    step(f"adapter:    {args.adapter}  (max_lora_rank={args.max_lora_rank})")
    step("loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    step("tokenizer loaded.")

    step("constructing LLM with enable_lora=True ...")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        enforce_eager=True,
        gpu_memory_utilization=0.90,
        max_model_len=16384,
        max_num_seqs=16,
        max_num_batched_tokens=16384,
    )
    step(f"LLM ready in {time.time() - t0:.1f}s.")
    lora_request = LoRARequest("trained-lora", 1, args.adapter)

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.8,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )
    step("sampling params set.")

    # ── 6. Generate ──────────────────────────────────────────────────────────
    subset = data[:args.n_samples] if args.n_samples else data
    banner(f"STEP 6 / 7  Generate responses for {len(subset)} question(s)")
    step("building chat-template prompts ...")
    prompts = []
    for item in subset:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)
    step(f"built {len(prompts)} prompts.")

    step("calling llm.generate WITH LoRA adapter ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_request)
    step(f"generation done in {time.time() - t0:.1f}s ({len(outputs)} outputs).")

    responses = [out.outputs[0].text.strip() for out in outputs]
    for i in range(min(3, len(responses))):
        step(f"── Response {i} (id={subset[i].get('id')}, len={len(responses[i])} chars) ──")
        print(responses[i][:400], "..." if len(responses[i]) > 400 else "")

    # ── 7. Score ─────────────────────────────────────────────────────────────
    banner("STEP 7 / 7  Score responses" + ("  [SKIPPED: --no-eval]" if args.no_eval else ""))
    results = []
    if args.no_eval:
        for item, response in zip(subset, responses):
            results.append({
                "id":       item.get("id"),
                "is_mcq":   bool(item.get("options")),
                "response": response,
            })
    else:
        step("loading judger ...")
        sys.path.insert(0, ".")
        from judger import Judger
        judger = Judger(strict_extract=False)
        step("judger ready.")

        for item, response in tqdm(zip(subset, responses), total=len(subset), desc="Scoring"):
            is_mcq = bool(item.get("options"))
            gold   = item["answer"]
            if is_mcq:
                correct = score_mcq(response, str(gold))
            else:
                gold_list = gold if isinstance(gold, list) else [gold]
                try:
                    correct = judger.auto_judge(
                        pred=response,
                        gold=gold_list,
                        options=[[]] * len(gold_list),
                    )
                except Exception:
                    correct = False
            results.append({
                "id":       item.get("id"),
                "is_mcq":   is_mcq,
                "gold":     gold,
                "response": response,
                "correct":  correct,
            })
        step(f"scored {len(results)} responses.")

        mcq_res  = [r for r in results if r["is_mcq"]]
        free_res = [r for r in results if not r["is_mcq"]]
        def acc(s):
            return sum(r["correct"] for r in s) / len(s) * 100 if s else 0.0
        banner("EVALUATION RESULTS (LoRA)")
        print(f"  Adapter    : {args.adapter}")
        print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
        print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
        print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    step(f"writing results to {out_path} ...")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    step(f"saved {len(results)} records.")
    banner("DONE")


if __name__ == "__main__":
    main()

