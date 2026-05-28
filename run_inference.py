"""
CSE 151B Competition — base-model inference with vLLM.

ONE-TIME setup (run in a shell):
    wget -qO- https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv venv .venv --seed
    .venv/bin/python -m pip install sympy "numpy<2" transformers vllm tqdm \
        bitsandbytes antlr4-python3-runtime==4.11.1

To run with defaults (full public set, no adapter):
    .venv/bin/python run_inference.py

Common toggles:
    --data PATH           dataset jsonl  (default: data/public.jsonl)
    --output PATH         results jsonl  (default: results/starter_results.jsonl)
    --adapter PATH        optional LoRA adapter dir; if omitted, runs base model
    --n-samples N         only run the first N items  (default: all)
    --no-eval             write {id, is_mcq, response} only (private test set)

If --output ends in .csv, results are written as a 2-column submission CSV
(header: id,response) instead of jsonl. Use this for private/test submissions:
    python run_inference.py --data data/private.jsonl --output sub.csv --no-eval
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
DEFAULT_OUTPUT_PATH = "results/starter_results.jsonl"
MAX_TOKENS          = 64000


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
    p = argparse.ArgumentParser(description="Run vLLM inference on the math dataset.")
    p.add_argument("--model",     default=DEFAULT_MODEL_ID,    help="HF model id or local path.")
    p.add_argument("--data",      default=DEFAULT_DATA_PATH,   help="Input jsonl.")
    p.add_argument("--output",    default=DEFAULT_OUTPUT_PATH, help="Output jsonl.")
    p.add_argument("--adapter",   default=None,                help="Optional LoRA adapter directory.")
    p.add_argument("--n-samples", type=int, default=None,      help="Limit to first N items (default: all).")
    p.add_argument("--no-eval",   action="store_true",         help="Skip scoring; write only {id, is_mcq, response}.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. CUDA sanity check ─────────────────────────────────────────────────
    banner("STEP 1 / 7  CUDA sanity check")
    import torch
    step(f"torch {torch.__version__} (built for CUDA {torch.version.cuda})")
    step(f"cuda available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available — fix the environment before continuing.")
    step(f"GPU 0: {torch.cuda.get_device_name(0)}")

    # ── 2. Imports ───────────────────────────────────────────────────────────
    banner("STEP 2 / 7  Imports (transformers + vllm)")
    step("importing transformers ...")
    from transformers import AutoTokenizer
    step("importing vllm ...")
    from vllm import LLM, SamplingParams
    if args.adapter:
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

    mcq_sample  = next((d for d in data if d.get("options")), None)
    free_sample = next((d for d in data if not d.get("options")), None)
    if mcq_sample:
        step("MCQ sample (truncated):")
        print("    ", json.dumps(mcq_sample, indent=2)[:400].replace("\n", "\n    "))
    if free_sample:
        step("Free-form sample (truncated):")
        print("    ", json.dumps(free_sample, indent=2)[:400].replace("\n", "\n    "))

    # ── 4. Prompt construction ───────────────────────────────────────────────
    banner("STEP 4 / 7  Prompt construction")
    for label, item in [("MCQ", mcq_sample), ("Free-form", free_sample)]:
        if item is None:
            continue
        sys_p, usr_p = build_prompt(item["question"], item.get("options"))
        step(f"{label} user prompt (first 200 chars): {usr_p[:200]!r}")

    # ── 5. Load model with vLLM ──────────────────────────────────────────────
    banner("STEP 5 / 7  Load model with vLLM (~30-60s)")
    step(f"loading tokenizer for {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    step("tokenizer loaded.")

    llm_kwargs = dict(
        model=args.model,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=32000,
        max_num_seqs=32,
        max_num_batched_tokens=32000,
    )
    lora_request = None
    if args.adapter:
        step(f"LoRA adapter requested: {args.adapter}")
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 32  # covers rank<=32 adapters
        lora_request = LoRARequest("adapter", 1, args.adapter)
    else:
        step("running base model (no adapter).")

    step("constructing LLM (bf16, FlashAttention 2, A30) ...")
    t0 = time.time()
    llm = LLM(**llm_kwargs)
    step(f"LLM ready in {time.time() - t0:.1f}s.")

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )
    step("sampling params set.")

    # ── 6. Generate responses ────────────────────────────────────────────────
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

    step("calling llm.generate (vLLM will print its own progress bar) ...")
    t0 = time.time()
    gen_kwargs = {"sampling_params": sampling_params}
    if lora_request is not None:
        gen_kwargs["lora_request"] = lora_request
    outputs = llm.generate(prompts, **gen_kwargs)
    step(f"generation done in {time.time() - t0:.1f}s ({len(outputs)} outputs).")

    responses = [out.outputs[0].text.strip() for out in outputs]
    for i in range(min(3, len(responses))):
        step(f"── Response {i} (id={subset[i].get('id')}, len={len(responses[i])} chars) ──")
        print(responses[i][:400], "..." if len(responses[i]) > 400 else "")

    # ── 7. Score (skipped if --no-eval) ──────────────────────────────────────
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
        banner("EVALUATION RESULTS")
        print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
        print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
        print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        import csv
        step(f"writing CSV submission (id,response) to {out_path} ...")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "response"])
            for item, response in zip(subset, responses):
                writer.writerow([item.get("id"), response])
    else:
        step(f"writing results to {out_path} ...")
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    step(f"saved {len(results) if results else len(subset)} records.")
    banner("DONE")


if __name__ == "__main__":
    main()
