"""
generate_reasoning.py
─────────────────────
Generates chain-of-thought reasoning for each question using
Qwen3-4B-Thinking via vLLM.

Uses judger.py (+ utils.py) for accurate answer matching —
handles LaTeX, fractions, numerical tolerance, MCQ, multi-answer, etc.

Per-question logic:
  - Try up to MAX_ATTEMPTS (3) times
  - Stop early as soon as judger says correct
  - Save BEST attempt (first correct one, else last attempt)

Usage:
    python generate_reasoning.py \
        --data-path data/selected_512.jsonl \
        --output-path data/selected_512_with_reasoning.jsonl \
        --model-id Qwen/Qwen3-4B-Thinking-2507 \
        --max-attempts 3
"""

import argparse
import json
import os
import re
import sys
from typing import Optional

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ── Import judger ──────────────────────────────────────────────────────────────
# judger.py and utils.py must be in the same directory as this script
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

try:
    from judger import Judger
    _judger = Judger(strict_extract=False)
    print("✅ Judger loaded successfully")
except Exception as e:
    print(f"⚠️  Could not load Judger: {e}")
    print("   Make sure judger.py and utils.py are in the same folder as this script.")
    print("   Falling back to simple string matching.")
    _judger = None


# ── System prompts ─────────────────────────────────────────────────────────────

SYSTEM_MCQ = (
    "You are an expert mathematician. Solve the problem step by step.\n"
    "Rules:\n"
    "- Show ALL reasoning inside <think>...</think> tags.\n"
    "- After </think>, output ONLY: \\boxed{X} where X is the answer letter (A, B, C, ...).\n"
    "- Do not write anything after the \\boxed{} line.\n"
)

SYSTEM_FREE = (
    "You are an expert mathematician. Solve the problem step by step.\n"
    "Rules:\n"
    "- Show ALL reasoning inside <think>...</think> tags.\n"
    "- After </think>, output ONLY: \\boxed{answer} with the final answer.\n"
    "- If multiple sub-answers are needed, separate inside the box: \\boxed{a, b, c}.\n"
    "- Do not write anything after the \\boxed{} line.\n"
)


# ── Answer checking — mirrors notebook scoring exactly ────────────────────────

def score_mcq(response: str, gold_letter: str) -> bool:
    """Extract letter from boxed or last uppercase letter, compare to gold."""
    m = re.search(r"\\boxed\{([A-Za-z])\}", response)
    if m:
        return m.group(1).upper() == gold_letter.strip().upper()
    matches = re.findall(r"\b([A-Z])\b", str(response).upper())
    extracted = matches[-1] if matches else ""
    return extracted == gold_letter.strip().upper()


def check_answer(response: str, ground_truth, is_mcq: bool, options: list = None) -> bool:
    """
    Mirrors the notebook scoring logic exactly:
      - MCQ:       score_mcq (extract letter from \\boxed{X} or last uppercase)
      - Free-form: judger.auto_judge with gold as list, options=[[]] per answer
    """
    if ground_truth is None:
        return False

    if is_mcq:
        # MCQ: gold is always a single letter string
        return score_mcq(response, str(ground_truth))

    # Free-form: use judger.auto_judge
    gold_list = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    gold_list = [str(g) for g in gold_list]

    if _judger is not None:
        try:
            return bool(_judger.auto_judge(
                pred=response,
                gold=gold_list,
                options=[[]] * len(gold_list),
            ))
        except Exception:
            pass

    # Fallback if judger not available
    extracted = extract_boxed(response)
    if extracted is None:
        return False
    ext = extracted.strip().lower()
    for gt in gold_list:
        if ext == gt.strip().lower():
            return True
        try:
            if abs(float(ext) - float(gt)) / (abs(float(gt)) + 1e-9) < 0.01:
                return True
        except (ValueError, ZeroDivisionError):
            pass
    return False


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_boxed(text: str) -> Optional[str]:
    """Extract content of last \\boxed{} in text."""
    results = []
    for match in re.finditer(r"\\boxed\{", text):
        start = match.end()
        depth, i = 1, start
        while i < len(text) and depth > 0:
            if text[i] == "{": depth += 1
            elif text[i] == "}": depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start:i-1].strip())
    return results[-1] if results else None


def extract_think(response: str) -> str:
    """Pull <think>...</think> block."""
    m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    if "\\boxed{" in response:
        return response[:response.rfind("\\boxed{")].strip()
    return response.strip()


def build_prompt(item: dict, tokenizer, attempt: int) -> str:
    question = item["question"]
    options  = item.get("options")
    is_mcq   = bool(options)

    if is_mcq:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user_msg  = f"{question}\n\nOptions:\n{opts_text}"
        system    = SYSTEM_MCQ
    else:
        user_msg = question
        system   = SYSTEM_FREE

    if attempt > 1:
        user_msg = (
            f"[Attempt {attempt} — think differently if your previous approach failed]\n\n"
            + user_msg
        )

    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_done_ids(path: str) -> dict:
    done = {}
    if os.path.exists(path):
        for item in load_jsonl(path):
            if "think" in item:
                done[item["id"]] = item
    return done


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path",    default="data/selected_512.jsonl")
    p.add_argument("--output-path",  default="data/selected_512_with_reasoning.jsonl")
    p.add_argument("--model-id",     default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--max-attempts", type=int,   default=3)
    p.add_argument("--max-new-tokens",       type=int,   default=8000)
    p.add_argument("--max-model-len",        type=int,   default=12000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--temperature",  type=float, default=0.6)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--resume",       action="store_true")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"Loading data from {args.data_path}")
    data = load_jsonl(args.data_path)
    print(f"  {len(data)} items")

    # Resume
    existing = {}
    if args.resume:
        existing = load_done_ids(args.output_path)
        print(f"  Resuming — {len(existing)} already done")

    todo = [item for item in data if item["id"] not in existing]
    print(f"  {len(todo)} items to process")
    if not todo:
        print("Nothing to do.")
        return

    # Load model
    print(f"\nLoading tokenizer + vLLM: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, use_fast=False)

    llm = LLM(
        model=args.model_id,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )

    # Sampling params per attempt — increasing temperature
    sampling_params = [
        SamplingParams(temperature=0.3, max_tokens=args.max_new_tokens),   # attempt 1
        SamplingParams(temperature=0.6, max_tokens=args.max_new_tokens),   # attempt 2
        SamplingParams(temperature=0.8, max_tokens=args.max_new_tokens),   # attempt 3
    ]

    # Per-item state
    item_state = {
        item["id"]: {"attempts": 0, "best": None, "done": False}
        for item in todo
    }

    # Multi-attempt loop
    for attempt_num in range(1, args.max_attempts + 1):
        pending = [item for item in todo if not item_state[item["id"]]["done"]]
        if not pending:
            print(f"\nAll resolved before attempt {attempt_num}.")
            break

        print(f"\n{'='*60}")
        print(f"Attempt {attempt_num}/{args.max_attempts} — {len(pending)} items")
        print(f"{'='*60}")

        sp = sampling_params[attempt_num - 1]

        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start:batch_start + args.batch_size]
            prompts = [build_prompt(item, tokenizer, attempt_num) for item in batch]

            print(f"  Batch {batch_start//args.batch_size + 1}/"
                  f"{(len(pending)-1)//args.batch_size + 1} "
                  f"({len(batch)} items)...", flush=True)

            vllm_outputs = llm.generate(prompts, sp)

            for item, vllm_out in zip(batch, vllm_outputs):
                response  = vllm_out.outputs[0].text.strip()
                think     = extract_think(response)
                extracted = extract_boxed(response)
                is_mcq    = bool(item.get("options"))

                correct = check_answer(
                    response=response,
                    ground_truth=item.get("answer"),
                    is_mcq=is_mcq,
                    options=item.get("options"),
                )

                state = item_state[item["id"]]
                state["attempts"] += 1

                enriched = dict(item)
                enriched["think"]            = think
                enriched["response"]         = response
                enriched["extracted_answer"] = extracted
                enriched["correct"]          = correct
                enriched["attempts_used"]    = state["attempts"]

                # Keep first correct attempt; if none correct keep last
                if state["best"] is None or correct:
                    state["best"] = enriched

                if correct:
                    state["done"] = True
                    status = "✓ CORRECT"
                else:
                    if state["attempts"] >= args.max_attempts:
                        state["done"] = True
                    status = "✗ wrong" + (" (exhausted)" if state["done"] else "")

                print(f"    id={item['id']:4d}  attempt={state['attempts']}  "
                      f"extracted={str(extracted)[:30]:30s}  {status}")

        n_done    = sum(1 for s in item_state.values() if s["done"])
        n_correct = sum(1 for s in item_state.values() if s["done"] and s["best"] and s["best"]["correct"])
        print(f"\n  After attempt {attempt_num}: {n_done}/{len(todo)} done, "
              f"{n_correct}/{len(todo)} correct")

    # Collect results
    results = dict(existing)
    for item in todo:
        best = item_state[item["id"]]["best"]
        if best is None:
            best = dict(item)
            best.update({"think": "", "response": "", "extracted_answer": None,
                         "correct": False, "attempts_used": item_state[item["id"]]["attempts"]})
        results[item["id"]] = best

    output_data = [results[item["id"]] for item in data if item["id"] in results]
    save_jsonl(output_data, args.output_path)

    # Summary
    total        = len(output_data)
    n_correct    = sum(1 for d in output_data if d.get("correct"))
    avg_attempts = sum(d.get("attempts_used", 0) for d in output_data) / max(total, 1)
    dist = {}
    for d in output_data:
        a = d.get("attempts_used", 0)
        dist[a] = dist.get(a, 0) + 1

    print(f"\n{'='*60}")
    print(f"DONE — {args.output_path}")
    print(f"{'='*60}")
    print(f"  Total:        {total}")
    print(f"  Correct:      {n_correct} ({n_correct/max(total,1):.1%})")
    print(f"  Avg attempts: {avg_attempts:.2f}")
    print(f"  Dist:         {dict(sorted(dist.items()))}")
    print(f"\nUse correct=True items for LoRA SFT.")


if __name__ == "__main__":
    main()
