"""
generate_reasoning.py
─────────────────────
Generates chain-of-thought reasoning for each question in the input JSONL
using Qwen3-4B-Thinking (via vLLM).

Per-question logic:
  - Try up to MAX_ATTEMPTS (3) times
  - Stop early as soon as the model's extracted answer matches ground truth
  - Save the BEST attempt (correct one if found, otherwise last attempt)
  - Records: think, response, extracted_answer, correct (bool), attempts_used

Usage:
    python generate_reasoning.py \
        --data-path data/selected_512.jsonl \
        --output-path data/selected_512_with_reasoning.jsonl \
        --model-id Qwen/Qwen3-4B-Thinking-2507 \
        --max-attempts 3

For the FULL dataset (for GRPO):
    python generate_reasoning.py \
        --data-path data/public.jsonl \
        --output-path data/public_with_reasoning.jsonl \
        --max-attempts 3
"""

import argparse
import json
import os
import re
import string
from typing import Optional

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


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
    "- After </think>, output ONLY: \\boxed{answer} with the final numerical or symbolic answer.\n"
    "- If multiple sub-answers are needed, separate them with commas inside the box: \\boxed{a, b, c}.\n"
    "- Do not write anything after the \\boxed{} line.\n"
)


# ── Answer extraction ──────────────────────────────────────────────────────────

def extract_boxed(text: str) -> Optional[str]:
    """Extract the content of the last \\boxed{} in the text."""
    # Find all \boxed{...} occurrences, handle nested braces
    pattern = r"\\boxed\{"
    results = []
    for match in re.finditer(pattern, text):
        start = match.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start : i - 1].strip())
    return results[-1] if results else None


def normalise_answer(ans: str) -> str:
    """Normalise an answer string for comparison."""
    if ans is None:
        return ""
    ans = ans.strip().lower()
    # Remove surrounding quotes, brackets, parentheses
    ans = re.sub(r"^[\"\'\(\[\{]+|[\"\'\)\]\}]+$", "", ans).strip()
    # Remove trailing punctuation
    ans = ans.rstrip(".,;:")
    # Collapse whitespace
    ans = re.sub(r"\s+", " ", ans)
    return ans


def normalise_mcq_letter(ans: str) -> str:
    """For MCQ, just keep the letter."""
    if not ans:
        return ""
    # Take first uppercase letter A-Z
    m = re.search(r"\b([A-Za-z])\b", ans)
    if m:
        return m.group(1).upper()
    # Fall back: first letter
    return ans.strip()[0].upper() if ans.strip() else ""


def answers_match(extracted: str, ground_truth, is_mcq: bool) -> bool:
    """
    Compare extracted model answer to ground truth.
    ground_truth may be:
      - str  (MCQ letter, or single numeric/symbolic answer)
      - list (multiple acceptable answers for free-form multi-part)
    """
    if extracted is None:
        return False

    if is_mcq:
        ext = normalise_mcq_letter(extracted)
        if isinstance(ground_truth, list):
            gt_list = [normalise_mcq_letter(str(g)) for g in ground_truth]
        else:
            gt_list = [normalise_mcq_letter(str(ground_truth))]
        return ext in gt_list

    # Free-form
    ext = normalise_answer(extracted)
    if isinstance(ground_truth, list):
        gt_list = [normalise_answer(str(g)) for g in ground_truth]
    else:
        gt_list = [normalise_answer(str(ground_truth))]

    # Direct match
    if ext in gt_list:
        return True

    # Numeric fuzzy match
    try:
        ext_f = float(ext.replace(",", ""))
        for gt in gt_list:
            try:
                gt_f = float(gt.replace(",", ""))
                if abs(ext_f - gt_f) / (abs(gt_f) + 1e-9) < 0.01:
                    return True
            except ValueError:
                pass
    except ValueError:
        pass

    return False


# ── Prompt building ────────────────────────────────────────────────────────────

def build_prompt(item: dict, tokenizer, attempt: int) -> str:
    """
    Build a chat-template prompt for the question.
    On attempt > 1 we add a small nudge to think more carefully.
    """
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
            f"[Attempt {attempt}] Please re-read the problem carefully "
            f"and try a different approach if needed.\n\n{user_msg}"
        )

    messages = [
        {"role": "system",  "content": system},
        {"role": "user",    "content": user_msg},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ── Think extraction ───────────────────────────────────────────────────────────

def extract_think(response: str) -> str:
    """Pull <think>...</think> block, or return everything before \\boxed{}."""
    m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: everything before the boxed answer
    if "\\boxed{" in response:
        idx = response.rfind("\\boxed{")
        return response[:idx].strip()
    return response.strip()


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_done_ids(path: str) -> set:
    """For resume: collect IDs already written to output."""
    done = set()
    if os.path.exists(path):
        for item in load_jsonl(path):
            if "think" in item:
                done.add(item["id"])
    return done


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path",    default="data/selected_512.jsonl",
                   help="Input JSONL (questions + answers)")
    p.add_argument("--output-path",  default="data/selected_512_with_reasoning.jsonl",
                   help="Output JSONL with reasoning fields")
    p.add_argument("--model-id",     default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="Max generation attempts per question")
    p.add_argument("--max-new-tokens",       type=int, default=8000)
    p.add_argument("--max-model-len",        type=int, default=12000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--temperature",  type=float, default=0.6,
                   help="Sampling temperature (use 0 for greedy on attempt 1)")
    p.add_argument("--batch-size",   type=int, default=16,
                   help="Number of questions to send to vLLM at once")
    p.add_argument("--resume",       action="store_true",
                   help="Skip items that already have a think field in the output file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading data from {args.data_path} ...")
    data = load_jsonl(args.data_path)
    print(f"  {len(data)} items loaded")

    # ── Resume support ─────────────────────────────────────────────────────────
    done_ids: set = set()
    existing_results: dict = {}
    if args.resume and os.path.exists(args.output_path):
        for item in load_jsonl(args.output_path):
            if "think" in item:
                done_ids.add(item["id"])
                existing_results[item["id"]] = item
        print(f"  Resuming — {len(done_ids)} already done")

    todo = [item for item in data if item["id"] not in done_ids]
    print(f"  {len(todo)} items to process")

    if not todo:
        print("Nothing to do.")
        return

    # ── Load tokenizer + vLLM ─────────────────────────────────────────────────
    print(f"\nLoading tokenizer from {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_fast=False,
    )

    print(f"Loading vLLM from {args.model_id} ...")
    llm = LLM(
        model=args.model_id,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )

    # ── Sampling params per attempt ────────────────────────────────────────────
    # Attempt 1: slightly greedy (temp=0.3) for best first try
    # Attempt 2+: more exploratory (higher temp) to try different reasoning
    sampling_params = [
        SamplingParams(temperature=0.3,              max_tokens=args.max_new_tokens),  # attempt 1
        SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens),  # attempt 2
        SamplingParams(temperature=min(args.temperature + 0.2, 1.0),
                       max_tokens=args.max_new_tokens),                                # attempt 3
    ]

    # ── Per-item state ─────────────────────────────────────────────────────────
    # We track per-item: best result so far, remaining attempts
    # We do batch inference per attempt round for efficiency.

    results: dict = dict(existing_results)  # id → enriched item

    # Initialise tracking
    item_state: dict = {}   # id → {"attempts": 0, "best": None, "done": False}
    for item in todo:
        item_state[item["id"]] = {
            "attempts": 0,
            "best":     None,    # best enriched item so far
            "done":     False,   # True once correct or exhausted
        }

    # ── Multi-attempt loop ─────────────────────────────────────────────────────
    for attempt_num in range(1, args.max_attempts + 1):
        # Items still needing generation this round
        pending = [item for item in todo if not item_state[item["id"]]["done"]]

        if not pending:
            print(f"\nAll items resolved before attempt {attempt_num}. Done.")
            break

        print(f"\n{'='*60}")
        print(f"Attempt {attempt_num}/{args.max_attempts}  —  {len(pending)} items")
        print(f"{'='*60}")

        sp = sampling_params[attempt_num - 1]

        # Build prompts
        prompts = [build_prompt(item, tokenizer, attempt_num) for item in pending]

        # Batch inference
        all_outputs: list[str] = []
        for batch_start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[batch_start : batch_start + args.batch_size]
            batch_items   = pending[batch_start : batch_start + args.batch_size]

            print(f"  Generating batch {batch_start // args.batch_size + 1}"
                  f"/{(len(prompts) - 1) // args.batch_size + 1}"
                  f" ({len(batch_prompts)} items) ...", flush=True)

            vllm_outputs = llm.generate(batch_prompts, sp)

            for item, vllm_out in zip(batch_items, vllm_outputs):
                response  = vllm_out.outputs[0].text.strip()
                extracted = extract_boxed(response)
                think     = extract_think(response)
                is_mcq    = bool(item.get("options"))
                correct   = answers_match(extracted, item.get("answer"), is_mcq)

                state = item_state[item["id"]]
                state["attempts"] += 1

                enriched = dict(item)
                enriched["think"]            = think
                enriched["response"]         = response
                enriched["extracted_answer"] = extracted
                enriched["correct"]          = correct
                enriched["attempts_used"]    = state["attempts"]

                # Update best: prefer correct, then latest
                if state["best"] is None or correct:
                    state["best"] = enriched

                if correct:
                    state["done"] = True
                    status = "✓ CORRECT"
                else:
                    status = "✗ wrong"
                    if state["attempts"] >= args.max_attempts:
                        state["done"] = True
                        status += " (exhausted)"

                print(f"    id={item['id']:4d}  attempt={state['attempts']}  "
                      f"extracted={str(extracted)[:30]:30s}  {status}")

                all_outputs.append(response)

        # Count resolved this round
        n_correct_total = sum(1 for s in item_state.values() if s["done"] and s["best"] and s["best"]["correct"])
        n_done_total    = sum(1 for s in item_state.values() if s["done"])
        print(f"\n  After attempt {attempt_num}: "
              f"{n_done_total}/{len(todo)} done, "
              f"{n_correct_total}/{len(todo)} correct so far")

    # ── Collect final results ──────────────────────────────────────────────────
    print("\nCollecting final results ...")
    for item in todo:
        state = item_state[item["id"]]
        best  = state["best"]

        if best is None:
            # Shouldn't happen, but safety fallback
            best = dict(item)
            best["think"]            = ""
            best["response"]         = ""
            best["extracted_answer"] = None
            best["correct"]          = False
            best["attempts_used"]    = state["attempts"]

        results[item["id"]] = best

    # ── Write output in original order ─────────────────────────────────────────
    output_data = [results[item["id"]] for item in data if item["id"] in results]
    save_jsonl(output_data, args.output_path)

    # ── Summary ────────────────────────────────────────────────────────────────
    total         = len(output_data)
    n_correct     = sum(1 for d in output_data if d.get("correct"))
    n_has_think   = sum(1 for d in output_data if d.get("think"))
    avg_attempts  = (
        sum(d.get("attempts_used", 0) for d in output_data) / total
        if total else 0
    )
    attempts_dist = {}
    for d in output_data:
        a = d.get("attempts_used", 0)
        attempts_dist[a] = attempts_dist.get(a, 0) + 1

    print(f"\n{'='*60}")
    print(f"DONE — Results saved to {args.output_path}")
    print(f"{'='*60}")
    print(f"  Total items:      {total}")
    print(f"  Correct answers:  {n_correct} ({n_correct/total:.1%})")
    print(f"  Has think field:  {n_has_think}")
    print(f"  Avg attempts:     {avg_attempts:.2f}")
    print(f"  Attempts dist:    {dict(sorted(attempts_dist.items()))}")
    print()
    print("Tip: Use items where correct=True for LoRA SFT training.")
    print("     Use the full output (all items) for GRPO reward signal.")


if __name__ == "__main__":
    main()
