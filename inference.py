import os
import sys
import json
import re
import random
import argparse
import time
from pathlib import Path
from typing import Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen3-4B-Thinking-2507"
ADAPTER_PATH  = "data/lora_math_adaper/lora_math_adapter/final_adapter"
DATA_PATH     = "data/public.jsonl"
OUTPUT_PATH   = "results/inference_results.csv"
MAX_NEW_TOKENS = 8192
N_SAMPLES     = 200   # set to None to run on full dataset
SEED          = 42
BATCH_SIZE    = 4

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "You MUST always attempt the problem and provide a final answer in \\boxed{}. "
    "Never say the problem is too complex or that you cannot answer. "
    "Even if uncertain, make your best attempt. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "You MUST always choose an answer — never say the problem is too complex. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def extract_boxed(text: str) -> Optional[str]:
    """Extract last \\boxed{} content from text — post-processing fix for truncation."""
    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        return None
    # take the last \boxed{} found
    last = matches[-1]
    start = last.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    content = text[start:i - 1].strip()
    return content if content else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default=MODEL_ID)
    p.add_argument("--adapter",      default=ADAPTER_PATH, help="LoRA adapter path, or 'none' for base model")
    p.add_argument("--data",         default=DATA_PATH)
    p.add_argument("--output",       default=OUTPUT_PATH)
    p.add_argument("--n-samples",    type=int, default=N_SAMPLES)
    p.add_argument("--max-tokens",   type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    p.add_argument("--seed",         type=int, default=SEED)
    p.add_argument("--no-adapter",   action="store_true", help="Run base model without LoRA")
    p.add_argument("--score",        action="store_true", help="Score against ground truth (public set only)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    print(f"Loading data from {args.data}...")
    data = [json.loads(l) for l in open(args.data) if l.strip()]

    # Random sample
    if args.n_samples and args.n_samples < len(data):
        data = random.sample(data, args.n_samples)
        print(f"Randomly sampled {args.n_samples} questions (seed={args.seed})")
    print(f"Total questions: {len(data)}")

    print("\nLoading model...")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if not args.no_adapter and args.adapter.lower() != "none":
        adapter_path = args.adapter
        if Path(adapter_path).exists():
            print(f"Loading LoRA adapter from {adapter_path}...")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()
            print("Adapter merged.")
        else:
            print(f"WARNING: adapter path {adapter_path} not found — running base model")
    else:
        print("Running base model (no adapter)")

    model.eval()

    # Build prompts
    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(prompt)

    # Run inference in batches
    print(f"\nRunning inference (batch_size={args.batch_size}, max_tokens={args.max_tokens})...")
    os.makedirs(Path(args.output).parent, exist_ok=True)

    responses = []
    t0 = time.time()

    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i:i + args.batch_size]
        batch_items = data[i:i + args.batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        for j, (output, item) in enumerate(zip(outputs, batch_items)):
            new_tokens = output[input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append({"id": item["id"], "response": response})

        elapsed = time.time() - t0
        done = min(i + args.batch_size, len(prompts))
        rate = done / elapsed
        eta = (len(prompts) - done) / rate if rate > 0 else 0
        print(f"  {done}/{len(prompts)} done — {elapsed/60:.1f}min elapsed, ~{eta/60:.1f}min remaining")

    # Save as CSV
    import csv
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()
        writer.writerows(responses)
    print(f"\nSaved {len(responses)} responses to {args.output}")

    # Score if requested and answers available
    if args.score:
        print("\nScoring...")
        sys.path.insert(0, ".")
        try:
            from judger import Judger
            judger = Judger(strict_extract=False)
        except ImportError:
            print("judger.py not found — skipping scoring")
            return

        correct = 0
        no_boxed = 0
        for item, resp in zip(data, responses):
            text = resp["response"]
            boxed = extract_boxed(text)

            if boxed is None:
                no_boxed += 1
                continue

            gold = item.get("answer", [])
            if not isinstance(gold, list):
                gold = [gold]

            if item.get("options"):
                # MCQ
                letter = re.search(r'\b([A-Z])\b', boxed.upper())
                pred_letter = letter.group(1) if letter else ""
                if pred_letter == str(gold[0]).strip().upper():
                    correct += 1
            else:
                try:
                    result = judger.auto_judge(pred=text, gold=gold, options=[[]] * len(gold))
                    if result:
                        correct += 1
                except Exception:
                    pass

        total = len(data)
        print(f"\n=== RESULTS ===")
        print(f"Total:      {total}")
        print(f"No boxed:   {no_boxed} ({no_boxed/total:.1%})")
        print(f"Correct:    {correct} ({correct/total:.1%})")
        print(f"Accuracy:   {correct/(total-no_boxed):.1%} (of answered)")

    print("\nDone!")


if __name__ == "__main__":
    main()
