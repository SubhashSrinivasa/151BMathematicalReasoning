"""
generate_reasoning.py
─────────────────────
Populates a "think" field on every training item using AutoCoT:
  1. Embed all questions with SBERT
  2. Cluster with KMeans
  3. Generate full solutions for cluster representatives
  4. For every item, use its cluster's representative solutions as
     few-shot exemplars to generate a "think" field
  5. Save enriched JSONL ready for MathDataset

Usage:
    python generate_reasoning.py \
        --data-path data/public.jsonl \
        --output-path data/public_with_reasoning.jsonl \
        --model-id Qwen/Qwen3-4B-Thinking-2507 \
        --k 8 \
        --representatives-per-cluster 2
"""

import argparse
import gc
import json
import os
import re
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ── System prompts ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician reasoning. Think concisely. Step by step.\n\n "
    "MANDATORY RULE:\n"
    "- Show ALL reasoning inside <think>...</think> tags."
    "- The FINAL token MUST be: \\boxed{final_answer}\n "
    "- If multiple sub-answers are required, put them in one box separated by commas, e.g. \\boxed{3, 7}.\n "
    "- If this is not the final token, the response is invalid.\n\n"
    "FORMAT:\n"
    "\\boxed{final_answer}\n"
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician reasoning. Think concisely. Step by step.\n\n"
    "MANDATORY RULE:\n"
    "- Show ALL reasoning inside <think>...</think> tags."
    "- The FINAL token MUST be: \\boxed{X} where X is the answer choice given the options A-Z.\n "
    "- If this is not the final token, the response is invalid.\n\n"
    "FORMAT:\n"
    "\\boxed{X}\n"
)

# For representative generation — ask for boxed answer first, then explanation
SYSTEM_PROMPT_REP_MATH = (
    "You are an expert mathematician. "
    "MANDATORY RULE: If multiple sub-answers are required, put them in one box separated by commas e.g. \\boxed{3,7}. "
    "Do NOT think before answering and do NOT delay the boxed answer. "
    "THEN give a concise explanation of the steps reaching the answer."
)

SYSTEM_PROMPT_REP_MCQ = (
    "You are an expert mathematician. "
    "MANDATORY RULE: FIRST line must be the SINGLE best answer inside \\boxed{}, e.g. \\boxed{C}. "
    "If this is not the first token, the response is invalid. "
    "Do NOT think before answering and do NOT delay the boxed answer. "
    "Do not write more than 120 words."
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def item_to_text(item: dict) -> str:
    question = item["question"]
    options = item.get("options")
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = " ".join(f"{lbl}. {opt}" for lbl, opt in zip(labels, options))
        return f"Question: {question} Options: {opts_text}"
    return f"Question: {question}"


def build_rep_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Prompt for generating representative solutions (answer-first style)."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_REP_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_REP_MATH, question


def build_fewshot_prompt(
    item: dict,
    exemplars: list[dict],
    max_exemplar_words: int = 1200,
) -> tuple[str, str]:
    """
    Build a few-shot prompt for generating the think field.
    exemplars: list of {"question", "options", "solution"} dicts
    """
    parts = []

    for ex in exemplars:
        sol_words = ex["solution"].split()
        if len(sol_words) > max_exemplar_words:
            sol = " ".join(sol_words[:max_exemplar_words])
        else:
            sol = ex["solution"]

        q, opts = ex["question"], ex.get("options")
        if opts:
            labels = [chr(65 + i) for i in range(len(opts))]
            opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, opts))
            block = (
                f"Example Problem:\n{q}\n\n"
                f"Options:\n{opts_text}\n\n"
                f"Step-by-step Solution:\n{sol}\n"
            )
        else:
            block = f"Example Problem:\n{q}\n\nStep-by-step Solution:\n{sol}\n"

        parts.append(block)

    question, options = item["question"], item.get("options")

    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        target = (
            "Now solve this new problem in the same style.\n"
            "Show ALL reasoning inside <think>...</think> tags.\n"
            "End with \\boxed{answer}.\n\n"
            f"Problem:\n{question}\n\n"
            f"Options:\n{opts_text}"
        )
        system = SYSTEM_PROMPT_MCQ
    else:
        target = (
            "Now solve this new problem in the same style.\n"
            "Show ALL reasoning inside <think>...</think> tags.\n"
            "End with \\boxed{answer}.\n\n"
            f"Problem:\n{question}"
        )
        system = SYSTEM_PROMPT_MATH

    parts.append(target)
    return system, "\n\n".join(parts)


def shorten_solution(text: str, max_chars: int = 1200) -> str:
    """Keep up to the boxed answer, truncating after it."""
    if "\\boxed{" in text:
        box_pos = text.rfind("\\boxed{")
        return text[: box_pos + 200]
    return text[:max_chars]


def extract_think(response: str) -> str:
    """
    Pull reasoning from <think>...</think> if present.
    Falls back to the full response stripped of the boxed answer line.
    """
    m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: strip the final \boxed{} line
    lines = response.strip().splitlines()
    lines = [l for l in lines if not re.match(r"^\s*\\boxed\{", l)]
    return "\n".join(lines).strip()


def generate_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    prompts: list[str],
    batch_size: int = 32,
) -> list[str]:
    results = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        outputs = llm.generate(batch, sampling_params)
        for o in outputs:
            results.append(o.outputs[0].text.strip())
        print(f"  Generated {start + len(batch)}/{len(prompts)}", flush=True)
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path",           default="data/public.jsonl")
    p.add_argument("--output-path",         default="data/public_with_reasoning.jsonl")
    p.add_argument("--model-id",            default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--k",                   type=int,   default=8)
    p.add_argument("--representatives-per-cluster", type=int, default=2)
    p.add_argument("--max-model-len",       type=int,   default=8192)
    p.add_argument("--max-input-tokens",    type=int,   default=4000)
    p.add_argument("--max-new-tokens",      type=int,   default=3000)
    p.add_argument("--rep-max-new-tokens",  type=int,   default=1000)
    p.add_argument("--batch-size",          type=int,   default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--resume",              action="store_true",
                   help="Skip items that already have a 'think' field in output file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading data from {args.data_path}")
    data = load_jsonl(args.data_path)
    print(f"Loaded {len(data)} items")

    # ── Resume: load already-processed items ──────────────────────────────────
    done_ids: set = set()
    existing: dict = {}
    if args.resume and os.path.exists(args.output_path):
        for item in load_jsonl(args.output_path):
            if "think" in item:
                done_ids.add(item["id"])
                existing[item["id"]] = item
        print(f"Resuming — {len(done_ids)} items already have 'think' field")

    # ── SBERT embeddings ──────────────────────────────────────────────────────
    print("Computing SBERT embeddings...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [item_to_text(item) for item in data]
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # ── KMeans clustering ─────────────────────────────────────────────────────
    print(f"Clustering into k={args.k} clusters...")
    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)

    # Find representative indices per cluster (closest to centroid)
    cluster_rep_indices: dict[int, list[int]] = {}
    for cid in range(args.k):
        idxs = np.where(cluster_ids == cid)[0]
        embeds = embeddings[idxs]
        centroid = kmeans.cluster_centers_[cid]
        dists = np.linalg.norm(embeds - centroid, axis=1)
        top = np.argsort(dists)[: args.representatives_per_cluster]
        cluster_rep_indices[cid] = idxs[top].tolist()
        print(f"  Cluster {cid}: {len(idxs)} items, reps at global indices {cluster_rep_indices[cid]}")

    del embedder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Load tokenizer + vLLM ─────────────────────────────────────────────────
    print(f"Loading tokenizer and vLLM from {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_fast=False,
        padding_side="left",
    )

    llm = LLM(
        model=args.model_id,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=8,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )

    # ── Step 1: Generate representative solutions ─────────────────────────────
    print("Generating representative solutions...")
    rep_sampling = SamplingParams(temperature=0.0, max_tokens=args.rep_max_new_tokens)

    rep_prompts, rep_items = [], []
    for rep_idxs in cluster_rep_indices.values():
        for idx in rep_idxs:
            item = data[idx]
            system, user = build_rep_prompt(item["question"], item.get("options"))
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
            rep_prompts.append(prompt_text)
            rep_items.append(item)

    rep_outputs = generate_batch(llm, rep_sampling, rep_prompts, batch_size=args.batch_size)

    # Store representative solutions keyed by item id
    rep_solutions: dict = {}
    for item, output in zip(rep_items, rep_outputs):
        sol = shorten_solution(output)
        if "\\boxed{" in sol:
            rep_solutions[item["id"]] = {
                "question": item["question"],
                "options":  item.get("options"),
                "solution": sol,
            }
            print(f"  Rep id={item['id']} OK — {len(sol)} chars")
        else:
            print(f"  Rep id={item['id']} SKIPPED — no \\boxed{{}} in output")

    # ── Step 2: Generate think field for every training item ──────────────────
    print("Building few-shot prompts for all training items...")
    think_sampling = SamplingParams(temperature=0.6, max_tokens=args.max_new_tokens)

    prompts, prompt_items = [], []

    for i, (item, cid) in enumerate(zip(data, cluster_ids)):
        if item["id"] in done_ids:
            continue

        # Get exemplars from this item's cluster
        rep_idxs = cluster_rep_indices[int(cid)]
        exemplars = []
        for idx in rep_idxs:
            ex_item = data[idx]
            if ex_item["id"] == item["id"]:
                continue  # don't use item as its own exemplar
            sol_entry = rep_solutions.get(ex_item["id"])
            if sol_entry:
                exemplars.append(sol_entry)

        # Try to build few-shot prompt, shrinking exemplars if too long
        final_prompt = None
        max_words = 1200
        while max_words >= 200:
            if exemplars:
                system, user = build_fewshot_prompt(item, exemplars, max_exemplar_words=max_words)
            else:
                # No valid exemplars — fall back to zero-shot
                system, user = build_rep_prompt(item["question"], item.get("options"))

            prompt_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
            tok_len = len(tokenizer.encode(prompt_text))
            if tok_len <= args.max_input_tokens:
                final_prompt = prompt_text
                break
            max_words -= 200

        if final_prompt is None:
            print(f"  Item id={item['id']} skipped — prompt too long even at 200 words")
            continue

        prompts.append(final_prompt)
        prompt_items.append(item)

    print(f"Generating think fields for {len(prompts)} items...")
    think_outputs = generate_batch(llm, think_sampling, prompts, batch_size=args.batch_size)

    # ── Assemble enriched dataset ─────────────────────────────────────────────
    # Start with already-done items
    enriched = {item_id: ex_item for item_id, ex_item in existing.items()}

    for item, output in zip(prompt_items, think_outputs):
        think = extract_think(output)
        enriched_item = dict(item)
        enriched_item["think"] = think
        # Also store the full response so you can inspect it
        enriched_item["think_raw"] = output
        enriched[item["id"]] = enriched_item

    # Items that were skipped (prompt too long, no exemplars, etc.) — save without think
    all_ids = {item["id"] for item in data}
    missing = all_ids - set(enriched.keys())
    for item in data:
        if item["id"] in missing:
            enriched[item["id"]] = dict(item)
            print(f"  Warning: item id={item['id']} has no think field")

    # Write out in original order
    output_data = [enriched[item["id"]] for item in data]
    save_jsonl(output_data, args.output_path)

    # Summary
    n_with_think = sum(1 for item in output_data if "think" in item and item["think"])
    n_with_box   = sum(1 for item in output_data if "think_raw" in item and "\\boxed{" in item.get("think_raw", ""))
    print(f"\nDone!")
    print(f"  {n_with_think}/{len(output_data)} items have a 'think' field")
    print(f"  {n_with_box}/{len(output_data)} items have \\boxed{{}} in their raw output")
    print(f"  Saved to {args.output_path}")


if __name__ == "__main__":
    main()
