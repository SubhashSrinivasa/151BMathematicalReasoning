                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  autoImplement.py
import argparse
import gc
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT_MATH_REPRESENTATIVES = (
    "You are an expert mathematician. "
    "MANDATORY RULE: If multiple sub-answers are required, put them in one box separated by commas e.g. \\boxed{3,7}. "
    "Do NOT think before answering and do NOT delay the boxed answer. "
    "THEN give a concise explanation of the steps reaching the answer."
)

SYSTEM_PROMPT_MCQ_REPRESENTATIVES = (
    "You are an expert mathematician. "
    "MANDATORY RULE: FIRST line must be the SINGLE best answer inside \\boxed{}, e.g. \\boxed{C}. "
    "If this is not the first token, the response is invalid. "
    "Do NOT think before answering and do NOT delay the boxed answer. "
    "Do not write more than 120 words."
)

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. "
    "MANDATORY RULE: If multiple sub-answers are required, put them in one box separated by commas e.g. \\boxed{3,7}. "
    "If this is not the first token, the response is invalid. "
    "Do NOT think before answering and do NOT delay the boxed answer. "
    "THEN give a concise explanation."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "MANDATORY RULE: FIRST line must be the SINGLE best answer inside \\boxed{}, e.g. \\boxed{C}. "
    "If this is not the first token, the response is invalid. "
    "Do NOT think before answering and do NOT delay the boxed answer. "
    "Do not write more than 120 words."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--data-path", default="data/private.jsonl")
    parser.add_argument("--output-csv", default="submission_partial.csv")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--representatives-per-cluster", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=500, help="Number of remaining questions to run this time.")
    parser.add_argument("--vllm-batch-size", type=int, default=40)
    parser.add_argument("--rep-vllm-batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=32000)
    parser.add_argument("--max-input-tokens", type=int, default=20000)
    parser.add_argument("--max-new-tokens", type=int, default=10000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def item_to_text(item: dict) -> str:
    question = item["question"]
    options = item.get("options")
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = " ".join(f"{label}. {opt}" for label, opt in zip(labels, options))
        return f"Question: {question} Options: {opts_text}"
    return f"Question: {question}"


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{label}. {opt.strip()}" for label, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ_REPRESENTATIVES, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH_REPRESENTATIVES, question


def build_fewshot_prompt(item: dict, exemplars: list[dict]) -> tuple[str, str]:
    parts = []

    for ex in exemplars:
        question = ex["question"]
        options = ex.get("options")
        solution = ex["solution"]

        if options:
            labels = [chr(65 + i) for i in range(len(options))]
            opts_text = "\n".join(f"{label}. {opt.strip()}" for label, opt in zip(labels, options))
            block = (
                f"Example Problem:\n{question}\n\n"
                f"Options:\n{opts_text}\n\n"
                f"Step-by-step Solution:\n{solution}\n"
            )
        else:
            block = f"Example Problem:\n{question}\n\nStep-by-step Solution:\n{solution}\n"

        parts.append(block)

    question = item["question"]
    options = item.get("options")

    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{label}. {opt.strip()}" for label, opt in zip(labels, options))
        target = (
            "Now solve this new problem in the same style.\n"
            "The FIRST line must include the final answer inside \\boxed{}.\n\n"
            f"Problem:\n{question}\n\n"
            f"Options:\n{opts_text}"
        )
        system_prompt = SYSTEM_PROMPT_MCQ
    else:
        target = (
            "Now solve this new problem in the same style.\n"
            "The FIRST line must include the final answer inside \\boxed{}.\n\n"
            f"Problem:\n{question}"
        )
        system_prompt = SYSTEM_PROMPT_MATH

    parts.append(target)
    return system_prompt, "\n\n".join(parts)


def trim_solution_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def shorten_solution(text: str, max_chars: int = 1200) -> str:
    if "\\boxed{" in text:
        box_pos = text.rfind("\\boxed{")
        return text[: box_pos + 200]
    return text[:max_chars]


def generate_batch_vllm(llm: LLM, sampling_params: SamplingParams, prompts: list[str], batch_size: int) -> list[str]:
    all_responses = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)

        for output in outputs:
            all_responses.append(output.outputs[0].text.strip())

        print(f"Generated {start + len(batch_prompts)} / {len(prompts)}", flush=True)

    return all_responses


def append_row_csv(path: str, row: dict) -> None:
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False,
    )


def main() -> None:
    args = parse_args()

    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("Current folder:", os.getcwd())

    data = load_jsonl(args.data_path)
    n_mcq = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options") for d in data)
    print(f"Loaded {len(data)} questions ({n_mcq} MCQ, {n_free} free-form)")

    # ----------------------------
    # SBERT embeddings + clustering
    # ----------------------------
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [item_to_text(item) for item in data]

    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print("Embeddings:", embeddings.shape)

    kmeans = KMeans(n_clusters=args.k, random_state=23, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)

    cluster_rep_indices = {}
    for cid in range(args.k):
        cluster_idxs = np.where(cluster_ids == cid)[0]
        cluster_embeds = embeddings[cluster_idxs]
        centroid = kmeans.cluster_centers_[cid]
        dists = np.linalg.norm(cluster_embeds - centroid, axis=1)
        sorted_local = np.argsort(dists)[: args.representatives_per_cluster]
        chosen_global = cluster_idxs[sorted_local]
        cluster_rep_indices[cid] = chosen_global.tolist()

    print("Representative indices:", cluster_rep_indices)

    # Free memory before loading LLM.
    del embedder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ----------------------------
    # Load tokenizer + vLLM
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_fast=False,
        padding_side="left",
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    llm = LLM(
        model=args.model_id,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=args.max_model_len,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )
    # ----------------------------
    # Generate representative solutions
    # ----------------------------
    rep_prompts = []
    rep_items = []

    for rep_idxs in cluster_rep_indices.values():
        for idx in rep_idxs:
            item = data[idx]
            system, user = build_prompt(item["question"], item.get("options"))
            prompt_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            rep_prompts.append(prompt_text)
            rep_items.append(item)

    print(f"Generating representative solutions for {len(rep_prompts)} examples...")
    rep_outputs = generate_batch_vllm(
        llm,
        sampling_params,
        rep_prompts,
        batch_size=args.rep_vllm_batch_size,
    )

    rep_solutions = {
        item["id"]: shorten_solution(output)
        for item, output in zip(rep_items, rep_outputs)
    }
    print("Representative solutions generated!")

    # ----------------------------
    # Load existing progress
    # ----------------------------
    if os.path.exists(args.output_csv):
        existing_df = pd.read_csv(args.output_csv)
        existing_df = existing_df.dropna(subset=["id"])
        existing_df["id"] = existing_df["id"].astype(str)
        completed_ids = set(existing_df["id"])
        print(f"Found {len(completed_ids)} completed answers.")
    else:
        completed_ids = set()
        print("No existing CSV found.")

    for item, cid in zip(data, cluster_ids):
        item["cluster_id"] = int(cid)

    remaining_items = [item for item in data if str(item["id"]) not in completed_ids]
    batch_items = remaining_items[: args.batch_size]

    print(f"Total questions: {len(data)}")
    print(f"Remaining questions: {len(remaining_items)}")
    print(f"Running {len(batch_items)} questions this time.")

    # ----------------------------
    # Build final prompts
    # ----------------------------
    prompts = []
    prompt_items = []

    for item in batch_items:
        cid = item["cluster_id"]
        rep_idxs = cluster_rep_indices[cid][:2]

        exemplar_candidates = []

        for idx in rep_idxs:
            ex_item = data[idx]

            if ex_item["id"] == item["id"]:
                continue

            sol = rep_solutions.get(ex_item["id"], "")
            if not sol or "\\boxed" not in sol:
                continue

            exemplar_candidates.append({
                "question": ex_item["question"],
                "options": ex_item.get("options"),
                "solution": sol,
            })

        final_prompt = None
        max_words = 1200

        while max_words >= 200:
            exemplars = []

            for ex in exemplar_candidates:
                exemplars.append({
                     "question": ex["question"],
                     "options": ex["options"],
                     "solution": trim_solution_words(ex["solution"], max_words),
                })

            if len(exemplars) == 0:
               break

            system, user = build_fewshot_prompt(item, exemplars)
            prompt_text = tokenizer.apply_chat_template(
                [
                        {"role": "system", "content": system},
                     {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            tok_len = len(tokenizer.encode(prompt_text))
            if tok_len <= args.max_input_tokens:
                final_prompt = prompt_text
                break

            max_words -= 200

        # Fallback: no exemplar.
        if final_prompt is None:
             system, user = build_prompt(item["question"], item.get("options"))
             prompt_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
             )

             tok_len = len(tokenizer.encode(prompt_text))
             if tok_len <= args.max_input_tokens:
                final_prompt = prompt_text

        if final_prompt is None:
            append_row_csv(args.output_csv, {"id": item["id"], "answer": "unsolvable"})
            print(f"Marked unsolvable | id={item['id']}")
            continue

        prompts.append(final_prompt)
        prompt_items.append(item)

    print(f"Sending {len(prompts)} prompts to vLLM.")

    # ----------------------------
    # Generate + save incrementally
    # ----------------------------
    outputs = generate_batch_vllm(
        llm,
        sampling_params,
        prompts,
        batch_size=args.vllm_batch_size,
    )

    for item, output in zip(prompt_items, outputs):
        append_row_csv(args.output_csv, {"id": item["id"], "answer": output})
        print(f"Saved id={item['id']}")

    print("Batch complete!")


if __name__ == "__main__":
    main()
