"""
inference.py — CSE 151B Competition
====================================
Combines three methodologies:
  1. Auto Chain-of-Thought (AutoCoT)  — SBERT embeddings + KMeans clustering to
                                        select per-cluster few-shot exemplars
  2. Self-Consistency                 — sample N completions per question and
                                        majority-vote the boxed answer
  3. LoRA / Reflection                — optional LoRA adapter loading + a second
                                        reflection pass that re-evaluates the
                                        majority-voted answer

Pipeline per question
─────────────────────
  [AutoCoT] Build few-shot prompt from nearest cluster representatives
      ↓
  [Self-Consistency] Generate N completions → majority vote → best answer
      ↓
  [Reflection] Feed best answer back to the model for one refinement pass
      ↓
  Final answer saved to JSONL

ONE-TIME setup:
    pip install "numpy<2" torch transformers accelerate peft \
                sentence-transformers scikit-learn vllm tqdm bitsandbytes \
                antlr4-python3-runtime==4.11.1

Run:
    python inference.py [--lora-path path/to/adapter] [--no-reflection] ...
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ── Set env vars BEFORE importing vllm ───────────────────────────────────────
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified inference: AutoCoT + Self-Consistency + LoRA Reflection")
    # Paths
    p.add_argument("--model-id",       default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--lora-path",      default=None,
                   help="Path to a LoRA adapter directory (optional). "
                        "If omitted, the base model is used as-is.")
    p.add_argument("--data-path",      default="data/private.jsonl")
    p.add_argument("--output-path",    default="results/inference_results.jsonl")
    # Dataset
    p.add_argument("--n-samples",      type=int, default=None,
                   help="Cap the number of questions (None = full dataset).")
    p.add_argument("--save-eval",      action="store_true",
                   help="Include gold answers and correctness in the output.")
    # AutoCoT clustering
    p.add_argument("--k",              type=int, default=8,
                   help="Number of KMeans clusters for AutoCoT.")
    p.add_argument("--reps-per-cluster", type=int, default=2,
                   help="Representative exemplars drawn from each cluster.")
    p.add_argument("--max-input-tokens", type=int, default=18000,
                   help="Max prompt tokens before falling back to zero-shot.")
    # Self-consistency
    p.add_argument("--sc-samples",     type=int, default=5,
                   help="Number of completions sampled per question for majority vote.")
    # Reflection
    p.add_argument("--no-reflection",  action="store_true",
                   help="Skip the reflection (second-pass) step.")
    p.add_argument("--max-tokens-think",   type=int, default=7000)
    p.add_argument("--max-tokens-reflect", type=int, default=1000)
    # vLLM
    p.add_argument("--max-model-len",  type=int, default=30000)
    p.add_argument("--vllm-batch-size", type=int, default=40)
    p.add_argument("--rep-vllm-batch-size", type=int, default=8)
    p.add_argument("--gpu-mem-util",   type=float, default=0.90)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_MATH_REP = (
    "You are an expert mathematician. "
    "MANDATORY RULE: If multiple sub-answers are required, put them in one box "
    "separated by commas e.g. \\boxed{3,7}. "
    "Do NOT think before answering and do NOT delay the boxed answer. "
    "THEN give a concise explanation of the steps reaching the answer."
)

SYSTEM_MCQ_REP = (
    "You are an expert mathematician. "
    "MANDATORY RULE: FIRST line must be the SINGLE best answer inside \\boxed{}, "
    "e.g. \\boxed{C}. "
    "If this is not the first token, the response is invalid. "
    "Do NOT think before answering and do NOT delay the boxed answer. "
    "Do not write more than 120 words."
)

SYSTEM_MATH = (
    "You are an expert mathematician. "
    "MANDATORY RULE: If multiple sub-answers are required, put them in one box "
    "separated by commas e.g. \\boxed{3,7}. "
    "If this is not the first token, the response is invalid. "
    "Do NOT think before answering and do NOT delay the boxed answer. "
    "THEN give a concise explanation."
)

SYSTEM_MCQ = (
    "You are an expert mathematician. "
    "MANDATORY RULE: FIRST line must be the SINGLE best answer inside \\boxed{}, "
    "e.g. \\boxed{C}. "
    "If this is not the first token, the response is invalid. "
    "Do NOT think before answering and do NOT delay the boxed answer. "
    "Do not write more than 120 words."
)

SYSTEM_REFLECT_FREE = (
    "Pick ONLY ONE method to reevaluate the answer to the question. "
    "MANDATORY RULE: "
    "- Final answer MUST be: \\boxed{X} where X = correct answer. "
    "- If multiple sub-answers are required, put them in one box separated by commas, "
    "e.g. \\boxed{3, 7}. "
    "- \\boxed{answer} MUST EXIST within the first 100 tokens. "
    "CONSTRAINTS: "
    "- DO NOT evaluate using more than ONE METHOD."
)

SYSTEM_REFLECT_MCQ = (
    "Pick ONLY ONE method to reevaluate the answer to the question. "
    "MANDATORY RULE: "
    "- Final answer MUST be: \\boxed{X} where X = correct option. "
    "- \\boxed{answer} MUST EXIST within the first 100 tokens. "
    "CONSTRAINTS: "
    "- DO NOT evaluate using more than ONE METHOD."
)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def build_rep_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Prompt used when generating representative / exemplar solutions."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{l}. {o.strip()}" for l, o in zip(labels, options))
        return SYSTEM_MCQ_REP, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_MATH_REP, question


def build_fewshot_prompt(item: dict, exemplars: list[dict]) -> tuple[str, str]:
    """AutoCoT: inject per-cluster exemplar solutions before the target question."""
    parts = []
    for ex in exemplars:
        q, opts, sol = ex["question"], ex.get("options"), ex["solution"]
        if opts:
            labels    = [chr(65 + i) for i in range(len(opts))]
            opts_text = "\n".join(f"{l}. {o.strip()}" for l, o in zip(labels, opts))
            block = (
                f"Example Problem:\n{q}\n\n"
                f"Options:\n{opts_text}\n\n"
                f"Step-by-step Solution:\n{sol}\n"
            )
        else:
            block = f"Example Problem:\n{q}\n\nStep-by-step Solution:\n{sol}\n"
        parts.append(block)

    q    = item["question"]
    opts = item.get("options")
    if opts:
        labels    = [chr(65 + i) for i in range(len(opts))]
        opts_text = "\n".join(f"{l}. {o.strip()}" for l, o in zip(labels, opts))
        target = (
            "Now solve this new problem in the same style.\n"
            "The FIRST line must include the final answer inside \\boxed{}.\n\n"
            f"Problem:\n{q}\n\nOptions:\n{opts_text}"
        )
        system = SYSTEM_MCQ
    else:
        target = (
            "Now solve this new problem in the same style.\n"
            "The FIRST line must include the final answer inside \\boxed{}.\n\n"
            f"Problem:\n{q}"
        )
        system = SYSTEM_MATH

    parts.append(target)
    return system, "\n\n".join(parts)


def build_reflection_system(options: Optional[list]) -> str:
    return SYSTEM_REFLECT_MCQ if options else SYSTEM_REFLECT_FREE


# ─────────────────────────────────────────────────────────────────────────────
# Answer utilities
# ─────────────────────────────────────────────────────────────────────────────

def extract_boxed(text: str) -> Optional[str]:
    matches = re.findall(r"\\boxed\{(.+?)\}", text)
    return matches[-1].strip() if matches else None


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


def majority_vote(texts: list[str]) -> str:
    """
    Self-consistency majority vote over a list of raw model outputs.
    Returns the most common boxed answer wrapped in \\boxed{}, or the
    first raw text if no boxed answer is found in any completion.
    """
    answers = [extract_boxed(t) for t in texts]
    valid   = [a for a in answers if a]
    if not valid:
        return texts[0].strip()
    winner, _ = Counter(valid).most_common(1)[0]
    return f"\\boxed{{{winner}}}"


def shorten_solution(text: str, max_chars: int = 1200) -> str:
    if "\\boxed{" in text:
        pos = text.rfind("\\boxed{")
        return text[: pos + 200]
    return text[:max_chars]


def trim_solution_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def item_to_text(item: dict) -> str:
    q    = item["question"]
    opts = item.get("options")
    if opts:
        labels    = [chr(65 + i) for i in range(len(opts))]
        opts_text = " ".join(f"{l}. {o}" for l, o in zip(labels, opts))
        return f"Question: {q} Options: {opts_text}"
    return f"Question: {q}"


# ─────────────────────────────────────────────────────────────────────────────
# vLLM helpers
# ─────────────────────────────────────────────────────────────────────────────

def batched_generate(llm, sampling_params, prompts: list[str], batch_size: int, save_callback=None) -> list[list[str]]:
    """
    Returns a list of lists: for each prompt, a list of `n` completion strings
    (n = sampling_params.n).
    Optionally calls save_callback(batch_index, batch_outputs) after each batch.
    """
    all_outputs: list[list[str]] = []
    for start in range(0, len(prompts), batch_size):
        chunk   = prompts[start : start + batch_size]
        results = llm.generate(chunk, sampling_params)
        batch_outputs = []
        for req_out in results:
            batch_outputs.append([c.text.strip() for c in req_out.outputs])
        all_outputs.extend(batch_outputs)
        if save_callback:
            save_callback(start, batch_outputs)
        print(f"  Generated {start + len(chunk)} / {len(prompts)}", flush=True)
    return all_outputs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    t_start = time.time()

    def banner(msg):
        print(f"\n{'='*70}\n[{time.strftime('%H:%M:%S')}] {msg}\n{'='*70}", flush=True)

    def step(msg):
        print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    # ── CUDA ─────────────────────────────────────────────────────────────────
    banner("STEP 1 / 8  CUDA check")
    step(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available — aborting.")
    step(f"GPU 0: {torch.cuda.get_device_name(0)}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    banner("STEP 2 / 8  Load dataset")
    data = [json.loads(line) for line in open(args.data_path)]
    if args.n_samples:
        data = data[: args.n_samples]
    # ── Resume: load already completed IDs ───────────────────────────────────
    completed_ids = set()
    out_path = Path(args.output_path)
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("id") is not None:
                        completed_ids.add(str(r["id"]))
                except Exception:
                    pass
        step(f"Resuming — found {len(completed_ids)} already completed questions, skipping them.")
    else:
        step("No existing results found — starting fresh.")

    # Filter out already completed items
    data = [d for d in data if str(d.get("id")) not in completed_ids]
    step(f"Remaining questions to process: {len(data)}")

    # ── AutoCoT: SBERT embeddings + KMeans ───────────────────────────────────
    banner("STEP 3 / 8  AutoCoT — embed + cluster")
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    texts      = [item_to_text(item) for item in data]
    embeddings = embedder.encode(
        texts, batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    step(f"Embeddings shape: {embeddings.shape}")

    k       = min(args.k, len(data))
    kmeans  = KMeans(n_clusters=k, random_state=23, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)

    cluster_rep_indices: dict[int, list[int]] = {}
    for cid in range(k):
        idxs     = np.where(cluster_ids == cid)[0]
        centroid = kmeans.cluster_centers_[cid]
        dists    = np.linalg.norm(embeddings[idxs] - centroid, axis=1)
        chosen   = idxs[np.argsort(dists)[: args.reps_per_cluster]]
        cluster_rep_indices[cid] = chosen.tolist()

    for item, cid in zip(data, cluster_ids):
        item["cluster_id"] = int(cid)

    del embedder
    gc.collect()
    torch.cuda.empty_cache()
    step("Clustering done; embedder freed.")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    banner("STEP 4 / 8  Load tokenizer")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, use_fast=False, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    step("Tokenizer ready.")

    # ── Load vLLM (optionally with LoRA) ─────────────────────────────────────
    banner("STEP 5 / 8  Load vLLM model")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    lora_request = None
    llm_kwargs   = dict(
        model                   = args.model_id,
        quantization            = "bitsandbytes",
        load_format             = "bitsandbytes",
        enforce_eager           = True,
        gpu_memory_utilization  = args.gpu_mem_util,
        max_model_len           = args.max_model_len,
        max_num_seqs            = 4,
        max_num_batched_tokens  = args.max_model_len,
        dtype                   = "float16",
        trust_remote_code       = True,
        disable_custom_all_reduce = True,
    )

    if args.lora_path:
        step(f"LoRA adapter detected at: {args.lora_path}")
        llm_kwargs["enable_lora"] = True
        lora_request = LoRARequest("lora_adapter", 1, args.lora_path)

    t0  = time.time()
    llm = LLM(**llm_kwargs)
    step(f"LLM ready in {time.time() - t0:.1f}s.")

    # Sampling params: n > 1 enables self-consistency
    sc_params = SamplingParams(
        n                 = args.sc_samples,
        max_tokens        = args.max_tokens_think,
        temperature       = 0.6,
        top_p             = 0.95,
        top_k             = 20,
        min_p             = 0.0,
        repetition_penalty= 1.0,
    )
    # Reflection uses greedy / low temperature
    reflect_params = SamplingParams(
        n                 = 1,
        max_tokens        = args.max_tokens_reflect,
        temperature       = 0.3,
        min_p             = 0.1,
        repetition_penalty= 1.0,
    )
    step(f"Sampling: sc_samples={args.sc_samples}, reflect={'enabled' if not args.no_reflection else 'disabled'}")

    # ── Generate representative (exemplar) solutions ──────────────────────────
    banner("STEP 6 / 8  Generate AutoCoT exemplar solutions")
    rep_prompts: list[str] = []
    rep_items:   list[dict] = []

    for rep_idxs in cluster_rep_indices.values():
        for idx in rep_idxs:
            item        = data[idx]
            system, user = build_rep_prompt(item["question"], item.get("options"))
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
            rep_prompts.append(prompt_text)
            rep_items.append(item)

    step(f"Generating solutions for {len(rep_prompts)} representative examples …")
    # Use n=1, greedy for exemplars (deterministic, compact)
    rep_params = SamplingParams(n=1, max_tokens=args.max_tokens_think, temperature=0.0)
    rep_raw    = batched_generate(llm, rep_params, rep_prompts, args.rep_vllm_batch_size)

    rep_solutions: dict[str, str] = {
        item["id"]: shorten_solution(outputs[0])
        for item, outputs in zip(rep_items, rep_raw)
    }
    step(f"Exemplar solutions ready ({len(rep_solutions)}).")

    # ── Build AutoCoT few-shot prompts ────────────────────────────────────────
    banner("STEP 7 / 8  Build few-shot prompts → Self-Consistency → Reflection")
    prompts:      list[str]  = []
    prompt_items: list[dict] = []

    for item in data:
        cid      = item["cluster_id"]
        rep_idxs = cluster_rep_indices[cid][: args.reps_per_cluster]

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
                "options":  ex_item.get("options"),
                "solution": sol,
            })

        final_prompt = None
        max_words    = 1200

        while max_words >= 200:
            exemplars = [
                {**ex, "solution": trim_solution_words(ex["solution"], max_words)}
                for ex in exemplar_candidates
            ]
            if not exemplars:
                break
            system, user = build_fewshot_prompt(item, exemplars)
            prompt_text  = tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
            if len(tokenizer.encode(prompt_text)) <= args.max_input_tokens:
                final_prompt = prompt_text
                break
            max_words -= 200

        # Fallback: zero-shot
        if final_prompt is None:
            system, user = build_rep_prompt(item["question"], item.get("options"))
            prompt_text  = tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
            if len(tokenizer.encode(prompt_text)) <= args.max_input_tokens:
                final_prompt = prompt_text

        if final_prompt is None:
            step(f"  WARNING: prompt too long for id={item['id']} — skipping.")
            continue

        prompts.append(final_prompt)
        prompt_items.append(item)

    step(f"Built {len(prompts)} prompts.")

    # ── Self-Consistency: generate N completions + majority vote ──────────────
    step("Running self-consistency generation …")

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save incrementally after each batch so progress is never lost
    def save_sc_batch(start_idx: int, batch_completions: list[list[str]]) -> None:
        with open(out_path, "a") as f:
            for i, completions in enumerate(batch_completions):
                item     = prompt_items[start_idx + i]
                response = majority_vote(completions)
                record   = {"id": item.get("id"), "is_mcq": bool(item.get("options")), "response": response, "_sc_done": True}
                f.write(json.dumps(record) + "\n")

    all_completions = batched_generate(llm, sc_params, prompts, args.vllm_batch_size, save_callback=save_sc_batch)
    sc_responses = [majority_vote(completions) for completions in all_completions]
    step("Self-consistency majority vote complete.")

    # ── Reflection pass ───────────────────────────────────────────────────────
    if args.no_reflection:
        final_responses = sc_responses
        step("Reflection skipped (--no-reflection).")
    else:
        step("Building reflection prompts …")
        reflect_prompts: list[str] = []
        for item, sc_resp in zip(prompt_items, sc_responses):
            sys_r   = build_reflection_system(item.get("options"))
            _, user = build_rep_prompt(item["question"], item.get("options"))
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "system",    "content": sys_r},
                 {"role": "user",      "content": user},
                 {"role": "assistant", "content": sc_resp},
                 {"role": "user",      "content": "Please reevaluate your answer."}],
                tokenize=False, add_generation_prompt=True,
            )
            reflect_prompts.append(prompt_text)

        step(f"Running reflection for {len(reflect_prompts)} questions …")
        reflect_raw      = batched_generate(llm, reflect_params, reflect_prompts, args.vllm_batch_size)
        final_responses  = [outputs[0] for outputs in reflect_raw]
        step("Reflection complete.")

    # ── Score ─────────────────────────────────────────────────────────────────
    banner("STEP 8 / 8  Score & save")
    sys.path.insert(0, ".")
    from judger import Judger
    from tqdm import tqdm

    judger  = Judger(strict_extract=False)
    results = []

    for item, response in tqdm(zip(prompt_items, final_responses), total=len(prompt_items), desc="Scoring"):
        is_mcq = bool(item.get("options"))
        gold   = item.get("answer")
        correct = None

        if gold is not None:
            if is_mcq:
                correct = score_mcq(response, str(gold))
            else:
                gold_list = gold if isinstance(gold, list) else [gold]
                try:
                    correct = judger.auto_judge(
                        pred=response, gold=gold_list,
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

    # Summary
    scored   = [r for r in results if r["correct"] is not None]
    mcq_res  = [r for r in scored if r["is_mcq"]]
    free_res = [r for r in scored if not r["is_mcq"]]

    def acc(subset):
        return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
    print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
    print(f"  Overall    : {sum(r['correct'] for r in scored):4d} / {len(scored):4d}  ({acc(scored):.2f}%)")
    print("=" * 50)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:  # append mode for resume support
        for r in results:
            if args.save_eval and r["correct"] is not None:
                record = {k: r[k] for k in ("id", "is_mcq", "gold", "response", "correct")}
            else:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "response": r["response"]}
            f.write(json.dumps(record) + "\n")

    step(f"Saved {len(results)} records to {out_path}.")
    step(f"Total wall time: {(time.time() - t_start) / 60:.1f} min.")
    banner("DONE")


if __name__ == "__main__":
    main()
