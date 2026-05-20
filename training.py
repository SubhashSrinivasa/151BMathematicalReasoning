"""
training.py — CSE 151B Competition
=====================================
Fine-tunes the base model with LoRA (PEFT), using training data constructed
with the AutoCoT methodology:
  • SBERT embeddings + KMeans clustering to group similar questions
  • Per-cluster representative solutions as few-shot exemplars
  • Each training example = (few-shot AutoCoT prompt, gold answer)

The LoRA adapter produced here can be passed directly to inference.py via
  --lora-path results/lora_adapter

ONE-TIME setup:
    pip install "numpy<2" torch transformers accelerate peft \
                sentence-transformers scikit-learn tqdm bitsandbytes \
                datasets antlr4-python3-runtime==4.11.1

Run:
    python training.py \
        --data-path data/public.jsonl \
        --output-dir results/lora_adapter \
        [--model-id Qwen/Qwen3-4B-Thinking-2507] \
        [--epochs 3] [--batch-size 2] [--lr 2e-4]
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning with AutoCoT data construction")
    # Paths
    p.add_argument("--model-id",       default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--data-path",      default="data/public.jsonl",
                   help="Labelled JSONL with 'answer' field (public split).")
    p.add_argument("--output-dir",     default="results/lora_adapter",
                   help="Where to save the final LoRA adapter.")
    # AutoCoT clustering
    p.add_argument("--k",              type=int, default=8)
    p.add_argument("--reps-per-cluster", type=int, default=2)
    p.add_argument("--max-input-tokens", type=int, default=10000,
                   help="Hard cap on tokenised prompt length during training.")
    # LoRA hyper-params
    p.add_argument("--lora-r",         type=int,   default=16)
    p.add_argument("--lora-alpha",     type=int,   default=32)
    p.add_argument("--lora-dropout",   type=float, default=0.05)
    p.add_argument("--target-modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
                   help="Which linear layers to attach LoRA adapters to.")
    # Training hyper-params
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--batch-size",     type=int,   default=1,
                   help="Per-device training batch size.")
    p.add_argument("--grad-accum",     type=int,   default=8,
                   help="Gradient accumulation steps (effective batch = batch-size × grad-accum).")
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--warmup-ratio",   type=float, default=0.03)
    p.add_argument("--max-seq-len",    type=int,   default=4096,
                   help="Maximum sequence length (prompt + completion) for training.")
    p.add_argument("--val-split",      type=float, default=0.1,
                   help="Fraction of data held out for validation.")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--fp16",           action="store_true",
                   help="Use FP16 mixed precision (default: BF16 when available).")
    p.add_argument("--load-in-8bit",   action="store_true",
                   help="Load base model in INT8 (saves VRAM, slower training).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Prompts  (same vocabulary as inference.py)
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


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def build_rep_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
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


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def item_to_text(item: dict) -> str:
    q    = item["question"]
    opts = item.get("options")
    if opts:
        labels    = [chr(65 + i) for i in range(len(opts))]
        opts_text = " ".join(f"{l}. {o}" for l, o in zip(labels, opts))
        return f"Question: {q} Options: {opts_text}"
    return f"Question: {q}"


def gold_to_completion(item: dict) -> str:
    """
    Format the gold answer as the expected model completion.
    MCQ  → \\boxed{A}  (just the letter)
    Math → \\boxed{42} (the numeric/symbolic answer)
    """
    gold = item.get("answer", "")
    opts = item.get("options")
    if opts and isinstance(gold, str) and len(gold.strip()) == 1:
        return f"\\boxed{{{gold.strip().upper()}}}"
    if isinstance(gold, list):
        return f"\\boxed{{{', '.join(str(g) for g in gold)}}}"
    return f"\\boxed{{{gold}}}"


def shorten_solution(text: str, max_chars: int = 1200) -> str:
    if "\\boxed{" in text:
        pos = text.rfind("\\boxed{")
        return text[: pos + 200]
    return text[:max_chars]


def trim_solution_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


# ─────────────────────────────────────────────────────────────────────────────
# Dataset construction
# ─────────────────────────────────────────────────────────────────────────────

def build_training_examples(
    data:               list[dict],
    cluster_ids:        np.ndarray,
    cluster_rep_indices: dict[int, list[int]],
    rep_solutions:      dict[str, str],
    tokenizer,
    args:               argparse.Namespace,
) -> list[dict]:
    """
    For each item, build an AutoCoT few-shot prompt and pair it with the gold
    completion.  Returns a list of {"prompt": str, "completion": str} dicts.
    """
    examples = []

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

        # Fallback: zero-shot prompt
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
            print(f"  WARNING: skipping id={item['id']} — prompt too long.")
            continue

        completion = gold_to_completion(item)
        examples.append({"prompt": final_prompt, "completion": completion})

    return examples


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PromptCompletionDataset(torch.utils.data.Dataset):
    """
    Tokenises (prompt + completion) sequences and masks the prompt tokens in
    the labels so that loss is computed only on the completion.
    """

    def __init__(self, examples: list[dict], tokenizer, max_seq_len: int):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.records     = []

        for ex in examples:
            full_text   = ex["prompt"] + ex["completion"] + tokenizer.eos_token
            prompt_text = ex["prompt"]

            full_ids   = tokenizer.encode(full_text,   add_special_tokens=False)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

            if len(full_ids) > max_seq_len:
                full_ids = full_ids[:max_seq_len]

            prompt_len = min(len(prompt_ids), len(full_ids))
            labels     = [-100] * prompt_len + full_ids[prompt_len:]

            # Pad to max_seq_len
            pad_len   = max_seq_len - len(full_ids)
            input_ids = full_ids + [tokenizer.pad_token_id] * pad_len
            labels    = labels   + [-100]                  * pad_len
            attn_mask = [1] * len(full_ids) + [0] * pad_len

            self.records.append({
                "input_ids":      torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
                "labels":         torch.tensor(labels,    dtype=torch.long),
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    t_start = time.time()
    torch.manual_seed(args.seed)

    def banner(msg):
        print(f"\n{'='*70}\n[{time.strftime('%H:%M:%S')}] {msg}\n{'='*70}", flush=True)

    def step(msg):
        print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    # ── CUDA ─────────────────────────────────────────────────────────────────
    banner("STEP 1 / 7  CUDA check")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available — aborting.")
    step(f"torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    banner("STEP 2 / 7  Load dataset")
    data = [json.loads(line) for line in open(args.data_path)]
    # Keep only items that have gold answers (training split)
    data = [d for d in data if d.get("answer") is not None]
    step(f"Loaded {len(data)} labelled questions")

    # ── AutoCoT clustering ────────────────────────────────────────────────────
    banner("STEP 3 / 7  AutoCoT — embed + cluster")
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
    banner("STEP 4 / 7  Load tokenizer + base model")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, use_fast=False, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Build representative solutions from gold answers ──────────────────────
    # For training we use the gold answer directly as the representative solution
    # (avoids a separate generation pass and keeps training deterministic).
    rep_solutions: dict[str, str] = {}
    for cid, rep_idxs in cluster_rep_indices.items():
        for idx in rep_idxs:
            item = data[idx]
            rep_solutions[item["id"]] = gold_to_completion(item)

    step(f"Built {len(rep_solutions)} representative solutions from gold answers.")

    # ── Build AutoCoT training examples ──────────────────────────────────────
    banner("STEP 5 / 7  Build training examples")
    examples = build_training_examples(
        data, cluster_ids, cluster_rep_indices, rep_solutions, tokenizer, args,
    )
    step(f"Built {len(examples)} training examples.")

    # Train / val split
    rng      = np.random.default_rng(args.seed)
    indices  = rng.permutation(len(examples)).tolist()
    n_val    = max(1, int(len(examples) * args.val_split))
    val_idx  = set(indices[:n_val])
    train_ex = [ex for i, ex in enumerate(examples) if i not in val_idx]
    val_ex   = [ex for i, ex in enumerate(examples) if i in val_idx]
    step(f"Train: {len(train_ex)}, Val: {len(val_ex)}")

    train_ds = PromptCompletionDataset(train_ex, tokenizer, args.max_seq_len)
    val_ds   = PromptCompletionDataset(val_ex,   tokenizer, args.max_seq_len)

    # ── Load base model ───────────────────────────────────────────────────────
    bnb_config = None
    if args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        step("Loading model in INT8 (bitsandbytes).")
    else:
        step("Loading model in BF16 / FP32.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config = bnb_config,
        device_map          = "auto",
        trust_remote_code   = True,
        torch_dtype         = torch.float16 if args.fp16 else torch.bfloat16,
    )

    if args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    # ── Attach LoRA adapters ──────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type    = TaskType.CAUSAL_LM,
        r            = args.lora_r,
        lora_alpha   = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        target_modules = args.target_modules,
        bias         = "none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Train ─────────────────────────────────────────────────────────────────
    banner("STEP 6 / 7  Fine-tune with LoRA")
    from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

    use_bf16 = (not args.fp16) and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir                  = args.output_dir,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        gradient_checkpointing      = True,
    	gradient_checkpointing_kwargs = {"use_reentrant": False},
	per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        warmup_ratio                = args.warmup_ratio,
        lr_scheduler_type           = "cosine",
        fp16                        = args.fp16,
        bf16                        = use_bf16,
        logging_steps               = 10,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        report_to                   = "none",
        seed                        = args.seed,
        dataloader_num_workers      = 0,
        remove_unused_columns       = False,
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
    )

    step("Starting training …")
    from transformers.trainer_utils import get_last_checkpoint
    last_ckpt = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if last_ckpt:
        step(f"Resuming from checkpoint: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
    step("Training complete.")

    # ── Save LoRA adapter ─────────────────────────────────────────────────────
    banner("STEP 7 / 7  Save LoRA adapter")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    step(f"LoRA adapter saved to: {out_dir}")
    step(f"Total wall time: {(time.time() - t_start) / 60:.1f} min.")
    banner("DONE — run inference.py --lora-path " + str(out_dir))


if __name__ == "__main__":
    main()
