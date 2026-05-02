"""
FULL fine-tune of the LAST 3 transformer blocks (+ final norm + lm_head).
Everything else is frozen. No quantization, no LoRA.

Differences vs the LoRA scripts:
  - Base model loaded in bf16 (no 4-bit) so the unfrozen weights have full
    precision for SGD updates.
  - Embeddings + first (N - 3) transformer blocks have requires_grad=False.
  - Saved checkpoint is the FULL model (~8 GB), not a small adapter — there
    is no PEFT here, so no compact way to ship just the deltas.

Memory budget on a 24 GB A30:
    base bf16 weights         ~8.0 GB
    optimizer state (last 3)  ~1.3 GB  (paged_adamw_8bit on ~330M params)
    activations + KV          ~3-5 GB  (bs=1, grad_checkpointing on)
    -----------------------------------
    fits comfortably; if you OOM, drop PER_DEVICE_BS to 1 (already 1) and/or
    MAX_SEQ_LEN to 1024.

** Disk warning **: saved checkpoint is ~8 GB. With your 9 GB budget you can
keep AT MOST ONE epoch's checkpoint at a time. We set save_total_limit=1.

Deps:
    .venv/bin/python -m pip install datasets accelerate

Run:
    .venv/bin/python train_full_last3.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID         = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH        = "data/public.jsonl"
OUTPUT_DIR       = "checkpoints/qwen3-4b-fullft-last3"
MAX_SEQ_LEN      = 2048
NUM_EPOCHS       = 2
PER_DEVICE_BS    = 1            # full FT is heavier; start small
GRAD_ACCUM       = 16           # effective batch = 16
LR               = 5e-5         # full FT needs much lower LR than LoRA
N_UNFROZEN_LAYERS = 3
SEED             = 42


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


def format_answer(ans) -> str:
    if isinstance(ans, list):
        return ", ".join(str(a) for a in ans)
    return str(ans)


def freeze_all_but_last_n(model, n: int) -> None:
    """Freeze every parameter, then unfreeze the last n transformer blocks,
    the final layernorm, and lm_head. Prints a summary."""
    for p in model.parameters():
        p.requires_grad = False

    num_layers = model.config.num_hidden_layers
    keep = set(range(num_layers - n, num_layers))
    unfrozen_keys = []
    for name, p in model.named_parameters():
        # transformer blocks: "model.layers.<idx>.*"
        if name.startswith("model.layers."):
            idx = int(name.split(".")[2])
            if idx in keep:
                p.requires_grad = True
                unfrozen_keys.append(name)
        # final norm and head
        elif name.startswith("model.norm") or name.startswith("lm_head"):
            p.requires_grad = True
            unfrozen_keys.append(name)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  unfrozen layers: {sorted(keep)} + final norm + lm_head")
    print(f"  trainable params: {n_train:,} / {n_total:,}  ({100 * n_train / n_total:.2f}%)")


def main() -> None:
    banner("STEP 1 / 5  CUDA sanity check")
    import torch
    step(f"torch {torch.__version__} cuda {torch.version.cuda}")
    step(f"cuda available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available.")

    banner("STEP 2 / 5  Imports")
    step("importing ...")
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    )
    from datasets import Dataset
    step("imports done.")

    banner("STEP 3 / 5  Load tokenizer + bf16 base, then freeze all-but-last-3")
    step("loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    step("loading base model in bf16 (no quantization) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    step(f"base model loaded in {time.time() - t0:.1f}s.")

    # Required so gradient checkpointing works with frozen embeddings.
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    step(f"freezing all parameters except last {N_UNFROZEN_LAYERS} blocks + norm + lm_head ...")
    freeze_all_but_last_n(model, N_UNFROZEN_LAYERS)

    banner("STEP 4 / 5  Build training dataset")
    step(f"reading {DATA_PATH} ...")
    raw = [json.loads(line) for line in open(DATA_PATH)]
    step(f"loaded {len(raw)} items.")

    def tokenize_example(item: dict) -> dict:
        system, user = build_prompt(item["question"], item.get("options"))
        answer_str = format_answer(item["answer"])
        target = (
            f"\\boxed{{{answer_str.strip().upper()}}}"
            if item.get("options")
            else f"\\boxed{{{answer_str}}}"
        )
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt_text + target + tokenizer.eos_token
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids   = tokenizer(full_text,   add_special_tokens=False)["input_ids"]
        if len(full_ids) > MAX_SEQ_LEN:
            overflow = len(full_ids) - MAX_SEQ_LEN
            full_ids   = full_ids[overflow:]
            prompt_ids = prompt_ids[overflow:] if len(prompt_ids) > overflow else []
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}

    step("tokenizing examples ...")
    tokenized = [tokenize_example(it) for it in raw]
    n_supervised = sum(any(l != -100 for l in ex["labels"]) for ex in tokenized)
    step(f"built {len(tokenized)} examples; {n_supervised} have non-empty supervision.")
    ex0 = tokenized[0]
    cut = next((i for i, l in enumerate(ex0["labels"]) if l != -100), len(ex0["input_ids"]))
    print("    PROMPT:", tokenizer.decode(ex0["input_ids"][:cut])[-300:].replace("\n", "\\n"))
    print("    TARGET:", tokenizer.decode(ex0["input_ids"][cut:]).replace("\n", "\\n"))
    dataset = Dataset.from_list(tokenized).shuffle(seed=SEED)

    banner("STEP 5 / 5  Train")
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,         # ~8 GB per checkpoint; keep only one
        optim="paged_adamw_8bit",
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=dataset,
        tokenizer=tokenizer, data_collator=collator,
    )
    step("starting training ...")
    t0 = time.time()
    trainer.train()
    step(f"training done in {(time.time() - t0) / 60:.1f} min.")

    step(f"saving FULL model to {OUTPUT_DIR} (~8 GB) ...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    banner("DONE")


if __name__ == "__main__":
    main()
