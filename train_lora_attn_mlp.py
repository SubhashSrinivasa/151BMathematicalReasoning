"""
QLoRA on attention + MLP projections, restricted to the LAST 12 LAYERS.

Compared to train_lora.py:
  - LoRA rank 8 (was 16), alpha 16 (was 32)
  - LoRA only on the top 12 transformer layers (was all layers)
  - Same target modules: attn (q,k,v,o) + MLP (gate,up,down)

Smaller, faster adapter (~10-15 MB on disk). Cheaper memory footprint.
Likely outcome: comparable or slightly weaker than train_lora.py on free-form
reasoning, similar on MCQ. Good ablation if the full-layer version overfits.

Deps (same as train_lora.py):
    .venv/bin/python -m pip install peft datasets accelerate

Run:
    .venv/bin/python train_lora_attn_mlp.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID        = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH       = "data/public.jsonl"
OUTPUT_DIR      = "checkpoints/qwen3-4b-lora-attnmlp-last12"
MAX_SEQ_LEN     = 1024
NUM_EPOCHS      = 2
PER_DEVICE_BS   = 2
GRAD_ACCUM      = 8
LR              = 2e-4
LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05
LAST_N_LAYERS   = 12
SEED            = 42


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
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
        TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    step("imports done.")

    banner("STEP 3 / 5  Load tokenizer + 4-bit base + LoRA on last layers")
    step("loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    step("loading base model in 4-bit NF4 ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    step(f"base model loaded in {time.time() - t0:.1f}s.")
    model = prepare_model_for_kbit_training(model)

    num_layers = model.config.num_hidden_layers
    first_trained = max(0, num_layers - LAST_N_LAYERS)
    layers_to_transform = list(range(first_trained, num_layers))
    step(f"model has {num_layers} layers; LoRA on layers {first_trained}..{num_layers - 1} ({len(layers_to_transform)} layers).")

    step(f"attaching LoRA adapter (r={LORA_R}, alpha={LORA_ALPHA}, attn+MLP) ...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=layers_to_transform,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
        save_total_limit=2,
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

    step(f"saving adapter to {OUTPUT_DIR} ...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    banner("DONE")


if __name__ == "__main__":
    main()
