"""
CoT QLoRA fine-tuning of Qwen3-4B-Thinking on data/rft_sft.jsonl.

This is the SFT stage of the RFT (rejection-sampling fine-tuning) pipeline. It
differs from train_lora.py in one essential way: train_lora.py supervises ONLY
the "\\boxed{answer}" token (which suppresses reasoning), whereas this script
supervises the WHOLE assistant turn — the chain-of-thought plus the boxed
answer — so the resulting adapter both reasons and boxes.

Pipeline position:
    generate_rft_traces.py  ->  data/rft_sft.jsonl
    train_lora_cot.py       ->  checkpoints/qwen3-4b-lora-cot   (this script)
    train_grpo.py           ->  GRPO LoRA adapter   (use --init-adapter <above>)
    test_inference.py       ->  evaluate

ONE-TIME extra deps (on top of the run_inference.py setup):
    .venv/bin/python -m pip install peft datasets accelerate

Run (after generate_rft_traces.py has produced data/rft_sft.jsonl):
    .venv/bin/python train_lora_cot.py

The target is the trace verbatim: the prompt is masked, the completion (which
already contains the model's <think>...</think> reasoning and \\boxed{answer})
is supervised, with an EOS appended so the model learns to stop.
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
DATA_PATH       = "data/rft_sft.jsonl"
OUTPUT_DIR      = "checkpoints/qwen3-4b-lora-cot"
MAX_SEQ_LEN     = 4096     # CoT traces are long — much larger than train_lora.py's 1024
NUM_EPOCHS      = 2
PER_DEVICE_BS   = 1        # long sequences -> small micro-batch
GRAD_ACCUM      = 16       # effective batch = 16
LR              = 2e-4
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
SEED            = 42


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    print("=" * 70, flush=True)


def step(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Same prompts as generate_rft_traces.py — the prompt must be byte-identical to
# the one the traces were sampled under, or the masking boundary drifts.
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


def main() -> None:
    # ── 1. CUDA + imports ───────────────────────────────────────────────────
    banner("STEP 1 / 5  CUDA sanity check")
    import torch
    step(f"torch {torch.__version__} cuda {torch.version.cuda}")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    banner("STEP 2 / 5  Imports (transformers, peft, datasets)")
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
        TrainingArguments, Trainer, DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    step("imports done.")

    # ── 2. Tokenizer + model (4-bit NF4) ────────────────────────────────────
    banner("STEP 3 / 5  Load tokenizer + 4-bit base model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    step("loading base model in 4-bit NF4 (this takes a minute) ...")
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
        attn_implementation="flash_attention_2",
    )
    step(f"base model loaded in {time.time() - t0:.1f}s.")
    model = prepare_model_for_kbit_training(model)

    step("attaching LoRA adapter ...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 3. Build dataset (tokenize + mask prompt tokens) ────────────────────
    banner("STEP 4 / 5  Build training dataset from rft_sft.jsonl")
    if not Path(DATA_PATH).exists():
        sys.exit(f"{DATA_PATH} not found — run generate_rft_traces.py first.")
    raw = [json.loads(line) for line in open(DATA_PATH)]
    step(f"loaded {len(raw)} traces.")

    def tokenize_example(item: dict) -> Optional[dict]:
        """Render system+user as a prompt, append the CoT trace verbatim, and
        return input_ids with labels masked over the prompt portion only.
        Over-long traces are dropped (returns None) rather than truncated —
        truncating would cut off reasoning and leave a danging \\boxed{}."""
        system, user = build_prompt(item["question"], item.get("options"))
        completion   = item["completion"].strip()

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,   # ends with the assistant marker
        )
        # The trace already contains the reasoning + \boxed{answer}; append EOS
        # so the model learns to stop.
        full_text = prompt_text + completion + tokenizer.eos_token

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids   = tokenizer(full_text,   add_special_tokens=False)["input_ids"]
        if len(full_ids) > MAX_SEQ_LEN:
            return None

        # Mask the prompt; supervise the whole completion (CoT + boxed answer).
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        assert len(labels) == len(full_ids), (len(labels), len(full_ids))
        return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}

    step("tokenizing traces ...")
    tokenized = [ex for ex in (tokenize_example(it) for it in raw) if ex is not None]
    n_dropped = len(raw) - len(tokenized)
    step(f"built {len(tokenized)} examples ({n_dropped} dropped for exceeding {MAX_SEQ_LEN} tokens).")
    if not tokenized:
        sys.exit("No examples survived tokenization — raise MAX_SEQ_LEN.")

    lens = sorted(len(ex["input_ids"]) for ex in tokenized)
    step(f"sequence length: min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}")
    step("first example (decoded prompt | target):")
    ex0 = tokenized[0]
    cut = next((i for i, l in enumerate(ex0["labels"]) if l != -100), len(ex0["input_ids"]))
    print("    PROMPT:", tokenizer.decode(ex0["input_ids"][:cut])[-200:].replace("\n", "\\n"))
    print("    TARGET:", tokenizer.decode(ex0["input_ids"][cut:])[:400].replace("\n", "\\n"))

    dataset = Dataset.from_list(tokenized).shuffle(seed=SEED)

    # ── 4. Train ────────────────────────────────────────────────────────────
    banner("STEP 5 / 5  Train")
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        tf32=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    step("starting training ...")
    t0 = time.time()
    trainer.train()
    step(f"training done in {(time.time() - t0) / 60:.1f} min.")

    step(f"saving adapter to {OUTPUT_DIR} ...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    banner(f"DONE  —  next: train_grpo.py --init-adapter {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
