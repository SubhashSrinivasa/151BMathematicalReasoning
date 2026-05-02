# train_lora_qwen_math.py

import os
import re
import gc
import sys
import json
import torch
import pandas as pd

from tqdm import tqdm
from datasets import Dataset
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# CONFIG

MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"

PUBLIC_PATH = "data/public.jsonl"
PRIVATE_PATH = "data/private.jsonl"

ADAPTER_DIR = "qwen_math_lora_adapter"
SUBMISSION_PATH = "submission.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)


# LOAD DATA


def load_jsonl(path):
    return [json.loads(line) for line in open(path)]


public_data = load_jsonl(PUBLIC_PATH)
private_data = load_jsonl(PRIVATE_PATH)

print("Public examples:", len(public_data))
print("Private examples:", len(private_data))



# PROMPTS

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. "
    "Your FIRST token must be \\boxed{}. "
    "Put the final answer inside \\boxed{}. "
    "If multiple sub-answers are required, put them in one box separated by commas, e.g. \\boxed{3,7}. "
    "After the boxed answer, give a concise explanation."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Your FIRST line must be the SINGLE best answer choice inside \\boxed{}, e.g. \\boxed{C}. "
    "After the boxed answer, give a concise explanation."
)


def build_prompt(question, options=None):
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(
            f"{lbl}. {opt.strip()}"
            for lbl, opt in zip(labels, options)
        )
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"

    return SYSTEM_PROMPT_MATH, question


def format_gold_answer(item):
    gold = item["answer"]

    if item.get("options"):
        return f"\\boxed{{{gold}}}"

    if isinstance(gold, list):
        return "\\boxed{" + ",".join(map(str, gold)) + "}"

    return f"\\boxed{{{gold}}}"


# TOKENIZER


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_fast=False,
    padding_side="left",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# SPLIT DATA

train_data, val_data = train_test_split(
    public_data,
    test_size=0.30,
    random_state=42,
    shuffle=True,
)

print("Train:", len(train_data))
print("Validation:", len(val_data))


# BUILD TRAIN DATASET

def build_training_text(item):
    system, user = build_prompt(item["question"], item.get("options"))
    gold_text = format_gold_answer(item)

    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": gold_text},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


train_dataset = Dataset.from_list([
    {"text": build_training_text(item)}
    for item in train_data
])

print(train_dataset)


# LOAD BASE QWEN FOR QLORA TRAINING
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

qlora_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

qlora_model.config.use_cache = False
qlora_model = prepare_model_for_kbit_training(qlora_model)

# LORA TRAINING

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

training_args = SFTConfig(
    output_dir="qwen_math_lora",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=False,
    bf16=True,
    max_length=1024,
    packing=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=qlora_model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
)

trainer.train()

trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

print("Saved QLoRA adapter to:", ADAPTER_DIR)


#CLEAR TRAINING MODEL

del trainer
del qlora_model

gc.collect()
torch.cuda.empty_cache()

print("Cleared training model from memory.")

# LOAD BASE QWEN + LORA ADAPTER FOR INFERENCE

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
)

model.eval()

print("Loaded base Qwen + LoRA adapter.")


# ============================================================
# GENERATION
# ============================================================

def build_inference_prompts(items):
    prompts = []

    for item in items:
        system, user = build_prompt(item["question"], item.get("options"))

        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        prompts.append(prompt_text)

    return prompts


def generate_batch_peft(
    items,
    batch_size=1,
    max_new_tokens=3000,
    max_input_length=8000,
):
    prompts = build_inference_prompts(items)
    all_responses = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(model.device)

        input_lengths = inputs["attention_mask"].sum(dim=1)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for i in range(len(batch_prompts)):
            generated_ids = outputs[i][input_lengths[i]:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_responses.append(text.strip())

        print(f"Generated {min(start + batch_size, len(prompts))} / {len(prompts)}")

    return all_responses


# ============================================================
# VALIDATION
# ============================================================
"""
val_outputs = generate_batch_peft(
    val_data,
    batch_size=1,
    max_new_tokens=3000,
)
"""

# ============================================================
# SCORING
# ============================================================
"""
def extract_letter(text):
    text = str(text)

    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()

    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response, gold_letter):
    return extract_letter(response) == str(gold_letter).strip().upper()


sys.path.insert(0, ".")

from judger import Judger

judger = Judger(strict_extract=False)

results = []

for item, response in tqdm(zip(val_data, val_outputs), total=len(val_data), desc="Scoring"):
    is_mcq = bool(item.get("options"))
    gold = item["answer"]

    if is_mcq:
        correct = score_mcq(response, gold)
    else:
        gold_list = gold if isinstance(gold, list) else [gold]

        try:
            correct = judger.auto_judge(
                pred=response,
                gold=gold_list,
                options=[[]] * len(gold_list),
            )
        except Exception:
            correct = False

    results.append({
        "id": item.get("id"),
        "is_mcq": is_mcq,
        "gold": gold,
        "response": response,
        "correct": correct,
    })


mcq_res = [r for r in results if r["is_mcq"]]
free_res = [r for r in results if not r["is_mcq"]]


def acc(subset):
    return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0


print("=" * 50)
print("VALIDATION RESULTS")
print("=" * 50)
print(f"MCQ       : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
print(f"Free-form : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
print(f"Overall   : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")
print("=" * 50)

pd.DataFrame(results).to_csv("validation_results.csv", index=False)
"""
# ============================================================
# FINAL PRIVATE TEST INFERENCE
# ============================================================

# ============================================================
# FINAL PRIVATE TEST INFERENCE WITH RESUME
# ============================================================

SUBMISSION_PATH = "submission.csv"

if os.path.exists(SUBMISSION_PATH):
    existing_df = pd.read_csv(SUBMISSION_PATH)
    completed_ids = set(existing_df["id"].astype(str))
    print(f"Found {len(completed_ids)} completed answers.")
else:
    completed_ids = set()
    pd.DataFrame(columns=["id", "answer"]).to_csv(SUBMISSION_PATH, index=False)

remaining_private = [
    item for item in private_data
    if str(item["id"]) not in completed_ids
]

print(f"Remaining private examples: {len(remaining_private)}")

for idx, item in enumerate(remaining_private, start=1):
    output = generate_batch_peft(
        [item],
        batch_size=1,
        max_new_tokens=3000,   # reduce from 3000
        max_input_length=3000  # reduce from 8000
    )[0]

    row = pd.DataFrame([{
        "id": item["id"],
        "answer": output.strip(),
    }])

    row.to_csv(
        SUBMISSION_PATH,
        mode="a",
        header=False,
        index=False
    )

    print(f"Saved {idx} / {len(remaining_private)} | id={item['id']}")

print("Done.")
print("Saved:", SUBMISSION_PATH)
print("Rows:", len(pd.read_csv(SUBMISSION_PATH)))

