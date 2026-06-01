"""
training.py — LoRA fine-tuning for CSE 151B Math Reasoning Competition
Mirrors the exact steps, configs, and token sizes from the notebook.
"""

import gc
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset

from transformers import TrainerCallback
import json
import re


class SaveAnswersCallback(TrainerCallback):
    """
    Runs inference on the full training set at the end of every epoch and
    saves the results to a JSON file.  Each record contains:
        - question     : the raw question text
        - gold         : the ground-truth answer string
        - full_response: the raw model output (includes <think> block)
        - final_box    : the content of the last \\boxed{} in the output
        - correct      : True/False exact-match against gold
        - epoch        : which epoch produced this prediction
    """

    MAX_NEW_TOKENS = 4096   # cap generation length per sample
    BATCH_SIZE     = 4      # inference batch size (reduce if OOM)

    def __init__(self, dataset, raw_data, tokenizer, save_path="data/training_answers.json"):
        self.tokenizer    = tokenizer
        self.save_path    = save_path
        self.raw_data     = raw_data   # original list of dicts from the JSONL

        # Build inference prompts (prompt-only, no assistant turn) and gold answers
        self.prompts      = []
        self.gold_answers = []
        for item in raw_data:
            system, user = build_prompt(item["question"], item.get("options"))
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,   # appends <|im_start|>assistant\n
            )
            self.prompts.append(prompt_text)

            answer = item["answer"]
            if isinstance(answer, list):
                answer = ", ".join(str(a) for a in answer)
            self.gold_answers.append(answer)

    # ------------------------------------------------------------------
    # Helper: extract last \boxed{...} from generated text
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_boxed(text: str) -> Optional[str]:
        matches = re.findall(r"\\boxed\{([^}]*)\}", text)
        return matches[-1].strip() if matches else None

    # ------------------------------------------------------------------
    # Helper: naive exact-match (normalised)
    # ------------------------------------------------------------------
    @staticmethod
    def _is_correct(pred: Optional[str], gold: str) -> bool:
        if pred is None:
            return False
        return pred.strip().lower() == gold.strip().lower()

    # ------------------------------------------------------------------
    # Run inference over the whole dataset in mini-batches
    # ------------------------------------------------------------------
    def _generate_answers(self, model) -> list:
        model.eval()
        results = []

        with torch.no_grad():
            for start in range(0, len(self.prompts), self.BATCH_SIZE):
                batch_prompts = self.prompts[start : start + self.BATCH_SIZE]

                encodings = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=20000,
                ).to(model.device)

                outputs = model.generate(
                    **encodings,
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    do_sample=False,          # greedy — deterministic & fast
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Decode only the newly generated tokens (skip the prompt)
                prompt_lengths = encodings["input_ids"].shape[1]
                for i, output in enumerate(outputs):
                    global_idx  = start + i
                    new_tokens  = output[prompt_lengths:]
                    full_resp   = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    final_box   = self._extract_boxed(full_resp)
                    gold        = self.gold_answers[global_idx]

                    results.append({
                        "question"     : self.raw_data[global_idx]["question"],
                        "gold"         : gold,
                        "full_response": full_resp,
                        "final_box"    : final_box,
                        "correct"      : self._is_correct(final_box, gold),
                    })

        model.train()
        return results

    # ------------------------------------------------------------------
    # Called at the end of every epoch
    # ------------------------------------------------------------------
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(state.epoch)
        print(f"\n[Epoch {epoch}] Running inference on {len(self.prompts)} samples ...")

        results = self._generate_answers(model)

        # Tag each record with the epoch number
        for r in results:
            r["epoch"] = epoch

        # Load existing records (from previous epochs) and append
        save_path = Path(self.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if save_path.exists():
            with open(save_path) as f:
                existing = json.load(f)

        existing.extend(results)
        with open(save_path, "w") as f:
            json.dump(existing, f, indent=2)

        # Quick accuracy summary
        correct = sum(r["correct"] for r in results)
        total   = len(results)
        no_box  = sum(r["final_box"] is None for r in results)
        print(
            f"[Epoch {epoch}] Saved {total} predictions -> {save_path}\n"
            f"           Accuracy : {correct}/{total} ({correct/total*100:.1f}%)\n"
            f"           No boxed : {no_box}/{total} ({no_box/total*100:.1f}%)"
        )


# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_ID     = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH    = "data/public_with_reasoning.jsonl"
ADAPTER_PATH = "data/lora_math_adapter/final_adapter"
MERGED_PATH  = "/tmp/lora_math_merged"
OUTPUT_DIR   = "data/lora_math_adapter"
MAX_TOKENS   = 32000   # max output tokens at inference
MAX_SEQ_LEN  = 25000   # max input token length during training

# ── System Prompts ─────────────────────────────────────────────────────────────
EQUATION_SHEET = """
REFERENCE EQUATION SHEET:
Use only when relevant. Do not force a formula if the problem is conceptual.

FINAL ANSWER FORMAT:
- If there is one answer: \\boxed{answer}
- If there are multiple answers: \\boxed{a, b, c, ...}
- Do NOT output separate final boxes like \\boxed{a}\\boxed{b}.
- For multi-part questions, preserve the order asked in the problem.

ARITHMETIC / ALGEBRA:
- Order of operations: parentheses, exponents, multiplication/division left-to-right, addition/subtraction left-to-right.
- Difference of squares: a^2 - b^2 = (a-b)(a+b)
- Perfect square: (a+b)^2 = a^2 + 2ab + b^2, (a-b)^2 = a^2 - 2ab + b^2
- Quadratic formula: for ax^2+bx+c=0, x = (-b +/- sqrt(b^2-4ac))/(2a)
- Discriminant: D = b^2 - 4ac
  D > 0: two real roots; D = 0: one repeated real root; D < 0: no real roots.
- Arithmetic sequence: a_n = a_1 + (n-1)d
- Arithmetic sum: S_n = n(a_1+a_n)/2
- Geometric sequence: a_n = a_1 r^(n-1)
- Geometric sum: S_n = a_1(1-r^n)/(1-r), if r != 1

FUNCTIONS / COORDINATES:
- Slope: m = (y_2-y_1)/(x_2-x_1)
- Line form: y - y_1 = m(x-x_1), or y = mx+b
- Distance: d = sqrt((x_2-x_1)^2 + (y_2-y_1)^2)
- Midpoint: ((x_1+x_2)/2, (y_1+y_2)/2)

PROBABILITY:
- Complement: P(A^c) = 1 - P(A)
- Union: P(A union B) = P(A) + P(B) - P(A intersection B)
- Conditional probability: P(A|B) = P(A intersection B)/P(B)
- Independent events: P(A intersection B) = P(A)P(B)
- Mutually exclusive events: P(A intersection B) = 0
- Bayes: P(A|B) = P(B|A)P(A)/P(B)
- Expected value: E[X] = sum x P(X=x)
- Variance: Var(X) = E[X^2] - (E[X])^2

COUNTING / COMBINATORICS:
- Permutations/order matters: nP r = n!/(n-r)!
- Combinations/order does not matter: nC r = n!/(r!(n-r)!)
- With replacement: n^r
- Without replacement and order matters: nP r
- Without replacement and order does not matter: nC r
- Binomial probability: P(X=k) = C(n,k) p^k (1-p)^(n-k)
- At least one: P(at least one) = 1 - P(none)

STATISTICS:
- Mean: xbar = (sum x_i)/n
- Weighted mean: xbar = (sum w_i x_i)/(sum w_i)
- Median: middle value after sorting; if even n, average the two middle values.
- Sample variance: s^2 = sum(x_i - xbar)^2/(n-1)
- Sample standard deviation: s = sqrt(s^2)
- z-score: z = (x - mean)/standard deviation
- Confidence interval for mean: xbar +/- critical_value * standard_error
- If population sigma known: SE = sigma/sqrt(n), use z.
- If population sigma unknown: SE = s/sqrt(n), use t.
- Confidence interval for proportion: phat +/- z * sqrt(phat(1-phat)/n)
- Margin of error: E = critical_value * standard_error
- Sample size for proportion: n = p(1-p)(z/E)^2; if p unknown, use p=0.5.
- Sample size for mean: n = (z*sigma/E)^2.
- Always round sample size UP.

GEOMETRY:
- Rectangle area: A = lw; perimeter = 2l+2w
- Triangle area: A = (1/2)bh
- Pythagorean theorem: a^2 + b^2 = c^2
- Circle area: A = pi r^2
- Circle circumference: C = 2pi r = pi d
- Sector arc length: s = r theta, theta in radians
- Sector area: A = (1/2)r^2 theta, theta in radians
- Cylinder volume: V = pi r^2 h
- Cone volume: V = (1/3)pi r^2 h
- Sphere volume: V = (4/3)pi r^3
- Sphere surface area: A = 4pi r^2
- Similar figures: side lengths scale by k, areas by k^2, volumes by k^3

TRIGONOMETRY / POLAR:
- sin^2(x) + cos^2(x) = 1
- tan(x) = sin(x)/cos(x)
- Common angles:
  sin(0)=0, cos(0)=1
  sin(pi/6)=1/2, cos(pi/6)=sqrt(3)/2
  sin(pi/4)=sqrt(2)/2, cos(pi/4)=sqrt(2)/2
  sin(pi/3)=sqrt(3)/2, cos(pi/3)=1/2
  sin(pi/2)=1, cos(pi/2)=0
- Polar to rectangular: x = r cos(theta), y = r sin(theta)
- Rectangular to polar: r = sqrt(x^2+y^2), theta = atan(y/x) adjusted by quadrant
- Law of sines: a/sin(A) = b/sin(B) = c/sin(C)
- Law of cosines: c^2 = a^2 + b^2 - 2ab cos(C)

CALCULUS:
- Power rule: d/dx x^n = n x^(n-1)
- Integral power rule: integral x^n dx = x^(n+1)/(n+1) + C, n != -1
- Fundamental theorem: integral_a^b f(x) dx = F(b)-F(a)
- Product rule: (fg)' = f'g + fg'
- Chain rule: d/dx f(g(x)) = f'(g(x))g'(x)

CHECK BEFORE FINAL:
- Did the question ask for multiple values? If yes, use one comma-separated box.
- Did the question ask for an option letter? If yes, final must be \\boxed{A}, \\boxed{B}, etc.
- Did the reasoning choose one option but the final box shows another? Fix the final box.
- Did the problem ask for rounding, units, or an interval? Include exactly that.
"""

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician reasoning. Think concisely. Step by step.\n\n "
    "MANDATORY RULE:\n"
    "- Show ALL reasoning inside <think>...</think> tags."
    "- The FINAL token MUST be: \\boxed{final_answer}\n "
    "- If multiple answers are required, the FINAL token MUST be exactly: \\boxed{a, b, c, ...}\n"
    "- Do NOT use separate boxes like \\boxed{a}\\boxed{b}. Use one box only.\n"
    "- If this is not the final token, the response is invalid.\n\n"
    + EQUATION_SHEET +
    "FORMAT:\n"
    "\\boxed{final_answer}\n"
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician reasoning. Think concisely. Step by step.\n\n"
    "MANDATORY RULE:\n"
    "- Show ALL reasoning inside <think>...</think> tags.\n"
    "- The FINAL token MUST be: \\boxed{X} where X is the answer choice given the options A-Z.\n"
    "- If reasoning points to one option, the final box must match that option.\n"
    "- If this is not the final token, the response is invalid.\n\n"
    + EQUATION_SHEET +
    "\nFORMAT:\n"
    "<think>\nreasoning here\n</think>\n"
    "\\boxed{X}\n"
)

def build_prompt(question: str, options: Optional[list]) -> tuple:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


# ── Dataset ────────────────────────────────────────────────────────────────────
class MathDataset(Dataset):
    """
    Tokenizes question->answer pairs for supervised fine-tuning.

    Training target per sample:
        <think>
        {real reasoning from item["think"]}
        </think>

        \\boxed{answer}

    Only the assistant turn is trained on (prompt tokens are masked to -100).
    """

    MAX_THINK_CHARS = 8000  # reasoning trimmed to this many chars at sentence boundary

    def __init__(self, data, tokenizer, max_length=20000):
        self.samples = []
        skipped = 0

        for item in data:
            system, user = build_prompt(item["question"], item.get("options"))

            answer = item["answer"]
            if isinstance(answer, list):
                answer = ", ".join(str(a) for a in answer)

            # Use real CoT if available, else short fallback
            reasoning = item.get("think", "Let me solve this step by step.")
            # Strip embedded </think> tag and everything after it
            if "</think>" in reasoning:
                reasoning = reasoning.split("</think>")[0].strip()
            if len(reasoning) > self.MAX_THINK_CHARS:
                # Trim at sentence boundary to avoid cutting mid-word
                trimmed = reasoning[: self.MAX_THINK_CHARS].rsplit(".", 1)[0]
                reasoning = trimmed + "." if trimmed else reasoning[: self.MAX_THINK_CHARS]

            assistant_content = (
                f"<think>\n{reasoning}\n</think>\n\n"
                f"\\boxed{{{answer}}}"
            )

            messages = [
                {"role": "system",    "content": system},
                {"role": "user",      "content": user},
                {"role": "assistant", "content": assistant_content},
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors="pt",
            )

            input_ids      = tokenized["input_ids"].squeeze()
            attention_mask = tokenized["attention_mask"].squeeze()
            labels         = input_ids.clone()

            # Mask the prompt (system + user) — only train on assistant output
            assistant_header_tokens = tokenizer.encode(
                "<|im_start|>assistant\n", add_special_tokens=False
            )
            masked = False
            for i in range(len(labels) - len(assistant_header_tokens)):
                if labels[i : i + len(assistant_header_tokens)].tolist() == assistant_header_tokens:
                    labels[: i + len(assistant_header_tokens)] = -100
                    masked = True
                    break

            if not masked:
                skipped += 1
                continue

            self.samples.append({
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "labels":         labels,
            })

        print(f"MathDataset: {len(self.samples)} samples built, {skipped} skipped (no assistant header found)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Main training function ─────────────────────────────────────────────────────
def main():
    # 1. Load dataset
    print(f"Loading data from {DATA_PATH} ...")
    data = [json.loads(line) for line in open(DATA_PATH)]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options")   for d in data)
    print(f"Loaded {len(data)} questions ({n_mcq} MCQ, {n_free} free-form)")

    # 2. Load tokenizer (must come before dataset construction)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=False,
        padding_side="left",
        model_max_length=MAX_TOKENS,  # full 32k context window
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded:", tokenizer.__class__.__name__)

    # 3. Load model in 4-bit
    print("Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
    )
    train_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 4. Prepare for k-bit training
    train_model = prepare_model_for_kbit_training(train_model)

    # 5. Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.06,
        bias="none",
        task_type="CAUSAL_LM",
    )
    train_model = get_peft_model(train_model, lora_config)
    train_model.print_trainable_parameters()

    # 6. Build dataset
    train_dataset = MathDataset(data, tokenizer, max_length=MAX_SEQ_LEN)
    print(f"Training samples: {len(train_dataset)}")

    # 7. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=train_model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # Pass both the dataset and the raw data so the callback can build prompts
    # and record gold answers alongside model predictions.
    answers_callback = SaveAnswersCallback(
        dataset=train_dataset,
        raw_data=data,
        tokenizer=tokenizer,
        save_path="data/training_answers.json",
    )

    # 8. Train
    trainer = Trainer(
        model=train_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[answers_callback],
    )

    trainer.train()

    # 9. Save adapter + tokenizer
    Path(ADAPTER_PATH).mkdir(parents=True, exist_ok=True)
    train_model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    print(f"Adapter saved to {ADAPTER_PATH}")

    # 10. Free training model before loading full-precision base for merge
    train_model.cpu()
    del train_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"Free GPU after unloading train model: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    # 11. Merge LoRA adapter into base model
    print("Merging LoRA adapter into base model...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda",
        local_files_only=True,
    )
    model_merged = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model_merged = model_merged.merge_and_unload()
    model_merged.save_pretrained(MERGED_PATH)
    AutoTokenizer.from_pretrained(ADAPTER_PATH).save_pretrained(MERGED_PATH)
    print(f"Merged model saved to {MERGED_PATH}")

    # 12. Final cleanup
    del base
    del model_merged
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"Free GPU after merge: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")


if __name__ == "__main__":
    main()