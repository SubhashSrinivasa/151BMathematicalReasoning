"""
GRPO reinforcement learning of Qwen3-4B-Thinking on public.jsonl.

This is the RL stage that sits *between* SFT (train_lora.py / the CoT RFT-SFT)
and inference (test_inference.py). It takes a LoRA policy, samples groups of
chain-of-thought rollouts per problem, scores each rollout with the competition
judger, and does a GRPO policy-gradient update — no value model, no reward
model, just the verifiable judge.

ONE-TIME extra deps (on top of the run_inference.py / train_lora.py setup):
    .venv/bin/python -m pip install "trl>=0.19.0" peft datasets accelerate

To run (continuing from a CoT-capable SFT adapter — recommended):
    .venv/bin/python train_grpo.py --init-adapter checkpoints/qwen3-4b-lora-cot

To run from the base model directly (Qwen3-Thinking already reasons natively):
    .venv/bin/python train_grpo.py

What it does:
  - Loads Qwen3-4B-Thinking-2507 in 4-bit NF4 (QLoRA), attaches/loads a LoRA.
  - For each problem, vLLM (colocate) samples NUM_GENERATIONS rollouts.
  - Each rollout is scored: 0.8 * binary-correct + 0.2 * partial-credit, plus a
    small format reward for a well-formed \\boxed{}. The judger is the same one
    used by run_inference.py / test_inference.py.
  - GRPO normalizes reward within each group and updates the LoRA.
  - Saves the trained adapter to checkpoints/qwen3-4b-grpo.

After training, evaluate exactly like an SFT adapter:
    .venv/bin/python test_inference.py --adapter checkpoints/qwen3-4b-grpo

MEMORY (24 GB MIG slice): vLLM colocate keeps a bf16 base copy *and* the 4-bit
training copy resident. If you OOM, in rough order: lower --vllm-mem, drop
PER_DEVICE_BS to 1, or cut --max-completion.
"""

import os
import sys
import json
import re
import time
import signal
import argparse
from pathlib import Path
from typing import Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID         = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH        = "data/public.jsonl"
OUTPUT_DIR       = "checkpoints/qwen3-4b-grpo"

NUM_GENERATIONS  = 8       # rollouts per problem (the GRPO "group")
MAX_PROMPT_LEN   = 1536    # longest question ~1.4k tokens; left-truncated above this
MAX_COMPLETION   = 2048    # "medium" reasoning budget
PER_DEVICE_BS    = 2       # completions per forward in the loss pass (raise if VRAM allows)
GRAD_ACCUM       = 8       # global batch = PER_DEVICE_BS*GRAD_ACCUM = 16 -> 2 prompts/step
NUM_EPOCHS       = 1
LR               = 1e-5
BETA             = 0.04    # KL coefficient vs. the (adapter-disabled) reference
TEMPERATURE      = 1.0     # rollout sampling temperature (diversity for the group)
VLLM_MEM         = 0.40    # fraction of the 24 GB slice handed to colocate vLLM

LORA_R           = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.0     # dropout off for RL: keeps gen/train log-probs consistent
SEED             = 42

# Reward weights: [correctness, format]. Correctness already blends binary +
# partial credit internally; the format term only breaks ties in all-wrong groups.
REWARD_WEIGHTS   = [1.0, 0.2]
JUDGE_TIMEOUT_S  = 5       # per-rollout sympy judging wall-clock cap


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    print("=" * 70, flush=True)


def step(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Same prompts as train_lora.py / run_inference.py so the policy sees a
# consistent task description across SFT, RL and inference.
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


def extract_letter(text: str) -> str:
    """MCQ answer extraction — mirrors run_inference.py."""
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def has_boxed(text: str) -> bool:
    """True iff the text contains a non-empty \\boxed{...}."""
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    return bool(m and m.group(1).strip())


# ── Reward functions ─────────────────────────────────────────────────────────
# Built by make_reward_fns(judger) so they close over the (heavy) Judger
# instance. TRL calls reward functions in the trainer's main process/thread,
# so signal.alarm()-based timeouts are safe here.

def _alarm_guard(seconds: int):
    """Context-manager-ish helper: arm SIGALRM, used around sympy calls."""
    def handler(signum, frame):
        raise TimeoutError("judge timeout")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)


def make_reward_fns(judger):
    """Return (correctness_reward, format_reward) closures for GRPOTrainer."""

    def _judge_one(text: str, gold_list: list, is_mcq: bool) -> tuple[float, float]:
        """Return (binary, partial) in [0,1]. binary is the strict all-or-nothing
        verdict (the real competition metric); partial is the fraction of
        sub-answers correct, used to densify reward on multi-answer problems."""
        if is_mcq:
            ok = float(extract_letter(text) == str(gold_list[0]).strip().upper())
            return ok, ok

        # Free-form: strict binary via the same auto_judge used at eval time.
        binary = 0.0
        try:
            _alarm_guard(JUDGE_TIMEOUT_S)
            binary = float(judger.auto_judge(
                pred=text, gold=gold_list, options=[[]] * len(gold_list),
            ))
        except Exception:
            binary = 0.0
        finally:
            signal.alarm(0)

        if binary == 1.0:
            return 1.0, 1.0

        # Partial credit: element-wise comparison, mirroring auto_judge's steps.
        partial = 0.0
        try:
            _alarm_guard(JUDGE_TIMEOUT_S)
            extracted = judger.extract_ans(text)
            if extracted:
                preds = [judger.norm_ans_str(p) for p in judger.split_by_comma(extracted)]
                gold_n = [judger.norm_ans_str(g) for g in gold_list]
                if len(preds) == len(gold_n) and gold_n:
                    judger.precision = 1e-8
                    hits = 0
                    for p, g in zip(preds, gold_n):
                        try:
                            hits += int(judger.is_equal(p, g, options=[]))
                        except Exception:
                            pass
                    partial = hits / len(gold_n)
        except Exception:
            partial = 0.0
        finally:
            signal.alarm(0)
        return binary, partial

    def _text(completion) -> str:
        """Extract assistant text from a TRL completion (conversational or str)."""
        if isinstance(completion, str):
            return completion
        # conversational: list of message dicts
        return completion[-1]["content"] if completion else ""

    def correctness_reward(completions, gold, is_mcq, **kwargs):
        """0.8 * strict-correct + 0.2 * partial-credit, per rollout."""
        rewards = []
        for comp, g_json, mcq in zip(completions, gold, is_mcq):
            text = _text(comp)
            gold_list = json.loads(g_json)
            if not isinstance(gold_list, list):
                gold_list = [gold_list]
            binary, partial = _judge_one(text, gold_list, bool(mcq))
            rewards.append(0.8 * binary + 0.2 * partial)
        return rewards

    def format_reward(completions, **kwargs):
        """1.0 if the rollout emits a well-formed non-empty \\boxed{}, else 0.0.
        Constant within a group when every rollout is well-formed (the common
        case) -> zero advantage -> no effect; only matters early / on failures."""
        return [1.0 if has_boxed(_text(c)) else 0.0 for c in completions]

    return correctness_reward, format_reward


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO RL fine-tuning on the math dataset.")
    p.add_argument("--model",          default=MODEL_ID,       help="Base model id or path.")
    p.add_argument("--init-adapter",   default=None,           help="LoRA adapter to continue RL from (e.g. the CoT SFT adapter). If omitted, a fresh LoRA is attached to the base model.")
    p.add_argument("--data",           default=DATA_PATH,      help="Training jsonl.")
    p.add_argument("--output",         default=OUTPUT_DIR,     help="Where to save the trained adapter.")
    p.add_argument("--epochs",         type=int,   default=NUM_EPOCHS)
    p.add_argument("--lr",             type=float, default=LR)
    p.add_argument("--beta",           type=float, default=BETA, help="KL coefficient (0 disables the reference model).")
    p.add_argument("--num-generations", type=int,  default=NUM_GENERATIONS)
    p.add_argument("--max-prompt",     type=int,   default=MAX_PROMPT_LEN)
    p.add_argument("--max-completion", type=int,   default=MAX_COMPLETION)
    p.add_argument("--temperature",    type=float, default=TEMPERATURE)
    p.add_argument("--vllm-mem",       type=float, default=VLLM_MEM, help="Fraction of the GPU for colocate vLLM.")
    p.add_argument("--per-device-bs",  type=int,   default=PER_DEVICE_BS)
    p.add_argument("--grad-accum",     type=int,   default=GRAD_ACCUM)
    p.add_argument("--n-samples",      type=int,   default=None, help="Limit to first N problems (debug).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.init_adapter and not Path(args.init_adapter).exists():
        sys.exit(f"--init-adapter path does not exist: {args.init_adapter}")

    # ── 1. CUDA + perf knobs ────────────────────────────────────────────────
    banner("STEP 1 / 6  CUDA sanity check")
    import torch
    step(f"torch {torch.__version__} cuda {torch.version.cuda}")
    step(f"cuda available: {torch.cuda.is_available()}, "
         f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # ── 2. Imports ──────────────────────────────────────────────────────────
    banner("STEP 2 / 6  Imports (transformers, peft, trl, datasets)")
    step("importing ...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        sys.exit("trl not installed. Run: .venv/bin/python -m pip install 'trl>=0.19.0'")
    step("imports done.")

    # judger is the same verifier used by run_inference.py / test_inference.py.
    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)
    correctness_reward, format_reward = make_reward_fns(judger)
    step("judger + reward functions ready.")

    # ── 3. Tokenizer + 4-bit base model + LoRA ──────────────────────────────
    banner("STEP 3 / 6  Load tokenizer + 4-bit base model + LoRA")
    step("loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left padding for generation

    step("loading base model in 4-bit NF4 (this takes a minute) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    step(f"base model loaded in {time.time() - t0:.1f}s.")
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
    )

    if args.init_adapter:
        step(f"loading existing LoRA adapter (trainable) from {args.init_adapter} ...")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        step("attaching a fresh LoRA adapter (no --init-adapter given) ...")
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    # ── 4. Build the GRPO dataset ───────────────────────────────────────────
    banner("STEP 4 / 6  Build prompt dataset from public.jsonl")
    step(f"reading {args.data} ...")
    raw = [json.loads(line) for line in open(args.data)]
    if args.n_samples:
        raw = raw[:args.n_samples]
    step(f"loaded {len(raw)} problems.")

    records = []
    for item in raw:
        system, user = build_prompt(item["question"], item.get("options"))
        gold = item["answer"]
        if not isinstance(gold, list):
            gold = [gold]
        records.append({
            # conversational prompt: GRPOTrainer applies the chat template and
            # appends the generation prompt (Qwen3-Thinking opens a <think> block).
            "prompt": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            # gold stashed as JSON so the column has a uniform (string) dtype;
            # the reward fn json.loads() it back.
            "gold":   json.dumps(gold),
            "is_mcq": bool(item.get("options")),
        })
    dataset = Dataset.from_list(records).shuffle(seed=SEED)
    n_mcq = sum(r["is_mcq"] for r in records)
    step(f"built {len(dataset)} prompts  ({n_mcq} MCQ, {len(records) - n_mcq} free-form).")

    # ── 5. GRPO config ──────────────────────────────────────────────────────
    banner("STEP 5 / 6  Configure GRPO (Dr.GRPO loss, vLLM colocate)")
    global_batch = args.per_device_bs * args.grad_accum
    if global_batch % args.num_generations != 0:
        sys.exit(f"per-device-bs*grad-accum ({global_batch}) must be divisible "
                 f"by num-generations ({args.num_generations}).")
    step(f"global batch {global_batch} -> {global_batch // args.num_generations} prompt(s)/step, "
         f"{args.num_generations} rollouts each.")

    grpo_config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        bf16=True,
        tf32=True,
        max_grad_norm=0.2,

        # ── GRPO-specific ──
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_completion,
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        beta=args.beta,                      # KL vs. adapter-disabled reference
        loss_type="dr_grpo",                 # length-bias-free GRPO variant
        scale_rewards=False,                 # recommended with dr_grpo
        mask_truncated_completions=True,     # 2k-truncated rollouts don't poison loss
        num_iterations=1,                    # GRPO inner epochs (mu)
        reward_weights=REWARD_WEIGHTS,

        # ── vLLM colocate generation ──
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_mem,
        vllm_tensor_parallel_size=1,

        # ── logging / checkpointing ──
        logging_steps=2,
        log_completions=True,
        num_completions_to_print=2,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[correctness_reward, format_reward],
        processing_class=tokenizer,
    )

    # ── 6. Train ────────────────────────────────────────────────────────────
    banner("STEP 6 / 6  Train (watch 'reward' and 'reward_std' climb)")
    step("starting GRPO ...")
    t0 = time.time()
    trainer.train()
    step(f"training done in {(time.time() - t0) / 60:.1f} min.")

    step(f"saving adapter to {args.output} ...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    step("done.")
    banner("DONE  —  evaluate with: "
           f"test_inference.py --adapter {args.output}")


if __name__ == "__main__":
    main()
