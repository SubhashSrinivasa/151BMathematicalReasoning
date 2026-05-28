"""
GRPO reinforcement learning of Qwen3-4B-Thinking on public.jsonl.

Variant of train_grpo.py with fixes for multi-GPU-type pools (e.g. Titan RTX / T4 /
V100 vs Ampere+):

  - PEFT checkpoint resume: neutralises tensor-parallel sharding import that breaks
    on transformers 4.57 + peft 0.19 (single-GPU training).
  - Pre-Ampere fp16 training: TRL 1.x casts QLoRA trainable weights to bf16
    unconditionally; this script re-casts them to fp32 so fp16 autocast + GradScaler
    work (GradScaler cannot unscale bf16 or fp16 gradients — only fp32).

This is the RL stage that sits *between* SFT (train_lora.py / the CoT RFT-SFT)
and inference (test_inference.py). It takes a LoRA policy, samples groups of
chain-of-thought rollouts per problem, scores each rollout with the competition
judger, and does a GRPO policy-gradient update — no value model, no reward
model, just the verifiable judge.

ONE-TIME extra deps (on top of the run_inference.py / train_lora.py setup):
    .venv/bin/python -m pip install "trl>=0.19.0" peft datasets accelerate

To run (continuing from a CoT-capable SFT adapter — recommended):
    .venv/bin/python train_grpo1.py --init-adapter checkpoints/qwen3-4b-lora-cot

To run from the base model directly (Qwen3-Thinking already reasons natively):
    .venv/bin/python train_grpo1.py

What it does:
  - Loads Qwen3-4B-Thinking-2507 in 4-bit NF4 (QLoRA), attaches/loads a LoRA.
  - For each problem, vLLM (colocate) samples NUM_GENERATIONS rollouts.
  - Each rollout is scored: 0.8 * binary-correct + 0.2 * partial-credit, plus a
    small format reward for a well-formed \\boxed{}. The judger is the same one
    used by run_inference.py / test_inference.py.
  - GRPO normalizes reward within each group and updates the LoRA.
  - Saves the trained adapter to checkpoints/qwen3-4b-grpo1.

After training, evaluate exactly like an SFT adapter:
    .venv/bin/python test_inference.py --adapter checkpoints/qwen3-4b-grpo1

MEMORY (24 GB GPU): vLLM colocate keeps a bf16 base copy (~8.2 GB) AND the
4-bit training copy resident. --vllm-mem splits the card; 0.42 leaves the
loss step (which materializes a big logits tensor) enough room. If the
training step OOMs, lower --vllm-mem further; if vLLM fails to initialize,
raise it. NOTE: a CUDA OOM in the backward pass may surface as an
"NVML_SUCCESS ... INTERNAL ASSERT" — that is just PyTorch's OOM reporter
failing inside the container; treat it as a plain out-of-memory.

CHECKPOINTS: a full resumable checkpoint (adapter + optimizer + scheduler +
RNG + step counter) is written every --save-every-min minutes; only the
newest is kept. Re-running the script auto-resumes from it, so a GPU/server
crash costs at most that many minutes — not the whole run.
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
# expandable_segments curbs VRAM fragmentation in the GRPO loss step.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID         = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH        = "data/public.jsonl"
OUTPUT_DIR       = "checkpoints/qwen3-4b-grpo1"

NUM_GENERATIONS  = 6       # rollouts per problem (GRPO group); 6 = less noisy advantage
MAX_PROMPT_LEN   = 2048    # prompt-length budget; with MAX_COMPLETION sizes the vLLM context
MAX_COMPLETION   = 2048    # reasoning budget; full length (avoids truncating ~half the rollouts)
PER_DEVICE_BS    = 1       # completions per loss forward; bs=1 fits 24 GB (logits tensor is huge)
GRAD_ACCUM       = 12      # global batch = PER_DEVICE_BS*GRAD_ACCUM = 12 -> 2 prompts/step
NUM_EPOCHS       = 1
LR               = 1e-5
BETA             = 0.0     # KL coeff; 0 skips the reference-model forward each step (faster)
NUM_ITERATIONS   = 1       # GRPO inner epochs (mu); 1 = on-policy, fewest loss passes (fastest)
TEMPERATURE      = 1.0     # rollout sampling temperature (diversity for the group)
VLLM_MEM         = 0.42    # colocate vLLM share; lower = more VRAM for the loss step (anti-OOM)

LORA_R           = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.0     # dropout off for RL: keeps gen/train log-probs consistent
SEED             = 42
SAVE_EVERY_MIN   = 30      # wall-clock minutes between resumable checkpoints (newest kept)

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


def align_lora_trainable_dtype(model, dtype, reason: str) -> int:
    """Cast trainable params to *dtype* (undo TRL's unconditional bf16 QLoRA cast).

    On pre-Ampere use fp32: Trainer fp16=True runs autocast for compute but GradScaler
    requires fp32 trainable weights / gradients. 4-bit matmuls still use fp16 via
    bnb_4bit_compute_dtype.

    Returns the number of tensors updated."""
    import torch
    n = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype != dtype:
            param.data = param.data.to(dtype)
            n += 1
    if n:
        step(f"aligned {n} trainable tensor(s) to {dtype} ({reason})")
    return n


def best_attn_impl() -> str:
    """Use flash_attention_2 if the flash-attn package is importable AND the GPU
    is Ampere+ (FA2 requires it), else fall back to PyTorch-native sdpa. Lets the
    same script run on any GPU/env, whether or not flash-attn is installed."""
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
        step("attention backend: sdpa (pre-Ampere GPU)")
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
        impl = "flash_attention_2"
    except Exception:
        impl = "sdpa"
    step(f"attention backend: {impl}")
    return impl


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
    p.add_argument("--save-every-min", type=int,   default=SAVE_EVERY_MIN, help="Minutes between resumable checkpoints (only the newest is kept).")
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
    # Ampere+ (compute capability >= 8.0) has bf16 + tf32 tensor cores; older
    # GPUs (V100/T4/...) do not, so fall back to fp16 and disable tf32.
    cap = torch.cuda.get_device_capability(0)
    use_bf16 = cap[0] >= 8
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    step(f"compute capability {cap[0]}.{cap[1]} -> "
         f"{'bf16 + tf32' if use_bf16 else 'fp16 (pre-Ampere: no bf16/tf32)'}")
    if not use_bf16:
        step("note: pre-Ampere GPU — runs in fp16 (no bf16/tf32), slower than "
             "Ampere; fine with ~24 GB VRAM (Titan RTX), will OOM on a 16 GB card.")
        # vLLM can't use FlashAttention-2 on pre-Ampere and falls back to
        # FlashInfer, which JIT-compiles CUDA kernels with nvcc (often missing
        # in containers). Force the Triton attention backend — Triton does its
        # own JIT, so no nvcc is needed.
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
    torch.backends.cuda.matmul.allow_tf32 = use_bf16
    torch.backends.cudnn.allow_tf32 = use_bf16
    torch.set_float32_matmul_precision("high")

    # ── 2. Imports ──────────────────────────────────────────────────────────
    banner("STEP 2 / 6  Imports (transformers, peft, trl, datasets)")
    step("importing ...")
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              BitsAndBytesConfig, TrainerCallback)
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    # peft 0.19's checkpoint loader calls a tensor-parallel adapter-sharding
    # helper that imports a name absent from transformers 4.57 (a version skew).
    # We train on a single GPU (no tensor parallelism), so that helper is a
    # no-op here — neutralise it so --resume can load the adapter cleanly.
    import peft.utils.save_and_load as _peft_sl
    if hasattr(_peft_sl, "_maybe_shard_state_dict_for_tp"):
        _peft_sl._maybe_shard_state_dict_for_tp = lambda model, state_dict, *a, **k: state_dict
    from datasets import Dataset
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        sys.exit("trl not installed. Run: .venv/bin/python -m pip install 'trl>=0.19.0'")
    step("imports done.")

    # Force a full resumable checkpoint every --save-every-min minutes of wall
    # clock — survives GPU/server crashes. With save_total_limit=1 only the
    # newest is kept, so checkpoints don't accumulate.
    class TimedCheckpoint(TrainerCallback):
        def __init__(self, interval_s: float):
            self.interval_s = interval_s
            self.last = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if time.time() - self.last >= self.interval_s:
                control.should_save = True
                self.last = time.time()
            return control

    # TRL casts QLoRA trainable weights to bf16 in GRPOTrainer.__init__; on
    # pre-Ampere we train with fp16 autocast + GradScaler, which needs fp32 LoRA
    # weights. Re-align after checkpoint load (on_train_begin runs after resume).
    class LoraTrainableDtypeCallback(TrainerCallback):
        def __init__(self, train_dtype: torch.dtype):
            self.train_dtype = train_dtype

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            align_lora_trainable_dtype(model, self.train_dtype, "on_train_begin (post-resume)")
            return control

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
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=compute_dtype,
        attn_implementation=best_attn_impl(),
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

    records, prompt_lens, n_dropped = [], [], 0
    for item in raw:
        system, user = build_prompt(item["question"], item.get("options"))
        # conversational prompt: GRPOTrainer applies the chat template and
        # appends the generation prompt (Qwen3-Thinking opens a <think> block).
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        # TRL 1.x does NOT truncate prompts — an over-long one crashes vLLM.
        # Drop those problems here (measured exactly as GRPOTrainer tokenizes).
        n_tok = len(tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True))
        if n_tok > args.max_prompt:
            n_dropped += 1
            continue
        prompt_lens.append(n_tok)
        gold = item["answer"]
        if not isinstance(gold, list):
            gold = [gold]
        records.append({
            "prompt": messages,
            # gold stashed as JSON so the column has a uniform (string) dtype;
            # the reward fn json.loads() it back.
            "gold":   json.dumps(gold),
            "is_mcq": bool(item.get("options")),
        })
    dataset = Dataset.from_list(records).shuffle(seed=SEED)
    n_mcq = sum(r["is_mcq"] for r in records)
    prompt_lens.sort()
    step(f"prompt tokens: min={prompt_lens[0]} "
         f"median={prompt_lens[len(prompt_lens) // 2]} max={prompt_lens[-1]}")
    step(f"built {len(dataset)} prompts  ({n_mcq} MCQ, {len(records) - n_mcq} free-form); "
         f"dropped {n_dropped} with prompt > {args.max_prompt} tokens.")

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
        bf16=use_bf16,
        fp16=not use_bf16,
        tf32=use_bf16,
        max_grad_norm=0.2,
        cast_lm_head_to_fp32=False,          # bf16 logits -> halve the loss-step logits tensor

        # ── GRPO-specific ──
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        beta=args.beta,                      # KL vs. adapter-disabled reference
        loss_type="dr_grpo",                 # length-bias-free GRPO variant
        scale_rewards=False,                 # recommended with dr_grpo
        mask_truncated_completions=True,     # 2k-truncated rollouts don't poison loss
        num_iterations=NUM_ITERATIONS,       # reuse each rollout batch for mu updates
        reward_weights=REWARD_WEIGHTS,

        # ── vLLM colocate generation ──
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_mem,
        vllm_max_model_length=args.max_prompt + args.max_completion + 128,  # ceiling + small margin
        vllm_tensor_parallel_size=1,

        # ── logging / checkpointing ──
        logging_steps=2,
        log_completions=False,
        num_completions_to_print=2,
        save_strategy="no",          # checkpoint saves are driven by TimedCheckpoint
        save_total_limit=1,          # keep only the newest checkpoint (delete older)
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
    trainer.add_callback(TimedCheckpoint(args.save_every_min * 60))
    lora_trainable_dtype = torch.bfloat16 if use_bf16 else torch.float32
    if not use_bf16:
        align_lora_trainable_dtype(
            trainer.model, lora_trainable_dtype,
            "post-GRPOTrainer init (undo TRL bf16 cast; fp32 for GradScaler)",
        )
        trainer.add_callback(LoraTrainableDtypeCallback(lora_trainable_dtype))

    # Record dtype policy next to checkpoints (helps when moving between GPU types).
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "training_meta.json"), "w") as f:
        json.dump({
            "use_bf16": use_bf16,
            "compute_dtype": str(compute_dtype),
            "lora_trainable_dtype": str(lora_trainable_dtype),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(cap),
        }, f, indent=2)

    # ── 6. Train ────────────────────────────────────────────────────────────
    banner("STEP 6 / 6  Train (watch 'reward' and 'reward_std' climb)")
    # Auto-resume: if a checkpoint exists in output_dir, continue from it
    # (optimizer/scheduler/RNG/step all restored); otherwise start fresh.
    last_ckpt = get_last_checkpoint(args.output) if os.path.isdir(args.output) else None
    if last_ckpt:
        step(f"resuming from checkpoint: {last_ckpt}")
    else:
        step("no checkpoint found — starting GRPO from scratch.")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last_ckpt)
    step(f"training done in {(time.time() - t0) / 60:.1f} min.")

    step(f"saving adapter to {args.output} ...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    step("done.")
    banner("DONE  —  evaluate with: "
           f"test_inference.py --adapter {args.output}")


if __name__ == "__main__":
    main()
