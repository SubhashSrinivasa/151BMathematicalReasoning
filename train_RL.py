"""
GRPO reinforcement learning on public.jsonl — tuned for NVIDIA L40 / L40S (~50 GB, Ada).

Variant: SKIP SymPy judging for unboxed outputs.

Motivation:
  - With strict_extract=False, judger.extract_ans() falls back to "last LaTeX" or "last number"
    when there is no explicit answer. On long reasoning traces this can feed SymPy garbage and
    cause multi-hour CPU stalls.
  - This variant gates correctness judging on the presence of a non-empty \\boxed{...}.

Behavior change vs train_grpo_l40s.py:
  - If a completion has no non-empty \\boxed{...}, correctness_reward is forced to 0.0 and
    judger.auto_judge / is_equal are NOT called for that completion.
  - Logs skipped ratio via log_metric: unboxed_skip_ratio.
  - Appends each skipped completion to <output>/unboxed_audit.jsonl (disable with --no-unboxed-audit).
    Offline counterfactual scoring: audit_unboxed_skipped.py or score_results.py --recompute.
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
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ── Configuration (~50 GB L40 / L40S Ada) — same tiers as train_grpo_l40s.py ─
MODEL_ID         = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH        = "data/public.jsonl"
OUTPUT_DIR       = "checkpoints/qwen3-4b-grpo-l40s-skip-unboxed"

MIN_CC_MAJOR       = 8
MIN_VRAM_GB        = 40.0
TARGET_VRAM_GB     = 50

NUM_GENERATIONS  = 6       # rollouts/prompt for the GRPO advantage estimate
MAX_PROMPT_LEN   = 2048
MAX_COMPLETION   = 4096    # long ceiling for the thinking model; seq can reach 2048+4096=6144
PER_DEVICE_BS    = 1       # 1 keeps the backward logits/activation spike safe at 6144-token seqs
GRAD_ACCUM       = 24      # global batch 1*12=12 -> 2 prompts/step (12 % 6 == 0)
NUM_EPOCHS       = 1
LR               = 1e-5
BETA             = 0.0
NUM_ITERATIONS   = 1
TEMPERATURE      = 0.8
VLLM_MEM         = 0.50    # more KV cache for the longer completions; bf16 train side still ~24 GB

LORA_R           = 32      # was 16 — more adapter capacity, negligible extra VRAM on 50 GB
LORA_ALPHA       = 64      # 2 * r
LORA_DROPOUT     = 0.0
SEED             = 42
SAVE_EVERY_STEPS = 10

REWARD_WEIGHTS   = [1.0, 0.2]
JUDGE_TIMEOUT_S  = 5

COMPUTE_DTYPE_NAME = "bfloat16"


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    print("=" * 70, flush=True)


def step(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] WARNING: {msg}", flush=True)


def patch_peft_tp_resume() -> None:
    """peft 0.19 + transformers 4.57: TP sharding import breaks single-GPU resume."""
    import peft.utils.save_and_load as _peft_sl
    if hasattr(_peft_sl, "_maybe_shard_state_dict_for_tp"):
        _peft_sl._maybe_shard_state_dict_for_tp = lambda model, state_dict, *a, **k: state_dict


def patch_trainer_scaler_resume() -> None:
    """bf16 training has no GradScaler; skip stale scaler.pt from fp16 checkpoints."""
    from transformers import Trainer

    if getattr(Trainer, "_l40s_scaler_patch", False):
        return
    _orig = Trainer._load_scaler

    def _load_scaler_safe(self, checkpoint):
        if checkpoint is None:
            return
        scaler_path = os.path.join(checkpoint, "scaler.pt")
        if os.path.isfile(scaler_path) and self.accelerator.scaler is None:
            step(
                "skipping scaler.pt on resume (bf16 has no GradScaler; "
                "checkpoint may be from fp16/Titan — optimizer state may still load)"
            )
            return
        return _orig(self, checkpoint)

    Trainer._load_scaler = _load_scaler_safe
    Trainer._l40s_scaler_patch = True


def attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401
        impl = "flash_attention_2"
    except Exception:
        impl = "sdpa"
        step("flash-attn not installed — using sdpa (pip install flash-attn --no-build-isolation)")
    step(f"attention backend: {impl}")
    return impl


def assert_l40s_gpu() -> tuple[tuple[int, int], float]:
    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA not available.")
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    step(f"torch {torch.__version__} cuda {torch.version.cuda}")
    step(f"device: {name}")
    step(f"compute capability {cap[0]}.{cap[1]}")
    step(f"VRAM: {vram_gb:.1f} GB")
    if cap[0] < MIN_CC_MAJOR:
        sys.exit(
            f"train_grpo_l40s_skip_unboxed.py requires compute capability >= {MIN_CC_MAJOR}.0 "
            f"(got {cap[0]}.{cap[1]}). Use train_grpo1.py for Titan / V100 / T4."
        )
    if vram_gb < MIN_VRAM_GB:
        warn(
            f"VRAM {vram_gb:.1f} GB < {MIN_VRAM_GB:.0f} GB expected for L40S defaults — "
            "tune down --max-completion / --vllm-mem / --num-generations"
        )
    elif vram_gb < TARGET_VRAM_GB - 3:
        step(
            f"note: {vram_gb:.1f} GB detected — defaults assume ~{TARGET_VRAM_GB} GB; "
            "if OOM try --max-completion 2048 --vllm-mem 0.46"
        )
    return cap, vram_gb


# ── Prompts / rewards ───────────────────────────────────────────────────────
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
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def has_boxed(text: str) -> bool:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    return bool(m and m.group(1).strip())


class UnboxedAuditWriter:
    """Append skipped (no \\boxed{}) rollouts for offline judger audit."""

    def __init__(self, path: Path):
        self.path = path
        self.global_step = 0
        self._count = 0
        path.parent.mkdir(parents=True, exist_ok=True)

    def set_global_step(self, step: int) -> None:
        self.global_step = int(step)

    def append(self, response: str, gold_json: str, is_mcq: bool) -> None:
        row = {
            "global_step": self.global_step,
            "response": response,
            "gold": json.loads(gold_json),
            "is_mcq": is_mcq,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._count += 1


def _alarm_guard(seconds: int):
    def handler(signum, frame):
        raise TimeoutError("judge timeout")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)


def make_reward_fns(judger, audit_writer: Optional[UnboxedAuditWriter] = None,
                    reward_acc: Optional[dict] = None):
    def _judge_one(text: str, gold_list: list, is_mcq: bool) -> tuple[float, float]:
        if is_mcq:
            ok = float(extract_letter(text) == str(gold_list[0]).strip().upper())
            return ok, ok
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
        if isinstance(completion, str):
            return completion
        return completion[-1]["content"] if completion else ""

    def correctness_reward(completions, gold, is_mcq, log_metric=None, **kwargs):
        rewards = []
        unboxed = 0
        for comp, g_json, mcq in zip(completions, gold, is_mcq):
            text = _text(comp)
            if not has_boxed(text):
                unboxed += 1
                if audit_writer is not None:
                    audit_writer.append(text, g_json, bool(mcq))
                rewards.append(0.0)
                continue
            gold_list = json.loads(g_json)
            if not isinstance(gold_list, list):
                gold_list = [gold_list]
            binary, partial = _judge_one(text, gold_list, bool(mcq))
            rewards.append(0.8 * binary + 0.2 * partial)
        n = len(rewards)
        fmt = [1.0 if has_boxed(_text(c)) else 0.0 for c in completions]
        combined = [
            REWARD_WEIGHTS[0] * r + REWARD_WEIGHTS[1] * f
            for r, f in zip(rewards, fmt, strict=True)
        ]
        # Accumulate per-batch reward stats; the step-end callback flushes them
        # as a single line per optimizer step (the reward fn is called once per
        # generation group, i.e. potentially many times per step).
        if reward_acc is not None and n:
            reward_acc["rewards"].extend(rewards)
            reward_acc["combined"].extend(combined)
            reward_acc["unboxed"] += unboxed
        if log_metric is not None and n:
            log_metric("unboxed_skip_ratio", unboxed / n)
            log_metric(
                "correctness_reward_zero_ratio",
                sum(1 for r in rewards if r == 0.0) / n,
            )
            log_metric("reward_zero_ratio", sum(1 for x in combined if x == 0.0) / n)
        return rewards

    def format_reward(completions, log_metric=None, **kwargs):
        rewards = [1.0 if has_boxed(_text(c)) else 0.0 for c in completions]
        if log_metric is not None and rewards:
            log_metric(
                "format_reward_zero_ratio",
                sum(1 for r in rewards if r == 0.0) / len(rewards),
            )
        return rewards

    return correctness_reward, format_reward


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO RL tuned for NVIDIA L40 ~50 GB (bf16, cc>=8). "
                    "This variant skips SymPy judging for unboxed completions.",
    )
    p.add_argument("--model",           default=MODEL_ID)
    p.add_argument("--init-adapter",    default=None,
                   help="LoRA to start from (CoT SFT or prior GRPO checkpoint dir).")
    p.add_argument("--data",            default=DATA_PATH)
    p.add_argument("--output",          default=OUTPUT_DIR)
    p.add_argument("--epochs",          type=int,   default=NUM_EPOCHS)
    p.add_argument("--lr",              type=float, default=LR)
    p.add_argument("--beta",            type=float, default=BETA)
    p.add_argument("--num-generations", type=int,   default=NUM_GENERATIONS)
    p.add_argument("--max-prompt",      type=int,   default=MAX_PROMPT_LEN)
    p.add_argument("--max-completion",  type=int,   default=MAX_COMPLETION)
    p.add_argument("--temperature",     type=float, default=TEMPERATURE)
    p.add_argument("--vllm-mem",        type=float, default=VLLM_MEM)
    p.add_argument("--per-device-bs",   type=int,   default=PER_DEVICE_BS)
    p.add_argument("--grad-accum",      type=int,   default=GRAD_ACCUM)
    p.add_argument(
        "--optim",
        default="adamw_torch",
        help="Optimizer. Default adamw_torch avoids bitsandbytes managed memory "
             "(cuMemAllocManaged), which is disabled on Thunder prototyping / UVM-off "
             "instances. Use paged_adamw_8bit only where CUDA UVM works (e.g. DSMLP).",
    )
    p.add_argument("--n-samples",       type=int,   default=None)
    p.add_argument("--save-every-steps", type=int,  default=SAVE_EVERY_STEPS)
    p.add_argument("--no-resume",       action="store_true")
    p.add_argument(
        "--load-4bit",
        action="store_true",
        help="QLoRA NF4 base instead of bf16. Lossy; only needed on <24 GB cards. "
             "On a ~50 GB L40 leave this OFF for best quality.",
    )
    p.add_argument(
        "--unboxed-audit",
        default=None,
        help="Append skipped unboxed completions here (default: <output>/unboxed_audit.jsonl).",
    )
    p.add_argument(
        "--no-unboxed-audit",
        action="store_true",
        help="Do not write unboxed_audit.jsonl during training.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.init_adapter and not Path(args.init_adapter).exists():
        sys.exit(f"--init-adapter path does not exist: {args.init_adapter}")

    banner("STEP 1 / 6  CUDA sanity check (bf16, L40 ~50 GB profile)")
    import torch
    cap, vram_gb = assert_l40s_gpu()
    compute_dtype = torch.bfloat16
    step("training precision: bf16 + tf32")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    banner("STEP 2 / 6  Imports")
    step("importing ...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    patch_peft_tp_resume()
    patch_trainer_scaler_resume()
    from datasets import Dataset
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        sys.exit("trl not installed. Run: .venv/bin/python -m pip install 'trl>=0.19.0'")
    step("imports done.")

    audit_writer: Optional[UnboxedAuditWriter] = None
    if not args.no_unboxed_audit:
        audit_path = Path(args.unboxed_audit or os.path.join(args.output, "unboxed_audit.jsonl"))
        audit_writer = UnboxedAuditWriter(audit_path)
        step(f"unboxed audit log: {audit_path} (append; offline: audit_unboxed_skipped.py)")

    class TrainingLogCallback(TrainerCallback):
        def __init__(self, save_every_steps: int, audit: Optional[UnboxedAuditWriter] = None,
                     reward_acc: Optional[dict] = None):
            self.save_every_steps = save_every_steps
            self.audit = audit
            self.reward_acc = reward_acc
            self._pending_save_step: Optional[int] = None

        def _flush_rewards(self, global_step: int) -> None:
            acc = self.reward_acc
            if not acc or not acc["rewards"]:
                return
            rw, cb = acc["rewards"], acc["combined"]
            n = len(rw)
            nonzero = sum(1 for r in rw if r > 0.0)
            step(
                f"step {global_step} rewards: n={n} "
                f"correct_mean={sum(rw) / n:.3f} total_mean={sum(cb) / n:.3f} "
                f"correct_range=[{min(rw):.2f},{max(rw):.2f}] "
                f"nonzero={nonzero}/{n} unboxed={acc['unboxed']}/{n} ({acc['unboxed'] / n:.0%})"
            )
            acc["rewards"].clear()
            acc["combined"].clear()
            acc["unboxed"] = 0

        def _list_checkpoint_dirs(self, output_dir: str) -> list[str]:
            if not os.path.isdir(output_dir):
                return []
            names = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            return sorted(names, key=lambda n: int(n.rsplit("-", 1)[-1]))

        def on_step_end(self, args, state, control, **kwargs):
            self._flush_rewards(state.global_step)
            if self.audit is not None:
                self.audit.set_global_step(state.global_step)
            if state.global_step > 0 and state.global_step % self.save_every_steps == 0:
                control.should_save = True
                self._pending_save_step = state.global_step
                step(f"checkpoint REQUESTED at global_step={state.global_step}")
            return control

        def on_save(self, args, state, control, **kwargs):
            step_num = self._pending_save_step or state.global_step
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-{step_num}")
            if os.path.isdir(ckpt_path):
                step(f"checkpoint SAVED OK: {ckpt_path}")
                ckpts = self._list_checkpoint_dirs(args.output_dir)
                step(f"checkpoint dirs now: {', '.join(ckpts) if ckpts else '(none)'}")
            else:
                warn(f"checkpoint directory missing after save: {ckpt_path}")
            self._pending_save_step = None
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return control
            # Print the new skip metric if present
            if "unboxed_skip_ratio" in logs:
                step(f"step {state.global_step} unboxed_skip_ratio={float(logs['unboxed_skip_ratio']):.1%}")
            return control

    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)
    reward_acc = {"rewards": [], "combined": [], "unboxed": 0}
    correctness_reward, format_reward = make_reward_fns(
        judger, audit_writer=audit_writer, reward_acc=reward_acc)
    step("judger + reward functions ready.")

    precision_label = "4-bit NF4 QLoRA" if args.load_4bit else "bf16"
    banner(f"STEP 3 / 6  Load tokenizer + base model ({precision_label}) + LoRA")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs = dict(
        device_map={"": 0},
        dtype=compute_dtype,
        attn_implementation=attn_implementation(),
    )
    if args.load_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        step("bf16 base (no quantization) — a 4B model is ~8 GB, fits with room on ~50 GB")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    step(f"base model loaded in {time.time() - t0:.1f}s.")
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # Non-quantized: enable input grads so gradient checkpointing works with a
        # frozen base + LoRA (prepare_model_for_kbit_training does this for the 4-bit path).
        model.enable_input_require_grads()

    if args.init_adapter:
        step(f"loading LoRA from {args.init_adapter} ...")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        step("attaching fresh LoRA ...")
        model = get_peft_model(model, LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ))
    model.print_trainable_parameters()
    model.config.use_cache = False

    banner("STEP 4 / 6  Build prompt dataset")
    raw = [json.loads(line) for line in open(args.data)]
    if args.n_samples:
        raw = raw[:args.n_samples]
    step(f"loaded {len(raw)} problems.")

    records, prompt_lens, n_dropped = [], [], 0
    for item in raw:
        system, user = build_prompt(item["question"], item.get("options"))
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
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
            "gold": json.dumps(gold),
            "is_mcq": bool(item.get("options")),
        })
    dataset = Dataset.from_list(records).shuffle(seed=SEED)
    prompt_lens.sort()
    step(f"prompt tokens: min={prompt_lens[0]} "
         f"median={prompt_lens[len(prompt_lens) // 2]} max={prompt_lens[-1]}")
    step(f"built {len(dataset)} prompts; dropped {n_dropped} overlong prompts.")

    banner("STEP 5 / 6  Configure GRPO")
    global_batch = args.per_device_bs * args.grad_accum
    if global_batch % args.num_generations != 0:
        sys.exit(f"per-device-bs*grad-accum ({global_batch}) must be divisible "
                 f"by num-generations ({args.num_generations}).")
    prompts_per_step = global_batch // args.num_generations
    vllm_ctx = args.max_prompt + args.max_completion + 128
    step(f"global batch {global_batch} -> {prompts_per_step} prompt(s)/step")
    step(
        f"L40 ~50GB profile: max_completion={args.max_completion}, num_generations={args.num_generations}, "
        f"vllm_mem={args.vllm_mem}, vllm_max_model_length≈{vllm_ctx}"
    )
    if not args.load_4bit and vram_gb < 44:
        warn(
            f"bf16 base on {vram_gb:.0f} GB is tight — if you OOM, add --load-4bit "
            "or lower --vllm-mem / --max-completion"
        )
    if args.max_completion >= 4096 and args.num_generations > 6:
        warn(
            f"max_completion={args.max_completion} with num_generations={args.num_generations} "
            f"is aggressive on {vram_gb:.0f} GB — drop --num-generations to 4-6 if backward OOMs"
        )
    if args.max_completion >= 5120:
        warn(
            f"max_completion={args.max_completion} is experimental on ~50 GB colocate GRPO"
        )
    if args.load_4bit:
        step("4-bit base requested — quality is lower than bf16; only needed on <24 GB cards")
    step("boxed gate: unboxed completions skip SymPy judging (correctness=0)")

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
        optim=args.optim,
        bf16=True,
        fp16=False,
        tf32=True,
        max_grad_norm=0.2,
        cast_lm_head_to_fp32=False,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        beta=args.beta,
        loss_type="dr_grpo",
        scale_rewards=False,
        mask_truncated_completions=True,
        num_iterations=NUM_ITERATIONS,
        reward_weights=REWARD_WEIGHTS,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_mem,
        vllm_max_model_length=vllm_ctx,
        vllm_tensor_parallel_size=1,
        logging_steps=2,
        log_completions=False,
        num_completions_to_print=2,
        save_strategy="no",
        save_steps=args.save_every_steps,
        save_total_limit=3,
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
    trainer.add_callback(TrainingLogCallback(
        args.save_every_steps, audit=audit_writer, reward_acc=reward_acc))

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "training_meta.json"), "w") as f:
        json.dump({
            "script": "train_grpo_l40s_skip_unboxed.py",
            "profile": "l40s_48gb_skip_unboxed",
            "use_bf16": True,
            "compute_dtype": COMPUTE_DTYPE_NAME,
            "base_4bit": bool(args.load_4bit),
            "lora_r": LORA_R,
            "max_completion": args.max_completion,
            "num_generations": args.num_generations,
            "vllm_mem": args.vllm_mem,
            "save_every_steps": args.save_every_steps,
            "gpu": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1),
            "compute_capability": list(cap),
            "boxed_gate": True,
            "unboxed_audit": None if args.no_unboxed_audit else str(
                Path(args.unboxed_audit or os.path.join(args.output, "unboxed_audit.jsonl"))
            ),
        }, f, indent=2)

    banner("STEP 6 / 6  Train")
    last_ckpt = None
    if not args.no_resume and os.path.isdir(args.output):
        from transformers.trainer_utils import get_last_checkpoint
        last_ckpt = get_last_checkpoint(args.output)
    if last_ckpt:
        step(f"resuming from checkpoint: {last_ckpt}")
    elif args.init_adapter:
        step("starting GRPO (--init-adapter weights; no checkpoint in --output).")
    else:
        step("starting GRPO from scratch.")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last_ckpt)
    step(f"training done in {(time.time() - t0) / 60:.1f} min.")
    if audit_writer is not None:
        step(f"unboxed audit: {audit_writer._count} rows appended -> {audit_writer.path}")

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    banner(f"DONE  —  test_inference.py --adapter {args.output}")


if __name__ == "__main__":
    main()

