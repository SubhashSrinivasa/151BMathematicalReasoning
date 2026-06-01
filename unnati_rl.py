"""
GRPO reinforcement learning on public.jsonl — bf16 GPUs (Blackwell, A30, Ampere+).

Tuned for 24 GB cards with native bf16 (compute capability >= 8): e.g. RTX
Blackwell 6000, NVIDIA A30, RTX 3090/4090, A100-40GB (also fine with more VRAM).
For pre-Ampere 24 GB cards (Titan RTX, V100, T4) use train_grpo1.py instead.

Differences from train_grpo1.py:
  - bf16 + tf32 only (no fp16 / GradScaler / fp32 LoRA workarounds).
  - Same 24 GB memory budget as train_grpo1 (vllm-mem ~0.42, per-device-bs=1).
  - Flash Attention 2 when installed; no Triton-only vLLM hack.
  - Safe resume: skips scaler.pt when checkpoint came from fp16 training.
  - Refuses to run on GPUs with compute capability < 8.

A30: supported (cc 8.0, bf16). Slower than Blackwell but same code path.
Blackwell: supported (cc >= 10); needs a recent torch build for your CUDA.

Requires torch built for your CUDA (see requirements.txt).

Run (recommended: continue from CoT SFT):
    .venv/bin/python train_grpo_blackwell.py --init-adapter checkpoints/qwen3-4b-lora-cot

Continue policy from a Titan checkpoint (weights only — new output dir):
    .venv/bin/python train_grpo_blackwell.py \\
        --init-adapter checkpoints/qwen3-4b-grpo1/checkpoint-74 \\
        --output checkpoints/qwen3-4b-grpo-blackwell

Fresh GRPO (auto-resumes from latest checkpoint-* in --output):
    .venv/bin/python train_grpo_blackwell.py

Evaluate:
    .venv/bin/python test_inference.py --adapter checkpoints/qwen3-4b-grpo-blackwell
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
# Prefer PYTORCH_ALLOC_CONF (PYTORCH_CUDA_ALLOC_CONF is deprecated in torch 2.9+).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ── Configuration (24 GB bf16 GPUs — Blackwell, A30, etc.) ───────────────────
MODEL_ID         = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH        = "data/public_with_reasoning.jsonl"
OUTPUT_DIR       = "checkpoints/qwen3-4b-grpo-blackwell"

MIN_CC_MAJOR     = 8       # refuse pre-Ampere; use train_grpo1.py below this
BLACKWELL_CC_MAJOR = 10    # informational only; A30/Ampere (cc 8–9) still run fine

NUM_GENERATIONS  = 6
MAX_PROMPT_LEN   = 2000
MAX_COMPLETION   = 1536  
PER_DEVICE_BS    = 1       # logits tensor is huge; bs=1 for 24 GB
GRAD_ACCUM       = 12      # global batch 12 -> 2 prompts/step with 6 generations
NUM_EPOCHS       = 1
LR               = 1e-5
BETA             = 0.0
NUM_ITERATIONS   = 1
TEMPERATURE      = 0.7
VLLM_MEM         = 0.70    # 24 GB split: vLLM colocate + 4-bit train + loss logits

LORA_R           = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.0
SEED             = 42
SAVE_EVERY_STEPS = 5       # full resumable checkpoint every N optimizer steps

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

    if getattr(Trainer, "_blackwell_scaler_patch", False):
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
    Trainer._blackwell_scaler_patch = True


def attn_implementation() -> str:
    """Prefer FlashAttention 2 on Ampere+ / Blackwell."""
    try:
        import flash_attn  # noqa: F401
        impl = "flash_attention_2"
    except Exception:
        impl = "sdpa"
        step("flash-attn not installed — using sdpa (pip install flash-attn --no-build-isolation for speed)")
    step(f"attention backend: {impl}")
    return impl


def assert_modern_gpu() -> tuple[int, int]:
    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA not available.")
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    step(f"torch {torch.__version__} cuda {torch.version.cuda}")
    step(f"device: {name}")
    step(f"compute capability {cap[0]}.{cap[1]}")
    if cap[0] < MIN_CC_MAJOR:
        sys.exit(
            f"train_grpo_blackwell.py requires compute capability >= {MIN_CC_MAJOR}.0 "
            f"(got {cap[0]}.{cap[1]}). Use train_grpo1.py for Titan / V100 / T4."
        )
    if cap[0] < BLACKWELL_CC_MAJOR:
        step(
            f"note: cc {cap[0]}.{cap[1]} (e.g. A30/Ampere) — bf16 path OK; "
            f"Blackwell is cc>={BLACKWELL_CC_MAJOR}. Defaults assume 24 GB VRAM."
        )
    return cap


# ── Prompts / rewards (same as train_grpo1.py) ───────────────────────────────
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put ONLY the final required answer(s) inside \\boxed{}. "
    "For single-answer questions, output exactly ONE final answer inside \\boxed{}."
    " If and ONLY if the problem explicitly asks for multiple final answers "
    "(multiple blanks, multiple parts, select-all-that-apply), put them in one box separated by commas. "
    "Do NOT include intermediate values, calculations, candidate answers, or extra quantities inside \\boxed{}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and answer choices carefully and select the SINGLE best answer. "
    "Think step-by-step internally, but output ONLY ONE final answer letter "
    "inside exactly one \\boxed{}, e.g. \\boxed{C}. "
    "Never output multiple letters, multiple choices, commas, intermediate reasoning, "
    "or extra text inside \\boxed{}. "
    "The final answer must be exactly one of: A, B, C, D, E, F, G."
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


def _alarm_guard(seconds: int):
    def handler(signum, frame):
        raise TimeoutError("judge timeout")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)


def make_reward_fns(judger):
    csv_path = Path("results/grpo_generated_answers.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

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

    def log_generation(text: str, gold_list: list, is_mcq_val: bool, reward_val: float):
        import csv
        exists = csv_path.exists()
        boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
        final_box = boxes[-1].strip() if boxes else ""

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow([
                    "is_mcq",
                    "gold",
                    "final_box",
                    "reward",
                    "full_response"
                ])
            writer.writerow([
                is_mcq_val,
                json.dumps(gold_list),
                final_box,
                reward_val,
                text
            ])

    def correctness_reward(completions, gold, is_mcq, log_metric=None, **kwargs):
        rewards = []
        for comp, g_json, mcq in zip(completions, gold, is_mcq):
            text = _text(comp)
            gold_list = json.loads(g_json)
            if not isinstance(gold_list, list):
                gold_list = [gold_list]
            binary, partial = _judge_one(text, gold_list, bool(mcq))
            reward_val = 0.8 * binary + 0.2 * partial
            rewards.append(reward_val)

            log_generation(
                text=text,
                gold_list=gold_list,
                is_mcq_val=bool(mcq),
                reward_val=reward_val,
            )
        if log_metric is not None and rewards:
            n = len(rewards)
            log_metric(
                "correctness_reward_zero_ratio",
                sum(1 for r in rewards if r == 0.0) / n,
            )
            # Combined weighted reward (same as TRL: w0*correctness + w1*format).
            fmt = [1.0 if has_boxed(_text(c)) else 0.0 for c in completions]
            combined = [
                REWARD_WEIGHTS[0] * r + REWARD_WEIGHTS[1] * f
                for r, f in zip(rewards, fmt, strict=True)
            ]
            log_metric("reward_zero_ratio", sum(1 for x in combined if x == 0.0) / n)
        return rewards

    def format_reward(completions, gold=None, is_mcq=None, log_metric=None, **kwargs):
        rewards = []
        for comp, g_json, mcq in zip(completions, gold, is_mcq):
            text = _text(comp)
            boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
            if not boxes or not boxes[-1].strip():
                rewards.append(0.0)
                continue

            final = boxes[-1].strip()
            gold_list = json.loads(g_json)
            if not isinstance(gold_list, list):
                gold_list = [gold_list]

            r = 1.0

            # Penalize multiple boxed outputs
            if len(boxes) > 1:
                r -= 0.4

            # Penalize comma answers when gold expects one answer
            if len(gold_list) == 1 and "," in final:
                r -= 0.5

            # MCQ should be exactly one letter
            if bool(mcq) and not re.fullmatch(r"[A-Za-z]", final):
                r -= 0.4

            rewards.append(max(0.0, r))

        if log_metric is not None and rewards:
            log_metric(
                "format_reward_zero_ratio",
                sum(1 for r in rewards if r == 0.0) / len(rewards),
            )
        return rewards
    return correctness_reward, format_reward


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO RL on bf16 GPUs with cc>=8 (Blackwell, A30, 3090/4090). "
                    "24 GB defaults. For Titan RTX use train_grpo1.py.",
    )
    p.add_argument("--model",           default=MODEL_ID)
    p.add_argument("--init-adapter",    default="data/lora_math_adapter/final_adapter",
                   help="LoRA to start from (SFT adapter or a prior GRPO checkpoint dir).")
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
    p.add_argument("--n-samples",        type=int,   default=None)
    p.add_argument("--save-every-steps", type=int,   default=SAVE_EVERY_STEPS,
                   help="Save a full resumable checkpoint every N optimizer steps (save_total_limit keeps the newest).")
    p.add_argument("--no-resume",        action="store_true",
                   help="Ignore existing checkpoints in --output and train from scratch "
                        "(still uses --init-adapter if set).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.init_adapter and not Path(args.init_adapter).exists():
        sys.exit(f"--init-adapter path does not exist: {args.init_adapter}")

    banner("STEP 1 / 6  CUDA sanity check (bf16, 24 GB profile)")
    import torch
    cap = assert_modern_gpu()
    compute_dtype = torch.bfloat16
    step("training precision: bf16 + tf32 (no fp16 GradScaler path)")
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

    # Force saves via control.should_save (train_grpo1 TimedCheckpoint pattern).
    class GrpoTrainingLogCallback(TrainerCallback):
        """Log checkpoint triggers/saves and flag suspicious training metrics."""

        def __init__(self, save_every_steps: int):
            self.save_every_steps = save_every_steps
            self._pending_save_step: Optional[int] = None

        def _list_checkpoint_dirs(self, output_dir: str) -> list[str]:
            if not os.path.isdir(output_dir):
                return []
            names = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            return sorted(names, key=lambda n: int(n.rsplit("-", 1)[-1]))

        def on_train_begin(self, args, state, control, **kwargs):
            step(
                f"checkpoint policy: StepCheckpoint every {self.save_every_steps} "
                f"global_steps, save_total_limit={args.save_total_limit}, "
                f"output_dir={args.output_dir}"
            )
            ckpts = self._list_checkpoint_dirs(args.output_dir)
            if ckpts:
                step(f"existing checkpoints: {', '.join(ckpts)}")
            else:
                step("existing checkpoints: (none)")
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % self.save_every_steps == 0:
                control.should_save = True
                self._pending_save_step = state.global_step
                step(
                    f"checkpoint REQUESTED at global_step={state.global_step} "
                    f"(every {self.save_every_steps} steps)"
                )
            return control

        def on_save(self, args, state, control, **kwargs):
            step_num = self._pending_save_step or state.global_step
            ckpt_name = f"checkpoint-{step_num}"
            ckpt_path = os.path.join(args.output_dir, ckpt_name)
            if os.path.isdir(ckpt_path):
                has_state = os.path.isfile(os.path.join(ckpt_path, "trainer_state.json"))
                has_adapter = os.path.isfile(os.path.join(ckpt_path, "adapter_config.json")) or os.path.isfile(
                    os.path.join(ckpt_path, "adapter_model.safetensors")
                )
                step(
                    f"checkpoint SAVED OK: {ckpt_path} "
                    f"(trainer_state.json={has_state}, adapter={has_adapter})"
                )
                ckpts = self._list_checkpoint_dirs(args.output_dir)
                step(f"checkpoint dirs now: {', '.join(ckpts) if ckpts else '(none)'}")
            else:
                warn(
                    f"checkpoint save finished but directory missing: {ckpt_path} "
                    f"(global_step={state.global_step})"
                )
            self._pending_save_step = None
            return control

        @staticmethod
        def _fmt_ratio(logs: dict, key: str) -> Optional[float]:
            v = logs.get(key)
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return control
            # Per-batch fraction of rollouts with zero reward (logged via reward fn log_metric).
            z_combined = self._fmt_ratio(logs, "reward_zero_ratio")
            z_corr = self._fmt_ratio(logs, "correctness_reward_zero_ratio")
            z_fmt = self._fmt_ratio(logs, "format_reward_zero_ratio")
            if z_combined is not None or z_corr is not None or z_fmt is not None:
                parts = []
                if z_combined is not None:
                    parts.append(f"combined reward=0: {z_combined:.1%}")
                if z_corr is not None:
                    parts.append(f"correctness=0: {z_corr:.1%}")
                if z_fmt is not None:
                    parts.append(f"no \\boxed{{}} (format=0): {z_fmt:.1%}")
                step(f"step {state.global_step} rollout zero-reward ratios — {', '.join(parts)}")

            issues: list[str] = []
            if logs.get("loss") == 0.0:
                issues.append("loss=0 (expected for GRPO — use reward / step_time)")
            reward = logs.get("reward")
            if reward is not None:
                try:
                    if reward != reward:  # NaN
                        issues.append("reward is NaN")
                    elif reward == 0.0:
                        issues.append("mean reward=0 on this log window")
                except TypeError:
                    pass
            if z_combined is not None and z_combined >= 0.9:
                issues.append(
                    f"reward_zero_ratio={z_combined:.1%} (almost all rollouts scored 0 total reward)"
                )
            elif z_combined is not None and z_combined >= 0.75:
                issues.append(f"reward_zero_ratio={z_combined:.1%} (mostly zero rewards)")
            clipped = logs.get("completions/clipped_ratio")
            if clipped is not None:
                try:
                    if float(clipped) >= 0.9:
                        issues.append(
                            f"completions/clipped_ratio={float(clipped):.3f} "
                            "(most rollouts truncated — consider lower --max-completion)"
                        )
                except (TypeError, ValueError):
                    pass
            step_time = logs.get("step_time")
            if step_time is not None:
                try:
                    st = float(step_time)
                    if st >= 900:
                        issues.append(f"step_time={st:.0f}s (very slow step)")
                    elif st >= 600:
                        issues.append(f"step_time={st:.0f}s (slow step)")
                except (TypeError, ValueError):
                    pass
            grad_norm = logs.get("grad_norm")
            if grad_norm is not None:
                try:
                    gn = float(grad_norm)
                    if gn == 0.0:
                        issues.append("grad_norm=0")
                    elif gn > 10.0:
                        issues.append(f"grad_norm={gn:.3f} (large)")
                except (TypeError, ValueError):
                    pass
            if issues:
                warn(f"step {state.global_step} metrics: " + "; ".join(issues))
            return control

    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)
    correctness_reward, format_reward = make_reward_fns(judger)
    step("judger + reward functions ready.")

    banner("STEP 3 / 6  Load tokenizer + 4-bit base model + LoRA")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
        attn_implementation=attn_implementation(),
    )
    step(f"base model loaded in {time.time() - t0:.1f}s.")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

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
    n_mcq = sum(r["is_mcq"] for r in records)
    prompt_lens.sort()
    step(f"prompt tokens: min={prompt_lens[0]} "
         f"median={prompt_lens[len(prompt_lens) // 2]} max={prompt_lens[-1]}")
    step(f"built {len(dataset)} prompts ({n_mcq} MCQ); dropped {n_dropped} overlong prompts.")

    banner("STEP 5 / 6  Configure GRPO")
    global_batch = args.per_device_bs * args.grad_accum
    if global_batch % args.num_generations != 0:
        sys.exit(f"per-device-bs*grad-accum ({global_batch}) must be divisible "
                 f"by num-generations ({args.num_generations}).")
    step(f"global batch {global_batch} -> {global_batch // args.num_generations} prompt(s)/step")
    step(f"checkpoint every {args.save_every_steps} optimizer step(s) (newest kept via save_total_limit)")

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
        optim="adamw_8bit",
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
        vllm_max_model_length=args.max_prompt + args.max_completion + 128,
        vllm_tensor_parallel_size=1,
        logging_steps=2,
        log_completions=False,
        num_completions_to_print=2,
        save_strategy="no",           # saves driven by StepCheckpoint (like train_grpo1)
        save_steps=args.save_every_steps,
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
    trainer.add_callback(GrpoTrainingLogCallback(args.save_every_steps))

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "training_meta.json"), "w") as f:
        json.dump({
            "script": "train_grpo_blackwell.py",
            "use_bf16": True,
            "compute_dtype": COMPUTE_DTYPE_NAME,
            "save_every_steps": args.save_every_steps,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(cap),
        }, f, indent=2)

    banner("STEP 6 / 6  Train")
    last_ckpt = None
    if not args.no_resume and os.path.isdir(args.output):
        last_ckpt = get_last_checkpoint(args.output)
    if last_ckpt:
        step(f"resuming from checkpoint: {last_ckpt}")
    elif args.init_adapter:
        step("starting GRPO (--init-adapter weights; no checkpoint in --output).")
    else:
        step("starting GRPO from scratch.")
    t0 = time.time()
    try:
        trainer.train(resume_from_checkpoint=last_ckpt)
    except Exception as exc:
        banner("TRAINING FAILED")
        warn(f"{type(exc).__name__}: {exc}")
        ckpts = [
            d for d in os.listdir(args.output)
            if os.path.isdir(os.path.join(args.output, d)) and d.startswith("checkpoint-")
        ] if os.path.isdir(args.output) else []
        if ckpts:
            step(f"latest checkpoint dirs on disk: {', '.join(sorted(ckpts)[-3:])}")
        else:
            warn("no checkpoint-* directories found under output_dir")
        raise
    step(f"training done in {(time.time() - t0) / 60:.1f} min.")

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    banner(f"DONE  —  test_inference.py --adapter {args.output}")


if __name__ == "__main__":
    main()

