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

# ── Configuration (L40 48GB, cc 8.9) ─────────────────────────────────────────
MODEL_ID           = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH          = "data/public.jsonl"
OUTPUT_DIR         = "checkpoints/qwen3-4b-grpo-v2"

MIN_CC_MAJOR       = 8
BLACKWELL_CC_MAJOR = 10

NUM_GENERATIONS    = 6
MAX_PROMPT_LEN     = 2048
MAX_COMPLETION     = 2048
PER_DEVICE_BS      = 1
GRAD_ACCUM         = 12
NUM_EPOCHS         = 1          # CHANGED: 2 -> 1 (prevent shortcut learning on small datasets)
LR                 = 5e-6
BETA               = 0.01
NUM_ITERATIONS     = 1
TEMPERATURE        = 1.0
VLLM_MEM           = 0.5

LORA_R             = 16
LORA_ALPHA         = 32
LORA_DROPOUT       = 0.0
SEED               = 42
SAVE_EVERY_STEPS   = 5

REWARD_WEIGHTS     = [1.0, 0.2, 0.3, 0.3]  # CHANGED: added 0.3 for thinking length reward
JUDGE_TIMEOUT_S    = 5

COMPUTE_DTYPE_NAME = "bfloat16"

HEDGING_PHRASES = [
    "complex or challenging question",
    "difficult to provide a direct",
    "i need to think about it",
    "cannot answer",
    "i cannot provide",
    "unable to answer",
]

# ADDED: minimum thinking length to prevent empty <think> shortcut
MIN_THINK_LEN      = 100   # chars — below this is penalized
GOOD_THINK_LEN     = 500   # chars — above this gets a small bonus


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    print("=" * 70, flush=True)


def step(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] WARNING: {msg}", flush=True)


def patch_peft_tp_resume() -> None:
    import peft.utils.save_and_load as _peft_sl
    if hasattr(_peft_sl, "_maybe_shard_state_dict_for_tp"):
        _peft_sl._maybe_shard_state_dict_for_tp = lambda model, state_dict, *a, **k: state_dict


def patch_trainer_scaler_resume() -> None:
    from transformers import Trainer
    if getattr(Trainer, "_blackwell_scaler_patch", False):
        return
    _orig = Trainer._load_scaler

    def _load_scaler_safe(self, checkpoint):
        if checkpoint is None:
            return
        scaler_path = os.path.join(checkpoint, "scaler.pt")
        if os.path.isfile(scaler_path) and self.accelerator.scaler is None:
            step("skipping scaler.pt on resume (bf16 has no GradScaler)")
            return
        return _orig(self, checkpoint)

    Trainer._load_scaler = _load_scaler_safe
    Trainer._blackwell_scaler_patch = True


def attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401
        impl = "flash_attention_2"
    except Exception:
        impl = "sdpa"
        step("flash-attn not installed — using sdpa")
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
        sys.exit(f"requires compute capability >= {MIN_CC_MAJOR}.0 (got {cap[0]}.{cap[1]})")
    if cap[0] < BLACKWELL_CC_MAJOR:
        step(f"note: cc {cap[0]}.{cap[1]} (Ampere/Ada) — bf16 path OK")
    return cap


SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "You MUST always attempt the problem and provide a final answer in \\boxed{}. "
    "Never say the problem is too complex or that you cannot answer. "
    "Even if uncertain, make your best attempt. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "You MUST always choose an answer — never say the problem is too complex. "
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


def _alarm_guard(seconds: int):
    def handler(signum, frame):
        raise TimeoutError("judge timeout")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)


def make_reward_fns(judger):
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
        for comp, g_json, mcq in zip(completions, gold, is_mcq):
            text = _text(comp)
            gold_list = json.loads(g_json)
            if not isinstance(gold_list, list):
                gold_list = [gold_list]
            binary, partial = _judge_one(text, gold_list, bool(mcq))
            rewards.append(0.8 * binary + 0.2 * partial)
        if log_metric is not None and rewards:
            n = len(rewards)
            log_metric(
                "correctness_reward_zero_ratio",
                sum(1 for r in rewards if r == 0.0) / n,
            )
            fmt = [1.0 if has_boxed(_text(c)) else 0.0 for c in completions]
            combined = [
                REWARD_WEIGHTS[0] * r + REWARD_WEIGHTS[1] * f
                for r, f in zip(rewards, fmt)
            ]
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

    def anti_hedge_reward(completions, log_metric=None, **kwargs):
        rewards = []
        for comp in completions:
            text = _text(comp).lower()
            hedging = any(phrase in text for phrase in HEDGING_PHRASES)
            rewards.append(-0.5 if hedging else 0.0)
        if log_metric is not None:
            n_hedging = sum(1 for r in rewards if r < 0)
            log_metric("hedge_ratio", n_hedging / len(rewards) if rewards else 0.0)
            step(f"  anti_hedge: {n_hedging}/{len(rewards)} hedging responses this batch")
        return rewards

    # ADDED: thinking length reward — prevents empty <think> shortcut
    # Model was learning to skip reasoning entirely and just guess
    # This reward penalizes short/empty <think> blocks and rewards proper reasoning
    def thinking_length_reward(completions, log_metric=None, **kwargs):
        rewards = []
        short_count = 0
        for comp in completions:
            text = _text(comp)
            think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
            if think_match:
                think_len = len(think_match.group(1).strip())
                if think_len < MIN_THINK_LEN:
                    # empty or near-empty <think> — model is skipping reasoning
                    rewards.append(-0.5)
                    short_count += 1
                elif think_len < GOOD_THINK_LEN:
                    # some reasoning but not much — neutral
                    rewards.append(0.0)
                else:
                    # proper reasoning chain — small bonus
                    rewards.append(0.2)
            else:
                # no <think> block at all — penalize
                rewards.append(-0.3)
                short_count += 1
        if log_metric is not None:
            log_metric("short_think_ratio", short_count / len(rewards) if rewards else 0.0)
            step(f"  thinking_length: {short_count}/{len(rewards)} short/empty think blocks")
        return rewards

    return correctness_reward, format_reward, anti_hedge_reward, thinking_length_reward


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model",            default=MODEL_ID)
    p.add_argument("--init-adapter",     default=None)
    p.add_argument("--data",             default=DATA_PATH)
    p.add_argument("--output",           default=OUTPUT_DIR)
    p.add_argument("--epochs",           type=int,   default=NUM_EPOCHS)
    p.add_argument("--lr",               type=float, default=LR)
    p.add_argument("--beta",             type=float, default=BETA)
    p.add_argument("--num-generations",  type=int,   default=NUM_GENERATIONS)
    p.add_argument("--max-prompt",       type=int,   default=MAX_PROMPT_LEN)
    p.add_argument("--max-completion",   type=int,   default=MAX_COMPLETION)
    p.add_argument("--temperature",      type=float, default=TEMPERATURE)
    p.add_argument("--vllm-mem",         type=float, default=VLLM_MEM)
    p.add_argument("--per-device-bs",    type=int,   default=PER_DEVICE_BS)
    p.add_argument("--grad-accum",       type=int,   default=GRAD_ACCUM)
    p.add_argument("--n-samples",        type=int,   default=None)
    p.add_argument("--save-every-steps", type=int,   default=SAVE_EVERY_STEPS)
    p.add_argument("--no-resume",        action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.init_adapter and not Path(args.init_adapter).exists():
        sys.exit(f"--init-adapter path does not exist: {args.init_adapter}")

    banner("STEP 1 / 6  CUDA sanity check (bf16, L40 48GB profile)")
    import torch
    cap = assert_modern_gpu()
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
        sys.exit("trl not installed. Run: pip install 'trl>=0.19.0'")
    step("imports done.")

    class GrpoTrainingLogCallback(TrainerCallback):
        def __init__(self, save_every_steps: int):
            self.save_every_steps = save_every_steps
            self._pending_save_step: Optional[int] = None

        def _list_checkpoint_dirs(self, output_dir: str) -> list[str]:
            if not os.path.isdir(output_dir):
                return []
            names = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            return sorted(names, key=lambda n: int(n.rsplit("-", 1)[-1]))

        def on_train_begin(self, args, state, control, **kwargs):
            step(f"checkpoint policy: every {self.save_every_steps} steps")
            ckpts = self._list_checkpoint_dirs(args.output_dir)
            step(f"existing checkpoints: {', '.join(ckpts) if ckpts else '(none)'}")
            return control

        def on_step_end(self, args, state, control, **kwargs):
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
            else:
                warn(f"checkpoint directory missing: {ckpt_path}")
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
            z_combined   = self._fmt_ratio(logs, "reward_zero_ratio")
            z_corr       = self._fmt_ratio(logs, "correctness_reward_zero_ratio")
            z_fmt        = self._fmt_ratio(logs, "format_reward_zero_ratio")
            hedge        = self._fmt_ratio(logs, "hedge_ratio")
            short_think  = self._fmt_ratio(logs, "short_think_ratio")
            if any(v is not None for v in [z_combined, z_corr, z_fmt, hedge, short_think]):
                parts = []
                if z_combined  is not None: parts.append(f"combined=0: {z_combined:.1%}")
                if z_corr      is not None: parts.append(f"correct=0: {z_corr:.1%}")
                if z_fmt       is not None: parts.append(f"no boxed: {z_fmt:.1%}")
                if hedge       is not None: parts.append(f"hedging: {hedge:.1%}")
                if short_think is not None: parts.append(f"short think: {short_think:.1%}")
                step(f"step {state.global_step} zero-reward ratios — {', '.join(parts)}")

            issues = []
            reward = logs.get("reward")
            if reward is not None:
                try:
                    if reward != reward: issues.append("reward is NaN")
                    elif reward == 0.0:  issues.append("mean reward=0")
                except TypeError:
                    pass
            if z_combined is not None and z_combined >= 0.9:
                issues.append(f"reward_zero_ratio={z_combined:.1%} (almost all zero)")
            clipped = logs.get("completions/clipped_ratio")
            if clipped is not None:
                try:
                    if float(clipped) >= 0.9:
                        issues.append(f"clipped_ratio={float(clipped):.3f} — increase --max-completion")
                except (TypeError, ValueError):
                    pass
            grad_norm = logs.get("grad_norm")
            if grad_norm is not None:
                try:
                    gn = float(grad_norm)
                    if gn == 0.0:    issues.append("grad_norm=0")
                    elif gn > 10.0:  issues.append(f"grad_norm={gn:.3f} (large)")
                except (TypeError, ValueError):
                    pass
            if issues:
                warn(f"step {state.global_step}: " + "; ".join(issues))
            return control

    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)
    correctness_reward, format_reward, anti_hedge_reward, thinking_length_reward = make_reward_fns(judger)
    step("judger + reward functions ready (correctness + format + anti-hedge + thinking-length).")

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
            {"role": "user",   "content": user},
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
            "gold":   json.dumps(gold),
            "is_mcq": bool(item.get("options")),
        })
    dataset = Dataset.from_list(records).shuffle(seed=SEED)
    n_mcq = sum(r["is_mcq"] for r in records)
    prompt_lens.sort()
    step(f"prompt tokens: min={prompt_lens[0]} median={prompt_lens[len(prompt_lens)//2]} max={prompt_lens[-1]}")
    step(f"built {len(dataset)} prompts ({n_mcq} MCQ); dropped {n_dropped} overlong.")

    banner("STEP 5 / 6  Configure GRPO")
    global_batch = args.per_device_bs * args.grad_accum
    if global_batch % args.num_generations != 0:
        sys.exit(f"per-device-bs*grad-accum ({global_batch}) must be divisible by num-generations ({args.num_generations}).")
    step(f"global batch {global_batch} -> {global_batch // args.num_generations} prompt(s)/step")

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
        save_strategy="no",
        save_steps=args.save_every_steps,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[correctness_reward, format_reward, anti_hedge_reward, thinking_length_reward],
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
            step(f"latest checkpoints: {', '.join(sorted(ckpts)[-3:])}")
        else:
            warn("no checkpoint-* directories found")
        raise
    step(f"training done in {(time.time() - t0) / 60:.1f} min.")

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    banner(f"DONE  —  test_inference.py --adapter {args.output}")


if __name__ == "__main__":
    main()
