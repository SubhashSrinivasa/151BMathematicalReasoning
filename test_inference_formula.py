"""
Inference with a trained LoRA adapter on top of the base Qwen3-4B-Thinking.

Uses vLLM's runtime LoRA support (no merging — adapter stays as a separate
~50 MB file, base model is loaded once).

Usage:
    .venv/bin/python test_inference.py --adapter checkpoints/qwen3-4b-lora

Required:
    --adapter PATH    LoRA adapter directory produced by train_lora.py

Optional toggles (with defaults):
    --model PATH      base model id  (default: Qwen/Qwen3-4B-Thinking-2507)
    --data PATH       dataset jsonl  (default: data/public.jsonl)
    --output PATH     results jsonl  (default: results/lora_results.jsonl)
    --n-samples N     limit to first N items (default: all)
    --no-eval         write {id, is_mcq, response} only (private test set)
    --max-lora-rank   max rank vLLM should allocate slots for (default: 32)
"""

import os
import sys
import json
import re
import time
import argparse
from pathlib import Path
from typing import Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_DATA_PATH   = "data/private.jsonl"
DEFAULT_OUTPUT_PATH = "results/lora_results.jsonl"
MAX_TOKENS          = 32000


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    print("=" * 70, flush=True)


def step(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with a LoRA adapter.")
    p.add_argument("--adapter",        required=True,                help="LoRA adapter directory (REQUIRED).")
    p.add_argument("--model",          default=DEFAULT_MODEL_ID,     help="Base model id or path.")
    p.add_argument("--data",           default=DEFAULT_DATA_PATH,    help="Input jsonl.")
    p.add_argument("--output",         default=DEFAULT_OUTPUT_PATH,  help="Output jsonl.")
    p.add_argument("--n-samples",      type=int, default=None,       help="Limit to first N items (default: all).")
    p.add_argument("--no-eval",        action="store_true",          help="Skip scoring; private-test format.")
    p.add_argument("--max-lora-rank",  type=int, default=32,         help="Max LoRA rank slots (>= adapter rank).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.adapter).exists():
        sys.exit(f"Adapter path does not exist: {args.adapter}")

    # ── 1. CUDA ──────────────────────────────────────────────────────────────
    banner("STEP 1 / 7  CUDA sanity check")
    import torch
    step(f"torch {torch.__version__} cuda {torch.version.cuda}")
    step(f"cuda available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available.")
    step(f"GPU 0: {torch.cuda.get_device_name(0)}")

    # ── 2. Imports ───────────────────────────────────────────────────────────
    banner("STEP 2 / 7  Imports")
    step("importing transformers ...")
    from transformers import AutoTokenizer
    step("importing vllm ...")
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    step("importing tqdm ...")
    from tqdm import tqdm
    step("imports done.")

    # ── 3. Load dataset ──────────────────────────────────────────────────────
    banner("STEP 3 / 7  Load dataset")
    step(f"reading {args.data} ...")
    data = [json.loads(line) for line in open(args.data)]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options")   for d in data)
    step(f"loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    # ── 4. Prompt construction (sanity preview) ─────────────────────────────
    banner("STEP 4 / 7  Prompt construction")
    mcq_sample  = next((d for d in data if d.get("options")), None)
    free_sample = next((d for d in data if not d.get("options")), None)
    for label, item in [("MCQ", mcq_sample), ("Free-form", free_sample)]:
        if item is None:
            continue
        sys_p, usr_p = build_prompt(item["question"], item.get("options"))
        step(f"{label} user prompt (first 200 chars): {usr_p[:200]!r}")

    # ── 5. Load base + adapter ───────────────────────────────────────────────
    banner("STEP 5 / 7  Load base model + LoRA adapter")
    step(f"base model: {args.model}")
    step(f"adapter:    {args.adapter}  (max_lora_rank={args.max_lora_rank})")
    step("loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    step("tokenizer loaded.")

    step("constructing LLM with enable_lora=True ...")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        enforce_eager=False,
        gpu_memory_utilization=0.90,
        max_model_len=24000,
        max_num_seqs=16,
        max_num_batched_tokens=16384,
    )
    step(f"LLM ready in {time.time() - t0:.1f}s.")
    lora_request = LoRARequest("trained-lora", 1, args.adapter)

    sampling_params = SamplingParams(
        n=3,
        max_tokens=MAX_TOKENS,
        temperature=0.8,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )
    step("sampling params set.")

    # ── 6. Generate ──────────────────────────────────────────────────────────
    subset = data[:args.n_samples] if args.n_samples else data
    banner(f"STEP 6 / 7  Generate responses for {len(subset)} question(s)")
    step("building chat-template prompts ...")
    prompts = []
    for item in subset:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)
    step(f"built {len(prompts)} prompts.")

    step("calling llm.generate WITH LoRA adapter ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_request)
    step(f"generation done in {time.time() - t0:.1f}s ({len(outputs)} outputs).")

    def extract_boxed(text):
        m = re.findall(r"\\boxed\{(.+?)\}", text)
        return m[-1].strip() if m else text.strip()

    def majority_vote(completions):
        from collections import Counter
        answers = [extract_boxed(c) for c in completions]
        winner = Counter(answers).most_common(1)[0][0]
        return f"\\boxed{{{winner}}}"

    responses = [
        majority_vote([o.text.strip() for o in out.outputs])
        for out in outputs
    ]

    for i in range(min(3, len(responses))):
        step(f"── Response {i} (id={subset[i].get('id')}, len={len(responses[i])} chars) ──")
        print(responses[i][:400], "..." if len(responses[i]) > 400 else "")

    # ── 7. Score ─────────────────────────────────────────────────────────────
    banner("STEP 7 / 7  Score responses" + ("  [SKIPPED: --no-eval]" if args.no_eval else ""))
    results = []
    if args.no_eval:
        for item, response in zip(subset, responses):
            results.append({
                "id":       item.get("id"),
                "is_mcq":   bool(item.get("options")),
                "response": response,
            })
    else:
        step("loading judger ...")
        sys.path.insert(0, ".")
        from judger import Judger
        judger = Judger(strict_extract=False)
        step("judger ready.")

        for item, response in tqdm(zip(subset, responses), total=len(subset), desc="Scoring"):
            is_mcq = bool(item.get("options"))
            gold   = item["answer"]
            if is_mcq:
                correct = score_mcq(response, str(gold))
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
                "id":       item.get("id"),
                "is_mcq":   is_mcq,
                "gold":     gold,
                "response": response,
                "correct":  correct,
            })
        step(f"scored {len(results)} responses.")

        mcq_res  = [r for r in results if r["is_mcq"]]
        free_res = [r for r in results if not r["is_mcq"]]
        def acc(s):
            return sum(r["correct"] for r in s) / len(s) * 100 if s else 0.0
        banner("EVALUATION RESULTS (LoRA)")
        print(f"  Adapter    : {args.adapter}")
        print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
        print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
        print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    step(f"writing results to {out_path} ...")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    step(f"saved {len(results)} records.")
    banner("DONE")


if __name__ == "__main__":
    main()

