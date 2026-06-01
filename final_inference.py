"""
final_inference.py — single-entry-point inference for the private test set.

Exposes ONE function, run_inference(), that does the whole pipeline end-to-end:
  1. Loads the fine-tuned model FROM THE HUGGINGFACE HUB (auto-detects whether the
     repo is a LoRA adapter or a full merged model and loads it the right way).
  2. Runs inference on the private dataset.
  3. Applies post-processing: self-consistency majority vote over N samples, keeping
     the FULL winning trace (reasoning + final \\boxed{}), nothing stripped.
  4. Writes the final submission CSV (id,response).

Nothing manual or external — `python final_inference.py` (or calling
run_inference() from Python) produces the final CSV by itself.

────────────────────────────────────────────────────────────────────────────
CONFIG — change HF_HUB_MODEL to your pushed repo (or pass --hf-model).
You pushed your model with, e.g.:
    from huggingface_hub import login
    login()
    model.push_to_hub("your-username/your-model-name")
Then set HF_HUB_MODEL below to "your-username/your-model-name".
────────────────────────────────────────────────────────────────────────────

Usage:
    python final_inference.py                                  # uses config defaults
    python final_inference.py --hf-model user/my-model         # override hub repo
    python final_inference.py --data data/private.jsonl --output submission.csv

If the Hub repo is PRIVATE, authenticate first (any one of):
    huggingface-cli login          # caches a token, OR
    export HF_TOKEN=hf_xxx         # env var, OR
    python final_inference.py --hf-token hf_xxx
"""

import os
import sys
import re
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ── CONFIG ───────────────────────────────────────────────────────────────────
# CHANGE THIS to the HuggingFace Hub repo you pushed your fine-tuned model to:
HF_HUB_MODEL      = "your-username/your-model-name"
# Base model the LoRA was trained on (only used if the Hub repo is a LoRA adapter):
BASE_MODEL_ID     = "Qwen/Qwen3-4B-Thinking-2507"
PRIVATE_DATA_PATH = "data/private.jsonl"
OUTPUT_CSV        = "submission.csv"

MAX_TOKENS        = 32000   # cap on generated tokens per sample
N_VOTES           = 3       # self-consistency: samples per question for majority vote
TEMPERATURE       = 0.8
TOP_P             = 0.95
TOP_K             = 20
MAX_LORA_RANK     = 32      # >= the rank your adapter was trained with
GPU_MEM_UTIL      = 0.90
MAX_MODEL_LEN     = 24000


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    print("=" * 70, flush=True)


def step(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── System prompts (identical to test_inference_formula_old.py) ──────────────
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


def extract_boxed(text: str) -> str:
    """Last \\boxed{...} contents, else the whole (stripped) text."""
    m = re.findall(r"\\boxed\{(.+?)\}", text)
    return m[-1].strip() if m else text.strip()


def majority_vote(completions: list) -> str:
    """
    Self-consistency post-processing: vote on the boxed answer across the N
    samples, then return the FULL text of a completion that produced the
    winning answer — reasoning/<think> intact, nothing stripped.
    """
    from collections import Counter
    answers = [extract_boxed(c) for c in completions]
    winner = Counter(answers).most_common(1)[0][0]
    for comp, ans in zip(completions, answers):
        if ans == winner:
            return comp
    return completions[0]


def _resolve_model_source(hf_model: str, hf_token: Optional[str]):
    """
    Download the Hub repo and decide how to load it.

    Returns (is_adapter, local_dir):
      - is_adapter True  -> repo is a LoRA adapter (has adapter_config.json);
                            load BASE_MODEL_ID in vLLM with enable_lora + LoRARequest(local_dir).
      - is_adapter False -> repo is a full model; load local_dir directly in vLLM.
    """
    from huggingface_hub import snapshot_download, login
    if hf_token:
        login(token=hf_token)
    step(f"downloading model repo from HF Hub: {hf_model} ...")
    local_dir = snapshot_download(repo_id=hf_model)
    is_adapter = os.path.isfile(os.path.join(local_dir, "adapter_config.json"))
    kind = "LoRA adapter" if is_adapter else "full model"
    step(f"resolved {hf_model} -> {local_dir}  ({kind})")
    return is_adapter, local_dir


def run_inference(
    hf_model: str = HF_HUB_MODEL,
    data_path: str = PRIVATE_DATA_PATH,
    output_csv: str = OUTPUT_CSV,
    base_model: str = BASE_MODEL_ID,
    n_samples: Optional[int] = None,
    n_votes: int = N_VOTES,
    max_tokens: int = MAX_TOKENS,
    max_lora_rank: int = MAX_LORA_RANK,
    hf_token: Optional[str] = None,
) -> str:
    """
    Full end-to-end pipeline. Loads the model from the HF Hub, runs inference on
    the private dataset, applies majority-vote post-processing, and writes the
    submission CSV (id,response). Returns the CSV path.

    Callable with no arguments — uses the CONFIG defaults above.
    """
    if hf_model == "your-username/your-model-name":
        sys.exit("Set HF_HUB_MODEL in final_inference.py (or pass --hf-model / hf_model=...) "
                 "to your pushed HuggingFace repo.")
    if not Path(data_path).exists():
        sys.exit(f"Private dataset not found: {data_path}")

    # ── 1. CUDA ──────────────────────────────────────────────────────────────
    banner("STEP 1 / 6  CUDA sanity check")
    import torch
    step(f"torch {torch.__version__} cuda {torch.version.cuda}")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available.")
    step(f"GPU 0: {torch.cuda.get_device_name(0)}")

    # ── 2. Imports ───────────────────────────────────────────────────────────
    banner("STEP 2 / 6  Imports")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    step("imports done.")

    # ── 3. Resolve + load model from the Hub ─────────────────────────────────
    banner("STEP 3 / 6  Load model from HuggingFace Hub")
    is_adapter, local_dir = _resolve_model_source(hf_model, hf_token)

    # Tokenizer: from base for an adapter, from the repo itself for a full model.
    tok_source = base_model if is_adapter else local_dir
    step(f"loading tokenizer from {tok_source} ...")
    tokenizer = AutoTokenizer.from_pretrained(tok_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t0 = time.time()
    if is_adapter:
        step(f"constructing vLLM (base={base_model}, enable_lora=True) ...")
        llm = LLM(
            model=base_model,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            enforce_eager=False,
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=16,
            max_num_batched_tokens=16384,
        )
        lora_request = LoRARequest("submission-lora", 1, local_dir)
    else:
        step("constructing vLLM (full merged model) ...")
        llm = LLM(
            model=local_dir,
            enforce_eager=False,
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=16,
            max_num_batched_tokens=16384,
        )
        lora_request = None
    step(f"model ready in {time.time() - t0:.1f}s.")

    sampling_params = SamplingParams(
        n=n_votes,
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    # ── 4. Load private dataset + build prompts ──────────────────────────────
    banner("STEP 4 / 6  Load private dataset + build prompts")
    data = [json.loads(line) for line in open(data_path) if line.strip()]
    subset = data[:n_samples] if n_samples else data
    n_mcq = sum(bool(d.get("options")) for d in subset)
    step(f"loaded {len(subset)} questions  ({n_mcq} MCQ, {len(subset) - n_mcq} free-form)")

    prompts = []
    for item in subset:
        system, user = build_prompt(item["question"], item.get("options"))
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False, add_generation_prompt=True,
        ))
    step(f"built {len(prompts)} prompts (n={n_votes} samples each).")

    # ── 5. Generate + post-process (majority vote, full trace kept) ──────────
    banner("STEP 5 / 6  Generate + majority-vote post-processing")
    t0 = time.time()
    gen_kwargs = {"sampling_params": sampling_params}
    if lora_request is not None:
        gen_kwargs["lora_request"] = lora_request
    outputs = llm.generate(prompts, **gen_kwargs)
    step(f"generation done in {time.time() - t0:.1f}s ({len(outputs)} outputs).")

    # vLLM returns outputs in input order -> ids stay aligned with `subset`.
    responses = [majority_vote([o.text.strip() for o in out.outputs]) for out in outputs]
    for i in range(min(3, len(responses))):
        step(f"── id={subset[i].get('id')}  final box={extract_boxed(responses[i])!r}  "
             f"(len {len(responses[i])} chars) ──")

    # ── 6. Write submission CSV (id,response) ────────────────────────────────
    banner("STEP 6 / 6  Write submission CSV")
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)                 # QUOTE_MINIMAL — matches the sample CSV
        writer.writerow(["id", "response"])
        for item, response in zip(subset, responses):
            writer.writerow([item.get("id"), response])
    step(f"wrote {len(subset)} rows -> {out_path}")
    banner(f"DONE  —  submission CSV: {out_path}")
    return str(out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-entry-point private-set inference -> submission CSV.")
    p.add_argument("--hf-model",      default=HF_HUB_MODEL,
                   help=f"HuggingFace Hub repo of your fine-tuned model (default: {HF_HUB_MODEL}).")
    p.add_argument("--base-model",    default=BASE_MODEL_ID,
                   help="Base model id (used only if the Hub repo is a LoRA adapter).")
    p.add_argument("--data",          default=PRIVATE_DATA_PATH, help="Private dataset jsonl.")
    p.add_argument("--output",        default=OUTPUT_CSV,        help="Output submission CSV.")
    p.add_argument("--n-samples",     type=int, default=None,    help="Limit to first N items (debug).")
    p.add_argument("--n-votes",       type=int, default=N_VOTES, help="Samples per question for majority vote.")
    p.add_argument("--max-tokens",    type=int, default=MAX_TOKENS, help="Max generated tokens per sample.")
    p.add_argument("--max-lora-rank", type=int, default=MAX_LORA_RANK, help="Max LoRA rank slots (>= adapter rank).")
    p.add_argument("--hf-token",      default=None, help="HF token if the repo is private (else use huggingface-cli login / HF_TOKEN).")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run_inference(
        hf_model=a.hf_model,
        data_path=a.data,
        output_csv=a.output,
        base_model=a.base_model,
        n_samples=a.n_samples,
        n_votes=a.n_votes,
        max_tokens=a.max_tokens,
        max_lora_rank=a.max_lora_rank,
        hf_token=a.hf_token,
    )
