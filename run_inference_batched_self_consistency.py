"""
CSE 151B Competition — converted from starter_code_cse151b_comp (2).ipynb.

ONE-TIME setup (run in a shell, not in this script):
    wget -qO- https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv venv .venv --seed
    .venv/bin/python -m pip install sympy "numpy<2" transformers vllm tqdm \
        bitsandbytes antlr4-python3-runtime==4.11.1

To run this file:
    .venv/bin/python run_inference.py
or:
    source .venv/bin/activate && python run_inference.py
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from typing import Optional
from collections import Counter

# Must be set BEFORE vllm is imported (and before any subprocess re-imports this file).
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH   = "data/public.jsonl"
OUTPUT_PATH = "results/starter_results.jsonl"
MAX_TOKENS  = 512
N_SAMPLES   = None       # None = run the full dataset
SELF_CONSISTENCY_SAMPLES = 3  # number of completions per question for majority vote
SAVE_EVAL   = True       # False when running on the private test set


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    print("=" * 70, flush=True)


def step(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the content inside the last \boxed{} in a model response."""
    matches = re.findall(r"\\boxed\{(.+?)\}", text)
    return matches[-1].strip() if matches else None


def majority_voted_response(vllm_request_output) -> str:
    """
    Given one vLLM RequestOutput with multiple sampled completions,
    return the majority-voted boxed answer. Falls back to the first raw output
    if no completion contains a boxed answer.
    """
    boxed_answers = []

    for completion in vllm_request_output.outputs:
        ans = extract_boxed_answer(completion.text)
        if ans:
            boxed_answers.append(ans)

    if not boxed_answers:
        return vllm_request_output.outputs[0].text.strip()

    final_answer, _ = Counter(boxed_answers).most_common(1)[0]
    return f"\\boxed{{{final_answer}}}"


def main() -> None:
    # ── 1. CUDA sanity check ─────────────────────────────────────────────────
    banner("STEP 1 / 7  CUDA sanity check")
    import torch
    step(f"torch {torch.__version__} (built for CUDA {torch.version.cuda})")
    step(f"cuda available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available — fix the environment before continuing.")
    step(f"GPU 0: {torch.cuda.get_device_name(0)}")

    # ── 2. Imports ───────────────────────────────────────────────────────────
    banner("STEP 2 / 7  Imports (transformers + vllm)")
    step("importing transformers ...")
    from transformers import AutoTokenizer
    step("importing vllm ...")
    from vllm import LLM, SamplingParams
    step("importing tqdm ...")
    from tqdm import tqdm
    step("imports done.")

    # ── 3. Load dataset ──────────────────────────────────────────────────────
    banner("STEP 3 / 7  Load dataset")
    step(f"reading {DATA_PATH} ...")
    data = [json.loads(line) for line in open(DATA_PATH)]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options")   for d in data)
    step(f"loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    mcq_sample  = next(d for d in data if d.get("options"))
    free_sample = next(d for d in data if not d.get("options"))
    step("MCQ sample (truncated):")
    print("    ", json.dumps(mcq_sample, indent=2)[:400].replace("\n", "\n    "))
    step("Free-form sample (truncated):")
    print("    ", json.dumps(free_sample, indent=2)[:400].replace("\n", "\n    "))

    # ── 4. Prompt construction ───────────────────────────────────────────────
    banner("STEP 4 / 7  Prompt construction")
    for label, item in [("MCQ", mcq_sample), ("Free-form", free_sample)]:
        sys_p, usr_p = build_prompt(item["question"], item.get("options"))
        step(f"{label} user prompt (first 200 chars): {usr_p[:200]!r}")

    # ── 5. Load model with vLLM ──────────────────────────────────────────────
    banner("STEP 5 / 7  Load model with vLLM (this can take ~30-60s)")
    step(f"loading tokenizer for {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    step("tokenizer loaded.")

    step("constructing LLM (bitsandbytes int8, eager mode, Triton attention) ...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enforce_eager=True,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        dtype="float16",
    )
    step(f"LLM ready in {time.time() - t0:.1f}s.")

    sampling_params = SamplingParams(
        n=SELF_CONSISTENCY_SAMPLES,
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )
    step(f"sampling params set (self-consistency n={SELF_CONSISTENCY_SAMPLES}).")

    # ── 6. Generate responses ────────────────────────────────────────────────
    subset = data[:N_SAMPLES] if N_SAMPLES else data
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

    step("calling llm.generate (vLLM will print its own progress bar) ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    step(f"generation done in {time.time() - t0:.1f}s ({len(outputs)} outputs).")

    responses = [majority_voted_response(out) for out in outputs]
    step("self-consistency majority vote complete.")
    for i in range(min(3, len(responses))):
        step(f"── Response {i} (id={subset[i].get('id')}, len={len(responses[i])} chars) ──")
        print(responses[i][:400], "..." if len(responses[i]) > 400 else "")

    # ── 7. Score responses ───────────────────────────────────────────────────
    banner("STEP 7 / 7  Score responses")
    step("loading judger ...")
    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)
    step("judger ready.")

    results = []
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

    # ── Summary ──────────────────────────────────────────────────────────────
    mcq_res  = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]

    def acc(subset_):
        return sum(r["correct"] for r in subset_) / len(subset_) * 100 if subset_ else 0.0

    banner("EVALUATION RESULTS")
    print(f"  MCQ        : {sum(r['correct'] for r in mcq_res):4d} / {len(mcq_res):4d}  ({acc(mcq_res):.2f}%)")
    print(f"  Free-form  : {sum(r['correct'] for r in free_res):4d} / {len(free_res):4d}  ({acc(free_res):.2f}%)")
    print(f"  Overall    : {sum(r['correct'] for r in results):4d} / {len(results):4d}  ({acc(results):.2f}%)")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    step(f"writing results to {out_path} (SAVE_EVAL={SAVE_EVAL}) ...")
    with open(out_path, "w") as f:
        for r in results:
            if SAVE_EVAL:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "gold": r["gold"],
                          "response": r["response"], "correct": r["correct"]}
            else:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "response": r["response"]}
            f.write(json.dumps(record) + "\n")
    step(f"saved {len(results)} records.")
    banner("DONE")


if __name__ == "__main__":
    main()