"""
Re-score a results jsonl file the same way run_inference.py does.

Expects each line to have at least: is_mcq, gold, response.
(The 'correct' field, if present, is ignored — we recompute it.)

Usage:
    .venv/bin/python score_results.py results/lora_results.jsonl
"""

import json
import re
import sys
from pathlib import Path


def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == str(gold_letter).strip().upper()


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python score_results.py <results.jsonl>")
    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f"File not found: {path}")

    sys.path.insert(0, ".")
    from judger import Judger
    judger = Judger(strict_extract=False)

    rows = [json.loads(line) for line in open(path)]
    mcq_correct = mcq_total = free_correct = free_total = 0

    for r in rows:
        if "gold" not in r:
            sys.exit(f"Row {r.get('id')} has no 'gold' field — was this written with --no-eval?")
        is_mcq   = bool(r.get("is_mcq"))
        gold     = r["gold"]
        response = r["response"]

        if is_mcq:
            ok = score_mcq(response, str(gold))
            mcq_correct += ok
            mcq_total   += 1
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                ok = judger.auto_judge(
                    pred=response,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            except Exception:
                ok = False
            free_correct += ok
            free_total   += 1

    def pct(c, t):
        return f"{c:4d} / {t:4d}  ({100 * c / t:.2f}%)" if t else "N/A"

    total_correct = mcq_correct + free_correct
    total         = mcq_total + free_total

    print(f"\nResults file: {path}")
    print(f"  MCQ        : {pct(mcq_correct, mcq_total)}")
    print(f"  Free-form  : {pct(free_correct, free_total)}")
    print(f"  Overall    : {pct(total_correct, total)}")


if __name__ == "__main__":
    main()

