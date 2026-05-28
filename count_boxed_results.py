"""
Count how many test_inference.py responses contain a non-empty \\boxed{...}.

Usage:
    .venv/bin/python count_boxed_results.py results/grpo_blackwell_results.jsonl
"""

import json
import re
import sys
from pathlib import Path


def has_boxed(text: str) -> bool:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    return bool(m and m.group(1).strip())


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python count_boxed_results.py <results.jsonl>")
    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f"File not found: {path}")

    rows = [json.loads(line) for line in open(path)]
    if not rows or "response" not in rows[0]:
        sys.exit("Expected jsonl rows with a 'response' field (test_inference.py output).")

    boxed = [r for r in rows if has_boxed(r["response"])]
    n = len(rows)
    k = len(boxed)

    print(f"\nResults file: {path}")
    print(f"  With \\boxed{{}} : {k:4d} / {n:4d}  ({100 * k / n:.2f}%)")
    print(f"  Missing/empty   : {n - k:4d} / {n:4d}  ({100 * (n - k) / n:.2f}%)")

    if "is_mcq" in rows[0]:
        for label, subset in [("MCQ", [r for r in rows if r.get("is_mcq")]),
                              ("Free-form", [r for r in rows if not r.get("is_mcq")])]:
            if not subset:
                print(f"  {label:10s}: N/A")
                continue
            b = sum(1 for r in subset if has_boxed(r["response"]))
            print(f"  {label:10s}: {b:4d} / {len(subset):4d}  ({100 * b / len(subset):.2f}%)")


if __name__ == "__main__":
    main()
