"""
cluster_questions.py
--------------------
Reads public.jsonl, embeds each question with SBERT, clusters into 8 groups
via KMeans, then selects the 8 questions closest to each cluster centroid
(64 total) and writes them to public_shorten.jsonl.

Usage:
    python cluster_questions.py [--input public.jsonl] [--output public_shorten.jsonl]
                                [--clusters 8] [--per-cluster 8]

Dependencies (install once):
    pip install sentence-transformers scikit-learn numpy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [warn] skipping line {lineno}: {exc}", file=sys.stderr)
    return records


def detect_question_key(records: list[dict]) -> str:
    """Return the first key that looks like it holds the question text."""
    candidates = ["question", "text", "prompt", "input", "content", "query"]
    if not records:
        raise ValueError("JSONL file is empty.")
    sample = records[0]
    for key in candidates:
        if key in sample:
            return key
    # fall back to the first string-valued key
    for key, val in sample.items():
        if isinstance(val, str):
            print(f"  [info] auto-detected question key: '{key}'", file=sys.stderr)
            return key
    raise KeyError(
        f"Cannot find a string field in record: {list(sample.keys())}. "
        "Pass --question-key <key> to specify it."
    )


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"  Loading SBERT model '{model_name}' …")
    model = SentenceTransformer(model_name)
    print(f"  Encoding {len(texts)} texts …")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity ≡ dot product
    )
    return embeddings  # shape: (N, D)


def kmeans_cluster(embeddings: np.ndarray, n_clusters: int, seed: int = 42):
    from sklearn.cluster import KMeans

    print(f"  Running KMeans with {n_clusters} clusters …")
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = km.fit_predict(embeddings)
    centroids = km.cluster_centers_       # shape: (K, D)
    return labels, centroids


def select_center_questions(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    n_clusters: int,
    per_cluster: int,
) -> list[int]:
    """
    For each cluster return the indices of the `per_cluster` questions
    whose embeddings are closest (L2) to that cluster's centroid.
    """
    selected = []
    for k in range(n_clusters):
        mask = np.where(labels == k)[0]          # indices in cluster k
        if len(mask) == 0:
            print(f"  [warn] cluster {k} is empty – skipping", file=sys.stderr)
            continue
        cluster_embs = embeddings[mask]           # (Nk, D)
        centroid = centroids[k]                   # (D,)
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        top_k = per_cluster if per_cluster <= len(mask) else len(mask)
        nearest = np.argsort(dists)[:top_k]
        selected.extend(mask[nearest].tolist())
    return selected


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster JSONL questions with SBERT + KMeans.")
    parser.add_argument("--input",        default="data/public.jsonl",         help="Input JSONL file")
    parser.add_argument("--output",       default="public_shorten.jsonl", help="Output JSONL file")
    parser.add_argument("--clusters",     type=int, default=8,            help="Number of KMeans clusters (default: 8)")
    parser.add_argument("--per-cluster",  type=int, default=8,            help="Questions to keep per cluster (default: 8)")
    parser.add_argument("--model",        default="all-MiniLM-L6-v2",     help="SBERT model name (default: all-MiniLM-L6-v2)")
    parser.add_argument("--question-key", default=None,                   help="JSON key holding the question text (auto-detected if omitted)")
    parser.add_argument("--seed",         type=int, default=42,           help="Random seed for KMeans (default: 42)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    # 1. Load -----------------------------------------------------------------
    print(f"\n[1/5] Loading '{input_path}' …")
    if not input_path.exists():
        sys.exit(f"ERROR: '{input_path}' not found.")
    records = load_jsonl(input_path)
    print(f"      Loaded {len(records)} records.")

    if len(records) < args.clusters * args.per_cluster:
        print(
            f"  [warn] Only {len(records)} records; the output may have fewer than "
            f"{args.clusters * args.per_cluster} questions.",
            file=sys.stderr,
        )

    # 2. Extract question texts -----------------------------------------------
    print("\n[2/5] Extracting question texts …")
    q_key = args.question_key or detect_question_key(records)
    print(f"      Using key: '{q_key}'")
    texts = [str(rec.get(q_key, "")) for rec in records]

    # 3. Embed ----------------------------------------------------------------
    print("\n[3/5] Embedding with SBERT …")
    embeddings = embed_texts(texts, args.model)
    print(f"      Embedding matrix: {embeddings.shape}")

    # 4. Cluster --------------------------------------------------------------
    print("\n[4/5] Clustering …")
    labels, centroids = kmeans_cluster(embeddings, args.clusters, args.seed)
    unique, counts = np.unique(labels, return_counts=True)
    for k, c in zip(unique, counts):
        print(f"      Cluster {k}: {c} questions")

    # 5. Select & save --------------------------------------------------------
    print(f"\n[5/5] Selecting {args.per_cluster} center question(s) per cluster …")
    selected_idx = select_center_questions(
        embeddings, labels, centroids, args.clusters, args.per_cluster
    )
    # Annotate each record with its cluster label for reference
    output_records = []
    for idx in sorted(set(selected_idx)):   # deduplicate & preserve order
        rec = dict(records[idx])
        rec["_cluster"] = int(labels[idx])
        output_records.append(rec)

    # Sort by cluster so the output file is human-readable
    output_records.sort(key=lambda r: r["_cluster"])

    write_jsonl(output_records, output_path)
    print(f"\n✓  Saved {len(output_records)} questions → '{output_path}'")
    print(f"   (across {args.clusters} clusters, up to {args.per_cluster} each)\n")


if __name__ == "__main__":
    main()