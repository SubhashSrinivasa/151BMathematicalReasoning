"""
select_512.py
─────────────
Selects 512 representative questions from public.jsonl using
cosine-similarity-based clustering (KMeans on SBERT embeddings).

Strategy:
  1. Embed all questions with SBERT (all-MiniLM-L6-v2)
  2. KMeans into N_CLUSTERS clusters
  3. From each cluster, pick samples proportional to cluster size
     (so larger/denser topic areas get more representation)
  4. Within each cluster, pick samples that are closest to the centroid
     (most representative) but also spread out (avoid near-duplicates)
  5. Save selected_512.jsonl

Usage:
    python select_512.py \
        --data-path data/public.jsonl \
        --output-path data/selected_512.jsonl \
        --n-samples 512 \
        --n-clusters 32
"""

import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def item_to_text(item: dict) -> str:
    """Convert a dataset item to a single string for embedding."""
    question = item["question"]
    options = item.get("options")
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = " | ".join(f"{lbl}:{opt[:60]}" for lbl, opt in zip(labels, options))
        return f"{question} OPTIONS: {opts_text}"
    return question


def diverse_sample_from_cluster(
    indices: np.ndarray,
    embeddings: np.ndarray,
    centroid: np.ndarray,
    n_pick: int,
) -> list[int]:
    """
    Pick n_pick indices from a cluster that are:
    - Close to centroid (representative)
    - Spread out from each other (diverse, no near-duplicates)

    Uses a greedy max-min-distance approach seeded from the
    closest-to-centroid point.
    """
    if len(indices) <= n_pick:
        return indices.tolist()

    cluster_embeds = embeddings[indices]  # (m, d)

    # Start with the point closest to centroid
    dists_to_centroid = np.linalg.norm(cluster_embeds - centroid, axis=1)
    selected = [int(np.argmin(dists_to_centroid))]

    # Greedily add the point that maximises minimum distance to already-selected
    for _ in range(n_pick - 1):
        min_dists = np.full(len(indices), np.inf)
        for sel_local_idx in selected:
            d = np.linalg.norm(cluster_embeds - cluster_embeds[sel_local_idx], axis=1)
            min_dists = np.minimum(min_dists, d)
        # Don't re-pick
        for sel_local_idx in selected:
            min_dists[sel_local_idx] = -np.inf
        next_pick = int(np.argmax(min_dists))
        selected.append(next_pick)

    return [int(indices[i]) for i in selected]


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path",    default="data/public.jsonl")
    p.add_argument("--output-path",  default="data/selected_512.jsonl")
    p.add_argument("--n-samples",    type=int, default=512)
    p.add_argument("--n-clusters",   type=int, default=32,
                   help="Number of KMeans clusters. 32 works well for ~1100 items.")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading data from {args.data_path} ...")
    data = load_jsonl(args.data_path)
    print(f"  Loaded {len(data)} items")

    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = len(data) - n_mcq
    print(f"  MCQ: {n_mcq}  |  Free-form: {n_free}")

    if len(data) <= args.n_samples:
        print("Dataset is already ≤ target size — saving all items.")
        save_jsonl(data, args.output_path)
        return

    # ── Embed ──────────────────────────────────────────────────────────────────
    print("Computing SBERT embeddings (all-MiniLM-L6-v2) ...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [item_to_text(item) for item in data]
    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit vectors → cosine sim = dot product
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    # ── KMeans clustering ──────────────────────────────────────────────────────
    n_clusters = min(args.n_clusters, len(data))
    print(f"Running KMeans with k={n_clusters} ...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=15)
    cluster_ids = kmeans.fit_predict(embeddings)
    centroids   = kmeans.cluster_centers_     # (k, d)

    # ── Proportional allocation ────────────────────────────────────────────────
    # Each cluster gets floor(n_samples * cluster_size / total) samples,
    # with remainders distributed to the largest clusters first.
    cluster_sizes   = np.bincount(cluster_ids, minlength=n_clusters)
    raw_allocations = args.n_samples * cluster_sizes / len(data)
    floor_alloc     = np.floor(raw_allocations).astype(int)
    remainder       = args.n_samples - floor_alloc.sum()

    # Distribute remainder to clusters with largest fractional parts
    fractions = raw_allocations - floor_alloc
    top_clusters = np.argsort(-fractions)[:remainder]
    floor_alloc[top_clusters] += 1

    print(f"\nAllocation summary (total={floor_alloc.sum()}):")
    for cid in range(n_clusters):
        print(f"  Cluster {cid:2d}: size={cluster_sizes[cid]:4d}  →  pick={floor_alloc[cid]}")

    # ── Select diverse representatives from each cluster ──────────────────────
    selected_global_indices: list[int] = []

    for cid in range(n_clusters):
        n_pick = floor_alloc[cid]
        if n_pick == 0:
            continue

        cluster_indices = np.where(cluster_ids == cid)[0]
        centroid = centroids[cid]

        picked = diverse_sample_from_cluster(
            cluster_indices, embeddings, centroid, n_pick
        )
        selected_global_indices.extend(picked)

    print(f"\nTotal selected: {len(selected_global_indices)}")

    # Sort by original order so the output file is deterministic
    selected_global_indices.sort()

    selected_data = [data[i] for i in selected_global_indices]

    # ── Sanity check: MCQ / free-form balance ─────────────────────────────────
    sel_mcq  = sum(bool(d.get("options")) for d in selected_data)
    sel_free = len(selected_data) - sel_mcq
    print(f"Selected breakdown — MCQ: {sel_mcq}  |  Free-form: {sel_free}")
    print(f"  (original ratio — MCQ: {n_mcq/len(data):.1%}, "
          f"selected ratio — MCQ: {sel_mcq/len(selected_data):.1%})")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_jsonl(selected_data, args.output_path)
    print(f"\n✅ Saved {len(selected_data)} items to {args.output_path}")

    # Also save the cluster assignments for inspection
    meta_path = args.output_path.replace(".jsonl", "_meta.json")
    meta = {
        "total_items":   len(data),
        "n_clusters":    int(n_clusters),
        "n_selected":    len(selected_data),
        "selected_ids":  [data[i]["id"] for i in selected_global_indices],
        "cluster_sizes": cluster_sizes.tolist(),
        "allocations":   floor_alloc.tolist(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"   Cluster metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
