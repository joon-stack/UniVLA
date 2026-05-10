#!/usr/bin/env python3
"""t-SNE views of cached LAM latents colored by action-sum k-means labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


METHODS = [
    ("Hyper <RAD><DIRx4>", "factorized_hyper_30k.npz", "hyper_rad_dir"),
    ("Hyper <DIRx4> only", "factorized_hyper_30k.npz", "hyper_dir_only"),
    ("Hyper raw <Z_Qx4>", "factorized_hyper_30k.npz", "raw"),
    ("Euclid raw <Z_Qx4>", "euclidean_visual_vq_30k.npz", "raw"),
    ("UniVLA stage2 raw <Z_Qx4>", "univla_stage2_60k.npz", "raw"),
]


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + eps)


def load_feature(npz_path: Path, kind: str) -> np.ndarray:
    data = np.load(npz_path)
    if kind == "raw":
        return data["z_raw"].astype(np.float32)

    token_unit = data["z_token_unit"].astype(np.float32)
    if kind == "hyper_dir_only":
        return token_unit

    if kind == "hyper_rad_dir":
        norms = data["z_token_norms"].astype(np.float32)
        n = token_unit.shape[0]
        dim = token_unit.shape[1] // norms.shape[1]
        radius_token = np.zeros((n, dim), dtype=np.float32)
        radius_token[:, : norms.shape[1]] = norms
        return np.concatenate([radius_token, token_unit], axis=1)

    raise ValueError(f"Unknown feature kind: {kind}")


def action_sum_labels(cache_dir: Path, clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(cache_dir / "factorized_hyper_30k.npz")
    action_seq = data["action_seq"].astype(np.float32)
    action_sum = action_seq.reshape(action_seq.shape[0], -1, 7).sum(axis=1)
    action_sum_z = zscore(action_sum)
    kmeans = KMeans(n_clusters=clusters, random_state=seed, n_init=20)
    labels = kmeans.fit_predict(action_sum_z)
    centers = kmeans.cluster_centers_
    return labels, centers


def between_total_ratio(x2d: np.ndarray, labels: np.ndarray) -> float:
    total = np.sum((x2d - x2d.mean(axis=0, keepdims=True)) ** 2)
    between = 0.0
    for label in np.unique(labels):
        member = x2d[labels == label]
        between += len(member) * np.sum((member.mean(axis=0) - x2d.mean(axis=0)) ** 2)
    return float(between / max(total, 1e-12))


def tsne_2d(x: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    x = StandardScaler().fit_transform(x)
    n_comp = min(50, x.shape[0] - 1, x.shape[1])
    if n_comp >= 2:
        x = PCA(n_components=n_comp, random_state=seed).fit_transform(x)
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        metric="euclidean",
        max_iter=1000,
        verbose=1,
    ).fit_transform(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/analysis/cache/zq_full_action_k9_all_2048"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/analysis/plots"))
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=40.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels, centers = action_sum_labels(args.cache_dir, args.clusters, args.seed)
    counts = np.bincount(labels, minlength=args.clusters)

    rows: list[dict[str, float | int | str]] = []
    embeddings: list[tuple[str, np.ndarray, float]] = []
    for title, filename, kind in METHODS:
        x = load_feature(args.cache_dir / filename, kind)
        x2d = tsne_2d(x, seed=args.seed, perplexity=args.perplexity)
        ratio = between_total_ratio(x2d, labels)
        embeddings.append((title, x2d, ratio))
        for i, (x0, x1) in enumerate(x2d):
            rows.append(
                {
                    "method": title,
                    "sample_idx": i,
                    "tsne_0": float(x0),
                    "tsne_1": float(x1),
                    "action_sum_kmeans10": int(labels[i]),
                    "between_total_ratio": ratio,
                }
            )

    stem = f"zq_tsne_scatter_action_sum_kmeans{args.clusters}_k9_2048"
    csv_path = args.out_dir / f"{stem}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    centers_path = args.out_dir / f"action_sum_kmeans{args.clusters}_centers_k9_2048_tsne.csv"
    np.savetxt(centers_path, centers, delimiter=",")

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, len(embeddings), figsize=(20, 4.2), constrained_layout=True)
    for ax, (title, x2d, ratio) in zip(axes, embeddings):
        for label in range(args.clusters):
            idx = labels == label
            ax.scatter(
                x2d[idx, 0],
                x2d[idx, 1],
                s=8,
                alpha=0.72,
                c=[cmap(label % 10)],
                linewidths=0,
                label=str(label) if ax is axes[0] else None,
            )
        ax.set_title(f"{title}\ncluster sep={ratio:.3f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=args.clusters, frameon=False)
    fig.suptitle(
        "LAM latent t-SNE colored by k-means clusters of action-sum sequence (k=9)",
        fontsize=12,
        y=1.04,
    )
    png_path = args.out_dir / f"{stem}.png"
    pdf_path = args.out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print("Saved:", png_path)
    print("Saved:", pdf_path)
    print("Saved:", csv_path)
    print("Saved:", centers_path)
    print("Cluster counts:", counts.tolist())
    for title, _, ratio in embeddings:
        print(f"{title}: between_total_ratio={ratio:.4f}")


if __name__ == "__main__":
    main()
