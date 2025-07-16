import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
import numba as nb
from sklearn.metrics import auc
from numba import njit, prange

# === Config ===
K = 15  # Max neighborhood size to evaluate
metric = "euclidean"  # or 'cosine', etc.


# === Step 1: Compute KNN in original and embedded space ===
def get_knn(X, k):
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=-1)
    nn.fit(X)
    neighbors = nn.kneighbors(X, return_distance=False)[:, 1:]  # exclude self
    return neighbors


# Assume you have:
# X_orig: high-dim data, shape (n_samples, n_features)
# X_emb:  low-dim embedding, shape (n_samples, n_embed_dim)

# Example dummy data
# X_orig = np.random.randn(1000, 50)
# X_emb = np.random.randn(1000, 2)


# === Step 2: Compute True Positives count across K for each point ===
@nb.njit(parallel=True)
def compute_tp_curve(orig_nb, emb_nb, K):
    n = orig_nb.shape[0]
    tp = np.zeros((n, K), dtype=np.int32)
    for i in nb.prange(n):
        orig_set = set(orig_nb[i])
        for k in range(1, K + 1):
            emb_k = emb_nb[i, :k]
            count = 0
            for j in emb_k:
                if j in orig_set:
                    count += 1
            tp[i, k - 1] = count
    return tp


def compute_nnp(X_orig, X_embed, K=30):
    orig_neighbors = get_knn(X_orig, K)
    emb_neighbors = get_knn(X_embed, K)
    tp_counts = compute_tp_curve(orig_neighbors, emb_neighbors, K)

    # === Step 3: Precision & Recall curves ===
    n_samples = tp_counts.shape[0]
    precision = np.mean(tp_counts / np.arange(1, K + 1), axis=0)
    recall = np.mean(tp_counts / K, axis=0)

    # === Step 4: Compute AUC of Precision-Recall ===
    nnp_auc = auc(recall, precision)

    print(f"NNP AUC (area under precision-recall curve): {nnp_auc:.4f}")
    return nnp_auc


@njit(parallel=True)
def compute_neigh_preservation(indexes_high, indexes_low, nr_neighbors):
    n_samples = indexes_high.shape[0]
    scores = np.zeros(n_samples)
    for i in prange(n_samples):
        count = 0
        for p in range(nr_neighbors):
            for q in range(nr_neighbors):
                if indexes_high[i, p] == indexes_low[i, q]:
                    count += 1
        scores[i] = count / nr_neighbors
    return scores


def neighborhood_preservation(X, embed, nr_neighbors=30, metric="euclidean"):
    dists_high, indexes_high = KDTree(X, leaf_size=2, metric=metric).query(
        X, k=nr_neighbors
    )
    dists_low, indexes_low = KDTree(embed, leaf_size=2, metric=metric).query(
        embed, k=nr_neighbors
    )

    scores = compute_neigh_preservation(indexes_high, indexes_low, nr_neighbors)
    return np.mean(scores)
