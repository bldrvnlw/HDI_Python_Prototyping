import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
import numba as nb
from sklearn.metrics import auc
from scipy import stats
from numba import njit, prange
import torch
import math
from optimize.nn_points_torch import NNPointsTorch
from utils.base import (
    knn_brute_force,
)

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


def orig_HAP_neighborhood_preservation(X, y, nr_neighbors=10, metric="euclidean"):
    dists_high, indexes_high = KDTree(X, leaf_size=2, metric=metric).query(
        X, k=nr_neighbors
    )
    dists_low, indexes_low = KDTree(y, leaf_size=2, metric=metric).query(
        y, k=nr_neighbors
    )

    neigh_pres = np.zeros(len(X))
    for i in range(len(X)):
        for p in range(nr_neighbors):
            for q in range(nr_neighbors):
                if indexes_high[i][p] == indexes_low[i][q]:
                    neigh_pres[i] = neigh_pres[i] + 1
        neigh_pres[i] = neigh_pres[i] / nr_neighbors

    return np.average(neigh_pres)


def orig_HAP_neighborhood_hit(y, label, nr_neighbors=10, metric="euclidean"):
    dists_low, indexes_low = KDTree(y, leaf_size=2, metric=metric).query(
        y, k=nr_neighbors
    )

    neigh_hit = np.zeros(len(y))
    for i in range(len(y)):
        for j in range(nr_neighbors):
            if label[i] == label[indexes_low[i][j]]:
                neigh_hit[i] = neigh_hit[i] + 1
        neigh_hit[i] = neigh_hit[i] / nr_neighbors

    return np.average(neigh_hit)


def neighborhood_hit_torch(y, label, nr_neighbors=10):
    """Measures average fraction of labels of the nn points
    that match the point label

    Args:
        y (_type_): The embedding
        label (_type_): labels
        nr_neighbors (int, optional): _description_. Defaults to 10.
    """
    indexes_low = knn_brute_force(y, nr_neighbors)
    label_t = torch.tensor(label, device="cuda", dtype=torch.int32)

    # Lookup neighbor labels
    neighbor_labels = label_t[indexes_low]  # (N, k)

    # Expand labels for comparison: (N, 1) vs (N, k)
    matches = neighbor_labels == label_t.unsqueeze(1)  # (N, k), boolean

    # Count matches and normalize
    neigh_hit = matches.sum(dim=1).float() / indexes_low.shape[1]
    return np.average(neigh_hit.cpu().numpy())


def compute_neigh_preservation_torch(indexes_high, indexes_low):
    # Shape manipulation for broadcasting
    # indexes_high: (N, K) -> (N, K, 1)
    # indexes_low:  (N, K) -> (N, 1, K)
    high_expanded = indexes_high.unsqueeze(2)
    low_expanded = indexes_low.unsqueeze(1)

    # Compare and count matches along the neighbor dimension
    matches = high_expanded == low_expanded  # (N, K, K)
    row_match_counts = matches.any(dim=2).sum(dim=1)  # (N,)

    scores = row_match_counts.float() / indexes_high.shape[1]
    return scores.cpu().numpy()


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


def neighborhood_preservation_torch(
    X: NNPointsTorch, embed: NNPointsTorch, nr_neighbors=30, metric="euclidean"
):
    """Compare nearest neighbours between the two data sets
    Metric is hardcoded as euclidean.

    Args:
        X (_type_): original data set NxM
        embed (_type_): 2D embedding set Nx2
        nr_neighbors (int, optional): number of neighbors to compare. Defaults to 30.

    Returns:
        float: neighbourhood preservation metric. Range: 0 - 1
    """
    indexes_high = X.getNN(nr_neighbors)
    # dists_high, indexes_high = KDTree(X, leaf_size=2, metric=metric).query(
    #    X, k=nr_neighbors
    # )
    indexes_low = embed.getNN(nr_neighbors)
    # dists_low, indexes_low = KDTree(embed, leaf_size=2, metric=metric).query(
    #    embed, k=nr_neighbors
    # )
    scores = compute_neigh_preservation_torch(indexes_high, indexes_low)
    # scores = compute_neigh_preservation(indexes_high, indexes_low, nr_neighbors)
    return np.mean(scores)


def metric_stress(D_high, D_low):
    return math.sqrt(np.sum(((D_high - D_low) ** 2) / np.sum(D_high**2)))


def metric_stress_torch(D_high: NNPointsTorch, D_low: NNPointsTorch):
    return math.sqrt(
        torch.sum(((D_high.tensor - D_low.tensor) ** 2) / torch.sum(D_high.tensor**2))
    )


def metric_shepard_diagram_correlation(D_high, D_low):
    """Calculate Shepard Goodness.
    This is the not the shepard u the Spearman Rho (ranked) coefficient

    Args:
        D_high (np.ndarray): pairwise distances between high dimensional data
        D_low (np.ndarray): pairwise distances between low dimensional data

    Returns:
        float: Spearman correlation coefficient
    """
    return stats.spearmanr(D_high, D_low)[0]


def spearmanr_torch(x: NNPointsTorch, y: NNPointsTorch) -> torch.tensor:
    """A pytorch implementation of the Spearman rank correlation.

    The Spearman correlation is the Pearson correlation
    calculated on ranks rather then actual values. This makes
    it less sensitive to outliers and focuses less on
    actual distances in the low dimensional space which
    are not preserved by tSNE. Result varies beween:
        -1 - perfect anti-correlation
        1  - perfect correlation

    Based on the scipy implementation
    https://github.com/scipy/scipy/blob/4d3dcc103612a2edaec7069638b7f8d0d75cab8b/scipy/stats/_stats_py.py#L5181

    Args:
        x (NNPointsTorch): optimized points containing pairwise distances high dimensional data
        y (NNPointsTorch): optimized points containing pairwise distances low dimensional data

    Returns:
        torch.tensor: torch.tensor
    """
    # assert x.pointData.shape == y.pointData.shape

    rx = x.get_rank()
    print(rx)
    ry = y.get_rank()
    print(ry)

    # Pearson correlation of ranks
    rx_mean = rx.mean()
    ry_mean = ry.mean()
    cov = ((rx - rx_mean) * (ry - ry_mean)).mean()
    std_rx = rx.std(unbiased=False)
    std_ry = ry.std(unbiased=False)
    return cov / (std_rx * std_ry + 1e-8)  # Add small epsilon to avoid divide-by-zero


def get_spearman_and_stress(D_hd: NNPointsTorch, D_ld: NNPointsTorch):
    shepard = spearmanr_torch(D_hd, D_ld)
    stress = metric_stress_torch(
        D_hd,
        D_ld,
    )
    return shepard.cpu().numpy(), stress


def trustworthiness_torch(
    hd_X: NNPointsTorch, ld_X: NNPointsTorch, n_neighbors: int = 5
) -> float:
    # reimplement the trustworthiness in sklearn manifole _t_sne.py
    # https://github.com/scikit-learn/scikit-learn/blob/c5497b7f7eacfaff061cf68e09bcd48aa93d4d6b/sklearn/manifold/_t_sne.py#L456

    N = hd_X.pointData.shape[0]

    dist_X = hd_X.get_pairwise_ls_dist_full()
    dist_X.fill_diagonal_(torch.inf)
    sortIndex_X = torch.argsort(dist_X)

    embed_nn = ld_X.getNN(n_neighbors)

    inverted_index = torch.zeros(N, N, device="cuda", dtype=torch.int)
    ordered_index = torch.arange(N + 1, device="cuda", dtype=torch.int)
    inverted_index[ordered_index[:-1].unsqueeze(1), sortIndex_X] = ordered_index[1:]
    ranks = inverted_index[ordered_index[:-1, np.newaxis], embed_nn] - n_neighbors

    t = torch.sum(ranks[ranks > 0])
    t = 1.0 - t * (2.0 / (N * n_neighbors * (2.0 * N - 3.0 * n_neighbors - 1.0)))

    return float(t)


def continuity_torch(
    hd_X: NNPointsTorch, ld_X: NNPointsTorch, n_neighbors: int = 5
) -> float:
    return trustworthiness_torch(ld_X, hd_X, n_neighbors)
