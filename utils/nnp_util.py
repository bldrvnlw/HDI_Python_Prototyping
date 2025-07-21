import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
import numba as nb
from sklearn.metrics import auc
from scipy import stats
from numba import njit, prange
import torch
import math

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


# up to 100K rows depending on GPU memory
def knn_brute_force(Xnp, k):
    # x: (N, D), on GPU
    x = torch.tensor(Xnp, device="cuda", dtype=torch.float32)
    x_norm = (x**2).sum(dim=1).view(-1, 1)  # (N, 1)
    dist = x_norm + x_norm.t() - 2.0 * x @ x.t()  # (N, N)
    indices = dist.topk(k=k + 1, largest=False).indices[:, 1:]  # Skip self (distance 0)
    return indices


def neighborhood_preservation_torch(X, embed, nr_neighbors=30, metric="euclidean"):
    """Compare nearest neighbours between the two data sets
    Metric is hardcoded as euclidean.

    Args:
        X (_type_): original data set NxM
        embed (_type_): 2D embedding set Nx2
        nr_neighbors (int, optional): number of neighbors to compare. Defaults to 30.

    Returns:
        float: neighbourhood preservation metric. Range: 0 - 1
    """
    indexes_high = knn_brute_force(X, nr_neighbors)
    # dists_high, indexes_high = KDTree(X, leaf_size=2, metric=metric).query(
    #    X, k=nr_neighbors
    # )
    indexes_low = knn_brute_force(embed, nr_neighbors)
    # dists_low, indexes_low = KDTree(embed, leaf_size=2, metric=metric).query(
    #    embed, k=nr_neighbors
    # )
    scores = compute_neigh_preservation_torch(indexes_high, indexes_low)
    # scores = compute_neigh_preservation(indexes_high, indexes_low, nr_neighbors)
    return np.mean(scores)

def trustworthiness: float(X, embed, nr_neighbors=30, metric="euclidean"):




def metric_stress(D_high, D_low):
    return math.sqrt(np.sum(((D_high - D_low) ** 2) / np.sum(D_high**2)))


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


def rankdata_average_fast(x: torch.Tensor) -> torch.Tensor:
    """
    Fast average-ranking with ties, like scipy.stats.rankdata(..., method='average')
    Works on 1D tensors. Output is 0-based ranks.
    """
    x = x.float()
    N = x.shape[0]

    # Step 1: sort x and get sorted indices
    sort_idx = torch.argsort(x)
    sorted_x = x[sort_idx]

    # Step 2: assign provisional ranks (0-based :  0 - N-1)
    ranks = torch.arange(N, dtype=torch.float32, device=x.device)

    # Step 3: find run boundaries (where values change)
    diffs = torch.diff(sorted_x, prepend=sorted_x[:1] - 1)
    group_starts = (diffs != 0).cumsum(dim=0) - 1  # group ID per element

    # Step 4: compute mean rank per group (segment mean)
    group_ids = group_starts  # shape: (N,)
    num_groups = group_ids.max().item() + 1

    # Sum of ranks per group
    rank_sums = torch.zeros(num_groups, device=x.device).scatter_add(
        0, group_ids, ranks
    )
    counts = torch.bincount(group_ids, minlength=num_groups).float()
    group_means = rank_sums / counts  # shape: (num_groups,)

    # Step 5: assign mean ranks to elements
    avg_ranks = group_means[group_ids]  # shape: (N,)

    # Step 6: undo the sorting
    unsort_idx = torch.argsort(sort_idx)
    return avg_ranks[unsort_idx]


def spearmanr_torch(x: torch.tensor, y: torch.tensor) -> torch.tensor:
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
        x (torch.tensor): pairwise distances high dimensional data
        y (torch.tensor): pairwise distances low dimensional data

    Returns:
        torch.tensor: torch.tensor
    """
    assert x.shape == y.shape
    x = x.float()
    y = y.float()

    rx = rankdata_average_fast(x)
    print(rx)
    ry = rankdata_average_fast(y)
    print(ry)

    # Pearson correlation of ranks
    rx_mean = rx.mean()
    ry_mean = ry.mean()
    cov = ((rx - rx_mean) * (ry - ry_mean)).mean()
    std_rx = rx.std(unbiased=False)
    std_ry = ry.std(unbiased=False)
    return cov / (std_rx * std_ry + 1e-8)  # Add small epsilon to avoid divide-by-zero


def pairwise_l2_distances(x):
    # x: (N, D)
    x_norm = (x**2).sum(dim=1).unsqueeze(1)  # (N, 1)
    dist = x_norm + x_norm.t() - 2.0 * (x @ x.t())  # (N, N)
    dist = torch.clamp(dist, min=0.0)  # avoid negative due to precision
    dists = torch.sqrt(dist)
    dists = dists[torch.triu_indices(dists.size(0), dists.size(1), offset=1).unbind(0)]
    return dists / dists.max()


def get_spearman_and_stress(hd_X, ld_X):
    D_hd = pairwise_l2_distances(torch.tensor(hd_X, device="cuda", dtype=torch.float32))
    D_ld = pairwise_l2_distances(torch.tensor(ld_X, device="cuda", dtype=torch.float32))
    shepard = spearmanr_torch(D_hd, D_ld)
    stress = metric_stress(D_hd.cpu().numpy(), D_ld.cpu().numpy())
    return shepard.cpu().numpy(), stress
