import torch
import numpy as np


# up to 100K rows depending on GPU memory
def knn_brute_force(Xnp, k):
    # x: (N, D), on GPU
    x = torch.tensor(Xnp, device="cuda", dtype=torch.float32)
    x_norm = (x**2).sum(dim=1).view(-1, 1)  # (N, 1)
    dist = x_norm + x_norm.t() - 2.0 * x @ x.t()  # (N, N)
    indices = dist.topk(k=k + 1, largest=False).indices[:, 1:]  # Skip self (distance 0)
    return indices


def pairwise_l2_distances(x: torch.tensor):
    """_summary_

    Args:
        x (nd.array): An N X D data array

    Returns:
        torch.tensor: Upper triangle Containing pairwise l2 distances
    """
    # x: (N, D)
    x_norm = (x**2).sum(dim=1).unsqueeze(1)  # (N, 1)
    dist = x_norm + x_norm.t() - 2.0 * (x @ x.t())  # (N, N)
    dist = torch.clamp(dist, min=0.0)  # avoid negative due to precision
    dists = torch.sqrt(dist)
    dists = dists[torch.triu_indices(dists.size(0), dists.size(1), offset=1).unbind(0)]
    return dists / dists.max()


def pairwise_l2_distances_full(x: torch.tensor):
    """_summary_

    Args:
        x (nd.array): An N X D data array

    Returns:
        torch.tensor: Full matrix containing pairwise l2 distances
    """
    # x: (N, D)
    x_norm = (x**2).sum(dim=1).unsqueeze(1)  # (N, 1)
    dist = x_norm + x_norm.t() - 2.0 * (x @ x.t())  # (N, N)
    dist = torch.clamp(dist, min=0.0)  # avoid negative due to precision
    dists = torch.sqrt(dist)
    return dists


def rankdata_average_fast(x: torch.Tensor) -> torch.Tensor:
    """
    Fast average-ranking with ties, like scipy.stats.rankdata(..., method='average')
    Works on 1D tensors. Output is 0-based ranks.
    x is an array of pariwise distances for the N points
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
