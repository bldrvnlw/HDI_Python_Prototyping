import numpy as np
from annoy import AnnoyIndex
from joblib import cpu_count, Parallel, delayed


def euclidian_sqrdistance_matrix(
    data: np.ndarray[np.float32],
) -> np.ndarray[np.float32]:
    """
    Calculate the Euclidean distance matrix for a multidimension set of values as a numpy array(shape = (Nxd))

    Returns:
        NxN matrix of distances
    """
    num_points = data.shape[0]
    if num_points == 0:
        return np.zeros((0, 0), dtype=np.float32)

    result = np.sum((data[:, None] - data) ** 2, axis=-1)
    # Or with linalg - not sure which is faster
    # result = np.zeros((num_points, num_points), dtype=np.float32)
    # for i in range(num_points):
    #    result[i, ...] = np.linalg.norm(data - data[i], axis=1)
    # result = result**2
    return result


# Stolen from scipy tSNE implementation


def _h_beta(D_row, beta):
    P = np.exp(-D_row * beta)
    sumP = np.sum(P)
    P = P / sumP
    H = -np.sum(P * np.log(P + 1e-10))  # add epsilon for numerical stability
    return H, P


def compute_perplexity_probs(D, perplexity=30.0, tol=1e-5, max_iter=50):
    N = D.shape[0]
    target_entropy = np.log(perplexity)

    # Initialize output
    P = np.zeros((N, N))
    sigmas = np.ones(N)

    for i in range(N):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0  # beta = 1 / (2 * sigma^2)

        Di = np.delete(D[i], i)  # Exclude self-distance
        H, thisP = _h_beta(Di, beta)

        # Binary search for correct beta
        iter_count = 0
        while np.abs(H - target_entropy) > tol and iter_count < max_iter:
            if H > target_entropy:
                beta_min = beta
                beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2.0
            else:
                beta_max = beta
                beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2.0

            H, thisP = _h_beta(Di, beta)
            iter_count += 1

        # Fill row of P (with 0 at the i-th position)
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1 : N]))] = thisP
        sigmas[i] = np.sqrt(1 / (2 * beta))

    return P, sigmas


def symmetrize_P(P_cond):
    N = P_cond.shape[0]
    P = (P_cond + P_cond.T) / (2.0 * N)
    return P


def compute_annoy_probabilities(
    data: np.ndarray[np.float32],
    num_trees: int = 4,
    nn: int = 30,
) -> np.ndarray[np.float32]:
    """
    Compute the probabilities using Annoy for nearest neighbors (euclidean metric).
    Inspired by CoPilot and https://github.com/astrogilda/openTSNE

    Args:
        data (np.ndarray): Input data points.
        num_trees (int): Number of trees to build in Annoy.
        nn (int): Number of nearest neighbors to consider.

    Returns:
        np.ndarray: Probabilities matrix.
    """
    num_points, dim = data.shape
    annoy_index = AnnoyIndex(dim, "euclidean")

    for i in range(num_points):
        annoy_index.add_item(i, data[i])

    annoy_index.build(num_trees)

    # Sample check if enough neighbours are available
    for i in range(100):
        neighbours = annoy_index.get_nns_by_item(i, nn, include_distances=False)
        if len(neighbours) < nn:
            print(
                f"Warning: Not enough neighbours for point {i}. Found"
                f" {len(neighbours)} instead of {nn}."
            )
            return np.zeros((num_points, nn), dtype=np.float32)

    distances = np.zeros((num_points, nn))
    indices = np.zeros((num_points, nn)).astype(int)

    def getnns(i):
        # print(f"Processing point {i} of {num_points}")
        # Annoy returns the query point itself as the first element
        indices_i, distances_i = annoy_index.get_nns_by_item(
            i, nn + 1, include_distances=True
        )
        indices[i] = indices_i[1:]
        distances[i] = distances_i[1:]

    num_jobs = cpu_count()
    Parallel(n_jobs=num_jobs, require="sharedmem")(
        delayed(getnns)(i) for i in range(num_points)
    )

    return indices, distances
