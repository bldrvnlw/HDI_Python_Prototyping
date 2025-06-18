import numpy as np
from annoy import AnnoyIndex
from joblib import cpu_count, Parallel, delayed
from typing import Tuple, List
from numba import njit
from scipy.sparse import coo_matrix


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


# alternative with numba
@njit
def _h_beta_numba(D_row, beta):
    P = np.exp(-D_row * beta)
    sumP = np.sum(P)
    P = P / (sumP + 1e-10)
    H = -np.sum(P * np.log(P + 1e-10))
    return H, P


def compute_perplexity_probs(D, perplexity=30.0, tol=1e-5, max_iter=50):
    """
    Compute the conditional probabilities for a given distance matrix D
    using a binary search to find the correct sigma for each point.
    The probabilities are computed such that the perplexity of the distribution
    is equal to the specified perplexity value.
    Args:
        D (np.ndarray): Distance matrix of shape (N, N) where D[i, j] is the distance between points i and j.
        perplexity (float): Desired perplexity of the distribution.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations for binary search.
    Returns:
        P (np.ndarray): Conditional probability matrix of shape (N, nn) where P[i, j] is the conditional probability of point j given point i.
        sigmas (np.ndarray): Array of sigmas used for each point.
    """

    N = D.shape[0]
    nn = D.shape[1]
    target_entropy = np.log(perplexity)

    # Initialize output
    P = np.zeros((N, nn))
    sigmas = np.ones(N)

    for i in range(N):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0  # beta = 1 / (2 * sigma^2)

        Di = D[
            i
        ]  # distance from this point to defined neighbours (self already excluded)
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
        # P_row_i = np.zeros(nn, dtype=np.float32)
        # P_row_i[:i] = thisP[:i]
        # P_row_i[i + 1 :] = thisP[i:]
        P[i] = thisP
        sigmas[i] = np.sqrt(1 / (2 * beta))

    return P, sigmas


# alternative with numba
@njit
def binary_search_perplexity_numba(D_trunc, target_entropy, tol=1e-5, max_iter=50):
    N, nn = D_trunc.shape
    P = np.zeros((N, nn))
    sigmas = np.ones(N)

    for i in range(N):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0

        Di = D_trunc[
            i
        ]  # distance from this point to defined neighbours (self already excluded)
        H, thisP = _h_beta_numba(Di, beta)

        iter_count = 0
        while np.abs(H - target_entropy) > tol and iter_count < max_iter:
            if H > target_entropy:
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -np.inf:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

            H, thisP = _h_beta_numba(Di, beta)
            iter_count += 1

        P[i] = thisP
        sigmas[i] = np.sqrt(1 / (2 * beta))

    return P, sigmas


def compute_perplexity_probs_numba(D, perplexity=30.0, tol=1e-5, max_iter=50):
    target_entropy = np.log(perplexity)
    return binary_search_perplexity_numba(D, target_entropy, tol, max_iter)


def symmetrize_sparse_probs(P_cond, neighbors, N, nn):
    rows, cols, data = [], [], []

    for i in range(N):
        for k in range(nn):
            j = neighbors[i * nn + k]
            pij = P_cond[i, k]

            # Check if j also has i in its neighbor list
            try:
                rev_idx = np.where(neighbors[j] == i)[0]
                if rev_idx.size > 0:
                    pji = P_cond[j, rev_idx[0]]
                    pij_sym = (pij + pji) / (2 * N)
                else:
                    pij_sym = pij / (2 * N)
            except IndexError:
                pij_sym = pij / (2 * N)

            rows.append(i)
            cols.append(j)
            data.append(pij_sym)

    return coo_matrix((data, (rows, cols)), shape=(N, N))


def symmetrize_P(P_cond):
    N = P_cond.shape[0]
    P = (P_cond + P_cond.T) / (2.0 * N)
    return P


def compute_annoy_probabilities(
    data: np.ndarray[np.float32],
    num_trees: int = 20,
    nn: int = 30,
) -> Tuple[np.ndarray[np.float32], np.ndarray[np.uint32], np.ndarray[np.int32]]:
    """
    Compute the probabilities using Annoy for nearest neighbors (euclidean metric).
    Inspired by CoPilot and https://github.com/astrogilda/openTSNE

    Args:
        data (np.ndarray): Input data points.
        num_trees (int): Number of trees to build in Annoy.
        nn (int): Number of nearest neighbors to consider.

    Returns:
        tuple
            np.ndarray: distances
            np.ndarray: neighbours
            nd:array indices
    """
    num_points, dim = data.shape
    annoy_index = AnnoyIndex(dim, "euclidean")

    for i in range(num_points):
        annoy_index.add_item(i, data[i])

    annoy_index.build(num_trees)

    # Sample check if enough neighbours are available
    # for i in range(100):
    #    neighbours = annoy_index.get_nns_by_item(i, nn, include_distances=False)
    #    if len(neighbours) < nn:
    #        print(
    #            f"Warning: Not enough neighbours for point {i}. Found"
    #            f" {len(neighbours)} instead of {nn}."
    #        )
    #        return np.zeros((num_points, nn), dtype=np.float32)

    distances = np.zeros((num_points, nn))
    neighbours = np.zeros((num_points, nn)).astype(np.uint32)
    indices = np.zeros((num_points, 2)).astype(np.uint32)

    def getnns(i):
        # print(f"Processing point {i} of {num_points}")
        # Annoy returns the query point itself as the first element
        neighbours_i, distances_i = annoy_index.get_nns_by_item(
            np.int32(i), nn + 1, include_distances=True
        )
        neighbours[i] = neighbours_i[1:]
        if len(neighbours_i) < nn + 1:
            raise Exception(f"Too few neighbours {len(neighbours_i)} ")
        distances[i] = distances_i[1:]  # Append distances excluding the query point
        indices[i] = [i * nn, nn]  # offset size

    # Parallel processing to speed up the nearest neighbor search

    num_jobs = cpu_count()
    Parallel(n_jobs=num_jobs, require="sharedmem", prefer="threads")(
        delayed(getnns)(i) for i in range(num_points)
    )

    return (
        np.ascontiguousarray(distances),
        np.ascontiguousarray(neighbours.reshape(num_points * nn)),
        np.ascontiguousarray(indices.reshape(num_points * 2)),
    )


def get_random_uniform_circular_embedding(
    num_points: int, radius: float
) -> np.ndarray[np.float32]:
    """Generate a ndarray shape: (num_points, 2) containing x,y coordinates
    of points uniformly distributed within the circle of given radius
    centered on 0.

    Args:
        num_points (int): The number of 2D points to generate
        range (float): The radius of the circle

    Returns:
        np.ndarray[np.float32]: The uniform circular distrubution
    """
    rng = np.random.default_rng()
    x = radius + 0.1
    y = radius + 0.1
    sqr_rad = radius * radius
    result = np.zeros((num_points, 2), dtype=np.float32)
    for i in range(num_points):
        while x**2 + y**2 > sqr_rad:
            r_point = rng.uniform(-1, 1, 2).astype(np.float32) * radius
            x = r_point[0]
            y = r_point[1]
        result[i] = r_point
        x = radius + 0.1
        y = radius + 0.1

    return result


class DataMap:
    """
    Class to manage a mapping of data points to their indices and distances.
    This is used to store the results of the Annoy nearest neighbor search.
    """

    def __init__(
        self, indices: np.ndarray[np.int32], distances: np.ndarray[np.float32]
    ):
        self.indices = indices
        self.distances = distances
        # value = tuple(int, float)
        # storage = List[value]

    def __repr__(self):
        return (
            f"DataMap(indices.shape={self.indices.shape},"
            f" distances.shape={self.distances.shape})"
        )
