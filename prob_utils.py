import numpy as np
from annoy import AnnoyIndex
from joblib import cpu_count, Parallel, delayed
from typing import Tuple, List
from numba import njit
from scipy.sparse import coo_matrix, csr_matrix
import hnswlib

from openTSNE import affinity
import torch


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


def compute_perplexity_probs(D, perplexity=30.0, tol=1e-5, max_iter=200):
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
        H, thisP = _h_beta(Di, beta, P)

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


# reference https://nicola17.github.io/publications/2016_AtSNE.pdf (section 3 tSNE) for the math in original form


# alternative with numba
# Taking the log of the perplexity (called target_entropy or H here) we have
# https://latex.codecogs.com/gif.latex?log_2(\mu)=-\sum_{j}^{N}p_{j|i}%20log_2(p_{j|i})


@njit
def _h_beta_numba(D_i, beta_i):
    """_summary_
        Compute Gaussian kernel row
    Args:
        D_i (nd.array[np.float32]): all the neighbour distances squared for a point i
        beta_i (float): current 1/(2*sigma^2) estimate for row i

    Returns:
        tuple(float, nd.array[np.float32]): estimated total_entropy i.e. log(perplexity) and conditioal distribution
    """
    # Calculate the gaussian value array for each neighbour of point i
    # https://latex.codecogs.com/gif.latex?\exp(-(||x_i-x_j||^2)/(2\sigma^2_i))
    Gi = np.exp(-D_i * beta_i)
    # sum the gaussians on the current point
    # https://latex.codecogs.com/gif.latex?\sum_{k&space;\neq&space;i}^{N}\exp(-(||x_i-x_j||^2)/(2\sigma^2_i))
    sum_Gi = np.sum(Gi)
    Pjci = Gi / (sum_Gi + 1e-10)  # calculate the probability similarity distribution
    H = -np.sum(Pjci * np.log(Pjci + 1e-10))  # estimate of log(target_perplexity)
    return H, Pjci


# Implement with numba
#
# Iteratively search or a value of sigma that givea a p_j|i distribution such that
# https://latex.codecogs.com/gif.latex?\mu=2^{-\sum_{j}^{N}p_{j|i}%20log_2(p_{j|i})}
# where \mu is the perplexity
# log(\mu) is denoted by H a.k.a. target_entropy
#
# In practice we look for beta defined as 1/(2*sigma^2) and
# the comparison is performed with a tolerance
# Return the conditional probabilities Pj|i
# These need to be symmetrized and normalized for the
# high dimensional probabilities Pij


@njit
def binary_search_perplexity_numba(D, target_entropy, tol=1e-5, max_iter=200):
    N, nn = D.shape
    P = np.zeros((N, nn))
    sigmas = np.ones(N)

    for i in range(N):  # create the distribution on a point by point basis
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0

        # Calculate the log(perplexity) - H_est
        # and the corresponding probability array - P_est
        # for the current point distances - D[i]
        # given an estimated beta value
        H_est, Pjci_est = _h_beta_numba(D[i], beta)

        iter_count = 0
        # Is H_est withing the tolerance bounds or should we continue iterating
        while np.abs(H_est - target_entropy) > tol and iter_count < max_iter:
            if H_est > target_entropy:
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

            H_est, Pjci_est = _h_beta_numba(D[i], beta)
            iter_count += 1

        P[i] = Pjci_est
        sigmas[i] = np.sqrt(1 / (2 * beta))

    return P, sigmas


def compute_perplexity_probs_numba(D, perplexity=30.0, tol=1e-5, max_iter=200):
    target_entropy = np.log(perplexity)
    return binary_search_perplexity_numba(D, target_entropy, tol, max_iter)


def symmetrize_P(P_cond, neighbors, nn, denormalize=True):
    num_points = P_cond.shape[0]
    P = csr_matrix(
        (
            P_cond.ravel(),
            neighbors.ravel(),
            range(0, num_points * nn + 1, nn),
        ),
        shape=(num_points, num_points),
    )
    # Simple symmetrization creates missing probability values in sparse array
    # so the size will be > (num_points * nn)
    P = (P + P.T) / 2.0
    # denormalize
    P /= np.sum(P)
    # flatten the probabilities and neighbours into indexed 1D arrays
    offset = 0
    length = 0
    total = 0

    indices = np.empty((num_points * 2), dtype=np.uint32)
    for i in range(P.shape[0]):
        row = P.getrow(i)
        indices[i * 2] = offset
        length = row.data.shape[0]
        indices[i * 2 + 1] = length
        total = total + length
        offset += length
    print(f"Total number of symmetrized conditions probs: {total} ")
    neighbours = np.empty((total), dtype=np.uint32)
    probabilities = np.empty((total), dtype=np.float32)
    denorm_value = 1
    if denormalize:
        denorm_value = num_points

    for i in range(P.shape[0]):
        row = P.getrow(i)
        neighbours[range(indices[i * 2], indices[i * 2] + indices[i * 2 + 1])] = (
            row.indices
        )
        # probabilities from openTSNE affinity are normalized over num_points
        probabilities[range(indices[i * 2], indices[i * 2] + indices[i * 2 + 1])] = (
            row.data * denorm_value
        )

    return neighbours, probabilities, indices


def compute_hnsw_distances(
    data: np.ndarray[np.float32], nn: int = 30, ef: int = 200, M: int = 16
) -> Tuple[np.ndarray[np.float32], np.ndarray[np.uint32], np.ndarray[np.int32]]:
    num_points, dim = data.shape
    p = hnswlib.Index(space="l2", dim=dim)  # euclidean
    p.init_index(max_elements=num_points, ef_construction=ef, M=M)
    ids = np.arange(num_points)
    p.add_items(data, ids)
    p.set_ef(nn * 3)
    labels, dist = p.knn_query(data, k=nn + 1)

    distances = np.zeros((num_points, nn))
    neighbours = np.zeros((num_points, nn)).astype(np.uint32)
    indices = np.zeros((num_points, 2)).astype(np.uint32)

    def serialize(i):
        neighbours[i] = labels[i, 1:]
        distances[i] = dist[i, 1:]
        indices[i] = [i * nn, nn]

    num_jobs = cpu_count()
    Parallel(n_jobs=num_jobs, require="sharedmem", prefer="threads")(
        delayed(serialize)(i) for i in range(num_points)
    )
    return (
        distances,
        neighbours,
        indices,
    )


def compute_annoy_distances(
    data: np.ndarray[np.float32],
    num_trees: int = 20,
    nn: int = 30,
) -> Tuple[np.ndarray[np.float32], np.ndarray[np.uint32], np.ndarray[np.int32]]:
    """
    Compute the probabilities using Annoy for nearest neighbors (euclidean metric).
    Inspired by CoPilot and https://github.com/pavlin-policar/openTSNE

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

    annoy_index.build(num_trees, n_jobs=10)

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
        # print(f"Num neighbours {neighbours_i.shape[0]}")
        assert (
            len(neighbours_i) == nn + 1
        ), f"Unexpected number of neighbours {len(neighbours_i)}"
        neighbours[i] = neighbours_i[1:]
        if len(neighbours_i) < nn + 1:
            raise Exception(f"Too few neighbours {len(neighbours_i)} ")
        distances[i] = distances_i[1:]  # Append distances excluding the query point
        indices[i] = [i * nn, nn]  # offset size

    # Parallel processing to speed up the nearest neighbor search

    num_jobs = cpu_count()
    # neighbours_i, distances_i = annoy_index.get_nns_by_item(
    #    np.int32(0), nn + 1, include_distances=True
    # )
    Parallel(n_jobs=num_jobs, require="sharedmem", prefer="threads")(
        delayed(getnns)(i) for i in range(num_points)
    )

    distances = np.power(distances, 2)
    return (
        distances,
        neighbours,
        indices,
    )


def get_random_uniform_circular_embedding(
    num_points: int, radius: float, seed: int = 42
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
    rng = np.random.default_rng(seed)
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


def getProbabilitiesOpenTSNE(
    X: np.ndarray,
    perplexity: int = 30,
    metric: str = "euclidean",
    njobs: int = 20,
    symmetrize: bool = True,
    denormalize: bool = True,
    verbose: bool = True,
):
    # defaults to annot
    affinities = affinity.PerplexityBasedNN(
        X,
        perplexity=perplexity,
        metric=metric,
        n_jobs=njobs,
        symmetrize=symmetrize,
        verbose=verbose,
        method="annoy",
    )
    P = affinities.P
    # P.sort_indices()
    num_points = X.shape[0]
    indices = np.empty((num_points * 2), dtype=np.uint32)

    offset = 0
    length = 0
    total = 0
    for i in range(P.shape[0]):
        row = P.getrow(i)
        indices[i * 2] = offset
        length = row.data.shape[0]
        indices[i * 2 + 1] = length
        total = total + length
        offset += length

    neighbours = np.empty((total), dtype=np.uint32)
    probabilities = np.empty((total), dtype=np.float32)
    denorm_value = 1
    if denormalize:
        denorm_value = num_points
    for i in range(P.shape[0]):
        row = P.getrow(i)
        neighbours[range(indices[i * 2], indices[i * 2] + indices[i * 2 + 1])] = (
            row.indices
        )
        # probabilities from openTSNE affinity are normalized over num_points
        probabilities[range(indices[i * 2], indices[i * 2] + indices[i * 2 + 1])] = (
            row.data * denorm_value
        )

    return neighbours, probabilities, indices


from numba import njit, prange


@njit(parallel=True)
def calculate_normalization_Q(points):
    n = points.shape[0]
    total_sum = 0.0

    for i in prange(n):  # prange enables parallel execution of the outer loop
        for j in range(i + 1, n):  # Only calculate unique pairs (j > i)
            diff_x = points[i, 0] - points[j, 0]
            diff_y = points[i, 1] - points[j, 1]
            D = diff_x**2 + diff_y**2
            total_sum += 1.0 / (1.0 + D)

    total_sum = total_sum * 2
    return total_sum


def compute_Qnorm_cuda(points, eps=1e-12):
    p_tensor = torch.tensor(points, device="cuda")

    # Compute pairwise squared distances
    D = torch.cdist(p_tensor, p_tensor, p=2)
    # print(f"distance shape {D.shape} device {D.device}")

    # Compute Q matrix
    num = 1 / (1.0 + D)  # t-SNE kernel
    # print(f"num shape {num.shape} device {num.device}")
    num.fill_diagonal_(0.0)  # set Q_ii = 0

    return num.sum().item()
