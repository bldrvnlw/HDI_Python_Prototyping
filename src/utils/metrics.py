# Contains some numba accellerated metrics fro DR

# Copied from https://timsainburg.com/coranking-matrix-python-numba.html
# Fast computation of a coranking matrix for dimensionality reduction
# with python, joblib, and numba

from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from joblib import Parallel, delayed
import numba
from tqdm import tqdm
import numpy as np


def compute_ranking_matrix_parallel(D):
    """Compute ranking matrix in parallel. Input (D) is distance matrix"""
    # if data is small, no need for parallel
    if len(D) > 1000:
        n_jobs = -1
    else:
        n_jobs = 1
    r1 = Parallel(n_jobs, prefer="threads")(
        delayed(np.argsort)(i)
        for i in tqdm(D.T, desc="computing rank matrix", leave=False)
    )
    r2 = Parallel(n_jobs, prefer="threads")(
        delayed(np.argsort)(i)
        for i in tqdm(r1, desc="computing rank matrix", leave=False)
    )
    # write as a single array
    r2_array = np.zeros((len(r2), len(r2[0])), dtype="int32")
    for i, r2row in enumerate(tqdm(r2, desc="concatenating rank matrix", leave=False)):
        r2_array[i] = r2row
    return r2_array


@numba.njit(fastmath=True)
def populate_Q(Q, i, m, R1, R2):
    """populate coranking matrix using numba for speed"""
    for j in range(m):
        k = R1[i, j]
        l = R2[i, j]
        Q[k, l] += 1
    return Q


def iterate_compute_distances(data):
    """Compute pairwise distance matrix iteratively, so we can see progress"""
    n = len(data)
    D = np.zeros((n, n), dtype="float32")
    col = 0
    with tqdm(desc="computing pairwise distances", leave=False) as pbar:
        for i, distances in enumerate(
            pairwise_distances_chunked(data, n_jobs=-1),
        ):
            D[col : col + len(distances)] = distances
            col += len(distances)
            if i == 0:
                pbar.total = int(len(data) / len(distances))
            pbar.update(1)
    return D


def compute_coranking_matrix(data_ld, data_hd=None, D_hd=None):
    """Compute the full coranking matrix"""

    # compute pairwise probabilities
    if D_hd is None:
        D_hd = iterate_compute_distances(data_hd)

    D_ld = iterate_compute_distances(data_ld)
    n = len(D_ld)
    # compute the ranking matrix for high and low D
    rm_hd = compute_ranking_matrix_parallel(D_hd)
    rm_ld = compute_ranking_matrix_parallel(D_ld)

    # compute coranking matrix from_ranking matrix
    m = len(rm_hd)
    Q = np.zeros(rm_hd.shape, dtype="int16")
    for i in tqdm(range(m), desc="computing coranking matrix"):
        Q = populate_Q(Q, i, m, rm_hd, rm_ld)

    Q = Q[1:, 1:]
    return Q


# Area under curve from coranking matrix
# Code from https://github.com/ymlasu/CAMEL/blob/main/demo/eval_metrics.py
# See https://doi.org/10.48550/arXiv.2403.14813


@numba.njit(fastmath=True)
def qnx_crm(crm, k):
    """Average Normalized Agreement Between K-ary Neighborhoods (QNX)
    # QNX measures the degree to which an embedding preserves the local
    # neighborhood around each observation. For a value of K, the K closest
    # neighbors of each observation are retrieved in the input and output space.
    # For each observation, the number of shared neighbors can vary between 0
    # and K. QNX is simply the average value of the number of shared neighbors,
    # normalized by K, so that if the neighborhoods are perfectly preserved, QNX
    # is 1, and if there is no neighborhood preservation, QNX is 0.
    #
    # For a random embedding, the expected value of QNX is approximately
    # K / (N - 1) where N is the number of observations. Using RNX
    # (\code{rnx_crm}) removes this dependency on K and the number of
    # observations.
    #
    # @param crm Co-ranking matrix. Create from a pair of distance matrices with
    # \code{coranking_matrix}.
    # @param k Neighborhood size.
    # @return QNX for \code{k}.
    # @references
    # Lee, J. A., & Verleysen, M. (2009).
    # Quality assessment of dimensionality reduction: Rank-based criteria.
    # \emph{Neurocomputing}, \emph{72(7)}, 1431-1443.

    Python reimplmentation of code by jlmelville
    (https://github.com/jlmelville/quadra/blob/master/R/neighbor.R)
    """
    qnx_crm_sum = np.sum(crm[:k, :k])
    return qnx_crm_sum / (k * len(crm))


@numba.njit(fastmath=True)
def rnx_crm(crm, k):
    """Rescaled Agreement Between K-ary Neighborhoods (RNX)
    # RNX is a scaled version of QNX which measures the agreement between two
    # embeddings in terms of the shared number of k-nearest neighbors for each
    # observation. RNX gives a value of 1 if the neighbors are all preserved
    # perfectly and a value of 0 for a random embedding.
    #
    # @param crm Co-ranking matrix. Create from a pair of distance matrices with
    # \code{coranking_matrix}.
    # @param k Neighborhood size.
    # @return RNX for \code{k}.
    # @references
    # Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013).
    # Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in
    # dimensionality reduction based on similarity preservation.
    # \emph{Neurocomputing}, \emph{112}, 92-108.

    Python reimplmentation of code by jlmelville
    (https://github.com/jlmelville/quadra/blob/master/R/neighbor.R)
    """
    n = len(crm)
    return ((qnx_crm(crm, k) * (n - 1)) - k) / (n - 1 - k)


# @numba.njit(fastmath=True)
def rnx_auc_crm(crm, test_range=None):
    """Area Under the RNX Curve
    # The RNX curve is formed by calculating the \code{rnx_crm} metric for
    # different sizes of neighborhood. Each value of RNX is scaled according to
    # the natural log of the neighborhood size, to give a higher weight to smaller
    # neighborhoods. An AUC of 1 indicates perfect neighborhood preservation, an
    # AUC of 0 is due to random results.
    #
    # param crm Co-ranking matrix.
    # return Area under the curve.
    # references
    # Lee, J. A., Peluffo-Ordo'nez, D. H., & Verleysen, M. (2015).
    # Multi-scale similarities in stochastic neighbour embedding: Reducing
    # dimensionality while preserving both local and global structure.
    # \emph{Neurocomputing}, \emph{169}, 246-261.

    Python reimplmentation of code by jlmelville
    (https://github.com/jlmelville/quadra/blob/master/R/neighbor.R)
    """
    n = len(crm)
    num = 0
    den = 0

    qnx_crm_sum = 0
    rng = range(1, n - 2)
    # if test_range is not None:
    #    rng = test_range
    # for k in tqdm(rng):
    for k in range(1, n - 2):
        qnx_crm_sum += (
            np.sum(crm[(k - 1), :k]) + np.sum(crm[:k, (k - 1)]) - crm[(k - 1), (k - 1)]
        )
        qnx_crm = qnx_crm_sum / (k * len(crm))
        rnx_crm = ((qnx_crm * (n - 1)) - k) / (n - 1 - k)
        num += rnx_crm / k
        den += 1 / k
    return num / den
