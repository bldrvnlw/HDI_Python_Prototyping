import torch
import numpy as np


from utils.base import knn_brute_force
from utils.base import pairwise_l2_distances
from utils.base import pairwise_l2_distances_full
from utils.base import rankdata_average_fast


class NNPointsTorch:
    """Cache expensively calculated nearest neighbor,
    distance or rank information in torch.tensor
    objects.

    The initial pointData NxD array may represent a high or low
    dimensional space.
    """

    def __init__(self, pointData: np.ndarray, torch_type=torch.float32):
        self.pointData = pointData
        self.tensor = torch.tensor(pointData, device="cuda", dtype=torch_type)
        self.nn = {}
        self.pair_l2_dist = None
        self.pair_l2_dist_full = None
        self.ranked = None

    def getNN(self, k=30):
        if k in self.nn:
            return self.nn[k]
        self.nn[k] = knn_brute_force(self.pointData, k)
        return self.nn[k]

    def get_pairwise_l2_distances(self):
        if self.pair_l2_dist is not None:
            return self.pair_l2_dist

        self.pair_l2_dist = pairwise_l2_distances(self.tensor)
        return self.pair_l2_dist

    def get_pairwise_ls_dist_full(self):
        if self.pair_l2_dist_full is not None:
            return self.pair_l2_dist_full

        self.pair_l2_dist_full = pairwise_l2_distances_full(self.tensor)
        return self.pair_l2_dist_full

    def get_rank(self):
        if self.ranked is not None:
            return self.ranked
        self.ranked = rankdata_average_fast(self.tensor)
        return self.ranked
