import numpy as np
from kp import Manager


class LinearProbabilityMatrix:
    def __init__(
        self,
        neighbours: np.ndarray[np.uint32],
        probabilities: np.ndarray[np.float32],
        indices: np.ndarray[np.int32],
    ):
        """
        Initialize the LinearProbabilityMatrix with neighbours and probabilities.

        Args:
            neighbours (np.ndarray): Array of neighbour indices.
            probabilities (np.ndarray): Array of probabilities corresponding to neighbours.
        """
        self.neighbours = neighbours
        self.probabilities = probabilities
        self.indices = indices


class PersistentBuffers:
    """
    Class to manage persistent buffers for shaders.
    This class is used to create and manage buffers that persist across shader invocations.
    """

    def __init__(
        self, mgr: Manager, num_points: int, prob_matrix: LinearProbabilityMatrix
    ):
        self.num_points = num_points
        self.mgr = mgr
        self.prob_matrix = prob_matrix
        self.__create_tensors()

    def __create_tensors(self):
        """
        Create all the tensor buffer with the specified sizes.
        """
        # These are initially zero tensors, they will be filled by the shaders.
        self.position_tensor = self.mgr.tensor(
            np.zeros((self.num_points, 2), dtype=np.float32)
        )
        self.interp_fields_tensor = self.mgr.tensor(
            np.zeros((self.num_points, 4), dtype=np.float32)
        )
        self.sumQ_tensor = self.mgr.tensor(
            np.zeros((self.num_points, 1), dtype=np.float32)
        )
        self.gradients_tensor = self.mgr.tensor(
            np.zeros((self.num_points, 2), dtype=np.float32)
        )
        # These contain the probability matrix information
        self.neighbours_tensor = self.mgr.tensor(self.prob_matrix.neighbours)
        self.probabilities_tensor = self.mgr.tensor(self.prob_matrix.probabilities)
        self.indices_tensor = self.mgr.tensor(self.prob_matrix.indices)
