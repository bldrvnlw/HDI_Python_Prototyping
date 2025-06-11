import numpy as np
from enum import IntEnum
from kp import Manager, Tensor


class ShaderBuffers(IntEnum):
    """
    Enum for shader buffer types.
    This enum defines the different types of buffers used in shaders.
    This is more of an aide memorie (kp will deal with buffer numbers,
    Perhaps the .name will be actually useful
    """

    POSITION = (0,)  # float: num_points x 2 (x,y)
    INTERP_FIELDS = (1,)  # float: num_points x 4 (x,y,dx,dy)
    SUM_Q = (2,)  # float: 1 (sum of Q)
    NEIGHBOUR = (3,)  # uint32: num_points x nn (neighbours)
    PROBABILITIES = (4,)  # float: num_points x nn (probabilities)
    INDEX = (5,)  # int32: 2 x num_points (start index, nn)
    GRADIENTS = (6,)  # float: num_points x 2 (gradients)
    PREV_GRADIENTS = (7,)  # float: num_points x 2 (previous gradients)
    GAIN = (8,)  # float: num_points x 2 (gain)
    BOUNDS = (9,)  # float: 1 x 4 (min, max)


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
            indices (np.ndarray): Array of indices for the neighbours.
        """
        self.neighbours = neighbours
        self.probabilities = probabilities
        self.indices = indices


class PersistentTensors:
    """
    Class to manage persistent tensor buffers for shaders.
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
        self.tensor_map = {
            ShaderBuffers.POSITION: self.mgr.tensor(
                np.zeros((self.num_points, 2), dtype=np.float32)
            ),
            ShaderBuffers.INTERP_FIELDS: self.mgr.tensor(
                np.zeros((self.num_points, 4), dtype=np.float32)
            ),
            ShaderBuffers.SUM_Q: self.mgr.tensor(np.zeros((1,), dtype=np.float32)),
            ShaderBuffers.NEIGHBOUR: self.mgr.tensor_t(self.prob_matrix.neighbours),
            ShaderBuffers.PROBABILITIES: self.mgr.tensor(
                self.prob_matrix.probabilities
            ),
            ShaderBuffers.INDEX: self.mgr.tensor_t(self.prob_matrix.indices),
            ShaderBuffers.GRADIENTS: self.mgr.tensor(
                np.zeros((self.num_points, 2), dtype=np.float32)
            ),
            ShaderBuffers.PREV_GRADIENTS: self.mgr.tensor(
                np.zeros((self.num_points, 2), dtype=np.float32)
            ),
            ShaderBuffers.GAIN: self.mgr.tensor(
                np.ones((self.num_points, 2), dtype=np.float32)
            ),
            ShaderBuffers.BOUNDS: self.mgr.tensor(
                np.zeros((1, 4), dtype=np.float32)  # min and max bounds
            ),
        }

    def get_tensor(self, buffer_type: ShaderBuffers) -> Tensor:
        """
        Get the tensor for the specified buffer type.

        Args:
            buffer_type (ShaderBuffers): The type of buffer to retrieve.

        Returns:
            kp.Tensor: The tensor corresponding to the specified buffer type.
        """
        return self.tensor_map.get(buffer_type, None)

    def set_tensor_data(self, buffer_type: ShaderBuffers, data: np.ndarray) -> None:
        """
        Set the data for the specified tensor buffer.

        Args:
            buffer_type (ShaderBuffers): The type of buffer to set.
            data (np.ndarray): The data to set in the tensor.
        """
        tensor = self.get_tensor(buffer_type)
        if tensor is not None:
            tensor.data()[:] = data.reshape(tensor.data().shape)
        else:
            raise ValueError(f"Buffer type {buffer_type} not found.")

    def get_tensor_data(self, buffer_type: ShaderBuffers) -> np.ndarray:
        """
        Get the data from the specified tensor buffer.

        Args:
            buffer_type (ShaderBuffers): The type of buffer to retrieve data from.

        Returns:
            np.ndarray: The data contained in the specified tensor buffer.
        """
        tensor = self.get_tensor(buffer_type)
        if tensor is not None:
            return tensor.data()
        else:
            raise ValueError(f"Buffer type {buffer_type} not found.")
