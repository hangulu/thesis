import numpy as np
import tensorflow as tf

class Grid:
    """
    A probabilistic hypercube (PHC).

    Attributes:
        num_groups (int): the number of groups to be represented
        shape (int tuple): the dimensions of the PHC
        grid (Tensor): the Tensor representation of a PHC
        dtype (TensorFlow DType): the datatype of the PHC
    """
    def __init__(self, num_groups, matrix_size, grid=None, random_init=True):
        """
        Initializes a probabilistic hypercube.

        num_groups (int): the number of groups to be represented
        matrix_size (int): the dimensions of the PHC
        random (bool): whether the PHC should be initialized uniformly or
            randomly
        """
        if grid:
            self.num_groups = len(grid.shape)
            self.shape = grid.shape
            self.grid = grid
        else:
            self.num_groups = num_groups
            self.shape = tuple([matrix_size] * self.num_groups)

            if random_init:
                matrix = tf.random.uniform(self.shape)
            else:
                matrix = tf.ones(shape)

            self.grid = matrix / tf.math.reduce_sum(matrix)

        self.dtype = self.grid.dtype

    def __repr__(self):
        return f"Probabilistic Hypercube: shape={self.shape}"

    def normalize(self):
        self.grid = self.grid / tf.math.reduce_sum(self.grid)


def make_grid(num_groups, matrix_size, random=True):
    """
    Create a probabilistic hypercube.

    num_groups (int): the number of groups to be represented
    matrix_size (int): the dimensions of the matrix
    random (bool): whether the hypercube should be initialized uniformly
    or randomly

    return: a probabilistic hypercube
    """
    shape = tuple([matrix_size] * num_groups)
    if random:
        matrix = tf.random.uniform(shape)
    else:
        matrix = tf.ones(shape)
    return matrix / tf.math.reduce_sum(matrix)
