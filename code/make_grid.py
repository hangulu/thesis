import numpy as np
import tensorflow as tf

class Grid:
    def __init__(self, num_groups, matrix_size, random_init=True):
        self.num_groups = num_groups
        self.shape = tuple([])

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
