"""
This module creates probabilistic hypercubes for the Discrete Voter Model for ecological inference.
"""

import tensorflow as tf

def make_phc(num_groups, matrix_size, random=True):
    """
    Create a probabilistic hypercube (PHC).

    num_groups (int): the number of groups to be represented
    matrix_size (int): the dimensions of the PHC
    random (bool): whether the PHC should be initialized uniformly
    or randomly

    return: a probabilistic hypercube
    """
    shape = tuple([matrix_size] * num_groups)
    if random:
        matrix = tf.random.uniform(shape)
    else:
        matrix = tf.ones(shape)
    return matrix / tf.math.reduce_sum(matrix)
