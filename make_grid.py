import numpy as np

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
        matrix = np.random.rand(*shape)
    else:
        matrix = np.ones(shape)
    return matrix / matrix.sum()
