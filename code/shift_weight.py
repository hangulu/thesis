"""
This module implements the shift_weight subroutine for the Discrete Voter Model
for ecological inference.
"""

import numpy as np

def right_shift(grid, epsilon):
    """
    Shift weight to the right in a probabilistic hypercube.

    grid (NumPy array): the probabilistic hypercube to be shifted
    epsilon (float): the value to increase the right hypercube shift by

    return: a probabilistic hypercube
    """
    grid_shift = (1 / grid.size) * epsilon
    rolled = np.roll(grid, 1, axis=1)
    return (grid + (rolled + grid_shift) - (grid - grid_shift)) / 3


def left_shift(grid, epsilon):
    """
    Shift weight to the left in a probabilistic hypercube.

    grid (NumPy array): the probabilistic hypercube to be shifted
    epsilon (float): the value to increase the left hypercube shift by

    return: a probabilistic hypercube
    """
    grid_shift = (1 / grid.size) * epsilon
    rolled = np.roll(grid, -1, axis=1)
    return (grid + (rolled + grid_shift) - (grid - grid_shift)) / 3


def shuffle_grid(grid):
    """
    Shuffle a probabilistic hypercube.

    grid (NumPy array): the probabilistic hypercube to be shuffled

    return: a probabilistic hypercube
    """
    np.random.shuffle(grid)
    return grid


def add_single_uniform_to_grid(grid):
    """
    Add a single uniform random variable to each cell then re-normalize.

    grid (NumPy array): the probabilistic hypercube to be shifted

    return: a probabilistic hypercube
    """
    new_grid = grid + np.random.uniform()
    return new_grid / new_grid.sum()


def add_uniform_to_grid(grid):
    """
    Add a uniform random hypercube to the hypercube then re-normalize.

    grid (NumPy array): the probabilistic hypercube to be shifted

    return: a probabilistic hypercube
    """
    new_grid = grid + np.random.rand(*grid.shape)
    return new_grid / new_grid.sum()


def shift_weight(grid, shift_type="uniform", epsilon=1):
    """
    Shift the weight in a probabilistic hypercube.

    grid (NumPy array): the probabilistic hypercube to be perturbed
    shift_type (string): the type of shift to be done. One of:
        1. uniform (default): add a uniform random hypercube to the hypercube
        then re-normalize
        2. single_uniform: add a single uniform random variable to each cell
        then re-normalize
        3. shuffle: shuffle the hypercube
        4. right: shift hypercube weight to the right
        5. left: shift hypercube weight to the left
    epsilon (float): the value to increase the lateral hypercube shift by

    return: a probabilistic hypercube
    """
    if shift_type == "shuffle":
        return shuffle_grid(grid)
    elif shift_type == "right":
        return right_shift(grid, epsilon)
    elif shift_type == "left":
        return left_shift(grid, epsilon)
    elif shift_type == "single_uniform":
        return add_single_uniform_to_grid(grid)
    else:
        return add_uniform_to_grid(grid)
