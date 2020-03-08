"""
This module implements the expec_votes subroutine for the Discrete Voter Model
for ecological inference.
"""

import numpy as np
import tensorflow as tf
import functools
import tools

def get_vote_outcome(flat_index, grid_shape, demo):
    """
    Find the vote outcome of a given election
    for a candidate, with a given probabilistic
    grid.

    flat_index (int): the flat index of the selected cell
    grid_shape (Tensor): the shape of the PHC
    demo (dict): the demographics of the district

    return: the expectation of the vote outcome for that cell
    """
    # Find the corresponding index
    index = tf.unravel_index(flat_index, grid_shape)
    matrix_dim = grid_shape[0]

    # Calculate the vote outcomes of the cell
    demo_pop = np.fromiter(demo.values(), dtype=np.float32)
    demo_vote_pcts = index / matrix_dim

    vote_outcome = tf.math.reduce_sum(demo_pop * tf.cast(demo_vote_pcts, tf.float32))

    return vote_outcome

@tf.function
def expec_votes(grid, demo):
    """
    Find the expectation of the vote outcome
    for a candidate, with a given probabilistic
    grid.

    grid (Tensor): the Tensor representation of a PHC
    demo (dict): the demographics of the district

    return: the expectation for the vote outcomes over the grid
    """
    normalized_grid = tools.normalize(grid)
    flat_grid = tf.reshape(grid, [-1])

    get_vote_outcome_partial = functools.partial(
        get_vote_outcome,
        grid_shape=grid.shape,
        demo=demo)

    outcomes = tf.map_fn(get_vote_outcome_partial,
        tf.range(tf.size(flat_grid)), dtype=tf.float32)

    return tools.reduce_average(outcomes, flat_grid)

@tf.function
def prob_from_expec(grid, demo, observed):
    """
    Derive a probability of a PHC by comparing its expectation to
    the observed number of votes.

    grid (Tensor): the Tensor representation of a PHC
    demo (dict): the demographics of the district
    observed (int): the observed number of votes the candidate received

    return: the complement of the sigmoid function applied to the
    difference in expectation
    """
    return tf.math.log(1. - tf.math.sigmoid(
        tf.math.abs(observed - expec_votes(grid, demo))))
