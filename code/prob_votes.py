"""
This module implements the prob_votes subroutine for the Discrete Voter Model
for ecological inference.
"""

import numpy as np
import math
import tensorflow as tf
from tools import integer_partition, get_vote_pcts, permute_integer_partition
import scipy.special
from tqdm import tqdm, trange
from election import get_coefficients

@tf.function
def get_vote_probability(flat_index, grid, demo, coeff_dict):
    """
    Find the probability of a grid's cell producing a
    vote outcome of a given election for a candidate,
    with a given probabilistic grid.

    flat_index (int): the flat index of the selected cell
    grid (NumPy array): the probabilistic grid for the precinct
    and candidate
    demo (dict): the demographics of the district
    coeff_dict (dict): the binomial coefficients for each partition

    return: the probability that a cell produced the observed outcome
    """
    # Find the corresponding index
    index = tf.unravel_index(flat_index, grid.shape)
    matrix_dim = grid.shape[0]

    # Find the vote percentages for each demographic group
    vote_pcts = get_vote_pcts(index, matrix_dim, demo)

    # Find the probability of the outcome
    total_prob = 0

    # Go through the possible partitions of the vote outcome, by group
    for p, coeff in coeff_dict.items():
        # Assign the partitioned elements to groups
        partition = dict(zip(demo.keys(), p))

        # Find the probability of seeing that outcome
        prob = 1
        for group in demo:
            group_pct = vote_pcts[group]
            candidate_group_num = partition[group]
            total_group_num = demo[group]

            # Check if this is feasible with the current demographic
            # If infeasible, record the infeasibility and continue
            if candidate_group_num > total_group_num:
                prob *= 0
                break

            group_factor = (group_pct ** candidate_group_num) * ((1 - group_pct) ** (total_group_num - candidate_group_num))

            prob *= group_factor

        total_prob += prob * coeff

    return total_prob


def prob_votes(grid, demo, observed, coeff_dict):
    """
    Find the probability that a grid produced
    the observed number of votes that a candidate
    received in a given election, with a given
    probabilistic grid.

    grid (NumPy array): the probabilistic grid for the precinct
    and candidate
    demo (dict): the demographics of the district
    observed (int): the observed number of votes the candidate received
    coeff_dict (dict): the binomial coefficients for each partition

    return: the probability that a grid produced the observed outcomes
    """
    flat_grid = tf.reshape(grid, [-1])
    probs = enumerate(flat_grid)
    grid_prob = 1
    for flat_index, prob in probs:
        vote_prob = get_vote_probability(tf.constant(flat_index), grid, demo, coeff_dict)
        grid_prob *= (1 - (vote_prob * prob))

    return tf.math.log(1 - grid_prob)
