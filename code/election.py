"""
This module implements subroutines for creating elections and handling their
data.
"""

import tools
import math
import tensorflow as tf
import scipy.special
from tqdm import tqdm

def get_vote_pcts(index, matrix_dim, demo):
    """
    Find the vote percentages for each demographic group,
    given the index of an associated probabilistic grid.

    index (int tuple): the index of the grid
    matrix_dim (int): the size of one dimension of the grid
    demo (dict): the demographics of the district

    return: a dict of the vote percentages for each demographic
    group
    """
    return {group: (tf.cast(index[num], tf.float32) + 0.5) / matrix_dim for num, group in enumerate(demo)}

def generate_random_election(candidates, demo, beta):
    """
    Generate a random election.

    candidates (string list): the candidates
    demo (dict): the demographics of the electorate
    beta (dict): the theoretical voting percentages of
    each demographic group, for each candidate

    return: a dictionary of candidates and the vote breakdowns by
    demographic group
    """
    # Set up the result dictionary
    num_groups = len(demo)
    result = {'a': (0, [0] * num_groups),
              'b': (0, [0] * num_groups),
              'c': (0, [0] * num_groups)}

    # Iterate through each demographic group
    for group_index, group in enumerate(demo):
        # Simulate each voter
        for voter in range(demo[group]):
            vote = np.random.choice(candidates, 1, beta[group])[0]
            prev_total, prev_breakdown = result[vote]
            prev_breakdown[group_index] += 1
            result[vote] = prev_total + 1, prev_breakdown
    return result

def get_coefficients(demo, observed):
    coeff_dict = {}
    observed_factorial = math.factorial(observed)

    for p in tools.permute_integer_partition(observed, len(demo)):
        # Assign the partitioned elements to groups
        partition = dict(zip(demo.keys(), p))

        factorial_list = tf.convert_to_tensor(
            scipy.special.factorial(p), dtype=float)
        coefficient = observed_factorial / tf.math.reduce_prod(factorial_list)

        coeff_dict[p] = tf.cast(coefficient, tf.float32)

    return coeff_dict
