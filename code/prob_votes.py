"""
This module implements the prob_votes subroutine for the Discrete Voter Model
for ecological inference.
"""

import functools
import math
import scipy.special
import tensorflow as tf

import elect
import tools


def get_coefficients(demo, observed):
    """
    Get the binomial coefficients for calculating the probability of a PHC
    producing an election result.

    demo (dict): the demographics of the district
    observed (int): the number of votes a candidate got in an election

    return: a Python dictionary containing the integer partitions and their
    binomial coefficients
    """
    coeff_dict = {}
    observed_factorial = math.factorial(observed)

    for p in tools.permute_integer_partition(observed, len(demo)):
        factorial_list = tf.convert_to_tensor(
            scipy.special.factorial(p), dtype=float)
        coefficient = observed_factorial / tf.math.reduce_prod(factorial_list)

        coeff_dict[p] = tf.cast(coefficient, tf.float32)

    return coeff_dict


@tf.function
def get_vote_probability(flat_index, phc, demo, coeff_dict):
    """
    Find the probability of a PHC's cell producing a
    vote outcome of a given election for a candidate,
    with a given PHC.

    flat_index (int): the flat index of the selected cell
    phc (Tensor): the Tensor representation of a PHC
    demo (dict): the demographics of the district
    coeff_dict (dict): the binomial coefficients for each partition

    return: the probability that a PHC's cell produced the observed outcome
    """
    # Find the corresponding index
    index = tf.unravel_index(flat_index, phc.shape)
    matrix_dim = phc.shape[0]

    # Find the vote percentages for each demographic group
    vote_pcts = elect.get_vote_pcts(index, matrix_dim, demo)

    total_prob = [0] * len(coeff_dict)

    # Go through the possible partitions of the vote outcome, by group
    for index, (p, coeff) in enumerate(coeff_dict.items()):
        # Assign the partitioned elements to groups
        partition = dict(zip(demo.keys(), p))

        # Find the probability of seeing that outcome
        group_factors = [0.] * len(demo)

        for num, group in enumerate(demo):
            group_pct = vote_pcts[group]
            candidate_group_num = partition[group]
            total_group_num = demo[group]

            # Check if this is feasible with the current demographic
            # If infeasible, record the infeasibility and continue
            if candidate_group_num > total_group_num:
                break

            group_factor_1 = tf.math.pow(group_pct, candidate_group_num)
            group_factor_2 = tf.math.pow(1 - group_pct, total_group_num - candidate_group_num)
            group_factors[num] = tf.math.multiply(group_factor_1, group_factor_2)

        total_prob[index] = tf.math.multiply(tf.math.reduce_prod(group_factors), coeff)

    return tf.math.reduce_sum(total_prob)


@tf.function
def prob_votes(phc, demo, observed, coeff_dict, rwm=False):
    """
    Find the probability that a PHC produced
    the observed number of votes that a candidate
    received in a given election, with a given
    PHC.

    phc (Tensor): the Tensor representation of a PHC
    demo (dict): the demographics of the district
    observed (int): the observed number of votes the candidate received
    coeff_dict (dict): the binomial coefficients for each partition
    rwm (bool): whether this function serves the RWM or HMC kernel

    return: the probability that a PHC produced the observed outcomes
    """
    if rwm:
        flat_phc = tf.reshape(phc, [-1])
    else:
        normalized_phc = tools.prob_normalize(phc)
        flat_phc = tf.reshape(normalized_phc, [-1])

    get_vote_prob_partial = functools.partial(
        get_vote_probability,
        phc=normalized_phc,
        demo=demo,
        coeff_dict=coeff_dict)

    vote_prob = tf.map_fn(get_vote_prob_partial,
        tf.range(tf.size(flat_phc)), dtype=tf.float32)

    phc_prob_complement = tf.math.reduce_prod(
        1 - tf.math.multiply(vote_prob, flat_phc))

    return tf.math.log(1 - phc_prob_complement)
