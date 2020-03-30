"""
This module implements the expec_votes subroutine for the Discrete Voter Model
for ecological inference.
"""

import functools
import numpy as np
import tensorflow as tf

import tools


def get_vote_outcome(flat_index, phc_shape, demo):
    """
    Find the vote outcome of a given election
    for a candidate, with a given PHC.

    flat_index (int): the flat index of the selected cell
    phc_shape (Tensor): the shape of the PHC
    demo (dict): the demographics of the district

    return: the expectation of the vote outcome for that cell
    """
    # Find the corresponding index
    index = tf.unravel_index(flat_index, phc_shape)
    matrix_dim = phc_shape[0]

    # Calculate the vote outcomes of the cell
    demo_pop = np.fromiter(demo.values(), dtype=np.float32)
    demo_vote_pcts = index / matrix_dim

    vote_outcome = tf.math.reduce_sum(demo_pop * tf.cast(demo_vote_pcts, tf.float32))

    return vote_outcome


@tf.function
def expec_votes(phc, demo, rwm=False):
    """
    Find the expectation of the vote outcome
    for a candidate, with a given PHC.

    phc (Tensor): the Tensor representation of a PHC
    demo (dict): the demographics of the district
    rwm (bool): whether this function serves the RWM or HMC kernel

    return: the expectation for the vote outcomes over the PHC
    """
    if rwm:
        flat_phc = tf.reshape(phc, [-1])
    else:
        normalized_phc = tools.prob_normalize(phc)
        flat_phc = tf.reshape(normalized_phc, [-1])

    get_vote_outcome_partial = functools.partial(
        get_vote_outcome,
        phc_shape=phc.shape,
        demo=demo)

    outcomes = tf.map_fn(get_vote_outcome_partial,
        tf.range(tf.size(flat_phc)), dtype=tf.float32)

    return tools.reduce_average(outcomes, flat_phc)


@tf.function
def prob_from_expec(phc, demo, observed, rwm=False):
    """
    Derive a probability of a PHC by comparing its expectation to
    the observed number of votes.

    phc (Tensor): the Tensor representation of a PHC
    demo (dict): the demographics of the district
    observed (int): the observed number of votes the candidate received
    rwm (bool): whether this function serves the RWM or HMC kernel

    return: the complement of the sigmoid function applied to the
    difference in expectation
    """
    return tf.math.log(1. - tf.math.sigmoid(
        tf.math.abs(observed - expec_votes(phc, demo, rwm=rwm))))
