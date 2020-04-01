"""
This module implements general tools for the Discrete Voter Model for
ecological inference.
"""

import numpy as np
import tensorflow as tf

from itertools import chain, permutations


def integer_partition(n, k, min_size=0):
    """
    Partition an integer.

    n (int): the integer to partition
    k (int): the number of elements in a partition
    min_size (int): the minimum size of an element
    in the partition

    return: a generator of partitions as tuples
    """
    if k < 1:
        return
    if k == 1:
        if n >= min_size:
            yield (n,)
        return
    for i in range(min_size, n // k + 1):
        for result in integer_partition(n - i, k - 1, i):
            yield (i,) + result


def permute_integer_partition(n, k, min_size=0):
    """
    Partition an integer, with all permutations

    n (int): the integer to partition
    k (int): the number of elements in a partition
    min_size (int): the minimum size of an element
    in the partition

    return: a generator of all permutations of partitions as tuples
    """
    return chain.from_iterable(set(permutations(p)) for p in integer_partition(n, k, min_size))


def mse(a, b):
    """
    Find the mean squared error between
    two one-dimensional NumPy arrays.

    a (NumPy array-like): the first array
    b (NumPy array-like): the second array

    return: a float of the MSE
    """
    return np.mean((a - b) ** 2)


def find_last_finite(array, default=0):
    rev_array = array[::-1]
    index = np.argmax(tf.math.is_finite(rev_array))
    last_finite = rev_array[index]

    if tf.math.is_inf(last_finite):
        return default
    return last_finite


@tf.function
def reduce_average(x, weights):
    numerator = tf.reduce_sum(tf.math.multiply(x, weights))
    return numerator / tf.math.reduce_sum(weights)


def normalize(phc):
    """
    Normalize a probabilistic hypercube (PHC) to have values between
    0 and 1 and a sum of 1 to reflect probability.

    phc (Tensor): the Tensor representation of a PHC

    return: a normalized Tensor
    """
    numerator = tf.math.subtract(phc, tf.math.reduce_min(phc))
    denominator = tf.math.maximum(1e-12,
        tf.math.subtract(tf.math.reduce_max(phc), tf.math.reduce_min(phc)))
    return tf.math.divide(numerator, denominator)


def prob_normalize(phc):
    """
    Normalize a probabilistic hypercube (PHC) to have values between
    0 and 1 and a sum of 1 to reflect probability.

    phc (Tensor): the Tensor representation of a PHC

    return: a normalized Tensor
    """
    quotient = normalize(phc)

    return quotient / tf.math.reduce_sum(quotient)


def get_most_probable_cell(phc):
    """
    Find the most probable cell in a PHC.

    phc (Tensor): the Tensor representation of a PHC

    return: the index of the most probable cell
    """
    return tf.unravel_index(np.argmax(phc), phc.shape)
