"""
This module implements general tools for the Discrete Voter Model for
ecological inference.
"""

from itertools import chain, permutations
import tensorflow as tf
import numpy as np

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

def normalize(grid):
    """
    Normalize a grid to have values between
    0 and 1 and a sum of 1 to reflect probability.

    grid (Tensor): the Tensor representation of a PHC

    return: a normalized Tensor
    """
    numerator = tf.math.subtract(grid, tf.math.reduce_min(grid))
    denominator = tf.math.maximum(1e-12,
        tf.math.subtract(tf.math.reduce_max(grid), tf.math.reduce_min(grid)))
    return tf.math.divide(numerator, denominator)


def prob_normalize(grid):
    """
    Normalize a grid to have values between
    0 and 1 and a sum of 1 to reflect probability.

    grid (Tensor): the Tensor representation of a PHC

    return: a normalized Tensor
    """
    quotient = normalize(grid)

    return quotient / tf.math.reduce_sum(quotient)
