"""
This module implements subroutines for creating elections and handling their
data.
"""

import tools
import math
import tensorflow as tf
import scipy.special
from tqdm import tqdm

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
