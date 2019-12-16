"""
This module implements tools for the Discrete Voter Model for ecological
inference.
"""

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
    return {group: (index[num] + 0.5) / matrix_dim for num, group in enumerate(demo)}

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

def mse(a, b):
    """
    Find the mean squared error between
    two one-dimensional NumPy arrays.

    a (NumPy array-like): the first array
    b (NumPy array-like): the second array

    return: a float of the MSE
    """
    return np.mean((a - b) ** 2)
