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
