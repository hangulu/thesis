"""
This module implements the expec_votes subroutine for the Discrete Voter Model
for ecological inference.
"""

import numpy as np
import tensorflow as tf

def get_vote_outcome(flat_index, grid, demo, print_stats=False):
    """
    Find the vote outcome of a given election
    for a candidate, with a given probabilistic
    grid.

    flat_index (int): the flat index of the selected cell
    grid (NumPy array): the probabilistic grid for the precinct
    and candidate
    demo (dict): the demographics of the district
    print_stats (boolean): whether to print the statistics

    return: the expectation of the vote outcome for that cell
    """
    # Find the corresponding index
    index = np.unravel_index(flat_index, grid.shape)
    matrix_dim = grid.shape[0]

    if print_stats:
        print(f"The index is {index}.")

    # Calculate the vote outcomes given the cell selected

    vote_outcome = np.zeros(len(demo))

    for num, group in enumerate(demo):
        # Find the probabilities the cell represents for each group
        pct = index[num] / matrix_dim

        if print_stats:
            print(f"{int(pct * 100)}% of the {group} population voted for this candidate.")

        vote_outcome[num] += demo[group] * pct

    return np.sum(vote_outcome)


def expec_votes(grid, demo, print_all_stats=False):
    """
    Find the expectation of the vote outcome
    for a candidate, with a given probabilistic
    grid.

    grid (NumPy array): the probabilistic grid for the precinct
    and candidate
    demo (dict): the demographics of the district

    return: the expectation for the vote outcomes over the grid
    """
    probs = tf.reshape(grid, [-1])
    outcomes = []
    for flat_index, prob in enumerate(probs):
        # Set up printing
        print_stats = False
        if flat_index % 10 == 0 and print_all_stats:
            print_stats = True
        outcomes.append(get_vote_outcome(flat_index, grid, demo, print_stats))

    return np.average(outcomes, weights=probs)
