"""
This module implements the Discrete Voter Model for ecological inference in 
Python 3.
"""

import numpy as np
import random

from tqdm import trange

import make_grid
import shift_weight
import expec_votes
import prob_votes

def dvm(n_iter, initial_grid, observed_votes, demo,
        shift_type='uniform', scoring_type='prob',
        epsilon=1):
    """
    Run the Metropolis-Hastings MCMC algorithm to sample the space
    of probabilistic demographic grids in the discrete
    voter model.

    n_iter (int): the number of iterations to run
    initial_grid (NumPy array): the probabilistic grid to start with
    observed_votes (int): the number of votes a candidate got in an election
    demo (dict): the demographics of the district
    shift_type (string): the type of update to apply to the grid. One of:
        1. uniform (default): add a uniform random variable to each cell
        then re-normalize
        2. shuffle: shuffle the matrix
        3. right: shift grid weight to the right
        4. left: shift grid weight to the left
    scoring_type (string): the type of scoring to use. One of:
        1. prob (default): score by the probability of a grid to produce
        the outcome
        2. expec: score by the difference in the outcome and the expectation
        of a grid
    epsilon (float): the value to increase the lateral grid shift by

    return: a dictionary of the best scoring grid, the highest score
    it received,
    and a list of all the grids explored and their scores
    """
    grid = initial_grid
    current_score = 0
    rejection_count = 0
    results = {'rejection_rate': None,
               'best_grid': grid,
               'best_score': current_score,
               'all_grids': [grid],
               'all_scores': [current_score]}

    # Iterate
    for _ in trange(n_iter, desc='met-hastings'):
        # Generate a candidate
        candidate = shift_weight(grid, shift_type=shift_type, epsilon=epsilon)

        # Score the candidate and accept or reject
        if scoring_type == 'prob':
            score = prob_votes(candidate, demo, observed_votes)

        else:
            expectation = expec_votes(candidate, demo)
            # Negate the score for expectation to be consistent with lower scores
            # being better
            score = -abs(observed_votes - expectation)

        # Accept if higher than the current score, or with that probability
        # if lower, and implicitly reject
        if score >= current_score or (score / current_score) > random.uniform(0, 1):
            grid = candidate
            results['all_grids'].append(grid)
            results['all_scores'].append(score)
            # Check the best score
            if score >= results['best_score']:
                results['best_score'] = score
                results['best_grid'] = grid
        else:
            rejection_count += 1

    results['rejection_rate'] = rejection_count / n_iter
    return results
