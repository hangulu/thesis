import numpy as np
from operator import mul
import functools
import random
import pymc3 as pm
import time

from itertools import chain, permutations

def dvm_king_evaluator(election, demo, beta, label, n_iter=1, met_iter=200):
    """
    Run and compare the results of the Discrete
    Voter Model and King's EI on generated
    election data.

    election (dict): the random election to
    evaluate on
    demo (dict): the demographic dictionary of
    a precinct
    label (string): the label of the experiment
    n_iter (int): the number of times to repeat
    the experiment
    met_iter (int): the number of iterations to use
    for Metropolis-Hastings

    return: a dictionary of the label and times and MSEs for
    the Discrete Voter Model and King's EI
    """
    results = {}
    dvm_total_time = 0
    dvm_total_mse = 0

    king_total_time = 0
    king_total_mse = 0

    temp_pcts_array = [pcts[0] for group, pcts in beta.items()]
    true_pcts = np.fromiter(temp_pcts_array, dtype=float)

    for _ in range(n_iter):
        # Get the observed votes for candidate a
        candidate_a_obs = random_election_1_1['a'][0]

        # Run Metropolis-Hastings and time it
        dvm_total_time -= time.time()
        initial_grid = make_grid(len(demo), 10)
        met_result = metropolis_hastings(met_iter, initial_grid, candidate_a_obs, demo, scoring_type='prob')
        dvm_total_time += time.time()

        # Find the best grid and output the result
        best_grid = met_result['best_grid']
        best_cell = get_most_probable_cell(met_result['best_grid'])
        vote_pcts = get_vote_pcts(best_cell, 10, demo)

        # Find the MSE of the result
        dvm_mse_array = np.fromiter(vote_pcts.values(), dtype=float)
        dvm_total_mse += mse(dvm_mse_array, true_pcts)

        # Run King's EI and time it, if at the right dimension
        if len(demo) > 2:
            continue
        king_demo = list(demo.values())[0] / 100
        king_cand_vote = candidate_a_obs / 100
        king_prec_pop = np.array([100])

        king_total_time -= time.time()
        king_model = ei_two_by_two(king_demo, king_cand_vote, king_prec_pop)
        with king_model:
            king_trace = pm.sample()
        king_total_time += time.time()

        # Find the MSE of the result
        king_mse_array = np.fromiter([king_trace.get_values('b_1').mean(),
                                      king_trace.get_values('b_2').mean()],
                                     dtype=float)

        king_total_mse += mse(king_mse_array, true_pcts)


    return {'label': label,
            'dvm_time': dvm_total_time / n_iter,
            'dvm_mse': dvm_total_mse / n_iter,
            'king_time': king_total_time / n_iter,
            'king_mse': king_total_mse / n_iter}
