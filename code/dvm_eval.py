"""
This module evaluates the Discrete Voter Model for ecological inference and
King's Ecological Inference.
"""

import logging
import numpy as np
import pandas as pd
import pymc3 as pm
import time
from tqdm.autonotebook import trange, tqdm

import dvm
import elect
import kings_ei as kei
import phc
import tools


def dvm_evaluator(election, label, candidate=None, phc_granularity=10,
                  hmc=False, expec_scoring=False, burn_frac=0.3,
                  n_steps=200, n_iter=1, verbose=False):
    """
    Evaluate the accuracy and speed of the Discrete Voter
    Model.

    election (Election): the election to evaluate on
    label (string): the label of the experiment
    candidate (string): the candidate to analyze
    phc_granularity (int): the size of a dimension of the PHC
    hmc (bool): whether to use the HMC or RWM kernel
    expec_scoring (bool): whether to score by:
        1. the probability of a PHC to produce the outcome
        (False, default)
        2. the difference in the outcome and the PHC's expectation
        (True)
    burn_frac (float): the fraction of MCMC iterations to burn
    n_steps (int): the number of steps to run the MCMC for
    n_iter (int): the number of times to repeat the experiment
    verbose (bool): whether to display loogging and progress bars

    return: a dictionary of the label, times and MSEs for
    the Discrete Voter Model
    """
    total_time = 0

    total_mle_phc_mse = 0
    total_mean_phc_mse = 0

    initial_phc = phc.make_phc(election.num_demo_groups, phc_granularity)

    for _ in trange(n_iter, desc="Experiment progress", leave=verbose):
        # Get the observed votes for the desired candidate
        if not candidate:
            candidate = election.candidates[0]

        cand_obs_votes = {}
        for prec in election.precincts:
            cand_obs_votes[prec] = election.vote_totals[prec][candidate]

        # Run the MCMC with the specified kernel
        total_time -= time.time()

        if hmc:
            chain_results = dvm.hmc(n_steps, burn_frac, initial_phc,
                                    election.dpp, cand_obs_votes,
                                    expec_scoring=expec_scoring,
                                    verbose=verbose)
        else:
            chain_results = dvm.rwm(n_steps, burn_frac, initial_phc,
                                    election.dpp, cand_obs_votes,
                                    expec_scoring=expec_scoring,
                                    verbose=verbose)

        total_time += time.time()

        # Find the best PHC
        mle_phc = dvm.chain_mle(chain_results)[0]
        mean_phc = dvm.mean_phc(chain_results)

        # Find the most probable cell in the PHC
        best_cell_mle_phc = tools.get_most_probable_cell(mle_phc)
        best_cell_mean_phc = tools.get_most_probable_cell(mean_phc)

        vote_pcts_mle_phc = elect.get_vote_pcts(best_cell_mle_phc, phc_granularity, election.dpp)
        vote_pcts_mean_phc = elect.get_vote_pcts(best_cell_mean_phc, phc_granularity, election.dpp)

        # Find the MSE of the vote percentages if applicable
        if election.mock:
            # Get the demographic voting probabilities for the desired
            # candidate
            for prec, dvp in election.dvp.items():
                dvp_pct = np.fromiter([pcts[candidate] for group, pcts in dvp.items()], dtype=float)

                mle_phc_mse_array = np.fromiter(vote_pcts_mle_phc[prec].values(), dtype=float)
                mean_phc_mse_array = np.fromiter(vote_pcts_mean_phc[prec].values(), dtype=float)

                total_mle_phc_mse += tools.mse(mle_phc_mse_array, dvp_pct)
                total_mean_phc_mse += tools.mse(mean_phc_mse_array, dvp_pct)

    return {'label': label,
            'time': total_time / n_iter,
            'mle_phc_mse': total_mle_phc_mse / n_iter,
            'mean_phc_mse': total_mean_phc_mse / n_iter}


def batch_dvm_eval(experiments, n_steps, n_iter, phc_granularity, hmc=True,
                   expec_scoring=True):
    """
    Perform a batch evaluation of a set of experiments
    of the Discrete Voter Model.

    experiments (List of Elections): the list of elections to be evaluated
    n_steps (int): the number of steps to run the MCMC for
    n_iter (int): the number of times to repeat the experiment
    phc_granularity (int): the size of a dimension of the PHC
    hmc (bool): whether to use the HMC or RWM kernel
    expec_scoring (bool): whether to score by:
        1. the probability of a PHC to produce the outcome
        (False, default)
        2. the difference in the outcome and the PHC's expectation
        (True)

    return: a Pandas DataFrame of the result of the experiments
    """
    exper_results = []
    for election, label in tqdm(experiments, desc="Elections progress"):
        eval_result = dvm_evaluator(election,
                                    label,
                                    phc_granularity=phc_granularity,
                                    hmc=hmc,
                                    expec_scoring=expec_scoring,
                                    n_steps=n_steps,
                                    n_iter=n_iter)
        exper_results.append(eval_result)

    df = pd.DataFrame(exper_results)
    df['phc_granularity'] = phc_granularity
    df['n_steps'] = n_steps

    return df


# Suppress logging for pyMC3
pymc3_logger = logging.getLogger('pymc3')
pymc3_logger.setLevel(logging.CRITICAL)


def kei_evaluator(election, label, candidate=None, n_steps=500, n_iter=1,
                  verbose=False):
    """
    Evaluate the accuracy and speed of King's Ecological Inference
    method.

    election (Election): the election to evaluate on
    candidate (string): the candidate to analyze
    label (string): the label of the experiment
    n_steps (int): the number of steps to run the MCMC for
    n_iter (int): the number of times to repeat the experiment
    verbose (bool): whether to display loogging and progress bars

    return: a dictionary of the label, times and MSEs for
    the Discrete Voter Model
    """
    # Check if King's EI can be used
    if len(election.demo) > 2:
        raise ValueError("King's Ecological Inference method only works in the 2x2 case.")

    total_time = 0
    total_mse = 0

    for _ in trange(n_iter, desc="Experiment progress", leave=verbose):
        # Get the observed votes for the desired candidate
        if not candidate:
            candidate = election.candidates[0]
        cand_obs_votes = election.vote_totals[candidate]

        prec_demos = [election.demo]

        # Run King's EI and time it
        total_time -= time.time()

        king_model = kei.eco_inf(prec_demos, cand_obs_votes)
        with king_model:
            king_trace = pm.sample(draws=n_steps, progressbar=False)

        total_time += time.time()

        # Find the MSE of the vote percentages if applicable
        if election.mock:
            # Get the demographic voting probabilities for the first candidate
            dvp_pcts = np.fromiter([pcts[candidate] for group, pcts in election.dvp.items()], dtype=float)

            king_mse_array = np.fromiter([king_trace.get_values('b_1').mean(),
                                      king_trace.get_values('b_2').mean()],
                                     dtype=float)

            total_mse += tools.mse(king_mse_array, dvp_pcts)

    return {'label': label,
            'time': total_time / n_iter,
            'mse': total_mse / n_iter}


def batch_kei_eval(experiments, n_steps, n_iter):
    """
    Perform a batch evaluation of a set of experiments
    of King's Ecological Inference method.

    experiments (List of Elections): the list of elections to be evaluated
    n_steps (int): the number of steps to run the MCMC for
    n_iter (int): the number of times to repeat the experiment

    return: a Pandas DataFrame of the result of the experiments
    """
    exper_results = []
    for election, label in tqdm(experiments, desc="Elections progress"):
        eval_result = kei_evaluator(election,
                                    label,
                                    n_steps=n_steps,
                                    n_iter=n_iter)
        exper_results.append(eval_result)

    df = pd.DataFrame(exper_results)
    df['n_steps'] = n_steps

    return df
