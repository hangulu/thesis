"""
This module implements King's Ecological Inference for the 2x2 case.
The following gist was used as a reference:
https://gist.github.com/ColCarroll/9fb0e6714dc0369acf6549cededcc875
"""

import numpy as np
import pymc3 as pm


def eco_inf(prec_demos, first_cand_obs_votes, lmbda=0.5):
    """
    Run King's Ecological Inference method for the 2x2 case
    (2 demographic groups, and 2 candidates).

    prec_demos (list of dicts): the demographics of the precincts
    first_cand_obs_votes (NumPy array): the number of people in each
    precinct who voted for the first candidate
    lmbda (float): the hyperparameter for the Exponential distributions

    return: the probabilistic model
    """
    # Convert the demographics of the precincts to a NumPy array
    total_pop = np.array([sum(demo.values()) for demo in prec_demos])

    # Find the percentage of people in the first demographic group in the
    # precincts
    first_group = list(prec_demos[0].keys())[0]
    demo_pcts = np.array([demo[first_group] for demo in prec_demos]) / total_pop

    # Find the number of precincts
    p = total_pop.size
    with pm.Model() as model:
        c_1 = pm.Exponential('c_1', lmbda)
        d_1 = pm.Exponential('d_1', lmbda)
        c_2 = pm.Exponential('c_2', lmbda)
        d_2 = pm.Exponential('d_2', lmbda)

        b_1 = pm.Beta('b_1', alpha=c_1, beta=d_1, shape=p)
        b_2 = pm.Beta('b_2', alpha=c_2, beta=d_2, shape=p)

        theta = demo_pcts * b_1 + (1 - demo_pcts) * b_2
        Tprime = pm.Binomial('Tprime', n=total_pop , p=theta, observed=first_cand_obs_votes)

    return model
