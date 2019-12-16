import numpy as np
import pymc3 as pm

def eco_inf(demo_pcts, candidate_pcts, precint_populations, lmbda=0.5):
    """
    Run King's Ecological Inference method on
    a 2x2 example (2 demographic groups).

    group_demo_pcts (NumPy array): the percentage of people in the
    demographic group for each precinct
    group_voting_pcts (NumPy array): the percentage of people in the
    precinct who voted for a candidate
    precint_populations (NumPy array): the populations of the
    precincts
    lmbda (float):

    return: the probabilistic model
    """
    demo_counts = candidate_pcts * precint_populations
    # Number of populations
    p = len(precint_populations)
    with pm.Model() as model:
        c_1 = pm.Exponential('c_1', lmbda)
        d_1 = pm.Exponential('d_1', lmbda)
        c_2 = pm.Exponential('c_2', lmbda)
        d_2 = pm.Exponential('d_2', lmbda)

        b_1 = pm.Beta('b_1', alpha=c_1, beta=d_1, shape=p)
        b_2 = pm.Beta('b_2', alpha=c_2, beta=d_2, shape=p)

        theta = demo_pcts * b_1 + (1 - demo_pcts) * b_2
        Tprime = pm.Binomial('Tprime', n=precint_populations , p=theta, observed=demo_counts)
    return model
