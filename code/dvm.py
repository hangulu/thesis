"""
This module implements the Discrete Voter Model for ecological inference in
Python 3.
"""

import functools
import random
import tensorflow as tf
import tensorflow_probability as tfp
import time

from tqdm.autonotebook import trange

import make_grid
import expec_votes as ev
import prob_votes as pv
import election
import tools

def init_hmc_kernel(log_prob_fn, step_size, num_adaptation_steps=0):
    """
    Initialize the HMC kernel.

    log_prob_fn (function): the function to calculate the log probability
    step_size (float): the float size to use for the kernel
    num_adaptation_steps (int): the number of adaptation steps

    return: kernel
    """
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=3)
    return tfp.mcmc.SimpleStepSizeAdaptation(
        hmc_kernel,
        num_adaptation_steps=num_adaptation_steps)

def init_rwm_kernel(log_prob_fn):
    """
    Initialize the RWM kernel.

    log_prob_fn (function): the function to calculate the log probability

    return: kernel
    """
    return tfp.mcmc.RandomWalkMetropolis(log_prob_fn)

def hmc_trace_fn(_, pkr):
    return pkr.inner_results.accepted_results.target_log_prob, pkr.inner_results.log_accept_ratio, pkr.inner_results.accepted_results.step_size

def rwm_trace_fn(_, pkr):
    return pkr.accepted_results.target_log_prob, pkr.log_accept_ratio

def sample_chain(kernel, n_iter, current_state, trace_fn=None):
    return tfp.mcmc.sample_chain(
        num_results=n_iter,
        num_steps_between_results=2,
        current_state=current_state,
        kernel=kernel,
        trace_fn=trace_fn)

def burn_in(chain_result_tensor, burn_frac):
    num_samples = chain_result_tensor.shape[0]
    start = int(burn_frac * num_samples)

    begin = [start]
    size = [num_samples - start]

    for dim in chain_result_tensor.shape[1:]:
        begin.append(0)
        size.append(dim)

    return tf.slice(chain_result_tensor, begin, size)

def hmc(n_iter, burn_frac, initial_grid, demo, observed, init_step_size=0.03, adaptation_frac=0.6, pause_point=10):
    """
    Run the Hamiltonian Monte Carlo MCMC algorithm to sample the space
    of probabilistic demographic grids in the discrete
    voter model.

    n_iter (int): the number of iterations to run
    burn_frac (float): the fraction of iterations to burn
    initial_grid (Tensor): the probabilistic grid to start with
    observed_votes (int): the number of votes a candidate got in an election
    demo (dict): the demographics of the district
    init_step_size (float): the initial step size for the transition
    adaptation_frac (float): the fraction of the burn in steps to be used for step size adaptation
    pause_point (int): the number of iterations to run in each chain chunk

    return: a tuple of the chain and the trace
    """
    start_time = time.time()
    # Find the number of steps for adaptation
    num_adaptation_steps = int(burn_frac * adaptation_frac * n_iter)

    # Separate the number of iterations into chunks
    fixed_size_steps = n_iter - num_adaptation_steps
    num_chunks = fixed_size_steps // pause_point
    remainder = fixed_size_steps % pause_point

    print(f"This Hamiltonian Monte Carlo chain will be run in {num_chunks} chunks of size {pause_point}, with {num_adaptation_steps} steps of adaptation and {remainder} steps at the end.\n")

    sample_chunks = []
    log_prob_trace_chunks = []
    log_accept_trace_chunks = []

    current_state = initial_grid
    kernel_results = None

    print("[1/5] Creating the binomial coefficients...")
    # Get the coefficients for the binomial calculations
    coeff_dict = election.get_coefficients(demo, observed)

    # Partially apply `prob_votes`, so it only takes the grid
    prob_votes_partial = functools.partial(
        pv.prob_votes,
        demo=demo,
        observed=observed,
        coeff_dict=coeff_dict)

    # Initialize the adaptive HMC transition kernel
    adaptive_hmc_kernel = init_hmc_kernel(prob_votes_partial, init_step_size, num_adaptation_steps)

    print(f"[2/5] Running the chain for {num_adaptation_steps} steps to adapt the step size...")
    # Run the chain with adaptive HMC to adapt the step size
    if num_adaptation_steps:
        samples, trace = sample_chain(
            adaptive_hmc_kernel,
            num_adaptation_steps,
            current_state,
            trace_fn=hmc_trace_fn)

        step_size_trace = trace[2]

        sample_chunks.append(samples)
        log_prob_trace_chunks.append(trace[0])
        log_accept_trace_chunks.append(trace[1])

        adapted_step_size = tools.find_last_finite(step_size_trace, default=init_step_size)
    else:
        adapted_step_size = init_step_size

    # Intialize the HMC transition kernel with the final step size
    hmc_kernel = init_hmc_kernel(prob_votes_partial, adapted_step_size)

    print(f"[3/5] Running the chain with a step size of {adapted_step_size} on {num_chunks} chunks of {pause_point} iterations each...")
    # Run the chain in chunks to be able to monitor progress
    for i in trange(num_chunks):
        samples, (log_prob_trace, log_accept_trace, _) = sample_chain(
            adaptive_hmc_kernel,
            pause_point,
            current_state,
            trace_fn=hmc_trace_fn)

        current_state = tf.nest.map_structure(lambda x: x[-1], samples)

        sample_chunks.append(samples)
        log_prob_trace_chunks.append(log_prob_trace)
        log_accept_trace_chunks.append(log_accept_trace)

    print(f"[4/5] Running the chain for {remainder} more steps...")
    # Run the chain for the remainder of steps
    samples, (log_prob_trace, log_accept_trace, _) = sample_chain(
        hmc_kernel,
        remainder,
        current_state,
        trace_fn=hmc_trace_fn)

    sample_chunks.append(samples)
    log_prob_trace_chunks.append(log_prob_trace)
    log_accept_trace_chunks.append(log_accept_trace)

    # Consolidate the results
    full_chain = tf.nest.map_structure(
        lambda *chunks: tf.concat(chunks, axis=0), *sample_chunks)
    full_log_prob_trace = tf.nest.map_structure(
        lambda *chunks: tf.concat(chunks, axis=0), *log_prob_trace_chunks)
    full_log_accept_trace = tf.nest.map_structure(
        lambda *chunks: tf.concat(chunks, axis=0), *log_accept_trace_chunks)

    print(full_chain.shape)

    print(f"[5/5] Burning {burn_frac} of samples...")
    burned_chain = burn_in(full_chain, burn_frac)
    burned_log_prob_trace = burn_in(full_log_prob_trace, burn_frac)
    burned_log_accept_trace = burn_in(full_log_accept_trace, burn_frac)

    elapsed = int(time.time() - start_time)
    num_samples = n_iter - int(burn_frac * n_iter)

    print("Done.")
    print(f"Generated a sample of {num_samples} observations in ~{elapsed} seconds.")
    return {'sample': burned_chain, 'log_prob_trace': burned_log_prob_trace, 'log_accept_trace': burned_log_accept_trace}


def rwm(n_iter, burn_frac, initial_grid, demo, observed, pause_point=10):
    """
    Run the Random Walk Metropolis MCMC algorithm to sample the space
    of probabilistic demographic grids in the discrete
    voter model.

    n_iter (int): the number of iterations to run
    burn_frac (float): the fraction of iterations to burn
    initial_grid (Tensor): the probabilistic grid to start with
    observed_votes (int): the number of votes a candidate got in an election
    demo (dict): the demographics of the district
    pause_point (int): the number of iterations to run in each chain chunk

    return: a tuple of the chain and the trace
    """
    start_time = time.time()

    # Separate the number of iterations into chunks
    num_chunks = n_iter // pause_point
    remainder = n_iter % pause_point

    print(f"The Random Walk Metropolis chain will be run in {num_chunks} chunks of size {pause_point}, with {remainder} steps at the end.\n")

    sample_chunks = []
    log_prob_trace_chunks = []
    log_accept_trace_chunks = []

    current_state = initial_grid
    kernel_results = None

    print("[1/4] Creating the binomial coefficients...")
    # Get the coefficients for the binomial calculations
    coeff_dict = election.get_coefficients(demo, observed)

    # Partially apply `prob_votes`, so it only takes the grid
    prob_votes_partial = functools.partial(
        pv.prob_votes,
        demo=demo,
        observed=observed,
        coeff_dict=coeff_dict)

    # Initialize the RWM transition kernel
    rwm_kernel = init_rwm_kernel(prob_votes_partial)

    print(f"[2/4] Running the chain on {num_chunks} chunks of {pause_point} iterations each...")
    # Run the chain in chunks to be able to monitor progress
    for i in trange(num_chunks):
        samples, (log_prob_trace, log_accept_trace) = sample_chain(
            rwm_kernel,
            pause_point,
            current_state,
            trace_fn=rwm_trace_fn)

        current_state = tf.nest.map_structure(lambda x: x[-1], samples)

        sample_chunks.append(samples)
        log_prob_trace_chunks.append(log_prob_trace)
        log_accept_trace_chunks.append(log_accept_trace)

    print(f"[3/4] Running the chain for {remainder} more steps...")
    # Run the chain for the remainder of steps
    samples, (log_prob_trace, log_accept_trace) = sample_chain(
        rwm_kernel,
        remainder,
        current_state,
        trace_fn=rwm_trace_fn)

    sample_chunks.append(samples)
    log_prob_trace_chunks.append(log_prob_trace)
    log_accept_trace_chunks.append(log_accept_trace)

    # Consolidate the results
    full_chain = tf.nest.map_structure(
        lambda *chunks: tf.concat(chunks, axis=0), *sample_chunks)
    full_log_prob_trace = tf.nest.map_structure(
        lambda *chunks: tf.concat(chunks, axis=0), *log_prob_trace_chunks)
    full_log_accept_trace = tf.nest.map_structure(
        lambda *chunks: tf.concat(chunks, axis=0), *log_accept_trace_chunks)

    print(f"[4/4] Burning {burn_frac} of the sample...")
    burned_chain = burn_in(full_chain, burn_frac)
    burned_log_prob_trace = burn_in(full_log_prob_trace, burn_frac)
    burned_log_accept_trace = burn_in(full_log_accept_trace, burn_frac)

    elapsed = int(time.time() - start_time)
    num_samples = n_iter - int(burn_frac * n_iter)

    print("Done.")
    print(f"Generated a sample of {num_samples} observations in ~{elapsed} seconds.")
    return {'sample': burned_chain, 'log_prob_trace': burned_log_prob_trace, 'log_accept_trace': burned_log_accept_trace}

def mle(chain_results):
    """
    Find the Maximum Likelihood Estimate (MLE) of the distribution of
    PHCs indicated by the sample generated by HMC or RWM.

    chain_results (dict): Python dictionary containing the sample and
    traces of log probability and log acceptance

    return: the PHC with the maximum likelihood
    """
    index = tf.math.argmax(chain_results['log_prob_trace'])
    return chain_results['sample'][index]
