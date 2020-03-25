"""
This module implements subroutines for creating elections and handling their
data.
"""

import numpy as np
import tensorflow as tf


def get_vote_pcts(index, matrix_dim, demo):
    """
    Find the vote percentages for each demographic group,
    given the index of an associated PHC.

    index (int tuple): the index of the PHC
    matrix_dim (int): the size of one dimension of the PHC
    demo (dict): the demographics of the district

    return: a dict of the vote percentages for each demographic
    group
    """
    return {group: (tf.cast(index[num], tf.float32) + 0.5) / matrix_dim for num, group in enumerate(demo)}


class Election:
    """
    An election in a district.

    Attributes:
        candidates (string list): the candidates
        demo (dict): the demographics of the electorate
        dvp (dict): Demographic Voting Probabilities: the theoretical voting
        percentages of each demographic group, for each candidate
    """

    def __init__(self, candidates, demo, demo_vote_prob=None, mock=True):
        # Intialize the variables
        self.candidates = candidates
        self.demo = demo
        self.dvp = demo_vote_prob
        self.outcome = None
        self.winner = None
        self.mock = mock

        # Simulate a mock election and decide a winner
        if mock:
            self.simulate()

    def simulate(self):
        """
        Simulate an election.
        """
        # Set up the result dictionary for the electoral outcome
        result = {cand: (0, {group: 0 for group in self.demo}) for cand in self.candidates}

        # Iterate through each demographic group
        for group in self.demo:
            # Extract information from the DVP dictionary
            demo_vote_frac = np.fromiter(self.dvp[group].values(), dtype=float)
            n_voters = self.demo[group]

            # Simulate each voter
            votes = np.random.choice(self.candidates, n_voters, p=demo_vote_frac)
            for cand in votes:
                prev_total, prev_demo_votes = result[cand]
                prev_demo_votes[group] += 1
                result[cand] = prev_total + 1, prev_demo_votes

        self.outcome = result

        max_votes = 0
        winning_candidate = None

        for cand, ballot in result.items():
            if ballot[0] > max_votes:
                max_votes = ballot[0]
                winning_candidate = [cand]
            elif ballot[0] == max_votes and winning_candidate:
                winning_candidate.append(cand)

        self.winner = winning_candidate

    def get_demo_votes(self, cand):
        """
        Find the demographic breakdown of a candidate's electoral result.

        cand (str): the candidate

        return: a Python dictionary containing the demographic breakdown of
        a candidate's electoral result
        """
        return self.outcome[cand][1]

    def __repr__(self):
        if self.mock:
            return f"A mock election with {len(self.candidates)} candidates in a district with {len(self.demo)} demographic groups."

        return f"A real election with {len(self.candidates)} candidates in a district with {len(self.demo)} demographic groups."


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
