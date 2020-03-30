"""
This module implements subroutines for creating elections and handling their
data.
"""

import numpy as np
import pandas as pd
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
        winner (string): the candidate with the most votes
        vote_totals (dict): the vote totals of each candidate in the
        election
        dvp (dict): Demographic Voting Probabilities: the theoretical voting
        percentages of each demographic group, for each candidate
    """

    def __init__(self, candidates, demo, name=None, cand_vote_totals=None, demo_vote_prob=None, mock=True):
        # Intialize the variables
        # Always present
        self.candidates = candidates
        self.demo = demo
        self.vote_totals = cand_vote_totals
        self.winner = None
        self.name = name

        # None or False, unless a mock election
        self.dvp = demo_vote_prob
        self.outcome = None
        self.mock = mock

        # Decide a winner
        self.decide_election()

    def decide_election(self):
        """
        Decide the winner of an election.
        """
        if not self.vote_totals:
            self.simulate()

        self.winner = max(self.vote_totals, key=self.vote_totals.get)

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

        # Extract the vote totals
        vote_totals = {}
        for cand, demo_votes in result.items():
            vote_totals[cand] = demo_votes[0]

        self.vote_totals = vote_totals

        # Extract the winner
        self.decide_election()

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


def create_elections(voting_data, demo_data, name, full_map=True):
    """
    Create elections from Pandas DataFrames of voting and
    demographic data.

    voting_data (Pandas DataFrame): the voting data for an election
    demo_data (Pandas DataFrame): the demographic data for an election
    name (string): identifiers of the election(s)
    full_map (bool): whether to use the full map as an election, or do
    precinct-wise elections

    return: a list of election objects
    """
    demo_names = list(demo_data.columns)
    demo_names.remove('prec_id')

    candidates = list(voting_data.columns)
    candidates.remove('prec_id')

    combined_data = pd.merge(voting_data, demo_data, on='prec_id')

    elections = []
    if full_map:
        data = combined_data.sum()
        cand_vote_totals = {}
        for cand in candidates:
            cand_vote_totals[cand] = data[cand]

        demo = {}
        for demo_group in demo_names:
            demo[demo_group] = data[demo_group]

        election_name = f"{name}_full"

        elections.append(Election(candidates, demo, name=name,
                                  cand_vote_totals=cand_vote_totals,
                                  mock=False))
    else:
        for row in combined_data.itertuples(index=False):
            cand_vote_totals = {}
            for cand in candidates:
                cand_vote_totals[cand] = getattr(row, cand)

            demo = {}
            for demo_group in demo_names:
                demo[demo_group] = getattr(row, demo_group)

            prec_id = getattr(row, 'prec_id')

            election_name = f"{name}_{prec_id}"

            elections.append(Election(candidates, demo, name=election_name,
                                      cand_vote_totals=cand_vote_totals,
                                      mock=False))

    return elections
