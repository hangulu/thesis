"""
This module implements subroutines for creating elections and handling their
data.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
import tensorflow as tf


def get_vote_pcts(index, matrix_dim, demo_per_prec):
    """
    Find the vote percentages for each demographic group,
    given the index of an associated PHC.

    index (int tuple): the index of the PHC
    matrix_dim (int): the size of one dimension of the PHC
    demo_per_prec (dict): the demographics of the district

    return: a dict of the vote percentages for each demographic
    group
    """
    vote_pcts = {}
    for prec, prec_demo in demo_per_prec.items():
        vote_pcts[prec] = {group: (tf.cast(index[num], tf.float32) + 0.5) / matrix_dim for num, group in enumerate(prec_demo)}
    return vote_pcts


class Election:
    """
    An election in a district.

    Attributes:
        candidates (string list): the candidates
        dpp (dict): Demographics Per Precinct: the demographics of the
        electorate, per precinct
        winner (string): the candidate with the most votes
        vote_totals (dict): the vote totals of each candidate in the
        election, per precinct
        dvp (dict): Demographic Voting Probabilities: the theoretical voting
        percentages of each demographic group, for each candidate in each
        precinct
        outcome (dict): the outcome of the election, per precinct
    """

    def __init__(self, candidates, demo_per_prec, name=None, cand_vote_totals=None, demo_vote_prob=None, mock=True):
        # Intialize the variables
        # Always present
        self.name = name
        self.candidates = candidates
        self.dpp = demo_per_prec
        self.precincts = list(self.dpp.keys())
        self.vote_totals = cand_vote_totals
        self.winner = None

        # None or False, unless a mock election
        self.dvp = demo_vote_prob
        self.outcome = None
        self.mock = mock

        # Find the number of demographic groups in the electorate
        for prec, dpp in self.dpp.items():
            self.num_demo_groups = len(dpp)
            break

        # Decide a winner
        self.decide_election()

    def decide_election(self):
        """
        Decide the winner of an election.
        """
        if not self.vote_totals:
            self.simulate()

        map_wide_totals = defaultdict(int)
        for cand in self.candidates:
            for prec, votes in self.vote_totals.items():
                map_wide_totals[cand] += votes[cand]

        self.winner = max(map_wide_totals, key=map_wide_totals.get)

    def simulate(self):
        """
        Simulate an election.
        """
        result = {}
        for prec in self.precincts:
            # Set up the result dictionary for the electoral outcome
            prec_result = {cand: (0, {group: 0 for group in self.dpp[prec]}) for cand in self.candidates}

            # Iterate through each demographic group
            for group in self.dpp[prec]:
                # Extract information from the DVP dictionary
                demo_vote_frac = np.fromiter(self.dvp[prec][group].values(), dtype=float)
                n_voters = self.dpp[prec][group]

                # Simulate each voter
                votes = np.random.choice(self.candidates, n_voters, p=demo_vote_frac)
                for cand in votes:
                    prev_total, prev_demo_votes = prec_result[cand]
                    prev_demo_votes[group] += 1
                    prec_result[cand] = prev_total + 1, prev_demo_votes

            result[prec] = prec_result

        self.outcome = result

        # Extract the vote totals
        vote_totals = {}
        for prec, prec_result in result.items():
            prec_vote_totals = {}
            for cand, demo_votes in prec_result.items():
                prec_vote_totals[cand] = demo_votes[0]
            vote_totals[prec] = prec_vote_totals

        self.vote_totals = vote_totals

        # Extract the winner
        self.decide_election()

    def get_demo_votes(self, cand, prec_id):
        """
        Find the demographic breakdown of a candidate's electoral result in
        a precinct.

        cand (str): the candidate
        prec_id (str): the precinct

        return: a Python dictionary containing the demographic breakdown of
        a candidate's electoral result
        """
        return self.outcome[prec_id][cand][1]

    def __repr__(self):
        if self.mock:
            return f"A mock election with {len(self.candidates)} candidates in a district with {len(self.precincts)} precincts and {self.num_demo_groups} demographic groups."

        return f"A real election with {len(self.candidates)} candidates in a district with {len(self.precincts)} precincts and {self.num_demo_groups} demographic groups."


def create_elections(voting_data, demo_data, name):
    """
    Create elections from Pandas DataFrames of voting and
    demographic data.

    voting_data (Pandas DataFrame): the voting data for an election
    demo_data (Pandas DataFrame): the demographic data for an election
    name (string): identifiers of the election(s)

    return: a list of election objects
    """
    demo_names = list(demo_data.columns)
    demo_names.remove('prec_id')

    candidates = list(voting_data.columns)
    candidates.remove('prec_id')

    combined_data = pd.merge(voting_data, demo_data, on='prec_id')

    demo_per_prec = {}
    cand_vote_totals_per_prec = {}

    for row in combined_data.itertuples(index=False):
        prec_id = getattr(row, 'prec_id')

        prec_cand_vote_totals = {}
        for cand in candidates:
            prec_cand_vote_totals[cand] = getattr(row, cand)
        cand_vote_totals_per_prec[prec_id] = prec_cand_vote_totals

        prec_demo = {}
        for demo_group in demo_names:
            prec_demo[demo_group] = getattr(row, demo_group)
        demo_per_prec[prec_id] = prec_demo

    return Election(candidates, demo_per_prec, name=name,
                    cand_vote_totals=cand_vote_totals_per_prec,
                    mock=False)
