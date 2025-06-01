"""
The eligibility_traces.py module provides functions to manage eligibility traces in the Snake project.
"""
from collections import defaultdict


class Eligibility:
    """
    This class implements the eligibility traces mechanism for TD(λ) algorithms.
    """
    def __init__(self, lambda_value, gamma=0.9):
        self.lambda_value = lambda_value
        self.traces = defaultdict(float)
        self.gamma = gamma

    def reset(self):
        """
        Reset the eligibility traces.
        """
        self.traces.clear()

    def update(self, state_action):
        """
        state_action here is in the usual format (*state, action)
        """
        self.traces[state_action] += 1.0

    def decay(self):
        self.traces *= self.gamma * self.lambda_value








