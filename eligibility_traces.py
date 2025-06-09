"""
    This work is developed for academic purpose by Bredariol Francesco, Savorgnan Enrico, Tic Ruben.
    This work is part of the final project for the 2024-2025 Reinforcement Learning course at the University of Trieste.
"""
from collections import defaultdict


class Eligibility:
    """
        The eligibility_traces.py module provides functions to manage eligibility traces in the Snake project.
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
        # if state_action is not in self.traces, it will be initialized to 0.0
        if state_action not in self.traces:
            self.traces[state_action] = 0.0
        else :
            self.traces[state_action] += 1.0


    def decay(self):
        self.traces = {k: v * self.gamma * self.lambda_value for k, v in self.traces.items()}








