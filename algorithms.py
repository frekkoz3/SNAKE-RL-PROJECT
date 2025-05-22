"""
    This work is developed for academic purpose by Bredariol Francesco, Savorgnan Enrico, Tic Ruben.
    This work is part of the final project for the 2024-2025 Reinforcement Learning course at the University of Trieste.
"""

import numpy as np


from collections import defaultdict

class RLAlgorithm:
    """
        This is the generic class for an RL Algorithm using gymnasium environment
    """

    def __init__(self, env):
        self.env = env
        self.values = defaultdict(int())
        self.state, _ = env.reset()
        self.iterations = 0 # This is used to count how many iteration until convergence

    def episode_update(self):
        """
            This is the function where to take one trajectory in the environment
        """
        pass

    def loop(self):
        """
            This is the function where to optimize the policy or the policy evaluation
        """
        pass

class Montecarlo(RLAlgorithm):
    
    def __init__(self, env):
        super().__init__(env)
    
class SARSA(RLAlgorithm):
    
    def __init__(self, env):
        super().__init__(env)