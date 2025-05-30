"""
    This work is developed for academic purpose by Bredariol Francesco, Savorgnan Enrico, Tic Ruben.
    This work is part of the final project for the 2024-2025 Reinforcement Learning course at the University of Trieste.
"""

class Epsilon:
    def __init__(self, eps):
        """
            This is just an Epsilon schedule. In fact it is the constant one
        """
        self.first_eps = eps
        self.eps = eps
    
    def decay(self):
        return self.eps
    
    def reset(self):
        self.eps = self.first_eps
        
    def __str__(self):
        return f"Constant Epsilon Schedule {self.first_eps}"

class ConstantEpsilonDecay(Epsilon):
    def __init__(self, eps):
        super().__init__(eps)

class LinearEpsilonDecay(Epsilon):

    def __init__(self, eps, coefficient = 0.999, minimum = 0.1 ):
        """
            This is the linear epsilon decay.
            the decay works as follows:
                eps = max(minimum, eps*coef)
        """
        super().__init__(eps)
        self.coefficient = coefficient
        self.minimum = minimum
    
    def decay(self):
        self.eps = max(self.minimum, self.eps*self.coefficient)
        return self.eps
    
    def __str__(self):
        return f"Linear Epsilon eps0 {self.first_eps} coef {self.coefficient} min {self.minimum}"
