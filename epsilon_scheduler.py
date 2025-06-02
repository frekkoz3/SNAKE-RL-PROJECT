"""
    This work is developed for academic purpose by Bredariol Francesco, Savorgnan Enrico, Tic Ruben.
    This work is part of the final project for the 2024-2025 Reinforcement Learning course at the University of Trieste.
"""

class Epsilon:
    def __init__(self, eps):
        """
            This is just an Epsilon schedule. In fact it is the constant one
        """

        assert 0 < eps < 1, "Error in Epsilon: eps must be a probability (between 0 and 1)"
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

    def __init__(self, eps, coefficient = 0.999, minimum = 0.1, n_steps_to_minimum=None):
        """
            This is the linear epsilon decay.
            the decay works as follows:
                if steps_to_minimum is provided --> eps decays exponentially reaching the minimum in n_steps_to_minimum steps
                else                            --> eps = max(minimum, eps*coef)
        """
        super().__init__(eps)
        assert  0 <= minimum <= 1 , "Error in LinearEpsilonDecay: minimum must be a probabilty (between 0 and 1)"
        self.minimum = minimum
        self.n_steps_to_minimum = n_steps_to_minimum
        if n_steps_to_minimum is None:
            assert 0 < coefficient < 1, "Error in LinearEpsilonDecay: coefficient must be between 0 and 1" 
            self.coefficient = coefficient
        else:
            assert n_steps_to_minimum > 0, "Error in LinearEpsilonDecay: number of steps for minimum must be stictly positive"
            self.coefficient = (self.minimum/self.first_eps)**(1/n_steps_to_minimum)
        
    
    def decay(self):
        self.eps = max(self.minimum, self.eps*self.coefficient)
        return self.eps
    
    def __str__(self):
        if self.n_steps_to_minimum is None:
            return f"Linear Epsilon eps0 {self.first_eps} coef {self.coefficient} min {self.minimum}"
        else:
            return f"Linear Epsilon eps0 {self.first_eps} min {self.minimum} reached in {self.n_steps_to_minimum} steps"



