"""
    This work is developed for academic purpose by Bredariol Francesco, Savorgnan Enrico, Tic Ruben.
    This work is part of the final project for the 2024-2025 Reinforcement Learning course at the University of Trieste.
"""

class StateBracket():
    """
        A State Bracket is an object that takes a state as input and using some *defined* transformation returns a "bracket" for that state.
        Takes this example: 
            Define G a grid world nxn.
            You are in position (x, y).
            The goal is in position (gx, gy).
            A state representation for the problem could be G filled with all 0s with the exception of a 1 in (x, y) and a 2 in (gx, gy).
            Using this state representation you have a total space of 3^(nxn) (or something like this).
            A good idea is to "cluster" similar states.
            An idea could be to use as space representation only (gx-x, gy-y), that is the relative position of the goal wrt your position.
            This leads the total space to something like 4xnxn.
            This is great. 
            This is not always so easy, but this is the idea.
            Once you have defined the rule for the braket (that is, given a state, in wich cluster do I map it?), the game is done.
            Well, if you do stupid bracketing you can fuck up your solution.
            Good Luck finding your best brackets!
    """
    
    def __init__(self):
        pass

    def bracket(self, state):
        """
            Input : generic state
            Output : some feature of the state representing the bracket containing the state
        """
        pass

class FoodRelativePositionBraket(StateBracket):
    """
        Specific State Bracket for the snake game.
    """
    def __init__(self):
        super().__init__()

    def bracket(self, state):
        """
            Input : generic state
            Output : some feature of the state representing the bracket containing the state

            This bracketer takes as input the whole grid world. Returns as output the relative position of the food wrt the head of the snake
        """
        grid = state 
        hx, hy, fx, fy = 0, 0, 0, 0
        for i, row in enumerate(grid):
            for j, cel in enumerate(row):
                if cel == 2: # head has value 2
                    hx = j
                    hy = i
                if cel == 1: # food has value 1
                    fx = j
                    fy = i
        return (fx - hx, fy - hy)


