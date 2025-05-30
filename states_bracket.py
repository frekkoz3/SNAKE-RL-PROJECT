"""
    This work is developed for academic purpose by Bredariol Francesco, Savorgnan Enrico, Tic Ruben.
    This work is part of the final project for the 2024-2025 Reinforcement Learning course at the University of Trieste.
"""
def out_of_border(grid, pos):
    max_y = len(grid)
    max_x = len(grid[0])
    y = pos[0]
    x = pos[1]
    return y >= max_y or y < 0 or x >= max_x or x < 0

def von_neumann_neigh_radius_1(grid, pos):
    """
        The grid is the whole state made up of 0, 1, 2 and 3 (3 for blocks).
        The position is in the form (y, x) since it is how the grid is decoded (row first matrix)
    """
    mov = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    neigh = [0, 0, 0, 0]
    rel_pos = [(pos[0]+m[0], pos[1]+m[1]) for m in mov]
    outside = [out_of_border(grid, r) for r in rel_pos]
    for i, oob in enumerate(outside):
        if oob:
            neigh[i] = 1
        else:
            if grid[rel_pos[i][0]][rel_pos[i][1]] == 3: # there is a block
                neigh[i] = 1
    return tuple(neigh)

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
            Once you have defined the rule for the bracket (that is, given a state, in which cluster do I map it?), the game is done.
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

    def get_state_dim(self):
        """
            Output: the dimension of the state representation
        """
        pass

    def __str__(self):
        """
            Output: a generic string representation for the state bracketer
        """
        pass

    def to_string(self, state):
        """
            Input : a state
            Output : the string format of the state following the bracketer mechanism
        """
        pass

class FoodRelativePositionBracket(StateBracket):
    """
        Specific State Bracket for the snake game.
    """

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

    def get_state_dim(self):
        """
            Returns the dimension of the state space.
            In this case, the relative position of the food wrt the head of the snake can be represented as a 2D vector.
        """
        return 2
    
    def __str__(self):
        return "FRP"
    
    def to_string(self, state):
        dx, dy = state
        return f"(dx : {dx}, dy : {dy})"

class VonNeumann1NeighPlusFoodRelPosBracket(StateBracket):
    def bracket(self, state):
        """
            Input : generic state
            Output : some feature of the state representing the bracket containing the state

            This bracketer takes as input the whole grid world. Returns as output the relative position of the food wrt the head of the snake
            plus the type of block (0 if not blocked, 1 if blocked) in the von neumann neighborhood of radius 1 of the head of the snake.
            The state representation is (dx, dy, s, e, n, w). 
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
        
        return (fx - hx, fy - hy, *von_neumann_neigh_radius_1(grid, [hy, hx]))
    
    def get_state_dim(self):
        return 6 # 2 for the food rel position, 4 for the von neumann neighborhood
    
    def __str__(self):
        return "VN1 FRP"
    
    def to_string(self, state):
        dx, dy, s, e, n, w = state
        return f"(dx : {dx}, dy : {dy}) Neigh : [S:{s}][E:{e}][N:{n}][W:{w}]"
    
class FoodDirectionBracket(StateBracket):
    """
        Specific State Bracket for the snake game.
    """

    def bracket(self, state):
        """
            Input : generic state
            Output : some feature of the state representing the bracket containing the state

            This bracketer takes as input the whole grid world. Returns as output the direction of the food wrt the head of the snake
            It look for the axis and says : 1 if the food is above the head, -1 if the food is under the head and 0 otherwise
            (rel_y, rel_x)
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

        return (int(hy < fy), int(hx < fx), int(hy > fy), int(hx > fx))

    def get_state_dim(self):
        """
            Returns the dimension of the state space.
            In this case, the relative position of the food wrt the head of the snake can be represented as a 2D vector.
        """
        return 4
    
    def __str__(self):
        return "FD"

    def to_string(self, state):
        ds, de, dn, dw = state
        return f"(ds : {ds}, de : {de}, dn : {dn}, dw : {dw})"

class VonNeumann1NeighPlusFoodDirectionBracket(StateBracket):
    """
        Specific State Bracket for the snake game.
    """

    def bracket(self, state):
        """
            Input : generic state
            Output : some feature of the state representing the bracket containing the state

            This bracketer takes as input the whole grid world. Returns as output the direction of the food wrt the head of the snake plus the von neumann neighborhood
            It look for the axis and says : 1 if the food is above the head, -1 if the food is under the head and 0 otherwise
            (rel_y, rel_x)
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

        return (int(hy < fy), int(hx < fx), int(hy > fy), int(hx > fx), *von_neumann_neigh_radius_1(grid, [hy, hx]))

    def get_state_dim(self):
        """
            Returns the dimension of the state space.
            In this case, the relative position of the food wrt the head of the snake can be represented as a 2D vector.
        """
        return 8
    
    def __str__(self):
        return "VN1 FD"
    
    def to_string(self, state):
        ds, de, dn, dw, s, e, n, w = state
        return f"(ds : {ds}, de : {de}, dn : {dn}, dw : {dw}) Neigh : [S:{s}][E:{e}][N:{n}][W:{w}]"

if __name__ == "__main__":
    grid = [[0, 3, 3, 3], 
            [0, 2, 3, 3],
            [0, 0, 0, 0], 
            [1, 0, 0, 0]]
    bracketer = VonNeumann1NeighPlusFoodDirectionBracket()
    print(bracketer.to_string(bracketer.bracket(grid)))