"""
    This work is developed for academic purpose by Bredariol Francesco, Savorgnan Enrico, Tic Ruben.
    This work is part of the final project for the 2024-2025 Reinforcement Learning course at the University of Trieste.
"""
import numpy as np
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

class VonNeumann2Neigh(StateBracket):
    """
        Specific State Bracket for the snake game.
    """

    def bracket(self, state):
        """
            Input : generic state
            Output : some feature of the state representing the bracket containing the state

            This bracketer takes as input the whole grid world. 
            Returns as output the von neumann neighborhood of distance 2 as a 5x5 grid around the snake's head 
            (free cell = 0, block or outside = -1, food = 1) 
        """
        head_pos = np.argwhere(state == 2)[0].tolist()      

        neigh = np.zeros((5,5)) - 1   #this will hold the result of binning. The initial value is -1 because only the cells inside the grid will be changed
       
        for (i, j) in np.ndindex(state.shape):                  #we iterate over all indices of the cells in the grid
            neigh_index = (i - head_pos[0] + 2, j - head_pos[1] + 2)    #corresponding index for the output array neigh

            if abs(i - head_pos[0]) + abs(j - head_pos[1]) <= 2:          #we consider only the cells at distance <= 2 from the head
                if state[i, j] == 0:
                    neigh[neigh_index] = 0
                if state[i, j] == 1:
                    neigh[neigh_index] = 1
                elif state[i, j] == 3:
                   neigh[neigh_index] = -1       

        return tuple(tuple(map(int, row))  for row in neigh.tolist())
         
        

    def get_state_dim(self):
        """
            Returns the dimension of the state space.
            In this case, a 5x5 grid around the snake's head
        """
        return 25
    
    def __str__(self):
        return "VN2"
    
    def to_string(self, state):
        return f"(The head in the middle. Food = 1, block = -1 Neigh :\n {np.array(state)}"

class SquaredNeigh(StateBracket):
    """
        Specific State Bracket for the snake game.
    """
    def __init__(self, size):
        super().__init__()
        self.size = size

    def bracket(self, state):
        """
            Input : generic state
            Output : some feature of the state representing the bracket containing the state

            This bracketer takes as input the whole grid world. 
            Returns as output the sizexsize grid around the snake's head
            (free cell = 0, block or outside = -1, food = 1) 
        """
        head_pos = np.argwhere(state == 2)[0].tolist()      

        neigh = np.zeros((self.size,self.size)) - 1   #this will hold the result of binning. The initial value is -1 because only the cells inside the grid will be changed
        
        for (i, j) in np.ndindex(state.shape):                          #we iterate over all indices of the cells in the grid
            neigh_index = (i - head_pos[0] + self.size//2, j - head_pos[1] + self.size//2)    #corresponding index for the output array neigh
            
            if all(0 <= neigh_index[k] < self.size for k in range(2)):        #this checks if we are inside the sizexsize grid
                
                if state[i, j] == 0:
                    neigh[neigh_index] = 0
                if state[i, j] == 1:
                    neigh[neigh_index] = 1
                elif state[i, j] == 3:
                    neigh[neigh_index] = -1       

        return tuple(tuple(map(int, row))  for row in neigh.tolist())

    def get_state_dim(self):
        """
            Returns the dimension of the state space.
            In this case, a sizexsize grid around the snake's head
        """
        return self.size**2
    
    def __str__(self):
        return "squared neigh"
    
    def to_string(self, state):
        return f"The head in the middle. Food = 1, block = -1 Neigh :\n {np.array(state)}"

class VonNeumann2NeighPlusFoodDirectionBracket(StateBracket):
    """
        Specific State Bracket for the snake game.
    """

    def bracket(self, state):
        """
            Input : generic state
            Output : some feature of the state representing the bracket containing the state

            This bracketer takes as input the whole grid world. 
            Returns as output the von neumann neighborhood of distance 2 as a 5x5 grid around the snake's head 
            (free cell = 0, block or outside = -1, food = 1) 
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
        return (int(hy < fy), int(hx < fx), int(hy > fy), int(hx > fx), *von_neumann_neigh_radius_2(grid))
         
        

    def get_state_dim(self):
        """
            Returns the dimension of the state space.
            In this case, a 5x5 grid around the snake's head plus for bits for food direction
        """
        return 29   #25(VN2) + 4(food direction)
    
    def __str__(self):
        return "VN2 with food direction"
    
    def to_string(self, state):
        return f"(ds : {state[0]}, de : {state[1]}, dn : {state[2]}, dw : {state[3]}) The head in the middle. Food = 1, block = -1  Neigh :\n {np.array(state[4:])} "

if __name__ == "__main__":
    grid = np.array(
           [[0, 0, 0, 0, 0, 3, 3, 0], 
            [0, 0, 0, 3, 3, 3, 3, 0],
            [0, 0, 0, 3, 0, 0, 3, 0], 
            [0, 0, 0, 3, 3, 0, 3, 0],
            [0, 0, 0, 0, 3, 0, 3, 0],
            [0, 0, 0, 0, 2, 0, 3, 0],
            [0, 0, 0, 0, 0, 0, 3, 0]])
    bracketer = VonNeumann2Neigh()
    
    bin = bracketer.bracket(grid)
    print(bracketer.to_string(bin))
