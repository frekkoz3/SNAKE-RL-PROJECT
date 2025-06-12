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

def out_of_border_faster(max_y, max_x, y, x):
    return y >= max_y or y < 0 or x >= max_x or x < 0

def moore_neigh(grid, pos, rad):
    """
        This is a general function to compute, given a position on a grid world, its correspective Moore neighborhood of radius r.
    """
    max_y = len(grid)
    max_x = len(grid[0])
    y, x = pos
    y_pos = [p for p in range (y - rad, y + rad + 1)]
    x_pos = [p for p in range (x - rad, x + rad + 1)]
    neigh_pos = [[y_p, x_p] for y_p in y_pos for x_p in x_pos]
    neigh = []
    for n in neigh_pos:
        n_y, n_x = n
        if out_of_border_faster(max_y, max_x, n_y, n_x):
            neigh.append(1)
        elif grid[n_y][n_x] >= 3:
            neigh.append(1)
        else:
            neigh.append(0)
    return tuple(neigh)

def von_neumann_neigh(grid, pos, rad):
    """
        This is a general function to compute, given a position on a grid world, its correspective Von Neumann neighborhood of radius r.
    """
    max_y = len(grid)
    max_x = len(grid[0])
    y, x = pos
    # We take the moore and substract those positions that doesn't hold |x-x0| + |y - y0| <= r
    y_pos = [p for p in range (y - rad, y + rad + 1)]
    x_pos = [p for p in range (x - rad, x + rad + 1)]
    neigh_pos = [[y_p, x_p] for y_p in y_pos for x_p in x_pos]
    neigh = []
    for n in neigh_pos:
        n_y, n_x = n
        if (abs(x - n_x) + abs(y - n_y)) <= rad:
            if out_of_border_faster(max_y, max_x, n_y, n_x):
                neigh.append(1)
            elif grid[n_y][n_x] >= 3:
                neigh.append(1)
            else:
                neigh.append(0)
    return neigh


def get_object_position(grid, code, width, height):
    positions = []

    for x in range(width):
        for y in range(height):
            if grid[x, y] == code:
                positions.append((x, y))
                if code != 3:
                    break

    return positions

# SUPER CLASS
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
            Input : generic state (as a np.array)
            Output : some feature of the state representing the bracket containing the state (this must be a tuple for its use in a dictonary)
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

# ONLY FOOD BRACKETER
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

# FOOD PLUS NEIGHBORHOOD BRACKETER
class NeighPlusFoodRelativePositionBracket(StateBracket):
    def __init__(self, neigh = "V", radius = 1):
        """
            This bracketer is a generalization of all others bracketers that combines neighborhood and food relative position informations. 
            It does not implement to_string(state) and neither get_state_dim().
            Input are the neighborhood type, "V" for Von Neumann, "M" for Moore, and the radius.
        """
        super().__init__()
        self.neigh_name = neigh
        self.neigh = moore_neigh if neigh == "M" else von_neumann_neigh
        self.radius = radius
    
    def bracket(self, state):
        """
            Input : generic state
            Output : some feature of the state representing the bracket containing the state

            This bracketer takes as input the whole grid world. Returns as output the relative position of the food wrt the head of the snake plus the desider neighborhood information.
            In the neighborhood it says : 0 if the cell is empty, 1 if the cell is occupied by a block or by the tail.
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

        return (fx - hx, fy - hy, *self.neigh(grid, [hy, hx], self.radius))
    
    def __str__(self):
        return f"{self.neigh_name}{self.radius} FRP"

    def get_state_dim(self):
        if self.neigh_name == "M":
            return (2*self.radius + 1)*(2*self.radius + 1) + 2
        return self.radius*self.radius + (self.radius + 1)*(self.radius + 1) + 2    
    
class NeighPlusFoodDirectionBracket(StateBracket):
    def __init__(self, neigh = "V", radius = 1):
        """
            This bracketer is a generalization of all others bracketers that combines neighborhood and food direction informations. 
            It does not implement to_string(state) and neither get_state_dim().
            Input are the neighborhood type, "V" for Von Neumann, "M" for Moore, and the radius.
        """
        super().__init__()
        self.neigh_name = neigh
        self.neigh = moore_neigh if neigh == "M" else von_neumann_neigh
        self.radius = radius
    
    def bracket(self, state):
        """
            Input : generic state
            Output : some feature of the state representing the bracket containing the state

            This bracketer takes as input the whole grid world. Returns as output the direction of the food wrt the head of the snake plus the desider neighborhood information.
            It look for the axis and says : 1 if the food is above the head, -1 if the food is under the head and 0 otherwise
            (rel_y, rel_x).
            In the neighborhood it says : 0 if the cell is empty, 1 if the cell is occupied by a block or by the tail.
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

        return (int(hy < fy), int(hx < fx), int(hy > fy), int(hx > fx), *self.neigh(grid, [hy, hx], self.radius))
    
    def __str__(self):
        return f"{self.neigh_name}{self.radius} FD"

    def get_state_dim(self):
        if self.neigh_name == "M":
            return (2*self.radius + 1)*(2*self.radius + 1) + 4
        return self.radius*self.radius + (self.radius + 1)*(self.radius + 1) + 4    

# ONLY NEIGHBORHOOD BRACKETER
class NeighborhoodBracket(StateBracket):
    def __init__(self, neigh = "V", radius = 1):
        """
            This bracketer is a generalization of all others bracketers that combines neighborhood and food direction informations. 
            It does not implement to_string(state) and neither get_state_dim().
            Input are the neighborhood type, "V" for Von Neumann, "M" for Moore, and the radius.
        """
        super().__init__()
        self.neigh_name = neigh
        self.neigh = moore_neigh if neigh == "M" else von_neumann_neigh
        self.radius = radius

    def bracket(self, state):
        """
            Input : generic state
            Output : some feature of the state representing the bracket containing the state

            This bracketer takes as input the whole grid world. 
            Returns as output the desidered neighboorhood informations. 
            In the neighborhood it says : 0 if the cell is empty, 1 if the cell is occupied by a block or by the tail.
        """
        grid = state 
        hx, hy, = 0, 0
        for i, row in enumerate(grid):
            for j, cel in enumerate(row):
                if cel == 2: # head has value 2
                    hx = j
                    hy = i

        return (*self.neigh(grid, [hy, hx], self.radius), )

    def __str__(self):
        return self.neigh_name
    
    def get_state_dim(self):
        if self.neigh_name == "M":
            return (2*self.radius + 1)*(2*self.radius + 1)
        return self.radius*self.radius + (self.radius + 1)*(self.radius + 1)  
    
# FOOD PLUS NEIGHBORHOOD PLUS TAIL BRACKETER 
class NeighPlusFoodRelativePositionPlusTailBracket(StateBracket):
    def __init__(self, neigh = "V", radius = 1):
        """
            This bracketer is a generalization of all others bracketers that combines neighborhood and food relative position informations. 
            It does not implement to_string(state) and neither get_state_dim().
            Input are the neighborhood type, "V" for Von Neumann, "M" for Moore, and the radius.
        """
        super().__init__()
        self.neigh_name = neigh
        self.neigh = moore_neigh if neigh == "M" else von_neumann_neigh
        self.radius = radius
    
    def bracket(self, state):
        """
            Input : generic state
            Output : some feature of the state representing the bracket containing the state

            This bracketer takes as input the whole grid world. Returns as output the length of the tail plus the relative position of the food wrt the head of the snake plus the desider neighborhood information.
            In the neighborhood it says : 0 if the cell is empty, 1 if the cell is occupied by a block or by the tail.
        """
        grid = state 
        hx, hy, fx, fy = 0, 0, 0, 0
        tail = 0
        for i, row in enumerate(grid):
            for j, cel in enumerate(row):
                if cel == 3: # tail has value 3
                    tail += 1
                if cel == 2: # head has value 2
                    hx = j
                    hy = i
                if cel == 1: # food has value 1
                    fx = j
                    fy = i

        return (tail, fx - hx, fy - hy, *self.neigh(grid, [hy, hx], self.radius))
    
    def __str__(self):
        return f"{self.neigh_name}{self.radius} FRP T"

    def get_state_dim(self):
        if self.neigh_name == "M":
            return (2*self.radius + 1)*(2*self.radius + 1) + 2 + 1
        return self.radius*self.radius + (self.radius + 1)*(self.radius + 1) + 2 + 1

class NeighPlusFoodDirectionPlusTailBracket(StateBracket):
    def __init__(self, neigh = "V", radius = 1):
        """
            This bracketer is a generalization of all others bracketers that combines neighborhood and food relative position informations. 
            It does not implement to_string(state) and neither get_state_dim().
            Input are the neighborhood type, "V" for Von Neumann, "M" for Moore, and the radius.
        """
        super().__init__()
        self.neigh_name = neigh
        self.neigh = moore_neigh if neigh == "M" else von_neumann_neigh
        self.radius = radius
    
    def bracket(self, state):
        """
            Input : generic state
            Output : some feature of the state representing the bracket containing the state

            This bracketer takes as input the whole grid world. Returns as output the length of the tail plus the relative position of the food wrt the head of the snake plus the desider neighborhood information.
            In the neighborhood it says : 0 if the cell is empty, 1 if the cell is occupied by a block or by the tail.
        """
        grid = state 
        hx, hy, fx, fy = 0, 0, 0, 0
        tail = 0
        for i, row in enumerate(grid):
            for j, cel in enumerate(row):
                if cel == 3: # tail has value 3
                    tail += 1
                if cel == 2: # head has value 2
                    hx = j
                    hy = i
                if cel == 1: # food has value 1
                    fx = j
                    fy = i

        return tail, int(hy < fy), int(hx < fx), int(hy > fy), int(hx > fx), *self.neigh(grid, [hy, hx], self.radius)
    
    def __str__(self):
        return f"{self.neigh_name}{self.radius} FRP T"

    def get_state_dim(self):
        if self.neigh_name == "M":
            return (2*self.radius + 1)*(2*self.radius + 1) + 4 + 1
        return self.radius*self.radius + (self.radius + 1)*(self.radius + 1) + 4 + 1


class FullGrid(StateBracket):

    def __init__(self, height, width):
        if width != height:
            raise ValueError("The grid must be square, width must be equal to height.")

        super().__init__()
        self.width = width
        self.height = height

    def bracket(self, state):
        """
            Input : generic state
            Output : the full grid as a tuple of tuples, where each inner tuple represents a row in the grid.
        """
        if state.shape != (self.height, self.width):
            raise ValueError(f"State must be of shape ({self.height}, {self.width}).")

        return tuple(tuple(row) for row in state)

    def get_state_dim(self):
        """
            Returns the dimension of the state space.
            In this case, the full grid is represented as a tuple of tuples.
        """
        return self.height * self.width


class FullGridEncoded(StateBracket):

    def __init__(self, height, width):
        if width != height:
            raise ValueError("The grid must be square, width must be equal to height.")

        super().__init__()
        self.width = width
        self.height = height


    def bracket(self, state):
        if state.shape != (self.height, self.width):
            raise ValueError(f"State must be of shape ({self.height}, {self.width}).")

        food_position = np.array(get_object_position(state, code=1, width=self.width, height=self.height), ndmin=1)
        head_position = np.array(get_object_position(state, code=2, width=self.width, height=self.height), ndmin=1)
        tail_position = np.array(get_object_position(state, code=3, width=self.width, height=self.height), ndmin=1)

        food_grid = np.zeros((self.height, self.width), dtype=int)
        food_grid[food_position[0, 0], food_position[0, 1]] = 1

        head_grid = np.zeros((self.height, self.width), dtype=int)
        head_grid[head_position[0, 0], head_position[0, 1]] = 1

        tail_grid = np.zeros((self.height, self.width), dtype=int)
        for pos in tail_position:
            tail_grid[pos[0], pos[1]] = 1

        food_grid = food_grid.tolist()
        head_grid = head_grid.tolist()
        tail_grid = tail_grid.tolist()

        return (
            tuple(tuple(row) for row in food_grid),
            tuple(tuple(row) for row in head_grid),
            tuple(tuple(row) for row in tail_grid)
        )
    
if __name__ == "__main__":
    grid = np.array(
           [[0, 0, 0, 0, 0, 3, 3, 0], 
            [0, 0, 0, 3, 3, 3, 3, 0],
            [0, 0, 0, 3, 0, 0, 3, 0], 
            [0, 1, 0, 3, 3, 0, 3, 0],
            [0, 0, 0, 0, 3, 0, 3, 0],
            [0, 0, 0, 0, 2, 0, 3, 0],
            [0, 0, 0, 0, 0, 0, 3, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]])

    bracketer = FullGridEncoded(8, 8)
    print(bracketer.bracket(grid))
