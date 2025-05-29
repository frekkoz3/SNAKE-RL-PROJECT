"""
    This work is developed for academic purpose by Bredariol Francesco, Savorgnan Enrico, Tic Ruben.
    This work is part of the final project for the 2024-2025 Reinforcement Learning course at the University of Trieste.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

CELL_SIZE = 20
GRID_WIDTH, GRID_HEIGHT = 20, 20  # 400x300 if CELL_SIZE=20
WIDTH, HEIGHT = GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE

# In this environment the snake takes -1 for moving, -5 for dying, 10 for eating

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None, max_step = 1000, **kwargs):
        """
            This is the class implementing the gymnasium protocol for the snake environment. 
            As input there are the render mode and the maximum amount of step.
            If wanted one can also pass (in a **dictionary) :
                reward_food, the reward for eating food
                reward_death, the reward for dying
                reward_step, the reward givern at each step
        """
        self.action_space = spaces.Discrete(4)  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8
        )
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.total_step = 0
        self.max_step = max_step
        self.reset()
        if kwargs == {}:
            self.reward_food = 10
            self.reward_death = -10
            self.reward_step = -1
        else:
            self.reward_food = kwargs["reward_food"]
            self.reward_death = kwargs["reward_death"]
            self.reward_step = kwargs["reward_step"]
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        #initial conditions are random
        self.snake = [(random.randrange(GRID_HEIGHT), random.randrange(GRID_WIDTH))]
        self.direction = random.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
        
        self._place_food()
        self.done = False
        self.score = 0
        self.total_step = 0
        self.info = {}
        return self._get_obs(), self.info

    def _place_food(self):
        while True:
            self.food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if self.food not in self.snake:
                break

    def _get_obs(self):
        grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        for x, y in self.snake:
            grid[y, x] = 3 # 3 for the body of the snake
        head_x, head_y = self.snake[0]
        grid[head_y, head_x] = 2 # 2 for the head of the snake
        food_x, food_y = self.food
        grid[food_y, food_x] = 1 # 1 for the food of the snake
        return grid

    def get_possible_actions(self, action):
        """
        Returns all the possible actions given the current one.
        """
        if action is None:
            return list(range(self.action_space.n))
        forbidden_action = {0:1, 1:0, 2:3, 3:2}[action]  # Prevent reversing direction
        possible_actions = [i for i in range(self.action_space.n) if i != forbidden_action]
        return possible_actions

    def step(self, action):
        self.info = {}
        if self.done:
            return self._get_obs(), 0.0, True, False, self.info
        
        self.total_step += 1

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        new_dir = directions[action]

        # Prevent reversing
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir
        else:
            self.info = {"act" : directions.index(self.direction)}

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # Check collision
        if (
            new_head in self.snake or
            not (0 <= new_head[0] < GRID_WIDTH) or
            not (0 <= new_head[1] < GRID_HEIGHT)
        ):
            self.done = True
            reward = self.reward_death
            return self._get_obs(), reward, True, False, self.info

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward = self.reward_food
            self._place_food()
        else:
            self.snake.pop()
            reward = self.reward_step

        if self.total_step > self.max_step:
            self.done = True # this should be truncated 
            truncated = True
            reward = 0
            return self._get_obs(), reward, self.done, truncated, self.info

        return self._get_obs(), reward, self.done, False, self.info

    def render(self):
        if self.render_mode != "human":
            return True
        
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Snake Environment")
            self.clock = pygame.time.Clock()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False

        self.window.fill((0, 0, 0))

        for i, s in enumerate(self.snake):
            x, y = s
            if i == 0:
                pygame.draw.rect(self.window, (0, 255, 0), pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            else:
                pygame.draw.rect(self.window, (0, 181, 0), pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        food_x, food_y = self.food
        pygame.draw.rect(self.window, (255, 0, 0), pygame.Rect(food_x * CELL_SIZE, food_y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

        return True

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None

if __name__ == "__main__":
    pass



