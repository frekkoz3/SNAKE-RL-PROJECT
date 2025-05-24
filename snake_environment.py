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

# In this environment the snake takes -1 for moving and 0 for eating

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        self.action_space = spaces.Discrete(4)  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8
        )
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = [(5, 5)]
        self.direction = (1, 0)
        self._place_food()
        self.done = False
        self.score = 0
        return self._get_obs(), {}

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

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        new_dir = directions[action]

        # Prevent reversing
        if (new_dir[0] * -1, new_dir[1] * -1) != self.direction:
            self.direction = new_dir

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
            reward = -10
            return self._get_obs(), reward, True, False, {}

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward = 100
            self._place_food()
        else:
            self.snake.pop()
            reward = -0.5

        return self._get_obs(), reward, self.done, False, {}

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



