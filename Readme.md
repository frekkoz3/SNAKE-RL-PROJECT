# SNAKE 

---

This is the final project for the Reinforcement Learning course of University of Trieste, Italy.

This is a group-project. The team is made by:
Bredariol Francesco, Savorgnan Enrico, Tic Ruben

---

Here we try to solve the Snake game using different approaches.

---
## Structure

- `main.py`
  - This is the main file. Currently just runs a SARSA agent for the Snake game. 
  - Probably to update
  

- `main.ipynb`
  - This notebook runs a QLearning Agent and a SARSA Agent, saving the results in `models/` folder.


- `algorithms.py`
  - Defines the superclass **RLAlgorithm** which defines the main methods for the specific algorithms that are built upon it.
  - Algorithm s currently implemented:
    - **QLearning**
    - **SARSA**
  - Algorithms to implement:
    - **Double QLearning**
    

- `snake_environment.py`
  - In this file is defined the Snake environment.
  - 6 main methods are implemented:
    - `place_food()`: places food in a random position.
    - `_get_obs()`: returns the current observation of the environment.
    - `step(action)`: takes an action and returns the next state, reward, done flag, and info.
    - `render()`: renders the environment.
    - `reset()`: resets the environment to the initial state.


- `states_bracket.py`
  - This file defines a class **StateBracket** which, given a state, is able to retun the most similar "super-state"


- `models/`
  - This folder contains the saved models of the algorithms.


--- 
## TODO
- Implement Monte Carlo
- Implement Double QLearning
- Implement dynamic parameters
- Implement dynamic learning bracket

