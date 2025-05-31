# This file is aimed to be the main entry point for the model.

import os
from algorithms import *
from main import returns
from snake_environment import *
from states_bracket import *
from epsilon_scheduler import *
from utils import *

def file_load_params():
    pass

def manual_load_params():
    # Ask the user to choose the algorithm for the training
    user_algorithm = input("Choose the algorithm (Q: Q-learning, D: DQN, S: SARSA, M: MonteCarlo): ").strip().upper()
    if user_algorithm not in ['Q', 'D', 'S', 'M']:
        print("Invalid algorithm selected. Please choose 'Q' for Q-learning, 'D' for DQN, 'S' for SARSA, or 'M' for Monte Carlo.")
        return

    # Ask the user to choose the state bracket
    user_bracket = input("Choose the state bracket (FP for FoodRelativePosition, NFP for VonNeumannFoodRelativePosition, FD for FoodDirection, NFD for VonNeumannFoodDirection): ").strip().upper()
    if user_bracket not in ['FP', 'NFP', 'FD', 'NFD']:
        print("Invalid state bracket selected. Please choose 'FP', 'NFP', 'FD', or 'NFD'.")
        return

    # Ask the user to choose the number of episodes
    try:
        user_num_episodes = int(input("Enter the number of episodes for training: ").strip())
        if user_num_episodes <= 0:
            raise ValueError("Number of episodes must be greater than 0.")
    except ValueError as e:
        print(f"Invalid input for number of episodes: {e}")
        return

    # Ask the user to choose the learning rate
    try:
        user_lr_v = float(input("Enter the learning rate (e.g., 0.01): ").strip())
        if user_lr_v <= 0:
            raise ValueError("Learning rate must be greater than 0.")
    except ValueError as e:
        print(f"Invalid input for learning rate: {e}")
        return

    # Ask the user to choose the discount factor
    try:
        user_gamma = float(input("Enter the discount factor (e.g., 0.99): ").strip())
        if not (0 < user_gamma <= 1):
            raise ValueError("Discount factor must be between 0 and 1.")
    except ValueError as e:
        print(f"Invalid input for discount factor: {e}")
        return

    # Ask the user to choose the epsilon decay
    user_epsilon_decay = float(input("Enter the epsilon decay (L for linear, E for exponential): ").strip().upper())
    if user_epsilon_decay not in ['L', 'E']:
        print("Invalid epsilon decay selected. Please choose 'L' for linear or 'E' for exponential.")
        return


def train():
    # Ask the user to choose for manual parameters or file loading
    manual_input = input("Do you want to enter parameters manually (Y/N)? ").strip().upper()
    if manual_input not in ['Y', 'N']:
        print("Invalid input. Please enter 'Y' for Yes or 'N' for No.")
        return -1
    if manual_input == 'N':
        params = file_load_params()
    else:
        params = manual_load_params()

    model_name = params[0]
    bracketer = params[1]
    gamma = params[2]
    lr_v = params[3]
    epsilon_scheduler = params[4]
    num_episodes = params[5]
    model_path = params[6]

    env = SnakeEnv(render_mode='nonhuman')
    if model_name == 'DDQN':

        state_dim = bracketer.get_state_dim()
        batch_size = params[7]
        memory_size = params[8]
        target_update_freq = params[9]
        device = params[10]

        model = DeepDoubleQLearning(action_space=env.action_space,
                                  state_dim=state_dim,
                                  gamma=gamma,
                                  lr_v=lr_v,
                                  batch_size=batch_size,
                                  memory_size=memory_size,
                                  target_update_freq=target_update_freq,
                                  device=device)

    elif model_name == 'QLearning':
        model = QLearning(action_space=env.action_space, gamma=gamma, lr_v=lr_v)

    elif model_name == 'SARSA':
        model = SARSA(action_space=env.action_space, gamma=gamma, lr_v=lr_v)

    elif model_name == 'MC':
        model = Montecarlo(action_space=env.action_space, gamma=gamma, lr_v=lr_v)

    model.learning(env=env, epsilon_schedule=epsilon_scheduler, n_episodes=num_episodes,
                   bracketer=bracketer)
    model.save(model_path)

    print(f"Training completed. Model saved to {model_path}")
    avg_reward = get_model_average_score(model_name, env.action_space, gamma, lr_v, model_path, bracketer, num_episodes=100, render_mode='nonhuman')
    print(f"Average reward over 100 episodes: {avg_reward}")
    return 0

def play():
    """
    This method is used to just play with a given model.
    """
    model_path = input("Enter the path to the model you want to play with: ").strip()
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist. Please check the path and try again.")
        return

    model_name = input("Enter the model name (DDQN, QLearning, SARSA, MC): ").strip().upper()
    if model_name not in ['DDQN', 'QLearning', 'SARSA', 'MC']:
        print(f"Invalid model name {model_name}. Supported models are: DDQN, QLearning, SARSA, MC.")
        return

    bracketer = VonNeumann1NeighPlusFoodDirectionBracket()  # Example bracket, can be changed
    env = SnakeEnv(render_mode='human')

    try:
        if model_name == 'DDQN':
            state_dim = bracketer.get_state_dim()
            model = DeepDoubleQLearning(action_space=env.action_space, state_dim=state_dim)
        elif model_name == 'QLearning':
            model = QLearning(action_space=env.action_space)
        elif model_name == 'SARSA':
            model = SARSA(action_space=env.action_space)
        elif model_name == 'MC':
            model = Montecarlo(action_space=env.action_space)

        model.upload(model_path)
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    print("Starting the game. Press Ctrl+C to exit.")
    model.play(env=env, bracketer=bracketer)

    print(f'Would you like to play again with the same model? (Y/N)')
    play_again = input().strip().upper()
    if play_again == 'Y':
        play()
    else:
        print("Exiting the game. Thank you for playing!")
        return



def main():
    # Ask the user to choose the Mode between "train"(T) and "play"(P)

    mode = input("Choose the mode (T for train, P for play): ").strip().upper()

    if mode not in ['T', 'P']:
        print("Invalid mode selected. Please choose 'T' for train or 'P' for play.")
        return

    if mode == 'T':
        train()
    else:
        play()


if __name__ == "__main__":
    main()


