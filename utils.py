import numpy as np
import scipy
from collections import defaultdict

import algorithms as alg
from snake_environment import SnakeEnv

from IPython.display import clear_output


def action_name(index):
    dict = {0 : "south", 1 : "east", 2 : "north", 3 : "west"}
    return dict[index]


def argmax_over_dict(dictionary):
    """
        Input : dictionary
        Output : argmax of the dictionary
        This function retrieve the argmax in a dictionary.
    """
    d = dictionary
    # Convert keys and values to lists
    keys = list(d.keys())
    values = np.array(list(d.values()))
    # Find the key with the max value
    max_key = keys[np.argmax(values)]
    return max_key


def argmax_over_dict_given_subkey(dictionary, sub_key, default = [0, 1, 2, 3]):
    """
        Input : dictionary, sub_key
        Output : argmax of all the elements in the dictionary sharing the sub_key   
        We are assuming that the subkey must be the first part of the real key, ergo 
        If the key is composed as follow : (*subkey, other) and so we can search for the 
        subkey in the key just by looking at the first |subkey| components of the key.
    """
    sub_d = defaultdict(int)
    for d in dictionary:
        if tuple([d[i] for i in range (len(sub_key))]) == sub_key:
            sub_d[d] = dictionary[d]
    if sub_d == {}: # this is used if a new state is sampled (so for all the actions 0 is given as value)
        for d in default:
            sub_d[(*sub_key, d)] = 0
    return argmax_over_dict(sub_d)


def argmax_over_dict_given_subkey_and_possible_action(dictionary, sub_key, possible_actions = [0, 1, 2, 3], default = [0, 1, 2, 3]):
    """
        Input : dictionary, sub_key
        Output : argmax of all the elements in the dictionary sharing the sub_key  
        We are assuming that the subkey must be the first part of the real key, ergo 
        If the key is composed as follow : (*subkey, other) and so we can search for the 
        subkey in the key just by looking at the first |subkey| components of the key. 
    """
    complete_subkey(dictionary, sub_key, default)
    sub_d = defaultdict(int)
    for d in dictionary:
        if tuple([d[i] for i in range (len(sub_key))]) == sub_key and d[-1] in possible_actions:
            sub_d[d] = dictionary[d]
    return argmax_over_dict(sub_d)


def max_over_dict(dictionary):
    """
        Input : dictionary
        Output : argmax of the dictionary
        This function retrieve the max in a dictionary.
    """
    d = dictionary
    # Convert the values to a list
    values = np.array(list(d.values()))
    # Find the max value
    return np.max(values)


def max_over_dict_given_subkey(dictionary, sub_key, default = [0, 1, 2, 3]):
    """
        Input : dictionary, sub_key
        Output : argmax of all the elements in the dictionary sharing the sub_key   
        We are assuming that the subkey must be the first part of the real key, ergo 
        If the key is composed as follow : (*subkey, other) and so we can search for the 
        subkey in the key just by looking at the first |subkey| components of the key.
    """
    complete_subkey(dictionary, sub_key, default)
    sub_d = defaultdict(int)
    for d in dictionary:
        if tuple([d[i] for i in range (len(sub_key))]) == sub_key:
            sub_d[d] = dictionary[d]
    return max_over_dict(sub_d)


def complete_subkey(dictionary, sub_key, default = [0, 1, 2, 3]):
    """
        This function is used to complete a Qstate(s, a) dictionary.
        We are assuming that the subkey must be the first part of the real key, ergo 
        If the key is composed as follows : (*subkey, other) and so we can search for the
        subkey in the key just by looking at the first |subkey| components of the key.
    """
    s = 0
    for d in dictionary:
        if tuple([d[i] for i in range (len(sub_key))]) == sub_key:
            s += 1
    if s < len(default):
        for d in default:
            dictionary[(*sub_key, d)] = dictionary[(*sub_key, d)]



def opposite_action(action):
    """
        Input : an action in the space [0, 1, 2, 3]
        Output : an action in the space [0, 1, 2, 3]
        This function returns the opposite action to the one already taken.
        The rule is the following : 0 and 1 are opposite, 2 and 3 are opposite
    """
    return {0: 1, 1: 0, 2: 3, 3: 2}[action]


def discount_cumsum(x, discount):
    """
    magic for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0, x1, x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def get_model_average_performance(model_name, action_space, gamma, lr_v, model_path, bracketer, num_episodes=100, render_mode='nonhuman', **kwargs):
    """
    Computes the average performance of a given model over a number of episodes.
    In particular, it computes the average reward and the average number of food eaten.
    """

    assert num_episodes > 0, "Number of episodes must be greater than 0."
    model_types = ['DDQL', 'QLearning', 'SARSA', 'MC', 'AtariDQL']
    env = SnakeEnv(render_mode=render_mode)

    if model_name not in model_types:
        print(f'Model {model_name} is not supported. Supported models are: {model_types}.\nReturning...')
        return None

    # Define the model given model_name
    if model_name == 'DDQL':
        if len(kwargs) < 5:
            print('Not enough parameters for Deep Double Q-Learning. Returning...')
            return None

        state_dim = kwargs['state_dim']
        batch_size = kwargs['batch_size']
        memory_size = kwargs['memory_size']
        target_update_freq = kwargs['target_update_freq']
        device = kwargs['device']

        model = alg.DeepDoubleQLearning(action_space=action_space, state_dim=state_dim, gamma=gamma, lr_v=lr_v, batch_size=batch_size,
                                        memory_size=memory_size, target_update_freq=target_update_freq, device=device)

    elif model_name == 'AtariDQL':
        if len(kwargs) < 6:
            print('Not enough parameters for Atari-like Deep Double Q-Learning. Returning...')
            return None

        batch_size = kwargs['batch_size']
        memory_size = kwargs['memory_size']
        target_update_freq = kwargs['target_update_freq']
        device = kwargs['device']
        width = kwargs['width']
        height = kwargs['height']
        n_layers = kwargs['n_layers']

        model = alg.AtariDeepQLearning(action_space=action_space, gamma=gamma, lr_v=lr_v, batch_size=batch_size, memory_size=memory_size, target_update_freq=target_update_freq, device=device, width=width, height=height, n_layers=n_layers)

    elif model_name  == 'QLearning':
        model = alg.QLearning(action_space=action_space, gamma=gamma, lr_v=lr_v)

    elif model_name == 'SARSA':
        model = alg.SARSA(action_space=action_space, gamma=gamma, lr_v=lr_v)

    elif model_name == 'MC':
        model = alg.Montecarlo(action_space=action_space, gamma=gamma, lr_v=lr_v)

    # Upload the model if it exists
    try:
        model.upload(model_path)
    except Exception as e:
        print(f'Error uploading model {model_name}. Exception {e}. Returning...')
        return None

    total_rewards = 0
    total_food = 0

    for episode in range(num_episodes):
        clear_output(wait=False)
        print(f'Episode {episode + 1}/{num_episodes}')
        total_rewards += model.play(env=env, bracketer=bracketer)
        total_food += env.get_score()

    avg_reward = total_rewards / num_episodes
    avg_eaten_food = total_food / num_episodes

    return avg_reward, avg_eaten_food



