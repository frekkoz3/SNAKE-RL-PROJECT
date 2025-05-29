import numpy as np
import scipy
from collections import defaultdict

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
        If the key is composed as follow : (*subkey, other) and so we can search for the 
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
