"""
    This work is developed for academic purpose by Bredariol Francesco, Savorgnan Enrico, Tic Ruben.
    This work is part of the final project for the 2024-2025 Reinforcement Learning course at the University of Trieste.
"""

import numpy as np
from collections import defaultdict
import pickle 

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
        This is implemented exactly with the subkey being the first two elements of the key of the dictionary, 
        in which key are tuple of three elements. [ key = (x, y, z), subkey = (x, y)] 
    """
    sub_d = defaultdict(int)
    for d in dictionary:
        if tuple((d[0], d[1])) == sub_key:
            sub_d[d] = dictionary[d]
    if sub_d == {}: # this is used if a new state is sampled (so for all the actions 0 is given as value)
        for d in default:
            sub_d[(*sub_key, d)] = 0
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
        This is implemented exactly with the subkey being the first two elements of the key of the dictionary, 
        in which key are tuple of three elements. [ key = (x, y, z), subkey = (x, y)] 
    """
    sub_d = defaultdict(int)
    for d in dictionary:
        if tuple((d[0], d[1])) == sub_key:
            sub_d[d] = dictionary[d]
    if sub_d == {}: # this is used if a new state is sampled (so for all the actions 0 is given as value)
        for d in default:
            sub_d[(*sub_key, d)] = 0
    return max_over_dict(sub_d)

def complete_subkey(dictionary, sub_key, default = [0, 1, 2, 3]):
    """
        This function is used to complete a Qstate(s, a) dictionary.
    """
    s = 0
    for d in dictionary:
        if tuple((d[0], d[1])) == sub_key:
            s += 1
    if s < len(default):
        for d in default:
            dictionary[(*sub_key, d)] = dictionary[(*sub_key, d)] 

class RLAlgorithm:
    """
        This is the generic class for an RL Algorithm using gymnasium environment
    """

    def __init__(self, action_space):
        self.Qvalues = defaultdict(int)
        self.action_space = action_space
        self.iterations = 0 # This is used to count how many iteration until convergence

    def single_step_update(self):
        """
            This is the function where to take one trajectory in the environment
        """
        pass

    def get_action_epsilon_greedy(self, s, eps):
        """
        Chooses action at random using an epsilon-greedy policy wrt the current Q(s,a).
        """
        complete_subkey(self.Qvalues, s, default=[i for i in range (self.action_space)])
        ran = np.random.rand()
        
        if (ran < eps):
            prob_actions = np.ones(self.action_space)/self.action_space
        else:
            prob_actions = np.zeros(self.action_space)
            prob_actions[argmax_over_dict_given_subkey(self.Qvalues, (*s, ), default=[i for i in range (self.action_space)])[2]] = 1
            
        # take one action from the array of actions with the probabilities as defined above.
        a = np.random.choice(self.action_space, p=prob_actions)
        return a 
        
    def get_action_greedy(self, s):
        complete_subkey(self.Qvalues, s, default=[i for i in range (self.action_space)])
        a = argmax_over_dict_given_subkey(self.Qvalues, (*s, ), default=[i for i in range (self.action_space)])[2]
        return a

    def save(self, path):
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(self.Qvalues, f)
    
    def upload(self, path):
        with open(f"{path}.pkl", 'rb') as f:
            self.Qvalues = pickle.load(f)

    def learning(self, env, eps, n_episodes, bracketer):
        
        performance_traj = np.zeros(n_episodes)

        state, _ = env.reset()

        for i in range(n_episodes):

            done = False
            keep = True

            env.reset()
            state = bracketer.bracket(env._get_obs())
            action = self.get_action_epsilon_greedy(state, eps = eps)

            while not done and keep:

                new_s, reward, done, trunc, inf = env.step(action)
                if inf != {}:
                    action = inf["act"]
                new_s = bracketer.bracket(new_s)
                
                # Keeps track of performance for each episode
                performance_traj[i] += reward
                
                new_a = self.get_action_epsilon_greedy(new_s, eps)

                self.single_step_update(state, action, reward, new_s, new_a, done)
                
                action = new_a
                state = new_s

                keep = env.render()

            if i % 500 == 0:
                print(i)

        env.close()

    def play(self, env, bracketer):
        done = False
        keep = True

        state, _ = env.reset()
        state = bracketer.bracket(state)

        while not done and keep:
            action = self.get_action_greedy(state)
            state, reward, done, trunc, inf = env.step(action)
            state = bracketer.bracket(state)
            keep = env.render()

        env.close()

    def __str__(self):
        s = ""
        for d in self.Qvalues:
            s = s + f"State ({d[0]}, {d[1]}), Action {d[2]} : Value {self.Qvalues[d]}\n"
        return s
    
class Montecarlo(RLAlgorithm):
    
    def __init__(self, env):
        super().__init__(env)
    
class SARSA(RLAlgorithm):
    
    def __init__(self, action_space, gamma=1, lr_v=0.01):
        super().__init__(action_space)
        # the discount factor
        self.gamma = gamma
        # the learning rate
        self.lr_v = lr_v
    
    def single_step_update(self, s, a, r, new_s, new_a, done):
        """
        Uses a single step to update the values, using Temporal Difference for Q values.
        Employs the EXPERIENCED action in the new state  <- Q(S_new, A_new).
        """
        if done:
            # SARSA: deltaQ = R - Q(s,a)
            deltaQ = r - self.Qvalues[(*s, a)]
        else:
            # SARSA: deltaQ = R + gamma*Q(new_s, new_a) - Q(s,a)
            deltaQ = r + self.gamma*self.Qvalues[(*new_s, new_a)] - self.Qvalues[(*s, a)]
            
        self.Qvalues[(*s, a)] += self.lr_v * deltaQ

class QLearning(RLAlgorithm):

    def __init__(self, action_space, gamma=1, lr_v=0.01):
        super().__init__(action_space)
        # the discount factor
        self.gamma = gamma
        # the learning rate
        self.lr_v = lr_v
    
    def single_step_update(self, s, a, r, new_s, new_a, done):
        """
        Uses a single step to update the values, using Temporal Difference for Q values.
        Employs the EXPERIENCED action in the new state  <- Q(S_new, A_new).
        """
        if done:
            # QLearning: deltaQ = R - Q(s,a)
            deltaQ = r - self.Qvalues[(*s, a)]
        else:
            # QLearning: deltaQ = R + gamma*maxQ(*new_s,) - Q(s,a)
            deltaQ = r + self.gamma*max_over_dict_given_subkey(self.Qvalues, (*new_s,), default=[i for i in range (self.action_space)]) - self.Qvalues[(*s, a)]
            
        self.Qvalues[(*s, a)] += self.lr_v * deltaQ

if __name__ == "__main__":
    q = {(0, 0, 1) : 1, (0, 0, 2) : 2, (0, 1, 1) : 3}
    print(argmax_over_dict_given_subkey(q, (0, 2), [1, 2]))