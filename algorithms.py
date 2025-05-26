"""
    This work is developed for academic purpose by Bredariol Francesco, Savorgnan Enrico, Tic Ruben.
    This work is part of the final project for the 2024-2025 Reinforcement Learning course at the University of Trieste.
"""

import numpy as np
from collections import defaultdict, deque
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy


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


def argmax_over_dict_given_subkey_and_possible_action(dictionary, sub_key, possible_actions = [0, 1, 2, 3], default = [0, 1, 2, 3]):
    """
        Input : dictionary, sub_key
        Output : argmax of all the elements in the dictionary sharing the sub_key   
        This is implemented exactly with the subkey being the first two elements of the key of the dictionary, 
        in which key are tuple of three elements. [ key = (x, y, z), subkey = (x, y)] 
    """
    complete_subkey(dictionary, sub_key, default)
    sub_d = defaultdict(int)
    for d in dictionary:
        if tuple((d[0], d[1])) == sub_key and d[2] in possible_actions:
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
        This is implemented exactly with the subkey being the first two elements of the key of the dictionary, 
        in which key are tuple of three elements. [ key = (x, y, z), subkey = (x, y)] 
    """
    complete_subkey(dictionary, sub_key, default)
    sub_d = defaultdict(int)
    for d in dictionary:
        if tuple((d[0], d[1])) == sub_key:
            sub_d[d] = dictionary[d]
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


class DQN(nn.Module):
    """
    Deep Q_network for approximating Q-values.
    This is the standard PyTorch implementation. DO NOT MODIFY IT
    """
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, state):
        return self.fc(state)


class ReplayBuffer:
    """
    A simple replay buffer to store transitions.
    This is the standard PyTorch implementation. DO NOT MODIFY IT
    """
    def __init__(self, memory_size, batch_size):
        self.buffer = deque(maxlen=memory_size)
        self.batch_size = batch_size

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self):
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        return zip(*[self.buffer[i] for i in indices])

    def __len__(self):
        return len(self.buffer)


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

    def get_action_epsilon_greedy(self, s, eps, possible_actions=None):
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
        
    def get_action_greedy(self, s, possible_action = None):
        if possible_action is None:
            complete_subkey(self.Qvalues, s, default=[i for i in range (self.action_space)])
            a = argmax_over_dict_given_subkey(self.Qvalues, (*s, ), default=[i for i in range (self.action_space)])[2]
        else:
            a = argmax_over_dict_given_subkey_and_possible_action(self.Qvalues, (*s, ), possible_action, default=[i for i in range (self.action_space)])[2]
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
            eps = max(eps * 0.99, 0.05)  # Decay epsilon

        env.close()

    def play(self, env, bracketer):
        done = False
        keep = True

        state, _ = env.reset()
        state = bracketer.bracket(state)

        possible_action = [0, 1, 2, 3]
        last_action = None

        while not done and keep:
            if last_action is not None:
                possible_action = [0, 1, 2, 3]
                possible_action.remove(opposite_action(last_action))
            action = self.get_action_greedy(state, possible_action)
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
    
    def __init__(self, env, gamma, lr_v):
        super().__init__(env)
        self.gamma = gamma
        self.lr_v = lr_v
        self.returns = defaultdict(list)  # To store returns for each state-action pair

    def learning(self, env, eps, n_episodes, bracketer):

        performance_traj = np.zeros(n_episodes)

        state, _ = env.reset()

        for i in range(n_episodes):

            done = False
            keep = True

            env.reset()
            state = bracketer.bracket(env._get_obs())
            possible_actions = env.get_possible_actions(None)
            action = self.get_action_epsilon_greedy(state, eps=eps, possible_actions=possible_actions)

            episode = []

            while not done and keep:

                new_s, reward, done, trunc, inf = env.step(action)
                if inf != {}:
                    action = inf["act"]
                new_s = bracketer.bracket(new_s)

                # Keeps track of performance for each episode
                performance_traj[i] += reward

                episode.append((state, action, reward))

                possible_actions = env.get_possible_actions(action)
                new_a = self.get_action_epsilon_greedy(new_s, eps, possible_actions)

                state = new_s
                action = new_a

                keep = env.render()


            # After the episode ends, we update the Q-values
            G = 0
            for state, action, reward in reversed(episode):
                G = reward + self.gamma * G
                self.returns[(*state, action)].append(G)
                # Update the Q-value for the state-action pair
                self.Qvalues[(*state, action)] = np.mean(self.returns[(*state, action)])

            if i % 500 == 0:
                print(i)
            eps = max(eps * 0.99, 0.05)  # Decay epsilon

    
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




class DeepDoubleQLearning(RLAlgorithm):
    """
    This algorithm implements Deep Double Q-learning by using a Deep Q-Network (DQN) to approximate the Q-values.
    """

    def __init__(self, action_space, gamma, lr_v, state_dim, batch_size=32, memory_size=10000, target_update_freq=1000, device='cpu'):
        super().__init__(action_space)
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update_freq = target_update_freq
        self.memory = ReplayBuffer(self.memory_size, self.batch_size)

        self.gamma = gamma
        self.lr_v = lr_v
        self.device = torch.device(device)

        self.dqn_online = DQN(state_dim=state_dim, action_dim=action_space).to(self.device)
        self.dqn_target = DQN(state_dim=state_dim, action_dim=action_space).to(self.device)

        self.optimizer = torch.optim.Adam(self.dqn_online.parameters(), lr=self.lr_v)


        # Initialize the target network with the same weights as the online network
        self.dqn_target.load_state_dict(self.dqn_online.state_dict())
        self.dqn_target.eval()




    def get_action_epsilon_greedy(self, state, eps, possible_actions=None):
        if np.random.rand() < eps:
            if possible_actions is None or len(possible_actions) == 0:
                # Fallback if no possible actions are specified (should ideally not happen in constrained envs)
                # Or if the environment truly allows all actions from self.action_space
                return np.random.choice(self.action_space) # Assumes self.action_space is an int for range, or a list
            return np.random.choice(possible_actions)
        else: # Exploit
            s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values_all = self.dqn_online(s_tensor)[0]  # Q-values for state s, shape [action_dim]

            if possible_actions is None or len(possible_actions) == 0:
                return q_values_all.argmax().item()
            else:
                # Select best action from the subset of possible_actions
                q_values_subset = q_values_all[possible_actions]
                best_action_idx_in_subset = q_values_subset.argmax().item()
                return possible_actions[best_action_idx_in_subset]


    def get_action_greedy(self, state, possible_actions=None):
        # Se vuoi rispettare possibili vincoli:
        qvals = self.dqn_online(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
        best = qvals.argmax(dim=1).item()
        if possible_actions is None or best in possible_actions:
            return best
        # altrimenti scegli il massimo tra quelli permessi
        sub_q = qvals[0, possible_actions]
        return possible_actions[sub_q.argmax().item()]



    def single_step_update(self, s, a, r, new_s, new_a, done):
        """
        This method is a bit more technical.
        Each time it is called, it stores the transition in memory.
        If the memory is full, it samples a batch of transitions and updates the online DQN.
        The target DQN is updated every `self.target_update_freq` steps.
        """
        self.memory.append((s, a, r, new_s, done))

        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Compute Q-values for the current states using the ONLINE DQN
        current_q_values = self.dqn_online(states).gather(1, actions).squeeze(1)

        # Compute Q-values for the next states using the TARGET DQN
        with torch.no_grad():
            # Get the best action given the next states using the ONLINE DQN
            next_actions = self.dqn_online(next_states).argmax(1, keepdim=True)
            # Compute the Q-values for the next states using the TARGET DQN
            next_q_values = self.dqn_target(next_states).gather(1, next_actions).squeeze(1)

            # Compute the expected q_values
            target_q_values = rewards + self.gamma * next_q_values * (1-dones)

        # Compute the loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Update the online DQN
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target DQN every `self.target_update_freq` iterations
        if self.iterations % self.target_update_freq == 0:
            self.dqn_target.load_state_dict(self.dqn_online.state_dict())

        # Increment the iteration counter
        self.iterations += 1


if __name__ == "__main__":
    q = {(0, 0, 1) : 1, (0, 0, 2) : 2, (0, 1, 1) : 3}
    print(argmax_over_dict_given_subkey(q, (0, 2), [1, 2]))