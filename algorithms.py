"""
    This work is developed for academic purpose by Bredariol Francesco, Savorgnan Enrico, Tic Ruben.
    This work is part of the final project for the 2024-2025 Reinforcement Learning course at the University of Trieste.
"""

import numpy as np
from collections import defaultdict, deque
import pickle
from IPython.display import clear_output
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import signal
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from eligibility_traces import *

class DQN(nn.Module):
    """
    Deep Q_network for approximating Q-values.
    This is the standard PyTorch implementation. DO NOT MODIFY IT
    """
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, state):
        return self.fc(state)


class ConvolutionalDQN(nn.Module):

    def __init__(self, action_dim, n_layers, height, width):
        super(ConvolutionalDQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=n_layers, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * ((height - 3) // 2) * ((width - 3) // 2), 32),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        x is expected to be of shape (batch_size, state_dim, height, width)
        """
        x = self.layers(x)
        return x


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
        """
            The action space is required to be an integer which state the maximum index for an action (from 0 up to the action space).
        """
        self.Qvalues = defaultdict(int)
        self.action_space = action_space
        self.iterations = 0         # This is used to count how many iteration until convergence
        signal.signal(signal.SIGINT, self.handle_sigint)        # for printing data at premature exit
        
    def handle_sigint(self, signum, frame):

        self.print_results_learning()
        sys.exit(0)

    def single_step_update(self, s, a, r, new_s, new_a, done):
        """
            This is the function where to take one trajectory in the environment
        """
        pass

    def single_episode_update(self, episode):
        """
            This function does an update with a whole trajectory
            Chooses action at random using an epsilon-greedy policy wrt the current Q(s,a).
            It also automatically complete the dictionary for the state with all possible actions.

        """
        pass

    def get_action_during_learning(self, s, possible_actions=None):
        """
            Chooses action depending on the current policy used during learning
        """
        pass
        
    def get_action_during_evaluation(self, s, possible_action = None):
        """
            Chooses action depending on the policy used during final evaluation
        """
        pass

    def get_action_greedy(self, s, possible_action = None):
        """
            Return the action from the greedy policy.
            If there are no possible action it firstly complete the subkey and then excract one action.
        """
        if possible_action is None:
            complete_subkey(self.Qvalues, s, default=[i for i in range (self.action_space)])
            a = argmax_over_dict_given_subkey(self.Qvalues, (*s, ), default=[i for i in range (self.action_space)])[-1]
        else:
            a = argmax_over_dict_given_subkey_and_possible_action(self.Qvalues, (*s, ), possible_action, default=[i for i in range (self.action_space)])[-1]
        return a

    def save(self, path):
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(self.Qvalues, f)
    
    def upload(self, path):
        with open(f"{path}.pkl", 'rb') as f:
            self.Qvalues = pickle.load(f)

    def learning(self, env, epsilon_schedule, n_episodes, bracketer):
        
        self.bracketer = bracketer
        self.performance_traj = np.zeros(n_episodes)

        state, _ = env.reset()

        for i in range(n_episodes):
            
            if epsilon_schedule is not None:    #only if epsilon is used
                self.eps = epsilon_schedule.decay() # it decays over episodes

            done = False
            keep = True

            env.reset()
            state = self.bracketer.bracket(env._get_obs())
            
            action = self.get_action_during_learning(state)

            episode = []

            while not done and keep:

                new_s, reward, done, trunc, inf = env.step(action)
                if inf != {}:       #if inf is not empty => the action performed is NOT the action intended(it was unfeasible)
                    action = inf["act"]
                new_s = self.bracketer.bracket(new_s)
                
                # Keeps track of performance for each episode
                self.performance_traj[i] += reward

                episode.append((state, action, reward))

                possible_actions = env.get_possible_actions(action)
                new_a = self.get_action_during_learning(new_s, possible_actions=possible_actions)

                if self.single_step_update.__func__ is not RLAlgorithm.single_step_update:   #single_step_update was overridden
                    self.single_step_update(state, action, reward, new_s, new_a, done)
                
                action = new_a
                state = new_s

                keep = env.render()
            
            if self.single_episode_update.__func__ is not RLAlgorithm.single_episode_update:   #single_episode_update was overidden
                self.single_episode_update(episode)

            if i % 100 == 0 and i != 0:
                clear_output(wait=False)
                if epsilon_schedule is not None:    #only if epsilon is used
                    print(f"Episode {i}/{n_episodes} : epsilon {self.eps} : Average performance {np.mean(self.performance_traj[i-100:i])}")
                else:
                    print(f"Episode {i}/{n_episodes} : Average performance {np.mean(self.performance_traj[i-100:i])}")
        
        self.print_results_learning()

        env.close()

    def play(self, env, bracketer):
        done = False
        keep = True
        total_reward = 0

        state, _ = env.reset()
        state = bracketer.bracket(state)

        action = None

        while not done and keep:
            possible_actions = env.get_possible_actions(action)
            action = self.get_action_during_evaluation(state, possible_actions)
            state, reward, done, trunc, inf = env.step(action)
            state = bracketer.bracket(state)
            keep = env.render()

            total_reward += reward

        env.close()

        return total_reward

    def print_q_values(self, bracketer):
        """
            Deprecated. To be redifined.
        """
        print("\033[93m[WARNING] This method is deprecated and does not actually works as it should.\033[0m")
        print(str(self.Qvalues))

    def print_results_learning(self):
        """
            At the and of learining, this function is executed. Print here information on learining
        """
        print("\n\nLearning finished\n\n")
        for i in range(len(self.performance_traj)):
            if i % 100 == 0 and i != 0:
                print(f"Episode {i} : Average performance {np.mean(self.performance_traj[i-100:i])}")
        
        n_moving_average = 60
        average_performance = np.convolve(self.performance_traj, np.ones(n_moving_average)/n_moving_average, mode='valid')

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=average_performance,
            mode='lines',
            name='Average Performance'
        ))

        fig.update_layout(
            title='Performance for episode',
            xaxis_title='Episode',
            yaxis_title='Performance',
            template='plotly_white'
        )

        fig.show()


    def __str__(self):
        return ""

    def name(self):
        return "Generic RL Algorithm"


class Montecarlo(RLAlgorithm):
    
    def __init__(self, action_space, gamma, lr_v):
        super().__init__(action_space)
        self.gamma = gamma
        self.lr_v = lr_v
        self.returns = defaultdict(list)  # To store returns for each state-action pair

    def get_action_during_learning(self, s, possible_actions=None):
        
        #   Chooses action at random using an epsilon-greedy policy wrt the current Q(s,a).
        #   It also automatically complete the dictionary for the state with all possible actions.
       
        complete_subkey(self.Qvalues, s, default=[i for i in range (self.action_space)])
        ran = np.random.rand()
        
        if (ran < self.eps):
            prob_actions = np.ones(self.action_space)/self.action_space
        else:
            prob_actions = np.zeros(self.action_space)
            prob_actions[argmax_over_dict_given_subkey(self.Qvalues, (*s, ), default=[i for i in range (self.action_space)])[-1]] = 1
            
        # take one action from the array of actions with the probabilities as defined above.
        a = np.random.choice(self.action_space, p=prob_actions)
        return a 

    def get_action_during_evaluation(self, s, possible_action = None):
        #   Return the action from the greedy policy.
        #  If there are no possible action it firstly complete the subkey and then excract one action.
        
        if possible_action is None:
            complete_subkey(self.Qvalues, s, default=[i for i in range (self.action_space)])
            a = argmax_over_dict_given_subkey(self.Qvalues, (*s, ), default=[i for i in range (self.action_space)])[-1]
        else:
            a = argmax_over_dict_given_subkey_and_possible_action(self.Qvalues, (*s, ), possible_action, default=[i for i in range (self.action_space)])[-1]
        return a
    
    def single_episode_update(self, episode):
        # After the episode ends, we update the Q-values
        G = 0
        self.returns.clear()
        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G
            self.returns[(*state, action)].append(G)
            # Update the Q-value for the state-action pair
            self.Qvalues[(*state, action)] = np.mean(self.returns[(*state, action)])

    def name(self):
        return "Montecarlo"


class SARSA(RLAlgorithm):
    
    def __init__(self, action_space, gamma=1, lr_v=0.01):
        super().__init__(action_space)
        # the discount factor
        self.gamma = gamma
        # the learning rate
        self.lr_v = lr_v

    def get_action_during_learning(self, s, possible_actions=None):
        
        #   Chooses action at random using an epsilon-greedy policy wrt the current Q(s,a).
        #   It also automatically complete the dictionary for the state with all possible actions.
       
        complete_subkey(self.Qvalues, s, default=[i for i in range (self.action_space)])
        ran = np.random.rand()
        
        if (ran < self.eps):
            prob_actions = np.ones(self.action_space)/self.action_space
        else:
            prob_actions = np.zeros(self.action_space)
            prob_actions[argmax_over_dict_given_subkey(self.Qvalues, (*s, ), default=[i for i in range (self.action_space)])[-1]] = 1
            
        # take one action from the array of actions with the probabilities as defined above.
        a = np.random.choice(self.action_space, p=prob_actions)
        return a 

    
    def get_action_during_evaluation(self, s, possible_action = None):
        #   Return the action from the greedy policy.
        #  If there are no possible action it firstly complete the subkey and then excract one action.
        
        if possible_action is None:
            complete_subkey(self.Qvalues, s, default=[i for i in range (self.action_space)])
            a = argmax_over_dict_given_subkey(self.Qvalues, (*s, ), default=[i for i in range (self.action_space)])[-1]
        else:
            a = argmax_over_dict_given_subkey_and_possible_action(self.Qvalues, (*s, ), possible_action, default=[i for i in range (self.action_space)])[-1]
        return a

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
    
    def name(self):
        return "SARSA"


class SARSALambda(RLAlgorithm):

    def __init__(self, lambda_value, action_space, gamma=1, lr_v=0.01):
        super().__init__(action_space)
        # the discount factor
        self.gamma = gamma
        # the learning rate
        self.lr_v = lr_v
        # eligibility traces
        self.e = Eligibility(lambda_value, gamma)

    def get_action_during_learning(self, s, possible_actions=None):
        
        #   Chooses action at random using an epsilon-greedy policy wrt the current Q(s,a).
        #   It also automatically complete the dictionary for the state with all possible actions.
       
        complete_subkey(self.Qvalues, s, default=[i for i in range (self.action_space)])
        ran = np.random.rand()
        
        if (ran < self.eps):
            prob_actions = np.ones(self.action_space)/self.action_space
        else:
            prob_actions = np.zeros(self.action_space)
            prob_actions[argmax_over_dict_given_subkey(self.Qvalues, (*s, ), default=[i for i in range (self.action_space)])[-1]] = 1
            
        # take one action from the array of actions with the probabilities as defined above.
        a = np.random.choice(self.action_space, p=prob_actions)
        return a 


    def get_action_during_evaluation(self, s, possible_action = None):
        #   Return the action from the greedy policy.
        #  If there are no possible action it firstly complete the subkey and then excract one action.
        
        if possible_action is None:
            complete_subkey(self.Qvalues, s, default=[i for i in range (self.action_space)])
            a = argmax_over_dict_given_subkey(self.Qvalues, (*s, ), default=[i for i in range (self.action_space)])[-1]
        else:
            a = argmax_over_dict_given_subkey_and_possible_action(self.Qvalues, (*s, ), possible_action, default=[i for i in range (self.action_space)])[-1]
        return a

    def single_step_update(self, s, a, r, new_s, new_a, done):
        if done:
            # SARSA(λ): δ = R - Q(s,a)
            deltaQ = r - self.Qvalues[(*s, a)]
        else:
            deltaQ = r + self.gamma * self.Qvalues[(*new_s, new_a)] - self.Qvalues[(*s, a)]

        # Decay the traces
        self.e.decay()
        # Update the trace for the current state-action pair
        self.e.update((*s, a))

        self.Qvalues[(*s, a)] += self.lr_v * deltaQ * self.e.traces[(*s, a)]

    def name(self):
        return "SARSALambda"


class QLearning(RLAlgorithm):

    def __init__(self, action_space, gamma=1, lr_v=0.01):
        super().__init__(action_space)
        # the discount factor
        self.gamma = gamma
        # the learning rate
        self.lr_v = lr_v
    
    def get_action_during_learning(self, s, possible_actions=None):
        
        #   Chooses action at random using an epsilon-greedy policy wrt the current Q(s,a).
        #   It also automatically complete the dictionary for the state with all possible actions.
       
        complete_subkey(self.Qvalues, s, default=[i for i in range (self.action_space)])
        ran = np.random.rand()
        
        if (ran < self.eps):
            prob_actions = np.ones(self.action_space)/self.action_space
        else:
            prob_actions = np.zeros(self.action_space)
            prob_actions[argmax_over_dict_given_subkey(self.Qvalues, (*s, ), default=[i for i in range (self.action_space)])[-1]] = 1
            
        # take one action from the array of actions with the probabilities as defined above.
        a = np.random.choice(self.action_space, p=prob_actions)
        return a 

    def get_action_during_evaluation(self, s, possible_action = None):
        #   Return the action from the greedy policy.
        #  If there are no possible action it firstly complete the subkey and then excract one action.
        
        if possible_action is None:
            complete_subkey(self.Qvalues, s, default=[i for i in range (self.action_space)])
            a = argmax_over_dict_given_subkey(self.Qvalues, (*s, ), default=[i for i in range (self.action_space)])[-1]
        else:
            a = argmax_over_dict_given_subkey_and_possible_action(self.Qvalues, (*s, ), possible_action, default=[i for i in range (self.action_space)])[-1]
        return a

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
    
    def name(self):
        return "QLearning"


class DeepDoubleQLearning(RLAlgorithm):
    """
    This algorithm implements Deep Double Q-learning by using Deep Q-Networks (DQN) to approximate the Q-values.
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


    def get_action_during_learning(self, state, possible_actions=None):
        if np.random.rand() < self.eps: # Explore
            if possible_actions is None:
                possible_actions = list(range(self.action_space))
            return np.random.choice(possible_actions)
        else: # Exploit
            if possible_actions is None:
                possible_actions = list(range(self.action_space))
            s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values_all = self.dqn_online(s_tensor)[0]
                # this s is in the format (state, action) ? frekko

                # Select best action from the subset of possible_actions
                q_values_subset = q_values_all[possible_actions]
                best_action_idx_in_subset = q_values_subset.argmax().item()

                return possible_actions[best_action_idx_in_subset]


    def get_action_during_evaluation(self, state, possible_actions=None):
        qvals = self.dqn_target(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
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

        if len(self.memory) < 2*self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Compute Q-values for the current states using the ONLINE DQN
        current_q_values = self.dqn_online(states).gather(1, actions).squeeze(1)

        # Get the best action given the next states using the ONLINE DQN
        next_actions = self.dqn_online(next_states).argmax(1, keepdim=True)

        # Compute Q-values for the next states using the TARGET DQN
        with torch.no_grad():
            next_q_values = self.dqn_target(next_states).gather(1, next_actions).squeeze(1)
            # Compute the target q_values
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            
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


    def save(self, path):
        super().save(path)
        torch.save(self.dqn_target.state_dict(), path)


    def upload(self, path, map_location='cpu'):
        self.dqn_target.load_state_dict(torch.load(path, map_location=map_location))
        self.dqn_target.eval()


    def name(self):
        return "DDQL"


class PolicyGradient(RLAlgorithm):

    def __init__(self, action_space, gamma, lr_a, bracketer):
        super().__init__(action_space)

        self.gamma = gamma
        self.lr_a = lr_a
        
        
        self.parameters = np.zeros((2**bracketer.get_state_dim()) * 4)  #this array holds the parameters theta
        self.powers = 2 ** np.arange(bracketer.get_state_dim())         #for index convertion in get_index_sa

    def index_sa(self, sa):
            #this utility performes a one to one mapping between state-action pairs and index for the arrays

        index = np.dot(self.powers, sa[:-1])    #the state is a string of 0 and 1, we get the base2 number
        index = index*4 + sa[-1]                #every state has 4 actions, found in the last element of (*state, action)
        
        return int(index)

    def policy(self, action, state):
            # This functions performes the Softmax, obtaining prob(A = action | S = state ) = exp( parameters(s, a) )  /  (normaliz. factor over actions)

        return np.exp(self.parameters[self.index_sa((*state, action))])  /  sum(  np.exp(self.parameters[self.index_sa((*state, a))])  for a in range(self.action_space)  )

    def get_action_during_learning(self, state, possible_actions=None):     
            #this function takes an action given the actual policy
    
        prob_a = np.zeros(self.action_space)

        for a in range(self.action_space):
            prob_a[a] = self.policy(a, state)       #we construct the probability distribution of actions following the policy in the current state 
        
        chosen_action = np.random.choice(range(self.action_space), p=prob_a)     #we choose an action following the distribution "prob_a"
        return chosen_action
        
    def get_action_during_evaluation(self, state, possible_actions=None):
        #no difference from learning for Policy Gradient algorithms
        return self.get_action_during_learning(state, possible_actions=possible_actions)

    def save(self, path):
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(self.parameters, f)
    
    def upload(self, path):
        with open(f"{path}.pkl", 'rb') as f:
            self.parameters = pickle.load(f)
     
    
class ActorOnly(PolicyGradient):

    def single_episode_update(self, episode):
        # After the episode ended, we update the policy
        # The policy is parametrized via soft-max, and theta is a vector with entries for every couple bin-action
        G = 0
        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G

            # Update the parameters
            #                                       1 - pi(a|s)   if a is the performed action
            # we recall that grad log(pi(a|s)) = 
            #                                       - pi(a|s)     else

            for a in range(self.action_space):
                if a == action:  #for the performed action
                    self.parameters[self.index_sa((*state, a))] += self.lr_a*G*(1-self.policy(action, state))
                else:
                    self.parameters[self.index_sa((*state, a))] += self.lr_a*G*( -self.policy(action, state))

            #for numerical stability, we subtract from the parameters their max value
            max_parameter = np.max( [self.parameters[self.index_sa((*state, a))]   for a in range(self.action_space)])
            for a in range(self.action_space):
                self.parameters[self.index_sa((*state, a))] -= float(max_parameter)

class ActorCritic(PolicyGradient):

    def __init__(self, action_space, gamma, lr_a, lr_v, bracketer):
        super().__init__(action_space, gamma, lr_a, bracketer)

        self.lr_v = lr_v
        self.value = np.zeros(2**bracketer.get_state_dim())     #we store the value function for all states

    def index_s(self, s):
            #this utility performes a one to one mapping between states and index for the arrays
        index = np.dot(self.powers, s)    #the state in a string of 0 and 1, we get the base2 number
        return int(index)

    def single_step_update(self, s, a, r, new_s, new_a, done):
        # After each step, we update the value and the policy
        # The policy is parametrized via soft-max, and theta is a vector with entries for all pair state-action

        # critic update
        if done:   
            delta = r + 0 - self.value[self.index_s(s)]
        else:
            delta = r + self.gamma*self.value[self.index_s(new_s)]    -    self.value[self.index_s(s)]

        self.value[self.index_s(s)] += self.lr_v*delta

        # actor update (parameters)
        #                                       1 - pi(a|s)   if a is the performed action
        # we recall that grad log(pi(a|s)) = 
        #                                       - pi(a|s)     else
        for ap in range(self.action_space):
            if ap == a:  #for the performed action
                self.parameters[self.index_sa((*s, ap))] += self.lr_a*delta*(1-self.policy(a, s))
            else:
                self.parameters[self.index_sa((*s, ap))] += self.lr_a*delta*( -self.policy(a, s))

        # For numerical stability, we subtract from the parameters their max value
        max_parameter = np.max( [self.parameters[self.index_sa((*s, ap))]   for ap in range(self.action_space)])
        for ap in range(self.action_space):
            self.parameters[self.index_sa((*s, ap))] -= float(max_parameter)


class GAE(PolicyGradient):
    def __init__(self, action_space, gamma, lr_a, lr_v, Lambda, bracketer):
        super().__init__(action_space, gamma, lr_a, bracketer)

        self.lr_v = lr_v
        self.value = np.zeros(2**bracketer.get_state_dim())     #we store the value function for all states
        self.Lambda = Lambda

    def index_s(self, s):
            #this utility performes a one to one mapping between states and index for the arrays
        index = np.dot(self.powers, s)    #the state in a string of 0 and 1, we get the base2 number
        return int(index)

    def single_episode_update(self, episode):
        # After the episode ended, we update the value and the policy
        # The policy is parametrized via soft-max, and theta is a vector with entries for every couple bin-action

        GAE = 0       # GAE is a variable used to store the Generalized Advantage Estimator for all t 
        
        for t in range(len(episode) -1, -1, -1):
            
            state, action, reward = episode[t]

            if t != len(episode) -1:
                next_state = episode[t+1][0]

                delta = reward + self.gamma*self.value[self.index_s(next_state)] - self.value[self.index_s(state)]
            else:

                delta = reward - self.value[self.index_s(state)]

            GAE = delta + self.gamma*self.Lambda*GAE        #GAE is overwritten at each step, and we have GAE = GAE(t)
            
            #update the value

            self.value[self.index_s(state)] += self.lr_v*GAE

            # Update the parameters
            #                                       1 - pi(a|s)   if a is the performed action
            # we recall that grad log(pi(a|s)) = 
            #                                       - pi(a|s)     else

            for a in range(self.action_space):
                if a == action:  #for the performed action
                    self.parameters[self.index_sa((*state, a))] += self.lr_a*GAE*(1-self.policy(action, state))
                else:
                    self.parameters[self.index_sa((*state, a))] += self.lr_a*GAE*( -self.policy(action, state))
                

            #for numerical stability, we subtract from the parameters their max value
            
            max_parameter = np.max( [self.parameters[self.index_sa((*state, a))]   for a in range(self.action_space)])
            for a in range(self.action_space):
                self.parameters[self.index_sa((*state, a))] -= float(max_parameter)
        

if __name__ == "__main__":
    
    # General Settings 
    gamma = 0.99
    lr_v = 0.15
    lr_a = 0.01
    n_episodes = 2500
    env = SnakeEnv(render_mode="nonhuman", max_step=1000)
    L = GAE(env.action_space.n, gamma=gamma, lr_v=lr_v, lr_a=lr_a, Lambda=0.95)
    
   


