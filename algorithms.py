"""
    This work is developed for academic purpose by Bredariol Francesco, Savorgnan Enrico, Tic Ruben.
    This work is part of the final project for the 2024-2025 Reinforcement Learning course at the University of Trieste.
"""

import numpy as np
from collections import defaultdict, deque
import pickle
from IPython.display import clear_output

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
            nn.Conv2d(in_channels=n_layers, 32, kernel_size=4, stride=2),
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
        self.Qvalues = defaultdict(int)
        self.action_space = action_space
        self.iterations = 0 # This is used to count how many iteration until convergence

    def single_step_update(self, s, a, r, new_s, new_a, done):
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
            prob_actions[argmax_over_dict_given_subkey(self.Qvalues, (*s, ), default=[i for i in range (self.action_space)])[-1]] = 1
            
        # take one action from the array of actions with the probabilities as defined above.
        a = np.random.choice(self.action_space, p=prob_actions)
        return a 
        
    def get_action_greedy(self, s, possible_action = None):
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
        
        performance_traj = np.zeros(n_episodes)

        state, _ = env.reset()

        for i in range(n_episodes):

            eps = epsilon_schedule.decay() # it decays over episodes

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

                possible_actions = env.get_possible_actions(action)
                new_a = self.get_action_epsilon_greedy(new_s, eps, possible_actions=possible_actions)

                self.single_step_update(state, action, reward, new_s, new_a, done)
                
                action = new_a
                state = new_s

                keep = env.render()

            if i % 100 == 0:
                clear_output(wait=False)
                print(f"Episode {i}/{n_episodes} : epsilon {eps}")


        print("\n\nLearning finished\n\n")
        for i in range(len(performance_traj)):
            if i % 100 == 0:
                print(f"Episode {i} : Performance {performance_traj[i]}")
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
            action = self.get_action_greedy(state, possible_actions)
            state, reward, done, trunc, inf = env.step(action)
            state = bracketer.bracket(state)
            keep = env.render()

            total_reward += reward

        env.close()

        return total_reward


    def print_q_values(self, bracketer):
        s = ""
        for d in self.Qvalues:
            s = s + f"State ({bracketer.to_string(tuple(d[i] for i in range (len(d) - 1)))}), Action {action_name(d[-1])} : Value {self.Qvalues[d]}\n"
        print(s)

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

    def learning(self, env, epsilon_schedule, n_episodes, bracketer):

        performance_traj = np.zeros(n_episodes)

        state, _ = env.reset()

        for i in range(n_episodes):

            eps = epsilon_schedule.decay()

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

            clear_output()
            print(f'Episode {i}/{n_episodes}')

        print("\n\nLearning finished\n\n")
        for i in range(len(performance_traj)):
            if i % 100 == 0:
                print(f"Episode {i}/{n_episodes} : Performance {performance_traj[i]}")

    def name(self):
        return "Montecarlo"


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


    def get_action_epsilon_greedy(self, state, eps, possible_actions=None):
        if np.random.rand() < eps: # Explore
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


    def get_action_greedy(self, state, possible_actions=None):
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

        next_actions = self.dqn_online(next_states).argmax(1, keepdim=True)


        # Compute Q-values for the next states using the TARGET DQN
        with torch.no_grad():
            # Get the best action given the next states using the ONLINE DQN
            # Compute the Q-values for the next states using the TARGET DQN
            next_q_values = self.dqn_target(next_states).gather(1, next_actions).squeeze(1)

            # Compute the expected q_values
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

    
    def name(self):
        return "DDQL"


    def save(self, path):
        super().save(path)
        torch.save(self.dqn_target.state_dict(), path)


    def upload(self, path):
        self.dqn_target.load_state_dict(torch.load(path))
        self.dqn_target.eval()


class AtariDQN(DeepDoubleQLearning):
    """
    This class implements the DDQL using an Atari-like structure
    It uses convolutional layers to process the input state.
    """

    def __init__(self, action_space, gamma, lr_v, n_layers, height, width, batch_size=32, memory_size=10000, target_update_freq=1000, device='cpu'):
        super().__init__(action_space, gamma, lr_v, n_layers*height*width, batch_size, memory_size, target_update_freq, device)

        self.n_layers = n_layers
        self.recent_history = deque(maxlen=n_layers)

        self.dqn_online = ConvolutionalDQN(action_dim=action_space, n_layers=n_layers, height=height, width=width).to(self.device)
        self.dqn_target = ConvolutionalDQN(action_dim=action_space, n_layers=n_layers, height=height, width=width).to(self.device)
        self.dqn_target.load_state_dict(self.dqn_online.state_dict())
        self.dqn_target.eval()

        self.optimizer = torch.optim.Adam(self.dqn_online.parameters(), lr=self.lr_v)


    def single_step_update(self, s, a, r, new_s, new_a, done):
        """
        TODO: è DA SISTEMARE UN PO' QUESTO METODO.
        """

        # Update the recent history
        self.recent_history.append((s, a, r, new_s, done))

        # if the recent history is full, add to the memory
        if len(self.recent_history) == self.n_layers:
            # Convert the recent history to a state
            state = np.array([self.recent_history[i][0] for i in range(self.n_layers)])
            action = self.recent_history[-1][1]
            reward = self.recent_history[-1][2]
            new_state = np.array([self.recent_history[i][3] for i in range(self.n_layers)])
            done = self.recent_history[-1][4]

            # Store the transition in memory
            self.memory.append((state, action, reward, new_state, done))

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

        next_actions = self.dqn_online(next_states).argmax(1, keepdim=True)


        # Compute Q-values for the next states using the TARGET DQN
        with torch.no_grad():
            # Get the best action given the next states using the ONLINE DQN
            # Compute the Q-values for the next states using the TARGET DQN
            next_q_values = self.dqn_target(next_states).gather(1, next_actions).squeeze(1)

            # Compute the expected q_values
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




if __name__ == "__main__":
    eps = Epsilon(0.99, "linear", **{"coef" : 0.9, "minimum" : 0.05})
    print(eps.decay())
