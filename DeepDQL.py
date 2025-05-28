import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import time


class NN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(NN, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.out = nn.Linear(16, n_actions)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x) 

class DeepDQL:
    class ReplayBuffer:     
        #replay buffer: we store ALL history(up to capacity), then every time we update the weights, we take a random sample of batch_size element
        #this gives uncorrelation, and makes it possible to reuse experience for learining many times. 
        #Using batch_size >> 1 is a good idea for parallelization
        def __init__(self, capacity, device):
            self.buffer = deque(maxlen=capacity)    #deque is a FIFO
            self.device = device

        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))   #we append tuples, done is necessary to now if to add future rewards in the estimation of Q

        def sample(self, batch_size):
            transitions = random.sample(self.buffer, batch_size)    #random sample of data, more specifically a list of tuples (s, a, r, s', done)
            states, actions, rewards, next_states, dones = zip(*transitions)    #with * we get all tuples sepreted, with zip we group the i-th elements together
            return (
                torch.stack(states).to(self.device),
                torch.tensor(actions, device=self.device),     #device=self.device is more efficient
                torch.tensor(rewards, dtype=torch.float32,  device=self.device),
                torch.stack(next_states).to(self.device),
                torch.tensor(dones, dtype=torch.bool, device=self.device)
            )

        def __len__(self):
            return len(self.buffer) 

    def __init__(self,
                  env, 
                  NN, 
                  gamma = 0.99, 
                  epsilon_start = 1, epsilon_end = 0.01, epsilon_decay = 0.995,
                  learining_rate_optimizer = 0.001):     #e = max(e_end, e_start * e_decay^n)  where n is the trajectory's number
        
        self.env = env
        self.state_dim, self.n_actions = self.get_env_specs(env) 

        self.NN = NN
        self.device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")      #if there is a GPU available, we use it

        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        #we initialized two neural network, for the main Q (which will be updated via TD learning) and Qtarget (which copies Q every 10 steps)
        self.Q = self.NN(self.state_dim, self.n_actions).to(self.device)
        self.Qtarget = self.NN(self.state_dim, self.n_actions).to(self.device)
        self.Qtarget.load_state_dict(self.Q.state_dict())

        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=learining_rate_optimizer)    #Adam is more efficient than SGD
        self.criterion = nn.MSELoss()    #loss = squared error

    def get_env_specs(self, env):       
        # Action space
        if isinstance(env.action_space, gym.spaces.Discrete):
            n_actions = env.action_space.n
        #elif isinstance(env.action_space, gym.spaces.Box):
        #    action_space_shape = env.action_space.shape
        else:
            raise NotImplementedError(f"Unsupported action space type: {type(env.action_space)}")

        # Observation space
        if isinstance(env.observation_space, gym.spaces.Box):
            obs_shape = env.observation_space.shape
            state_dim = int(np.prod(obs_shape))  # Flattened size
        elif isinstance(env.observation_space, gym.spaces.Discrete):
            state_dim = 1
        else:
            raise NotImplementedError(f"Unsupported observation space type: {type(env.observation_space)}")

        return state_dim, n_actions

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    def set_criterion(self, criterion):
        self.criterion = criterion

    def choose_action(self, state, epsilon):
        with torch.no_grad():   #when we do backward propagation, we do again forward, so now we don't need grad
            q = self.Q(state)       #q[a] = Q(state, a) 
            if random.random() < epsilon:
                return random.choice(range(self.n_actions))
            else:
                return int(torch.argmax(q).item())

    def learnQ(self, n_traj = 500, batch_size = 100, n_traj_for_Qtarget_update = 20, buffer_capacity = 3000, t_step_for_backpropagation = 5):
        start = time.time()

        buffer = self.ReplayBuffer(capacity=buffer_capacity, device=self.device)     #replay buffer, which creates tensor in device




        returns = np.zeros(n_traj)     #we store the performance for every trajectory
        best_return = 0

        t = 0
        
        for n in range(n_traj):
            
            #tau = 0.005  # soft update rate
            #with torch.no_grad():
            #    for target_param, param in zip(self.Qtarget.parameters(), self.Q.parameters()):
            #        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            if n%n_traj_for_Qtarget_update == 0:       #every one in a while we update the target network, making it equal to the main one
                self.Qtarget.load_state_dict(self.Q.state_dict())

            state, _ = self.env.reset()    #initial state                                            
            state = torch.tensor(state, dtype=torch.float32, device=self.device).flatten()   #we need a tensor because otherwise ReplayBuffer.sample() doesn't work
                                                                                        #.flatten() because we need a 1 dim vector for the neural network

            epsilon = max(self.epsilon_end, self.epsilon_start*(self.epsilon_decay**n))     #we update epsilon

            done = False
            while not done:

                action = self.choose_action(state, epsilon)    #we take an epsilon greedy action

                next_state, reward, term, trunc, _ = self.env.step(action)
                t += 1
                #epsilon = epsilon*(t/(t+1))**0.3
                returns[n] += reward       #we accumulate rewards in the n-th comonent of the returns vector

                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).flatten()  #.flatten() because we need a 1 dim vector for the neural network

                done = term or trunc

                #saving data
                buffer.push(state, action, reward, next_state, done)

                #updating the weights using a whole batch of batch_size elements extracting randomly from memory by sample()
                if (len(buffer) > 2*batch_size) and (t%t_step_for_backpropagation == 0):
                    
                    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                    
                    #Double Q-learining ==> delta = r +  gamma*Qtarget(S', argmax(Q(S', a)) )    -    Q(S, A)

                    next_actions = torch.argmax(self.Q(next_states), 1)          #we find optimal action for every new state, using the main Q 

                    with torch.no_grad():                   #we MUST NOT update the weights of the target network, so we must not keep track of the gradient
                        q_next = self.Qtarget(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)     #for every next state, we have to take the value of the optimal action
                        q_next = torch.clamp(q_next, 0, 500)        #avoids degenerate value of Q
                        targets = rewards + self.gamma*q_next*(~dones)                 #if done = True we have   r + gamma*Q(s', a)*0 = r  (no future rewards)   
                    
                    predictions = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)     #for every state, we have to take value of the action we actually took
                                                                                           #unsqueeze(1) adds the second dimention
                    #back propagation
                    loss = self.criterion(predictions, targets)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


                state = next_state      #we move to the next step

                if returns[n] >= best_return:
                    best_return = returns[n]
                    torch.save(self.Q.state_dict(), "best_model.pth")
        end = time.time()

        print("Time span = ", end-start)
        print("epsilon = ",epsilon,"total steps = ", t)

        return returns
        
    def evaluation_averaged(self):

        with torch.no_grad():
            average_return = 0
            self.Q.load_state_dict(torch.load("best_model.pth"))
            for n in range(20):

                state, _ = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=self.device).flatten()
                
                done = False
                ret = 0
                while not done:
                    q = self.Q(state)       #q[a] = Q(state, a)
                    action = int(torch.argmax(q).item())
                
                    next_state, reward, term, trunc, _ = self.env.step(action)

                    next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).flatten()

                    done = term or trunc

                    ret += reward
                    state = next_state
                average_return += (ret-average_return)/(n+1)        #calculating on line average
                
            print(f"average return with greedy policy= {ret}")

    def single_evaluation(self):
        self.env.render_mode = "human"

        self.Q.load_state_dict(torch.load("best_model.pth"))
            
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).flatten()
        
        done = False
        
        while not done:
            q = self.Q(state)       #q[a] = Q(state, a)
            action = int(torch.argmax(q).item())
        
            next_state, reward, term, trunc, _ = self.env.step(action)

            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).flatten()

            done = term or trunc

            self.env.render()
            state = next_state


env = gym.make("CartPole-v1")

deepDQL = DeepDQL(env, NN)

returns = deepDQL.learnQ(n_traj = 700, n_traj_for_Qtarget_update=30, batch_size=150)
deepDQL.evaluation_averaged()
plt.plot(returns)
plt.show()