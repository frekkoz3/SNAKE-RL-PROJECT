import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import time

class DeepDQL:
    """
    For initialization it requires ONLY:
        an INSTANCE of an enviroment(that follows gymnasium conventions)
        a neural network CLASS, with __init__(self, state_dim, n_actions).
    During the initialization state_dim and n_actions are obtained from the enviroment, and 2 nets(mainQ and targetQ) are creatied from the class NN
    Functions:
        learnQ: trains the network, and the Q associated with the best performance is stored in best_model.pth. 
                It returns a vector containing the sum of the rewards for each trajectory
                Argoments:
                    batch_size: number of states that the networks process simultaneously (a batch extracted from replay buffer)
                    n_traj_for_Qtarget_update: Q_target is set equal to Q every n trajectories
                    t_step_for_backpropagation: we train Q every t steps, not every step
                    buffer_capacity: max capacity of the buffer used to extract random batch
                    max_steps: after this number of STEPS(!= traj) the training is cut short (so we break out of infinite loops)
        
        evaluation_averaged: gives an average of the sum of rewards of 20 trajectory WITH THE GREEDY POLICY

        human_evaluations(n): shows visually the greedy policy at work in n trajectories(using render in human mode)

    """
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
                  gamma = 0.99):     
        
        self.env = env
        self.state_dim, self.n_actions = self.get_env_specs(env) 

        self.NN = NN
        self.device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")      #if there is a GPU available, we use it

        self.gamma = gamma
    
        #we initialized two neural network, for the main Q (which will be updated via TD learning) and Qtarget (which copies Q every 10 steps)
        self.Q = self.NN(self.state_dim, self.n_actions).to(self.device)
        self.Qtarget = self.NN(self.state_dim, self.n_actions).to(self.device)
        self.Qtarget.load_state_dict(self.Q.state_dict())


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

    def choose_action(self, state, epsilon):
        with torch.no_grad():   #when we do backward propagation, we do again forward, so now we don't need grad
            q = self.Q(state)       #q[a] = Q(state, a) 
            if random.random() < epsilon:
                return random.choice(range(self.n_actions))
            else:
                return int(torch.argmax(q).item())

    def learnQ(self, n_traj = 500, batch_size = 100, 
               n_traj_for_Qtarget_update = 20, t_step_for_backpropagation = 5, 
               buffer_capacity = 3000, max_steps = 500000,
               epsilon_start = 1, epsilon_end = 0.01,    #we update epsilon so it decays exponentially reaching espilon_end for n = 2/3 * n_traj
               learining_rate_optimizer = 0.001):
        
        start = time.time()

        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=learining_rate_optimizer)    #Adam is more efficient than SGD
        self.criterion = nn.MSELoss()    #loss = squared error

        buffer = self.ReplayBuffer(capacity=buffer_capacity, device=self.device)     #replay buffer, which creates tensor in device

        returns = np.zeros(n_traj)     #we store the performance for every trajectory
        best_return = 0                #we save the Q value when we got the best score

        t = 0
        
        for n in range(n_traj):
            #instead of udating rarely the target network, we smoothly change it
            tau = 0.005  # soft update rate
            with torch.no_grad():
                for target_param, param in zip(self.Qtarget.parameters(), self.Q.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            #if n%n_traj_for_Qtarget_update == 0:       #every one in a while we update the target network, making it equal to the main one
            #    self.Qtarget.load_state_dict(self.Q.state_dict())

            if t > max_steps:       #if the comulative number of steps is too high(e.g. we got stuck), we cut the training short
                break 

            state, _ = self.env.reset()    #initial state                                            
            state = torch.tensor(state, dtype=torch.float32, device=self.device).flatten()  #we need a tensor because otherwise ReplayBuffer.sample() doesn't work
                                                                                            #.flatten() because we need a 1 dim vector for the neural network
            
            
            #epsilon = max(epsilon_end, epsilon_start*(epsilon_decay)**(n))             #version where epsilon_decay is fixed
            epsilon = max(epsilon_end, epsilon_start*(epsilon_end/epsilon_start)**(3.0*n/(2.0*n_traj)))    #we update epsilon so it decays explnentially reaching espilon_end for n = 2/3 * n_traj

            done = False
            while (not done) and (t <= max_steps):

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
                if (len(buffer) > batch_size) and (t%t_step_for_backpropagation == 0):
                    
                    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                    
                    #Double Q-learining ==> delta = r +  gamma*Qtarget(S', argmax(Q(S', a)) )    -    Q(S, A)

                    next_actions = torch.argmax(self.Q(next_states), 1)          #we find optimal action for every new state, using the main Q 

                    with torch.no_grad():                   #we MUST NOT update the weights of the target network, so we must not keep track of the gradient
                        q_next = self.Qtarget(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)     #for every next state, we have to take the value of the optimal action
                        
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
                    action = int(torch.argmax(q).item())        #we use greedy policy
                
                    next_state, reward, term, trunc, _ = self.env.step(action)

                    next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).flatten()

                    done = term or trunc

                    ret += reward
                    state = next_state
                average_return += (ret-average_return)/(n+1)        #calculating on line average
                
            print(f"average return with greedy policy= {ret}")

    def human_evaluations(self, n):

        for _ in range(n):
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

