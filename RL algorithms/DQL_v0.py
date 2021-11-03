import os
os.environ['LANG']='en_US'
#import pyglet
#window = pyglet.window.Window()
import numpy as np
import torchvision
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim

import random
import copy
import gym 
import matplotlib.pyplot as plt

from render import save_frames_as_gif
### This is the first gym implimantation notebook
#Here I will get femiliar with gym with the simple CartPole task and build a simple RNN to solve it

#### First, let's call the CartPole enviorment and see the state and action spaces:

env = gym.make('CartPole-v0')
#PRint the state space:
print('size of state space = ',env.observation_space)

#Print action space:
print('size of action space = ',env.action_space)

### Let's see a sample random run



#%%
#First we reset the env to the initial state:
state = env.reset()

#Now run the network for 100 ts
for t in range(100):
    #env.render()
    print(state)
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    if done:
        print('Boo hoo the pole fell')
        break
env.close()
    
#%%
########################### Call a network learner ############################

class Q(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.action_space= action_space
        self.state_space = state_space
        
        self.fc1 = nn.Linear(state_space, 16)

        self.fc2 = nn.Linear(16, action_space)
        with torch.no_grad():
            self.fc2.weight = nn.Parameter(nn.init.uniform_(self.fc2.weight))
        
        self.relu = nn.ReLU()
    def forward(self, state):
        state = torch.tensor(state).double()

        q = self.relu(self.fc1(state.double()))
        q = torch.abs(self.fc2(q))
        
        return q
    
#%%
#Set backprop learning parameters

q = Q(4, 2)
q = q.double()
b = copy.deepcopy(q)
b = b.double()

epochs = 10_001
T = 500 #max duration of each run

#Set DQL parameters
minibatch = 32 #Minibatch size to sample in each time step
memory_size = 10_00_000 #Size of the replay memory, delate old memory to place new ones
gamma = 0.99 #Weight of future values in calculations
lr = 2.5e-4 
epsilon = 1 #Greedy parameter, decreases with the learning process
min_epsilon = 0.1 #the minimum value epsilon can take
optimizer = optim.Adam(q.parameters(), lr)
loss_func = nn.MSELoss()

replay_memory = []
reward_list = [0]
loss_list = [0]
c = 0
render_it = False
for epoch in range(epochs):
    if len(reward_list) > 5:
        if np.mean(reward_list[-5:]) > 194:
            print('Succsess!!', reward_list[-5:])
            break
        elif np.mean(reward_list[-3:]) > 194:
            render_it = True
    if epoch%50 == 0:
        print('epoch = {}, 5 last duration = {}, loss = {}'.format(epoch, 
                                                                 reward_list[-5:], 
                                                                 loss_list[-1]))
        
    epsilon = epsilon * ( 1 - epoch/epochs)
    if epsilon < 0.1:
        epsilon = min_epsilon
    state = env.reset()
    frames = []
    for t in range(T):
        if np.random.choice([1,0], p = [epsilon, 1-epsilon]):
            action = torch.tensor(np.random.choice([0,1]))
        else:
            Q_val = q(state)
            action = torch.argmax(Q_val)
        old_state = state * 1
        if epoch%1000 == 0 or render_it:
            frames.append(env.render(mode="rgb_array"))
        
        state, reward, done, _ = env.step(action.detach().numpy())

        replay_memory.append([old_state, action, reward, state, done])
        epoch_loss = []
        if len(replay_memory) > 50_000:
            samples = random.sample(replay_memory, minibatch)
            for s in samples:
                optimizer.zero_grad()
                if s[-1]: #if the action is the final action, i.e. if done
                    y = torch.tensor(0) #the reward at time j
                else:
                    y = s[2] + gamma * torch.max(b(s[3]))
                
                Q_choice = q(s[0])[s[1]] #Get the Q value of the chosen action 
                                         #from the trained policy
                                         #s[0] is the old state and s[1] is the 
                                         #action
                loss = loss_func(Q_choice.double(), y.double())
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                c+=1
                if c > 1_000:
                    b = copy.deepcopy(q)
                if len(replay_memory) > memory_size:
                    replay_memory = replay_memory[-memory_size:]
        if done: 
            break
    if epoch%1000 == 0 or render_it:
        save_frames_as_gif(frames, filename = 'CartPole-epoch_{}.gif'.format(epoch))
    reward_list.append(t)
    loss_list.append(np.mean(epoch_loss))
    
        
plt.figure()
plt.plot(reward_list)
plt.figure()
plt.plot(loss_list)

save_frames_as_gif()





















