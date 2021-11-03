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

import gym 
import matplotlib.pyplot as plt

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
b = Q(4, 2)
b = b.double()
lr = 3e-5
optimizer = optim.Adam(q.parameters(), lr)
loss_func = nn.MSELoss()

#Set td learning paramaters
alpha = 1
lam = 0.5
epsilon = 0.5
epochs = 1_000
T = 500
duration = []
reward_list = []
loss_list = []
discaunting_vec = lam**np.arange(0,10+1)
for epoch in range(epochs):
    epsilon = epsilon * ( 1 - epoch/epochs)
    state = env.reset()
    r = 0
    epoch_loss = []
    memory = []
    for t in range(T):
        optimizer.zero_grad()
        output = q(state)
        if epoch < 4:
            if t == 0:
                print('epoch = ', epoch)
            print(output)
            
        action = np.argmax(output.detach().numpy())
        if np.random.rand() < epsilon:
            action = np.random.choice([0,1])
        old_state = state * 1
        state, reward, done, _ = env.step(action)
        if done:
            reward = 0
        r += reward
        action_mask = torch.zeros(2).double()
        action_mask[action] = 1
        other_mask = torch.ones(2).double()
        other_mask[action] = 0
        q_value = action_mask * alpha*(reward + \
               lam*torch.max(q(state))) + output * other_mask
        if done:
            q_value = torch.zeros(2).double()
        loss = loss_func(output, q_value)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        if done:
            #print(reward)
            break
        memory.append((state, reward))
        if len(memory)>10:
            1
            
    #for mem in range(len(memory)):
        
    
    loss_list.append(np.mean(epoch_loss))
    duration.append(t)
    reward_list.append(r)
    
    
   
        
        
        
plt.figure()
plt.plot(reward_list)
plt.figure()
plt.plot(duration)
plt.figure()
plt.plot(loss_list)

#%%
state = env.reset()
r = 0
epoch_loss = []
for t in range(T):
    output = q(state)
    #if epoch < 4:
    #    if t == 0:
    #        print('epoch = ', epoch)
    #    print(output)
        
    action = np.argmax(output.detach().numpy())
    if np.random.rand() < 0:
        action = np.random.choice([0,1])
    old_state = state * 1
    state, reward, done, _ = env.step(action)
    r += reward
    action_mask = torch.zeros(2).double()
    action_mask[action] = 1
    other_mask = torch.ones(2).double()
    other_mask[action] = 0
    q_value = action_mask * alpha*(reward + \
           lam*torch.max(q(state))) + output * other_mask
    loss = loss_func(output, q_value)

    if done:
        break


#%%
####################### Run Again with Memory ##################################
























