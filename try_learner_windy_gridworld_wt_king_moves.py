#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solving 
Windy Gridworld with King’s Moves 
From the book - 
Reinforment Lerning - an introduction 
page - 130
Using SARSA TD(0)

"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(__file__))
from Tabular_Learners import td_0,td_n

def windy_gridworld(location, action):
    '''
    Actions that takes the agent out of the grid keep the state unchanged.
    
    Parameters
    ----------
    location : the corrent location in the grid (x,y)
    action : the action to be played - actions are encoded as int (0-9) for 
             simplicity and converted to (action_x,action_y)

    Returns
    -------
    new_location : the next location as calculated
    reward : The reward from the action
    '''
    action_space = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1],[0,0]])
    action_corr = action_space[int(action)]
    wind_vec = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
    starting_corr = np.array([0,3])
    max_x_corr = 9
    max_y_corr = 6
    goal_corr = np.array([7,3])
    location = np.array(location)
    action = np.array(action)
    new_location = location + action_corr + np.array([0,wind_vec[location[0]]])
    #print(new_location,location, action_corr)
    if new_location[0] == 7:
        if new_location[1] == 3:
            7
            #print(new_location)
    if any(new_location < 0):
        new_location[np.where(new_location<0)] = 0
    if new_location[0] > max_x_corr:
        new_location[0] = max_x_corr
    if new_location[1] > max_y_corr:
        new_location[1] = max_y_corr
    if sum(new_location == goal_corr) == 2:
        reward = 0
        new_location = starting_corr
    else:
        reward = -1
    

    return new_location, reward
#%%
#Initialize Q(S,A) = 0 for all S,A
#Begin with the regular problem, only up, down, right, left
#The grid of size 9X6

grid_size_x = 9+1
grid_size_y = 6+1
num_actions = 9

starting_corr = np.array([0,3])
epochs = 15_000
epsilon = 0.1
alpha = 0.5
lmbda = 0.5
learner = td_n(state_space = [grid_size_x,grid_size_y], action_space = num_actions,\
                n=0,alpha = alpha, lmbda = lmbda, epsilon = epsilon, off_policy = False)
action = learner.act(starting_corr)
location = starting_corr * 1
count_steps = 0
#keep track of some of the runs
memory = []
#keep track of the rewards
episodes_memory = [0]
local_memory = []
action_memory = []
for epoch in range(epochs):
    
    local_reward_memory = [] 
    old_location = location * 1 
    
    local_memory.append(location*1)
    #extract the action values for the particular location
    max_action = learner.act(location, epsilon_greedy = False)
    action_memory.append([location,action, max_action])
    location, reward = windy_gridworld(location, action)
    #Keeps track on the number of finished runs in an epoch as there is no
    #end goal.
    if reward == -1:
        episodes_memory.append(episodes_memory[-1]+0)
    else:
        episodes_memory.append(episodes_memory[-1]+1)
        learner.post_terminal_learn()
        
    #extract the value of the new state in order to update the old Q(S,A)
    action = learner.learn(old_location, action, location, reward)
    count_steps += 1
    if epoch%500 == 0:
        print(epoch, episodes_memory[-1])

plt.figure()       
plt.plot(episodes_memory)
plt.title('Accumulated Times Reached Goal\n Windy Gridworld wt King Moves')
plt.xlabel('Time Steps')
plt.ylabel('Times Reached Goal')
#%%
#let's plot the last best policy
best_path = []
new_location = starting_corr   
finish = False
steps = 0
while not finish:
    location = new_location*1
    action = learner.act(location, off_policy = False)
    best_path.append([location,action])
    new_location, reward = windy_gridworld(location, action)
    if reward == 0:
        finish = True
    steps+=1
    if steps > 40:
        break

grid = np.zeros([grid_size_y,grid_size_x])
best_path = np.array(best_path)
count = 1
for i,j in best_path[:,0]:
    grid[grid_size_y-j-1,i] = count 
    count += 1
grid[3,7] = count + 1
plt.figure()
plt.imshow(grid)   
plt.title('The best policy path after {0} epochs\n Windy Gridworld wt King Moves'.format(epochs)) 
    
        
        
            
        
        
            
            
    
























