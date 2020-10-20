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
    wind = wind_vec[location[0]] + np.random.choice([-1,0,1])
    new_location = location + action_corr + np.array([0,wind])
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
'''
Q = {}
for x in range(9):
    for y in range(6):
        for a in range(4):
            Q[str([x,y,a])] = 0
'''
grid_size_x = 10
grid_size_y = 7
num_actions = 8
Q = np.zeros([ num_actions, grid_size_x, grid_size_y])

starting_corr = np.array([0,3])
epochs = 20_000
epsilon = 0.1
alpha = 0.5
lmbda = 0.5
count_steps = 0
#keep track of some of the runs
memory = []
#keep track of the rewards
episodes_memory = [0]
local_memory = []
action_memory = []
for epoch in range(epochs):
    
    local_reward_memory = []

        
    if count_steps == 0:
        location = starting_corr
    else: 
        location = new_location*1
    local_memory.append(location*1)
    action_space = Q[:,location[0],location[1]]
    #epsilon greedy:
    if np.random.rand() > 1 - epsilon:
        action = np.random.randint(0,len(action_space))
    else:
        action = np.where(action_space == max(action_space))[0]
        #if more then one max action, pick at random
        if len(action) > 1:
            action = action[np.random.randint(0,len(action))]
    action = int(action )
    action_memory.append([location,action, np.where(action_space == max(action_space))[0]])
    new_location, reward = windy_gridworld(location, action)
    if reward == -1:
        episodes_memory.append(episodes_memory[-1]+0)
    else:
        episodes_memory.append(episodes_memory[-1]+1)
    new_action_space = Q[:,new_location[0],new_location[1]]
    new_action = np.where(new_action_space == max(new_action_space))[0]
    #if more then one max action, pick at random
    if len(new_action) > 1:
        new_action = new_action[np.random.randint(0,len(new_action))]
    #update state action value
    Q[action, location[0], location[1]] \
        += alpha*(reward + \
           lmbda*Q[new_action,new_location[0],new_location[1]] -\
            Q[action,location[0],location[1]])
    count_steps += 1
    if epoch%500 == 0:
        print(epoch, episodes_memory[-1])
plt.figure()       
plt.plot(episodes_memory)
plt.title('Accumulated Times Reached Goal\n Stochastic Windy Gridworld wt King Moves')
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
    action_space = Q[:,location[0],location[1]]
    action = np.where(action_space == max(action_space))[0]
    #if more then one max action, pick at random
    if len(action) > 1:
        action = action[np.random.randint(0,len(action))]
    action = int(action)
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
plt.title('The best policy path after {0} epochs\n Stochastic Windy Gridworld wt King Moves'.format(epochs)) 
    
        
        
            
        
        
            
            
    
























