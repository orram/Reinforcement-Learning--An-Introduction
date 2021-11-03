#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Following the steps and instructions from Openais Spinning up

https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
"""
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch

import gym 


def mlp(sizes, activation = nn.Tanh, output_activation = nn.Identity):

    fc_layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            fc_layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
        else:
            act = output_activation
            fc_layers += ([nn.Linear(sizes[i], sizes[i+1]), act()])
            
    return nn.Sequential(*fc_layers)
    
# def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
#     # Build a feedforward neural network.
#     layers = []
#     for j in range(len(sizes)-1):
#         act = activation if j < len(sizes)-2 else output_activation
#         layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
#     return nn.Sequential(*layers)

# make function to compute action distribution
def action_dist(obs):
    logits = policy_net(obs)
    return Categorical(logits = logits) 

# make action selection function (outputs int actions, *sampled* from policy)
def select_action(obs):
    return action_dist(obs).sample().item()

# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(obs, act, reward_weight):
    logp = action_dist(obs).log_prob(act)
    #remember that the logp is in length of the length of the episode 
    #and the reward_weight is the total reward of the episode
    return -(logp * reward_weight).mean()

def reward_to_go(episode_rewards):
    #A minte-carlo like summation of the rewards
    rtg = np.zeros_like(episode_rewards)
    for i in reversed(range(len(episode_rewards))):
        rtg[i] = episode_rewards[i] + (rtg[i+1] if i+1<len(episode_rewards) else 0)
    return rtg

#call enviorment - CartPole-v0, MountainCar-v0
env = gym.make('CartPole-v0')
hidden = [32,64]
#Initialize policy network
policy_net = mlp(sizes = [env.observation_space.shape[0]] + hidden + [env.action_space.n])
policy_optimizer = optim.Adam(policy_net.parameters(), lr = 3e-3)
#Initialize value network - using the mlp with the same hidden states, should 
#try other veriations. 
value_function = mlp(sizes = [env.observation_space.shape[0]] + hidden + [1])
value_optimizer = optim.Adam(value_function.parameters(), lr = 3e-3)
value_loss = nn.MSELoss()
#Define one epoch of training
def train_one_epoch(training_batch = 100, rtg_mode = True, vfb = True, value_function = None):
    #make list placeholders
    #List of all observasions, all states of the network - will be used to update 
    #the policy network (get's the networks action probability) and and value
    # function
    obs_list = []
    #List all actions performed - will be used to update 
    #the policy network (get's the networks action probability) 
    act_list = []
    #List of the weight of each time step, it is computed as the accumalated reward
    #calculated at the end of the episode, with or without discounting and vfb.
    reward_weights = []
    reward_list = []
    length_list = []
    
    #Store all rewards of a single run, when the run is finished we'll take the sum 
    #of episode_R as the total accumalated reward and it's length as the length of 
    #the run. 
    episod_R = []
    #Store the predicted value from the value function if vfb=True
    episode_v = []
    
    v_loss = None
    #First we reset the env to the initial state:
    obs = env.reset()
    full = False
    while not full:
        obs_list.append(obs.copy())
        
        action = select_action(torch.as_tensor(obs, dtype=torch.float32))
        obs, reward, done, _ = env.step(action)
        episod_R.append(reward)

        act_list.append(action)
        
        if vfb:
            v = value_function(torch.as_tensor(obs, dtype=torch.float32))
            episode_v.append(v)
        if done:
            R, L = sum(episod_R), len(episod_R)
            reward_list.append(R)
            length_list.append(L)
            
            if rtg_mode:
                reward_weights += list(reward_to_go(episod_R))
            else:
                reward_weights += [R] * L
            if vfb:
                reward_weights = list(np.array(reward_weights) - np.array(episode_v))
            # reset episode-specific variables
            obs, done, episod_R = env.reset(), False, []
            
            if len(obs_list) > training_batch:
                full = True
                break
            
            
            
    #Take a step  
    optimizer.zero_grad()
    loss = compute_loss(torch.as_tensor(obs_list, dtype=torch.float32), 
                        torch.as_tensor(act_list, dtype=torch.int32), 
                        torch.as_tensor(reward_weights, dtype=torch.float32))
    loss.backward()
    optimizer.step()
    

    if vfb:
        #Take a step on the value function
        value_optimizer.zero_grad()
        values = value_function(torch.as_tensor(obs_list, dtype=torch.float32))
        v_loss = value_loss(values.squeeze(),
                            torch.as_tensor(reward_weights, dtype=torch.float32))
        v_loss.backward()
        value_optimizer.step()
        
    return reward_list, length_list, loss, v_loss

#%% Try code
epochs = 1
plt.figure()
vfb = False
for mode in [True]:
    policy_net = mlp(sizes = [env.observation_space.shape[0]] + hidden + [env.action_space.n])
    optimizer = optim.Adam(policy_net.parameters(), lr = 3e-3)
    rewards = []
    lengths = []
    
    value_function = mlp(sizes = [env.observation_space.shape[0]] + hidden + [1])
    value_optimizer = optim.Adam(value_function.parameters(), lr = 3e-3)
    value_loss = nn.MSELoss()
    value_loss_list = []
    for epoch in range(epochs):
        reward_list, length_list, loss, v_loss = train_one_epoch(rtg_mode = mode, vfb = True,value_function = value_function)
        rewards.append(np.mean(reward_list))
        lengths.append(np.mean(length_list))
        value_loss_list.append(v_loss.item())
    plt.plot(rewards, label = 'rtf={}, vfg={}'.format(mode,vfb))
    
plt.legend()
plt.figure()
plt.plot(value_loss_list, label = 'rtf={}, vfg={}'.format(mode,vfb))
#%% Compare different options
epochs = 500
plt.figure()
for mode in [False, True]:
    for vfb in [False, True]:
        policy_net = mlp(sizes = [env.observation_space.shape[0]] + hidden + [env.action_space.n])
        optimizer = optim.Adam(policy_net.parameters(), lr = 1e-3)
        rewards = []
        running_reward = []
        lengths = []
        
        value_function = mlp(sizes = [env.observation_space.shape[0]] + hidden + [1])
        value_optimizer = optim.Adam(value_function.parameters(), lr = 3e-3)
        value_loss = nn.MSELoss()
        value_loss_list = []
        for epoch in range(epochs):
            reward_list, length_list, loss, v_loss = train_one_epoch(rtg_mode = mode, vfb = vfb,value_function = value_function)
            rewards.append(np.mean(reward_list))
            lengths.append(np.mean(length_list))
            if len(rewards) > 10:
                running_reward.append(np.mean(rewards[-10:]))
            if vfb:
                value_loss_list.append(v_loss.item())
            # if np.mean(rewards[-10:]) > 190:
            #     for g in optimizer.param_groups:
            #         g['lr'] *= 0.85
        print(optimizer)
        plt.plot(running_reward, label = 'rtf={}, vfb={}'.format(mode,vfb))
plt.legend()

#TODO
# Add value function baselines and the advantage formulation of policy gradients
# Make sure the learning is stable - run for 30 different trainings for each
#                                    option and plot avarage plus 95% confidance
   
# Look at other gym enviorments
# Look at other networks (pixle only conv nets)
#%%
##########################
    