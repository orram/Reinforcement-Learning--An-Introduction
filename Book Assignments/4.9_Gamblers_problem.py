# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 11:16:53 2020

@author: or_ra

Jack's Car Rental problem
From  the book 
Reinforcement Learning - an introduction 
p. 81'

lambda_orig = 0.9
time steps == days
state == number of cars at trhe end of the day [(0)-(20)]
action == net number of cars moved between the locations [(-5)-(5)] 

Run the algorithm so we covere each state and each action in the state at least
once. 
So we have one example from each state. 
Every day is "new" in the sense that we don't need to simulate the problem
in a coherent time but just pool a state and progress it then another.
"""
import numpy as np
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt



#Initialize the vlaue function - set all to zero and the policy to equal prob:
states_base = np.arange(1,99)
states = []
v_func = np.zeros(100)
for i in range(0, 100):
    v_func[i] = np.random.random() * 1000
    
v_func[0] = 0
pi = np.zeros(100)
v_memory = np.zeros([100,30])
'''
for state in states_base:
        v_func[state] = 0
        temp_dict = dict()
        for i in range(state):
            actions = np.arange(0,min(state,100-state)+1)
            for a in actions:
                temp_dict[a] = 1/len(actions)
            pi[state] = temp_dict
'''
lambda_orig = 1 # for updates
eval_delta = True
policy_improvement = True
delta = 0.00005
epoch = 0
old_v = 0
num_eq = 0 # number of runs where the alg did not improve the value
no_improve_count = 0
p_h = 0.4

eval_time = 0
while eval_delta:
    #states = combinations(np.arange(0,21),2)
    for state in range(1, 100):
        v = v_func[state]
        actions = np.arange(0,min(state,100-state)+1)
        q = np.zeros(len(actions))
        for a in range(1, min(state, 100 - state) + 1):
            q[a] = 0
            if a + state < 100:
                q[a] += p_h*(0+lambda_orig*v_func[state + a])
                q[a] += (1-p_h)*(0+lambda_orig*v_func[state - a])
            elif a + state == 100:
                q[a] += p_h
                q[a] += (1-p_h)*(0+lambda_orig*v_func[state - a])

        max_act = np.argmax(q)
        pi[state] = max_act
        v_func[state] = q[max_act]

    eval_time += 1
    #print(eval_time)
    print(eval_time, np.mean(v_func))
    if eval_time<1:
        v_memory[:,eval_time] = v_func
    if eval_time >= 5000:
        print('takes to loong to eval', v-v_func[state])
        break

plt.plot(v_func)
plt.figure()
plt.bar(np.linspace(1, 100, num=100, endpoint=True),pi)




















    
    
        
    