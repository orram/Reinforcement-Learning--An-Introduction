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


def Jack_car_basic(in_1, in_2, action):

    lam_1_req = 3
    lam_1_ret = 4
    lam_2_req = 3
    lam_2_ret = 2
    
    max_car = 20
    if in_1 > max_car:
        in_1 = max_car
    if in_2 > max_car:
        in_2 = max_car
        
    req_1 = np.random.poisson(lam_1_req) #number requasted in vanue 1
    req_2 = np.random.poisson(lam_2_req) #vanue 2
    
    ret_1 = np.random.poisson(lam_1_ret) #number of returned in vanue 1
    ret_2 = np.random.poisson(lam_2_ret) #vanue 2
    
    if (in_1 - req_1) < 0:        
        earned_1 = (req_1 + (in_1-req_1))*10
        in_1 = 0 
    else:
        earned_1 = req_1 * 10
        in_1 -= req_1
    
    if (in_2 - req_2) < 0:
        earned_2 = (req_2 + (in_2-req_2))*10
        in_2 = 0
    else:
        earned_2 = req_2 * 10
        in_2 -= req_2
        
    in_1 += ret_1
    if in_1 > max_car:
        in_1 = max_car
    in_2 += ret_2
    if in_2 > max_car:
        in_2 = max_car
    
    expense = 0
    if action > 0 :
        expense = (abs(action))*2
        
    return in_1,in_2, earned_1 - expense, earned_2



def Jack_car_next(in_1, in_2,action):

    lam_1_req = 3
    lam_1_ret = 4
    lam_2_req = 3
    lam_2_ret = 2
    
    max_car = 20
    if in_1 > max_car:
        in_1 = max_car
    if in_2 > max_car:
        in_2 = max_car
        
    req_1 = np.random.poisson(lam_1_req) #number requasted in vanue 1
    req_2 = np.random.poisson(lam_2_req) #vanue 2
    
    ret_1 = np.random.poisson(lam_1_ret) #number of returned in vanue 1
    ret_2 = np.random.poisson(lam_2_ret) #vanue 2
    
    if (in_1 - req_1) < 0:        
        earned_1 = (req_1 + (in_1-req_1))*10
        in_1 = 0 
    else:
        earned_1 = req_1 * 10
        in_1 -= req_1
    
    if (in_2 - req_2) < 0:
        earned_2 = (req_2 + (in_2-req_2))*10
        in_2 = 0
    else:
        earned_2 = req_2 * 10
        in_2 -= req_2
        
    in_1 += ret_1
    if in_1 > max_car:
        in_1 = max_car
    in_2 += ret_2
    if in_2 > max_car:
        in_2 = max_car
    
    #Adding the extra cost of parking above 10 cars
    if in_2 > 10:
        earned_2  -= 4
    if in_1 > 10:
        earned_1 -= 4
    
    expense = 0
    if action > 0 :
        expense = (abs(action)-1)*2
    return in_1,in_2, earned_1 - expense, earned_2


#Initialize the vlaue function - set all to zero and the policy to equal prob:
states_base = np.arange(0,21)
states = []
v_func = dict()
pi = pd.Series()
for state1 in states_base:
    for state2 in states_base:
        v_func[(state1,state2)] = 0
        pi[str((state1,state2))] = 0
        states.append((state1,state2))
#pi is gready and has only one option, the best option

lambda_orig = 0.9 # for updates
eval_delta = True
policy_improvement = True
delta = 0.005
epoch = 0
old_v = 0
num_eq = 0 # number of runs where the alg did not improve the value
no_improve_count = 0
while policy_improvement:
    #Policy evaluation
    eval_time = 0
    while eval_delta:
        tot_v = 0
        #states = combinations(np.arange(0,21),2)
        for state in states:
            eval_delta = False
            v = v_func[state]
            tot_v += v
            action = pi[str(state)]
            
            new_state = (state[0] + action, state[1] - action)
            if new_state[0]>20:
                new_state = (20,state[1] - action) 
            if new_state[1]>20:
                new_state = (state[0] + action, 20) 
            in_1, in_2, earned_1, earned_2 = Jack_car_basic(new_state[0],new_state[1], action)
            #if state == (20,0) or state == (20,10):
                #print(state, earned_1, earned_2)
            v_func[state] = earned_1 + earned_2 + lambda_orig*v_func[(new_state[0],new_state[1])]
            if abs(v-v_func[state]) > delta:
                eval_delta = True
        eval_time += 1
        if eval_time >= 3000:
            print('takes to loong to eval', v-v_func[state])
            break
    if tot_v == old_v:    
        no_improve_count +=1
        if no_improve_count > 4:
            print('No improvment in value of last 5 epochs,', tot_v,old_v)
            policy_improvement = False
    old_v = tot_v
    action_not_changed = 0 #keep count of the change in the system if no change break
    #states = combinations(np.arange(0,21),2)
    for state in states:
        if state[0]<5:
            if state[1] < 5:
                actions = np.arange(-state[0],state[1]+1)
            else:
                actions = np.arange(-state[0],6)
        elif state[1] < 5:
            actions = np.arange(-5,state[1]+1)
        else:
            actions = np.arange(-5,6)
        old_action = pi[str(state)]
        max_earned = 0
        new_action = 0
        for action in actions:
            new_state = (state[0] + action, state[1] - action)
            if new_state[0]>20:
                new_state = (20,state[1] - action) 
            if new_state[1]>20:
                new_state = (state[0] + action, 20) 
            #in_1, in_2, earned_1, earned_2 = Jack_car_basic(new_state[0],new_state[1])
            #q_val = earned_1 + earned_2- abs(action)*2 + lambda_orig*v_func[(in_1,in_2)]
            
            q_val = v_func[new_state]- abs(action)*2 +lambda_orig*v_func[state]
            if q_val > max_earned:
                new_action = action
                max_earned = q_val
        if new_action == old_action:
            action_not_changed += 1
        pi[str(state)] = new_action
    if action_not_changed == 210:
        policy_improvement = False
        print('Policy stable, no changes, after {0} epochs'.format(epoch))
    epoch += 1 
    if epoch > 20:
        print('more than 20 epochs to convarge')
        break

    
    
pi_f = []
for state in states:
    pi_f.append([state[0],state[1],pi[str(state)]])
pi_f = np.array(pi_f)
                
        
    






















    
    
        
    