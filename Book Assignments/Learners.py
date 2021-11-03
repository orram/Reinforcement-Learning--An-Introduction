#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:59:34 2021

@author: orram
"""


import itertools
import numpy as np


class td_0():
    '''
    Implimintation of td(0) learner
    
    The learner class will hold the Q value and propogate the learning process
    The class will hold the following:
        __init__ - initialize the learner parameters and working functions
        learn    - each time step, another experiance - [S,A,R] - will be 
                   inserted to the learner and update the Q value.
        act      - will output a
    
    '''
    def __init__(state_space, action_space, alpha = 0.5, lmbda = 0.5, epsilon = 0.1):
        '''
        

        Parameters
        ----------
        state_space  : the possible states of the system, of a tabular problem
                       it should hold the size of the table and other veriables 
                       such as speed, direction, etc...
        action_space : the allowed actions  
        alpha        : the learning rate. The default is 0.5.
        lmbda : TYPE, optional
            DESCRIPTION. The default is 0.5.
        epsilon      : Epsilon gready policy. The default is 0.1.

        Returns
        -------
        Q - the Q values orgenized as a zero or random values with shape:
            [num_action,state[0]*state[1]*...*state[n]]
            
            

        '''
        size = 1
        for state_size in state_space:
            size *= state_size
        
        #Q will be a 2d matrix with all the possible actions for all possible 
        #states.
        self.Q = np.zeros(shape = [size,action_space])
        #Build a list holding all possible combinations of states - 
        #ex - [location x = 0, y=19 and speed = 4]
        state_lists = []
        for i in state_space:
            state_list.append(np.arange(0,i))
        self.state_comb_list = list(itertools.product(*state_list))
        self.alpha = alpha
        self.lmbda = lmbda
        self.epsilon = epsilon
        
    def act(state):
        #get index of state in Q
        state_index = self.state_comb_list.index(state)
        action_values = self.Q[state_index,:]
        #epsilon greedy choose an action
        
        


class td_n():
    '''
    Implimantation of the TD(n) on policy learner
    Defining the TD(n) algorithm on-policy as:
    
    Initialize Q(s, a) arbitrarily, for all s in S,a in A 
    Algorithm parameters: 
        step size alpha in (0, 1] 
        small epsilon > 0, 
        a positive integer n 
    All store and access operations (for St, At, and Rt) can take their index mod n +1
    set pi(s,a) from Q
    
    Loop for each episode: 
        Initialize and store S0 ~= terminal 
        Select and store an action A0 from pi(·|S0) 
        set T -> inf
        Loop for t =0, 1, 2,. .. : 
            If t< T,then:
                Take action A_t
                | | |
                Observe and store the next reward as Rt+1 and the next state as St+1 
                If St+1 is terminal, then:
                    T <- t +1
                else:
                    Select and store an action A_t+1 from pi(·|St+1) 
            tau  <- t - n +1 (tau is the time whose estimate is being updated)
            
            If tau >= 0:  
                G = sum_i(sigma**(i-tau-1)*R_i) - A descounting weighted sum over the n rewards
            If tau + n < T,then G = G+ (sigma**n)*Q(S_(tau+n),A_(tau+n))
            #update the Q value with the new experiance G
            Q(S_tau,A_tau) =  Q(S_tau,A_tau) + alpha * [G - Q(S_tau,A_tau)] 
            If pi is being learned, then ensure that pi(·|S⌧)is epsilon-greedy wrt Q
    until tau = T - 1
    
    '''
    def __init__(n = 5 , alpha, sigma, )