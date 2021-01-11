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
    def __init__(self,state_space, action_space, alpha = 0.5, lmbda = 0.5, epsilon = 0.1, off_policy = False):
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

        off_policy   : holds the off_policy - b - to be used if chosen, you can
                       set b as some matrix you want or 'random', default
                        is False..        
        Returns
        -------
        Q - the Q values orgenized as a zero or random values with shape:
            [state[0]*state[1]*...*state[n], num_actions]
            
            

        '''
        if off_policy:
            self.off_policy = True
            self.b = off_policy 
        size = 1
        for state_size in state_space:
            size *= state_size
        
        #Q will be a 2d matrix with all the possible actions for all possible 
        #states.
        self.Q = np.zeros(shape = [size,action_space])
        self.off_policy = off_policy
        if self.off_policy:
            if off_policy == 'random':
                self.b = np.random.rand(size,action_space)
            else:
                self.b = off_policy 
            self.off_policy = True
        #Build a list holding all possible combinations of states - 
        #ex - [location x = 0, y=19 and speed = 4]
        state_list = []
        for i in state_space:
            state_list.append(np.arange(0,i))
        self.state_comb_list = list(itertools.product(*state_list))
        self.alpha = alpha
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.num_updates = 0
        
    def act(self,state, epsilon_greedy = True, off_policy = False):
        '''
        

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.
        epsilon_greedy : TYPE, optional
            DESCRIPTION. The default is True.
        off_policy : an option to use the policy instade of the off_policy in case for evaluation 
                     or other needs, optional. The default is self.off_policy.

        Returns
        -------
        action : TYPE
            DESCRIPTION.

        '''
        #get index of state in Q
        state_index = self.state_comb_list.index(tuple(state))
        if off_policy:
            action_values = self.b[state_index,:]
            action_values = action_values/sum(action_values)
            if not sum(action_values) == 1:
                action_values = action_values/sum(action_values)
            action = np.random.choice(np.arange(0,len(action_values)),p = action_values)
        else:
            action_values = self.Q[state_index,:]
            #epsilon greedy choose an action
            if epsilon_greedy:
                if np.random.choice([0,1], p=[1-self.epsilon, self.epsilon]):
                    action = np.random.choice(np.arange(0,len(action_values)))
                    #Option to choose statisticaly and not gready
                else:
                    action = np.where(action_values == max(action_values))[0]
                    if len(action) > 0: #If there are two max actions, pick at random
                        action = np.random.choice(action)
            
            else:
                action = np.where(action_values == max(action_values))[0]
                if len(action) > 0: #If there are two max actions, pick at random
                    action = np.random.choice(action)
        action = int(action )
        return action
    
    def learn(self,old_state,old_action,new_state, reward):
        '''
        The learner is done at time t+1, i.e. a state is inserted into a one
        step memory and at the following state the Q function is updtatred 
        according to the new state and the Q value of the action the would 
        be selcted.
        The Q values is defined as such the the state action pair is that 
        of an action that is chosen in a state and NOT the action that led to 
        a state. 
        ---
        At t=0 I am at a state S[0] and choose an action A[0]
        This leads to S[1] at time t=1.
        ---
        We update the Q[S[0],A[0]] according to S[0], A[0], S[1] and an action
        A'[1] that will maximize Q in S[1]. In the on-policy case A'[1] could
        just be A[1] (epsilon greedy) but in the off policy case it is usually 
        not the case. 

        Parameters
        ----------
        state: state should hold [state[0],...,state[n]] the state of the system
               after taking the action, np.array(int)
        action: the action took by the learner - int 

        Returns
        -------
        None.

        '''
        action_to_update = old_action
        state_to_update_index = self.state_comb_list.index(tuple(old_state))
        new_state_index = self.state_comb_list.index(tuple(new_state))
        #Pick a new action according to new_state
        #For updating the Q function we need action from the Q not the off policy
        new_action = self.act(new_state, off_policy = False) 
        #LEARNING - update state action value
        self.Q[state_to_update_index,action_to_update] \
            += self.alpha*(reward + \
               self.lmbda*self.Q[new_state_index,new_action] -\
                self.Q[state_to_update_index,action_to_update])
        
        if self.off_policy:
            new_action = self.act(new_state, off_policy = self.off_policy)
        return new_action
        


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
                    T <- t + 1
                else:
                    Select and store an action A_t+1 from pi(·|St+1) 
            tau  <- t - n + 1 (tau is the time whose estimate is being updated)
            
            If tau >= 0:  
                G = sum_i(sigma**(i)*R_i) - A descounting weighted sum over the n rewards
            If tau + n < T,then G = G+ (sigma**n)*Q(S_(tau+n),A_(tau+n))
            #update the Q value with the new experiance G
            Q(S_tau,A_tau) =  Q(S_tau,A_tau) + alpha * [G - Q(S_tau,A_tau)] 
            If pi is being learned, then ensure that pi(·|S⌧)is epsilon-greedy wrt Q
    until tau = T - 1
    
    '''
    def __init__(self,state_space, action_space,n = 5, alpha = 0.5, lmbda = 0.5, epsilon = 0.1, off_policy = False):
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

        off_policy   : holds the off_policy - b - to be used if chosen, you can
                       set b as some matrix you want or 'random', default
                        is False..        
        Returns
        -------
        Q - the Q values orgenized as a zero or random values with shape:
            [state[0]*state[1]*...*state[n], num_actions]
            
            

        '''
        if off_policy:
            self.off_policy = True
            self.b = off_policy 
        size = 1
        for state_size in state_space:
            size *= state_size
        
        #Q will be a 2d matrix with all the possible actions for all possible 
        #states.
        self.Q = np.zeros(shape = [size,action_space])
        self.off_policy = off_policy
        if self.off_policy:
            if off_policy == 'random':
                self.b = np.random.rand(size,action_space)
            else:
                self.b = off_policy 
            self.off_policy = True
        #Build a list holding all possible combinations of states - 
        #ex - [location x = 0, y=19 and speed = 4]
        state_list = []
        for i in state_space:
            state_list.append(np.arange(0,i))
        self.state_comb_list = list(itertools.product(*state_list))
        self.alpha = alpha
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.num_updates = 0
        #Define a running memory to store the n last states and rewards for 
        #learning    
        self.td_state = []
        self.td_action = []
        self.td_reward = []
        self.tau = 0
        self.n = n+1
        #Create a discounting lambda vector to save some time on looping
        
        self.discaunting_vec = self.lmbda**np.arange(0,n+1)
            
        #set terminal to False, to facilitate learning
        self.terminal = False
    def act(self, state, epsilon_greedy = True, off_policy = False):
        '''
        

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.
        epsilon_greedy : TYPE, optional
            DESCRIPTION. The default is True.
        off_policy : an option to use the policy instade of the off_policy in case for evaluation 
                     or other needs, optional. The default is self.off_policy.

        Returns
        -------
        action : TYPE
            DESCRIPTION.

        '''
        #get index of state in Q
        state_index = self.state_comb_list.index(tuple(state))
        if off_policy:
            action_values = self.b[state_index,:]
            action_values = action_values/sum(action_values)
            if not sum(action_values) == 1:
                action_values = action_values/sum(action_values)
            action = np.random.choice(np.arange(0,len(action_values)),p = action_values)
        else:
            action_values = self.Q[state_index,:]
            #epsilon greedy choose an action
            if epsilon_greedy:
                if np.random.choice([0,1], p=[1-self.epsilon, self.epsilon]):
                    
                    action = np.random.choice(np.arange(0,len(action_values)))
                    
                    #Option to choose statisticaly and not gready
                else:
                    action = np.where(action_values == max(action_values))[0]
                    if len(action) > 0: #If there are two max actions, pick at random
                        action = np.random.choice(action)
            
            else:
                action = np.where(action_values == max(action_values))[0]
                if len(action) > 0: #If there are two max actions, pick at random
                    action = np.random.choice(action)
        action = int(action )
        #print(action, np.where(action_values == max(action_values))[0])
        return action
    
    def learn(self,old_state,old_action,new_state, reward):
        '''
        The learner is done at time t+n, i.e. a state is inserted into an 'n'
        step memory and following the n step it is updated like a mini monte-carlo
        algorithm.
        The Q values is defined as such the the state action pair is that 
        of an action that is chosen in a state and NOT the action that led to 
        a state. 
        ---
        At t=0 I am at a state S[0] and choose an action A[0]
        This leads to S[1] at time t=1.
        ---
        We update the Q[S[0],A[0]] according to [S[0], A[0]],..., [S[n],A[n]]
        and an action A'[n] that will maximize Q in S[n]. In the on-policy case 
        A'[n] could just be A[n] (epsilon greedy) but in the off policy case 
        it is usually not the case. 

        Parameters
        ----------
        state: state should hold [state[0],...,state[n]] the state of the system
               after taking the action, np.array(int)
        action: the action took by the learner - int 

        Returns
        -------
        None.

        '''
        if self.tau < self.n - 1:
            self.td_state.append(old_state)
            self.td_action.append(old_action)
            self.td_reward.append(reward)
            self.tau += 1
            new_action = self.act(new_state, off_policy = False) 
        else:
            self.td_state.append(old_state)
            self.td_action.append(old_action)
            self.td_reward.append(reward)
            
            G = np.sum(np.array(self.td_reward)*self.discaunting_vec)
            
            action_to_update = self.td_action[0]
            state_to_update_index = self.state_comb_list.index(tuple(self.td_state[0]))
            new_state_index = self.state_comb_list.index(tuple(new_state))
            #Pick a new action according to new_state
            #For updating the G function we need action from the Q not the off policy
            new_action = self.act(new_state, off_policy = False) 
            if not self.terminal:
                G += (self.lmbda**self.n)*self.Q[new_state_index,new_action]
                #print(G,(self.lmbda**self.n),self.Q[new_state_index,new_action])
            #LEARNING - update state action value
            self.Q[state_to_update_index,action_to_update] \
                += self.alpha*(G -\
                    self.Q[state_to_update_index,action_to_update])
            
            check = reward + \
                               self.lmbda*self.Q[new_state_index,new_action] -\
                                   self.Q[state_to_update_index,action_to_update]
                                   
            #print(G -\
            #        self.Q[state_to_update_index,action_to_update], check)
            #self.Q[state_to_update_index,action_to_update] \
            #    += self.alpha*(reward + \
            #                   self.lmbda*self.Q[new_state_index,new_action] -
            #                       self.Q[state_to_update_index,action_to_update])
            #remove the just updated spot
            self.state_temp = self.td_state[0]
            self.action_temp = self.td_action[0]
            self.state_reward = self.td_reward[0]
            self.td_state.remove(self.td_state[0])
            self.td_action.remove(self.td_action[0])
            self.td_reward.remove(self.td_reward[0])
        
        #if self.off_policy:
        #    new_action = self.act(new_state, off_policy = self.off_policy)
        return new_action
    
    def post_terminal_learn(self):
        if self.n == 1:
            return
        self.terminal = True
        for i in range(self.n-1):
            #print(self.state_temp, self.td_reward)
            _ = self.learn(old_state = self.state_temp, old_action = None, \
                       new_state = self.state_temp, reward = 0)
        self.terminal = False
        self.tau = 0
        self.td_state = []
        self.td_action = []
        self.td_reward = []
        self.tau = 0
            
        