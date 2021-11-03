# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:19:34 2020

@author: or_ra
"""

from k_armed_bandits import k_armed
import numpy as np
import matplotlib.pyplot as plt 

epochs = 10000
k = 10
max_p = 1
policy = True
stochastic = False
stochastic_plus = False
normal_noise = True
number_repeats = 10

Q = np.ones(10)*0.1 #np.random.rand(10)*max_p
Q[2] = 0.9
N = np.zeros(10)
epsilon = 0.1
exploring = 0
bandits = k_armed(policy , k , max_p , stochastic , stochastic_plus, normal_noise)
p_right = []
#Basic incramental algorithm
print('Basic incramental algorithm')
for j in range(number_repeats):
    bandits.reset()
    for i in range(epochs):
        if np.random.choice(2,1, p=[1-epsilon,epsilon])[0]:
            exploring += 1/epochs
            action = np.random.randint(0,10)
        else:
            action = np.argmax(Q)
        reward = bandits.step(action)[0]
        N[action] += 1
        Q[action] = Q[action] + (1/N[action])*(reward - Q[action])
        
    p_right.append(bandits.p_right_choice)

#bandits.render()
actions = bandits.actions
best = bandits.best_choice
p_right_incramental = np.array(p_right)



#weighted average method:
print('weighted average method')
Q = np.ones(10)*0.1 #np.random.rand(10)*max_p
Q[2] = 0.9
alpha = 0.1
epsilon = 0.1
exploring = 0
p_right = []
bandits = k_armed(policy , k , max_p , stochastic , stochastic_plus, normal_noise)

for j in range(number_repeats):
    bandits.reset()
    for i in range(epochs):
        if np.random.choice(2,1, p=[1-epsilon,epsilon])[0]:
            exploring += 1/epochs
            action = np.random.randint(0,10)
        else:
            action = np.argmax(Q)
        reward = bandits.step(action)[0]
        Q[action] = Q[action] + alpha*(reward - Q[action])
    p_right.append(bandits.p_right_choice)

#bandits.render()
actions = bandits.actions
best = bandits.best_choice
print(exploring)
p_right_weighted_average = np.array(p_right)
plt.figure(figsize=(12, 8))
plt.plot(np.arange(0,np.size(p_right_weighted_average,1)),np.mean(p_right_weighted_average,0), 'o', label = 'p_right_weighted_average')
plt.plot(np.arange(0,np.size(p_right_weighted_average,1)),np.mean(p_right_incramental,0), '*', label = 'p_right_incramental')
plt.legend()
plt.title('Percent policy picked the best action')
plt.xlabel('epochs')
plt.ylabel('%')
