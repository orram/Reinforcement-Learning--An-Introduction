#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:08:55 2020

@author: Or Ram 

A solution to the race track problem as defined in the book - 
Reinforment Lerning - an introduction 
page - 111
Only here we use TD(n) algorithm to compare with MC that we did before


"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

os.chdir(os.path.dirname(__file__))
from Tabular_Learners import td_n

def Race_track(size):
    '''
    

    Parameters
    ----------
    size : [N,M] - size of the race track

    Returns
    -------
    track : an N by M matrice describind the right angle race track.
            1 will symbolize an open squere and 0 boundery. 
            2 symbolizes the finish line
    
    The way we will build the track is as follows: 
       1) ruffle the size of the starting line and the y coordinates to start 
          the turn
       2) starting from the starting line we'll go line by line and add [-1,0,1]
          to the boundery, this way we are building the track from the ground.
          To make things easier changes are done only on the left side. 
       3) When we reach the turn line, the left side continues as before 
          and the right side move each step to the finish line on the x axis, 
          in adiition changes [-1,0,1] on the y axis are permited 
       

    '''


    track = np.ones(size)
    min_strt_pnt = int(size[1]*0.2)
    max_strt_pnt = int(size[1]*0.8)
    start_line_coordinates = [np.random.randint(min_strt_pnt,int(size[1]*0.5)),np.random.randint(int(size[1]*0.5),max_strt_pnt)]
    right_turn_coordinates = np.random.randint(int(size[0]*0.4),int(size[0]*0.8))
    left_bounderies = [start_line_coordinates[0]]
    right_bounderies = [start_line_coordinates[1]]
    left_choice = np.random.choice([-1,0,1])
    count = 0
    add_to_right = False
    for i in range(size[0]):
        if count < 5: # so the bounderies change more smoothly and less chaiotecly
            left_choice = left_choice
        else:
            count = 0
            left_choice = np.random.choice([-1,0,1])
        count += 1
        if left_bounderies[i]+left_choice == 0:
            left_bounderies.append(left_bounderies[i])
        elif right_bounderies[i] - left_bounderies[i]+left_choice < 8:
            #To make sure the there is no overlap and thet we always have some lane to race.
            #if the distance after update will be smaller then 5 we don't change
            left_bounderies.append(left_bounderies[i])
            count = 0
        else:
            left_bounderies.append(left_bounderies[i]+left_choice)
        
        
        track[-1-i,:left_bounderies[i]] = 0
        if i < right_turn_coordinates:
            right_bounderies.append(right_bounderies[i] + 0)
            track[-1-i,right_bounderies[i]:] = 0
        else:
            if not add_to_right:
                min_add_to_right = int((size[1] - right_bounderies[i])/(size[0] - right_turn_coordinates)+1)
            add_to_right = np.random.randint(min_add_to_right, min_add_to_right+4)
            right_bounderies.append(right_bounderies[i] + add_to_right)
            track[-1-i,right_bounderies[i] + 1:] = 0

            
    finish_line = list(track[:,-1]).index(0)
    track[:finish_line,-1] = 2
    plt.imshow(track)
     
    return track



def Race(track, coordinates, speed, action):
    '''
    NEED TO ADD - chance that actions will not work with probability 0.1    

    Parameters
    ----------
    track: the race track
    coordinates : Part of the state of the system, the x,y coordinates of the car
    speed : Part of the state, the x and y speed of the car
    action : What action is chosen

    Returns
    -------
    new_coordinates : 
    new_speed : Both velocity components are restricted to be nonnegative and 
                less than 5, and they cannot both be zero except at the 
                starting line.
    reward : The rewards are ?1 for each step until the car crosses the 
             finish line.
             
    If the car hits the track boundary, it is moved back to a random position 
    on the starting line, both velocity components are reduced to zero, and the
    episode continues.
    As a rule we update the speed and THEN update the coordinates.
    
    '''
    speed = np.array(speed)
    if np.random.choice([0,1],p = [0.9,0.1]):
        action = np.array([0,0])
    else:
        action = np.array(action)
    
    coordinates = np.array(coordinates)
    allowed_locations = np.argwhere(track>0)
    start_line_coordinates = allowed_locations[allowed_locations[:,0] == np.shape(track)[0] - 1]
    def start_location():
        new_location = start_line_coordinates[np.random.randint(0,len(start_line_coordinates))]
        return new_location
    new_speed = speed + action
    new_speed = np.array(new_speed)
    if any(i > 4 for i in new_speed):
        new_speed[new_speed > 4] = 4
    if any(i < -4 for i in new_speed):
        new_speed[new_speed < -4] = -4
    if all(i == 0 for i in new_speed):
        new_speed[np.random.randint(0,2)] = 1 
        
    #print(speed, action, new_speed)
    
    new_coordinates = coordinates*1 + new_speed*1
    #Let's look at the new coordinates and correct accordingly
    #First correct if the new corrdinates go out of bound
    if new_coordinates[0] > np.shape(track)[0] - 1: #the start line is track shape - 1
        new_coordinates = start_location()
        new_speed = [0,0] 
    elif new_coordinates[0] < 0: 
        #If you go above the first line (above the finish line) we need to correct 
        #according to if you hit the finish line or not.
        new_coordinates[0] = 0
        if new_coordinates[1] > np.shape(track)[1] - 1:
            new_coordinates[1] = np.shape(track)[1] - 1
            if track[new_coordinates[0],new_coordinates[1]] == 2:
                pass
            else:
                new_coordinates = start_location()
                new_speed = [0,0]
        else:
            if track[new_coordinates[0],new_coordinates[1]] == 2:
                pass
            else:
                new_coordinates = start_location()
                new_speed = [0,0]
    if new_coordinates[1] > np.shape(track)[1] - 1:
        #It's a right side turn so if you cross the right side you maybe in the finish line
        new_coordinates[1] = np.shape(track)[1] - 1 
        #print('exited from the right, coordinates are: {0} and value is {1}, speed and coordinates were'.format(new_coordinates,track[new_coordinates[0],new_coordinates[1]]),new_speed, coordinates)
    elif new_coordinates[1] < 0:
        new_coordinates = start_location()
        new_speed = np.array([0,0])
    
    
    if track[new_coordinates[0],new_coordinates[1]] == 0.0:
        while track[new_coordinates[0],new_coordinates[1]]== 0:
            #new_location = np.random.randint(np.min(np.argwhere(all_locations[:,0]==np.shape(track)[0] - 1)),np.max(np.argwhere(all_locations[:,0]==39)))
            #new_coordinates = [np.shape(track)[0]-1
            #                   ,allowed_locations[new_location,1]]
            new_coordinates = start_location()
            new_speed = np.array([0,0])
    '''
    in_track = False
    for i in range(len(allowed_locations)):
        if sum(new_coordinates == all_locations[i]) == 2:
            in_track = True
    if not in_track:
        print('got bug',track[new_coordinates[0],new_coordinates[1]])
        new_coordinates = start_location()
        new_speed = np.array([0,0])
    '''
    #calculate reward
    if track[new_coordinates[0],new_coordinates[1]] == 2:
        reward = 0 
        finish = True
    else:
        reward = -1
        finish = False 
   
    return np.array(new_coordinates), np.array(new_speed), np.array(reward), finish

#%%
     
            
     
'''
NOW LETS LEARN
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
#%%
#Initialize the race track map
y_size = 20
x_size = 15
track = Race_track([y_size,x_size])
#%%
#Inirtialize parameters:
all_locations = np.argwhere(track>0)*1
all_actions = []
for i in [-1,0,1]:
    for j in [-1,0,1]:
        all_actions.append([i,j])
action_space = len(all_actions)
all_speeds = []
for i in np.arange(-4,5):
    for j in np.arange(-4,5):
        all_speeds.append([i,j])

num_speeds = len(all_speeds)
state_space = [y_size,x_size, num_speeds]        
size = 1
for state_size in state_space:
    size *= state_size

Q = np.random.normal(loc = 1000, scale = 4, size = [size,action_space])
b = np.random.normal(loc = 1000, scale = 4, size = [size,action_space])
state_list = []
for i in state_space:
    state_list.append(np.arange(0,i))
state_comb_list = list(itertools.product(*state_list))

pi = {}
for location in all_locations:
    for speed in range(len(all_speeds)):
        state = [location[0], location[1], speed]
        state_index = state_comb_list.index(tuple(state))
        max_act = np.argmax(list(Q[state_index,:]))
        pi[str(location*1) + "," + str(np.array(speed)*1)] = all_actions[max_act]
        
track_visits = np.zeros(np.shape(track))
    
#%%
num_speeds = len(all_speeds)
num_actions = len(all_actions)

policy_reward = []
changed_act_at_loc = []
on_policy = True
epochs = 11
lam = 0.9
alpha = 0.1
epsilon = 0.5
mark_w = 1.25
all_tests = []
learner = td_n(state_space = [y_size,x_size, num_speeds], action_space = num_actions,\
                n=0,alpha = alpha, lmbda = lam, epsilon = epsilon, off_policy = False,\
                    MC = False, normal_dist = True)
learner.Q = Q
compare_class = []
compare_h = []
for epoch in range(epochs):
    if epoch%1000 == 0:
        print('Starting epoch -',epoch)
    start_location = np.argwhere(all_locations[:,0] == np.shape(track)[0]-1)[np.random.randint(0,len(np.argwhere(all_locations[:,0] == np.shape(track)[0]-1)))]
    start_location = all_locations[start_location[0]]
    action = [0,0]
    learner_action = all_actions.index(action)
    new_coordinates, new_speed, reward, finish = Race(track = track,
                                                      coordinates = start_location ,
                                                      speed = [0,0], 
                                                      action = action)
    
    race_memory = []
    run_time = 0
    didnt_reach_finish = False
    while not finish:
        run_time += 1
        if run_time % 10_000 == 0:
            print(run_time)
            didnt_reach_finish = True
            break

        old_coordinates = new_coordinates*1
        old_speed = new_speed * 1
        learner_old_speed = all_speeds.index(list(old_speed))
        old_state = [old_coordinates[0],old_coordinates[1],learner_old_speed]
        new_coordinates, new_speed, reward, finish = Race(track = track,
                                                      coordinates = new_coordinates,
                                                      speed = new_speed, 
                                                      action = action)
        learner_new_speed = all_speeds.index(list(new_speed))
        track_visits[new_coordinates[0],new_coordinates[1]] += 1
        new_state = [new_coordinates[0],new_coordinates[1],learner_new_speed]
        new_state_index = state_comb_list.index(tuple(new_state))
        old_learner_action = learner_action * 1
        
        race_memory.append([old_coordinates,old_speed, action, reward])
        old_action = action*1
        
        if on_policy:
            if np.random.choice([0,1], p = [epsilon, 1-epsilon]):
                action = all_actions[np.random.choice(np.arange(0,9))]
            else:
                #print('from pi')
                action = pi[str(new_coordinates*1) + "," + str(learner_new_speed*1)]
                
        else:
            p = list(b[new_state_index])
            action_index = np.random.choice(np.arange(0,9),p = p)
            action = all_actions[action_index]
        max_act_h = pi[str(new_coordinates*1) + "," + str(learner_new_speed*1)]
        max_act_h = all_actions.index(max_act_h)
        learner_action = all_actions.index(action)
        ###########################################
        #LEARNING - update state action value TD(0)
        ###########################################
        update_state_index = state_comb_list.index(tuple(old_state))
        new_state_index = state_comb_list.index(tuple(new_state))
        ########## Comperting #####################
        G_here = reward + \
               lam*Q[new_state_index,learner_action]
        #print('here Q', Q[new_state_index,learner_action], new_state_index, learner_action)
        compare_h.append((G_here, learner_action, max_act_h, Q[new_state_index,learner_action], reward, lam))
        learner_action_class, G_class, max_action, Q_class, r_class, factor_class = learner.learn(old_state, old_learner_action, new_state, reward, a = learner_action)
        #print('here reward', reward)
        compare_class.append((G_class, learner_action_class, max_action[0],Q_class, r_class, factor_class))
        ###########################################
        Q[update_state_index,old_learner_action] \
            += alpha*(reward + \
               lam*Q[new_state_index,learner_action]  -\
                Q[update_state_index,old_learner_action] )
        old_pi_act = pi[str(old_coordinates*1) + "," + str(learner_old_speed*1)]*1
        max_act = np.argmax(list(Q[update_state_index,:]))
        pi[str(old_coordinates*1) + "," + str(learner_old_speed*1)] = all_actions[max_act]
        if not old_pi_act == pi[str(old_coordinates*1) + "," + str(learner_old_speed*1)]:
            #print('changed action on location - ', coor)
            changed_act_at_loc.append(old_coordinates*1)
       
    race_memory = np.array(race_memory, dtype=object)
            
    if epoch%25== 0: 
        if epoch == 0:
            pass
        start_location = np.argwhere(all_locations[:,0] == np.shape(track)[0]-1)[np.random.randint(0,len(np.argwhere(all_locations[:,0] == np.shape(track)[0]-1)))]
        start_location = all_locations[start_location[0]]
        print(epoch, start_location,track[start_location[0],start_location[1]])
        #start_location[0] = np.shape(track)[0]-1
        track[start_location[0],start_location[1]] = 4
        plt.imshow(track)
        track[start_location[0],start_location[1]] = 1
        new_coordinates, new_speed, reward, finish = Race(track = track,
                                                      coordinates = start_location ,
                                                      speed = [0,0], 
                                                      action = [0,0])
        test_memory = []
        print('evaluating the learnt policy')
        run_time = 0
        while not finish:
            run_time += 1
            if run_time == 10_000:
                print('Evaluation exceeded time elocated', run_time)
                break
            learner_new_speed = all_speeds.index(list(new_speed))
            action = pi[str(new_coordinates) + "," + str(learner_new_speed)]
            act, m_act = learner.act(state, epsilon_greedy = False)
            act = all_actions[learner_action]
            old_coordinates = new_coordinates*1
            old_speed = new_speed * 1
            new_coordinates, new_speed, reward, finish = Race(track = track,
                                                          coordinates = new_coordinates,
                                                          speed = new_speed, 
                                                          action = action)
            
            test_memory.append([old_coordinates,old_speed, action, reward])
        
        test_memory = np.array(test_memory, dtype=object)
        sum_reward = sum(test_memory[:,-1])
        policy_reward.append([epoch,sum_reward])
        print('sum reward of policy after epoch {0} is {1}'.format(epoch,sum_reward))
        all_tests.append([test_memory,epoch,sum_reward])
policy_reward = np.array(policy_reward)
plt.figure()
track_visits[track_visits> np.mean(track_visits) + np.std(track_visits)] = np.mean(track_visits) + np.std(track_visits)
plt.imshow(track_visits)
plt.title('Represnts the number of VISITS\n the simulation took in each location\n on policy = {0}'.format(on_policy))
plt.colorbar()
track_changed = track*1
for i in range(len(changed_act_at_loc)):
    track_changed[changed_act_at_loc[i][0],changed_act_at_loc[i][1]] += 1
plt.figure()
plt.imshow(track_changed)
plt.title('Represnts the number of CHANGES\n of the policy in each location\n on policy = {0}'.format(on_policy))
plt.colorbar()
plt.figure()
plt.plot(policy_reward[int(len(policy_reward)*0.1):,0],np.abs(policy_reward[int(len(policy_reward)*0.1):,1]))
plt.xlabel('epochs')
plt.ylabel('reward')
plt.title('The rewards the policy got on the test run\n on policy = {0}'.format(on_policy))
avrg_reward = []
for i in range(len(policy_reward)):
    avrg_reward.append(np.mean(np.abs(policy_reward[:,1])[:i]))
plt.figure()
plt.plot(policy_reward[int(len(policy_reward)*0.1):,0],avrg_reward[int(len(policy_reward)*0.1):])
plt.xlabel('epochs')
plt.ylabel('avrg reward')
plt.title('The avrg rewards the policy got on the test runs\n on policy = {0}'.format(on_policy))
#%%
plt.subplots_adjust(left=5, bottom=5, right=6.5, top=6.5, wspace=None, hspace=None)
f = plt.figure(figsize=(20,20))
for j in range(1,10):
    i = np.linspace(1,len(all_tests), num=9)
    i = i[j-1]
    i = int(i)
    ax = f.add_subplot(3,3,j)
    ax.set_title('{0} epochs, reward = {1}'.format(all_tests[i-1][1],all_tests[i-1][2]))
    test_track = track*1
    temp_run = all_tests[i-1][0][:,0]
    for j in range(len(temp_run)):
        test_track[temp_run[j][0],temp_run[j][1]] += 4
    plt.imshow(test_track)



    
#%%
import matplotlib.animation as animation
# Set up formatting for the movie files
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
writergif = animation.PillowWriter(fps=50)

M = track*1
M[-1,-1] = np.max(track_changed)

def update(i):
    M[changed_act_at_loc[i][0],changed_act_at_loc[i][1]] += 1

    matrice.set_array(M)
    #if np.sum(M == track_changed):
    #   M = track*1
    

fig, ax = plt.subplots()
matrice = ax.matshow(M)
plt.colorbar(matrice)

ani = animation.FuncAnimation(fig, update, frames=len(changed_act_at_loc), interval=0.5, repeat =False)
ani.save('TD(0) changed_action_at_location on policy = {0}.gif'.format(on_policy), writer=writergif)
plt.show()
#%%
writergif = animation.PillowWriter(fps=5)
anim_memory = []
for i in range(len(all_tests)):
    temp_track = track*1
    temp_run = all_tests[i][0][:,0]
    for j in range(len(temp_run)):
        temp_track[temp_run[j][0],temp_run[j][1]] += 4
    anim_memory.append(temp_track)
for i in range(15):
    anim_memory.append(track*1)
M = track*1

def update(i):
    M = anim_memory[i]
    matrice.set_array(M)
    
fig, ax = plt.subplots()
plt.title('Examples of tracks chosen\n by the policy in differnet epochs')
matrice = ax.matshow(M)
plt.colorbar(matrice)

ani = animation.FuncAnimation(fig, update, frames=len(anim_memory), interval=200, repeat =False)
ani.save('TD(0) all_test_tracks on policy = {0}.gif'.format(on_policy), writer=writergif)
plt.show()
#%%




'''
allowed_locations = np.where(track>0)
new_location = np.random.randint(0,len(allowed_locations[0]))
coordinates = [allowed_locations[0][new_location],allowed_locations[1][new_location]]
speed = [0,0]
if track[coordinates[0],coordinates[1]] == 0:
        while track[coordinates[0],coordinates[1]] == 0:
            coordinates = [allowed_locations[0][new_location],allowed_locations[1][new_location]]
coor_vec = []
for i in range(10):
    action = np.random.choice([1,0,-1], size = 2)
    print(coordinates)
    coordinates, speed, reward = Race(track = track, coordinates = coordinates, speed = speed, action = action)
    coor_vec.append(coordinates)
    print('action = {0}, new speed is {1} and new coordinates {2}'.format(action, speed, coordinates))
    print('track value = {0}'.format(track[coordinates[0],coordinates[1]]))
    
add = 1
for i in range(len(coor_vec)):
    track[coor_vec[i][0],coor_vec[i][1]] = 3 + add 
    add+=1
plt.imshow(track)

'''