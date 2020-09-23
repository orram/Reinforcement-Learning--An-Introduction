#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:08:55 2020

@author: Or Ram 

A solution to the race track problem as defined in the book - 
Reinforment Lerning - an introduction 
page - 111


"""
#%%
import numpy as np
import matplotlib.pyplot as plt


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
Defining the Monte Carlo algorithm as:

off-policy MC control, for estimating pi

* Initialize, for all S, A(s): 
    Q(s, a) - (arbitrarily) 
    C(s, a) - 0
    pi(s) = argmaxa Q(s, a) (with ties broken consistently) 

* Loop forever (for each episode): 
    b - any soft policy 
    Generate an episode using b: S0,A0,R1,. .., S_T-1,A_T-1,R_T 
    W  = 1
    G  = 0 
    Loop for each step of episode, t = T-1,T-2,. .., 0: 
        G  = G+Rt+1
        C(St,At)  = C(St,At)+W
        Q(St,At)  = Q(St,At)+ (W/C(St,At))[G - Q(St,At)]    
        pi(St)  = argmaxa Q(St,a) (with ties broken consistently)
        If At ~= pi(St) then exit inner Loop (proceed to next episode) 
        W = W * 1/b(At|St)  

'''
#%%
track = Race_track([60,40])
#%%
#Inirtialize parameters:
all_locations = np.argwhere(track>0)*1
all_actions = []
for i in [-1,0,1]:
    for j in [-1,0,1]:
        all_actions.append([i,j])
all_speeds = []
for i in np.arange(-4,5):
    for j in np.arange(-4,5):
        all_speeds.append([i,j])
        
        

Q = {}
C = {}
b = {}
for location in all_locations:
    speed_dict = {}
    C_dict_temp = {}
    for speed in all_speeds:
        action_dict = {}
        c_dict = {}
        q = np.random.normal(loc = -10, scale = 4, size = 9)
        #p = p/p.sum()
        j = 0
        for action in all_actions:
            q_temp = q[j]
            j += 1
            action_dict[str(action)] = q_temp
            c_dict[str(action)] = 0
        speed_dict[str(np.array(speed))] = action_dict.copy()
        C_dict_temp[str(np.array(speed))] = c_dict.copy()
    Q[str(location)] = speed_dict.copy()
    C[str(location)] = C_dict_temp.copy()

b = {}
for location in all_locations:
    speed_dicp = {}
    for speed in all_speeds:
        action_dict = {}
        c_dict = {}
        q = abs(np.random.normal(loc = -5, scale = 4, size = 9))
        q = q/q.sum()
        j = 0
        for action in all_actions:
            q_temp = q[j]
            j += 1
            action_dict[str(action)] = q_temp
            c_dict[str(action)] = 0
        speed_dict[str(np.array(speed))] = action_dict
    b[str(location)] = speed_dict
    

track_visits = np.zeros(np.shape(track))

pi = {}
for location in all_locations:
    for speed in all_speeds:
        max_act = np.argmax(list(Q[str(location)][str(np.array(speed))].values()))
        pi[str(location*1) + "," + str(np.array(speed)*1)] = all_actions[max_act]
    
#%%
policy_reward = []
changed_act_at_loc = []
on_policy = False
epochs = 2_001
lam = 0.5
epsilon = 0.8 
mark_w = 1.25
all_tests = []

for epoch in range(epochs):
    if epoch%1000 == 0:
        print('Starting epoch -',epoch)
    start_location = np.argwhere(all_locations[:,0] == np.shape(track)[0]-1)[np.random.randint(0,len(np.argwhere(all_locations[:,0] == np.shape(track)[0]-1)))]
    start_location = all_locations[start_location[0]]
    new_coordinates, new_speed, reward, finish = Race(track = track,
                                                      coordinates = start_location ,
                                                      speed = [0,0], 
                                                      action = [0,0])
    
    race_memory = []
    run_time = 0
    didnt_reach_finish = False
    while not finish:
        run_time += 1
        if run_time % 50_000 == 0:
            print(run_time)
            didnt_reach_finish = True
            break
        if on_policy:
            if np.random.choice([0,1], p = [epsilon, 1-epsilon]):
                action = all_actions[np.random.choice(np.arange(0,9))]
            else:
                action = pi[str(new_coordinates*1) + "," + str(new_speed*1)]
                
        else:
            p = list(b[str(new_coordinates)][str(new_speed)].values())
            action_index = np.random.choice(np.arange(0,9),p = p)
            action = all_actions[action_index]
        old_coordinates = new_coordinates*1
        old_speed = new_speed * 1
        new_coordinates, new_speed, reward, finish = Race(track = track,
                                                      coordinates = new_coordinates,
                                                      speed = new_speed, 
                                                      action = action)
        track_visits[new_coordinates[0],new_coordinates[1]] += 1
        race_memory.append([old_coordinates,old_speed, action, reward])
        
    race_memory = np.array(race_memory, dtype=object)
    G = 0 
    W = 1
    for i in range(len(race_memory)):
        
        coor, speed, action, reward = race_memory[-1-i]
        if on_policy:
            if action == pi[str(coor*1) + "," + str(speed*1)]*1:
                on_policy_chance = epsilon*1
            else:
                on_policy_chance = (1-epsilon)*0.1
        G = lam * G + reward
        C[str(coor)][str(speed)][str(action)] += W
        Q[str(coor)][str(speed)][str(action)] += (W/C[str(coor)][str(speed)][str(action)]) * (G - Q[str(coor)][str(speed)][str(action)])
        old_pi_act = pi[str(coor*1) + "," + str(speed*1)]*1
        max_act = np.argmax(list(Q[str(coor)][str(speed)].values()))
        pi[str(coor*1) + "," + str(speed*1)] = all_actions[max_act]
        if not old_pi_act == pi[str(coor*1) + "," + str(speed*1)]:
            #print('changed action on location - ', coor)
            changed_act_at_loc.append(coor*1)
        if not pi[str(coor*1) + "," + str(speed*1)] == action:
            break
        if on_policy:
            W = W * (1/on_policy_chance)
            if W == mark_w:
                print(W)
                mark_w = mark_w * (1/on_policy_chance)
        else:
            W = W * (1/b[str(coor)][str(speed)][str(action)])
            
    if epoch%10== 0: 
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
            action = pi[str(new_coordinates) + "," + str(new_speed)]
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
ani.save('changed_action_at_location on policy = {0}.gif'.format(on_policy), writer=writergif)
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
ani.save('all_test_tracks on policy = {0}.gif'.format(on_policy), writer=writergif)
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