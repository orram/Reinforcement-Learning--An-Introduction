
import numpy as np
import matplotlib.pyplot as plt


'''
Project Discription:
    K armed bandit enviorment aloows three modes:
        Random               - if policy == False there will be no policy and 
                                the chance of each arm will 
                                be random at each epoch.  
                 
        Determenistic policy - If policy == True and stochastic == False
                               Will create a constand probability reward 
        Stochastic policy    - Will change the probabilities once every N times
        Stochastic plus      - Will change the probabilities once every N times
                                N will change each time with change of epsilon

'''




class k_armed():

    def __init__(self, policy = True, k = 10, max_p = 0.4, stochastic = False, stochastic_plus = False, normal_noise = False):
        self.policy = policy
        self.k = k
        self.stochastic = stochastic
        self.normal_noise = normal_noise
        self.max_p = max_p
        if self.stochastic:
            if not type(self.stochastic) == int:
                self.stochastic = 50
        self.stochastic_plus = stochastic_plus
        if self.stochastic_plus:
            if not type(self.stochastic_plus) == float:
                self.stochastic_plus = 0.5
                
    def create_k_armed(self):

            self.prob_array = np.abs(np.random.normal(loc = 0, scale = 1.0, size = 10))
        
    def reset(self):
        self.create_k_armed()
        self.time = 0
        self.tot_time = 0
        self.reward = [0]
        self.r_mean = []
        self.actions = []
        self.best_choice = []
        self.right_choice = []
        self.p_right_choice = [0]
        

    def step(self, action):
        self.time += 1
        self.tot_time += 1
        if self.policy: 
            #if policy == True the program will choose a distinct set 
            # of probabilities for each arm. If false will be random all.
            reward = np.random.normal(loc = self.prob_array[action], scale = 1)
            if self.stochastic:
                self.best_choice.append(np.argmax(self.prob_array))
                if self.time%self.stochastic == 0:
                    if self.normal_noise:
                        self.prob_array += np.random.normal(loc = 0, scale = 1, size = 10)
                        
                    else:
                        self.create_k_armed()
                        self.time = 0
                        if self.stochastic_plus:
                            if np.random.choice(2, 1, p=[1-self.stochastic_plus,self.stochastic_plus])[0]:         
                                self.stochastic = np.random.randint(1000)
            else:
                self.best_choice.append(np.argmax(self.prob_array))
        else:
            reward = np.random.normal()
        
        if action == self.best_choice[-1]:
            self.right_choice.append(1)
        else:
            self.right_choice.append(0)

        self.p_right_choice.append((self.p_right_choice[-1]*(self.tot_time-1) + self.right_choice[-1])/self.tot_time)

        self.actions.append(action)
        self.reward.append(self.reward[-1]+reward)
        self.r_mean.append(self.reward[-1]/len(self.reward))
        if self.reward[-1] > 2000:
            done = True
        else:
            done = False

        
        return reward, done
    
    def render(self):
        plt.plot(self.r_mean, 'o')
        plt.title('Mean Reward')
        plt.figure(figsize=(12, 8))
        plt.plot(np.arange(0,len(self.actions)),self.best_choice,label = 'Best Choice')
        plt.plot(np.arange(0,len(self.actions)),self.actions,label = 'Action')
        plt.legend()
        plt.title('Actions vs Best Actions Given by the Policy')
        plt.xlabel('epochs')
        plt.ylabel('Arm')
        plt.figure(figsize=(12, 8))
        plt.plot(np.arange(0,len(self.p_right_choice)),self.p_right_choice, 'o')
        plt.title('Percent policy picked the best action')
        plt.xlabel('epochs')
        plt.ylabel('Arm')
        



