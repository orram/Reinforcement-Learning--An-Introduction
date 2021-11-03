import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, gearJointDef, prismaticJointDef, contactListener)
import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle

'''
Project Discription:

'''




class NewEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }



    def __init__(self):
        EzPickle.__init__(self)
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        return
    def reset(self):
        

        return self.step(action = np.array([0,0]))[0]

    def step(self, action):
        
        state = []
        state = [np.array(state)]
        
       
        reward = 0
        #Need to add reward parameters.
        
        done = False
        #Need to add done indicators.
        
        return state, reward, done, {}

    def render(self, mode='human'):
        

        
        return 

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            
import gym
from gym import spaces

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, arg1, arg2,):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=255, shape=
                    (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

  def step(self, action):
    # Execute one time step within the environment
    ...
  def reset(self):
    # Reset the state of the environment to an initial state
    ...
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    ...
