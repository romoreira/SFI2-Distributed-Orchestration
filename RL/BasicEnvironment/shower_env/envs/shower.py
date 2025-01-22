from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import random

class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60
        
    def step(self, action):
        self.state += action - 1
        self.shower_length -= 1
        
        print("Temperature Adjustment on Step: "+str(self.state))
        if 37 <= self.state <= 39:
            reward = 1
        else:
            reward = -1
        
        done = self.shower_length <= 0
        info = {}
        
        return np.array([self.state]), reward, done, info
    
    def reset(self, seed=None, options=None):
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60
        return np.array([self.state])
    
    def render(self, mode='human'):
        pass
