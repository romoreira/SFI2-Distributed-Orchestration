from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
import tensorflow as tf

# ANN
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

# RL Policy
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import gymnasium
env = gymnasium.make('shower_env/ShowerEnv-v0')

class ShowerEnv(Env):
    def __init__(self):
        # Actions we can take: down, stay, up
        self.action_space = Discrete(3)
        # Temperature array
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        # Set start temp
        self.state = 38 + random.randint(-3, 3)
        # Set shower length
        self.shower_length = 60

    def step(self, action):
        # Apply action
        self.state += action - 1
        # Reduce shower length by 1 second
        self.shower_length -= 1 

        # Calculate reward
        if 37 <= self.state <= 39:
            reward = 1 
        else:
            reward = -1 

        # Check if shower is done
        done = self.shower_length <= 0

        # Placeholder for info
        info = {}

        # Return step information
        return self.state, reward, done, info

    def reset(self):
        # Reset shower temperature
        self.state = 38 + random.randint(-3, 3)
        # Reset shower time
        self.shower_length = 60 
        return self.state


def build_model(states, actions):
    model = Sequential()
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn


env = ShowerEnv()
states = env.observation_space.shape
actions = env.action_space.n

model = build_model((states[0],), actions)
model.summary()

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
