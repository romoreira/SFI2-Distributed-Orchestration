import gymnasium as gym
import shower_env.envs.shower
import torch
from agent import DQN_Agent
import random
import numpy as np
import matplotlib.pyplot as plt

seed = 42

env = gym.make("shower_env/ShowerEnv-v0")

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n


# Ajustes de hiperparâmetros
exp_replay_size = 1000
lr = 1e-3
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.995

agent = DQN_Agent(seed=42, layer_sizes=[input_dim, 64, output_dim], lr=lr, sync_freq=5, exp_replay_size=exp_replay_size)

num_episodes = 1000
batch_size = 64

rewards = []
losses = []  # Lista para armazenar os valores de loss
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
    
    while not done:
        action = agent.get_action(state, env.action_space.n, epsilon=epsilon)
        next_state, reward, done, _ = env.step(action)
        agent.collect_experience((state, action, reward, next_state))
        
        if len(agent.experience_replay) >= batch_size:
            loss = agent.train(batch_size)
            losses.append(loss)  # Armazenando o valor de loss
            print(f"Episode {episode}, Loss: {loss}")
        
        state = next_state
        total_reward += reward
    
    rewards.append(total_reward)
    print(f"Episode {episode}, Total Reward: {total_reward}")

# Plotando o gráfico de rewards
plt.figure()
plt.plot(rewards)
plt.xlabel('Episódios')
plt.ylabel('Total de Rewards')
plt.title('Desempenho do Agente ao Longo dos Episódios')
plt.savefig("reward.png")

# Plotando o gráfico de losses
plt.figure()
plt.plot(losses)
plt.xlabel('Passos de Treinamento')
plt.ylabel('Loss')
plt.title('Loss ao Longo do Tempo')
plt.savefig("loss.png")