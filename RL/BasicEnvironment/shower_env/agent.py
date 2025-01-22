import torch
from torch import nn
import copy
from collections import deque
import numpy as np
import random

class DQN_Agent:
    
    def __init__(self, seed, layer_sizes, lr, sync_freq, exp_replay_size):
        torch.manual_seed(seed)
        self.q_net = self.build_nn(layer_sizes)
        self.target_net = copy.deepcopy(self.q_net)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        
        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.gamma = torch.tensor(0.95).float()
        self.experience_replay = deque(maxlen=exp_replay_size)
        
    def build_nn(self, layer_sizes):
        layers = []
        for index in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act = nn.Tanh() if index < len(layer_sizes)-2 else nn.Identity()
            layers += [linear, act]
        return nn.Sequential(*layers)
    
    def get_action(self, state, action_space_len, epsilon):
        with torch.no_grad():
            Qp = self.q_net(torch.from_numpy(state).float())
        Q, A = torch.max(Qp, axis=0)
        A = A if random.random() > epsilon else random.randint(0, action_space_len-1)
        return A
    
    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_net(state)
        q, _ = torch.max(qp, axis=1)
        return q
    
    def collect_experience(self, experience):
        self.experience_replay.append(experience)
    
    def sample_from_experience(self, sample_size):
        if len(self.experience_replay) < sample_size:
            sample_size = len(self.experience_replay)
        sample = random.sample(self.experience_replay, sample_size)
        s = torch.tensor([exp[0] for exp in sample]).float()
        a = torch.tensor([exp[1] for exp in sample]).long()
        rn = torch.tensor([exp[2] for exp in sample]).float()
        sn = torch.tensor([exp[3] for exp in sample]).float()
        return s, a, rn, sn
    
    def train(self, batch_size):
        s, a, rn, sn = self.sample_from_experience(batch_size)
        if self.network_sync_counter == self.network_sync_freq:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.network_sync_counter = 0
        
        qp = self.q_net(s)
        pred_return = qp.gather(1, a.unsqueeze(-1)).squeeze(-1)
        
        q_next = self.get_q_next(sn)
        target_return = rn + self.gamma * q_next
        
        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.network_sync_counter += 1
        return loss.item()