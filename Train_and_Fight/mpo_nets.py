import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class CategoricalActor(nn.Module):

    def __init__(self, in_units, out_units):
        super(CategoricalActor, self).__init__()
        self.f1 = nn.Linear(in_units, 128)
        self.f2 = nn.Linear(128, 128)
        self.logits = nn.Linear(128, out_units)
        self.activation = torch.nn.Tanh()

    def forward(self, states):
        x = self.f1(states)
        x = self.activation(x)
        x = self.f2(x)
        x = self.activation(x)
        x = self.logits(x)
        x = x.softmax(dim=-1)
        return x
    
    def action(self, state):
        with torch.no_grad():
            # start = time.time()
            probs = self.forward(state)
            # print(0, time.time() - start)
            action_distribution = Categorical(probs=probs)
            # print(1, time.time() - start)
            action = action_distribution.sample()
            # print(2, time.time() - start)
            prob = action_distribution.probs
            # print(3, time.time() - start)
        return action, prob
    
    def get_action_prob(self, state):
        probs = self.forward(state)
        action_distribution = Categorical(probs=probs)
        return action_distribution.probs
    
    def evaluate_action(self, state, action):
        probs = self.forward(state)
        action_distribution = Categorical(probs=probs)
        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy().mean()
        return action_distribution.probs, log_prob, entropy


class Critic(nn.Module):
    def __init__(self, in_units, out_unit):
        super(Critic, self).__init__()
        self.f1 = nn.Linear(in_units, 256)
        self.f2 = nn.Linear(256, 256)
        self.value = nn.Linear(256, out_unit)
        self.activation = torch.nn.Tanh()

    def forward(self, state):
        x = self.f1(state)
        x = self.activation(x)
        x = self.f2(x)
        x = self.activation(x)
        x = self.value(x)
        return x
