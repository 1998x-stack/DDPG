
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Actor (Policy) Model for DDPG."""
    def __init__(self, state_dim, action_dim, max_action, min_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 16)
        self.layer_2 = nn.Linear(16, 16)
        self.layer_3 = nn.Linear(16, action_dim)
        self.max_action = max_action
        self.min_action = min_action
    
    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        if self.max_action and self.min_action:
            x = torch.tanh(self.layer_3(x)) * (self.max_action - self.min_action) + self.min_action
        else:
            x = self.layer_3(x)
        return x

class Critic(nn.Module):
    """Critic (Value) Model for DDPG."""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 16)
        self.layer_2 = nn.Linear(16, 16)
        self.layer_3 = nn.Linear(16, 1)
    
    def forward(self, state, action):
        x = F.relu(self.layer_1(torch.cat([state, action], 1)))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x