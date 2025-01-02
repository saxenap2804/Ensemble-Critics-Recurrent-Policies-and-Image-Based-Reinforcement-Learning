# Modules.py
import torch
import torch.nn as nn
import torch.distributions as distributions

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        layers = []
        for in_size, out_size in zip([obs_dim] + hidden_sizes[:-1], hidden_sizes):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], act_dim))
        self.mean = nn.Sequential(*layers)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def forward(self, obs):
        mean = self.mean(obs)
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(self, obs):
        mean, std = self(obs)
        dist = distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action.detach().numpy(), log_prob.detach()

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        layers = []
        for in_size, out_size in zip([obs_dim] + hidden_sizes[:-1], hidden_sizes):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)

class EnsembleValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, num_critics=5):
        super().__init__()
        self.critics = nn.ModuleList([ValueNetwork(obs_dim, hidden_sizes) for _ in range(num_critics)])
        self.num_critics = num_critics

    def forward(self, obs):
        values = torch.stack([critic(obs) for critic in self.critics], dim=0)
        return values.mean(dim=0)
