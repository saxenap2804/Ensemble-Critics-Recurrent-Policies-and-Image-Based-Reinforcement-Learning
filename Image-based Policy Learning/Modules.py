### Modules.py ###
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvolutionalPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        c, h, w = obs_dim
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_size = self._get_conv_out((c, h, w))
        self.fc_mean = nn.Linear(conv_out_size, act_dim)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs):
        conv_out = self.conv(obs).view(obs.size(0), -1)
        mean = self.fc_mean(conv_out)
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(self, obs):
        mean, std = self(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action.detach().numpy(), log_prob.detach()

class ConvolutionalValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        c, h, w = obs_dim
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_size = self._get_conv_out((c, h, w))
        self.fc = nn.Linear(conv_out_size, 1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs):
        conv_out = self.conv(obs).view(obs.size(0), -1)
        value = self.fc(conv_out)
        return value