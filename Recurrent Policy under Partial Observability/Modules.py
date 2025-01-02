import torch
import torch.nn as nn
import torch.distributions as distributions

class RecurrentPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_sizes[0], batch_first=True)
        self.fc_mean = nn.Linear(hidden_sizes[0], act_dim)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def forward(self, obs, hidden_state=None):
        obs = obs.unsqueeze(1)  # Add sequence dimension
        lstm_out, hidden_state = self.lstm(obs, hidden_state)
        mean = self.fc_mean(lstm_out.squeeze(1))
        std = torch.exp(self.log_std)
        return mean, std, hidden_state

    def get_action(self, obs, hidden_state=None):
        mean, std, hidden_state = self(obs, hidden_state)
        dist = distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action.detach().numpy(), log_prob.detach(), hidden_state

class RecurrentValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_sizes[0], batch_first=True)
        self.fc = nn.Linear(hidden_sizes[0], 1)  

    def forward(self, obs, hidden_state=None):
        obs = obs.unsqueeze(1)  
        lstm_out, hidden_state = self.lstm(obs, hidden_state)
        value = self.fc(lstm_out.squeeze(1))  
        return value, hidden_state  


class EnsembleValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, num_critics=5):
        super().__init__()
        self.critics = nn.ModuleList([ValueNetwork(obs_dim, hidden_sizes) for _ in range(num_critics)])
        self.num_critics = num_critics

    def forward(self, obs):
        values = torch.stack([critic(obs) for critic in self.critics], dim=0)
        return values.mean(dim=0)
