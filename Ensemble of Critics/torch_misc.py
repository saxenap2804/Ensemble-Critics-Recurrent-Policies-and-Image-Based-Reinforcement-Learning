### torch_misc.py ###
import numpy as np
import torch
from collections import deque

# Replay buffer class
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

# 7. Collect trajectories
def collect_trajectories(env, policy, steps):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    trajectories = []

    for _ in range(steps):
        obs_tensor = torch.as_tensor(np.array(obs), dtype=torch.float32)
        action, log_prob = policy.get_action(obs_tensor)
        next_obs, reward, done, truncated, _ = env.step(action)

        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]

        trajectories.append((obs, action, reward, log_prob))

        if done or truncated:
            obs = env.reset()[0]
        else:
            obs = next_obs

    return trajectories

# 8. Compute GAE
def compute_gae(trajectories, value_function, gamma, lam):
    states, actions, rewards, log_probs = zip(*trajectories)
    states = torch.as_tensor(np.array(states), dtype=torch.float32)
    actions = torch.as_tensor(np.array(actions), dtype=torch.float32)
    rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32)
    log_probs = torch.as_tensor(np.array(log_probs), dtype=torch.float32)

    values = value_function(states).detach()
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    advantages = torch.zeros_like(rewards)
    adv = 0.0
    for t in reversed(range(len(deltas))):
        adv = deltas[t] + gamma * lam * adv
        advantages[t] = adv
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    returns = rewards + gamma * advantages
    return states, actions, advantages, returns, log_probs

# 9. PPO update
def ppo_update(policy, value_function, pi_optimizer, vf_optimizer, states, actions,
               advantages, returns, log_probs_old, clip_ratio):
    total_policy_loss = 0
    for _ in range(10):
        # Policy update
        pi_optimizer.zero_grad()
        mean, std = policy(states)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        ratio = torch.exp(log_probs - log_probs_old)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -(torch.min(ratio * advantages, clipped_ratio * advantages)).mean()
        policy_loss.backward()
        pi_optimizer.step()
        total_policy_loss += policy_loss.item()

        # Skip value function update if no optimizer is provided
        if vf_optimizer is not None:
            vf_optimizer.zero_grad()
            value_loss = ((value_function(states) - returns) ** 2).mean()
            value_loss.backward()
            vf_optimizer.step()

    avg_policy_loss = total_policy_loss / 10
    return avg_policy_loss, None

