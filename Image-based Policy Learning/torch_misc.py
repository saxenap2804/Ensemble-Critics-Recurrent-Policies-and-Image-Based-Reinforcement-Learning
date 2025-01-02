import numpy as np
import torch
from collections import deque
import cv2

# Replay buffer class
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

def collect_trajectories(env, policy, steps, render_function, stack_frames):
    obs = env.reset()
    if isinstance(obs, tuple) and len(obs) == 2:  
        obs = obs[0]
    print("Initial observation:", obs)

    if isinstance(obs, dict):
        obs = obs.get('observation', obs)

    obs = np.array(obs)

    theta = np.arctan2(obs[1], obs[0]) 
    initial_frame = render_function(theta)
    stacked_frames = deque([initial_frame] * stack_frames, maxlen=stack_frames)
    trajectories = []

    for _ in range(steps):
        state = np.stack(stacked_frames, axis=0)
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0) 
        action, log_prob = policy.get_action(state_tensor)

        step_result = env.step(action)
        
        if len(step_result) == 4:
            next_obs, reward, done, info = step_result
        elif len(step_result) == 5: 
            next_obs, reward, done, info, extra = step_result
        else:
            raise ValueError(f"Unexpected number of values returned from env.step(): {len(step_result)}")

        if isinstance(next_obs, dict):
            next_obs = next_obs.get('observation', next_obs)
        next_obs = np.array(next_obs)  

        if not isinstance(next_obs, np.ndarray):
            raise TypeError(f"Expected a NumPy array, got {type(next_obs)}")

        theta = np.arctan2(next_obs[1], next_obs[0]) 
        next_frame = render_function(theta)
        stacked_frames.append(next_frame)

        trajectories.append((state, action, reward, log_prob.item()))

        if done:
            obs = env.reset()
            if isinstance(obs, dict):
                obs = obs.get('observation', obs)
            obs = np.array(obs)

            if not isinstance(obs, np.ndarray):
                raise TypeError(f"Expected a NumPy array, got {type(obs)}")

            theta = np.arctan2(obs[1], obs[0])
            initial_frame = render_function(theta)
            stacked_frames = deque([initial_frame] * stack_frames, maxlen=stack_frames)

    return trajectories

def compute_gae(trajectories, value_function, gamma, lam):
    states, actions, rewards, log_probs = zip(*trajectories)
    states = torch.as_tensor(np.array(states), dtype=torch.float32)
    actions = torch.as_tensor(np.array(actions), dtype=torch.float32)
    rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32).view(-1)  
    log_probs = torch.as_tensor(log_probs, dtype=torch.float32)

    values = value_function(states).detach().view(-1)  
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]

    advantages = torch.zeros(rewards.shape, dtype=torch.float32)
    adv = 0.0
    for t in reversed(range(len(deltas))):
        adv = deltas[t] + gamma * lam * adv
        advantages[t] = adv

    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  
    returns = rewards + gamma * advantages
    return states, actions, advantages, returns, log_probs

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 64)) 
    return resized / 255.0  

def ppo_update(policy, value_function, pi_optimizer, vf_optimizer, states, actions,
               advantages, returns, log_probs_old, clip_ratio):
    total_policy_loss = 0
    total_value_loss = 0

    for _ in range(10):  
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

        if vf_optimizer is not None:
            vf_optimizer.zero_grad()
            value_loss = ((value_function(states) - returns) ** 2).mean()
            value_loss.backward()
            vf_optimizer.step()
            total_value_loss += value_loss.item()

    avg_policy_loss = total_policy_loss / 10
    avg_value_loss = total_value_loss / 10 if vf_optimizer is not None else None
    return avg_policy_loss, avg_value_loss

