### main.py ###
import gym
import torch
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
from Modules import PolicyNetwork, ValueNetwork
from torch_misc import collect_trajectories, compute_gae, ppo_update, ReplayBuffer
from Modules import EnsembleValueNetwork

# Environment setup
env = gym.make('Pendulum-v1')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Hyperparameters
epochs = 30
steps_per_epoch = 4000
gamma = 0.99
lam = 0.95
clip_ratio = 0.2
pi_lr = 3e-4
vf_lr = 1e-3
hidden_sizes = [64, 64]
replay_buffer_capacity = 10000
batch_size = 64

# Initialize models and optimizers
policy = PolicyNetwork(obs_dim, act_dim, hidden_sizes)
value_function = EnsembleValueNetwork(obs_dim, hidden_sizes)
pi_optimizer = Adam(policy.parameters(), lr=pi_lr)
vf_optimizers = [Adam(v.parameters(), lr=vf_lr) for v in value_function.critics]
replay_buffer = ReplayBuffer(replay_buffer_capacity)

# Track performance
learning_curves = {}
policy_loss_curves = {}
value_loss_curves = {}

# Define settings for experiments
settings = {
    "with_clipping": (True, True),
    "without_clipping": (False, True),
    "with_GAE": (True, True),
    "without_GAE": (True, False)
}

# Training and evaluation
for setting_name, (clipping, use_gae) in settings.items():
    learning_curve = []
    policy_loss_curve = []
    value_loss_curve = []

    for epoch in range(epochs):
        # Collect trajectories
        trajectories = collect_trajectories(env, policy, steps_per_epoch)
        for transition in trajectories:
            replay_buffer.store(transition)

        batch = replay_buffer.sample(batch_size)
        states, actions, advantages, returns, log_probs_old = compute_gae(
            batch, value_function, gamma, lam if use_gae else 0.0
        )

        # Update policy
        policy_loss, _ = ppo_update(
            policy, value_function, pi_optimizer, None, states, actions, 
            advantages, returns, log_probs_old, clip_ratio if clipping else 1e6
        )

        # Update value networks
        for critic, optimizer in zip(value_function.critics, vf_optimizers):
            optimizer.zero_grad()
            value_loss = ((critic(states) - returns) ** 2).mean()
            value_loss.backward()
            optimizer.step()

        # Compute average reward
        cumulative_reward = np.mean([t[2] for t in trajectories])

        # Track metrics
        learning_curve.append(cumulative_reward)
        policy_loss_curve.append(policy_loss)
        value_loss_curve.append(value_loss)

        print(f"{setting_name} - Epoch {epoch + 1}: Reward: {cumulative_reward:.2f}")

    # Store results for the setting
    learning_curves[setting_name] = learning_curve
    policy_loss_curves[setting_name] = policy_loss_curve
    value_loss_curves[setting_name] = value_loss_curve

# Visualization
plt.figure(figsize=(16, 10))

# Learning curves
plt.subplot(2, 2, 1)
for setting_name, curve in learning_curves.items():
    plt.plot(range(1, epochs + 1), curve, label=setting_name)
plt.xlabel("Epoch")
plt.ylabel("Accumulated Discounted Reward")
plt.title("Learning Curves")
plt.legend()

# Policy loss curves
plt.subplot(2, 2, 2)
for setting_name, curve in policy_loss_curves.items():
    plt.plot(range(1, epochs + 1), curve, label=setting_name)
plt.xlabel("Epoch")
plt.ylabel("Policy Loss")
plt.title("Policy Loss Curves")
plt.legend()

# Value loss curves (normalized)
plt.subplot(2, 2, 3)
for setting_name, curve in value_loss_curves.items():
    normalized_curve = np.log(np.array([val.detach().numpy() if isinstance(val, torch.Tensor) else val for val in curve]) + 1e-5)
    plt.plot(range(1, epochs + 1), normalized_curve, label=setting_name)
plt.xlabel("Epoch")
plt.ylabel("Log(Value Loss)")
plt.title("Value Loss Curves (Log Scale)")
plt.legend()

plt.tight_layout()
plt.show()

# Simulate trained policy and plot trajectories
theta_trajectories = []
theta_dot_trajectories = []

obs = env.reset()[0]
trajectory_length = 200
for _ in range(trajectory_length):
    obs_tensor = torch.as_tensor(np.array(obs), dtype=torch.float32)
    action, _ = policy.get_action(obs_tensor)
    next_obs, _, done, _, _ = env.step(action)

    theta = np.arctan2(obs[1], obs[0])
    theta_dot = obs[2]

    theta_trajectories.append(theta)
    theta_dot_trajectories.append(theta_dot)

    if done:
        break
    obs = next_obs

# Plot trajectories
plt.figure(figsize=(10, 5))
plt.plot(range(len(theta_trajectories)), theta_trajectories, label="Theta (Angle)")
plt.plot(range(len(theta_dot_trajectories)), theta_dot_trajectories, label="Theta Dot (Angular Velocity)")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.title("Trajectories of Theta and Theta Dot Over Time")
plt.legend()
plt.show()

# 2D value landscape
theta_vals = np.linspace(-np.pi, np.pi, 100)
theta_dot_vals = np.linspace(-8, 8, 100)
V_vals = np.zeros((100, 100))

for i, theta in enumerate(theta_vals):
    for j, theta_dot in enumerate(theta_dot_vals):
        state = np.array([np.sin(theta), np.cos(theta), theta_dot])
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        V_vals[j, i] = value_function(state_tensor).item()

plt.figure(figsize=(6, 5))
plt.imshow(V_vals, extent=[-np.pi, np.pi, -8, 8], origin="lower", aspect="auto", cmap="viridis")
plt.colorbar(label="V(s)")
plt.xlabel("Theta")
plt.ylabel("Theta Dot")
plt.title("2D Landscape of V(s)")
plt.show()
