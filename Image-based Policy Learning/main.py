### main.py ###
import gym
import torch
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch_misc import collect_trajectories, compute_gae, ppo_update, ReplayBuffer
from Modules import ConvolutionalPolicyNetwork, ConvolutionalValueNetwork
from utils import render_pendulum_image

# Environment setup
env = gym.make('Pendulum-v1')
act_dim = env.action_space.shape[0]
obs_dim = (3, 64, 64)  

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
policy = ConvolutionalPolicyNetwork(obs_dim, act_dim, hidden_sizes)
value_function = ConvolutionalValueNetwork(obs_dim, hidden_sizes)
pi_optimizer = Adam(policy.parameters(), lr=pi_lr)
vf_optimizer = Adam(value_function.parameters(), lr=vf_lr)
replay_buffer = ReplayBuffer(replay_buffer_capacity)

# Track performance
learning_curves = {}
policy_loss_curves = {}
value_loss_curves = {}

# Training and evaluation
settings = {
    "with_clipping": (True, True),
    "without_clipping": (False, True),
    "with_GAE": (True, True),
    "without_GAE": (True, False)
}

for setting_name, (clipping, use_gae) in settings.items():
    learning_curve, policy_loss_curve, value_loss_curve = [], [], []

    for epoch in range(epochs):
        # Collect trajectories with image-based observations
        trajectories = collect_trajectories(env, policy, steps_per_epoch, render_pendulum_image, stack_frames=3)

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

        # Update value network
        vf_optimizer.zero_grad()
        value_loss = ((value_function(states)[0] - returns) ** 2).mean()
        value_loss.backward()
        vf_optimizer.step()

        cumulative_reward = np.mean([t[2] for t in trajectories])
        learning_curve.append(cumulative_reward)
        policy_loss_curve.append(policy_loss)
        value_loss_curve.append(value_loss.item())

        print(f"{setting_name} - Epoch {epoch + 1}: Reward: {cumulative_reward:.2f}")

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
    normalized_curve = np.log(np.array(curve) + 1e-5)
    plt.plot(range(1, epochs + 1), normalized_curve, label=setting_name)
plt.xlabel("Epoch")
plt.ylabel("Log(Value Loss)")
plt.title("Value Loss Curves (Log Scale)")
plt.legend()

plt.tight_layout()
plt.show()

# Save the trained models
torch.save(policy.state_dict(), "policy_model.pth")
torch.save(value_function.state_dict(), "value_model.pth")
