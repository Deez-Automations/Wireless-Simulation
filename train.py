"""
train.py
Trains two SAC agents:
  1. Baseline     — perfect Eve CSI   (csi_noise_std = 0)
  2. Contribution — imperfect Eve CSI (csi_noise_std > 0)

Results are saved to results/ for comparison plotting.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from cfj_env import WirelessJammingEnv

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

TIMESTEPS   = 50_000
NOISE_LEVELS = [0.0, 2.0, 5.0, 10.0]   # metres of Eve location uncertainty

# ── Callback to track reward per episode ──────────────────────────────
class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._ep_reward = 0.0

    def _on_step(self) -> bool:
        self._ep_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0.0
        return True


def train_agent(noise_std: float, timesteps: int = TIMESTEPS):
    """Train one SAC agent for a given Eve CSI noise level."""
    env      = WirelessJammingEnv(num_aps=4, num_users=2, num_eves=1,
                                   csi_noise_std=noise_std)
    callback = RewardLogger()

    model = SAC(
        "MlpPolicy", env,
        verbose=0,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),  # 9-layer equiv hidden units
    )
    model.learn(total_timesteps=timesteps, callback=callback)

    tag = f"noise_{noise_std:.1f}"
    model.save(f"models/sac_{tag}")
    print(f"  [noise={noise_std}m] trained and saved.")

    return model, callback.episode_rewards


# ── Train for each noise level ─────────────────────────────────────────
print("=" * 55)
print("Training SAC agents")
print("=" * 55)

all_rewards = {}
for noise in NOISE_LEVELS:
    label = "Baseline (perfect CSI)" if noise == 0.0 else f"Imperfect CSI σ={noise}m"
    print(f"\n→ {label}")
    _, rewards = train_agent(noise_std=noise)
    all_rewards[noise] = rewards

# ── Plot training convergence ──────────────────────────────────────────
plt.figure(figsize=(8, 4.5))
for noise, rewards in all_rewards.items():
    label = "Baseline (perfect CSI)" if noise == 0.0 else f"σ_ε = {noise} m"
    # Smooth with a rolling window
    window = max(1, len(rewards) // 50)
    smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
    plt.plot(smoothed, label=label)

plt.xlabel("Episode", fontsize=12)
plt.ylabel("Sum Secrecy Capacity (bps/Hz)", fontsize=12)
plt.title("Training Convergence — Baseline vs. Imperfect Eve CSI", fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/training_convergence.png", dpi=150)
print("\nSaved: results/training_convergence.png")