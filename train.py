"""
train.py
Trains two SAC agents on the identical current environment code, so the
comparison isolates the unknown-Eve-CSI contribution rather than being
confounded by different network capacity or association mechanics:
  1. Baseline SAC (noise=0.0)  — vanilla SAC, perfect Eve CSI
  2. UA-SAC                    — worst-case reward + rho-augmented state
                                  + rho-scaled entropy

Network depth matches Hoseini et al.'s reported architecture ("nine
layers of depth ... depending on the number of APs") rather than SAC's
generic 2-layer default — see NET_ARCH below.

Models saved to models/. Results (reward/entropy history, convergence
plot) saved to results2/.
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

# ── CPU performance ──────────────────────────────────────────────────────
# This machine has 4 PHYSICAL cores / 8 logical (hyperthreaded) — for
# dense linear algebra, hyperthreads don't give real extra throughput,
# so thread counts target 4, not 8. Must be set before numpy/torch are
# imported — BLAS libraries read these once at import time.
os.environ["OMP_NUM_THREADS"]      = "4"
os.environ["MKL_NUM_THREADS"]      = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import ctypes
import numpy as np
import matplotlib.pyplot as plt
import torch
import threadpoolctl
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from env.cfj_env import WirelessJammingEnv
from uasac import UASAC

torch.set_num_threads(4)

# Raise Windows process priority to High
try:
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(),
        0x00000080  # HIGH_PRIORITY_CLASS
    )
    print("Process priority: HIGH")
except Exception:
    pass

os.makedirs("models", exist_ok=True)
os.makedirs("results2", exist_ok=True)

# Matches Hoseini et al.'s reported "nine layers of depth" for the SAC
# critic/actor networks, re-interpreted for PyTorch: MATLAB's deep
# learning tooling counts each fullyConnectedLayer AND each reluLayer
# separately, so "nine layers" most plausibly means ~4 real weight
# layers with ReLUs between them, not 9 raw weight layers. A first
# attempt at a literal 9-weight-layer plain MLP (no normalization)
# showed exactly the training instability the literature predicts for
# deep, unnormalized actor-critic networks (Bjorck et al., "Towards
# Deeper Deep RL") — critic loss failed to converge, rising late in
# training instead of settling. This tapered 4-layer version targets
# their stated 32-256 hidden-unit range without that instability risk.
NET_ARCH = [256, 128, 64, 32]

# ── Callback to track reward per episode ──────────────────────────────
class RewardLogger(BaseCallback):
    """Tracks per-episode reward across ALL parallel envs, not just env 0."""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._ep_reward = None

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones   = self.locals["dones"]
        if self._ep_reward is None:
            self._ep_reward = np.zeros(len(rewards))
        self._ep_reward += rewards
        for i, done in enumerate(dones):
            if done:
                self.episode_rewards.append(self._ep_reward[i])
                self._ep_reward[i] = 0.0
        return True


def make_baseline_env():
    def _init():
        # Each worker subprocess otherwise inherits the main process's
        # BLAS thread count too — the per-step physics here is a few
        # tiny matrices, doesn't benefit from multithreading, and having
        # every worker also claim several threads causes severe
        # oversubscription against only 4 physical cores.
        threadpoolctl.threadpool_limits(1)
        return WirelessJammingEnv(num_aps=4, num_users=2, num_eves=1)
    return _init


def train_baseline(timesteps: int = 100_000):
    """Baseline SAC — vanilla SAC, perfect Eve CSI, same env/network depth
    as UA-SAC so the comparison isolates only the unknown-Eve contribution."""
    N_ENVS = 4
    env = SubprocVecEnv([make_baseline_env() for _ in range(N_ENVS)])

    model = SAC(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=NET_ARCH),
    )
    model.learn(total_timesteps=timesteps)
    model.save("models/sac_noise_0.0")
    env.close()
    print("Baseline SAC saved → models/sac_noise_0.0")


# ══════════════════════════════════════════════════════════════════════
# Phase 2 — UA-SAC (Uncertainty-Aware SAC)
#
# Single universal agent trained across all σ levels simultaneously.
# State: ℝ¹⁵  (adds ρ = σ/D_max as element 15)
# Reward: worst-case over M=5 sampled Eve locations per step
# ══════════════════════════════════════════════════════════════════════

def make_env(beta: float = 0.0):
    def _init():
        threadpoolctl.threadpool_limits(1)
        return WirelessJammingEnv(
            num_aps=4, num_users=2, num_eves=1,
            sigma_range=(0.0, 10.0),
            M=5,
            beta=beta,
            augment_rho=True,
        )
    return _init


def train_uasac(timesteps: int = 300_000, beta: float = 0.0, tag: str = "uasac_robust"):
    """beta=0 disables the rho-scaled entropy boost, isolating the
    worst-case-reward mechanism alone. tag controls the save name, so an
    ablation run (e.g. beta=1 at the same 300k budget) doesn't overwrite
    the working model — needed to isolate whether beta=0 or the extra
    training budget is what actually fixed convergence."""
    N_ENVS = 4   # parallel env workers — leaves 4 cores for PyTorch training
    env = SubprocVecEnv([make_env(beta=beta) for _ in range(N_ENVS)])

    model = UASAC(
        "MlpPolicy", env,
        beta=beta,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=200_000,      # larger buffer to match faster data collection
        batch_size=256,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=NET_ARCH),
    )

    callback = RewardLogger()
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(f"models/{tag}")
    model.save_ent_history(f"results2/{tag}_ent_history.npz")
    np.save(f"results2/{tag}_reward_history.npy", np.array(callback.episode_rewards))
    env.close()
    print(f"UA-SAC saved → models/{tag}")

    window   = max(1, len(callback.episode_rewards) // 50)
    smoothed = np.convolve(callback.episode_rewards,
                           np.ones(window) / window, mode="valid")
    plt.figure(figsize=(8, 4.5))
    plt.plot(smoothed, color="#8b5cf6", linewidth=2, label="UA-SAC (σ ~ U[0,10])")
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Worst-Case Sum Secrecy (bps/Hz)", fontsize=12)
    plt.title(f"UA-SAC Training Convergence (beta={beta})", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results2/{tag}_convergence.png", dpi=150)
    print(f"Saved: results2/{tag}_convergence.png")


if __name__ == "__main__":
    # Baseline SAC already trained and verified working (models/sac_noise_0.0) —
    # not retrained here. models/uasac_robust (beta=0, 300k) is the current
    # working result — also not retrained here, saved under its own tag so
    # this ablation run can't overwrite it.
    #
    # Ablation: beta=1 (entropy boost back ON) at the SAME 300k budget as
    # the working beta=0 run. Isolates whether beta=0 was the real fix, or
    # whether the extra training budget alone would have been enough —
    # the previous beta=0 vs beta=1 comparison changed both variables at
    # once (100k steps vs 300k steps too), so it couldn't tell us which.
    print("=== Ablation: UA-SAC, 300k steps, beta=1 (entropy boost ON) ===")
    train_uasac(timesteps=300_000, beta=1.0, tag="uasac_beta1_300k")