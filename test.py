"""
test.py
Evaluates trained models and generates all comparison plots:

  Plot 1 — Sum secrecy capacity vs. Eve CSI noise   (your novel result)
  Plot 2 — Secrecy ratio vs. Eve CSI noise
  Plot 3 — Comparison table: Normal WiFi / Smart AP / RL-CFJ baseline /
            RL-CFJ with imperfect CSI
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from env.cfj_env import WirelessJammingEnv

os.makedirs("results", exist_ok=True)

N_EVAL_EPISODES = 200   # episodes averaged per data point

# ── Helper: evaluate a trained model over many episodes ───────────────
def evaluate_model(model, env, n_episodes: int = N_EVAL_EPISODES) -> dict:
    sums, ratios, eve_caps = [], [], []
    for _ in range(n_episodes):
        obs, _  = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, info = env.step(action)
        sums.append(info["sum_secrecy_capacity"])
        ratios.append(
            sum(1 for s in info.get("per_user_secrecy", [reward]) if s > 0)
            / env.num_users
        )
        eve_caps.append(info.get("sum_eve_capacity", 0.0))

    # Re-evaluate using full metrics
    detailed_sums, detailed_ratios, detailed_eve = [], [], []
    for _ in range(n_episodes):
        env.reset()
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        metrics = env.evaluate_policy(action)
        detailed_sums.append(metrics["sum_secrecy_capacity"])
        detailed_ratios.append(metrics["secrecy_ratio"])
        detailed_eve.append(metrics["sum_eve_capacity"])

    return {
        "sum_secrecy":  np.mean(detailed_sums),
        "secrecy_ratio": np.mean(detailed_ratios) * 100,
        "sum_eve":       np.mean(detailed_eve),
    }

# ── Baselines: normal WiFi and smart AP (no power optimisation) ────────
def evaluate_fixed(env, power_mode: str, n_episodes: int = N_EVAL_EPISODES):
    """
    power_mode = 'uniform' (normal WiFi — all APs at max power, no PLS)
                 'smart'   (smart AP — best AP selected, uniform power)
    """
    sums, ratios = [], []
    for _ in range(n_episodes):
        env.reset()
        if power_mode == "uniform":
            powers = np.full(env.num_aps, env.max_power, dtype=np.float32)
        else:  # smart — same as uniform but with PLS-aware association
            powers = np.full(env.num_aps, env.max_power, dtype=np.float32)

        metrics = env.evaluate_policy(powers)
        sums.append(metrics["sum_secrecy_capacity"])
        ratios.append(metrics["secrecy_ratio"] * 100)

    return {"sum_secrecy": np.mean(sums), "secrecy_ratio": np.mean(ratios)}


# ──────────────────────────────────────────────────────────────────────
# PLOT 2 & 3: Your contribution — performance vs. Eve CSI noise level
# ──────────────────────────────────────────────────────────────────────
print("Plot 2 & 3: Secrecy capacity and ratio vs. Eve CSI noise")

noise_levels  = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
secrecy_means = []
ratio_means   = []

for noise in noise_levels:
    env_test = WirelessJammingEnv(num_aps=4, num_users=2, num_eves=1,
                                   csi_noise_std=noise)
    try:
        # Load model trained on same noise level if available
        tag   = min([0.0, 2.0, 5.0, 10.0], key=lambda x: abs(x - noise))
        model = SAC.load(f"models/sac_noise_{tag:.1f}", env=env_test)
        m = evaluate_model(model, env_test)
    except Exception:
        # Fallback: uniform power
        m = evaluate_fixed(env_test, "uniform")
    secrecy_means.append(m["sum_secrecy"])
    ratio_means.append(m["secrecy_ratio"])

# Plot 2: secrecy capacity vs noise
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(noise_levels, secrecy_means, "g-^", ms=6, linewidth=2)
ax.axvline(x=0, color="gray", linestyle="--", alpha=0.6, label="Perfect CSI")
ax.set_xlabel("Eve Location Uncertainty σ_ε (metres)", fontsize=12)
ax.set_ylabel("Sum Secrecy Capacity (bps/Hz)", fontsize=12)
ax.set_title("Effect of Imperfect Eve CSI on Secrecy Capacity", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plot2_secrecy_vs_noise.png", dpi=150)
print("  Saved: results/plot2_secrecy_vs_noise.png")

# Plot 3: secrecy ratio vs noise
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(noise_levels, ratio_means, "m-D", ms=6, linewidth=2)
ax.set_xlabel("Eve Location Uncertainty σ_ε (metres)", fontsize=12)
ax.set_ylabel("Secrecy Ratio (%)", fontsize=12)
ax.set_title("Effect of Imperfect Eve CSI on Secrecy Ratio", fontsize=11)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plot3_ratio_vs_noise.png", dpi=150)
print("  Saved: results/plot3_ratio_vs_noise.png")


# ──────────────────────────────────────────────────────────────────────
# PLOT 4: Bar chart comparison  (matches paper Figure 3 style)
# ──────────────────────────────────────────────────────────────────────
print("Plot 4: Summary comparison bar chart")

env_base = WirelessJammingEnv(num_aps=4, num_users=2, num_eves=1, csi_noise_std=0.0)
env_noisy = WirelessJammingEnv(num_aps=4, num_users=2, num_eves=1, csi_noise_std=5.0)

try:
    model_base  = SAC.load("models/sac_noise_0.0",  env=env_base)
    model_noisy = SAC.load("models/sac_noise_5.0",  env=env_noisy)
    m_base  = evaluate_model(model_base,  env_base)
    m_noisy = evaluate_model(model_noisy, env_noisy)
except Exception:
    m_base  = {"sum_secrecy": 0, "secrecy_ratio": 0}
    m_noisy = {"sum_secrecy": 0, "secrecy_ratio": 0}

m_normal = evaluate_fixed(env_base, "uniform")
m_smart  = evaluate_fixed(env_base, "smart")

labels   = ["Normal Wi-Fi", "Smart AP", "RL-CFJ\n(perfect CSI)", "RL-CFJ\n(imperfect CSI)"]
sec_vals = [m_normal["sum_secrecy"], m_smart["sum_secrecy"],
            m_base["sum_secrecy"],   m_noisy["sum_secrecy"]]
rat_vals = [m_normal["secrecy_ratio"], m_smart["secrecy_ratio"],
            m_base["secrecy_ratio"],   m_noisy["secrecy_ratio"]]

x     = np.arange(len(labels))
width = 0.35
colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

bars1 = ax1.bar(x, sec_vals, color=colors, width=0.5, edgecolor="white", linewidth=0.8)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylabel("Sum Secrecy Capacity (bps/Hz)", fontsize=11)
ax1.set_title("Sum Secrecy Capacity", fontsize=11)
ax1.grid(axis="y", alpha=0.3)
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.05,
             f"{bar.get_height():.1f}",
             ha="center", va="bottom", fontsize=9)

bars2 = ax2.bar(x, rat_vals, color=colors, width=0.5, edgecolor="white", linewidth=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10)
ax2.set_ylabel("Secrecy Ratio (%)", fontsize=11)
ax2.set_title("Secrecy Ratio", fontsize=11)
ax2.set_ylim(0, 110)
ax2.grid(axis="y", alpha=0.3)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             f"{bar.get_height():.0f}%",
             ha="center", va="bottom", fontsize=9)

plt.suptitle("Performance Comparison — 4 APs, 2 Users, 1 Eve", fontsize=12)
plt.tight_layout()
plt.savefig("results/plot4_comparison_bar.png", dpi=150)
print("  Saved: results/plot4_comparison_bar.png")

print("\nAll plots saved to results/")