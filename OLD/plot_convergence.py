"""
plot_convergence.py — Plot 5: UA-SAC Training Convergence
Reads raw episode rewards saved during training and draws:
  - Scatter dots every N episodes (showing highs and lows)
  - Polynomial best fit line (degree 4)
  - Light rolling average for context
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

RAW_PATH = "results/uasac_reward_history.npy"
OUT_PATH = "results/phase2/plot5_convergence.png"

if not os.path.exists(RAW_PATH):
    print(f"ERROR: {RAW_PATH} not found. Run train.py first.")
    exit(1)

rewards  = np.load(RAW_PATH)
episodes = np.arange(len(rewards))

# polynomial best fit (degree 4)
degree = 4
coeffs = np.polyfit(episodes, rewards, degree)
poly   = np.poly1d(coeffs)
x_fit  = np.linspace(0, len(rewards) - 1, 500)
y_fit  = poly(x_fit)

# rolling average — background context
window   = max(1, len(rewards) // 60)
smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
x_smooth = np.arange(len(smoothed)) + window // 2

# scatter every N-th episode so dots don't clutter
N      = max(1, len(rewards) // 300)
x_dots = episodes[::N]
y_dots = rewards[::N]

COLOR = "#8b5cf6"

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(x_smooth, smoothed, color=COLOR, linewidth=1.2,
        alpha=0.25, linestyle="-", zorder=1)

ax.scatter(x_dots, y_dots, color=COLOR, s=6, alpha=0.35,
           zorder=2, label="Episode reward")

ax.plot(x_fit, y_fit, color="#f59e0b", linewidth=2.5,
        zorder=3, label=f"Best fit (degree {degree})")

ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("Worst-Case Sum Secrecy R* (bps/Hz)", fontsize=12)
ax.set_title("UA-SAC Training Convergence\nWorst-case reward improving over σ ~ U[0, 10m]",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, linestyle="--")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
