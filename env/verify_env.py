"""
Week 1 verification script.
Run this to confirm your environment works before writing any DRL code.
Expected output: secrecy rate values, SNR values, and 2 saved plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from env.channel import IRSChannel

# ── 1. Basic sanity check ──────────────────────────────────────────────
print("=" * 55)
print("TEST 1: Single channel realization")
print("=" * 55)

env = IRSChannel(N_t=4, N_irs=32, sigma2=1e-3, seed=42)
env.generate_channels()

# random phases and beamforming
theta = np.random.uniform(-np.pi, np.pi, env.N_irs)
env.set_phases_continuous(theta)

w = np.random.randn(env.N_t) + 1j * np.random.randn(env.N_t)
w = w / np.linalg.norm(w)   # normalize beamforming vector

Rs, snr_bob, snr_eve = env.compute_secrecy_rate(w, P_tx=1.0)
print(f"  SNR Bob  : {10*np.log10(snr_bob):.2f} dB")
print(f"  SNR Eve  : {10*np.log10(snr_eve):.2f} dB")
print(f"  Secrecy rate Rs : {Rs:.4f} bits/s/Hz")
print()

# ── 2. Discrete phase test ────────────────────────────────────────────
print("=" * 55)
print("TEST 2: Discrete phase shifts (2-bit)")
print("=" * 55)
q_angles = env.set_phases_discrete(theta, bits=2)
unique_levels = np.unique(np.round(np.degrees(q_angles)))
print(f"  Quantized to levels (deg): {unique_levels}")
Rs_d, snr_bob_d, snr_eve_d = env.compute_secrecy_rate(w, P_tx=1.0)
print(f"  Rs continuous : {Rs:.4f}  |  Rs discrete : {Rs_d:.4f}")
print()

# ── 3. Worst-case secrecy (Contribution 1) ────────────────────────────
print("=" * 55)
print("TEST 3: Worst-case secrecy (no Eve CSI)")
print("=" * 55)
env.set_phases_continuous(theta)
Rs_wc = env.compute_worst_case_secrecy(w, P_tx=1.0, n_samples=50)
print(f"  True Rs       : {Rs:.4f}")
print(f"  Worst-case Rs : {Rs_wc:.4f}")
print()

# ── 4. State vector ───────────────────────────────────────────────────
print("=" * 55)
print("TEST 4: State vector dimensions")
print("=" * 55)
s_with    = env.get_state(include_eve_csi=True)
s_without = env.get_state(include_eve_csi=False)
print(f"  State dim (with Eve CSI)    : {len(s_with)}")
print(f"  State dim (without Eve CSI) : {len(s_without)}")
print(f"  Expected with    : {env.state_dim(True)}")
print(f"  Expected without : {env.state_dim(False)}")
print()

# ── 5. Plot 1: Secrecy rate vs SNR (random phases) ────────────────────
print("=" * 55)
print("PLOT 1: Secrecy rate vs transmit SNR")
print("=" * 55)

P_tx_dB_range = np.linspace(-10, 30, 30)   # dB
P_tx_range    = 10 ** (P_tx_dB_range / 10)  # linear

Rs_cont_list  = []
Rs_disc_list  = []
Rs_wc_list    = []

for P_tx in P_tx_range:
    env.generate_channels()
    theta = np.random.uniform(-np.pi, np.pi, env.N_irs)
    w     = np.random.randn(env.N_t) + 1j * np.random.randn(env.N_t)
    w     = w / np.linalg.norm(w)

    env.set_phases_continuous(theta)
    Rs_c, _, _ = env.compute_secrecy_rate(w, P_tx)
    Rs_cont_list.append(Rs_c)

    env.set_phases_discrete(theta, bits=2)
    Rs_d, _, _ = env.compute_secrecy_rate(w, P_tx)
    Rs_disc_list.append(Rs_d)

    env.set_phases_continuous(theta)
    Rs_wc_list.append(env.compute_worst_case_secrecy(w, P_tx, n_samples=30))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(P_tx_dB_range, Rs_cont_list, 'b-o',  ms=4, label='Continuous phases (known Eve CSI)')
ax.plot(P_tx_dB_range, Rs_disc_list, 'r-s',  ms=4, label='Discrete phases 2-bit (known Eve CSI)')
ax.plot(P_tx_dB_range, Rs_wc_list,   'g--^', ms=4, label='Worst-case (no Eve CSI)')
ax.set_xlabel('Transmit SNR (dB)', fontsize=12)
ax.set_ylabel('Secrecy Rate (bits/s/Hz)', fontsize=12)
ax.set_title('Secrecy Rate vs SNR — Random IRS Phases (Week 1 Baseline)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('../results/week1_secrecy_vs_snr.png', dpi=150)
print("  Saved: results/week1_secrecy_vs_snr.png")

# ── 6. Plot 2: Secrecy rate vs number of IRS elements ─────────────────
print("PLOT 2: Secrecy rate vs IRS elements N")

N_range  = [8, 16, 32, 64, 128]
Rs_N_cont = []
Rs_N_disc = []
P_tx_fixed = 10.0   # 10 dB

for N in N_range:
    env_n = IRSChannel(N_t=4, N_irs=N, sigma2=1e-3)
    env_n.generate_channels()
    theta = np.random.uniform(-np.pi, np.pi, N)
    w     = np.random.randn(4) + 1j * np.random.randn(4)
    w     = w / np.linalg.norm(w)

    env_n.set_phases_continuous(theta)
    Rs_c, _, _ = env_n.compute_secrecy_rate(w, P_tx_fixed)
    Rs_N_cont.append(Rs_c)

    env_n.set_phases_discrete(theta, bits=2)
    Rs_d, _, _ = env_n.compute_secrecy_rate(w, P_tx_fixed)
    Rs_N_disc.append(Rs_d)

fig2, ax2 = plt.subplots(figsize=(7, 4.5))
ax2.plot(N_range, Rs_N_cont, 'b-o', ms=5, label='Continuous phases')
ax2.plot(N_range, Rs_N_disc, 'r-s', ms=5, label='Discrete phases 2-bit')
ax2.set_xlabel('Number of IRS Elements (N)', fontsize=12)
ax2.set_ylabel('Secrecy Rate (bits/s/Hz)', fontsize=12)
ax2.set_title('Secrecy Rate vs IRS Size — Random Phases (Week 1)', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('../results/week1_secrecy_vs_N.png', dpi=150)
print("  Saved: results/week1_secrecy_vs_N.png")

print()
print("=" * 55)
print("Week 1 environment verified. Ready for DRL (Week 2).")
print("=" * 55)
