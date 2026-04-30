# Cooperative Friendly Jamming for Physical Layer Security
### Deep Reinforcement Learning under Imperfect Eavesdropper CSI

> **Course:** CY315 — Wireless and Mobile Security · GIKI · Spring 2026  
> **Track:** Track 2 — Implementation & Optimization  
> **Baseline Paper:** Hoseini et al., IEEE Globecom Workshops 2023, DOI: 10.1109/GCWKSHPS58843.2023.10465104

---

## Overview

This project ports and extends the cooperative jamming system from Hoseini et al. (2024) — originally written in MATLAB — to a fully open Python/PyTorch simulation. We introduce a novel contribution: training the SAC agent under **imperfect eavesdropper CSI**, removing the unrealistic assumption that the network knows exactly where the spy is.

In a real Wi-Fi deployment, passive eavesdroppers never transmit. They have no protocol presence and cannot be located. The original paper's state vector includes Eve's exact GPS coordinates — an assumption that does not hold in practice. We corrupt those coordinates with Gaussian noise, train under uncertainty, and show the system stays secure anyway — and actually outperforms the baseline when CSI uncertainty is present at test time.

---

## The Core Idea

A Wi-Fi network with N access points shares a single frequency band. If an AP has no associated user, instead of going idle, it transmits jamming noise — degrading every eavesdropper in the area without any dedicated hardware. A Soft Actor-Critic (SAC) agent learns the optimal transmit power for each AP to maximize total secrecy capacity across all legitimate users.

**Baseline:** agent observes Eve's exact location → learns tight, precise jamming  
**Ours:** agent observes Eve's location + Gaussian noise → learns broader, more robust jamming

The robust agent, when tested under real uncertainty, degrades slower and maintains higher secrecy capacity than the baseline that was never trained to handle noisy information.

---

## Team

| Name | Roll No. |
|---|---|
| M. Daniyal | 2023406 |
| M. Afeef Bari | 2023356 |
| Mahad Aqeel | 2023286 |

---

## System Model

| Parameter | Value |
|---|---|
| Coverage Area | 50m × 50m |
| Frequency | 2.4 GHz (Wi-Fi band) |
| Path Loss Model | Friis, exponent γ = 2 |
| Noise Floor | −85 dBm at all receivers |
| Max Transmit Power | 1 Watt per AP |
| Access Points (N) | 4 |
| Legitimate Users (K) | 2 |
| Eavesdroppers (J) | 1 |
| RL Algorithm | Soft Actor-Critic (SAC) |
| Training Episodes | 50,000 per agent |

**State vector:**
```
s = [ AP locations (8) | User locations (4) | Eve locations (2) ]
```
In our system, the Eve location entries are corrupted: `L_e → L_e + N(0, σ²)`

**Action:** Continuous power vector `P = [p₁, p₂, p₃, p₄]` where each `pᵢ ∈ [0, 1W]`

**Reward:** Sum secrecy capacity across all users
```
R = Σₖ Cs(uₖ)
```

**Secrecy capacity per user (Shannon):**
```
Cs(uₖ) = [ C(APₖ → Userₖ) − max_j C(APₖ → Eve_j) ]+
```
Positive only when the legitimate channel SNR exceeds the eavesdropper's best channel SNR on the same AP signal.

---

## Our Contribution

**Original state (Hoseini et al. 2024):**
```python
obs = [ap_locs, user_locs, eve_locs]           # exact Eve coordinates
```

**Our modified state:**
```python
obs = [ap_locs, user_locs, eve_locs + N(0,σ²)] # noisy Eve coordinates
```

We train four SAC agents at σ ∈ {0, 2, 5, 10} metres. At evaluation, all agents see identically noisy observations (fixed-seed topologies) but are scored against true Eve positions — isolating the effect of CSI uncertainty on policy quality without topology variance confounding the result.

---

## Results

### Plot 1 — Training Convergence

![Training Convergence](results/training_convergence.png)

All four agents converge from ~2.2 bps/Hz to ~3.0 bps/Hz over 50,000 training episodes. The curves overlap tightly across all noise levels — the SAC algorithm successfully learns effective cooperative jamming policies regardless of how uncertain the eavesdropper's location is. The entropy regularization in SAC encourages exploratory behavior that compensates for the blurred observations, explaining why higher noise levels do not visibly slow convergence.

**Takeaway:** Imperfect Eve CSI does not break training. All four agents are viable for deployment.

---

### Plot 2 — Robustness to Imperfect CSI

![Robustness to Imperfect Eve CSI](results/plot2_secrecy_vs_noise.png)

Both the baseline agent (σ=0, trained with perfect knowledge) and our robust agent (σ=10m, trained under maximum uncertainty) are evaluated across seven noise levels from 0 to 10m. All evaluations use the same 1000 fixed-seed network topologies and true Eve positions for scoring — only the observation noise changes.

**What the plot shows:**
- Both lines decrease as noise increases — expected, since the power allocation decisions degrade with worse information
- The green line (our agent) sits consistently above the blue line (baseline) at every noise level
- At σ=0 (perfect CSI), both agents perform equally — no cost to being trained robustly
- At σ=10m (Eve location uncertain within 20% of the map), our agent achieves **3.026 bps/Hz** versus **2.977 bps/Hz** for the baseline — a gap that widens with uncertainty

**Takeaway:** The baseline agent, when exposed to CSI uncertainty it was never trained for, degrades faster. Our agent was built for this and holds up.

---

### Plot 3 — Secrecy Ratio Robustness

![Secrecy Ratio](results/plot3_ratio_vs_noise.png)

Secrecy ratio measures the percentage of legitimate users who achieve positive secrecy capacity — i.e., users who are actually protected. Both agents maintain 97–99% across all noise levels.

This plot tells a complementary story: while absolute secrecy capacity does decrease with noise (Plot 2), almost no user loses protection entirely. The cooperative jamming mechanism is inherently robust — even a sub-optimal power allocation from an uncertain agent keeps most users secure because all APs are jamming simultaneously regardless.

**Takeaway:** The CFJ framework provides structural robustness. The agent only needs to optimize on top of an already-secure foundation.

---

### Plot 4 — System Comparison

![System Comparison](results/plot4_comparison_bar.png)

Four systems compared head-to-head on the same 4 AP / 2 User / 1 Eve scenario:

| System | Sum Secrecy Capacity | Secrecy Ratio | Description |
|---|---|---|---|
| Normal Wi-Fi | 2.3 bps/Hz | 100% | All APs at max power, no PLS awareness |
| Smart AP Selection | 2.3 bps/Hz | 100% | Best AP selected per user, uniform power |
| RL-CFJ — Perfect CSI | 3.0 bps/Hz | 98% | SAC agent with exact Eve location (baseline) |
| RL-CFJ — Imperfect CSI | 3.0 bps/Hz | 98% | SAC agent with σ=10m noise (ours) |

At matched evaluation conditions (σ=0 at test time), our imperfect CSI agent achieves identical secrecy capacity to the perfect CSI baseline — 3.0 bps/Hz. Both RL systems outperform the non-RL baselines by ~30%. The 2% secrecy ratio drop from 100% to 98% is negligible in practice.

The performance advantage of our contribution is visible in Plot 2, where uncertainty is actually applied. This bar chart confirms there is no peacetime cost — our agent is as good as the baseline when conditions are ideal, and better when they are not.

---

## Interactive Dashboard

The dashboard has two layers — a standalone browser simulation (no server needed), and a live AI mode that connects to the trained SAC agent via a local Flask server.

---

### Mode 1 — Static (no server)

```
Open dashboard/index.html directly in any browser.
```

All physics runs in JavaScript in real time. Drag nodes, adjust sliders — metrics update instantly.

---

### Mode 2 — Live AI (with Flask server)

When the server is running, switching to **RL-Based CFJ** mode connects the dashboard to the actual trained neural network. Every node drag triggers a request to the Python agent, which returns real power allocations computed by the SAC policy.

```bash
# Terminal — start the agent server
pip install flask flask-cors
python server.py
# → Model loaded: models/sac_noise_10.0
# → Running on http://127.0.0.1:5050

# Then open dashboard/index.html in browser
# Power section shows: ● AI Online
```

The power sliders lock in RL mode — the agent drives them. Switch to Normal Wi-Fi or Smart AP to unlock manual control.

---

### How the Dashboard Computes Physics

Every frame, `physics.js` runs the full Friis + Shannon pipeline in JavaScript, matching the Python environment exactly.

**Step 1 — Received power (Friis):**
```
p_r(AP_n → Node) = p_n · (λ / 4π)² · (1 / d)^γ

λ = c / f = 3×10⁸ / 2.4×10⁹ = 0.125 m
γ = 2  (free-space path loss exponent)
```

**Step 2 — SINR at legitimate user (Shannon SINR):**
```
SINR(n, k) = p_r(AP_n → User_k) / ( Σ_{ν≠n} p_r(AP_ν → User_k) + N₀ )

N₀ = noise power = 10^((-85 - 30) / 10) = 3.16 × 10⁻¹² W
```
All other APs act as interference — this is what limits user capacity and is also what jams Eve.

**Step 3 — Channel capacity:**
```
C(n, k) = log₂(1 + SINR(n, k))     [W = 1 Hz, normalized]
```

**Step 4 — Secrecy capacity per user:**
```
Cs(u_k) = [ C(AP_{α_k} → User_k) − max_j C(AP_{α_k} → Eve_j) ]+

The [·]+ means take max with 0 — secrecy is never negative.
α_k = the AP associated with user k (best secrecy AP at uniform power).
```

**Step 5 — Sum secrecy (the dashboard's main metric):**
```
Sum Secrecy = Σ_k Cs(u_k)
```

**Step 6 — Secrecy ratio:**
```
Secrecy Ratio = (number of users with Cs > 0) / total users
```

**What the heatmap shows:** For each pixel on the canvas, `physics.js` computes what the secrecy capacity would be if a user were placed at that point. Bright = high secrecy potential, dark = Eve-dominated zone.

**How the noise slider works:** When σ > 0, the dashboard samples `perceived_eve = true_eve + N(0, σ²)` — the orange dot drifts away from the red dot. In RL mode, the agent receives the orange dot's coordinates (what it was trained on), but all physics scoring uses the red dot's true position. This directly replicates the imperfect CSI experiment from our Python training.

**How the Flask bridge works:**
```
JS drag event
    → debounce 80ms
    → POST /predict { aps: [...], users: [...], eve: perceived_position }
    → Python: build 14-element obs vector, model.predict(obs)
    → return { powers: [p1, p2, p3, p4] }
    → JS: animate power sliders, re-render canvas
```

The observation vector order matches `cfj_env._build_obs()` exactly:
```
obs = [AP1x, AP1y, AP2x, AP2y, AP3x, AP3y, AP4x, AP4y,
       U1x,  U1y,  U2x,  U2y,
       Eve_perceived_x, Eve_perceived_y]    ← 14 floats, metres
```

---

## Repository Structure

```
├── env/
│   └── cfj_env.py                  ← Gymnasium environment (Friis physics, SAC MDP)
├── dashboard/
│   ├── index.html                  ← Interactive simulation — open in browser
│   ├── physics.js                  ← Friis path loss, secrecy capacity
│   ├── renderer.js                 ← Canvas rendering, heatmap
│   ├── state.js                    ← Network state management
│   ├── ui.js                       ← Event handlers, sliders, drag
│   └── style.css                   ← Dashboard UI
├── results/
│   ├── training_convergence.png    ← All 4 agents converging
│   ├── plot2_secrecy_vs_noise.png  ← Robustness comparison (key result)
│   ├── plot3_ratio_vs_noise.png    ← Secrecy ratio vs noise
│   └── plot4_comparison_bar.png    ← System comparison bar chart
├── models/
│   ├── sac_noise_0.0               ← Baseline agent (perfect CSI)
│   ├── sac_noise_2.0               ← Agent trained at σ=2m
│   ├── sac_noise_5.0               ← Agent trained at σ=5m
│   └── sac_noise_10.0              ← Robust agent (σ=10m)
├── train.py                        ← Trains all 4 agents sequentially
├── test.py                         ← Evaluates models, generates all plots
└── requirements.txt
```

---

## Setup & Run

```bash
# 1. Clone
git clone https://github.com/Deez-Automations/Wireless-Simulation.git
cd Wireless-Simulation

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install
pip install -r requirements.txt

# 4. Train all 4 agents (~30–60 min depending on hardware)
python train.py

# 5. Evaluate and generate all plots
python test.py
# → plots saved to results/
```

---

## Key Equations

**Friis received power:**
```
p_r = p_t · (λ / 4π)² · d^(−γ)
```

**Channel capacity (Shannon):**
```
C(n,k) = W · log₂(1 + SINR(n,k))
SINR(n,k) = p_r(n→k) / ( Σ_{ν≠n} p_r(ν→k) + N₀ )
```

**Secrecy capacity:**
```
Cs(uₖ) = [ C(αₖ, uₖ) − max_j C(αₖ, Eve_j) ]+
```

**RL Objective:**
```
maximize  Σₖ Cs(uₖ)   subject to  pₙ ∈ [0, P_max]  ∀n
```

---

## References

1. S.A. Hoseini, F. Bouhafs, N. Aboutorab, P. Sadeghi, F. den Hartog — *"Cooperative Jamming for Physical Layer Security Enhancement Using Deep Reinforcement Learning"* — 2023 IEEE Globecom Workshops (GC Wkshps), DOI: 10.1109/GCWKSHPS58843.2023.10465104
2. H. Yang, Z. Xiong, J. Zhao, D. Niyato, L. Xiao, Q. Wu — *"Deep Reinforcement Learning-Based IRS for Secure Wireless Communications"* — IEEE TWC, Vol. 20, No. 1, 2021
3. M. Cui, G. Zhang, R. Zhang — *"Secure Wireless Communication via Intelligent Reflecting Surface"* — IEEE WCL, 2019
4. Y. Zhang et al. — *"DRL for Secrecy Energy Efficiency in RIS-Assisted Networks"* — IEEE TVT, 2023
