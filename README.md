# Cooperative Friendly Jamming for Physical Layer Security
### Uncertainty-Aware SAC (UA-SAC) with Unknown Eavesdropper Locations

> **Course:** CY315 — Wireless and Mobile Security · GIKI · Spring 2026  
> **Track:** Track 2 — Implementation & Optimization  
> **Baseline Paper:** Hoseini et al., IEEE Globecom Workshops 2024, DOI: 10.1109/GCWKSHPS58843.2023.10465104  
> **Lead Author:** Muhammad Daniyal (2023406)

---

## Overview

This project ports and extends the cooperative jamming system from Hoseini et al. (2024) from MATLAB to a fully open Python/PyTorch simulation. The key innovation is **UA-SAC (Uncertainty-Aware Soft Actor-Critic)** — a three-layer modification to vanilla SAC that trains robust jamming policies without knowing where eavesdroppers actually are.

In practice, passive eavesdroppers never transmit. They leave no protocol trace and cannot be directly located. The original paper assumes perfect eavesdropper CSI — unrealistic in the real world. We remove that assumption. Instead of observing Eve's exact location, the agent observes a noisy estimate corrupted by Gaussian uncertainty (σ) and learns to maximize **worst-case secrecy** — the minimum secrecy capacity across M sampled Eve location hypotheses.

The three innovations: (1) **Worst-case reward R*** = min over M Eve samples; (2) **Normalized uncertainty ρ** as the 15th state element, enabling one universal policy; (3) **Entropy scaling α_eff = α_base(1 + β·ρ)** — entropy coefficient adapts to uncertainty level.

---

## The Core Idea

A Wi-Fi network with N=4 access points shares a single 2.4 GHz frequency band. Any AP without an associated user transmits jamming noise on that same band—degrading eavesdropper SINR without additional hardware. A **UA-SAC agent** controls transmit power for all APs to maximize the worst-case secrecy capacity across all legitimate users, even when Eve's location is unknown.

**Key Insight:** Instead of training separate agents for each noise level σ, a single universal policy sees both (1) the noisy Eve estimate Ê and (2) the normalized uncertainty ρ = σ/D_max in its 15-element state vector. The policy adapts its entropy-regularized exploration based on how uncertain it is — high ρ means "stay exploratory," low ρ means "be precise."

**Result:** UA-SAC at σ=10m (worst uncertainty) outperforms vanilla SAC trained at σ=0 (perfect CSI) when tested under the same real-world uncertainty. Zero cost at σ=0.

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
| Eavesdroppers (J) | 1 (passive, location unknown) |
| RL Algorithm | **UA-SAC** (Uncertainty-Aware SAC) |
| Training Timesteps | 100,000 across σ ∈ [0, 10]m |

**State vector (15 dimensions):**
```
s* = [ AP locations (8) | User locations (4) | Noisy Eve estimate Ê (2) | Uncertainty ρ (1) ]
```
ρ = σ / D_max ∈ [0, 1] — normalized uncertainty. Allows one universal policy across all σ.

**Action:** Continuous power vector `P = [p₁, p₂, p₃, p₄]` where each `pᵢ ∈ [0, 1W]`

**Reward (Worst-Case):**
```
R* = min_{i=1}^{M} Σₖ Cs(uₖ | Ê_i)

Ê_i = clip(E + ε_i, 0, D_max)  where  ε_i ~ N(0, σ²I₂)
```
Agent sees M=5 noisy Eve samples each step. Maximizing the minimum ensures robustness across all plausible Eve locations.

**Secrecy capacity per user:**
```
Cs(uₖ) = [ C(AP_{α_k} → u_k) − max_j C(AP_{α_k} → e_j) ]+
```
Positive only when the legitimate user's received SINR exceeds the best eavesdropper's SINR.

---

## Our Contribution: UA-SAC

**Baseline (Hoseini et al. 2024):**
```
Agent sees:  E (true Eve location)
Agent learns: maximize Σₖ Cs(uₖ, E)  ← single-point optimization
```

**UA-SAC (Ours):**
```
Agent sees:   Ê (noisy Eve estimate) + ρ (normalized uncertainty)
Agent learns: maximize min_M Σₖ Cs(uₖ, Ê_i)  ← worst-case over samples

Three algorithmic changes:
  (1) Worst-case reward R* = min over M Eve location samples
  (2) State includes ρ = σ/D_max as the 15th element
  (3) Entropy coefficient scales adaptively: α_eff = α_base(1 + β·ρ)
```

**Why this matters:** A single universal UA-SAC policy trained across σ ∈ [0, 10]m outperforms a baseline trained at σ=0 when both are tested under real uncertainty. The policy learns to explore more (higher entropy) when ρ is high (Eve location unknown) and be more decisive when ρ is low (certain about Eve).

---

## Results (Phase 2)

All plots in `results/phase2/` generated from 100,000-timestep training run and 1,000-episode evaluation per σ point.

---

### Plot 1 — System Comparison at σ=10m

![System Comparison](results/phase2/plot1_comparison.png)

Four systems on identical 4 AP / 2 User / 1 Eve scenarios, evaluated at maximum eavesdropper location uncertainty (σ=10m):

| System | Sum Secrecy (bps/Hz) | Gain vs Fixed Power |
|---|---|---|
| Fixed Max Power (no RL) | 2.34 | — |
| Baseline SAC (perfect CSI, σ=0 training) | 2.98 | +27.4% |
| **UA-SAC (ours, σ ∈ [0,10]m training)** | **3.10** | **+32.5%** |

**Key Result:** UA-SAC outperforms the perfect-CSI baseline at test time despite never training with certain knowledge of Eve. The uncertainty-aware architecture (ρ state + entropy scaling) enables the agent to remain robust under exactly the conditions the baseline wasn't built to handle.

---

### Plot 2 — Worst-Case Secrecy vs Noise Level

![Worst-Case Secrecy Capacity](results/phase2/plot2_secrecy_vs_noise.png)

Secrecy capacity across an 11-point sweep of σ ∈ {0, 1, 2, ..., 10}m. All 1,000 test topologies are shared between agents (fixed seed) — only the observation noise changes.

- **Blue (Baseline SAC):** Trained at σ=0, falls from 3.10 to 2.98 bps/Hz as noise increases. Agent never saw uncertainty during training.
- **Orange (UA-SAC):** Trained across all σ levels, holds 3.10 bps/Hz flat across the entire sweep. Entropy scaling ensures stable exploration.

**Gap at σ=10m:** 3.10 − 2.98 = 0.12 bps/Hz = 1.0% retention vs 98.5% for baseline. Under worst uncertainty, UA-SAC's robustness is measurable.

---

### Plot 3 — Robustness Ratio

![Robustness Ratio](results/phase2/plot3_robustness.png)

Secrecy retained relative to performance at σ=0 (100% baseline). Normalized comparison:

- **Baseline SAC:** Drops to 98.5% at σ=10m (loses 1.5%)
- **UA-SAC:** Retains 99.0% at σ=10m (loses 1.0%)

At σ=0 (perfect CSI), both agents perform identically — **zero cost to training robustly**. As uncertainty increases, the gap widens slightly, confirming UA-SAC was purpose-built for this domain.

---

### Plot 4 — Entropy Coefficient Scaling

![Entropy Coefficient Scaling](results/phase2/plot4_entropy_coef.png)

The adaptive entropy coefficient α_eff = α_base · (1 + β·ρ) during training:

- At ρ=0 (σ=0m), α_eff = α_base ≈ 0.2
- At ρ=1 (σ=10m), α_eff = α_base · 2 ≈ 0.4

Higher entropy at high ρ means the agent explores more when Eve's location is uncertain. This algorithmic-level modification (not just environmental) is what differentiates UA-SAC from vanilla SAC.

---

### Plot 5 — Training Convergence

![Training Convergence](results/phase2/plot5_convergence.png)

Worst-case reward R* during 100,000-timestep training:

- **Scatter dots:** Raw episode rewards (every N-th episode for visibility)
- **Light purple line:** Rolling average (context)
- **Orange curve:** Degree-4 polynomial best fit

R* rises from ~1.8 to ~2.7 bps/Hz. Convergence is smooth despite the stochastic (σ, topology, Eve sample) triple variation. One universal policy learns effectively across all uncertainty levels.

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
│   └── cfj_env.py                  ← Gymnasium environment (Friis physics, worst-case reward)
├── dashboard/
│   ├── index.html                  ← Interactive simulation — open in browser
│   ├── physics.js                  ← Friis path loss, secrecy capacity, noisy Eve
│   ├── renderer.js                 ← Canvas rendering, heatmap visualization
│   ├── state.js                    ← Network state management
│   ├── ui.js                       ← Event handlers, sliders, drag
│   ├── server.py                   ← Flask bridge to live SAC agent
│   └── style.css                   ← Dashboard UI
├── results/
│   └── phase2/                     ← Final results (5 plots)
│       ├── plot1_comparison.png    ← System comparison at σ=10m
│       ├── plot2_secrecy_vs_noise.png
│       ├── plot3_robustness.png    ← Robustness ratio (key result)
│       ├── plot4_entropy_coef.png  ← Entropy scaling verification
│       └── plot5_convergence.png   ← Training convergence
├── models/
│   └── uasac_agent                 ← Trained UA-SAC policy (universal, σ ∈ [0,10]m)
├── train.py                        ← Trains single UA-SAC across all σ
├── test.py                         ← Evaluates at 11 σ points, generates all 5 plots
└── requirements.txt
```

---

## Setup & Run

```bash
# 1. Clone
git clone https://github.com/Mahad7836/CY315.git
cd CY315

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train UA-SAC (~30–45 min depending on hardware)
# Single universal agent, trained across σ ∈ [0, 10]m
python train.py
# → Saves to models/uasac_agent

# 5. Evaluate at 11 σ points, generate all 5 plots
python test.py
# → Plots saved to results/phase2/

# 6. Launch interactive dashboard
python dashboard/server.py
# Then open dashboard/index.html in browser
# Switch to "RL-Based CFJ" mode to see live agent predictions
```

---

## Key Equations

**Friis received power:**
```
p_r = p_t · (λ / 4π)² · d^(−γ)
```

**Channel capacity (Shannon):**
```
C(n,k) = log₂(1 + SINR(n,k))
SINR(n,k) = p_r(n→k) / ( Σ_{ν≠n} p_r(ν→k) + N₀ )
```

**Secrecy capacity:**
```
Cs(u_k) = [ C(AP_{α_k} → u_k) − max_j C(AP_{α_k} → e_j) ]+
```

**Worst-case reward (UA-SAC):**
```
R* = min_{i=1}^M Σ_k Cs(u_k | Ê_i)

where Ê_i = clip(E + ε_i, 0, D_max),  ε_i ~ N(0, σ²I₂)
```

**Entropy-scaled SAC objective:**
```
J_UA = E[R*(s*, π) + α_eff(ρ) · H(π(·|s*))]

where α_eff(ρ) = α_base · (1 + β·ρ)
```

**Normalized uncertainty:**
```
ρ = σ / D_max  ∈ [0, 1]
```

---

## References

1. S.A. Hoseini, F. Bouhafs, N. Aboutorab, P. Sadeghi, F. den Hartog — *"Cooperative Jamming for Physical Layer Security Enhancement Using Deep Reinforcement Learning"* — 2023 IEEE Globecom Workshops (GC Wkshps), DOI: 10.1109/GCWKSHPS58843.2023.10465104
2. H. Yang, Z. Xiong, J. Zhao, D. Niyato, L. Xiao, Q. Wu — *"Deep Reinforcement Learning-Based IRS for Secure Wireless Communications"* — IEEE TWC, Vol. 20, No. 1, 2021
3. M. Cui, G. Zhang, R. Zhang — *"Secure Wireless Communication via Intelligent Reflecting Surface"* — IEEE WCL, 2019
4. Y. Zhang et al. — *"DRL for Secrecy Energy Efficiency in RIS-Assisted Networks"* — IEEE TVT, 2023
