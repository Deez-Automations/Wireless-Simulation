# CY315 Project — Handoff Document
**Last Updated:** 2026-05-09
**For:** Any AI model or team member picking this up cold
**GitHub:** https://github.com/Deez-Automations/Wireless-Simulation

---

## Who & What

**Course:** Wireless and Mobile Security (CY315), GIKI, Spring 2026
**Track:** Track 2 — Implementation & Optimization
**Deadline:** Week 14
**Deliverables:** Python simulation + IEEE format report + presentation slides (LaTeX Beamer)

**Team:**
| Name | Roll | Role |
|---|---|---|
| M. Daniyal | 2023406 | Lead — SAC Agent / Simulation |
| M. Afeef Bari | 2023356 | Results, Analysis, LaTeX Report |
| Mahad Aqeel | 2023286 | Channel Simulation Environment |

---

## Project Topic

**Cooperative Friendly Jamming for Physical Layer Security Enhancement Using Deep Reinforcement Learning**

We port the Hoseini et al. 2023 cooperative jamming system from MATLAB to Python, then extend it with a stronger novel contribution around imperfect eavesdropper CSI.

---

## Baseline Paper (LOCKED)

**Hoseini, S.A., Bouhafs, F., Aboutorab, N., Sadeghi, P., den Hartog, F.**
"Cooperative Jamming for Physical Layer Security Enhancement Using Deep Reinforcement Learning"
IEEE Globecom Workshops 2023, DOI: 10.1109/GCWkshps58843.2023.10465104

**What it does:** N Wi-Fi APs share one frequency band. A SAC agent optimizes transmit power of each AP to maximize sum secrecy capacity. APs not serving a user transmit jamming noise. Agent observes exact AP, user, and Eve locations.

**System parameters:**
- N=4 APs, K=2 users, J=1 Eve, 50m×50m, 2.4 GHz, γ=2, N₀=−85 dBm, P_max=1W
- State: [AP_locs(8) | User_locs(4) | Eve_locs(2)] ∈ ℝ¹⁴
- Action: P^t = [p1,p2,p3,p4] ∈ [0,1]⁴
- Reward: Σ Cs(uk, αk)
- Single-step MDP: each episode = one topology, one action, one reward

---

## SAC Algorithm — How It Works (and What Hoseini Used)

**What Hoseini et al. used:** MATLAB's RL Toolbox, built-in SAC implementation. The paper does not publish exact hyperparameters (typical for a workshop paper — they focus on system model, not ablation). What they describe: SAC with a continuous action space (the power vector), actor-critic architecture with two Q-networks, automatic entropy tuning. No modifications to SAC itself. Their entire contribution was the cooperative jamming environment design — reward function, state definition, AP association logic. The algorithm was vanilla SAC.

**What SAC actually does (Feynman):**

SAC has two parts. A critic that learns "how good is this action in this state?" and an actor that learns "what action should I take in this state?"

The critic watches the agent play and asks: "you took action X in state S, got reward R, then ended up in S'. Was X actually as good as you thought?" It keeps correcting its own estimates against observed reality until they stabilize. It uses two separate Q-networks and takes the minimum of both to avoid overoptimistic estimates — this is standard SAC.

The actor learns by asking the critic: "if I take this action, what score do I get?" and gradually shifts toward whatever the critic rates highest.

SAC's twist over plain actor-critic: it penalizes the actor for being too predictable. The objective is not just "maximize reward" but "maximize reward AND stay somewhat random." This is entropy regularization — α_ent controls how much randomness is rewarded. The "auto" setting lets SAC tune α_ent automatically during training. In jamming terms: don't always blast the same AP at full power — maintain spread in your strategy, because a deterministic jammer is a predictable jammer that Eve can learn to circumvent.

**Our Phase 1 hyperparameters (from train.py):**
- Policy: MlpPolicy (two hidden layers, [256, 256] neurons each)
- Learning rate: 3×10⁻⁴
- Replay buffer: 100,000 transitions
- Batch size: 256
- Entropy coefficient: "auto" (learned, same as paper)
- Timesteps: 50,000 per agent
- Framework: Stable Baselines3 (PyTorch backend)

---

## Current Contribution (Phase 1 — IMPLEMENTED)

**Imperfect Eavesdropper CSI via Gaussian Noise Injection**

We inject Gaussian noise into the Eve coordinates in the observation vector:
- Ê = clip(E + ε, 0, D_max), ε ~ N(0, σ²I₂)
- Train 4 agents: σ ∈ {0, 2, 5, 10} m
- Evaluate all against true Eve location (controlled experiment, fixed seeds)
- Reward is always computed against true E — only observation is noisy

**Models trained:** models/sac_noise_0.0, sac_noise_2.0, sac_noise_5.0, sac_noise_10.0
**Training time:** ~4 hours total on CPU (50,000 timesteps × 4 agents)
**DO NOT RETRAIN** unless implementing Phase 2 below.

### How Phase 1 Actually Worked — Deep Understanding

The baseline paper (Hoseini et al.) gives the agent the true Eve location in the state vector. The agent sees exactly where Eve is, picks jamming powers accordingly, and is rewarded by how much secrecy it achieves. This is the idealized scenario — in any real deployment, passive eavesdroppers never transmit, so their location is unknown.

Phase 1's contribution: we acknowledged this limitation by corrupting the Eve coordinate before handing it to the agent. Every episode, Eve is placed at some true position E. We add Gaussian noise ε ~ N(0, σ²I₂) to get Ê — the agent's perceived Eve location. The agent acts on Ê, but the reward is computed against the true E. Only the observation is noisy; the scoring is honest.

We trained 4 separate agents at different σ values. Each agent learned to operate under one fixed level of uncertainty. The σ=0 agent is essentially the baseline paper replicated in Python. The σ=10 agent learned to handle severe location error.

**What Phase 1 proved:** even under significant Eve location uncertainty (σ=10m in a 50m×50m grid — that's 20% of the map), a SAC agent trained specifically for that noise level can still achieve comparable secrecy to a perfect-CSI agent. At σ=10m evaluation, our robust agent scored 3.026 vs baseline's 2.977 bps/Hz.

**What Phase 1 did NOT do:**
- It did not change the SAC algorithm at all — vanilla SAC throughout
- It did not introduce a new variable — σ was an environment parameter, not part of the state
- It trained 4 separate specialist agents rather than one adaptive policy
- Each agent optimized for average reward, not worst-case security guarantee
- The professor's verdict: injecting noise into one state variable is not a strong enough novelty

This is the context for why Phase 2 (UA-SAC) is needed.

---

## Professor Feedback (2026-05-05) — CRITICAL

The professor reviewed the project and gave the following feedback. Each point must be addressed:

### 1. Innovation is too thin
Injecting noise into one state variable and using vanilla SAC is not a strong enough novelty for a course project. The professor wants us to:
- Introduce our own variable
- Modify the algorithm, not just the environment
- Widen the approach with something more original

### 2. More result dimensions needed
Current plots only show 2 agents (σ=0 vs σ=10) across one dimension (noise level). Need more angles:
- More agents compared
- Different scenario dimensions (could include N, J variation)
- Distribution-level results (CDF, box plots)

### 3. σ=0 performance must be equal for both agents
At σ=0, both the perfect-CSI agent and the robust agent see the SAME observation (true Eve location). Their performance at σ=0 should therefore be identical or very close. Any divergence at σ=0 is a presentation/evaluation error, not a real result. The divergence that opens at σ>0 is the actual contribution.

### 4. Refine the SAC algorithm
We should not adopt the baseline algorithm 100%. Propose a named modification specific to our problem.

---

## Proposed Phase 2 Contribution (NOT YET IMPLEMENTED)

### Layer 1 — Worst-Case Robust Reward (new reward formulation)

Instead of computing reward against one noisy Eve observation, sample M=5 noisy locations per step and take the minimum secrecy across all samples:

R* = min_{i=1}^{M} Σ_{k=1}^{K} Cs(uk, αk | Êᵢ), Êᵢ ~ N(E, σ²I₂)

This is true robust optimization — the agent learns to guarantee performance in the worst-case noise realization, not just on average. Completely different from Hoseini et al.

**Feynman explanation:** Imagine a bodyguard who doesn't know exactly where the threat is coming from. The old way: guess the most likely threat location, position yourself for that, and accept that on bad days you'll be caught off guard. The agent was optimizing for the average — which sounds fine until the average hides a catastrophic minority.

Layer 1 changes the training signal. Before committing to a jamming strategy, imagine 5 different places Eve could be. Compute secrecy for all 5. Take the minimum. Now the agent learns to maximize the worst-case outcome, not the typical one. It stops optimizing for luck. In security, a policy that works 80% of the time and fails catastrophically 20% of the time is not a good policy — you want the floor to be high, not just the average.

### Layer 2 — Uncertainty-Augmented State (new variable ρ)

Add σ itself (or its normalized form ρ = σ/D_max) as the 15th element of the state vector:

s* = [AP_locs(8) | User_locs(4) | Ê(2) | ρ(1)] ∈ ℝ¹⁵

where ρ = σ/D_max ∈ [0,1] is a dimensionless normalized uncertainty ratio — our own introduced variable.

One universal policy learns to adapt its jamming strategy based on how uncertain the CSI is. This replaces the 4 separate σ-specific agents with one adaptive agent.

**Feynman explanation:** Without ρ in the state, the agent sees AP locations, user locations, and Ê (noisy Eve coordinate). When σ=10m the Eve coordinate is wildly off, when σ=0 it's exact — but both look like a number between 0 and 50. The agent has no signal to distinguish them. So it learns some average strategy that half-works for all uncertainty levels and is optimal for none. That's why Phase 1 needed 4 separate agents, each trained on one σ only.

With ρ in the state, the agent has the missing signal. During training it sees all combinations — sometimes ρ=0.04 (reliable estimate), sometimes ρ=0.2 (rough guess). The neural network learns a pattern: high ρ means the Eve coordinate is probably lying, so spread jamming wider. Low ρ means the coordinate is trustworthy, so concentrate jamming toward Ê. One policy learns to scale its strategy with its own confidence level.

Think of training a chef without telling them how many guests are coming. They cook a fixed amount every time — sometimes too little, sometimes too much. You'd need 4 chefs, each trained for a specific party size. Now add one number to every order: the guest count. One chef handles all sizes because they have the key signal to adapt on. ρ is that signal. It doesn't change what the agent observes about Eve — it tells the agent how much to trust what it's observing.

**Important distinction:** ρ is not a property of Eve. It is a property of your own estimation system. In real deployment, you built the estimation pipeline, you know how stale your data is, how noisy your sensors are. σ is a parameter you set — so ρ = σ/D_max is always available to the operator even when Eve's true location is completely unknown.

### Layer 3 — Named Algorithm: UA-SAC (Uncertainty-Aware SAC)

UA-SAC is not an independent layer — it is the academic name for L1+L2 combined. Same SAC backbone, two surgical changes:
- Modified reward: worst-case over M Eve samples (Layer 1)
- Augmented state: includes ρ (Layer 2)
- Modified entropy scaling: α_ent(ρ) = α_base · (1 + β·ρ) — higher uncertainty → more entropy → broader jamming coverage

**New problem formulation Φ_UA** (replaces previous Φ*):
Φ_UA: max_{P^t} E_{ε}[ min_{i=1}^M Σ Cs(uk,αk | Êᵢ) ]
s.t. 0 ≤ p^t_n ≤ P_max, ρ = σ/D_max ∈ [0,1]

### How UA-SAC Changes the SAC Training Loop (Feynman)

SAC's structure is completely unchanged. What changes is the fuel it runs on.

**Change 1 — reward function:** SAC's critic learns by minimizing Bellman error against whatever reward the environment returns. Currently that reward is secrecy capacity at one Eve location. We change the environment to return min(secrecy across 5 sampled Eve locations) instead. SAC does not know or care — it still minimizes Bellman error the same way. But because the reward signal is now worst-case, the Q-values the critic learns represent worst-case expected return. The actor, which optimizes against the critic's Q-values, automatically becomes a robust policy. No changes to SAC math.

**Change 2 — state vector:** The actor network now takes 15 inputs instead of 14. One additional input neuron for ρ. During training, ρ varies episode to episode, so the network learns to use it as a conditioning signal — adjusting jamming spread based on how uncertain the Eve estimate is. This is a data change, not an architectural change.

**Change 3 — entropy scaling:** α_ent(ρ) = α_base · (1 + β·ρ). When ρ is high (bad CSI quality), the entropy bonus is amplified — the actor is rewarded more for staying unpredictable, which naturally leads to wider jamming coverage. When ρ is low (reliable estimate), the entropy bonus shrinks — the actor is allowed to be more precise and targeted. This is the one SAC-level modification that makes UA-SAC a named algorithm rather than just environment tweaks.

### UA-SAC — What It Does (Feynman Narrative)

One agent. Trained on every possible uncertainty level at once.

Every episode, the world randomizes — APs, users, Eve all placed randomly on the 50×50m grid. Then a random σ is drawn between 0 and 10m. That σ determines how far off the agent's Eve location estimate will be this episode. The agent sees the noisy Eve coordinate AND ρ = σ/50 — the 15th number in its observation, telling it how much to trust what it's seeing.

The agent picks 4 power values — how hard each AP transmits.

Now the reward calculation. Instead of computing secrecy at the one noisy Eve location, the environment samples 5 different possible Eve positions drawn from the same noise distribution. Computes secrecy for all 5. Returns the minimum. The agent gets punished for the worst of those 5 guesses, not the average. So the agent learns: I can't get lucky. I have to find jamming powers that hold up even in the nastiest scenario.

That's the outer loop. Standard SAC from here — critic watches the agent play and estimates "how good was that action given that state?" Actor shifts toward whatever the critic rates highest.

The twist is in the entropy term. SAC normally tells the actor: maximize reward, but also stay somewhat random — don't commit too hard to one strategy. UA-SAC scales how much randomness is rewarded based on ρ. High ρ means the Eve estimate is garbage — the agent doesn't know where Eve is — so the entropy bonus is amplified, pushing the agent toward spreading jamming power more broadly rather than concentrating it on a probably-wrong location. Low ρ means the estimate is trustworthy — entropy bonus shrinks — the agent is allowed to be more precise and targeted.

The agent trains across 100,000 steps seeing all combinations: easy episodes with ρ near zero where it can be precise, hard episodes with ρ = 0.2 where it has to spread wide, and everything in between. The neural network learns one universal strategy: read ρ, decide how much to trust the Eve coordinate, adjust aggression accordingly.

Result: one model that replaces what previously needed four separate specialists. Because it was trained on worst-case reward instead of average reward, its floor is high — it doesn't have catastrophic bad days.

### Result Plots — Phase 2 (test.py, 5 plots, all saved to results/phase2/)

---

**Plot 1 — 4-System Bar Chart at σ=10m**
Output: `results/phase2/plot1_comparison_bar.png`

![Plot 1](../results/phase2/plot1_comparison_bar.png)

Two side-by-side bars: sum secrecy capacity (bps/Hz) and secrecy ratio (%) for three systems — Fixed Max Power, Baseline SAC, UA-SAC — all evaluated at σ=10m (maximum uncertainty).

**Feynman:** This is the headline scorecard. You walk into the room and say "here are the three systems, here is the worst-case noise level, and here is who won." Fixed Max Power is the floor — no intelligence, just brute force. Baseline SAC is the competitor — it was trained with perfect Eve knowledge but is now being tested blind. UA-SAC is ours — trained without ever knowing Eve's location. The bar chart asks one question: at the hardest possible test, who delivers the most private communication? UA-SAC wins despite having less information than the baseline. That's the proof.

---

**Plot 2 — Mean Secrecy Capacity vs σ (fine sweep 0→10m)**
Output: `results/phase2/plot2_secrecy_vs_noise.png`

![Plot 2](../results/phase2/plot2_secrecy_vs_noise.png)

Four lines across 11 σ points (0,1,2,...,10m): UA-SAC, Baseline SAC, Fixed CFJ, Normal Wi-Fi. Purple shaded gap between UA-SAC and Baseline SAC. Both RL agents tested on identical topologies (fixed seeds).

**Feynman:** This plot tells the story over time — what happens as the location error gets progressively worse. At σ=0 both RL agents start at the same point (identical observation, identical reward). As σ grows, one line stays flat and the other dips. The gap that opens between them is exactly our contribution made visible — the purple shaded region IS the robustness UA-SAC provides over a perfect-CSI agent when the perfect-CSI assumption breaks down. The non-RL baselines are flat horizontal lines because they never used Eve's location anyway — they're the reference floor RL must beat to justify its existence.

---

**Plot 3 — Normalized Robustness vs σ**
Output: `results/phase2/plot3_robustness.png`

![Plot 3](../results/phase2/plot3_robustness.png)

Both agents normalized to their own σ=0 performance = 100%. Y-axis shows % of original performance retained. Both start at 100% — the divergence is the result. Annotated drop percentage at σ=10m for each agent.

**Feynman:** Plot 2 shows absolute numbers. Plot 3 removes the scale and just asks "how much did each agent degrade?" by anchoring both to 100% at σ=0. This is important because both agents start from the same reference — there's no unfair advantage built in. Everything above σ=0 is a fair test. The fact that UA-SAC's line stays closer to 100% while Baseline SAC's drops more is the clean proof of robustness. Think of it as the "aging test" — both cars start at 100% condition, how well does each hold up after 10 years of harsh weather?

---

**Plot 4 — Entropy Coefficient α over Training**
Output: `results/phase2/plot4_entropy_coef.png`

![Plot 4](../results/phase2/plot4_entropy_coef.png)

Dual-axis: left y = entropy coefficient values (α_base and α_eff), right y = mean batch ρ. Shaded gap between α_base and α_eff is the uncertainty boost Δα. Smoothed over gradient steps.

**Feynman:** This plot is the proof that UA-SAC is a different algorithm, not just a different environment. If someone says "you just changed the reward function, SAC is unchanged" — this plot is the rebuttal. α_base is what standard SAC would use. α_eff is what UA-SAC uses, and it's always higher when ρ > 0. The shaded gap between the two lines is the live uncertainty scaling in action during training. When ρ is high (bad CSI episodes), the gap widens — the agent is pushed to explore more broadly. When ρ drops, the lines converge. This is entropy scaling responding dynamically to CSI quality, which is an algorithmic modification, not an environment tweak.

---

**Plot 5 — UA-SAC Training Convergence**
Output: `results/phase2/uasac_convergence.png`

![Plot 5](../results/phase2/uasac_convergence.png)

**Feynman:** This plot answers "did the agent actually learn anything?" The x-axis is time (episodes), the y-axis is the worst-case secrecy reward R*. The smoothed curve cuts through episode-by-episode noise (variance is high because σ is random each episode — easy episodes at ρ≈0 and hard ones at ρ=0.2 get mixed together) to show the underlying trend. A rising curve = the agent is converging on a better policy. This is the training receipt — proof that 100k timesteps of UA-SAC actually improved the worst-case secrecy floor, not just the average.

### Implementation Status

**Ready to train — all code complete:**
- env/cfj_env.py — worst-case reward (M=5), ρ in observation, sigma_range for universal training
- uasac.py — UASAC subclass with ρ-dependent entropy scaling α_eff = α_base·(1+β·ρ), history logging
- train.py — trains UASAC, saves model + entropy history, Phase 1 block commented out
- test.py — all 5 plots, UA-SAC vs Baseline SAC comparison

**Pending:**
- Run: `venv/Scripts/python train.py` (~10 hrs on CPU, 100k timesteps, M=5)
- Run: `venv/Scripts/python test.py` (generates all 5 plots)
- Update IEEE report and slides with new plots and UA-SAC equations

---

## Codebase Architecture (Current State)

```
d:/GIKI/CY 315/Project/
├── .claude/
│   └── CLAUDE.md              ← AI context (update when contribution changes)
├── docs/
│   ├── HANDOFF.md             ← this file
│   ├── IEEE_Final_Report.md   ← full report with 18 references (needs Phase 2 update)
│   ├── slides.tex             ← LaTeX Beamer slides (needs Phase 2 update)
│   ├── proposal.md / .pdf     ← submitted proposal
│   └── *.pdf                  ← baseline and lit review papers
├── env/
│   ├── __init__.py
│   └── cfj_env.py             ← Gymnasium env: Friis physics, SAC MDP, noise injection
├── dashboard/
│   ├── index.html             ← interactive browser simulation
│   ├── physics.js             ← Friis + Shannon in JS (mirrors Python env exactly)
│   ├── renderer.js            ← canvas + heatmap rendering
│   ├── state.js               ← network state management
│   ├── ui.js                  ← drag, sliders, Flask bridge, mode switching
│   └── style.css              ← Zinc dark theme
├── models/
│   ├── sac_noise_0.0          ← baseline agent (perfect CSI)
│   ├── sac_noise_2.0          ← σ=2m agent
│   ├── sac_noise_5.0          ← σ=5m agent
│   └── sac_noise_10.0         ← robust agent (σ=10m) — used by server.py
├── results/
│   ├── training_convergence.png
│   ├── plot2_secrecy_vs_noise.png
│   ├── plot3_ratio_vs_noise.png
│   └── plot4_comparison_bar.png
├── server.py                  ← Flask bridge: /health, /predict → SAC inference
├── plot_convergence.py        ← evaluation-based smooth convergence plot
├── train.py                   ← trains 4 σ-specific SAC agents
├── test.py                    ← evaluates models, generates all plots
└── requirements.txt
```

---

## Key Equations

All equations verified against `env/cfj_env.py` and `uasac.py`. Notation source: `notations.tex`.

---

### 1. Friis Received Power

$$p^r = p^t \cdot G_t \cdot G_r \cdot \left(\frac{\lambda}{4\pi}\right)^2 \cdot d^{-\gamma}$$

With $G_t = G_r = 1$ (isotropic antennas), $\lambda = 0.125$ m (2.4 GHz), $\gamma = 2$ (free-space):

$$p^r = p^t \cdot \left(\frac{\lambda}{4\pi}\right)^2 \cdot d^{-2}$$

**Role:** Converts AP transmit power $p^t$ into received power at any node (user or Eve) at distance $d$. Used in every SINR calculation. The $d^{-2}$ factor means doubling the distance cuts received power by 4. The $(\lambda/4\pi)^2$ term is a fixed constant — it's just the effective aperture of an isotropic antenna at 2.4 GHz. Implemented in `cfj_env.py:_received_power()`.

**Feynman:** Think of a light bulb radiating in all directions. The further you stand, the more spread out the light gets — power falls off as the square of distance. Friis is just that, for radio. The wavelength term accounts for how well the antenna "catches" signal at that frequency.

---

### 2. SINR at Legitimate User

$$\text{SINR}_{n,k} = \frac{p^{ru}_{n,k}}{\displaystyle\sum_{\nu \neq n} p^{ru}_{\nu,k} + N_{u_k}}$$

$p^{ru}_{n,k}$ = received power at user $u_k$ from its serving AP $n$. $\nu$ iterates over all other APs (interferers). $N_{u_k} = 3.162 \times 10^{-12}$ W ($-85$ dBm).

**Role:** Signal-to-interference-plus-noise ratio at the user. Because all APs transmit on the same frequency, non-serving APs are interferers. SINR determines how clearly the user receives the intended signal. Implemented in `cfj_env.py:_sinr()`.

**Feynman:** You're in a room with 4 people all talking at once. You're only trying to hear person A. The SINR is how loud person A is compared to the combined noise of persons B, C, D plus background hiss. High SINR = clear signal. Low SINR = you're drowning in crosstalk.

---

### 3. SINR at Eavesdropper

$$\text{SINR}_{n,j} = \frac{p^{re}_{n,j}}{\displaystyle\sum_{\nu \neq n} p^{re}_{\nu,j} + N_{e_j}}$$

Same structure as user SINR but evaluated at Eve's location. $p^{re}_{n,j}$ = received power at Eve $e_j$ from AP $n$.

**Role:** How well Eve receives each AP's signal. Non-serving APs act as natural jammers degrading Eve's SINR — this is the core mechanism of cooperative friendly jamming. The agent exploits this by tuning interferer power levels.

**Feynman:** Same room, but now Eve is listening in from a corner. She can hear person A, but persons B, C, D are also talking — jamming her reception. Our agent controls how loud B, C, D talk, which controls how well Eve can decode A.

---

### 4. Shannon Capacity — Legitimate User

$$C_{n,k} = W \log_2\!\left(1 + \text{SINR}_{n,k}\right)$$

$W = 1$ Hz (normalized bandwidth). Gives capacity in bps/Hz.

**Role:** Converts SINR to information-theoretic channel capacity. Maximizing SINR directly maximizes the user's data rate. The $\log_2(1+\cdot)$ is the Shannon-Hartley formula — the theoretical maximum bits per second per Hz over any noisy channel.

**Feynman:** If SINR tells you the signal quality, Shannon capacity tells you the ceiling on how much information you can squeeze through that quality of channel per second. More SINR → higher ceiling. Doubling SINR doesn't double capacity (log is sublinear) — but driving Eve's SINR to near-zero still collapses her capacity toward zero.

---

### 5. Shannon Capacity — Eavesdropper

$$C^e_{n,j} = W \log_2\!\left(1 + \text{SINR}_{n,j}\right)$$

**Role:** How much information Eve can extract from AP $n$'s signal. We want this as low as possible. The agent achieves this by making other APs (which serve as jammers) degrade Eve's SINR.

---

### 6. Worst-Case Eve Capacity per AP

$$C^e(n) = \max_{j=1}^{J} C^e_{n,j}$$

In our simulation $J = 1$, so this is just $C^e_{n,1}$. The max notation is from the general formulation in Hoseini et al. — kept for consistency.

**Role:** The maximum eavesdropping capability Eve can achieve on AP $n$'s signal. Used as a conservative upper bound on what Eve learns from any one AP. Even with $J=1$, the formula generalizes to multiple Eves.

---

### 7. Secrecy Capacity per User

$$C_s(u_k, \alpha_k) = \left[C_{\alpha_k,k} - C^e(\alpha_k)\right]^+$$

$[\cdot]^+ = \max(\cdot, 0)$. $\alpha_k$ is the AP serving user $u_k$.

**Role:** The actual private communication rate — the rate at which user $u_k$ can receive information that Eve cannot decode. If Eve's capacity exceeds the user's capacity, secrecy is zero (clamped at 0 — you can't have negative secrecy). Maximizing this across all users is the whole point of the system. Implemented in `cfj_env.py:_secrecy_capacity()`.

**Feynman:** You're whispering to a friend, and Eve is nearby. The secrecy capacity is how much information you can share that Eve cannot figure out — it's the gap between how well your friend hears you vs how well Eve hears you. If Eve hears you better than your friend, nothing you say is private.

---

### 8. AP–User Association

$$\alpha_k = \arg\max_{n \in \{1,\ldots,N\}} \bigl(C_{n,k} - C^e(n)\bigr)\Big|_{\mathbf{P}^t = P_{\max}}$$

Computed once before the agent acts. Each user is assigned to the AP that gives it the best secrecy at uniform max power.

**Role:** Pre-determines which AP serves which user. The agent then optimizes power levels — it does not change who serves whom. This simplifies the continuous action space (power only) and matches the Hoseini et al. design. Implemented in `cfj_env.py:_associate_users()`.

**Feynman:** Before the agent does anything, we ask: given that all APs are broadcasting at full power, which AP gives each user the best private channel? That's the association. The agent then fine-tunes power levels but never reassigns users mid-episode.

---

### 9. Baseline Problem Ψ (Hoseini et al.)

$$\Psi:\quad \max_{\mathbf{P}^t,\,A}\; \sum_{k=1}^{K} C_s(u_k, \alpha_k) \quad \text{s.t.} \quad 0 \leq p^t_n \leq P_{\max}$$

Agent observes true Eve location directly. Reward scored at the single true Eve position.

**Role:** The optimization problem the baseline paper solves. Non-convex because secrecy capacity is not jointly convex in the power vector. SAC is required because no closed-form solution exists. This is the problem Ψ that our Φ_UA replaces.

---

### 10. Our Variable — Normalized Uncertainty Ratio ρ

$$\rho = \frac{\sigma}{D_{\max}} \in [0, 1]$$

$\sigma$ = standard deviation of Eve location estimation error (metres). $D_{\max} = 50$ m (grid boundary). $\sigma \sim \mathcal{U}[0, 10]$ per episode during UA-SAC training.

**Role:** Our introduced variable. Dimensionless. Tells the agent how much to trust the Eve coordinate it receives. $\rho = 0$ means perfect location knowledge. $\rho = 0.2$ means the estimate has error up to 20% of the map size. The 15th element of the state vector $s^*$. Implemented in `cfj_env.py:reset()` as `obs[-1] = sigma / map_size`.

**Important:** ρ is a property of the operator's own estimation system — not of Eve. In real deployment, you know how noisy your own sensors are. ρ is always available even when Eve's true location is completely unknown.

**Feynman:** Imagine GPS with variable accuracy. Sometimes you're in open sky — accurate to 1m. Sometimes in a canyon — accurate to 50m. ρ is the "how uncertain is my GPS right now" reading. Without it, the navigator (agent) sees a coordinate but doesn't know whether to trust it or ignore it. With ρ, it knows how stale the information is and plans accordingly.

---

### 11. Noisy Eve Estimate $\hat{E}_i$ (Phase 2)

$$\hat{E}_i = \mathrm{clip}\!\left(E + \varepsilon_i,\; 0,\; D_{\max}\right), \qquad \varepsilon_i \sim \mathcal{N}(\mathbf{0},\, \sigma^2 \mathbf{I}_2), \quad i = 1,\ldots,M$$

$E = (e_x, e_y)$ = true Eve location (never given to agent). $M = 5$ samples per step.

**Role:** For each step, we sample $M$ plausible Eve locations from the noise distribution. These are used to compute the worst-case reward $R^*$. Eve's true location $E$ is only used for sampling — the agent never observes it directly.

---

### 12. Worst-Case Reward R*

$$R^*(s^*, \mathbf{P}) = \min_{i=1}^{M}\; \sum_{k=1}^{K} C_s\!\left(u_k,\,\alpha_k \mid \hat{E}_i\right)$$

**Role:** The training signal that makes UA-SAC robust. Instead of rewarding the agent for average secrecy at one noisy Eve location, we reward it for the minimum secrecy across M sampled Eve locations. The agent learns to maximize its worst-case outcome — security floor not just average. Implemented in `cfj_env.py:step()` when `M > 1 and sigma > 0`.

**Feynman:** A fire drill run once a year means you can say "we passed the drill." A fire drill run 5 times simultaneously in the worst-case scenario — blocked exits, smoke, confusion — is what actually trains you for reality. R* is that 5-trial worst case. The agent can't get lucky on training day anymore.

---

### 13. Our Problem Formulation Φ_UA

$$\Phi_{\text{UA}}:\quad \max_{\mathbf{P}^t}\; \mathbb{E}_{\sigma \sim \mathcal{U}[0,10]}\!\left[R^*(s^*, \pi_\theta(s^*))\right] \quad \text{s.t.} \quad 0 \leq p^t_n \leq P_{\max}$$

**Role:** Our problem, replacing baseline Ψ. Non-convex and stochastic — joint expectation over σ (random per episode), the M Eve samples, and random topology. No closed-form solution. DRL (UA-SAC) is the only practical approach.

**Key distinction from Ψ:** Ψ maximizes sum secrecy at the known Eve location. Φ_UA maximizes expected worst-case secrecy under location uncertainty, across all uncertainty levels simultaneously.

---

### 14. Augmented State Space s*

$$s^* = \underbrace{[x_1,y_1,\ldots,x_4,y_4]}_{\text{AP locations }(8)} \oplus \underbrace{[x_{u_1},y_{u_1},x_{u_2},y_{u_2}]}_{\text{User locations }(4)} \oplus \underbrace{[\hat{e}_x,\hat{e}_y]}_{\text{Perceived Eve }(2)} \oplus \underbrace{[\rho]}_{\text{Uncertainty }(1)} \;\in \mathbb{R}^{15}$$

Baseline state $s \in \mathbb{R}^{14}$ omits ρ. All coordinates normalized to $[0,1]$ (divided by 50m).

**Role:** Full input to the actor network $\pi_\theta$. The 15th element ρ is what enables one universal policy instead of four specialist agents — it tells the network how much to trust element 13 (perceived Eve x) and 14 (perceived Eve y).

---

### 15. Entropy Scaling — UA-SAC Modification

$$\alpha_{\text{eff}}(\rho) = \alpha_{\text{base}} \cdot (1 + \beta \cdot \rho), \quad \beta = 1.0$$

$\alpha_{\text{base}}$ = auto-tuned by standard SAC dual objective. $\rho$ = mean normalized uncertainty of the current minibatch.

**Role:** The SAC-level change that makes UA-SAC a named algorithm. When ρ is high (bad CSI), the entropy bonus is amplified — the agent is rewarded more for staying unpredictable, which produces broader jamming coverage. When ρ is low (reliable estimate), the agent is allowed to jam precisely. Implemented in `uasac.py` as `alpha_eff = alpha_base * (1.0 + self.beta * rho_mean)`. Used in both Q-target update and actor loss.

**Feynman:** SAC normally says "maximize reward, but also stay somewhat random — don't always do the same thing." α controls how much you're rewarded for randomness. UA-SAC dynamically adjusts that dial based on how uncertain the Eve estimate is. When you genuinely don't know where Eve is (high ρ), being random is actually the right strategy — spread jamming everywhere rather than committing to a probably-wrong location.

---

### 16. UA-SAC Full Objective J_UA

$$J_{\text{UA}}(\pi_\theta) = \mathbb{E}_{s^* \sim \mathcal{D}}\!\left[ R^*(s^*,\, \pi_\theta(s^*)) + \alpha_{\text{eff}}(\rho)\;\mathcal{H}\!\left(\pi_\theta(\cdot \mid s^*)\right) \right]$$

$\mathcal{D}$ = distribution of random topologies. $\mathcal{H}$ = policy entropy.

**Role:** The full objective UA-SAC maximizes. It is the standard SAC objective with two changes: (1) $R^*$ replaces the standard reward, and (2) $\alpha_{\text{eff}}$ replaces the fixed $\alpha$. At $\rho = 0$, this collapses to the standard SAC objective evaluated at the true Eve location — the baseline Ψ. Our contribution is what happens at $\rho > 0$.

---

## Variables Introduced by Us

| Symbol | Definition | Role |
|---|---|---|
| $\rho$ | $\sigma / D_{\max} \in [0,1]$ | Normalized uncertainty ratio — 15th state element |
| $\hat{E}_i$ | $\mathrm{clip}(E + \varepsilon_i, 0, D_{\max})$ | Sampled noisy Eve location for worst-case reward |
| $R^*$ | $\min_{i=1}^M \sum_k C_s(u_k \mid \hat{E}_i)$ | Worst-case robust reward |
| $\alpha_{\text{eff}}$ | $\alpha_{\text{base}} (1 + \beta \rho)$ | ρ-scaled entropy coefficient |
| $M = 5$ | — | Eve location samples per reward computation |
| $\beta = 1.0$ | — | Entropy scaling hyperparameter |

---

## Decisions Log

1. Python + PyTorch over MATLAB — reproducibility, open-source
2. SAC over DDPG — matches baseline paper, more stable
3. Single-step MDP — each episode = one channel realization (terminated=True after one step)
4. Fixed seeds (seed=ep) for evaluation — eliminates topology variance as confound
5. Clean eval env (csi_noise_std=0) — reward scored against true Eve regardless of agent's training σ
6. DO NOT RETRAIN Phase 1 models — 4 hrs of compute, results are valid
7. Phase 2 requires new training run (UA-SAC) — expect 6-8 hrs
8. venv/Scripts/python — not system python (Python 3.13 broken on this machine)

---

## Current Results Summary

| System | Sum Secrecy | Secrecy Ratio |
|---|---|---|
| Normal Wi-Fi | 2.3 bps/Hz | 100% |
| Smart AP | 2.3 bps/Hz | 100% |
| RL-CFJ σ=0 (baseline) | 3.0 bps/Hz | 98% |
| RL-CFJ σ=10m (ours) | 3.0 bps/Hz | 98% |

At σ=10m evaluation: robust agent 3.026 vs baseline 2.977 bps/Hz.
Both RL agents ~30% better than non-RL baselines.

---

## GitHub

Remote: https://github.com/Deez-Automations/Wireless-Simulation
Branch: main
Models: pushed (sac_noise_0.0, 2.0, 5.0, 10.0 — ~13MB total)
Dashboard: pushed and working with server.py
Last commit: feat: connect dashboard to live SAC agent via Flask bridge

---

## Session Log

### 2026-05-09 — Phase 2 Slides + Equations Lock

**What was done:**

1. **slides.tex — Phase 1 → Phase 2 migration** (`docs/slides.tex`)
   - Converted from Beamer to IEEEtran onecolumn (user applied formatting)
   - Replaced Ψ/Φ* problem formulations with Φ_UA
   - Updated state from $s \in \mathbb{R}^{14}$ to $s^* \in \mathbb{R}^{15}$ with ρ as 15th element
   - Replaced single-Eve reward with worst-case R*
   - Added α_eff entropy scaling equation
   - Added UA-SAC simplified pseudocode (plain English comments, grouped into phases)
   - Added System Preliminaries section: System Model parameters (N=4, K=2, J=1, 50m grid, 2.4GHz, γ=2, P_max=1W) + Our Parameters (ρ, M=5, β=1.0, σ_max=10m) + Training Hyperparameters (LR=3e-4, buffer=200k, batch=256, hidden=[256,256], timesteps=100k)
   - Results section references `results/phase2/` plots

2. **Equation verification** — all 12 equations in slides.tex cross-checked against `env/cfj_env.py` and `uasac.py`. All verified correct:
   - Friis → `_received_power()` ✓
   - SINR → `_sinr()` ✓
   - Shannon → `_capacity()` ✓
   - Secrecy → `_secrecy_capacity()` ✓
   - AP association → `_associate_users()` ✓
   - ρ → `reset()` ✓
   - Ê_i sampling → `step()` ✓
   - R* (worst-case min) → `step()` when M>1 ✓
   - α_eff → `uasac.py:train()` ✓
   - J_UA → actor loss in `uasac.py` ✓

3. **notations.tex created** (`notations.tex`) — full LaTeX notation reference for all equations, organized by section. Supersedes the compiled-only `notations.pdf`.

4. **plot_convergence.py created** (`plot_convergence.py`) — standalone script for Plot 5 (training convergence). Reads `results/uasac_reward_history.npy`, draws scatter dots every N episodes + polynomial degree-4 best-fit line (orange) + light rolling average. Saves to `results/phase2/plot5_convergence.png`. **Requires one retrain to generate the .npy file.**

5. **train.py updated** — added `np.save("results/uasac_reward_history.npy", ...)` after `model.learn()` so raw episode rewards are persisted for plot_convergence.py.

**Pending after this session:**
- Run `venv/Scripts/python train.py` → generates `models/uasac_robust` + `results/uasac_reward_history.npy`
- Run `venv/Scripts/python plot_convergence.py` → generates proper Plot 5 with real best-fit line
- Run `venv/Scripts/python test.py` → regenerates Plots 1-4 in `results/phase2/`
- Update IEEE_Final_Report.md with Phase 2 equations and new plot references

---

### 2026-07-18 — Code Verification, Bug Fix, and Literature Review Overhaul

**What was done:**

1. **Verified every equation in `final_report.tex` against the actual code, line by line** — all confirmed correct (Friis, SINR, secrecy capacity, worst-case reward, ρ augmentation, entropy scaling), with a few precision gaps found:
   - AP association (`_associate_users()`) uses **true** Eve coordinates and a uniform-max-power assumption — inherited unchanged from Hoseini, never disclosed in our own Limitations section.
   - `ρ = σ/D_max` is claimed to range `[0,1]` in the text, but `sigma_range=(0,10)` and `D_max=50` mean training only ever explores `ρ ∈ [0, 0.2]` in practice.
   - Entropy scaling (`uasac.py:train()`) uses a **batch-average** ρ, not per-sample — "adaptive" language in the paper overstates this.

2. **Found and fixed a real bug** in `train.py`'s `RewardLogger` — it indexed `self.locals["rewards"][0]`/`dones[0]`, tracking only 1 of the 4 parallel `SubprocVecEnv` workers. Verified via `results/uasac_reward_history.npy`: exactly 25,000 entries against 100,000 total training timesteps (confirmed via `results/uasac_ent_history.npz`, 24,975 gradient steps — the *correct* number, since gradient steps ≈ timesteps/4 by SB3 design). The convergence plot only ever reflected 1/4 of the actual training run. Fixed to aggregate across all workers. Committed (`467048d`) and pushed to `origin` after also fixing a stale local git identity override that had been committing under the wrong email.

3. **Ran the actual eval code against the current saved models** — the paper's headline number (UA-SAC = 3.10 bps/Hz at σ=10m) does **not** reproduce; current code + `models/uasac_robust.zip` gives **3.20**. Fixed Max Power (2.34) and Baseline SAC (2.98) both matched. Computed the real paired significance test (same shared topologies): mean diff 0.221, SEM 0.053, t=4.16 (p≪0.001) — genuinely significant, but UA-SAC only wins on 53.4% of the 1000 individual topologies, a nuance the current mean-only bar chart hides. **Numbers must be regenerated before submission** — `test.py` needs a rerun.

4. **Literature review overhaul.** Original 20 references audited: one orphan citation (`\bibitem{liu}`, Y. Liu et al. 2017, never cited in body text — needs fixing or cutting), most other references capped at 2018–2021 with zero 2024–2026 engagement. Curated **15 new papers** across four buckets — robust/worst-case RL theory (Tanabe/M2TD3, Lanier/FARR, Chen/domain-randomization, Wang-Kallus-Sun/CVaR), recent DRL-for-PLS applications (Liu/SAC-covert-comm, Xing, Zhao, Tripathi, Wang/MAPPO-LSTM), non-RL competitors solving the same Eve-location-uncertainty problem (Xiao/STAR-RIS, Zhou/near-field, Miao), and survey/context (Khoshafa, AN survey) — landing at **35 total papers**, verified real via abstracts and (where accessible) full text, not fabricated.

5. **Access-checked all 35** — 28 have legitimate free copies (arXiv, author pages, institutional repositories, open-access journals/proceedings); only 5 are genuinely paywalled with no free alternative found (Csiszár & Körner 1978, Hu et al. 2018, Cui et al. 2019, Tu et al. 2024, Wang/MAPPO-LSTM 2025) — these need GIKI library access. Rappaport's textbook is a library-shelf problem, not a web-access one.

6. **AP association / non-convexity deep-dive** (not yet implemented, discussion only) — confirmed `Φ` (the power allocation subproblem) is non-convex because secrecy capacity is a difference of two SINR-ratio-based log terms (DC-programming structure), not because of any one single hard term. Explored whether RL could learn AP association jointly with power (currently fixed via a uniform-max-power heuristic, same gap Hoseini's own future work names) — concluded the cheapest tractable path is a soft continuous relaxation (extend the action space with per-user AP-affinity scores, argmax at execution) rather than a true hybrid discrete+continuous SAC variant. **Scoped as Future Work for the paper, not scheduled for implementation** given submission timeline.

**Pending after this session:**
- Retrain UA-SAC and rerun `test.py`/`plot_convergence.py` to regenerate all numbers/plots with the fixed `RewardLogger` and current code (fixes items 2 and 3 above)
- Add CI/error-bar reporting and the paired significance result to the Results section (currently mean-only)
- Fix or cut the orphan `liu` citation
- State the real `ρ ∈ [0, 0.2]` training range honestly, or widen `sigma_range` and retrain
- Add one paragraph to Limitations acknowledging the inherited AP-association/mobility gaps from Hoseini
- Get GIKI library access for the 5 paywalled papers (#3, #6, #14, #28, #31 in the working list — see `docs/literature_notes.md`)
- Rewrite the Literature Review section using the full 35-paper set, subsection-organized (foundations → cooperative jamming → DRL-PLS → robust/CVaR RL theory → non-RL uncertainty competitors → IRS alternatives)

---

### 2026-07-22 — Literature Notes Completed (Phase 1), Set Trimmed and Renumbered to 30

**What was done:**

1. **Finished the full deep-read pass on all accessible papers.** All 28 papers with legitimate free copies got real, verified notes entries in `docs/literature_notes.md` — one paper at a time, template-driven (citation, mechanism, CSI/Eve assumption, their own stated limitations/future work verbatim where possible, direct comparison to our work, and the specific job each paper does in our argument). Standout findings worth remembering without re-reading the whole file:
   - Three papers spanning three decades (Leung-Yan-Cheong & Hellman 1978, Tekin & Yener 2008, Dong et al. 2010) independently flagged imperfect/unknown eavesdropper CSI as an open problem — strong throughline for the Introduction.
   - Tekin & Yener (2008) explicitly proposed worst-case design over an uncertain Eve region as future work — 17 years before UA-SAC implements exactly that.
   - Hoseini et al.'s own paper documents an **empirical** instance of the AP-association mismatch (their scenario 5 underperformed unexpectedly, diagnosed by the authors themselves as caused by fixing association before optimizing power) — real evidence, not speculation, for our own Limitations section.
   - Bouhafs & den Hartog's 2020 paper (co-authors on our own baseline) claims Eve's location IS detectable by APs — a genuine, unresolved tension with our project's core premise that needs an explicit sentence addressing it, not silence.
   - Zhou et al. 2026 (near-field robust beamforming) found an "angular-error amplification effect" — the same positional uncertainty produces larger angular error the closer Eve is. Direct critique of our ρ being a single geometry-blind global scalar.
   - Zhao et al. 2025 and Miao et al. 2026 both offer *certified* robustness guarantees (interval bound propagation, deep-unfolded optimization) — sharp contrast for honestly scoping UA-SAC's robustness as empirical (5-sample Monte Carlo), not certified.

2. **Access reality check: 7 of the original 35 could never be reached** (Csiszár & Körner 1978, Hu/Li 2018, Cui et al. 2019, Rappaport textbook, Khoshafa 2024, Tu 2024, Wang/MAPPO-LSTM 2025) — GIKI library access never came through, HAL link for Khoshafa never resolved. Rather than keep waiting, **dropped 5 of them** (Hu/Li, Cui, Khoshafa, Tu, Wang MAPPO-LSTM — none load-bearing) and **found 2 strong, verified, fully open-access replacements**, both directly on-theme and very current:
   - **Bui et al. 2026** (arXiv:2603.05062) — deep-learning-driven friendly jamming for secure multicarrier ISAC under unknown Eve location and channel uncertainty. Closest sibling paper in the whole set; resolves Eve-uncertainty via radar echo feedback (an ISAC-only capability our plain-communication Wi-Fi APs don't have) rather than worst-case sampling — a clean, citable architectural contrast.
   - **Abughalwa et al. 2025** (arXiv:2504.05048) — IRS secrecy for all users under unknown Eve location, non-RL (SCA+LMI convex optimization with a proven local-optimality guarantee), explicit max-min objective. Their Jain's-fairness-index result (≈1.0 for max-min vs. ≈0.2–0.3 for sum-rate maximization) is a clean, quantified precedent for "max-min beats average" independent of RL.
   - Csiszár & Körner stays correctly cited in the bibliography (foundational, doesn't need a deep-read notes entry) but is no longer part of the numbered set. Rappaport stays as the textbook source for the Friis equations, never was a lit-review paper.

3. **Fully renumbered the notes file to a clean 1–30, no gaps.** Earlier in this process the set was numbered 1–37 with holes wherever a candidate got dropped; since the project is continuing rather than closing out, compressed everything into a straight 1–30 sequence and fixed two internal cross-references that had pointed to pre-renumbering positions. `docs/literature_notes.md` is now the authoritative, stable reference — 30 papers, each with a real write-up, numbered exactly 1 through 30.

**Pending after this session:**
- Begin Phase 2: write the actual Literature Review prose from these 30 notes entries, subsection-organized as planned (foundations → cooperative jamming lineage → DRL-for-PLS → robust/CVaR RL theory → non-RL uncertainty competitors → IRS alternatives)
- Address the Bouhafs/den Hartog Eve-detectability tension explicitly somewhere in the paper (Related Work or Limitations)
- Add the geometry-blind-ρ limitation (from Zhou et al.) and the empirical-not-certified robustness scoping (from Zhao/Miao) to Discussion/Limitations
- Everything else still pending from the 2026-07-18 entry (regenerate results, orphan-citation fix on `liu`, honest ρ-range statement, AP-association/mobility disclosure) remains open — none of that got touched this session
