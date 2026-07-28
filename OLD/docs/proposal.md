# Project Proposal
## CY315 — Wireless and Mobile Security | Spring 2026
### Ghulam Ishaq Khan Institute of Engineering Sciences and Technology

---

**Title:** Cooperative Friendly Jamming for Physical Layer Security Under Unknown Eavesdropper Locations Using Deep Reinforcement Learning

**Track:** Track 2 — Implementation and Optimization

**Team Members:**
- M. Daniyal (2023406) — SAC Agent & Simulation Lead
- M. Afeef Bari (2023356) — Results, Analysis, Report
- Mahad Aqeel (2023286) — Channel Simulation Environment

**Course Instructor:** [Instructor Name]

**Date:** April 2026

---

## 1. Problem Statement

Wireless signals don't stay between sender and receiver. They spread out. Anyone nearby — with the right hardware and some patience — can silently capture and decode them. This is the eavesdropping problem. It exists at the physical layer, before any encryption kicks in.

Physical Layer Security (PLS) addresses this at the source. The principle: if the legitimate receiver has a stronger channel than the eavesdropper, you can transmit in a way that is information-theoretically secure — meaning even infinite computing power cannot break it. No keys, no certificates. Just physics.

The core metric is secrecy capacity:

    C_s = max{ log2(1 + SNR_Bob) - log2(1 + SNR_Eve) , 0 }

Positive secrecy capacity means secure transmission is possible. The challenge is engineering the environment to guarantee this condition holds — especially against eavesdroppers whose locations you don't know.

One practical approach is cooperative jamming. In a dense Wi-Fi deployment, not every access point is serving a user at every moment. Idle APs are wasted infrastructure. But if they transmit jamming noise on the same frequency band, they actively degrade the eavesdropper's signal without affecting the intended receiver (who is associated with a different AP and can subtract known interference). The result is a built-in, infrastructure-level security layer.

Hoseini et al. (2024) formalized this. Their system has N Wi-Fi APs in a shared area. Each AP is either serving a user or acting as a jammer — continuously. A deep reinforcement learning agent (SAC) learns the optimal transmit power for each AP to maximize the total secrecy capacity across all legitimate users.

It works. But there is one assumption baked into their model that does not hold in the real world: the RL agent's state vector includes the eavesdropper locations. Passive eavesdroppers never transmit. They have no protocol presence. You cannot estimate their channel or locate them. A security system that requires knowing where the attacker is standing is not a deployable security system.

That is the gap this project addresses.

---

## 2. Proposed Contribution

We implement the Hoseini et al. (2024) system as our baseline — port it from MATLAB to Python, replicate their results, then modify it to remove the eavesdropper location assumption.

In the original system, the SAC agent's state is:

    s = { L_AP, L_u, L_e }

Where L_e is the set of eavesdropper locations. We strip L_e from the state entirely. The agent now operates with only:

    s = { L_AP, L_u }

To replace the lost information, we redesign the reward function. Instead of computing exact secrecy capacity using true Eve locations, we sample M random eavesdropper location realizations across the coverage area and compute the minimum secrecy capacity across all of them:

    r_wc = min_{m ∈ {1,...,M}} Σ_k C_s(u_k, α_k | Eve_m)

This is a worst-case reward. The agent learns power configurations that maintain positive secrecy even when Eve occupies the worst possible position — not just when she happens to be in a weak spot. That is the difference between a system that looks secure and one that actually is.

Contribution summary: Python port of MATLAB baseline + worst-case reward under unknown eavesdropper locations.

---

## 3. System Model

The simulation environment:

- **N Access Points** — scattered randomly in a 50m × 50m area
- **K Legitimate Users** — each associated with exactly one AP
- **J Eavesdroppers** — passive, unknown location (in our system)
- **Frequency:** 2.4 GHz Wi-Fi, all nodes share the same band
- **Path Loss:** Friis model, exponent γ = 2
- **Noise floor:** −85 dBm at all receivers
- **Max transmit power:** 1 Watt per AP

Every AP transmits at all times. If it has an associated user, it sends data. If not, it sends jamming noise. Because every AP's signal arrives at every receiver — including eavesdroppers — the jamming is inherently cooperative. Eavesdroppers see a wall of interference.

The SAC agent outputs a continuous power vector P^t = [p1, p2, ..., pN] at each time step. The environment returns the reward (sum secrecy capacity), the state updates with new random node placements, and training continues.

Channel capacity follows standard Shannon formula. Secrecy capacity per user:

    C_s(u_k, α_k) = [ C_{αk,k} − max_j(C^e_{αk,j}) ]^+

Where α_k is the AP associated with user k, and the max is over all eavesdroppers on that AP's signal.

In our modified system, the max is over all eavesdroppers in all M sampled realizations — the agent optimizes for the worst-case distribution, not one specific Eve placement.

---

## 4. Baseline Reference

S.A. Hoseini, F. Bouhafs, N. Aboutorab, P. Sadeghi, and F. den Hartog, "Cooperative Jamming for Physical Layer Security Enhancement Using Deep Reinforcement Learning," arXiv:2403.10342v1, March 2024.

We adopt their system model, AP-user association scheme, channel setup, and SAC-based MDP formulation. Our contribution is the modification that removes the eavesdropper location requirement from the state and replaces the reward function with a worst-case formulation.

---

## 5. Methodology

**Phase 1 — Port Baseline to Python**

Reimplement the Hoseini et al. MATLAB simulation in Python using PyTorch and Gymnasium. Validate that our implementation reproduces their results: secrecy ratio vs. number of APs across their 6 test scenarios (4 to 13 APs). This is a standalone engineering contribution.

**Phase 2 — Remove Eve Locations from State**

Strip L_e from the state vector. Keep everything else identical. Implement the worst-case reward: sample M Eve location realizations uniformly across the 50m × 50m area each step, compute secrecy capacity for each, reward with the minimum. Retrain the SAC agent.

**Phase 3 — Comparison and Evaluation**

Run both systems under identical conditions. Generate comparative plots:
- Secrecy ratio vs. number of APs (our 6-scenario replication + our modified system)
- Secrecy ratio vs. number of sampled Eve realizations M (ablation on M)
- Training convergence curves (baseline vs. modified)
- Secrecy ratio vs. eavesdropper density

---

## 6. Tools

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| DRL Framework | PyTorch |
| RL Environment | Gymnasium |
| Simulation | NumPy, SciPy |
| Plots | Matplotlib |
| Report | LaTeX — IEEE format (Overleaf) |

---

## 7. Expected Outcome

The baseline replication will validate our Python port matches Hoseini et al.'s results. Then the comparison tells a clear story: the original system, when Eve's location is hidden from it at test time, degrades significantly. Our system — trained without Eve's location from the start — maintains reasonable secrecy performance because the agent learned to be robust rather than just reactive.

The delta we are showing is not raw performance. It is robustness under realistic deployment conditions. A system that works when you feed it the attacker's GPS coordinates is not a real-world security system. Ours is.

---

## 8. References

[1] S.A. Hoseini, F. Bouhafs, N. Aboutorab, P. Sadeghi, and F. den Hartog, "Cooperative Jamming for Physical Layer Security Enhancement Using Deep Reinforcement Learning," arXiv:2403.10342v1, 2024.

[2] H. Yang, Z. Xiong, J. Zhao, D. Niyato, L. Xiao, and Q. Wu, "Deep Reinforcement Learning-Based Intelligent Reflecting Surface for Secure Wireless Communications," IEEE Trans. Wireless Commun., vol. 20, no. 1, pp. 375–388, Jan. 2021.

[3] M. Cui, G. Zhang, and R. Zhang, "Secure Wireless Communication via Intelligent Reflecting Surface," IEEE Wireless Commun. Lett., vol. 8, no. 5, pp. 1410–1414, Oct. 2019.

[4] Y. Zhang et al., "Deep Reinforcement Learning for Secrecy Energy Efficiency Maximization in RIS-Assisted Networks," IEEE Trans. Veh. Technol., vol. 72, no. 9, pp. 12413–12418, 2023.

[5] [RIS-Assisted Physical Layer Security Survey 2024 — full citation to be added once paper is downloaded]
