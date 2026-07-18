# Literature Review — Working Notes

Raw technical notes, one paper at a time. Not polished prose — this is the internal
"concept clearance" pass before we write the actual Literature Review section.
Each entry follows the same template so nothing gets a lazier treatment than
the rest, including the papers already in the bibliography.

Read → write the entry → move to the next. Not batched.

## Template per paper

```
## [N]. Author et al., Year — Short Title

**Citation:** full IEEE-ready citation
**Link:** url
**Source read:** full PDF / abstract-only (flagged honestly, never guessed)

**What it does:** 2-3 sentences, plain language.
**Core mechanism:** the actual technical method — equations/algorithm, not just a label.
**Eve/CSI assumption:** what they assume is known vs. unknown about the eavesdropper.
**Their stated limitations:** pulled from their own text, not my inference.
**Their stated future work:** verbatim/paraphrased, if the paper has one.
**Vs. our work:** direct comparison — same problem/different mechanism? Different
  problem/same mechanism? What would change in UA-SAC if we adopted their approach?
**Why it's in our lit review:** its specific job in our argument (competitor,
  theory grounding, recency signal, gap-filler, algorithm-choice justification, etc.)
```

---

## Index — status of all 35

PDF locations: existing 20's downloads are in `reference papers/`, except Hoseini
which is in `Baseline_Paper/` (has the actual formal IEEE GC Wkshps version, not
just arXiv — use `Hoseini Baseline IEEE Formal.pdf`). All files renamed to
`Author Keyword Keyword.pdf` for identifiability. `Kaur RIS Survey NOTINLIST.pdf`
is NOT #23 — a different survey entirely (Kaur et al.), verified by content, not
one of our 35 — keep it out of the reading queue.

| # | Paper | Status |
|---|---|---|
| 1 | Wyner (1975), wiretap channel | **Done** |
| 2 | Leung-Yan-Cheong & Hellman (1978), Gaussian wiretap | **Done** |
| 3 | Csiszár & Körner (1978), broadcast channels | Missing — paywalled |
| 4 | Tekin & Yener (2008) | **Done** |
| 5 | Dong et al. (2010), cooperating relays | **Done** |
| 6 | Hu/Li et al. (2018), IoT cooperative jamming | Missing — paywalled |
| 7 | Luong et al. (2019), DRL survey | **Done** |
| 8 | Nasir & Guo (2019) | **Done** |
| 9 | Li et al. (2018), network slicing | **Done** |
| 10 | Haarnoja et al. (2018), SAC (ICML) | **Done** |
| 11 | Haarnoja et al. (2018/19), SAC algorithms & applications | **Done** |
| 12 | Mukherjee & Swindlehurst (2011) | **Done** |
| 13 | Wang et al. (2014), outage-constrained MISO | **Done — flag: not actually a PLS paper, see notes** |
| 14 | Cui, Zhang & Zhang (2019), IRS secure comm | Missing — paywalled |
| 15 | Yang et al. (2021), DRL-based IRS | **Done** |
| 16 | Bouhafs et al. (2018), Wi-5 | **Done** |
| 17 | Bouhafs et al. (2020), spectrum programmability PLS | **Done — flags a real assumption conflict with our own baseline's co-authors, needs addressing** |
| 18 | Hoseini et al. (2023/2024), baseline | **Done — found empirical evidence (scenario 5) of the AP-association mismatch problem in their own results** |
| 19 | Rappaport (2002), textbook | Missing — needs library copy |
| 20 | Liu et al. (2017), PLS survey (orphan — fix or cut) | **Done — real fix found: cite for compound wiretap channels / statistical CSI, not generically** |
| 21 | Zhao et al. (2025) | **Done** |
| 22 | Liu et al. (2024), SAC covert comm | **Done** |
| 23 | Khoshafa et al. (2024) | Missing — HAL link loading slow, retry |
| 24 | Tanabe et al. (M2TD3), 2022 | **Done** |
| 25 | Lanier et al. (FARR), 2022 | **Done** |
| 26 | Chen et al. (domain randomization), 2022 | **Done** |
| 27 | Wang, Kallus, Sun (CVaR-RL), 2023 | **Done** |
| 28 | Tu et al. (2024) | Missing — paywalled |
| 29 | Xing et al. (2024) | **Done — real evidence joint association+power learning is tractable, relevant to our own limitation** |
| 30 | Tripathi et al. (2024) | **Done** |
| 31 | Wang et al. (MAPPO-LSTM), 2025 | Missing — paywalled |
| 32 | Miao et al. (2026) | **Done** |
| 33 | Zhou et al. (2026) | **Done** |
| 34 | Xiao et al. (STAR-RIS), 2025 | **Done** |
| 35 | AN survey (2025) | **Done** |

---

## Entries

## 1. Wyner, 1975 — The Wire-Tap Channel

**Citation:** A. D. Wyner, "The Wire-Tap Channel," *The Bell System Technical Journal*, vol. 54, no. 8, pp. 1355–1387, Oct. 1975.
**Link:** https://archive.org/details/bstj54-8-1355
**Source read:** Full PDF — image scan, no OCR text layer, read visually (pages 1–7 and 33–34 of 34; skipped the dense converse/direct proof sections in the middle since they don't change the conceptual takeaway for our purposes).

**What it does:** Establishes the foundational mathematical model for physical layer security. A legitimate receiver gets data through a "main channel"; an eavesdropper (the wire-tapper) intercepts a degraded version of the same signal through a separate "wire-tap channel." Proves it's possible to transmit information at a positive rate while keeping the eavesdropper's understanding of the message near zero — no encryption, purely by exploiting the difference in channel quality.

**Core mechanism:** Defines "equivocation" Δ = (1/K)H(S^K|Z^N) — how confused the eavesdropper is (higher = more secure). Characterizes the full rate/equivocation tradeoff as an achievable region in the (R,d) plane (Theorem 2), built around the key quantity Γ(R) = sup [I(X;Y) − I(X;Z)] over input distributions — the max achievable gap between how much information reaches the legitimate receiver vs. the eavesdropper. Proves (Theorem 3) a strictly positive "secrecy capacity" C_s exists, at which reliable transmission is possible in near-perfect secrecy.

**Eve/CSI assumption:** Eve's channel (Q_W) is fully known to the system designer — the entire result depends on exploiting a *known* degradation in Eve's channel relative to the main channel. No notion of Eve's location or imperfect CSI at all — this is 1975 abstract information theory (discrete memoryless channels), predates the wireless-specific "where is Eve" framing entirely.

**Their stated limitations:** None explicitly flagged — this is foundational theory, not an applied systems paper. Implicitly restricted to discrete memoryless channels, and the intuitive picture throughout assumes the main channel statistically dominates the wire-tap channel.

**Their stated future work:** None — no dedicated section. Closes with the completed proof and a 5-item reference list, notably citing Shannon's 1949 "Communication Theory of Secrecy Systems" as the direct precursor to this whole line of work.

**Vs. our work:** The ultimate theoretical ancestor of every secrecy equation in our own paper. Our secrecy capacity formula, `Cs(u_k,αk) = [C_{αk,k} − C^e(αk)]^+`, is a direct applied descendant of Wyner's `Γ(R) = sup[I(X;Y) − I(X;Z)]` — same core idea (gap between legitimate and eavesdropper mutual information), specialized from abstract DMCs down to a concrete Friis-path-loss SINR/Shannon-capacity setting, with power control as the mechanism instead of code design. Wyner assumes Eve's channel statistics are fully known; our entire contribution exists because that assumption breaks for real passive eavesdroppers.

**Why it's in our lit review:** The canonical origin citation for the whole field. Every PLS paper opens by citing Wyner 1975 as the reason secrecy-without-encryption is theoretically possible at all. Not a competitor or technical comparison point — it's the foundation the whole argument stands on. Belongs in the opening sentence of the Literature Review, not mid-paragraph.

## 2. Leung-Yan-Cheong & Hellman, 1978 — The Gaussian Wire-Tap Channel

**Citation:** S. K. Leung-Yan-Cheong and M. E. Hellman, "The Gaussian Wire-Tap Channel," *IEEE Trans. Inform. Theory*, vol. IT-24, no. 4, pp. 451–456, July 1978.
**Link:** https://www-ee.stanford.edu/~hellman/publications/29.pdf
**Source read:** Full PDF, real text layer (pdftotext, no image rendering needed). Note: the PDF's first extracted lines are leftover reference-list text from a *preceding* article in the same journal issue that happens to end on the same page (451) — the actual paper title starts partway down page 1. Confirmed correct paper by title, authors, abstract, and volume/issue/page match.

**What it does:** Extends Wyner's 1975 discrete-memoryless-channel result to the **Gaussian** wire-tap channel — the case that actually matters for real (AWGN-based) communication systems. Shows the secrecy capacity has an exceptionally clean closed form: `Cs = CM − CMW`, literally the *difference* between the main channel's capacity and the wire-tap channel's capacity. Also proves `Rd = Cs` is the upper boundary of the whole achievable rate-equivocation region (Theorem 1).

**Core mechanism:** Same wire-tap channel structure as Wyner (Fig. 2: source → encoder → main AWGN channel with noise σ₁² → legitimate receiver; the main channel output is also fed through a second AWGN stage with noise σ₂² → wire-tapper), but now with explicit power-limited Gaussian statistics instead of abstract DMCs. Defines `CM = ½log(1+P/σ₁²)` and `CMW = ½log(1+P/(σ₁²+σ₂²))`, and proves `Cs = CM − CMW` via a sequence of lemmas built on Wyner's framework plus entropy-power arguments (Blachman, Bergmans).

**Eve/CSI assumption:** Same as Wyner — Eve's channel noise variance σ₂² is treated as fully known to the system designer. But the Discussion section (Section IV) explicitly raises what happens when this assumption breaks: *"There is a potential problem if the SNR's on the channels are somewhat uncertain... if the wire-tap channel's SNR is only several dB below the main channel's nominal SNR, then secrecy is lost."* This is a genuinely surprising find — as early as 1978, the authors flagged CSI/SNR uncertainty as a real practical vulnerability, decades before "imperfect eavesdropper CSI" became its own subfield.

**Their stated limitations:** Explicitly named in Discussion: results are "of most use on power limited channels" (bandwidth-limited channels see diminishing returns), and the uncertain-SNR problem above is acknowledged as unresolved by this paper.

**Their stated future work:** Verbatim: *"We are currently investigating the wire-tap channel with feedback and have shown that even when the main channel is inferior to the wire-tapper's channel it is possible to transmit reliably and securely [their ref 10]. This would obviously eliminate many of the above problems."* (i.e., feedback-based secrecy, referencing Leung-Yan-Cheong's own PhD thesis.)

**Vs. our work:** This is the direct algebraic ancestor of the simple "secrecy = legit capacity minus eavesdropper capacity" formula our whole system is built on — our `Cs(u_k,αk) = [C_{αk,k} − C^e(αk)]^+` is structurally identical to their `Cs = CM − CMW`, just applied per-user in a multi-AP SINR setting instead of a single Gaussian channel pair. More importantly: their own 1978 discussion of SNR-uncertainty-breaks-secrecy is, in plain language, *exactly* the problem UA-SAC exists to solve — they identified the vulnerability and left it as an open problem citing feedback as a possible (different) fix; we're addressing the same class of vulnerability with worst-case-robust RL instead, 47 years later, in a cooperative-jamming multi-AP setting they never considered.

**Why it's in our lit review:** Two jobs at once — (1) it's the direct source of our secrecy capacity formula's exact algebraic shape (more precisely than Wyner's abstract mutual-information version), and (2) it's a striking, citable "even the foundational papers knew CSI uncertainty was a problem" moment that strengthens our motivation section — this isn't a new concern we invented, it's a 47-year-old open question we're finally giving a concrete answer to in a modern DRL setting.

## 4. Tekin & Yener, 2008 — The Gaussian Multiple Access Wire-Tap Channel

**Citation:** E. Tekin and A. Yener, "The Gaussian Multiple Access Wire-Tap Channel," *IEEE Trans. Inform. Theory*, vol. 54, no. 12, pp. 5747–5755, Dec. 2008.
**Link:** http://wcan.ee.psu.edu/papers/TY_IT2008Dec.pdf
**Source read:** Full PDF, real text layer.

**What it does:** Extends the wire-tap channel from a single transmitter to **K simultaneous transmitters** sharing one Gaussian multiple-access channel (GMAC-WT) — multiple users, one legitimate receiver, one eavesdropper who gets a degraded version of everyone's combined signal. Defines two flavors of secrecy (individual — each user's message protected even if others are compromised; collective — protect a group as a whole), finds achievable secrecy rate regions for both via superposition coding and TDMA, and derives the secrecy sum-capacity.

**Core mechanism:** System model: `Y = Σhk·Xk + NM` at the legitimate receiver, `Z = hY + NMW` at the eavesdropper (a further-degraded, noisier version of what the receiver got). The key mechanism their "main contribution" section names explicitly: **the undecoded messages of other users act as additional noise at the eavesdropper**, which is precisely the multi-transmitter analog of a cooperative-jamming effect — one user's transmission incidentally jams the eavesdropper's ability to decode another user's message, purely as a side effect of the multi-access structure, no dedicated jamming signal involved.

**Eve/CSI assumption:** Explicitly perfect — "the eavesdropper is intelligent and informed, i.e., it has the same decoding capability and has access to the same information as the legitimate receiver, including all channel parameters." Legitimate parties also assume Eve's channel gains `hkW` are fully known.

**Their stated limitations:** Named directly in the Conclusion: (1) results assume the eavesdropper's signal is a *stochastically degraded* version of the receiver's — the general (non-degraded, heterogeneous per-user channel gains) case is left as future work; (2) secrecy constraints assumed identical across all users — heterogeneous per-user secrecy requirements not explored; (3) most important for us — **"We assume in this correspondence that the eavesdropper's channel gains are known to the legitimate parties. These channel gains may not be easy to obtain in practice."**

**Their stated future work — verbatim, and this is the standout finding:** *"If the eavesdropper is known to be outside a certain area, we might opt to have a **worst case system design, considering the boundary of the "secure" area.**"* Written in 2008. They name the exact idea — worst-case design over an uncertain Eve-location region — that our worst-case reward (`R* = min over M sampled Eve locations`) is a concrete, DRL-based realization of, seventeen years later.

**Vs. our work:** Two distinct connections. (1) Mechanistic: their "other users' undecoded signals jam the eavesdropper" effect is the multi-access-channel cousin of our cooperative friendly jamming — same underlying idea (interference-from-others degrades Eve), different setting (information-theoretic coding vs. physical-layer power control). (2) Conceptual, and stronger: they explicitly proposed worst-case design over an uncertain Eve region as future work and never built it. UA-SAC's `min` over M sampled Eve locations *is* that worst-case design, just realized via reinforcement learning instead of a closed-form information-theoretic bound.

**Why it's in our lit review:** One of the strongest citations in the whole set. Most papers we cite are either mechanistically similar or conceptually similar to our contribution — this one is both, and it predicted the exact shape of our solution as an open problem in 2008. Worth a dedicated sentence in the Introduction or Related Work, not just a list-citation: "worst-case design over an uncertain eavesdropper region was proposed as future work as early as [Tekin & Yener, 2008] and, to our knowledge, remains unrealized in a learning-based cooperative-jamming setting until this work."

## 5. Dong, Han, Petropulu & Poor, 2010 — Improving Wireless Physical Layer Security via Cooperating Relays

**Citation:** L. Dong, Z. Han, A. P. Petropulu, and H. V. Poor, "Improving Wireless Physical Layer Security via Cooperating Relays," *IEEE Trans. Signal Processing*, vol. 58, no. 3, pp. 1875–1888, Mar. 2010.
**Link:** https://www.researchgate.net/publication/220322052
**Source read:** Full PDF, real text layer.

**What it does:** Uses dedicated helper relay nodes to improve secrecy for a single source-destination pair against one or more eavesdroppers, comparing three cooperative schemes: Decode-and-Forward (DF), Amplify-and-Forward (AF), and **Cooperative Jamming (CJ)** — one of the earliest formal uses of that exact term. Designs relay weights and power allocation to either maximize secrecy rate under a power budget, or minimize power under a secrecy-rate constraint.

**Core mechanism:** In the CJ scheme specifically: while the source transmits its encoded signal, the relays simultaneously transmit a weighted common jamming signal `V`, independent of the source message, purely to confound the eavesdropper(s). Relay weights and power are optimized (closed-form for some cases, iterative for others) assuming **global CSI is available** — the whole optimization is a function of known channel gains everywhere.

**Eve/CSI assumption:** Full/global CSI assumed throughout the main results — relay weight and power designs are derived as closed-form functions of the (known) source-eavesdropper and relay-eavesdropper channel gains.

**Their stated limitations:** None in the main body — the paper works entirely under the perfect-global-CSI assumption without flagging it as unrealistic mid-paper (unlike Tekin & Yener, this one doesn't editorialize about practicality until the conclusion).

**Their stated future work — verbatim, third paper in a row hitting the same open problem:** *"Areas that warrant further research include **performance degradation in the presence of imperfect channel estimates**, and **optimization based on partial channel knowledge only, e.g., only statistical information about the eavesdropper's channels is available**, or each relay knows its own channel only."*

**Vs. our work:** Mechanistic ancestor of the "jamming" half of cooperative friendly jamming — their CJ scheme (dedicated relays broadcasting a jamming signal while the source transmits) is the conceptual predecessor of our idle-AP-as-jammer design, just using purpose-built relay nodes instead of repurposing existing infrastructure APs. Their solution method (closed-form/iterative optimization requiring exact global CSI) is also a clean contrast point: it's precise but brittle — any CSI error breaks the closed-form optimality guarantee, which is exactly why they flag imperfect/statistical-only CSI as their own open problem, and exactly why an RL approach (ours) that doesn't require a closed-form solution is a structurally different way to handle that same gap.

**Why it's in our lit review:** Third consecutive paper in this chronological review (1978 → 2008 → 2010) to independently flag imperfect/statistical-only eavesdropper CSI as an acknowledged open limitation. That's a genuinely strong pattern to name explicitly in our own Related Work — this isn't a niche concern we invented, it's a recurring, repeatedly-unaddressed gap spanning three decades of foundational PLS work. Also useful as the "cooperative jamming" terminology's early lineage, distinct from our own "cooperative *friendly* jamming" framing (Hoseini's term, reusing existing AP infrastructure rather than dedicated relays).

## 7. Luong et al., 2019 — Applications of Deep Reinforcement Learning in Communications and Networking: A Survey

**Citation:** N. C. Luong, D. T. Hoang, S. Gong, D. Niyato, P. Wang, Y.-C. Liang, and D. I. Kim, "Applications of Deep Reinforcement Learning in Communications and Networking: A Survey," *IEEE Commun. Surveys Tuts.*, vol. 21, no. 4, pp. 3133–3174, 4th Quarter 2019.
**Link:** https://arxiv.org/abs/1810.07862
**Source read:** Full PDF, real text layer (33 pages) — read the intro, the network-security section, and the conclusion/future-directions sections in full; skimmed the domain-by-domain application sections (routing, caching, offloading) since they're outside our scope.

**What it does:** Broad tutorial-plus-survey of DRL applied across networking generally — not PLS-specific. Opens with a DRL fundamentals tutorial (MDPs, Q-learning, DQN, policy gradient, actor-critic), then surveys applications across dynamic network access, rate control, caching, offloading, network security, and connectivity preservation.

**Core mechanism:** Not a single method — it's a taxonomy/survey. The one section directly relevant to us, "Network Security," covers DRL for **anti-jamming** (defending against a jamming *attacker* using frequency-hopping / Q-learning-based channel selection) and cyber-physical attack defense. Notably, this is the *opposite* framing from our own work: here, jamming is the attacker's weapon and DRL is the defense; in cooperative friendly jamming, jamming is the defender's own tool.

**Eve/CSI assumption:** Not applicable in the PLS sense — this survey's "security" scope is anti-jamming/intrusion, not eavesdropping/secrecy capacity. No Eve-location or CSI framing at all.

**Their stated limitations:** Flagged directly as an open issue: most DRL-for-wireless work (this survey included, by implication) relies on **simulated data**, since real referential datasets aren't available the way they are in computer vision — and simulated data "is a simplification of the real system and may overlook hidden patterns," which "undermines the confidence of the DRL framework in practical systems." Relevant to us too, honestly — our own results are Friis-path-loss simulation, not hardware validation, same category of limitation.

**Their stated future work:** A long list of open directions (Section VII): DRL for channel estimation in massive MIMO, distributed/decentralized DRL frameworks for resource-constrained end devices, balancing information-gathering cost against learning performance, DRL for crowdsensing and cryptocurrency management, DRL for auctions. None specific to physical layer security or eavesdropper uncertainty — this survey's future-work agenda doesn't overlap with our contribution at all.

**Vs. our work:** Not a technical comparator — it's a breadth citation. Its main value is establishing that DRL for wireless resource/network optimization was already a broad, well-established, actively-growing research area by 2019, which supports the algorithm-choice justification (why DRL/SAC is a reasonable methodology, not an exotic one) without claiming any specific overlap with cooperative jamming or PLS.

**Why it's in our lit review:** Background/recency-lineage citation for "DRL for wireless" broadly, sitting alongside Nasir & Guo and Li et al. (network slicing) as evidence that DRL had already proven itself across multiple networking subdomains before Hoseini et al. applied it to cooperative jamming specifically. Keep it as a one-line supporting citation in the Introduction/Related Work, not a paragraph — it doesn't carry enough PLS-specific weight for more than that.

## 8. Nasir & Guo, 2019 — Multi-Agent Deep Reinforcement Learning for Dynamic Power Allocation in Wireless Networks

**Citation:** Y. S. Nasir and D. Guo, "Multi-Agent Deep Reinforcement Learning for Dynamic Power Allocation in Wireless Networks," *IEEE J. Sel. Areas Commun.*, vol. 37, no. 10, pp. 2239–2250, Oct. 2019.
**Link:** https://arxiv.org/abs/1808.00490
**Source read:** Full PDF, real text layer.

**What it does:** Distributed, multi-agent deep Q-learning (DQN, discrete actions) for transmit power control across many interfering links, maximizing a weighted sum-rate utility. **No secrecy/eavesdropping angle at all** — this is pure throughput/interference-management power control, not PLS. Each transmitter is its own agent, using only locally-available CSI/QoS info from a few neighbors, not full network CSI.

**Core mechanism:** Each link's transmitter is a DQN agent choosing a discrete power level based on local channel state + interference measurements from nearby links, trained to maximize a global weighted sum-rate objective despite fully decentralized execution. Explicitly designed to be robust to **delayed and imperfect CSI** — the abstract states the scheme is "especially suitable for practical scenarios where the system model is inaccurate and CSI delay is non-negligible."

**Eve/CSI assumption:** No eavesdropper exists in this model at all. On the CSI side: deliberately *does not* assume perfect/instantaneous CSI — one of its central selling points over classical WMMSE/FP benchmarks is exactly that it tolerates delayed, local-only CSI without needing the full cross-cell CSI those algorithms require.

**Their stated limitations:** Centralized training phase named as a limitation on scalability, though they show a workaround (train small, deploy large; warm-start from a differently-trained DQN).

**Their stated future work:** Regional/local retraining instead of full global retraining when new links join or performance drops; fully distributed (not just distributed execution, but distributed *training*) as a further extension; explicit note that a case with "inaccurate CSI measurements is left for future work" in one specific sub-scenario they didn't fully cover.

**Vs. our work:** Different objective (throughput, not secrecy) but a genuinely relevant methodological precedent: this is one of the earliest papers to show a DRL power-control agent can be trained to be robust to CSI imperfection *by construction*, rather than assuming it away. That's the same instinct behind UA-SAC, just applied to sum-rate instead of secrecy capacity, and via discrete DQN instead of continuous SAC. Also useful as a contrast on architecture: multi-agent/distributed (many independent DQN agents, local CSI only) vs. our single centralized SAC agent with full network-state observation — worth a sentence noting that direction as a possible extension of our own work (a decentralized UA-SAC), not something we've done.

**Why it's in our lit review:** Supports two claims at once — (1) DRL for power control robust to imperfect CSI has real precedent, predating our imperfect-Eve-CSI framing by six years, even outside the secrecy context; (2) it's a clean example of the field's broader move from centralized-full-CSI to distributed-partial-CSI designs, a trend our own uncertainty-aware framing fits into naturally.

## 9. Li et al., 2018 — Deep Reinforcement Learning for Resource Management in Network Slicing

**Citation:** R. Li, Z. Zhao, Q. Sun, C.-L. I, C. Yang, X. Chen, M. Zhao, and H. Zhang, "Deep Reinforcement Learning for Resource Management in Network Slicing," *IEEE Access*, vol. 6, pp. 74429–74441, 2018.
**Link:** https://doi.org/10.1109/ACCESS.2018.2881964
**Source read:** Full PDF, real text layer.

**What it does:** Applies deep Q-learning to 5G network slicing resource management — allocating radio and core-network resources across tenant "slices" to match fluctuating user demand, rather than static or prediction-based allocation. **No PLS/secrecy content whatsoever** — pure resource-management efficiency problem.

**Core mechanism:** DQN-based agents for two specific scenarios: radio resource slicing and priority-based core network slicing, benchmarked against demand-prediction-based and other heuristic baselines via simulation.

**Eve/CSI assumption:** Not applicable — no eavesdropper or channel-secrecy concept anywhere in this paper.

**Their stated limitations:** Named directly: successful DQL application to slicing "needs some careful considerations," starting with slice admission control for incoming requests for new slices — flagged as an open practical concern, not solved here.

**Their stated future work:** General — believe "DRL could play a crucial role in network slicing in the future," with admission control named as the next concrete problem to tackle. Nothing PLS-adjacent.

**Vs. our work:** No direct technical overlap — different domain (resource slicing vs. secrecy), different objective, different DRL flavor (discrete DQN vs. continuous SAC). Purely a breadth citation.

**Why it's in our lit review:** Same role as the Luong survey — evidence that DRL was already a proven, general-purpose tool across multiple 5G/networking subdomains (here: slicing) well before Hoseini et al. applied it to cooperative jamming specifically. One-line supporting citation only; doesn't warrant more space than that.

## 10. Haarnoja, Zhou, Abbeel & Levine, 2018 — Soft Actor-Critic (ICML)

**Citation:** T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor," in *Proc. ICML*, PMLR vol. 80, pp. 1856–1865, 2018.
**Link:** https://arxiv.org/abs/1801.01290
**Source read:** Full PDF, real text layer.

**What it does:** Introduces SAC itself — the algorithm both the baseline (Hoseini) and our own UA-SAC are built directly on top of. An off-policy actor-critic algorithm for continuous control, based on the maximum entropy RL framework: the actor maximizes expected reward *and* policy entropy simultaneously — "succeed at the task while acting as randomly as possible."

**Core mechanism:** Derives "soft policy iteration" and proves it converges to the optimal maximum-entropy policy, then turns that into a practical deep RL algorithm: two Q-networks (critics) trained via soft Bellman backups, a stochastic Gaussian actor trained to maximize `Q − α·log π` (reward minus entropy-weighted log-probability), and a slowly-tracking target value network updated via exponential moving average (smoothing constant τ). Off-policy — reuses a replay buffer instead of requiring fresh on-policy samples every gradient step, which is the main source of its sample efficiency advantage over PPO/TRPO.

**Eve/CSI assumption:** Not applicable — general-purpose continuous-control RL paper (MuJoCo benchmarks: hopper, walker, ant, humanoid), no wireless or security content at all.

**Their stated limitations:** Reward scale identified as "the only hyperparameter that requires tuning" in practice — too small and the policy becomes overly random/noisy for good asymptotic performance, too large and it becomes nearly deterministic with poor exploration, risking bad local minima. Also: target-network update rate τ trades off stability (small τ = more stable, slower) against training speed (large τ = faster, riskier).

**Their stated future work:** Verbatim: *"further exploration of maximum entropy methods, including methods that incorporate second order information (e.g., trust regions) or more expressive policy classes is an exciting avenue for future work."* General RL-methodology future work, nothing domain-specific.

**Vs. our work:** This is the direct algorithmic parent of both the baseline SAC and our UA-SAC. Two points worth being precise about: (1) the "reward scale is the only hyperparameter that matters" finding is exactly the temperature/entropy-coefficient sensitivity that motivates UA-SAC's α_eff scaling — we're not inventing a new sensitivity, we're exploiting a known one (uncertainty should shift the operating point along the exploration/exploitation axis this paper already identified as critical). (2) the original SAC treats the entropy coefficient α as a single global scalar tuned once (or auto-tuned via the dual-objective extension in the companion paper); UA-SAC's batch-average ρ-scaling is a direct, minimal modification of that single global scalar into a state-conditioned one — worth stating explicitly as "the smallest change to Haarnoja et al.'s formulation that makes α uncertainty-aware," since it clarifies exactly how surgical our algorithmic change is.

**Why it's in our lit review:** Not a competitor, not a comparison point — it's the algorithm itself. Every equation in our Methodology section that touches α, the entropy term, or the actor/critic loss traces directly back to this paper. Cite early, cite precisely (the auto-tuning behavior we rely on for α_base is actually from the *companion* paper — see #11 below, not this one).

## 11. Haarnoja et al., 2018/19 — Soft Actor-Critic Algorithms and Applications

**Citation:** T. Haarnoja, A. Zhou, K. Hartikainen, G. Tucker, S. Ha, J. Tan, V. Kumar, H. Zhu, A. Gupta, P. Abbeel, and S. Levine, "Soft Actor-Critic Algorithms and Applications," arXiv:1812.05905, 2018 (revised Jan. 2019).
**Link:** https://arxiv.org/abs/1812.05905
**Source read:** Full PDF, real text layer.

**What it does:** The companion/extension paper to the original ICML SAC paper. Two additions matter for us: (1) **automatic temperature (entropy coefficient) tuning** — turns α from a hand-tuned hyperparameter into something learned during training; (2) demonstrates SAC on real physical hardware (a Dynamixel Claw valve-rotation task from raw pixel observations), not just simulation.

**Core mechanism — this is the one that actually matters for our code:** Section 5, "Automating Entropy Adjustment," reformulates the maximum-entropy objective as a **constrained optimization**: maximize expected return subject to the policy's entropy staying above a minimum target `H`, rather than fixing a reward-bonus weight upfront. Solved via Lagrangian duality — the dual variable *is* the temperature α. This derivation is exactly what `ent_coef="auto"` implements in Stable-Baselines3, and exactly what our own `uasac.py`'s `self.log_ent_coef` and `ent_coef_loss = -(log_ent_coef · (log_prob + target_entropy)).mean()` update is doing — this paper is the literal mathematical source of that specific block of our own code, not just a general SAC citation.

**Eve/CSI assumption:** Not applicable — general RL methodology + robotics hardware demonstration, no wireless content.

**Their stated limitations:** The paper itself notes a subtlety worth being precise about: since reward magnitude varies both across tasks *and* over the course of training as the policy improves, a fixed entropy target is a poor solution on its own — "the policy should be free to explore more in regions where the optimal action is uncertain, but remain more deterministic in states with a clear distinction between good and bad actions." Their constrained-optimization solution addresses the *across-training* variation; it does not by itself make α state-conditional in the way UA-SAC's ρ-scaling does.

**Their stated future work:** General conclusion, no explicit forward-looking research agenda beyond continued application to harder real-world robotic tasks.

**Vs. our work:** This is the precise origin of `α_base` in our system — SB3's auto-tuned entropy coefficient, which UA-SAC then multiplies by `(1 + β·ρ)` to get `α_eff`. Worth stating this precisely in the Methodology section: UA-SAC does not replace or compete with this paper's automatic tuning mechanism, it *composes* with it — `α_base` is still learned exactly as this paper describes, and our contribution is the multiplicative uncertainty-scaling layered on top. The quote above about wanting entropy to vary "across states" but implementing only an average constraint is also directly relevant: it's an implicit admission that their own automatic tuning is a *global* average-entropy mechanism, not a per-state one — which is exactly the same category of limitation we flagged in our own α_eff (batch-average ρ, not per-sample). Worth citing as evidence this is a known, general tension in entropy-regularized RL, not a flaw unique to our implementation.

**Why it's in our lit review:** The direct mathematical source of our `α_base` auto-tuning mechanism — more precise and more load-bearing for our Methodology section than the original ICML paper, since this is specifically where the constrained-optimization/dual-variable derivation UA-SAC builds on top of actually comes from.

## 12. Mukherjee & Swindlehurst, 2011 — Robust Beamforming for Security in MIMO Wiretap Channels with Imperfect CSI

**Citation:** A. Mukherjee and A. L. Swindlehurst, "Robust Beamforming for Security in MIMO Wiretap Channels With Imperfect CSI," *IEEE Trans. Signal Processing*, vol. 59, no. 1, pp. 351–361, Jan. 2011.
**Link:** https://arxiv.org/abs/1009.2274
**Source read:** Full PDF, real text layer.

**What it does:** Two multi-antenna nodes want to communicate securely against a passive eavesdropper of *unknown* channel/location. The transmitter allocates power to hit a target SINR for the legitimate receiver, then broadcasts remaining power as **artificial noise (AN)** — deliberately structured interference designed to be invisible to the legitimate receiver (orthogonal to its signal) but disruptive to the eavesdropper.

**Core mechanism:** AN beamforming assuming no eavesdropper CSI at all in the baseline design, then a **second-order perturbation analysis** to precisely quantify how much performance degrades under channel estimation error, followed by two robust beamforming variants designed to recover most of the lost SINR under imperfect CSI.

**Eve/CSI assumption:** Explicitly *no* prior knowledge of Eve's channel required for the base AN scheme — a genuinely early example of designing for an unknown eavesdropper. But the robust *legitimate-channel* CSI (Bob's, not Eve's) is assumed accurate in the base case, and the whole paper is about what happens when even that assumption breaks.

**Their stated limitations — verbatim, and this is the key line:** *"The proposed approaches rely heavily on the availability of accurate CSI, and their performance can be quite sensitive to imprecise channel estimates... These techniques were shown to perform very well for moderate CSI errors, but ultimately a large enough channel mismatch can eliminate the secrecy advantage of using artificial noise."*

**Their stated future work:** Not a dedicated section — the perturbation-analysis and robust-scheme contributions are presented as addressing the CSI-sensitivity problem directly, with the closing caveat above rather than a forward-looking list.

**Vs. our work:** This is "Artificial Noise" (AN), a close terminological and mechanistic cousin of cooperative jamming — same idea (deliberate interference to blind Eve), different origin (transmitter's own excess power vs. other APs/relays). The critical finding to carry into our Discussion: even a *closed-form, analytically-derived* robust scheme has a breaking point — "large enough channel mismatch can eliminate the secrecy advantage" entirely, not degrade it gracefully. That's a sharp contrast with UA-SAC's design goal: rather than a closed-form robustness guarantee that holds up to some error bound and then fails, a learned worst-case policy has no such hard analytical cliff (though it also has no hard *guarantee* either — worth being honest that we trade certified degradation limits for empirical, sample-based robustness).

**Why it's in our lit review:** Strong technical comparator for the Related Work section on "prior robust approaches to CSI uncertainty" — it's beamforming/AN rather than cooperative jamming/RL, has no eavesdropper location model at all (works off statistics, not position), and its own stated failure mode (catastrophic breakdown past a CSI-error threshold) is a genuinely useful contrast against RL-learned robustness.

## 13. Wang, So, Chang, Ma & Chi, 2014 — Outage Constrained Robust Transmit Optimization for Multiuser MISO Downlinks

**Citation:** K.-Y. Wang, A. M.-C. So, T.-H. Chang, W.-K. Ma, and C.-Y. Chi, "Outage Constrained Robust Transmit Optimization for Multiuser MISO Downlinks: Tractable Approximations by Conic Optimization," *IEEE Trans. Signal Processing*, vol. 62, no. 21, pp. 5690–5705, Nov. 2014.
**Link:** https://www1.se.cuhk.edu.hk/~manchoso/papers/outage_txbeam-TSP.pdf
**Source read:** Full PDF, real text layer — including a full-text search confirming **zero occurrences of "secrecy" or "eavesdrop" anywhere in the document.**

**Important correction to how this paper is currently framed in our bibliography:** it's listed alongside Mukherjee (robust beamforming for *security*) as if it's a PLS precedent. It isn't. This is a general **QoS/rate-outage** robust MISO precoding paper — multiple legitimate users, Gaussian CSI errors, keeping each user's rate-outage probability below a threshold. No adversary, no eavesdropper, no secrecy objective anywhere in the formulation.

**What it does:** Solves a rate-outage-constrained MISO downlink precoding problem under imperfect CSI. The outage probability constraints are computationally intractable in exact form, so the paper derives three different convex-restriction techniques (sphere bounding, Bernstein-type inequality, decomposition-based large-deviation inequality) that each give an efficiently-computable, safe approximation.

**Core mechanism:** All three methods are ways of turning "keep P(rate < target) ≤ ε" into a tractable convex constraint, trading off approximation tightness against computational cost. Benchmarked against each other and prior state of the art via simulation.

**Eve/CSI assumption:** No Eve exists in this model. On CSI: legitimate-user channel errors are modeled as Gaussian, the whole point of the paper.

**Their stated limitations:** None flagged as unresolved — the three proposed methods are presented as each valid, with tradeoffs between tightness and complexity rather than open problems.

**Their stated future work:** The rate-outage-constrained formulation is noted as extensible to other problems (max-min-fairness, achievable-rate-region characterization), explored further in a companion technical report — no secrecy-related direction mentioned anywhere.

**Vs. our work:** No direct secrecy/PLS overlap at all — this is a pure methodology citation. What *is* legitimately transferable: the general technique of converting a probabilistic robustness requirement (here, rate outage; in our case, worst-case secrecy) into a tractable optimization target. Their route is closed-form convex restriction; ours is RL sampling (M=5 worst-case draws). Worth a passing methodological note, not a secrecy comparison.

**Why it's in our lit review — and this needs a decision, not just a citation:** As currently framed (grouped with Mukherjee as PLS precedent), this citation is **inaccurate** and a careful reviewer familiar with the paper would notice it has nothing to do with secrecy. Two honest options: (1) recast it purely as a general robust-optimization-under-CSI-uncertainty methodology citation, separate from the PLS-specific cluster, or (2) drop it and lean on Mukherjee alone for that role, since Mukherjee actually *is* a secrecy paper with the same "robust under CSI error" theme. Recommend option 1 if the citation count matters, option 2 if precision matters more — but do not leave it silently mislabeled as a PLS paper.

## 15. Yang, Xiong, Zhao, Niyato, Xiao & Wu, 2021 — DRL-Based Intelligent Reflecting Surface for Secure Wireless Communications

**Citation:** H. Yang, Z. Xiong, J. Zhao, D. Niyato, L. Xiao, and Q. Wu, "Deep Reinforcement Learning Based Intelligent Reflecting Surface for Secure Wireless Communications," *IEEE Trans. Wireless Commun.*, vol. 20, no. 1, pp. 375–388, Jan. 2021.
**Link:** https://arxiv.org/abs/2002.12271
**Source read:** Full PDF, real text layer.

**What it does:** Jointly optimizes a base station's active beamforming and an IRS's passive reflecting beamforming to maximize secrecy rate against multiple eavesdroppers, under time-varying channels — via DRL rather than closed-form optimization, since the joint problem is non-convex and the environment is dynamic.

**Core mechanism:** A "deep PDS-PER" learning approach — post-decision state (PDS) representation plus prioritized experience replay (PER), layered on top of a DRL base (not explicitly SAC) — designed specifically to track channel dynamics and improve learning efficiency/convergence speed.

**Eve/CSI assumption:** Worth being precise here — their robustness axis is **outdated CSI** (a staleness parameter ζ, ranging from fully outdated to current), not unknown eavesdropper location. Eve's CSI is *tracked over time* and degrades in accuracy due to delay, not fundamentally unobservable the way we assume. This is a meaningfully different uncertainty model from ours — delay-induced staleness vs. structural unknowability.

**Their stated limitations:** The paper's own motivation section critiques the cooperative-jamming/artificial-noise family directly — worth quoting: *"employing a large number of active antennas and relays in PLS systems incurs an excessive hardware cost... cooperative jamming and transmitting artificial noise require extra transmit power for security guarantees."* This is a direct critique of our own approach's family, from within the literature, and should be addressed honestly rather than ignored — cooperative friendly jamming genuinely does cost extra transmit power compared to a passive IRS; our counter is that it requires zero new hardware (reuses existing APs) where IRS requires deploying new physical reflecting surfaces.

**Their stated future work:** No dedicated section; the Conclusion restates the contribution (PDS-PER beamforming beats other approaches on secrecy rate and QoS satisfaction) without a forward-looking research agenda.

**Vs. our work:** A genuinely useful "different tool, same goal" comparator — passive reflecting elements instead of active jamming power, DRL for joint active+passive beamforming instead of pure power control, robustness to CSI *staleness* instead of *unknowability*. Good candidate for the IRS-alternatives subsection of Related Work, explicitly contrasted against cooperative jamming's hardware-reuse advantage and IRS's power-efficiency advantage.

**Why it's in our lit review:** Represents the IRS branch of DRL-for-PLS, parallel to (not competing directly with) the cooperative-jamming branch we're in. Also gives us a citable, in-literature critique of jamming-based approaches (extra power cost) that we should address head-on in Discussion rather than let a reviewer raise it first.

## 16. Bouhafs et al., 2018 — Wi-5: A Programming Architecture for Unlicensed Frequency Bands

**Citation:** F. Bouhafs, M. Mackay, A. Raschellà, Q. Shi, F. den Hartog, J. Saldana, R. Munilla, J. Ruiz-Mas, J. Fernández-Navajas, J. Almodovar, and N. van Adrichem, "Wi-5: A Programming Architecture for Unlicensed Frequency Bands," *IEEE Commun. Mag.*, 2018.
**Link:** https://zaguan.unizar.es/record/85397/files/texto_completo.pdf
**Source read:** Full PDF, real text layer. Systems/architecture magazine article, not a research algorithm paper — no secrecy content, cited purely for infrastructure feasibility.

**What it does:** Presents "Wi-5," an SDN-style spectrum programming architecture that extends the SDN control plane with a new "spectrum plane" — enabling centralized, fine-grained, QoS-aware control over Wi-Fi AP radio parameters, explicitly including **per-AP channel and transmit power configuration**, via open APIs rather than vendor-locked management systems.

**Core mechanism:** Not an algorithm — a systems architecture. Demonstrates via testbed emulation that a centralized "Wi-5 controller" can dynamically manage client-AP associations and mitigate spectral congestion in real time (de-authenticating/re-admitting clients, handing clients between APs).

**Eve/CSI assumption:** Not applicable — no security content in this paper at all.

**Their stated limitations:** None flagged as unresolved in the excerpt reviewed; presented as a working, demonstrated proof-of-concept.

**Their stated future work:** Positions the architecture as enabling "new possibilities for managing radio resources," implying further management applications beyond the three demonstrated, but no specific named research gap.

**Vs. our work:** Not a technical or algorithmic comparator at all — it's an infrastructure-feasibility citation. Co-authored by Bouhafs and den Hartog, both also co-authors of our Hoseini baseline paper — this is their own prior work establishing that **centralized real-time control of Wi-Fi AP transmit power is a practically deployable capability**, not a theoretical fantasy. Directly supports the practical premise our whole system depends on: a centralized SAC/UA-SAC controller commanding real AP power levels is implementable on existing infrastructure via architectures like this one.

**Why it's in our lit review:** Grounds the deployability claim. When the paper says cooperative friendly jamming "requires no additional hardware" and can run on "existing infrastructure," this is the citation that backs that claim with a real, demonstrated system rather than an assumption.

## 17. Bouhafs et al., 2020 — Realizing Physical Layer Security in Large Wireless Networks Using Spectrum Programmability

**Citation:** F. Bouhafs, F. den Hartog, A. Raschellà, M. Mackay, Q. Shi, and S. Sinanovic, "Realizing Physical Layer Security in Large Wireless Networks Using Spectrum Programmability," in *2020 IEEE Globecom Workshops (GC Wkshps)*, 2020, DOI: 10.1109/GCWkshps50303.2020.9367399.
**Link:** https://researchonline.gcu.ac.uk/ws/portalfiles/portal/44677213/
**Source read:** Full PDF, author-accepted-manuscript version (ResearchOnline GCU repository copy) — real text layer, read in full including the Adversary Model and Conclusion sections. My first pass at this entry (before actually reading the body) assumed it was about jamming/power control — it isn't. Corrected below.

**What it does:** Proposes a secrecy-aware **AP selection algorithm** — not jamming, not power control — that connects a Wi-Fi user to whichever AP offers the highest secrecy capacity, implemented on the Wi-5 spectrum programming architecture (#16). Evaluated in a 300m×300m simulated Wi-Fi network. First paper of its kind, per the authors, to apply PLS to secure a *complete* wireless network rather than a single link.

**Core mechanism:** Secrecy-rate-based AP selection: for each user, compute the secrecy rate achievable at each candidate AP against the nearest/strongest eavesdropper, pick the AP that maximizes it. No jamming, no power optimization — pure association-layer decision.

**Eve/CSI assumption — this is the important finding:** Their adversary model states plainly: *"the location of the eavesdropping stations are, however, known to the Wi-Fi APs of the network. This is a reasonable assumption as previous work proves APs can detect eavesdropping stations."* **This directly contradicts our project's central premise.** Worth being precise about the tension rather than glossing over it: this is co-authored by Bouhafs and den Hartog, who are *also* co-authors on our own baseline (Hoseini et al.). Two papers from the same author group take opposite positions on whether Eve's location is knowable. Their justification rests on citing prior AP-based eavesdropper-detection work — meaning their "known location" is itself an assumption built on a *separate, non-trivial detection capability*, not a given. That's actually a defensible way to reconcile the tension in our own Related Work: their detection-based approach requires deploying and trusting a wiretap-detection mechanism first; our contribution is what to do when you either don't have or don't trust that detection layer.

**Their stated limitations:** Named directly — the security improvement comes "at the expense of a lower capacity rate," an explicit security/performance tradeoff, not a free win.

**Their stated future work:** Verbatim, two named directions: (1) "the role of indoor vs outdoor environments and the inclusion of reflections," and (2) "the estimation of latency and overhead introduced by using the Wi-5 spectrum programming platform." Neither overlaps with eavesdropper-uncertainty — their acknowledged gaps are about propagation modeling and system overhead, not about the location-known assumption they explicitly defended in the adversary model.

**Vs. our work:** Genuinely important contrast, not just infrastructure lineage. (1) Mechanistically different: secure AP *selection* (who serves whom) vs. our secure *power control* given a fixed association — actually a nice complementary pairing, since our own paper's AP association step is the *one part of our system* still using a crude fixed heuristic (Section on Hoseini's inherited limitations) — this paper is evidence that smarter, secrecy-aware AP selection is itself a live research question, reinforcing that our own "AP association future work" gap is a real, currently-open direction, not a throwaway limitation. (2) Assumption-level: they assume detectable Eve location; we assume the opposite. This needs to be addressed explicitly in our paper, not left as an unacknowledged contradiction with our own baseline's co-authors' other work.

**Why it's in our lit review:** Two jobs — (1) completes the infrastructure-to-security lineage (Wi-5 general architecture → this paper applying it to PLS via AP selection → Hoseini et al. adding DRL-based power optimization on top), and (2) forces an honest reckoning with the "is Eve's location actually knowable" question using a source that can't be dismissed as external or irrelevant — it's our own baseline's co-authors' other work. Address this directly in the paper rather than hope nobody cross-references it.

## 18. Hoseini, Bouhafs, Aboutorab, Sadeghi & den Hartog, 2023/2024 — Cooperative Jamming for Physical Layer Security Enhancement Using Deep Reinforcement Learning (OUR BASELINE)

**Citation:** S. A. Hoseini, F. Bouhafs, N. Aboutorab, P. Sadeghi, and F. den Hartog, "Cooperative Jamming for Physical Layer Security Enhancement Using Deep Reinforcement Learning," in *2023 IEEE Globecom Workshops (GC Wkshps)*, pp. 1838–1843, DOI: 10.1109/GCWkshps58843.2023.10465104.
**Link:** `Baseline_Paper/Hoseini Baseline IEEE Formal.pdf` (formal published version, DOI header confirmed present in the file itself — not the arXiv preprint)
**Source read:** Full PDF, real text layer, both the formal IEEE version (this session) and the arXiv HTML version (read in full earlier in this project). Cross-checked — no meaningful content drift between the two; the future-work paragraph is word-for-word consistent across both.

**What it does:** The paper this entire project extends. N Wi-Fi APs share one 2.4 GHz band; APs not serving a user transmit jamming noise instead of going idle; a SAC agent learns per-AP transmit power to maximize sum secrecy capacity across all legitimate users, given the *exact* location of every eavesdropper as part of its observed state.

**Core mechanism:** State `S = {L_AP, L_u, L_e}` (AP, user, *and eavesdropper* locations, all exact), action `P^t = [p1,...,pN]` (continuous per-AP power), reward = sum secrecy capacity. AP-user association is a separate, deterministic pre-step (Eq. 7): each user assigned to whichever AP maximizes secrecy capacity *assuming uniform max power on all APs* — computed once before the SAC agent acts, never revisited.

**Eve/CSI assumption:** Perfect and explicit — Eve's exact coordinates are the third component of the state vector, no uncertainty modeled anywhere. This is precisely the assumption our whole project exists to remove.

**Their stated limitations — and this is the standout finding from this session's read:** Beyond the two future-work items (below), the formal version documents an **empirically observed failure mode**, not just a theoretical caveat: in their scenario 5 (adding 2 more APs relative to scenario 4), sum secrecy capacity *unexpectedly dropped* instead of improving. Their own diagnosis, verbatim: *"We conclude that since the AP selection algorithm is done separately and first, by assuming uniform power allocation, and then the RL model is used to find the optimal power vector, the whole process is not optimal."* This is real, documented, experimental evidence of the exact association/power mismatch problem discussed at length in this project's own working sessions — not a hypothetical concern, an observed result in the baseline's own paper.

**Their stated future work — verbatim:** *"we aim to integrate the AP selection algorithm into the RL model in our future work, to further optimize the secrecy of the wireless network. Finally, we are also considering the extension of this work to mobile legitimate stations and eavesdroppers, by training the RL model to the dynamic environment and mobility patterns."* Two gaps: (1) joint AP-selection + power learning, (2) mobility. Neither is addressed by our own UA-SAC — both remain open, inherited unchanged.

**Vs. our work:** This *is* the work being extended — not a comparator, the baseline itself. Every equation, every design decision in `env/cfj_env.py` traces back to this paper's system model. UA-SAC's three modifications (worst-case reward, ρ-augmented state, ρ-scaled entropy) replace exactly the `L_e` term in their state definition and nothing else — everything about the physical layer, jamming mechanism, and AP association is untouched, inherited as-is, limitations included.

**Why it's in our lit review:** The center of gravity for the entire paper. Cited most heavily, described most precisely, and — per this session's finding — should be cited for more than just "the baseline we extend": the scenario-5 empirical observation is a genuinely strong, concrete piece of evidence to cite when acknowledging the AP-association limitation in our own Limitations section, since it shows the problem isn't speculative, it's something the original authors watched happen in their own results.

## 20. Liu, Chen & Wang, 2017 — Physical Layer Security for Next Generation Wireless Networks: Theories, Technologies, and Challenges (the orphan)

**Citation:** Y. Liu, H.-H. Chen, and L. Wang, "Physical Layer Security for Next Generation Wireless Networks: Theories, Technologies, and Challenges," *IEEE Commun. Surveys Tuts.*, vol. 19, no. 1, pp. 347–376, 1st Quarter 2017.
**Link:** https://www.researchgate.net/publication/310824849
**Source read:** Full PDF, real text layer. This is the entry currently sitting in our bibliography as an uncited orphan — read specifically to determine whether it should be cited properly or cut.

**What it does:** Broad survey of PHY-security spanning confidentiality *and* authentication — wiretap coding, secure multi-antenna technologies (MISO/MIMO precoding), secure relay technologies, PHY-based key generation, and PHY-authentication against impersonation. Wider scope than our project needs (we don't touch key generation or authentication at all).

**Core mechanism — the part that actually matters for us:** Section on **"Partial/Imperfect CSI Solutions"** covers two relevant threads: (1) **statistical CSI approaches** — using a channel statistical model rather than exact CSI when Eve's channel is uncertain (same family as the Xiao/STAR-RIS paper's approach); (2) **compound wiretap channels** — formulating secrecy against the *worst case over a whole set of possible channel states* rather than one known state, covering both broadcast and multi-access variants. That compound-channel formulation is structurally the same idea as our worst-case reward — minimize over a set of possibilities — just from 1990s-2010s information theory rather than 2020s DRL.

**Eve/CSI assumption:** Survey-level, covers the full spectrum from perfect CSI to statistical-only to compound/worst-case formulations — not a single fixed assumption, which is exactly why it's useful as a citation for framing the *landscape* of CSI-uncertainty approaches.

**Their stated limitations:** None specific — survey format, presents each subfield's open issues rather than the authors' own unresolved problem.

**Their stated future work:** General restatement of the survey's scope in the Conclusion (fading effects, partial/imperfect CSI, compound channels, MIMO precoding, relay systems, key generation, authentication) — no single named research gap distinct from the body.

**Vs. our work:** Not a direct technical comparator, but its "compound wiretap channel" concept is a legitimate, older, information-theoretic ancestor of the worst-case-over-samples idea our reward function implements — worth a citation for exactly that lineage, distinct from the closed-form robust-beamforming citations (Mukherjee, Xiao) which handle uncertainty via convex approximation rather than a worst-case-over-a-set formulation.

**Why it's in our lit review, and the actual fix for the orphan problem:** Currently cited nowhere in the body — that's the defect flagged repeatedly this session. The fix: cite it specifically in the imperfect-CSI discussion, next to Mukherjee/Xiao, for its coverage of compound wiretap channels and statistical CSI approaches — not as a generic "PLS exists" citation (which is what an orphan citation usually turns into if forced in later). This gives it a genuine, specific job instead of just closing the orphan gap cosmetically.

## 21. Zhao, Li, Si, Huang, Hu, Quek & Al-Dhahir, 2025 — DRL-Based Robust Multi-Timescale Anti-Jamming Approaches under State Uncertainty

**Citation:** H. Zhao, Z. Li, J. Si, R. Huang, H. Hu, T. Q. S. Quek, and N. Al-Dhahir, "DRL-Based Robust Multi-Timescale Anti-Jamming Approaches under State Uncertainty," *IEEE Trans. Cognitive Commun. Networking*, 2025 (arXiv:2511.03305).
**Link:** https://arxiv.org/abs/2511.03305
**Source read:** Full text (arXiv HTML, read earlier this session) — real content, not abstract-only, cross-verified in this session by matching the local PDF's abstract against the earlier extraction (identical, including PGD-DDQN/NQC-DDQN algorithm names).

**What it does:** **Anti**-jamming (defending against a malicious jammer attacking the legitimate link) under sensor/state uncertainty — the mirror-image problem of our cooperative *friendly* jamming, where the DRL agent is the target of interference rather than the deployer of it. Proposes two robust DDQN variants that stay reliable when sensing the environment (not Eve, but the jamming/channel state) is imperfect.

**Core mechanism:** **PGD-DDQN** — derives worst-case perturbations under a norm-bounded sensing-error model via Projected Gradient Descent during training, using true-state optimal actions as supervised regularization labels. **NQC-DDQN** — uses interval bound propagation to compute certified Q-value ranges per action, then nonlinearly compresses them so the lower bound of the best action's Q-value provably exceeds the upper bounds of all others, eliminating "action aliasing" under perturbation.

**Eve/CSI assumption:** No eavesdropper at all — this is anti-jamming, not secrecy. The uncertainty is about sensing the *jammer's* state/the channel, not an eavesdropper's location.

**Their stated limitations — verbatim:** *"the agent has no prior knowledge of whether the input state is accurate"* — a structural admission that the robustness is trained-in, not detected at runtime. NQC-DDQN specifically: *"exhibit certain decision deviations"* — the certified bound reduces but does not eliminate residual errors.

**Their stated future work:** Thin — one line noting that *"explaining remaining decision deviations... warrants further investigation in subsequent research."* No broader agenda.

**Vs. our work:** The most important comparator in the whole set on the "how rigorous is your robustness claim" axis. NQC-DDQN's interval-bound-propagation approach gives a **certified** guarantee (the correct action's Q-value provably dominates, given the bounded-error assumption) — categorically stronger than UA-SAC's empirical `min` over M=5 sampled Eve locations, which is a Monte Carlo estimate with no formal guarantee at all. This is the single clearest citation to use when honestly scoping UA-SAC's robustness claim in Discussion/Limitations: *"unlike certified approaches such as [Zhao et al.], UA-SAC's robustness is empirical, estimated via a finite number of worst-case samples rather than a formally verified bound."* Also worth noting: their bounded-error model (`‖Δp‖ ≤ ε`) is a fixed, known uncertainty radius — closer to classical robust optimization than to our randomized-σ training distribution.

**Why it's in our lit review:** Sharpest available contrast on rigor. Cite it specifically where the paper claims robustness, to preempt the "is this actually guaranteed?" question a careful reviewer will ask — better to raise and honestly answer it ourselves using this exact comparator than have a reviewer raise it first.

## 22. Liu, Wen, Zhang, Zhu, Zhang, Nie & Kang, 2024 — Learning-Based Power Control for Secure Covert Semantic Communication

**Citation:** Y. Liu, J. Wen, Z. Zhang, K. Zhu, Y. Zhang, J. Nie, and J. Kang, "Learning-based Power Control for Secure Covert Semantic Communication," arXiv:2407.07475, 2024 (revised 2025).
**Link:** https://arxiv.org/abs/2407.07475
**Source read:** Full text (arXiv HTML, earlier this session) plus local PDF re-confirmed this session — consistent content, including the term **"friendly jammer,"** used explicitly by the authors.

**What it does:** Covert semantic communication — hiding not just message content but the *fact that communication is happening* from a warden performing statistical hypothesis testing — using a power regulator plus **a friendly jammer**, jointly optimized via SAC to balance semantic transmission quality (measured via BLEU score on reconstructed data) against covertness and energy constraints.

**Core mechanism:** SAC jointly controls transmitter power and friendly-jammer power. Benchmarked directly against PPO and a Generative Diffusion Model (GDM)-based algorithm — SAC converges faster than PPO (PPO needs ~3500 iterations to catch up) and outperforms both, attributed explicitly to SAC's entropy-maximization property helping it escape suboptimal solutions faster.

**Eve/CSI assumption:** No eavesdropper location model — the adversary here is a covert-communication "warden" performing statistical detection, not a PLS eavesdropper decoding content. Assumes perfect CSI between legitimate transmitter/receiver; the warden is modeled only via detection-error-probability constraints, not a channel/location model at all.

**Their stated limitations:** Text-only semantic communication demonstrated ("for simplicity, taking text SemCom as an example"), other modalities claimed generalizable but not empirically validated. Fixed node geometry in all experiments — no mobility.

**Their stated future work — verbatim:** *"we will explore ways to integrate the concept of the Mixture of Experts (MoE) system into our covert SemCom framework, ensuring reliable communication across varied network conditions."*

**Vs. our work:** The closest SAC-plus-jammer methodological analog to our own algorithm outside the baseline itself — same base algorithm (SAC), same two-role structure (a transmitter and a deliberately-adversarial-to-the-attacker jammer, jointly power-controlled). Genuinely useful evidence that SAC is an actively current, non-arbitrary choice for exactly this kind of jammer-power-control problem, published the same year work on this project's Phase 1 began. Key structural difference: their "friendly jammer" is a single dedicated role distinguishing transmitter-power from jammer-power explicitly in the action space; ours has no such distinction — every AP's power is a single undifferentiated continuous output, serving or jamming determined only by the (fixed) association step.

**Why it's in our lit review:** Strongest available "SAC is a live, current choice for this exact class of problem" citation — not physical layer secrecy, but structurally the nearest thing to our own algorithm-plus-role setup found anywhere in the 35-paper set.

## 24. Tanabe, Sato, Fukuchi, Sakuma & Akimoto, 2022 — Max-Min Off-Policy Actor-Critic Method Focusing on Worst-Case Robustness to Model Misspecification (M2TD3)

**Citation:** T. Tanabe, R. Sato, K. Fukuchi, J. Sakuma, and Y. Akimoto, "Max-Min Off-Policy Actor-Critic Method Focusing on Worst-Case Robustness to Model Misspecification," in *Proc. NeurIPS*, 2022 (arXiv:2211.03413).
**Link:** https://arxiv.org/abs/2211.03413
**Source read:** Full PDF, real text layer.

**What it does:** Sim-to-real robust RL — trains a policy in simulation to perform well not just on average, but in the **worst case** across a predefined set of possible real-world parameter values (friction, mass, etc.), so that transferring from sim to real doesn't catastrophically fail if the real world lands at an unlucky point in that uncertainty set. Not wireless, not PLS — general continuous-control robotics (MuJoCo, 19 tasks).

**Core mechanism:** M2TD3 — Max-Min TD3. Formulates the problem as a **tri-level optimization**: actor, critic, and an uncertainty-parameter selector, solved via **simultaneous gradient ascent-descent** — the uncertainty parameter is adversarially *learned*, not sampled, actively searching for the worst-case parameter value as training proceeds. Critically, the uncertainty parameter is fed directly into the **critic network as an input**, conceptually the same move as our ρ being appended to the state vector — both approaches make the network's value estimate explicitly conditional on the level of uncertainty being faced.

**Eve/CSI assumption:** Not applicable — general robust RL, no wireless/security content.

**Their stated limitations:** Framed as contributions rather than named gaps; ablation studies show each component (critic-input uncertainty parameter, simultaneous gradient ascent-descent, stabilization techniques) contributes measurably, implying removing any one degrades worst-case performance — an implicit acknowledgment that the method is not robust to being simplified.

**Their stated future work:** None beyond the acknowledgments — no forward-looking research agenda in the conclusion.

**Vs. our work:** The closest RL-theory analog to UA-SAC's core mechanism, and worth being precise about the actual difference: M2TD3 *learns* the worst-case uncertainty parameter via an adversarial min-max game (gradient ascent-descent between policy and an uncertainty-selecting adversary) — genuinely finding the worst case through optimization. UA-SAC instead *samples* M=5 candidate Eve locations from a known Gaussian and takes the empirical minimum — approximating the worst case through Monte Carlo draws, not searching for it adversarially. M2TD3's approach is more rigorous (actually optimizes toward the worst case) but requires a differentiable, learnable adversary; ours is simpler and cheaper (no adversary network to train and stabilize) but only as good as the M samples happen to cover. Worth stating explicitly: this is a real methodological choice we made, not an oversight — M2TD3-style adversarial worst-case search is a legitimate future-work direction if M=5 sampling turns out insufficient.

**Why it's in our lit review:** The single best piece of RL theory explaining *why* training against a worst-case signal (rather than an average one) is a principled, established approach — not something invented for this project. Also the clearest way to state, precisely, what UA-SAC's simplification actually costs relative to the more rigorous adversarial alternative.

## 25. Lanier, McAleer, Baldi & Fox, 2022 — Feasible Adversarial Robust Reinforcement Learning for Underspecified Environments (FARR)

**Citation:** J. B. Lanier, S. McAleer, P. Baldi, and R. Fox, "Feasible Adversarial Robust Reinforcement Learning for Underspecified Environments," in *Proc. NeurIPS*, 2022 (arXiv:2207.09597).
**Link:** https://arxiv.org/abs/2207.09597
**Source read:** Full PDF, real text layer.

**What it does:** Addresses a specific failure mode of robust RL: if you pick your uncertainty set too narrowly, the agent is vulnerable to reasonable conditions you didn't cover; too broadly, and some conditions in the set are literally infeasible (no policy can succeed), so training against them wastes capacity and makes the agent needlessly conservative. FARR **implicitly defines the feasible parameter set** as whichever conditions an agent could plausibly succeed at given enough training, rather than requiring the researcher to hand-specify a "reasonable" uncertainty range.

**Core mechanism:** A two-player zero-sum game — an adversary proposes environment parameters, the protagonist policy tries to perform well; solved for an approximate Nash equilibrium using PSRO (Policy-Space Response Oracles), which jointly produces an adversarial distribution over *feasible* parameters and a policy robust across that distribution.

**Eve/CSI assumption:** Not applicable — general robust RL (gridworld + 3 MuJoCo environments), no wireless content.

**Their stated limitations — verbatim, and a useful nuance:** *"our current method for optimizing FARR does not directly optimize the adversary, instead relying on random search and the PSRO restricted game solution."* Worth noting precisely: even this "adversarial" method leans on **random search** for part of its solution, not a fully learned/differentiable adversary — which narrows the gap between FARR's approach and our own random-sampling-based worst-case estimate more than the "adversarial vs. sampling" framing might suggest at first glance.

**Their stated future work:** If high-dimensional adversary best-responses with feasibility estimates become sample-efficiently optimizable, FARR could offer "a prescribable solution to avoid the manual creation of complex rules to limit robust RL adversaries" — i.e., automating what is currently a hand-tuned uncertainty range.

**Vs. our work:** Directly justifies our choice to bound σ∈[0,10]m rather than train against unbounded worst-case uncertainty. FARR's whole premise is that unbounded/unfeasible worst-case training makes agents needlessly conservative — training UA-SAC against, say, σ up to 50m (essentially "Eve could be anywhere, including on top of an AP") would likely produce exactly the over-conservative failure mode FARR is built to avoid. Our fixed, physically-motivated bound (σ_max=10m, a real fraction of the 50m map) is a manual version of what FARR tries to make automatic — worth stating that connection explicitly rather than leaving the σ_max choice looking arbitrary.

**Why it's in our lit review:** The clearest citation for defending *why* σ is bounded at all rather than trained against an unlimited worst case — FARR's entire contribution is a formal treatment of exactly that design question.

## 26. Chen, Hu, Jin, Li & Wang, 2022 — Understanding Domain Randomization for Sim-to-Real Transfer

**Citation:** X. Chen, J. Hu, C. Jin, L. Li, and L. Wang, "Understanding Domain Randomization for Sim-to-Real Transfer," in *Proc. ICLR*, 2022 (arXiv:2110.03239).
**Link:** https://arxiv.org/abs/2110.03239
**Source read:** Full PDF, real text layer.

**What it does:** Pure theory paper explaining *why* domain randomization (training across a distribution of simulated conditions) transfers well to the real world, even with zero real-world training samples. Frames domain randomization as an oracle over a **Latent MDP** with a *uniform* initialization distribution over simulator parameters, and proves bounds on how much value is lost (the "sim-to-real gap") relative to the true optimal real-world policy.

**Core mechanism:** Central result: the sim-to-real gap can be as small as `o(H)` (H = interaction horizon) when the randomized simulator class is finite or satisfies a smoothness condition — with a matching lower bound showing these conditions are *necessary*, not just convenient. Also shows history-dependent (memory-based) policies matter for the bound to hold.

**Eve/CSI assumption:** Not applicable — pure sim-to-real RL theory, no wireless content.

**Their stated limitations:** The bound requires the simulator class to be finite or smooth — domain randomization over an unbounded or pathological parameter space isn't covered by their guarantee.

**Their stated future work:** *"We hope our formulation and analysis can provide insight to design more efficient algorithms for sim-to-real transfer in the future."* General, no specific named direction.

**Vs. our work — the precise, now-confirmed distinction:** This paper explicitly frames domain randomization around the **"uniform sampling nature"** of the parameter distribution — meaning the training signal is fundamentally an *average*-case objective over the randomized simulators, not a worst-case one. Our `sigma_range=(0,10)` random draw per episode is structurally domain randomization in exactly this sense (uniform sampling over σ), but our reward is *not* just scored against whatever σ was drawn that episode — it's the worst case over M=5 Eve samples *given* that σ. So we're layering a worst-case objective (Tanabe/Lanier-style) on top of a domain-randomization-style uniform parameter sampling (this paper's style) — two different mechanisms operating at two different levels of our design, worth being precise about rather than conflating.

**Why it's in our lit review:** Background theoretical justification for why training across randomized σ generalizes at all — but use it narrowly. It explains why sampling σ uniformly per episode is a sound way to get one policy that works across all uncertainty levels; it does *not* explain or justify the worst-case-over-M-Eve-samples part of R*, which needs Tanabe/Lanier/Wang-Kallus-Sun instead. Don't let one citation do the work of two different ideas.

## 27. Wang, Kallus & Sun, 2023 — Near-Minimax-Optimal Risk-Sensitive Reinforcement Learning with CVaR

**Citation:** K. Wang, N. Kallus, and W. Sun, "Near-Minimax-Optimal Risk-Sensitive Reinforcement Learning with CVaR," in *Proc. ICML*, 2023 (arXiv:2302.03201).
**Link:** https://arxiv.org/abs/2302.03201
**Source read:** Full PDF, real text layer. Pure theory paper — multi-armed bandits and tabular MDPs, regret bounds, no application domain of any kind.

**What it does:** Establishes near-minimax-optimal regret bounds for RL under a **Conditional Value-at-Risk (CVaR)** objective — instead of maximizing average return, maximize the average outcome among the worst τ-fraction of cases. Gives a UCB-style algorithm for bandits and a bonus-driven value-iteration procedure for tabular MDPs, both provably near-optimal.

**Core mechanism:** Formal definition worth quoting exactly since it's the precise theoretical anchor for our own reward: `CVaR_τ(X) := sup_b [b − τ⁻¹E[(b−X)⁺]]`, which for continuous X equals `E[X | X ≤ F_X(τ)]` — **"the average outcome among the worst τ-percent of cases."** As τ→0, CVaR_τ converges toward the single worst-case outcome — i.e., our `R* = min` over M samples is structurally a small-τ, finite-sample CVaR estimate, not an unrelated ad hoc heuristic.

**Eve/CSI assumption:** Not applicable — no application domain at all, pure regret-bound theory.

**Their stated limitations:** Results are minimax-optimal for *constant* τ; scaling behavior for τ→0 (which is closer to our own effectively-near-worst-case regime) is not the paper's focus.

**Their stated future work:** Not a dedicated section — theory paper, contributions stated as the bounds themselves.

**Vs. our work — precise framing, not oversold:** Our `R*` is best understood as an **empirical, finite-sample (M=5) approximation of a small-τ CVaR objective**, not the provably-optimal CVaR estimator this paper constructs. We inherit none of their regret guarantees — different setting entirely (continuous control via SAC vs. tabular MDPs/bandits via value iteration). The honest citation is: "our worst-case reward is a Monte Carlo approximation of the CVaR objective formalized in [this paper], not a novel construct," stated in one sentence, not extended into a claim of shared theoretical guarantees.

**Why it's in our lit review:** The correct formal name and definition for the risk measure our reward function approximates. Cite once, precisely, to give R* a real theoretical name (CVaR-style worst-case objective) rather than leaving it as an unnamed heuristic — but resist the temptation to borrow their regret bounds, which don't transfer to our setting.

## 29. Xing, Qin, Du, Wang & Zhang, 2024 — Deep Reinforcement Learning-Driven Jamming-Enhanced Secure UAV Communications

**Citation:** Z. Xing, Y. Qin, C. Du, W. Wang, and Z. Zhang, "Deep Reinforcement Learning-Driven Jamming-Enhanced Secure Unmanned Aerial Vehicle Communications," *Sensors*, vol. 24, no. 22, article 7328, Nov. 2024.
**Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11598403/
**Source read:** Full PDF, real text layer.

**What it does:** UAV base stations serve legitimate users while dedicated UAV jammers disrupt eavesdroppers, with the explicit objective of **maximizing the minimum secrecy rate across users** — a genuine max-min/worst-case formulation, not an average-case one. Jointly optimizes user association, UAV trajectory, and power allocation, comparing single-agent SAC, TD3, and multi-agent SAC (MASAC).

**Core mechanism:** Formulates the joint optimization as an MDP due to its complexity, then benchmarks three DRL algorithms directly against each other. Result, stated plainly: SAC is the most stable and outperforms TD3 on cumulative reward; MASAC accumulates more reward but at higher training-time cost. **Critically, user association is jointly learned as part of the same DRL problem** — not a fixed pre-step like in our own system.

**Eve/CSI assumption:** Not stated as a primary contribution in the excerpt reviewed — the paper's novelty is the max-min objective and joint trajectory/association/power optimization via multi-agent DRL, not eavesdropper-uncertainty modeling specifically.

**Their stated limitations:** MASAC's higher time complexity during training is named directly as a real cost of the multi-agent approach, not glossed over.

**Their stated future work — verbatim:** *"we will explore secure transmission for integrated sensing and communication (ISAC)-enabled UAV networks... These single-agent and multi-agent DRL algorithms will be exploited to solve the user association variables, UAV trajectory planning, and power allocation problems in UAV-ISAC networks."*

**Vs. our work — two distinct, both important connections:** (1) Their **"maximize the minimum secrecy rate"** objective is a max-min formulation over *users*, structurally the same worst-case philosophy as our `R* = min` over *Eve samples* — different thing being minimized over, same underlying principle that averages hide catastrophic minorities. (2) **This paper is concrete proof that jointly learning user association alongside power control via DRL is tractable and already done in the literature** — directly undercuts any implicit assumption that fixing association was *necessary* rather than a simplifying choice. Worth citing exactly here when discussing our own AP-association limitation: "joint learning of association and power allocation has been demonstrated in DRL-based secure UAV systems [Xing et al. 2024]; extending UA-SAC similarly remains future work."

**Why it's in our lit review:** Does double duty — reinforces that max-min/worst-case objectives are an active, current design choice in DRL-for-secrecy (not unique to us), and gives concrete, citable evidence that our own AP-association gap is genuinely closeable, since someone else has already closed the equivalent gap in a UAV setting.

## 30. Tripathi, Kundu, Yadav, Bansal, Claussen & Ho, 2024 — Joint Transmit and Jamming Power Optimization for Secrecy in Energy Harvesting Networks

**Citation:** S. Tripathi, C. Kundu, A. Yadav, A. Bansal, H. Claussen, and L. Ho, "Joint Transmit and Jamming Power Optimization for Secrecy in Energy Harvesting Networks: A Reinforcement Learning Approach," arXiv:2407.17435, 2024 (revised 2025).
**Link:** https://arxiv.org/abs/2407.17435
**Source read:** Full PDF, real text layer.

**What it does:** Source and destination nodes both run on energy-harvesting batteries with limited capacity; the destination has full-duplex jamming capability. Jointly optimizes transmit power and jamming power over the network's *entire operational lifetime* to maximize cumulative secure bits transmitted before the battery-limited network dies.

**Core mechanism — worth being precise, this is not deep RL:** Formulated as an **infinite-horizon MDP**, solved with classical **tabular Policy Iteration** (OJPA — optimal joint power allocation) — no neural networks anywhere. A reduced-state sub-optimal variant (RSJPA) cuts computation by ~75% using only 50% of states, benchmarked against greedy (GA) and naive (NA) baselines. This is meaningfully different from every other RL paper in our set: genuinely sequential/multi-step (battery state evolves over time slots), and tabular rather than function-approximation-based — closer to classical dynamic programming than to modern DRL.

**Eve/CSI assumption — explicit and defended, not assumed away:** Requires full global CSI including the eavesdropper's channel. The authors don't hide this — they explicitly justify it with three scenarios where it's realistic: an *active* eavesdropper being monitored, dual-role nodes (legitimate receiver for some transmissions, eavesdropper for others, so its CSI is naturally available), and military settings where "all other nodes" are treated as potential eavesdroppers. They cite this as "a common assumption" in the secure-communication literature, with eight supporting references.

**Their stated limitations:** None named as unresolved beyond the inherent optimal/sub-optimal complexity tradeoff already addressed by RSJPA.

**Their stated future work:** Not a dedicated forward-looking section — conclusion restates comparative results.

**Vs. our work:** Useful contrast on two axes. (1) Methodological: tabular RL vs. our deep RL (SAC) — theirs is more analytically tractable but doesn't scale past small discretized state spaces the way a neural-network policy does, which is exactly why our multi-AP, continuous-power setting needs SAC instead. (2) Assumption-level: they explicitly defend *why* full Eve CSI is sometimes realistic (active/monitored/dual-role eavesdroppers) — genuinely useful for our own Related Work, since it shows the "Eve CSI known" assumption isn't universally naive, just inapplicable to our specific passive-eavesdropper threat model. Worth citing to sharpen our own scoping: "known-CSI assumptions are defensible under active or monitored-eavesdropper threat models [Tripathi et al. 2024]; our contribution targets the complementary case of a purely passive eavesdropper."

**Why it's in our lit review:** Best available citation for precisely scoping *when* the "Eve CSI is known" assumption is and isn't reasonable — turns a potential weakness (why do so many papers assume known CSI?) into a clean, defensible boundary condition for our own contribution.

## 32. Miao, Song, Li, Li, Liu, Shao & Wang, 2026 — RIS-Aided Physical Layer Security with Imperfect CSI: A Robust Model-Driven Deep Learning Approach

**Citation:** R. Miao, Z. Song, Y. Li, X. Li, L. Liu, G. Shao, and B. Wang, "RIS-Aided Physical Layer Security with Imperfect CSI: A Robust Model-Driven Deep Learning Approach," *Entropy*, vol. 28, no. 4, article 457, Apr. 2026.
**Link:** https://www.mdpi.com/1099-4300/28/4/457
**Source read:** Full PDF, real text layer.

**What it does:** RIS-aided multi-user MISO secure system against multiple eavesdroppers under imperfect Eve CSI, solved with a **model-driven deep learning** approach — not reinforcement learning. Unrolls a gradient descent-ascent optimization algorithm into a GRU-aided "deep unfold" network with learnable, adaptive per-iteration step sizes.

**Core mechanism:** Standard deep-unfolding networks fix the number of iterations upfront, a hyperparameter usually tuned empirically. This paper's contribution is letting a GRU generate the step size adaptively at each iteration, so the network can dynamically decide how many iterations it actually needs based on convergence conditions, rather than a fixed count.

**Eve/CSI assumption:** Imperfect (bounded-error) CSI for the eavesdroppers explicitly — the paper's whole premise. Worth flagging a passage from their own related-work discussion, since it's directly relevant to our own future-work framing: they note that in practical deployment, *"the location of RIS may change... the locations of users and potential eavesdroppers can also change,"* and that this dynamism itself "brings challenges to deep unfold network design" — i.e., even this paper acknowledges its own robust design doesn't yet handle *dynamic* (mobile) eavesdropper locations, only static imperfect-CSI uncertainty.

**Their stated limitations:** Implicit rather than a dedicated section — the mobility/dynamics passage above functions as an acknowledged gap in their own approach and in the deep-unfolding paradigm generally (fixed-iteration-count networks handle a fixed problem difficulty poorly).

**Their stated future work:** Not a dedicated section — conclusion restates the contribution (GRU-driven adaptive iteration count beats fixed-iteration deep unfolding and non-robust baselines on weighted sum secrecy rate).

**Vs. our work:** The cleanest "different ML paradigm, same problem" comparator in the whole set. Model-driven deep unfolding requires a differentiable, closed-form-derivable optimization algorithm to unroll in the first place (gradient descent-ascent here) — it works because RIS beamforming has that structure. UA-SAC needs no such differentiable surrogate; it learns the robust policy purely through interaction, which is precisely why RL (not unfolding) is the right tool for cooperative jamming, where the joint discrete-association/continuous-power/multi-AP-interference structure doesn't reduce to a clean unrollable optimization the way single-RIS beamforming does.

**Why it's in our lit review:** Sharpest "why RL and not something else" comparator — shows a real, current (2026), legitimate alternative ML paradigm for handling imperfect Eve CSI, and gives a precise, honest reason why it doesn't fit our problem structure as well as RL does.

## 33. Zhou, You, Zhou, Xing & Zhang, 2026 — Near-Field Physical Layer Security: Robust Beamforming under Location Uncertainty

**Citation:** C. Zhou, C. You, C. Zhou, C. Xing, and J. Zhang, "Near-Field Physical Layer Security: Robust Beamforming under Location Uncertainty," *IEEE Trans. Wireless Commun.*, vol. 25, pp. 17384–17398, 2026 (arXiv:2601.13549).
**Link:** https://arxiv.org/abs/2601.13549
**Source read:** Full PDF, real text layer — read in two passes across this project (methodology earlier, conclusion/future-work this session).

**What it does:** A base station with an extremely large-scale antenna array (near-field regime) serves legitimate users while defending against eavesdroppers whose *location* is imperfectly known — closest non-RL competitor to our exact problem statement (secrecy under Eve location uncertainty), solved via analytical robust beamforming rather than learning.

**Core mechanism:** Discovers the **near-field angular-error amplification effect** — the same Cartesian positional uncertainty produces a *larger* angular error the closer Eve actually is to the array, degrading conventional robust beamforming disproportionately at close range. Solves this with a two-stage method: partition the uncertainty region into fan-shaped sub-regions (bounding Taylor-approximation error within each), then solve a refined linear-matrix-inequality (LMI) reformulation per sub-region.

**Eve/CSI assumption:** Bob's location perfectly known; Eve's location uncertain, modeled as Gaussian positional error, LoS-dominant near-field channel, Eve assumed capable of perfectly cancelling multi-user interference before decoding (a conservative/worst-case receiver assumption in Eve's favor).

**Their stated limitations:** The conservative error-bound approach becomes loose ("conservative") when range errors are large, tightening transmit power constraints more than strictly necessary at close range — a direct consequence of the angular-error amplification effect they discovered.

**Their stated future work — verbatim, two items:** (1) Extension to a scenario requiring SCA and S-Procedure techniques — *"the detailed derivations, however, are mathematically intricate and thus are left for future work."* (2) Broader extensions named in the conclusion: XL-IRS-aided systems, space-air-ground integrated networks, and beam tracking for ISAC.

**Vs. our work:** The single most direct non-RL competitor in the entire 35-paper set — same exact problem (secrecy under Eve location uncertainty), entirely different mechanism (closed-form geometric/LMI robust optimization vs. our worst-case-sampled RL). Their headline finding (angular-error amplification — uncertainty effects are geometry-dependent, not uniform) is a genuine, sophisticated critique applicable to our own design: our ρ = σ/D_max is a single global scalar, identical regardless of where Eve actually sits relative to the APs. Worth naming explicitly as a limitation/future-work item: UA-SAC's uncertainty signal is currently geometry-blind, and Zhou et al.'s result suggests it shouldn't be.

**Why it's in our lit review:** Does the most work of any single citation on the "here's exactly who else solves our problem, and here's what they teach us about our own blind spot" front. The angular-error amplification finding is worth a dedicated sentence in Discussion/Future Work, not just a passing citation.

## 34. Xiao, Hu, Li, Wang & Yang, 2025 — Robust Full-Space Physical Layer Security for STAR-RIS-Aided Wireless Networks: Eavesdropper with Uncertain Location and Channel

**Citation:** H. Xiao, X. Hu, A. Li, W. Wang, and K. Yang, "Robust Full-Space Physical Layer Security for STAR-RIS-Aided Wireless Networks: Eavesdropper With Uncertain Location and Channel," *IEEE Trans. Wireless Commun.*, vol. 24, pp. 7206–7220, 2025 (arXiv:2503.12233).
**Link:** https://arxiv.org/abs/2503.12233
**Source read:** Full PDF, real text layer.

**What it does:** Secures a STAR-RIS-assisted network (simultaneous transmitting and reflecting — covers 360° around the surface, both "reflection" and "transmission" regions) against an eavesdropper whose location is uncertain within *either* region. Derives an asymptotic security-rate expression via large-system analysis, assuming the base station has only **statistical CSI** of the eavesdropper — genuinely no location point-estimate at all, a probabilistic region model instead.

**Core mechanism:** Models which side of the STAR-RIS Eve occupies with a Bernoulli random variable (probability she's in the Reflection region vs. Transmission region), derives the asymptotic average security rate in the large-antenna/large-RIS-element limit (where randomness from Eve's exact channel realization washes out), then jointly optimizes active (BS) and passive (RIS phase-shift) beamforming via an MMSE + cross-entropy-optimization iterative algorithm to solve the resulting non-convex, discrete-phase-shift problem.

**Eve/CSI assumption:** Only statistical CSI available — no location point-estimate, no Gaussian positional-error model (unlike Zhou et al.'s near-field paper) — genuinely closer to "we don't know where Eve is, only which broad region she's plausibly in and her channel statistics."

**Their stated limitations:** Explicitly scoped to a single eavesdropper under the STAR-RIS energy-splitting (ES) protocol — acknowledged directly, not left implicit.

**Their stated future work — verbatim:** *"it would be valuable to explore alternative protocols of STAR-RIS, such as mode selection and time switching, as well as to extend the scenarios with multiple eavesdroppers. Additionally, integrating active or hybrid..."* (RIS elements, text cut off in source but direction is clear — richer RIS hardware modes and multi-eavesdropper extension).

**Vs. our work:** The most philosophically aligned non-RL competitor — like us, they genuinely don't assume a location point-estimate for Eve, working instead from region-level statistics. The mechanism is entirely different (closed-form asymptotic security rate + MMSE/cross-entropy optimization vs. our RL-learned worst-case policy), and their approach requires the large-system (many antennas/RIS elements) limit to be analytically tractable — a scaling assumption our small-N (4 AP) setting doesn't share. Worth stating plainly: their method needs *scale* for its guarantees to kick in; ours needs *training data* (episodes) instead — different resource each approach trades on.

**Why it's in our lit review:** Second-strongest direct competitor after Zhou et al. — same "no location point-estimate" philosophy, different mechanism, different regime (large-system asymptotics vs. small-network RL). Together with Zhou et al., these two give the Related Work section genuine, current, non-RL competitors to differentiate against, not just older RL/beamforming papers.

## 35. Niu, Xiao, Lei, Chen, Xiao & Yuen, 2025 — A Survey on Artificial Noise for Physical Layer Security: Opportunities, Technologies, Guidelines, Advances, and Trends

**Citation:** H. Niu, Y. Xiao, X. Lei, J. Chen, Z. Xiao, M. Li, and C. Yuen, "A Survey on Artificial Noise for Physical Layer Security: Opportunities, Technologies, Guidelines, Advances, and Trends," arXiv:2507.06500, 2025.
**Link:** https://arxiv.org/abs/2507.06500
**Source read:** Full PDF, real text layer.

**What it does:** Comprehensive, current survey specifically on Artificial Noise (AN) as a PLS technique — evolution, modeling, applications, and a dedicated Future Directions section. Broader than our scope (covers many AN application scenarios: ISAC, space-air-ground networks, RIS combinations, vehicle comms, etc.) but with real, direct relevance in its CSI-uncertainty coverage.

**Core mechanism — the directly relevant part:** Covers **AN design without Eve's CSI at all** — hyperplane clustering and machine-learning-based artificial-noise-elimination (ANE) countermeasures from Eve's *attacking* side, and separately, the **statistical-CSI** case for the defender, explicitly named as *"a more operational assumption... which may occur when the location of Eve can be confirmed."* Notes AN still helps under statistical-only CSI, just less than with full instantaneous CSI — a quantified, not just claimed, secrecy/energy-efficiency gap.

**Eve/CSI assumption:** Survey-level — covers the full spectrum from full instantaneous CSI down to zero-CSI-with-ML-countermeasures, more thorough on this specific axis than any other survey in our set.

**Their stated limitations:** Named directly in their own Future Direction subsection: approaches that work without Eve's CSI "often come with high computational complexity and deviate from the original intention of AN (to provide secure communication with low complexity)."

**Their stated future work — verbatim, and directly actionable for us:** *"practical AN optimization algorithms with low computational complexity... may involve developing novel quantization standards for situations where Eve's CSI is unavailable and pursuing closed-form solutions for AN design. These efforts could bridge the gap between theoretical developments and practical implementation."* Also flags ISAC integration, space-air-ground/6G high-mobility scenarios (Doppler-robust AN), and combination with numerous other technologies (RIS, fluid antennas, cell-free MIMO) as open directions.

**Vs. our work:** Directly names the exact tension our own paper resolves differently: existing no-CSI/statistical-CSI AN approaches are flagged as too computationally expensive for practical low-complexity deployment. UA-SAC sidesteps this specific tension — training is expensive, but *inference* (a trained SAC policy forward pass) is cheap, unlike closed-form per-instance optimization under uncertainty. Worth a sentence: "unlike AN approaches under statistical CSI, which trade complexity for robustness at inference time [Niu et al.], UA-SAC shifts that cost entirely to training, leaving inference as a single fast forward pass."

**Why it's in our lit review:** Best available "state of the field" citation for the AN/friendly-interference literature specifically, current (2025) and comprehensive, with a future-work agenda that our own low-inference-cost RL approach directly and favorably contrasts against.
