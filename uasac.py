"""
uasac.py  —  Uncertainty-Aware SAC (UA-SAC)

Subclasses Stable Baselines3's SAC with two modifications:

  1. ρ-dependent entropy scaling:
       α_eff(ρ) = α_base · (1 + β·ρ)
     ρ is the last element of every observation (uncertainty ratio = σ/D_max).
     High uncertainty → higher entropy bonus → broader jamming coverage.
     α_base is still auto-tuned by SAC's standard dual-objective update.

  2. History logging:
     Stores α_base and α_eff per gradient step for Plot 4 (algorithm proof).
     Saved to results/uasac_ent_coef_history.npz after training.
"""

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class UASAC(SAC):
    """
    UA-SAC: SAC with uncertainty-scaled entropy regularization.

    Extra constructor args:
        beta (float): scaling coefficient for entropy modulation.
                      α_eff = α_base * (1 + beta * rho_mean).
                      beta=0 reduces to standard SAC.
    """

    def __init__(self, *args, beta: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self._alpha_base_hist = []   # α_base per gradient step
        self._alpha_eff_hist  = []   # α_eff  per gradient step
        self._rho_hist        = []   # mean ρ per gradient step

    # ------------------------------------------------------------------ #
    # Core training loop override
    # ------------------------------------------------------------------ #

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):

            # ── Sample replay buffer ───────────────────────────────────
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            # ── Per-sample ρ from batch observations (last element) ────
            rho_batch = replay_data.observations[:, -1].detach().reshape(-1, 1)
            self._rho_hist.append(float(rho_batch.mean().item()))

            # ── Base entropy coefficient (auto-tuned scalar) ───────────
            if self.ent_coef == "auto":
                alpha_base = th.exp(self.log_ent_coef.detach())
            else:
                alpha_base = self.ent_coef_tensor

            # ── Effective entropy = base × (1 + β·ρ), per sample ───────
            # Each transition is scaled by its OWN stored ρ, not a
            # batch-average — so entropy scaling actually tracks each
            # transition's own uncertainty level rather than one shared
            # scalar for the whole batch.
            alpha_eff = alpha_base * (1.0 + self.beta * rho_batch)

            # Log both for Plot 4 (batch mean, for a single scalar curve)
            alpha_eff_mean = alpha_eff.mean()
            self._alpha_base_hist.append(float(alpha_base.item()))
            self._alpha_eff_hist.append(float(alpha_eff_mean.item()))
            ent_coefs.append(float(alpha_eff_mean.item()))

            # ── Update base entropy coefficient (standard SAC dual obj) ─
            if self.ent_coef == "auto":
                with th.no_grad():
                    _, log_prob_ent = self.actor.action_log_prob(
                        replay_data.observations
                    )
                ent_coef_loss = -(
                    self.log_ent_coef
                    * (log_prob_ent + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # ── Critic update (uses α_eff in Q-target) ────────────────
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q = th.cat(
                    self.critic_target(
                        replay_data.next_observations, next_actions
                    ),
                    dim=1,
                )
                next_q, _ = th.min(next_q, dim=1, keepdim=True)
                next_q    = next_q - alpha_eff * next_log_prob.reshape(-1, 1)
                target_q  = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q
                )

            current_q   = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(
                F.mse_loss(cq, target_q) for cq in current_q
            )
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # ── Actor update (uses α_eff in policy gradient) ───────────
            actions_pi, log_prob = self.actor.action_log_prob(
                replay_data.observations
            )
            log_prob = log_prob.reshape(-1, 1)

            q_pi = th.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            min_q_pi, _ = th.min(q_pi, dim=1, keepdim=True)
            actor_loss   = (alpha_eff * log_prob - min_q_pi).mean()

            actor_losses.append(actor_loss.item())
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # ── Soft update target networks ────────────────────────────
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )
                batch_norm = getattr(self, "batch_norm_stats", [])
                batch_norm_tgt = getattr(self, "batch_norm_stats_target", [])
                if batch_norm:
                    polyak_update(batch_norm, batch_norm_tgt, 1.0)

        # ── Bookkeeping ────────────────────────────────────────────────
        self._n_updates += gradient_steps
        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard"
        )
        self.logger.record("train/ent_coef_eff",  np.mean(ent_coefs))
        self.logger.record("train/ent_coef_base",
                           float(alpha_base.item()))
        self.logger.record("train/actor_loss",  np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if ent_coef_losses:
            self.logger.record(
                "train/ent_coef_loss", np.mean(ent_coef_losses)
            )

    # ------------------------------------------------------------------ #
    # Save entropy history after training
    # ------------------------------------------------------------------ #

    def save_ent_history(self, path: str = "results/uasac_ent_history.npz"):
        np.savez(
            path,
            alpha_base=np.array(self._alpha_base_hist),
            alpha_eff=np.array(self._alpha_eff_hist),
            rho_mean=np.array(self._rho_hist),
        )
        print(f"Entropy history saved → {path}")
