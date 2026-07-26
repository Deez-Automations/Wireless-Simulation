import gymnasium as gym
import numpy as np
from gymnasium import spaces


class WirelessJammingEnv(gym.Env):
    """
    Cooperative Friendly Jamming environment.

    Implements the system model from:
        Hoseini et al., "Cooperative Jamming for Physical Layer Security
        Enhancement Using Deep Reinforcement Learning", arXiv:2403.10342, 2024.

    Phase 1 (default):
        csi_noise_std > 0  — agent observes noisy Eve location.
        M=1, augment_rho=False — 14-dim observation, point-estimate reward.
        Backward compatible with all saved Phase 1 models.

    Phase 2 (UA-SAC):
        augment_rho=True   — appends ρ = σ/D_max as 15th observation element.
        M=5                — reward = min secrecy over M sampled Eve locations.
        sigma_range=(0,10) — σ sampled uniformly each episode for universal policy.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_aps: int = 4,
        num_users: int = 2,
        num_eves: int = 1,
        max_power: float = 1.0,
        noise_power_dbm: float = -85.0,
        freq_hz: float = 2.4e9,
        path_loss_exp: float = 2.0,
        csi_noise_std: float = 0.0,
        map_size: float = 50.0,
        M: int = 1,
        beta: float = 1.0,
        sigma_range=None,
        augment_rho: bool = False,
    ):
        super().__init__()

        self.num_aps       = num_aps
        self.num_users     = num_users
        self.num_eves      = num_eves
        self.max_power     = max_power
        self.map_size      = map_size
        self.gamma         = path_loss_exp
        self.csi_noise_std = csi_noise_std
        self.M             = M
        self.beta          = beta
        self.sigma_range   = sigma_range
        self.augment_rho   = augment_rho

        self.current_sigma = csi_noise_std
        self.rho           = csi_noise_std / map_size

        self.noise_power = 10 ** ((noise_power_dbm - 30) / 10)

        c = 3e8
        self.wavelength = c / freq_hz

        self.action_space = spaces.Box(
            low=0.0, high=max_power,
            shape=(num_aps,), dtype=np.float32
        )

        n_coords = (num_aps + num_users + num_eves) * 2
        if augment_rho:
            n_obs    = n_coords + 1
            obs_low  = np.zeros(n_obs, dtype=np.float32)
            obs_high = np.concatenate([
                np.full(n_coords, map_size, dtype=np.float32),
                np.array([1.0], dtype=np.float32),
            ])
        else:
            n_obs    = n_coords
            obs_low  = np.zeros(n_obs, dtype=np.float32)
            obs_high = np.full(n_obs, map_size, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        self.ap_locs   = None
        self.user_locs = None
        self.eve_locs  = None
        self.assoc     = None

    # ------------------------------------------------------------------ #
    # Friis received power
    # ------------------------------------------------------------------ #

    def _received_power(self, p_tx: float, dist: float) -> float:
        d = max(dist, 0.1)
        return p_tx * ((self.wavelength / (4 * np.pi)) ** 2) * (d ** (-self.gamma))

    # ------------------------------------------------------------------ #
    # SINR
    # ------------------------------------------------------------------ #

    def _sinr(self, receiver_loc: np.ndarray, serving_ap: int,
              powers: np.ndarray) -> float:
        desired_dist = np.linalg.norm(self.ap_locs[serving_ap] - receiver_loc)
        desired_pwr  = self._received_power(powers[serving_ap], desired_dist)

        interference = 0.0
        for nu in range(self.num_aps):
            if nu == serving_ap:
                continue
            d = np.linalg.norm(self.ap_locs[nu] - receiver_loc)
            interference += self._received_power(powers[nu], d)

        return desired_pwr / (interference + self.noise_power)

    # ------------------------------------------------------------------ #
    # Shannon capacity
    # ------------------------------------------------------------------ #

    def _capacity(self, receiver_loc: np.ndarray, serving_ap: int,
                  powers: np.ndarray) -> float:
        return np.log2(1.0 + self._sinr(receiver_loc, serving_ap, powers))

    # ------------------------------------------------------------------ #
    # Secrecy capacity for one user (uses self.eve_locs)
    # ------------------------------------------------------------------ #

    def _secrecy_capacity(self, user_idx: int, powers: np.ndarray) -> float:
        ap_idx   = self.assoc[user_idx]
        user_cap = self._capacity(self.user_locs[user_idx], ap_idx, powers)

        eve_cap = max(
            self._capacity(self.eve_locs[j], ap_idx, powers)
            for j in range(self.num_eves)
        )

        return max(user_cap - eve_cap, 0.0)

    # ------------------------------------------------------------------ #
    # Sum secrecy at a given Eve position (for worst-case reward sampling)
    # ------------------------------------------------------------------ #

    def _sum_secrecy_at(self, powers: np.ndarray,
                        eve_locs_override: np.ndarray) -> float:
        saved          = self.eve_locs
        self.eve_locs  = eve_locs_override
        total = sum(
            self._secrecy_capacity(k, powers) for k in range(self.num_users)
        )
        self.eve_locs  = saved
        return total

    # ------------------------------------------------------------------ #
    # AP–user association (power-aware: re-evaluated against whatever
    # powers are actually scored, not a fixed uniform-max-power guess)
    # ------------------------------------------------------------------ #

    def _associate_users(self, powers: np.ndarray) -> np.ndarray:
        assoc = np.zeros(self.num_users, dtype=int)

        for k in range(self.num_users):
            best_ap  = 0
            best_sec = -np.inf
            for n in range(self.num_aps):
                user_cap = self._capacity(self.user_locs[k], n, powers)
                eve_cap  = max(
                    self._capacity(self.eve_locs[j], n, powers)
                    for j in range(self.num_eves)
                )
                sec = user_cap - eve_cap
                if sec > best_sec:
                    best_sec = sec
                    best_ap  = n
            assoc[k] = best_ap

        return assoc

    # ------------------------------------------------------------------ #
    # Build observation vector
    # ------------------------------------------------------------------ #

    def _build_obs(self) -> np.ndarray:
        obs_eve = self.eve_locs.copy()

        if self.current_sigma > 0.0:
            noise   = np.random.normal(0.0, self.current_sigma, obs_eve.shape)
            obs_eve = np.clip(obs_eve + noise, 0.0, self.map_size)

        parts = [
            self.ap_locs.flatten(),
            self.user_locs.flatten(),
            obs_eve.flatten(),
        ]

        if self.augment_rho:
            parts.append(np.array([self.rho], dtype=np.float32))

        return np.concatenate(parts).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        rng = np.random.default_rng(seed)

        self.ap_locs   = rng.uniform(0, self.map_size, (self.num_aps,   2))
        self.user_locs = rng.uniform(0, self.map_size, (self.num_users, 2))
        self.eve_locs  = rng.uniform(0, self.map_size, (self.num_eves,  2))

        if self.sigma_range is not None:
            self.current_sigma = float(
                np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            )
            self.rho = self.current_sigma / self.map_size
        else:
            self.current_sigma = self.csi_noise_std
            self.rho           = self.csi_noise_std / self.map_size

        return self._build_obs(), {}

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self, action: np.ndarray):
        powers = np.clip(action, 0.0, self.max_power).astype(np.float32)
        self.assoc = self._associate_users(powers)

        if self.M > 1 and self.current_sigma > 0.0:
            secrecies = []
            for _ in range(self.M):
                noise      = np.random.normal(0.0, self.current_sigma,
                                              self.eve_locs.shape)
                eve_sample = np.clip(self.eve_locs + noise, 0.0, self.map_size)
                secrecies.append(self._sum_secrecy_at(powers, eve_sample))
            reward = float(min(secrecies))
        else:
            reward = sum(
                self._secrecy_capacity(k, powers)
                for k in range(self.num_users)
            )

        terminated = True
        truncated  = False

        obs  = self._build_obs()
        info = {
            "sum_secrecy_capacity": reward,
            "powers":     powers.tolist(),
            "user_assoc": self.assoc.tolist(),
            "sigma":      self.current_sigma,
            "rho":        self.rho,
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Evaluate a fixed power allocation (for baselines / testing)
    # ------------------------------------------------------------------ #

    def evaluate_policy(self, powers: np.ndarray) -> dict:
        powers = np.clip(powers, 0.0, self.max_power)
        self.assoc = self._associate_users(powers)

        per_user_secrecy = [
            self._secrecy_capacity(k, powers) for k in range(self.num_users)
        ]
        sum_sec   = sum(per_user_secrecy)
        sec_ratio = sum(1 for s in per_user_secrecy if s > 0) / self.num_users

        eve_cap = []
        for n in range(self.num_aps):
            ec = max(
                self._capacity(self.eve_locs[j], n, powers)
                for j in range(self.num_eves)
            )
            eve_cap.append(ec)

        return {
            "sum_secrecy_capacity": sum_sec,
            "secrecy_ratio":        sec_ratio,
            "sum_eve_capacity":     sum(eve_cap),
            "per_user_secrecy":     per_user_secrecy,
        }
