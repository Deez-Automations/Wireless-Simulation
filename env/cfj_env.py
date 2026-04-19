import gymnasium as gym
import numpy as np
from gymnasium import spaces


class WirelessJammingEnv(gym.Env):
    """
    Cooperative Friendly Jamming environment.

    Implements the system model from:
        Hoseini et al., "Cooperative Jamming for Physical Layer Security
        Enhancement Using Deep Reinforcement Learning", arXiv:2403.10342, 2024.

    Baseline behaviour: agent observes TRUE eavesdropper location.
    With csi_noise_std > 0: agent observes NOISY eavesdropper location
        (your novel contribution — imperfect Eve CSI).

    The environment:
        - Randomly places N APs, K users, J eavesdroppers on a 50x50m grid.
        - Associates each user to the AP that maximises secrecy capacity
          (Eq. 7 of the paper) assuming uniform max power first.
        - Agent chooses transmit power p_n for each AP (continuous, 0–1 W).
        - Reward = sum secrecy capacity across all users (Eq. 6 / 9).
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        num_aps: int = 4,
        num_users: int = 2,
        num_eves: int = 1,
        max_power: float = 1.0,          # Watts
        noise_power_dbm: float = -85.0,  # dBm  (paper: -85 dBm)
        freq_hz: float = 2.4e9,          # Hz   (paper: Wi-Fi 2.4 GHz)
        path_loss_exp: float = 2.0,      # γ    (paper: γ = 2)
        csi_noise_std: float = 0.0,      # metres — 0 = perfect (baseline)
        map_size: float = 50.0,          # metres
    ):
        super().__init__()

        self.num_aps   = num_aps
        self.num_users = num_users
        self.num_eves  = num_eves
        self.max_power = max_power
        self.map_size  = map_size
        self.gamma     = path_loss_exp
        self.csi_noise_std = csi_noise_std

        # Convert noise power from dBm to Watts
        self.noise_power = 10 ** ((noise_power_dbm - 30) / 10)

        # Wavelength λ = c / f
        c = 3e8
        self.wavelength = c / freq_hz

        # -------- action space ---------------------------------------- #
        # Agent controls transmit power of every AP: p_n in [0, max_power]
        self.action_space = spaces.Box(
            low=0.0, high=max_power,
            shape=(num_aps,), dtype=np.float32
        )

        # -------- observation space ------------------------------------ #
        # [AP coords (2*N) | User coords (2*K) | Eve coords (2*J)]
        # Eve coords in the observation may be noisy (contribution).
        total_coords = (num_aps + num_users + num_eves) * 2
        self.observation_space = spaces.Box(
            low=0.0, high=map_size,
            shape=(total_coords,), dtype=np.float32
        )

        # Internal state initialised in reset()
        self.ap_locs   = None   # (N, 2)
        self.user_locs = None   # (K, 2)
        self.eve_locs  = None   # (J, 2)
        self.assoc     = None   # (K,) index of serving AP for each user

    # ------------------------------------------------------------------ #
    # Friis received power (Eq. 1 of paper)
    # ------------------------------------------------------------------ #

    def _received_power(self, p_tx: float, dist: float) -> float:
        """
        Friis: p_r = p_t * (λ / 4π)^2 * (1/d)^γ
        Guard against d=0 with a minimum distance of 0.1 m.
        """
        d = max(dist, 0.1)
        return p_tx * ((self.wavelength / (4 * np.pi)) ** 2) * (d ** (-self.gamma))

    # ------------------------------------------------------------------ #
    # SINR at a receiver (Eq. 2 / 3 of paper)
    # ------------------------------------------------------------------ #

    def _sinr(self, receiver_loc: np.ndarray, serving_ap: int,
              powers: np.ndarray) -> float:
        """
        SINR at receiver from serving_ap, with all other APs as interference.

        SINR_{n,k} = p_n^r / (Σ_{ν≠n} p_ν^r + N_0)
        """
        desired_dist = np.linalg.norm(self.ap_locs[serving_ap] - receiver_loc)
        desired_pwr  = self._received_power(powers[serving_ap], desired_dist)

        interference = 0.0
        for nu in range(self.num_aps):
            if nu == serving_ap:
                continue
            d   = np.linalg.norm(self.ap_locs[nu] - receiver_loc)
            interference += self._received_power(powers[nu], d)

        return desired_pwr / (interference + self.noise_power)

    # ------------------------------------------------------------------ #
    # Shannon capacity  C = log2(1 + SINR)  (Eq. 2 / 3)
    # ------------------------------------------------------------------ #

    def _capacity(self, receiver_loc: np.ndarray, serving_ap: int,
                  powers: np.ndarray) -> float:
        sinr = self._sinr(receiver_loc, serving_ap, powers)
        return np.log2(1.0 + sinr)

    # ------------------------------------------------------------------ #
    # Secrecy capacity for one user  (Eq. 5)
    # ------------------------------------------------------------------ #

    def _secrecy_capacity(self, user_idx: int, powers: np.ndarray) -> float:
        """
        C_s(u_k) = [C_{α_k, k}  -  max_j C^e_{α_k, j}]+

        The worst eavesdropper (max capacity among all J eves) is used.
        """
        ap_idx    = self.assoc[user_idx]
        user_cap  = self._capacity(self.user_locs[user_idx], ap_idx, powers)

        # Eve capacity: max over all eavesdroppers
        eve_cap = max(
            self._capacity(self.eve_locs[j], ap_idx, powers)
            for j in range(self.num_eves)
        )

        return max(user_cap - eve_cap, 0.0)

    # ------------------------------------------------------------------ #
    # AP–user association  (Eq. 7 of paper)
    # ------------------------------------------------------------------ #

    def _associate_users(self) -> np.ndarray:
        """
        Each user is served by the AP that maximises their secrecy capacity
        assuming uniform max-power allocation.
        """
        uniform_powers = np.full(self.num_aps, self.max_power, dtype=np.float32)
        assoc = np.zeros(self.num_users, dtype=int)

        for k in range(self.num_users):
            best_ap  = 0
            best_sec = -np.inf
            for n in range(self.num_aps):
                user_cap = self._capacity(self.user_locs[k], n, uniform_powers)
                eve_cap  = max(
                    self._capacity(self.eve_locs[j], n, uniform_powers)
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
        """
        Flatten [AP locs | User locs | Eve locs (possibly noisy)].

        When csi_noise_std > 0, Gaussian noise is added to eavesdropper
        coordinates before handing them to the agent — this is the
        novel contribution (imperfect Eve CSI).
        """
        obs_eve = self.eve_locs.copy()

        if self.csi_noise_std > 0.0:
            noise   = np.random.normal(0.0, self.csi_noise_std, obs_eve.shape)
            obs_eve = np.clip(obs_eve + noise, 0.0, self.map_size)

        obs = np.concatenate([
            self.ap_locs.flatten(),
            self.user_locs.flatten(),
            obs_eve.flatten(),
        ]).astype(np.float32)

        return obs

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        rng = np.random.default_rng(seed)

        # Random placement on 50x50 grid
        self.ap_locs   = rng.uniform(0, self.map_size, (self.num_aps,   2))
        self.user_locs = rng.uniform(0, self.map_size, (self.num_users, 2))
        self.eve_locs  = rng.uniform(0, self.map_size, (self.num_eves,  2))

        # Static association (re-computed each episode)
        self.assoc = self._associate_users()

        obs  = self._build_obs()
        info = {}
        return obs, info

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self, action: np.ndarray):
        """
        Apply power vector chosen by the agent.
        Reward = sum secrecy capacity  (Eq. 9 / objective Eq. 6).
        Each episode is a single-step optimisation over one channel
        realisation (matching the paper's approach).
        """
        powers = np.clip(action, 0.0, self.max_power).astype(np.float32)

        # Sum secrecy capacity across all legitimate users
        reward = sum(
            self._secrecy_capacity(k, powers)
            for k in range(self.num_users)
        )

        # Episode ends after one power-allocation decision
        terminated = True
        truncated  = False

        obs  = self._build_obs()   # next obs (irrelevant since done=True)
        info = {
            "sum_secrecy_capacity": reward,
            "powers": powers.tolist(),
            "user_assoc": self.assoc.tolist(),
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Utility: evaluate a fixed policy (for baselines / testing)
    # ------------------------------------------------------------------ #

    def evaluate_policy(self, powers: np.ndarray) -> dict:
        """
        Compute detailed metrics for a given power allocation.
        Useful for comparing baseline vs. optimized.
        """
        powers = np.clip(powers, 0.0, self.max_power)

        per_user_secrecy = [
            self._secrecy_capacity(k, powers) for k in range(self.num_users)
        ]
        sum_sec   = sum(per_user_secrecy)
        sec_ratio = sum(1 for s in per_user_secrecy if s > 0) / self.num_users

        # Eve total capacity (sum over worst-case per AP)
        eve_cap = []
        for n in range(self.num_aps):
            ec = max(
                self._capacity(self.eve_locs[j], n, powers)
                for j in range(self.num_eves)
            )
            eve_cap.append(ec)

        return {
            "sum_secrecy_capacity": sum_sec,
            "secrecy_ratio": sec_ratio,
            "sum_eve_capacity": sum(eve_cap),
            "per_user_secrecy": per_user_secrecy,
        }