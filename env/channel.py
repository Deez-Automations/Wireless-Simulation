import numpy as np

class IRSChannel:
    """
    IRS-aided wiretap channel environment.
    
    Entities:
        Alice  : base station with N_t transmit antennas
        Bob    : single-antenna legitimate receiver
        Eve    : single-antenna passive eavesdropper
        IRS    : N_irs reflecting elements (uniform linear array)
    
    All channels follow Rayleigh fading (complex Gaussian).
    """

    def __init__(
        self,
        N_t    = 4,      # Alice antennas
        N_irs  = 32,     # IRS elements
        sigma2 = 1e-3,   # noise power (linear)
        seed   = None
    ):
        self.N_t    = N_t
        self.N_irs  = N_irs
        self.sigma2 = sigma2
        self.rng    = np.random.default_rng(seed)

        # Channel path loss constants (simplified, normalized)
        self.beta_AB  = 1.0   # Alice -> Bob direct
        self.beta_AE  = 1.0   # Alice -> Eve direct
        self.beta_AI  = 1.0   # Alice -> IRS
        self.beta_IB  = 1.0   # IRS   -> Bob
        self.beta_IE  = 1.0   # IRS   -> Eve

        # Current channel realizations (set by reset/step)
        self.h_AB  = None   # (N_t,)       Alice->Bob
        self.h_AE  = None   # (N_t,)       Alice->Eve
        self.H_AI  = None   # (N_irs, N_t) Alice->IRS
        self.h_IB  = None   # (N_irs,)     IRS->Bob
        self.h_IE  = None   # (N_irs,)     IRS->Eve
        self.Theta = None   # (N_irs,)     IRS phase shifts (complex)

    # ------------------------------------------------------------------
    # Channel generation
    # ------------------------------------------------------------------

    def _rayleigh(self, shape):
        """Generate complex Gaussian channel coefficients CN(0, 0.5 per dim)."""
        return (self.rng.standard_normal(shape) +
                1j * self.rng.standard_normal(shape)) / np.sqrt(2)

    def generate_channels(self):
        """
        Sample a fresh set of channel realizations.
        Call this at the start of every episode.
        """
        self.h_AB = np.sqrt(self.beta_AB) * self._rayleigh((self.N_t,))
        self.h_AE = np.sqrt(self.beta_AE) * self._rayleigh((self.N_t,))
        self.H_AI = np.sqrt(self.beta_AI) * self._rayleigh((self.N_irs, self.N_t))
        self.h_IB = np.sqrt(self.beta_IB) * self._rayleigh((self.N_irs,))
        self.h_IE = np.sqrt(self.beta_IE) * self._rayleigh((self.N_irs,))
        return self

    # ------------------------------------------------------------------
    # IRS phase shift matrix
    # ------------------------------------------------------------------

    def set_phases_continuous(self, theta):
        """
        Set IRS phases from a real-valued array of angles in [-pi, pi].
        theta : (N_irs,) numpy array
        """
        self.Theta = np.exp(1j * theta)   # unit-modulus complex

    def set_phases_discrete(self, theta, bits=2):
        """
        Quantize continuous angles to discrete levels (2-bit = 4 levels).
        theta : (N_irs,) real-valued angles
        bits  : phase resolution (1=2 levels, 2=4 levels, 3=8 levels)
        """
        n_levels = 2 ** bits
        levels   = np.linspace(0, 2 * np.pi, n_levels, endpoint=False)
        # snap each angle to nearest discrete level
        theta_wrapped  = theta % (2 * np.pi)
        idx            = np.argmin(
            np.abs(theta_wrapped[:, None] - levels[None, :]), axis=1
        )
        self.Theta = np.exp(1j * levels[idx])
        return levels[idx]   # return the quantized angles for logging

    # ------------------------------------------------------------------
    # Effective channel computation
    # ------------------------------------------------------------------

    def _effective_channel_bob(self, w):
        """
        Compute effective channel gain at Bob.
        
        h_eff_Bob = h_AB^H w + h_IB^H Theta H_AI w
        
        w : (N_t,) transmit beamforming vector (normalized)
        Returns |h_eff_Bob|^2
        """
        direct    = self.h_AB.conj() @ w
        reflected = self.h_IB.conj() @ (self.Theta * (self.H_AI @ w))
        h_eff     = direct + reflected
        return np.abs(h_eff) ** 2

    def _effective_channel_eve(self, w):
        """
        Compute effective channel gain at Eve.
        
        h_eff_Eve = h_AE^H w + h_IE^H Theta H_AI w
        
        Returns |h_eff_Eve|^2
        """
        direct    = self.h_AE.conj() @ w
        reflected = self.h_IE.conj() @ (self.Theta * (self.H_AI @ w))
        h_eff     = direct + reflected
        return np.abs(h_eff) ** 2

    # ------------------------------------------------------------------
    # SNR and secrecy rate
    # ------------------------------------------------------------------

    def compute_snr(self, w, P_tx=1.0):
        """
        Compute SNR at Bob and Eve.
        
        SNR = P_tx * |h_eff|^2 / sigma^2
        
        P_tx : transmit power (linear)
        Returns (snr_bob, snr_eve)
        """
        gain_bob = self._effective_channel_bob(w)
        gain_eve = self._effective_channel_eve(w)
        snr_bob  = P_tx * gain_bob / self.sigma2
        snr_eve  = P_tx * gain_eve / self.sigma2
        return snr_bob, snr_eve

    def compute_secrecy_rate(self, w, P_tx=1.0):
        """
        Rs = max{ log2(1 + SNR_Bob) - log2(1 + SNR_Eve), 0 }
        
        Returns (Rs, snr_bob, snr_eve)
        """
        snr_bob, snr_eve = self.compute_snr(w, P_tx)
        C_bob  = np.log2(1 + snr_bob)
        C_eve  = np.log2(1 + snr_eve)
        Rs     = max(C_bob - C_eve, 0.0)
        return Rs, snr_bob, snr_eve

    # ------------------------------------------------------------------
    # Worst-case secrecy (your Contribution 1 — no Eve CSI)
    # ------------------------------------------------------------------

    def compute_worst_case_secrecy(self, w, P_tx=1.0, n_samples=50):
        """
        Estimate worst-case secrecy rate when Eve's CSI is unknown.
        
        Strategy: sample n_samples random Eve channels from the same
        distribution (statistical model), compute secrecy rate for each,
        return the MINIMUM (worst case for Alice).
        
        This is what the DRL agent will use as reward in Contribution 1.
        """
        snr_bob, _ = self.compute_snr(w, P_tx)
        C_bob      = np.log2(1 + snr_bob)

        worst_Rs = np.inf
        for _ in range(n_samples):
            # sample a random Eve channel — we don't know hers, so we try many
            h_AE_rand = np.sqrt(self.beta_AE) * self._rayleigh((self.N_t,))
            h_IE_rand = np.sqrt(self.beta_IE) * self._rayleigh((self.N_irs,))

            direct    = h_AE_rand.conj() @ w
            reflected = h_IE_rand.conj() @ (self.Theta * (self.H_AI @ w))
            h_eff_e   = direct + reflected
            gain_eve  = np.abs(h_eff_e) ** 2
            snr_eve   = P_tx * gain_eve / self.sigma2
            C_eve     = np.log2(1 + snr_eve)
            Rs        = max(C_bob - C_eve, 0.0)
            worst_Rs  = min(worst_Rs, Rs)

        return worst_Rs

    # ------------------------------------------------------------------
    # State vector for DRL agent
    # ------------------------------------------------------------------

    def get_state(self, include_eve_csi=True):
        """
        Build the state vector the DRL agent observes.
        
        Baseline (include_eve_csi=True):
            [real(h_AB), imag(h_AB),
             real(H_AI), imag(H_AI),
             real(h_IB), imag(h_IB),
             real(h_AE), imag(h_AE),   <- removed in Contribution 1
             real(h_IE), imag(h_IE)]   <- removed in Contribution 1
        
        Contribution 1 (include_eve_csi=False):
            Eve's channels removed from state vector.
        """
        state = np.concatenate([
            self.h_AB.real,  self.h_AB.imag,
            self.H_AI.real.flatten(), self.H_AI.imag.flatten(),
            self.h_IB.real,  self.h_IB.imag,
        ])
        if include_eve_csi:
            state = np.concatenate([
                state,
                self.h_AE.real, self.h_AE.imag,
                self.h_IE.real, self.h_IE.imag,
            ])
        return state.astype(np.float32)

    def state_dim(self, include_eve_csi=True):
        """Return the state vector dimension."""
        base = 2*self.N_t + 2*self.N_irs*self.N_t + 2*self.N_irs
        if include_eve_csi:
            base += 2*self.N_t + 2*self.N_irs
        return base

    def action_dim(self):
        """Action = IRS phases (N_irs) + beamforming vector (N_t complex = 2*N_t real)."""
        return self.N_irs + 2 * self.N_t
