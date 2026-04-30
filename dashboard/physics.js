/**
 * physics.js
 * Accurate implementation of the Hoseini et al. 2024 CFJ simulation math.
 * All functions use SI units (Watts, metres, Hz).
 *
 * Equations from arXiv:2403.10342:
 *   Eq.1  Friis path loss
 *   Eq.2  Shannon capacity (legitimate user)
 *   Eq.3  Shannon capacity (eavesdropper)
 *   Eq.5  Secrecy capacity
 *   Eq.7  AP selection rule
 */

const PHYSICS = (() => {
  // --- Constants (paper Section IV) ---
  const C_LIGHT     = 3e8;           // m/s
  const FREQ_HZ     = 2.4e9;        // 2.4 GHz Wi-Fi
  const WAVELENGTH  = C_LIGHT / FREQ_HZ;          // ~0.125 m
  const NOISE_DBM   = -85;
  const NOISE_WATTS = Math.pow(10, (NOISE_DBM - 30) / 10); // 3.16e-12 W
  const GAMMA       = 2.0;          // path loss exponent
  const MAP_SIZE    = 50;           // metres
  const MAX_POWER   = 1.0;          // Watts

  const LAMBDA_4PI  = WAVELENGTH / (4 * Math.PI);

  // --- Eq.1: Friis received power ---
  // p_r = p_t * (λ/4π)^2 * d^(-γ)
  function receivedPower(p_tx, dist) {
    const d = Math.max(dist, 0.1); // guard against d=0
    return p_tx * (LAMBDA_4PI ** 2) * Math.pow(d, -GAMMA);
  }

  // Euclidean distance between two {x,y} objects
  function dist(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
  }

  // --- Eq.2/3: SINR at a receiver node from a serving AP ---
  // SINR = desired_power / (sum_of_interference + noise)
  function sinrAt(receiverPos, servingIdx, aps, powers) {
    const desired = receivedPower(powers[servingIdx], dist(aps[servingIdx], receiverPos));
    let interference = 0;
    for (let i = 0; i < aps.length; i++) {
      if (i === servingIdx) continue;
      interference += receivedPower(powers[i], dist(aps[i], receiverPos));
    }
    return desired / (interference + NOISE_WATTS);
  }

  // --- Eq.2/3: Shannon capacity C = log2(1 + SINR) ---
  // W=1 Hz (spectral efficiency in bps/Hz, as in the paper)
  function capacityAt(receiverPos, servingIdx, aps, powers) {
    return Math.log2(1 + sinrAt(receiverPos, servingIdx, aps, powers));
  }

  // --- Eq.5: Secrecy capacity for one user ---
  // C_s = max( C_user - max_j(C_eve_j), 0 )
  function secrecyCapacity(userPos, servingIdx, eveLocs, aps, powers) {
    const userCap = capacityAt(userPos, servingIdx, aps, powers);
    const eveCap  = Math.max(...eveLocs.map(e => capacityAt(e, servingIdx, aps, powers)));
    return Math.max(userCap - eveCap, 0);
  }

  // --- Eq.7: AP selection ---
  // Each user connects to AP that maximises secrecy (assuming given powers).
  // mode='normal' → connect to AP with highest raw SINR (ignores Eve)
  // mode='smart' | 'rl' → connect to AP with highest secrecy capacity
  function associateUsers(users, eveLocs, aps, powers, mode) {
    return users.map(user => {
      let bestAP  = 0;
      let bestVal = -Infinity;
      for (let n = 0; n < aps.length; n++) {
        let val;
        if (mode === 'normal') {
          val = sinrAt(user, n, aps, powers);
        } else {
          const userCap = capacityAt(user, n, aps, powers);
          const eveCap  = Math.max(...eveLocs.map(e => capacityAt(e, n, aps, powers)));
          val = userCap - eveCap;
        }
        if (val > bestVal) { bestVal = val; bestAP = n; }
      }
      return bestAP;
    });
  }

  // --- Full state evaluation ---
  // Returns per-user secrecy, sum secrecy, secrecy ratio, eve capacity
  function evaluate(state) {
    const { aps, users, trueEve, powers, mode } = state;
    const eveLocs = [trueEve]; // single Eve (J=1)

    // Step 1: AP selection (Eq.7) — always uses max power for association
    const maxPowers = new Array(aps.length).fill(MAX_POWER);
    const assocPowers = (mode === 'rl') ? maxPowers : powers;
    const assoc = associateUsers(users, eveLocs, aps, assocPowers, mode);

    // Step 2: Evaluate secrecy with actual powers
    const evalPowers = (mode === 'rl') ? powers : new Array(aps.length).fill(MAX_POWER);

    const perUserSecrecy = users.map((u, k) =>
      secrecyCapacity(u, assoc[k], eveLocs, aps, evalPowers)
    );

    const sumSecrecy   = perUserSecrecy.reduce((a, b) => a + b, 0);
    const secrecyRatio = perUserSecrecy.filter(s => s > 0).length / users.length;

    // Eve total capacity (worst-case across all APs)
    const eveCap = aps.map((_, n) =>
      Math.max(...eveLocs.map(e => capacityAt(e, n, aps, evalPowers)))
    ).reduce((a, b) => a + b, 0);

    return { assoc, perUserSecrecy, sumSecrecy, secrecyRatio, eveCap };
  }

  // --- Heatmap: secrecy potential at every grid cell ---
  // Returns Float32Array of size GRID*GRID, values = secrecy capacity
  // for a hypothetical user placed at each cell centre.
  function computeHeatmap(state, gridRes = 50) {
    const { aps, trueEve, powers, mode } = state;
    const eveLocs   = [trueEve];
    const evalPowers = (mode === 'rl') ? powers : new Array(aps.length).fill(MAX_POWER);
    const maxPowers  = new Array(aps.length).fill(MAX_POWER);
    const data = new Float32Array(gridRes * gridRes);

    for (let row = 0; row < gridRes; row++) {
      for (let col = 0; col < gridRes; col++) {
        const pos = {
          x: (col + 0.5) * (MAP_SIZE / gridRes),
          y: (row + 0.5) * (MAP_SIZE / gridRes),
        };
        // Find best AP for this hypothetical user
        let bestAP = 0, bestVal = -Infinity;
        for (let n = 0; n < aps.length; n++) {
          const uCap = capacityAt(pos, n, aps, maxPowers);
          const eCap = Math.max(...eveLocs.map(e => capacityAt(e, n, aps, maxPowers)));
          const v = (mode === 'normal') ? sinrAt(pos, n, aps, maxPowers) : (uCap - eCap);
          if (v > bestVal) { bestVal = v; bestAP = n; }
        }
        data[row * gridRes + col] = secrecyCapacity(pos, bestAP, eveLocs, aps, evalPowers);
      }
    }
    return data;
  }

  return {
    receivedPower, dist, sinrAt, capacityAt,
    secrecyCapacity, associateUsers, evaluate, computeHeatmap,
    MAP_SIZE, MAX_POWER, NOISE_WATTS,
  };
})();
