/**
 * state.js
 * Simulation state and update functions.
 */

const STATE = (() => {
  const M = PHYSICS.MAP_SIZE;

  function defaultState() {
    return {
      mode: 'rl',
      csiNoiseSigma: 0,
      showHeatmap: true,
      showLabels: true,

      // 4 APs placed at 4 quadrant centres
      aps: [
        { x: 12, y: 12 },
        { x: 38, y: 12 },
        { x: 12, y: 38 },
        { x: 38, y: 38 },
      ],

      // Per-AP transmit powers (Watts, 0–1)
      powers: [0.8, 0.8, 0.8, 0.8],

      // 2 legitimate users
      users: [
        { x: 20, y: 25 },
        { x: 32, y: 28 },
      ],

      // True Eve location (known to physics, unknown to agent when σ>0)
      trueEve: { x: 25, y: 20 },

      // Perceived Eve (what agent observes — noisy when σ>0)
      perceivedEve: { x: 25, y: 20 },

      // Computed each frame
      result: null,
      heatmapData: null,
      heatmapDirty: true,
    };
  }

  // Gaussian noise sample using Box-Muller
  function gaussianNoise(sigma) {
    if (sigma === 0) return { dx: 0, dy: 0 };
    const u1 = Math.random(), u2 = Math.random();
    const mag = sigma * Math.sqrt(-2 * Math.log(u1));
    return {
      dx: mag * Math.cos(2 * Math.PI * u2),
      dy: mag * Math.sin(2 * Math.PI * u2),
    };
  }

  function resamplePerceivedEve(s) {
    const { dx, dy } = gaussianNoise(s.csiNoiseSigma);
    s.perceivedEve = {
      x: Math.max(0, Math.min(M, s.trueEve.x + dx)),
      y: Math.max(0, Math.min(M, s.trueEve.y + dy)),
    };
  }

  function randomize(s) {
    const rand = (lo, hi) => lo + Math.random() * (hi - lo);
    s.aps = [
      { x: rand(5, 20), y: rand(5, 20) },
      { x: rand(30, 45), y: rand(5, 20) },
      { x: rand(5, 20), y: rand(30, 45) },
      { x: rand(30, 45), y: rand(30, 45) },
    ];
    s.users = [
      { x: rand(10, 40), y: rand(10, 40) },
      { x: rand(10, 40), y: rand(10, 40) },
    ];
    s.trueEve = { x: rand(5, 45), y: rand(5, 45) };
    resamplePerceivedEve(s);
    s.heatmapDirty = true;
  }

  function update(s) {
    s.result = PHYSICS.evaluate(s);
    if (s.heatmapDirty) {
      s.heatmapData = PHYSICS.computeHeatmap(s, 50);
      s.heatmapDirty = false;
    }
  }

  const state = defaultState();
  resamplePerceivedEve(state);

  return { state, defaultState, resamplePerceivedEve, randomize, update };
})();
