/**
 * ui.js
 * Wires HTML controls to simulation state and triggers re-renders.
 */

const UI = (() => {
  const s = STATE.state;

  // --- Metrics display ---
  function updateMetrics() {
    const r = s.result;
    if (!r) return;

    // Summary cards
    const sumSec = r.sumSecrecy.toFixed(3);
    const ratio  = (r.secrecyRatio * 100).toFixed(0);
    const eveCap = r.eveCap.toFixed(3);

    el('metSumSec').textContent  = sumSec;
    el('metRatio').textContent   = ratio + '%';
    el('metEveCap').textContent  = eveCap;
    el('metNoise').textContent   = s.csiNoiseSigma.toFixed(1) + 'm';

    // Color coding
    const sumEl = el('metSumSec');
    sumEl.className = 'metric-value ' + (r.sumSecrecy > 5 ? 'green' : r.sumSecrecy > 2 ? 'yellow' : 'red');

    const ratEl = el('metRatio');
    ratEl.className = 'metric-value ' + (r.secrecyRatio === 1 ? 'green' : r.secrecyRatio > 0.5 ? 'yellow' : 'red');

    // Per-user bars (max scale = 15 bps/Hz)
    const MAX_BAR = 15;
    r.perUserSecrecy.forEach((sec, k) => {
      const fill = el(`userBar${k}`);
      const lbl  = el(`userLbl${k}`);
      if (!fill) return;
      const pct = Math.min(sec / MAX_BAR * 100, 100);
      fill.style.width = pct + '%';
      fill.style.background = sec > 0 ? '#00e676' : '#ff5252';
      lbl.textContent = sec.toFixed(3) + ' bps/Hz';
    });

    // CSI info
    const err = PHYSICS.dist(s.trueEve, s.perceivedEve).toFixed(1);
    el('csiError').textContent = s.csiNoiseSigma === 0
      ? 'Perfect CSI — exact Eve location known'
      : `σ = ${s.csiNoiseSigma.toFixed(1)}m  |  current error: ${err}m`;
  }

  function el(id) { return document.getElementById(id); }

  // --- Mode buttons ---
  function setMode(mode) {
    s.mode = mode;
    ['normal', 'smart', 'rl'].forEach(m => {
      el('mode' + m).classList.toggle('active', m === mode);
    });
    // Power sliders only active in RL mode
    el('powerSection').style.opacity = (mode === 'rl') ? '1' : '0.4';
    el('powerSection').style.pointerEvents = (mode === 'rl') ? 'auto' : 'none';
    s.heatmapDirty = true;
    refresh();
  }

  // --- CSI noise slider ---
  function updateNoise(val) {
    s.csiNoiseSigma = parseFloat(val);
    el('noiseVal').textContent = parseFloat(val).toFixed(1) + ' m';
    STATE.resamplePerceivedEve(s);
    s.heatmapDirty = true;
    refresh();
  }

  // --- AP power sliders ---
  function updatePower(apIdx, val) {
    s.powers[apIdx] = parseFloat(val);
    el('powerVal' + apIdx).textContent = parseFloat(val).toFixed(2) + ' W';
    s.heatmapDirty = true;
    refresh();
  }

  // --- Heatmap toggle ---
  function toggleHeatmap() {
    s.showHeatmap = !s.showHeatmap;
    el('btnHeatmap').textContent = s.showHeatmap ? 'Heatmap ON' : 'Heatmap OFF';
    el('btnHeatmap').classList.toggle('active', s.showHeatmap);
    refresh();
  }

  // --- Randomize ---
  function randomize() {
    STATE.randomize(s);
    refresh();
  }

  // --- Reset ---
  function resetState() {
    const fresh = STATE.defaultState();
    Object.assign(s, fresh);
    STATE.resamplePerceivedEve(s);
    s.heatmapDirty = true;
    // Sync slider UI
    el('noiseSlider').value = 0;
    el('noiseVal').textContent = '0.0 m';
    [0,1,2,3].forEach(i => {
      el('powerSlider' + i).value = s.powers[i];
      el('powerVal' + i).textContent = s.powers[i].toFixed(2) + ' W';
    });
    refresh();
  }

  // --- Main refresh ---
  function refresh() {
    STATE.update(s);
    RENDERER.render(s);
    updateMetrics();
  }

  // --- Canvas drag ---
  function initDrag(canvas) {
    let dragging = null; // { type: 'ap'|'user'|'eve', idx: number }

    canvas.addEventListener('mousedown', e => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const cx = (e.clientX - rect.left) * scaleX;
      const cy = (e.clientY - rect.top)  * scaleY;

      // Check APs
      for (let i = 0; i < s.aps.length; i++) {
        if (RENDERER.hitTest(cx, cy, s.aps[i], 'ap')) {
          dragging = { type: 'ap', idx: i }; return;
        }
      }
      // Check users
      for (let i = 0; i < s.users.length; i++) {
        if (RENDERER.hitTest(cx, cy, s.users[i], 'user')) {
          dragging = { type: 'user', idx: i }; return;
        }
      }
      // Check Eve
      if (RENDERER.hitTest(cx, cy, s.trueEve, 'user')) {
        dragging = { type: 'eve' };
      }
    });

    canvas.addEventListener('mousemove', e => {
      if (!dragging) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const cx = (e.clientX - rect.left) * scaleX;
      const cy = (e.clientY - rect.top)  * scaleY;
      const pos = RENDERER.toSim(cx, cy);

      if (dragging.type === 'ap') {
        s.aps[dragging.idx].x = pos.x;
        s.aps[dragging.idx].y = pos.y;
      } else if (dragging.type === 'user') {
        s.users[dragging.idx].x = pos.x;
        s.users[dragging.idx].y = pos.y;
      } else if (dragging.type === 'eve') {
        s.trueEve.x = pos.x;
        s.trueEve.y = pos.y;
        STATE.resamplePerceivedEve(s);
      }
      s.heatmapDirty = true;
      refresh();
    });

    canvas.addEventListener('mouseup',    () => { dragging = null; });
    canvas.addEventListener('mouseleave', () => { dragging = null; });

    // Cursor styling
    canvas.addEventListener('mousemove', e => {
      if (dragging) { canvas.style.cursor = 'grabbing'; return; }
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const cx = (e.clientX - rect.left) * scaleX;
      const cy = (e.clientY - rect.top)  * scaleY;
      const onNode = [...s.aps, ...s.users, s.trueEve].some((n, i) =>
        RENDERER.hitTest(cx, cy, n, i < s.aps.length ? 'ap' : 'user')
      );
      canvas.style.cursor = onNode ? 'grab' : 'crosshair';
    });
  }

  function init(canvas) {
    initDrag(canvas);
    // Expose to window for inline onclick handlers
    window.setMode      = setMode;
    window.updateNoise  = updateNoise;
    window.updatePower  = updatePower;
    window.toggleHeatmap = toggleHeatmap;
    window.randomize    = randomize;
    window.resetState   = resetState;
    refresh();
  }

  return { init, refresh };
})();
