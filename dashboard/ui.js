/**
 * ui.js
 * Wires HTML controls to simulation state and triggers re-renders.
 */

const UI = (() => {
  const s = STATE.state;
  const SERVER = 'http://127.0.0.1:5050';
  let _debounce  = null;
  let _agentOnline = false;

  // --- Agent status indicator ---
  function setAgentStatus(online) {
    _agentOnline = online;
    const el2 = document.getElementById('agentStatus');
    if (!el2) return;
    el2.textContent   = online ? '● AI Online' : '○ AI Offline';
    el2.style.color   = online ? 'var(--green)' : 'var(--muted)';
  }

  // Check server health once on load
  fetch(SERVER + '/health').then(() => setAgentStatus(true)).catch(() => setAgentStatus(false));

  // --- Ask the SAC agent for power allocations ---
  async function askAgent() {
    if (s.mode !== 'rl' || !_agentOnline) return;
    try {
      const res = await fetch(SERVER + '/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ aps: s.aps, users: s.users, eve: s.perceivedEve }),
      });
      const json = await res.json();
      json.powers.forEach((p, i) => {
        s.powers[i] = p;
        const sl = document.getElementById('powerSlider' + i);
        const vl = document.getElementById('powerVal'   + i);
        if (sl) sl.value = p;
        if (vl) vl.textContent = p.toFixed(2) + ' W';
      });
      // Re-render with updated powers (no re-debounce)
      STATE.update(s);
      RENDERER.render(s);
      updateMetrics();
    } catch (_) {
      setAgentStatus(false);
    }
  }

  function scheduleAgentCall() {
    clearTimeout(_debounce);
    _debounce = setTimeout(askAgent, 80);
  }

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
      fill.style.background = sec > 0 ? '#34d399' : '#71717a';
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
    const isRL = mode === 'rl';
    el('powerSection').style.opacity       = isRL ? '1' : '0.4';
    el('powerSection').style.pointerEvents = 'none'; // agent drives in RL, disabled otherwise
    [0,1,2,3].forEach(i => {
      const sl = el('powerSlider' + i);
      if (sl) sl.disabled = isRL; // agent drives in RL mode
    });
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
    if (s.mode === 'rl') scheduleAgentCall();
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
