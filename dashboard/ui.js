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
    const rho = s.csiNoiseSigma / 50;   // UA-SAC needs ρ in observation
    try {
      const res = await fetch(SERVER + '/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ aps: s.aps, users: s.users, eve: s.perceivedEve, rho }),
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

  // Shorthand DOM accessor
  function el(id) { return document.getElementById(id); }

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

    // ρ metric card
    const rho = s.csiNoiseSigma / 50;
    el('metRho').textContent = rho.toFixed(3);

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
      : `σ = ${s.csiNoiseSigma.toFixed(1)}m  |  current error: ${err}m  |  ρ = ${rho.toFixed(3)}`;

    // Update ρ confidence panel
    updateRho(rho);
  }

  // --- ρ (Rho) Confidence Panel ---
  // ρ = σ / D_max (D_max = 50m)  →  [0, 0.2] in practice (σ max = 10m)
  // The gauge maps [0, 0.2] across its full width so small changes are visible.
  const RHO_MAX_DISPLAY = 0.20;   // σ=10m → ρ=0.2 fills gauge 100%
  const BETA = 1.0;               // UA-SAC hyperparameter β

  function updateRho(rho) {
    // Gauge fill: map [0, RHO_MAX] → [0%, 100%]
    const fillPct = Math.min(rho / RHO_MAX_DISPLAY * 100, 100);
    el('confFill').style.width = fillPct + '%';

    // ρ value badge
    el('rhoValBadge').textContent = 'ρ = ' + rho.toFixed(3);

    // α_eff = α_base × (1 + β·ρ)  — β=1.0, α_base shown as 'auto' (SAC-tuned)
    const alphaEffMultiplier = (1 + BETA * rho).toFixed(3);
    el('alphaEff').textContent = '×' + alphaEffMultiplier;

    // Confidence tiers based on ρ
    let tier;
    if (rho < 0.04) {
      tier = {
        label: 'HIGH CONFIDENCE',
        sublabel: 'Agent jams precisely',
        color: '#34d399',   // green
        glow:  '#34d399',
        explain: `rho ~= 0: Eve estimate is nearly exact. UA-SAC concentrates jamming tightly on perceived location. alpha_eff ~= alpha_base — minimal entropy boost needed.`
      };
    } else if (rho < 0.08) {
      tier = {
        label: 'MODERATE CONFIDENCE',
        sublabel: 'Mild coverage spread',
        color: '#86efac',   // light green
        glow:  '#86efac',
        explain: `ρ = ${rho.toFixed(3)}: Small location error (~${(rho*50).toFixed(1)}m). UA-SAC slightly widens jamming spread. α_eff = α_base × ${alphaEffMultiplier} — subtle entropy lift.`
      };
    } else if (rho < 0.12) {
      tier = {
        label: 'MEDIUM CONFIDENCE',
        sublabel: 'Broader jamming pattern',
        color: '#fbbf24',   // amber
        glow:  '#fbbf24',
        explain: `ρ = ${rho.toFixed(3)}: Moderate error (~${(rho*50).toFixed(1)}m est.). UA-SAC spreads jamming across a wider zone. α_eff boosted to ${alphaEffMultiplier}× — agent explores broadly.`
      };
    } else if (rho < 0.16) {
      tier = {
        label: 'LOW CONFIDENCE',
        sublabel: 'Diffuse coverage mode',
        color: '#fb923c',   // orange
        glow:  '#fb923c',
        explain: `ρ = ${rho.toFixed(3)}: High error (~${(rho*50).toFixed(1)}m). Agent distrusts Ê — switches to diffuse jamming. α_eff = ${alphaEffMultiplier}× pushes broad exploration.`
      };
    } else {
      tier = {
        label: 'CRITICAL UNCERTAINTY',
        sublabel: 'Worst-case robust mode',
        color: '#f43f5e',   // red
        glow:  '#f43f5e',
        explain: `ρ = ${rho.toFixed(3)}: Ê is highly unreliable (~${(rho*50).toFixed(1)}m σ). UA-SAC activates worst-case robust mode: R* = min over M=5 Eve samples. α_eff = ${alphaEffMultiplier}× — maximum entropy, maximum coverage.`
      };
    }

    el('confLabel').textContent   = tier.label;
    el('confLabel').style.color   = tier.color;
    el('confSublabel').textContent = tier.sublabel;
    el('confDot').style.background = tier.color;
    el('confDot').style.boxShadow  = `0 0 8px ${tier.glow}`;
    el('rhoExplain').textContent   = tier.explain;
    el('rhoValBadge').style.color  = tier.color;
    el('rhoValBadge').style.borderColor = tier.color + '55';
    el('rhoValBadge').style.background  = tier.color + '18';
  }

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
