/**
 * renderer.js
 * Canvas rendering: grid, heatmap, nodes, labels, connections.
 */

const RENDERER = (() => {
  const PAD    = 30;
  const COLORS = {
    bg:           '#06060f',
    gridLine:     'rgba(255,255,255,0.04)',
    ap:           '#00d4ff',
    apGlow:       'rgba(0,212,255,0.35)',
    userSecure:   '#00e676',
    userInsecure: '#ff5252',
    trueEve:      '#ff4757',
    percEve:      '#ffd32a',
    conn:         'rgba(255,255,255,0.2)',
    csiLine:      'rgba(255,211,42,0.6)',
    label:        '#e0e0ff',
    labelMuted:   '#7070a0',
  };

  let canvas, ctx, W, H, scale;

  function init(canvasEl) {
    canvas = canvasEl;
    ctx = canvas.getContext('2d');
    W = canvas.width;
    H = canvas.height;
    scale = (W - 2 * PAD) / PHYSICS.MAP_SIZE;
  }

  // Convert simulation metres → canvas pixels
  function toCanvas(pos) {
    return { x: PAD + pos.x * scale, y: PAD + pos.y * scale };
  }

  // Convert canvas pixels → simulation metres
  function toSim(cx, cy) {
    return {
      x: Math.max(0, Math.min(PHYSICS.MAP_SIZE, (cx - PAD) / scale)),
      y: Math.max(0, Math.min(PHYSICS.MAP_SIZE, (cy - PAD) / scale)),
    };
  }

  function nodeRadius(type) {
    return type === 'ap' ? 12 : 9;
  }

  // Check if canvas point is near a node
  function hitTest(cx, cy, pos, type) {
    const c = toCanvas(pos);
    const r = nodeRadius(type) + 4;
    return (cx - c.x) ** 2 + (cy - c.y) ** 2 <= r * r;
  }

  // --- Background ---
  function drawBackground() {
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, W, H);

    // Grid lines every 10m
    ctx.strokeStyle = COLORS.gridLine;
    ctx.lineWidth = 1;
    for (let m = 0; m <= 50; m += 10) {
      const { x } = toCanvas({ x: m, y: 0 });
      const { y } = toCanvas({ x: 0, y: m });
      ctx.beginPath(); ctx.moveTo(x, PAD); ctx.lineTo(x, H - PAD); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(PAD, y); ctx.lineTo(W - PAD, y); ctx.stroke();
    }

    // Border
    ctx.strokeStyle = 'rgba(79,143,255,0.15)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(PAD, PAD, W - 2 * PAD, H - 2 * PAD);
  }

  // --- Heatmap ---
  function drawHeatmap(data, gridRes = 50) {
    if (!data) return;
    const cellW = (W - 2 * PAD) / gridRes;
    const cellH = (H - 2 * PAD) / gridRes;

    // Find max for normalisation
    let maxVal = 0;
    for (let i = 0; i < data.length; i++) maxVal = Math.max(maxVal, data[i]);
    if (maxVal === 0) maxVal = 1;

    const imgData = ctx.createImageData(W, H);
    const d = imgData.data;

    for (let row = 0; row < gridRes; row++) {
      for (let col = 0; col < gridRes; col++) {
        const val = data[row * gridRes + col];
        const t   = Math.min(val / maxVal, 1);

        // Color: dark red → orange → yellow → green
        let r, g, b;
        if (t < 0.5) {
          const f = t / 0.5;
          r = Math.round(180 + f * 75);
          g = Math.round(20 + f * 100);
          b = 20;
        } else {
          const f = (t - 0.5) / 0.5;
          r = Math.round(255 - f * 210);
          g = Math.round(120 + f * 110);
          b = Math.round(f * 30);
        }
        const alpha = Math.round(180 * t + 20);

        // Fill rectangle in ImageData
        const px0 = Math.round(PAD + col * cellW);
        const py0 = Math.round(PAD + row * cellH);
        const px1 = Math.round(PAD + (col + 1) * cellW);
        const py1 = Math.round(PAD + (row + 1) * cellH);

        for (let py = py0; py < py1; py++) {
          for (let px = px0; px < px1; px++) {
            const idx = (py * W + px) * 4;
            d[idx]     = r;
            d[idx + 1] = g;
            d[idx + 2] = b;
            d[idx + 3] = alpha;
          }
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  // --- Connection lines (user → serving AP) ---
  function drawConnections(state) {
    const { aps, users, result } = state;
    if (!result) return;
    result.assoc.forEach((apIdx, k) => {
      const a = toCanvas(aps[apIdx]);
      const u = toCanvas(users[k]);
      ctx.strokeStyle = COLORS.conn;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(u.x, u.y);
      ctx.stroke();
      ctx.setLineDash([]);
    });
  }

  // --- CSI error line (true Eve → perceived Eve) ---
  function drawCSILine(state) {
    const { trueEve, perceivedEve, csiNoiseSigma } = state;
    if (csiNoiseSigma === 0) return;

    const t = toCanvas(trueEve);
    const p = toCanvas(perceivedEve);

    ctx.strokeStyle = COLORS.csiLine;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(t.x, t.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Error distance label
    const errDist = PHYSICS.dist(trueEve, perceivedEve).toFixed(1);
    const midX = (t.x + p.x) / 2;
    const midY = (t.y + p.y) / 2;
    ctx.fillStyle = COLORS.percEve;
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${errDist}m`, midX, midY - 6);
  }

  // --- Draw a glowing circle ---
  function drawGlowCircle(cx, cy, r, color, glowColor, glowRadius) {
    ctx.shadowColor  = glowColor;
    ctx.shadowBlur   = glowRadius;
    ctx.fillStyle    = color;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  // --- Draw ring (for true Eve) ---
  function drawRing(cx, cy, r, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.globalAlpha = 0.4;
    ctx.beginPath();
    ctx.arc(cx, cy, r + 8, 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // --- All nodes ---
  function drawNodes(state) {
    const { aps, users, trueEve, perceivedEve, powers, result, mode } = state;

    // AP glow rings (power proportional)
    aps.forEach((ap, i) => {
      const c    = toCanvas(ap);
      const pwr  = (mode === 'rl') ? powers[i] : 1.0;
      const glow = 10 + pwr * 30;
      ctx.strokeStyle = `rgba(0,212,255,${0.08 + pwr * 0.12})`;
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.arc(c.x, c.y, 20 + pwr * 25, 0, Math.PI * 2);
      ctx.stroke();
    });

    // AP nodes
    aps.forEach((ap, i) => {
      const c   = toCanvas(ap);
      const pwr = (mode === 'rl') ? powers[i] : 1.0;
      drawGlowCircle(c.x, c.y, 10, COLORS.ap, COLORS.apGlow, 15 + pwr * 20);

      // AP label
      ctx.fillStyle  = '#e0e0ff';
      ctx.font       = 'bold 9px Inter, sans-serif';
      ctx.textAlign  = 'center';
      ctx.fillText(`AP${i + 1}`, c.x, c.y - 17);

      if (mode === 'rl') {
        ctx.fillStyle = 'rgba(0,212,255,0.8)';
        ctx.font      = '8px JetBrains Mono, monospace';
        ctx.fillText(`${pwr.toFixed(2)}W`, c.x, c.y + 22);
      }
    });

    // User nodes
    users.forEach((u, k) => {
      const c       = toCanvas(u);
      const secure  = result && result.perUserSecrecy[k] > 0;
      const color   = secure ? COLORS.userSecure : COLORS.userInsecure;
      const glow    = secure ? 'rgba(0,230,118,0.5)' : 'rgba(255,82,82,0.5)';
      drawGlowCircle(c.x, c.y, 8, color, glow, 18);
      ctx.fillStyle = '#e0e0ff';
      ctx.font      = 'bold 9px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`U${k + 1}`, c.x, c.y - 15);
      if (result) {
        ctx.fillStyle = color;
        ctx.font      = '8px JetBrains Mono, monospace';
        ctx.fillText(`${result.perUserSecrecy[k].toFixed(2)}`, c.x, c.y + 20);
      }
    });

    // True Eve
    const te = toCanvas(trueEve);
    drawRing(te.x, te.y, 8, COLORS.trueEve);
    drawGlowCircle(te.x, te.y, 8, COLORS.trueEve, 'rgba(255,71,87,0.6)', 20);
    ctx.fillStyle = COLORS.trueEve;
    ctx.font      = 'bold 9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('EVE', te.x, te.y - 16);
    ctx.fillStyle = '#ff8a80';
    ctx.font      = '8px Inter, sans-serif';
    ctx.fillText('(true)', te.x, te.y + 20);

    // Perceived Eve
    if (state.csiNoiseSigma > 0) {
      const pe = toCanvas(perceivedEve);
      drawGlowCircle(pe.x, pe.y, 7, COLORS.percEve, 'rgba(255,211,42,0.5)', 18);
      ctx.fillStyle = COLORS.percEve;
      ctx.font      = '8px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('perceived', pe.x, pe.y + 20);
    }
  }

  // --- Legend overlay ---
  function drawLegend() {
    const items = [
      { color: '#00d4ff', label: 'Access Point (AP)' },
      { color: '#00e676', label: 'User (secure)' },
      { color: '#ff5252', label: 'User (no secrecy)' },
      { color: '#ff4757', label: 'Eve — true location' },
      { color: '#ffd32a', label: 'Eve — perceived (noisy)' },
    ];
    let lx = PAD + 8, ly = H - PAD - 8 - items.length * 16;
    ctx.fillStyle = 'rgba(6,6,15,0.75)';
    ctx.fillRect(lx - 4, ly - 4, 185, items.length * 16 + 8);
    items.forEach((item, i) => {
      const y = ly + i * 16 + 8;
      ctx.fillStyle = item.color;
      ctx.beginPath(); ctx.arc(lx + 5, y, 4, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#a0a0c0';
      ctx.font = '9px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(item.label, lx + 14, y + 3);
    });
  }

  // --- Full render ---
  function render(state) {
    drawBackground();
    if (state.showHeatmap) drawHeatmap(state.heatmapData);
    drawConnections(state);
    drawCSILine(state);
    drawNodes(state);
    drawLegend();
  }

  return { init, render, toSim, toCanvas, hitTest, nodeRadius };
})();
