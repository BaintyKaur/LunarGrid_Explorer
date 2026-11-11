// Data, values, colours and the statistics panel. No 3D code in here.

export const R = 1737400, HALF = 770000, CELL = 20000, NC = 77, SIG = 2.58;
export const CATS = [
  { k: 'robbins', n: 'Robbins', v: '--c1' },
  { k: 'lu5m', n: 'LU5M812TGT', v: '--c2' },
  { k: 'luc', n: 'LUC-GRAS200', v: '--c3' }
];

export const S = { layer: 'dens', cat: 'robbins', thr: 1, tab: 'val', sel: null, polar: true };

// Everything loaded from dashboard_data.json ends up here.
export const D = {
  tiles: [], cells: [], N: [], S: [],
  tileGrid: new Int16Array(72 * 36).fill(-1),
  cellGrid: { N: new Int16Array(NC * NC).fill(-1), S: new Int16Array(NC * NC).fill(-1) }
};

/* ---------- palettes ---------- */
const PALS = {
  viridis: ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725'],
  div: ['#2166ac', '#67a9cf', '#e8eaee', '#ef8a62', '#b2182b'],
  terrain: ['#2b3a67', '#496a8c', '#7aa0a6', '#c8d6a3', '#e6c27a', '#b5835a', '#f2efe9'],
  slope: ['#fff5d6', '#fdc07a', '#e8743b', '#a5322b', '#4d0f2b'],
  sun: ['#10131f', '#4b2a6b', '#b4467b', '#f68f5a', '#fcdc85'],
  shadow: ['#f3f6fb', '#b4cce6', '#5d8fc6', '#2a5a9e', '#0f2a5c'],
  cv: ['#f4f0fa', '#c9b7e8', '#9a74cf', '#6b3fa8', '#3b1a6e']
};
const ROBUST = ['#6d7686', '#f7d56b', '#f08a3c', '#c0262d'];
const hex = (h) => [parseInt(h.substr(1, 2), 16), parseInt(h.substr(3, 2), 16), parseInt(h.substr(5, 2), 16)];
const pack = (r, g, b) => (0xff000000 | (b << 16) | (g << 8) | r) >>> 0;
const NODATA32 = pack(70, 76, 90);
const LUT = {};
Object.keys(PALS).forEach((name) => {
  const s = PALS[name].map(hex), lut = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (s.length - 1), j = Math.min(s.length - 2, Math.floor(x)), f = x - j, a = s[j], b = s[j + 1];
    lut[i] = pack(Math.round(a[0] + (b[0] - a[0]) * f), Math.round(a[1] + (b[1] - a[1]) * f), Math.round(a[2] + (b[2] - a[2]) * f));
  }
  LUT[name] = lut;
});
const ROBUST32 = ROBUST.map((h) => { const c = hex(h); return pack(c[0], c[1], c[2]); });
export const gradCss = (name) => 'linear-gradient(90deg,' + PALS[name].join(',') + ')';

/* ---------- helpers ---------- */
export const isNum = (v) => typeof v === 'number' && isFinite(v);
export const f = (v, d) => isNum(v) ? v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }) : '–';
export const fl = (lat) => Math.abs(lat).toFixed(1) + '° ' + (lat < 0 ? 'S' : 'N');
export const fo = (lon) => Math.abs(lon).toFixed(1) + '° ' + (lon < 0 ? 'W' : 'E');
const fl0 = (lat) => Math.abs(lat) + '° ' + (lat < 0 ? 'S' : 'N');
const fo0 = (lon) => Math.abs(lon) + '° ' + (lon < 0 ? 'W' : 'E');

/* ---------- layers ---------- */
export const LAYERS = {
  dens: { label: 'Crater density', cat: true, thr: true, pal: 'viridis', dom: 'pct', fmt: (v) => f(v, 1), unit: 'craters per 1,000 km²',
    note: 'Craters per 1,000 km² of true surface. Bright means crowded, dark means sparse.' },
  z: { label: 'Hotspot score (Gi*)', cat: true, thr: true, pal: 'div', dom: [-6, 6], fmt: (v) => f(v, 2), unit: 'Z-score',
    note: 'How unusual a place and its neighbours (within 61 km) are. Red is a hotspot, blue a coldspot. Beyond ±2.58 counts as significant at the 1% level.' },
  robust: { label: 'Hotspots that pass the permutation test', cat: false, thr: true, pal: null, dom: [0, 3], fmt: (v) => f(v, 0) + ' of 3', unit: 'catalogues', polarOnly: true,
    note: 'Polar cells only. How many of the three catalogues still show a hotspot after densities are shuffled within each latitude ring (the stricter test).' },
  cv: { label: 'Disagreement between catalogues', cat: false, thr: true, pal: 'cv', dom: 'pct', fmt: (v) => f(v, 2), unit: 'spread ÷ average',
    note: 'How much the three catalogues disagree about this place: the spread of their densities divided by the average. Dark means they disagree.' },
  elev: { label: 'Elevation', cat: false, thr: false, pal: 'terrain', dom: 'pct', fmt: (v) => f(v, 0), unit: 'm',
    note: 'Mean elevation from the 118 m lunar elevation model, in metres.' },
  slope: { label: 'Slope', cat: false, thr: false, pal: 'slope', dom: 'pct', fmt: (v) => f(v, 1), unit: 'degrees',
    note: 'Mean ground slope. Not computed poleward of 80°, those places are shown grey.' },
  sun: { label: 'Average sunlight fraction', cat: false, thr: false, pal: 'sun', dom: 'pct', fmt: (v) => f(v * 100, 0) + '%', unit: 'share of time lit', polarOnly: true,
    note: 'Polar cells only. How much of the time the ground is lit by the Sun, averaged over each 20 km cell.' },
  psr: { label: 'Permanent shadow fraction', cat: false, thr: false, pal: 'shadow', dom: [0, 1], fmt: (v) => f(v * 100, 0) + '%', unit: 'share of cell', polarOnly: true,
    note: 'Polar cells only. The share of each cell that never sees the Sun.' }
};
export const LAYER_ORDER = ['dens', 'z', 'robust', 'cv', 'elev', 'slope', 'sun', 'psr'];

/* ---------- build from the JSON ---------- */
export function build(raw) {
  const T = raw.tiles, C = raw.cells;
  const nT = T.tile_id.length;
  for (let i = 0; i < nT; i++) {
    const t = { kind: 'tile', i, lat0: T.lat_min[i], lat1: T.lat_max[i], lon0: T.lon_min[i], lon1: T.lon_max[i], area: T.area_km2[i],
      elev: T.mean_elev_m[i], slope: T.mean_slope_deg[i], sun: null, psr: null };
    t.lat = (t.lat0 + t.lat1) / 2; t.lon = (t.lon0 + t.lon1) / 2;
    CATS.forEach((ct) => { const k = ct.k;
      t['n1_' + k] = T['n_ge1_' + k][i]; t['n2_' + k] = T['n_ge2_' + k][i]; t['n5_' + k] = T['n_ge5_' + k][i]; t['n20_' + k] = T['n_ge20_' + k][i];
      t['d1_' + k] = T['dens1_' + k][i]; t['d2_' + k] = T['dens2_' + k][i]; t['z1_' + k] = T['z1_' + k][i]; t['z2_' + k] = T['z2_' + k][i]; });
    D.tiles.push(t);
    D.tileGrid[Math.round((t.lat0 + 90) / 5) * 72 + Math.round((t.lon0 + 180) / 5)] = i;
  }
  const nC = C.id.length, tmpN = [], tmpS = [];
  for (let i = 0; i < nC; i++) {
    const o = { kind: 'cell', pole: C.pole[i], id: C.id[i], x: C.x[i], y: C.y[i], lat: C.lat[i], lon: C.lon[i], area: C.area_km2[i],
      ix: C.ix[i], iy: C.iy[i], sun: C.sun_mean[i], psr: C.psr_frac[i], elev: C.elev_mean_m[i], slope: C.slope_mean_deg[i] };
    CATS.forEach((ct) => { const k = ct.k;
      [1, 2].forEach((th) => {
        o['n' + th + '_' + k] = C['n' + th + '_' + k][i]; o['d' + th + '_' + k] = C['d' + th + '_' + k][i]; o['z' + th + '_' + k] = C['z' + th + '_' + k][i];
        o['pA' + th + '_' + k] = C['pA' + th + '_' + k][i]; o['pB' + th + '_' + k] = C['pB' + th + '_' + k][i]; }); });
    (o.pole === 'N' ? tmpN : tmpS).push(o);
  }
  D.N = tmpN; D.S = tmpS; D.cells = tmpN.concat(tmpS);
  D.cells.forEach((o, gi) => { o.gi = gi; D.cellGrid[o.pole][o.iy * NC + o.ix] = gi; });
}

/* ---------- which item sits at a latitude and longitude ---------- */
export function itemAt(lat, lon) {
  if (S.polar && Math.abs(lat) >= 65) {
    const north = lat > 0, rho = 2 * R * Math.tan((90 - Math.abs(lat)) / 2 * Math.PI / 180), a = lon * Math.PI / 180;
    const x = rho * Math.sin(a), y = (north ? -1 : 1) * rho * Math.cos(a);
    const ix = Math.floor((x + HALF) / CELL), iy = Math.floor((y + HALF) / CELL);
    if (ix >= 0 && iy >= 0 && ix < NC && iy < NC) {
      const gi = D.cellGrid[north ? 'N' : 'S'][iy * NC + ix];
      if (gi >= 0) return D.cells[gi];
    }
  }
  const col = Math.max(0, Math.min(71, Math.floor((lon + 180) / 5))), row = Math.max(0, Math.min(35, Math.floor((lat + 90) / 5)));
  const ti = D.tileGrid[row * 72 + col];
  return ti >= 0 ? D.tiles[ti] : null;
}

/* ---------- values ---------- */
const mean = (a) => { let s = 0, n = 0; a.forEach((v) => { if (isNum(v)) { s += v; n++; } }); return n ? s / n : null; };
export function isRobust(it, k, th) { const z = it['z' + th + '_' + k], p = it['pA' + th + '_' + k]; return isNum(z) && z > 0 && isNum(p) && p <= 0.05; }
export function getVal(it, layer) {
  const th = S.thr, k = S.cat;
  switch (layer) {
    case 'dens': case 'z': {
      const p = layer === 'dens' ? 'd' : 'z';
      if (k === 'avg') return mean(CATS.map((c) => it[p + th + '_' + c.k]));
      return it[p + th + '_' + k];
    }
    case 'cv': {
      const a = CATS.map((c) => it['d' + th + '_' + c.k]), m = mean(a);
      if (!(m > 0)) return null;
      let v = 0; a.forEach((x) => { v += (x - m) * (x - m); });
      return Math.sqrt(v / a.length) / m;
    }
    case 'robust': {
      if (it.kind !== 'cell') return null;
      let n = 0; CATS.forEach((c) => { if (isRobust(it, c.k, th)) n++; }); return n;
    }
    default: return it[layer];
  }
}

const cache = {}, ringCache = {};
export function compute() {
  const key = [S.layer, S.cat, S.thr].join('|');
  if (cache[key]) return cache[key];
  const L = LAYERS[S.layer];
  const valsT = D.tiles.map((it) => getVal(it, S.layer));
  const valsC = D.cells.map((it) => getVal(it, S.layer));
  const num = (a) => a.filter(isNum).sort((x, y) => x - y);
  const sortedT = num(valsT), sortedC = num(valsC);
  const sortedN = num(D.cells.map((it, i) => it.pole === 'N' ? valsC[i] : null));
  const sortedS = num(D.cells.map((it, i) => it.pole === 'S' ? valsC[i] : null));
  const all = num(valsT.concat(valsC));
  const q = (p) => all[Math.min(all.length - 1, Math.max(0, Math.round(p * (all.length - 1))))];
  let dom = Array.isArray(L.dom) ? L.dom.slice() : [q(0.02), q(0.98)];
  if (L.dom === 'pct' && S.layer === 'sun') dom = [0, Math.max(dom[1], 0.01)];
  if (dom[1] <= dom[0]) dom[1] = dom[0] + 1e-9;
  const toColours = (vals) => {
    const out = new Uint32Array(vals.length);
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i];
      if (!isNum(v)) out[i] = NODATA32;
      else if (S.layer === 'robust') out[i] = ROBUST32[Math.max(0, Math.min(3, Math.round(v)))];
      else out[i] = LUT[L.pal][Math.max(0, Math.min(255, Math.round((v - dom[0]) / (dom[1] - dom[0]) * 255)))];
    }
    return out;
  };
  return (cache[key] = { valsT, valsC, sortedT, sortedC, sortedN, sortedS, dom, colT: toColours(valsT), colC: toColours(valsC) });
}
const sortedFor = (C, it) => it.kind === 'tile' ? C.sortedT : (it.pole === 'N' ? C.sortedN : C.sortedS);
const valueOf = (C, it) => it.kind === 'tile' ? C.valsT[it.i] : C.valsC[it.gi];
function percentile(sorted, v) {
  let lo = 0, hi = sorted.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] < v) lo = mid + 1; else hi = mid; }
  return sorted.length ? lo / sorted.length : 0;
}
const ringKey = (it) => it.kind === 'tile' ? it.lat0 : Math.floor(Math.abs(it.lat) / 5);
function ringMean(it, k, th) {
  const name = it.kind === 'tile' ? 'T' : it.pole, key = [name, k, th].join('|');
  if (ringCache[key]) return ringCache[key];
  const items = it.kind === 'tile' ? D.tiles : (it.pole === 'N' ? D.N : D.S), sum = {}, cnt = {};
  items.forEach((x) => { const v = x['d' + th + '_' + k]; if (!isNum(v)) return; const r = ringKey(x); sum[r] = (sum[r] || 0) + v; cnt[r] = (cnt[r] || 0) + 1; });
  const out = {}; Object.keys(sum).forEach((r) => { out[r] = sum[r] / cnt[r]; });
  return (ringCache[key] = out);
}

/* ---------- legend ---------- */
export const catName = () => S.cat === 'avg' ? 'Average of the three' : CATS.filter((c) => c.k === S.cat)[0].n;
export function legendHtml() {
  const L = LAYERS[S.layer], C = compute();
  if (S.layer === 'robust') {
    let h = '<div class="title">' + L.label + ' (' + (S.thr === 1 ? '1 km' : '2 km') + ' and larger)</div><div class="swatches">';
    ['0 catalogues', '1', '2', 'All 3'].forEach((t, i) => { h += '<span><i style="background:' + ROBUST[i] + '"></i>' + t + '</span>'; });
    return h + '</div>';
  }
  let title = L.label;
  if (L.cat) title += ' · ' + catName();
  if (L.thr) title += ' · ' + (S.thr === 1 ? '1 km' : '2 km') + ' and larger';
  const clip = L.dom === 'pct';
  return '<div class="title">' + title + ' (' + L.unit + ')</div><div class="bar" style="background:' + gradCss(L.pal) + '"></div>' +
    '<div class="ticks"><span>' + (clip ? '≤ ' : '') + L.fmt(C.dom[0]) + '</span><span>' + (clip ? '≥ ' : '') + L.fmt(C.dom[1]) + '</span></div>';
}

/* ---------- statistics panel ---------- */
const slopeText = (it) => isNum(it.slope) ? f(it.slope, 1) + '°' : '<span style="color:var(--muted)">not computed poleward of 80°</span>';
function histSvg(C, L, v) {
  const robust = S.layer === 'robust', nb = robust ? 4 : 24;
  let lo = C.dom[0], hi = C.dom[1];
  if (robust) { lo = -0.5; hi = 3.5; }
  const bins = new Array(nb).fill(0), all = C.sortedT.length ? C.sortedT.concat(C.sortedC) : C.sortedC;
  all.forEach((x) => { const b = Math.floor((x - lo) / (hi - lo) * nb); bins[Math.max(0, Math.min(nb - 1, b))]++; });
  const mx = Math.max.apply(null, bins) || 1, W = 300, H = 64, bw = W / nb;
  let out = '<svg class="hist" viewBox="0 0 300 84" role="img" aria-label="Distribution of values with the selected place marked">';
  for (let i = 0; i < nb; i++) { const bh = bins[i] / mx * H; out += '<rect x="' + (i * bw + 0.5) + '" y="' + (H - bh + 6) + '" width="' + (bw - 1) + '" height="' + bh + '" rx="1.5" fill="var(--line)"/>'; }
  if (isNum(v)) {
    const x = (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * W;
    out += '<line x1="' + x + '" y1="2" x2="' + x + '" y2="' + (H + 6) + '" stroke="var(--accent)" stroke-width="2.5"/><path d="M' + (x - 5) + ' 0 L' + (x + 5) + ' 0 L' + x + ' 7 Z" fill="var(--accent)"/>';
  }
  const l1 = robust ? '0' : L.fmt(C.dom[0]), l2 = robust ? '3' : L.fmt(C.dom[1]);
  return out + '<text x="0" y="80">' + l1 + '</text><text x="300" y="80" text-anchor="end">' + l2 + '</text></svg>';
}
const zPill = (z) => !isNum(z) ? '<span class="pill ns">no value</span>' : z > SIG ? '<span class="pill hot">hotspot</span>' : z < -SIG ? '<span class="pill cold">coldspot</span>' : '<span class="pill ns">not significant</span>';
function permPill(p, z) {
  if (!isNum(p)) return '–';
  const small = '<small style="color:var(--muted)">p=' + f(p, 3) + '</small>';
  return (isNum(z) && z > 0 && p <= 0.05) ? '<span class="pill ok">passes</span> ' + small : '<span class="pill ns">no</span> ' + small;
}
function reading(it, rm) {
  const th = S.thr, out = [], noun = it.kind === 'cell' ? 'cell' : 'tile';
  const ratios = CATS.map((c) => { const m = rm[c.k][ringKey(it)], d = it['d' + th + '_' + c.k]; return m > 0 && isNum(d) ? d / m : null; }).filter(isNum);
  if (ratios.length === 3) {
    const mn = Math.min.apply(null, ratios), mx = Math.max.apply(null, ratios);
    if (mn >= 1.25) out.push('All three catalogues find this ' + noun + ' crowded: ' + f(mn, 2) + ' to ' + f(mx, 2) + ' times the average of its latitude band.');
    else if (mx <= 0.8) out.push('All three catalogues find this ' + noun + ' sparse: only ' + f(mn, 2) + ' to ' + f(mx, 2) + ' times the average of its latitude band.');
    else if (mx - mn > 0.5) out.push('The catalogues disagree here. Compared with its latitude band, density runs from ' + f(mn, 2) + ' times to ' + f(mx, 2) + ' times the average depending on the catalogue.');
    else out.push('Crater density is close to the usual level for its latitude band in all three catalogues.');
  }
  let nHot = 0, nCold = 0, nRob = 0;
  CATS.forEach((c) => { const z = it['z' + th + '_' + c.k]; if (isNum(z)) { if (z > SIG) nHot++; if (z < -SIG) nCold++; } if (it.kind === 'cell' && isRobust(it, c.k, th)) nRob++; });
  if (nHot > 0) {
    let s = 'Flagged as a hotspot by ' + nHot + ' of 3 catalogues.';
    if (it.kind === 'cell') s += ' ' + (nRob ? nRob + ' of 3 still show it after the permutation test.' : 'None keeps it after the permutation test.');
    if (it.pole === 'N') s += ' In the north much of this comes from the steady rise in density towards the pole.';
    out.push(s);
  } else if (nCold > 0) out.push('Flagged as a coldspot by ' + nCold + ' of 3 catalogues, a place with fewer craters than its surroundings would suggest.');
  else out.push('Not a hotspot or coldspot in any of the catalogues.');
  if (it.kind === 'cell' && isNum(it.psr) && it.psr > 0.05) out.push('About ' + f(it.psr * 100, 0) + '% of this cell is in permanent shadow. Shadowed ground holds fewer small craters in these data, which may be real or may reflect how hard craters are to see in the dark.');
  return out.map((t) => '<p>' + t + '</p>').join('');
}

export function panelHtml(it) {
  if (!it) return '<div class="status">Click the Moon to choose a place.</div>';
  const L = LAYERS[S.layer], C = compute(), th = S.thr, isCell = it.kind === 'cell';
  const title = isCell ? 'Polar cell · 20 km × 20 km' : 'Tile · 5° × 5°';
  const where = isCell ? fl(it.lat) + ', ' + fo(it.lon) + ' · ' + f(it.area, 0) + ' km²'
    : fl0(it.lat0) + ' to ' + fl0(it.lat1) + ' · ' + fo0(it.lon0) + ' to ' + fo0(it.lon1) + ' · ' + f(it.area, 0) + ' km²';
  const head = '<h2>' + title + '</h2><p class="sub">' + where + '</p>';
  const thTxt = th === 1 ? '1 km' : '2 km';

  // tab: layer value
  const v = valueOf(C, it);
  let t1 = '<h3>' + L.label + (L.cat ? ' · ' + catName() : '') + (L.thr ? ' · ' + thTxt + '+' : '') + '</h3>';
  if (isNum(v)) {
    const pc = Math.round(percentile(sortedFor(C, it), v) * 100), n = sortedFor(C, it).length;
    t1 += '<div class="big"><b>' + L.fmt(v) + '</b><span>' + L.unit + '</span></div>' +
      '<div class="rank">Higher than ' + pc + '% of the ' + f(n, 0) + ' ' + (isCell ? 'polar cells in the ' + (it.pole === 'N' ? 'north' : 'south') : 'tiles') + '.</div>' + histSvg(C, L, v);
  } else {
    t1 += '<div class="rank">No value here' + (L.polarOnly ? ' (this layer exists only for the polar cells).' : S.layer === 'slope' ? ' (slope is not computed poleward of 80°).' : '.') + '</div>';
  }

  // tab: catalogues
  const rm = {}; let maxd = 0;
  CATS.forEach((c) => { rm[c.k] = ringMean(it, c.k, th); const d = it['d' + th + '_' + c.k]; if (isNum(d)) maxd = Math.max(maxd, d); const m = rm[c.k][ringKey(it)]; if (isNum(m)) maxd = Math.max(maxd, m); });
  maxd = maxd || 1;
  let t2 = '<h3>Crater density, ' + thTxt + ' and larger</h3><div class="bars">';
  CATS.forEach((c) => {
    const d = it['d' + th + '_' + c.k], m = rm[c.k][ringKey(it)], ratio = m > 0 ? d / m : null;
    t2 += '<div class="brow"><div class="nm"><span class="dot" style="background:var(' + c.v + ')"></span>' + c.n + '</div>' +
      '<div class="tr"><div class="fl" style="width:' + (isNum(d) ? Math.max(1, d / maxd * 100) : 0) + '%;background:var(' + c.v + ')"></div>' +
      (isNum(m) ? '<div class="tick" style="left:calc(' + (m / maxd * 100) + '% - 1px)" title="Average of this latitude band"></div>' : '') + '</div>' +
      '<div class="v">' + f(d, 1) + ' <small>' + (isNum(ratio) ? '× ' + f(ratio, 2) : '') + '</small></div></div>';
  });
  t2 += '</div><div class="hint">Black tick: average of ' + (isCell ? 'cells' : 'tiles') + ' in the same latitude band. × is the ratio to that average.</div>';
  const cols = isCell ? [1, 2] : [1, 2, 5, 20];
  t2 += '<h3>Crater counts in this place</h3><div class="tscroll"><table><thead><tr><th>Catalogue</th>' + cols.map((c) => '<th>≥ ' + c + ' km</th>').join('') + '</tr></thead><tbody>';
  CATS.forEach((c) => { t2 += '<tr><td><span class="dot" style="background:var(' + c.v + ')"></span>' + c.n + '</td>' + cols.map((k) => '<td>' + f(it['n' + k + '_' + c.k], 0) + '</td>').join('') + '</tr>'; });
  t2 += '</tbody></table></div>';

  // tab: hotspots
  let t3 = '<h3>Hotspot test, ' + thTxt + ' and larger</h3><div class="tscroll"><table><thead><tr><th>Catalogue</th><th>Gi* Z</th><th>Result</th>' +
    (isCell ? '<th>Test: rings</th><th>Test: rings + sun</th>' : '') + '</tr></thead><tbody>';
  CATS.forEach((c) => {
    const z = it['z' + th + '_' + c.k];
    t3 += '<tr><td><span class="dot" style="background:var(' + c.v + ')"></span>' + c.n + '</td><td>' + f(z, 2) + '</td><td>' + zPill(z) + '</td>' +
      (isCell ? '<td>' + permPill(it['pA' + th + '_' + c.k], z) + '</td><td>' + permPill(it['pB' + th + '_' + c.k], z) + '</td>' : '') + '</tr>';
  });
  t3 += '</tbody></table></div>' + (isCell ? '' : '<div class="hint">The permutation test was run for the polar cells only.</div>');

  // tab: terrain
  const t4 = '<h3>Terrain' + (isCell ? ' and light' : '') + '</h3><dl class="kv"><div><dt>Elevation</dt><dd>' + f(it.elev, 0) + ' m</dd></div><div><dt>Slope</dt><dd>' + slopeText(it) + '</dd></div>' +
    (isCell ? '<div><dt>Average sunlight fraction</dt><dd>' + f(it.sun * 100, 0) + '%</dd></div><div><dt>Permanent shadow fraction</dt><dd>' + f(it.psr * 100, 0) + '%</dd></div>' : '') + '</dl>';

  const t5 = '<h3>What this says</h3><div class="reading">' + reading(it, rm) + '</div>';

  const tabs = [{ id: 'val', title: 'Layer', body: t1 }, { id: 'cat', title: 'Catalogues', body: t2 }, { id: 'hot', title: 'Hotspots', body: t3 },
    { id: 'terrain', title: 'Terrain', body: t4 }, { id: 'reading', title: 'Summary', body: t5 }];
  const cur = tabs.filter((t) => t.id === S.tab)[0] || tabs[0];
  const bar = '<div class="tabs" role="tablist" aria-label="Details for the selected place">' + tabs.map((t) =>
    '<button type="button" role="tab" id="tab-' + t.id + '" data-tab="' + t.id + '" aria-selected="' + (t === cur) + '" aria-controls="tabpanel">' + t.title + '</button>').join('') + '</div>';
  return head + bar + '<div id="tabpanel" role="tabpanel" aria-labelledby="tab-' + cur.id + '">' + cur.body + '</div>';
}

export function strongestTile() {
  let best = null, bv = -Infinity;
  D.tiles.forEach((it) => { const z = it.z1_robbins; if (isNum(z) && z > bv && Math.abs(it.lat) < 60) { bv = z; best = it; } });
  return best;
}
