import './style.css';
import { createGlobe } from './globe.js';
import {
  S, D, CATS, LAYERS, LAYER_ORDER, build, itemAt, compute, getVal, legendHtml, panelHtml, strongestTile,
  isNum, fl, fo
} from './model.js';

const $ = (id) => document.getElementById(id);
let globe = null;

/* ---------- controls ---------- */
function fillSelects() {
  const ls = $('layerSel');
  LAYER_ORDER.forEach((id) => { const o = document.createElement('option'); o.value = id; o.textContent = LAYERS[id].label; ls.appendChild(o); });
  ls.value = S.layer;
  const cs = $('catSel');
  CATS.concat([{ k: 'avg', n: 'Average of the three' }]).forEach((c) => { const o = document.createElement('option'); o.value = c.k; o.textContent = c.n; cs.appendChild(o); });
  cs.value = S.cat;
}
function syncControls() {
  const L = LAYERS[S.layer];
  $('catSel').disabled = !L.cat;
  $('thrSel').disabled = !L.thr;
  $('layerNote').textContent = L.note;
}

/* ---------- drawing ---------- */
function repaint() {
  syncControls();
  const C = compute();
  globe.repaint(C.colT, C.colC, S.polar);
  $('legend').innerHTML = legendHtml();
  renderPanel();
}
function itemHalf(it) { return it.kind === 'tile' ? 2.5 * Math.PI / 180 : 10000 / 1737400; }
function select(it) {
  S.sel = it;
  if (it) globe.setSelected(it.lat, it.lon, itemHalf(it)); else globe.setSelected(null);
  renderPanel();
  $('readout').textContent = it ? 'Selected: ' + fl(it.lat) + ', ' + fo(it.lon) : 'Drag to rotate, scroll to zoom, click to select.';
}
function renderPanel() {
  $('panel').innerHTML = panelHtml(S.sel);
}

/* ---------- hover tooltip ---------- */
const tip = $('tip');
function onHover(p, ev) {
  if (!p || !ev) { tip.style.display = 'none'; return; }
  const it = itemAt(p.lat, p.lon);
  if (!it) { tip.style.display = 'none'; return; }
  const L = LAYERS[S.layer], v = getVal(it, S.layer);
  tip.textContent = fl(p.lat) + ', ' + fo(p.lon) + ' · ' + L.label + ': ' + (isNum(v) ? L.fmt(v) : 'no value');
  const box = $('globe').getBoundingClientRect();
  let x = ev.clientX - box.left + 14, y = ev.clientY - box.top + 14;
  tip.style.display = 'block';
  const w = tip.offsetWidth; if (x + w > box.width - 6) x = ev.clientX - box.left - w - 14;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}

/* ---------- start ---------- */
async function start() {
  try {
    const r = await fetch(import.meta.env.BASE_URL + 'dashboard_data.json');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    build(await r.json());
  } catch (err) {
    $('panel').innerHTML = '<div class="status">The crater data could not be loaded (' + String(err.message || err) + '). Reload the page to try again.</div>';
    return;
  }
  fillSelects();
  globe = createGlobe($('globe'), (lat, lon) => { const it = itemAt(lat, lon); if (it) select(it); }, onHover);
  globe.ready();

  $('layerSel').addEventListener('change', (e) => { S.layer = e.target.value; repaint(); });
  $('catSel').addEventListener('change', (e) => { S.cat = e.target.value; repaint(); });
  $('thrSel').addEventListener('change', (e) => { S.thr = parseInt(e.target.value, 10); repaint(); });
  $('chkPolar').addEventListener('change', (e) => { S.polar = e.target.checked; repaint(); if (S.sel) { const it = itemAt(S.sel.lat, S.sel.lon); if (it) select(it); } });
  $('chkGrid').addEventListener('change', (e) => globe.setGrid(e.target.checked));
  $('chkSpin').addEventListener('change', (e) => globe.setAutoRotate(e.target.checked));
  $('btnNear').addEventListener('click', () => globe.look(0, 0));
  $('btnFar').addEventListener('click', () => globe.look(0, 180));
  $('btnN').addEventListener('click', () => globe.look(90, 0));
  $('btnS').addEventListener('click', () => globe.look(-90, 0));
  $('panel').addEventListener('click', (e) => {
    const b = e.target.closest('[data-tab]');
    if (b) { S.tab = b.dataset.tab; renderPanel(); const nb = $('tab-' + S.tab); if (nb) nb.focus(); }
  });

  repaint();
  const first = strongestTile();
  if (first) { globe.look(first.lat * 0.5, first.lon, 3.1); select(first); }
}
start();
