/* RipeCheck front end. No libraries and no network calls except to this server. */
(() => {
'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const DEFAULT_BOX = [0.08, 0.08, 0.92, 0.92];
const GRADE_KEY = { 'Unripe': 'unripe', 'Ripe': 'ripe', 'Overripe': 'overripe', 'Uncertain': 'uncertain', 'Not gradable': 'none' };
const SLIDERS = [
  { k: 't_dark', label: 'Unripe when dark share is at least', min: 0.05, max: 0.95, step: 0.01 },
  { k: 't_orange', label: 'Ripe when orange share is at least', min: 0.05, max: 0.95, step: 0.01 },
  { k: 't_over', label: 'Overripe when over-ripe score is at least', min: 0.05, max: 1, step: 0.01 },
  { k: 'margin_min', label: 'Say uncertain when the margin is below', min: 0, max: 0.6, step: 0.01 },
];
const TOGGLES = [
  { k: 'bilateral', label: 'Smooth noise first (bilateral filter)' },
  { k: 'clahe', label: 'Boost local contrast (CLAHE) for edges' },
  { k: 'clahe_colour', label: 'Measure colour after the contrast boost' },
  { k: 'white_balance', label: 'Balance colour (gray-world). Risky when fruit fills the frame' },
  { k: 'glare_filter', label: 'Ignore glare and deep shadow' },
  { k: 'gates', label: 'Run photo checks before grading' },
];
const STEPS = [
  ['crop', 'Your box', 'The cropped photo.'],
  ['filtered', 'Bilateral filter', 'Smooths noise and keeps edges.'],
  ['enhanced', 'CLAHE', 'Local contrast boost. Used for the edge map.'],
  ['mask', 'Cut-out', 'Segmentation, then opening, closing and the largest piece.'],
  ['edges', 'Canny edges', 'Edge density inside the bunch is one feature.'],
  ['colormap', 'Colour classes', 'Dark, orange, red and other pixels inside the bunch.'],
  ['overlay', 'Outline', 'Gaps inside the outline feed the over-ripe score.'],
];

const S = { blob: null, name: '', img: null, box: DEFAULT_BOX.slice(), res: null, view: 'box', cfg: null, thr: {}, opts: {},
  views: [], blocks: [], seq: 0, fused: null, batchFiles: [], sample: null };

const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const pct = (v, d = 0) => (v * 100).toFixed(d) + '%';
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode: keep in memory only */ } },
};

/* ------------------------------------------------------------------ start */
async function init() {
  try {
    S.cfg = await (await fetch('/api/config')).json();
  } catch {
    showError('RipeCheck could not reach its server. Start it again and reload this page.');
    return;
  }
  S.opts = { ...S.cfg.params };
  S.thr = Object.fromEntries(SLIDERS.map((s) => [s.k, S.cfg.thresholds[s.k]]));
  buildSettings();
  await buildSamples();
  S.blocks = store.get('ripecheck.blocks') || [];
  bind();
  layoutCanvas(); drawCanvas();
  renderViews(); renderBlocks();
  $('#rulenote').textContent = 'Grading rules: ' + S.cfg.thresholds.note + '.';
}

function bind() {
  $('#btn-camera').onclick = () => $('#in-camera').click();
  $('#btn-file').onclick = () => $('#in-file').click();
  ['#in-camera', '#in-file'].forEach((id) => $(id).addEventListener('change', (e) => {
    const f = e.target.files[0];
    if (f) { S.sample = null; markSample(); setPhoto(f, f.name); }
    e.target.value = '';
  }));
  $$('.pages button').forEach((b) => b.onclick = () => showPage(b.dataset.page));
  $$('#viewtabs [role=tab]').forEach((b) => b.onclick = () => setView(b.dataset.view));
  $('#btn-reset-box').onclick = () => { S.box = DEFAULT_BOX.slice(); drawCanvas(); analyse(); };
  $('#btn-reset-rules').onclick = () => {
    SLIDERS.forEach((s) => S.thr[s.k] = S.cfg.thresholds[s.k]);
    $$('#sliders input').forEach((el) => { el.value = S.thr[el.dataset.k]; el.dispatchEvent(new Event('input')); });
  };
  $('#btn-add').onclick = addView;
  $('#btn-clear-views').onclick = () => { S.views = []; S.fused = null; renderViews(); renderFused(); };
  $('#save-form').onsubmit = saveBunch;
  $('#btn-export').onclick = exportCsv;
  $('#btn-clear-blocks').onclick = () => { if (confirm('Clear all blocks and saved bunches?')) { S.blocks = []; store.set('ripecheck.blocks', S.blocks); renderBlocks(); } };
  $('#in-batch').onchange = (e) => { S.batchFiles = [...e.target.files]; batchLabel(); };
  $('#btn-batch').onclick = runBatch;
  $('#btn-batch-samples').onclick = useSamplesForBatch;
  $('#viewlist').onclick = (e) => {
    const b = e.target.closest('button[data-i]');
    if (b) { S.views.splice(+b.dataset.i, 1); renderViews(); fuse(); }
  };
  window.addEventListener('resize', () => { layoutCanvas(); drawCanvas(); });
  bindCanvas();
}

/* ------------------------------------------------------------------ pages */
function showPage(name) {
  $$('.pages button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.page === name)));
  ['grade', 'views', 'blocks', 'test'].forEach((p) => { $('#p-' + p).hidden = p !== name; });
  if (name === 'grade') { layoutCanvas(); drawCanvas(); }
}

/* ------------------------------------------------------------------ settings */
function buildSettings() {
  const seg = $('#seg-mode');
  const names = { grabcut: 'Smart cut-out from your box (GrabCut)', hsv: 'Colour rules (HSV)', kmeans: 'Colour clusters (experimental)' };
  seg.innerHTML = S.cfg.seg_modes.map((m) => `<option value="${m}">${names[m] || m}</option>`).join('');
  seg.value = S.opts.seg_mode;
  seg.onchange = () => { S.opts.seg_mode = seg.value; schedule(); };
  $('#toggles').innerHTML = TOGGLES.map((t) => `<label class="toggle"><input type="checkbox" data-k="${t.k}" ${S.opts[t.k] ? 'checked' : ''}><span>${t.label}</span></label>`).join('');
  $$('#toggles input').forEach((el) => el.onchange = () => { S.opts[el.dataset.k] = el.checked; schedule(); });
  $('#sliders').innerHTML = SLIDERS.map((s) => `<div class="slider"><label for="sl-${s.k}"><span>${s.label}</span><output id="out-${s.k}">${S.thr[s.k].toFixed(2)}</output></label>
    <input id="sl-${s.k}" data-k="${s.k}" type="range" min="${s.min}" max="${s.max}" step="${s.step}" value="${S.thr[s.k]}"></div>`).join('');
  $$('#sliders input').forEach((el) => el.oninput = () => {
    S.thr[el.dataset.k] = +el.value;
    $('#out-' + el.dataset.k).textContent = (+el.value).toFixed(2);
    schedule(); if (S.views.length) fuse();
  });
}

async function buildSamples() {
  let names = [];
  try { names = await (await fetch('/api/samples')).json(); } catch { /* no samples */ }
  $('#samples').innerHTML = names.map((n) => `<button class="sample" type="button" data-name="${esc(n)}" aria-pressed="false">
    <img src="/samples/${encodeURIComponent(n)}" alt="" loading="lazy"><span>${esc(n.replace(/\.[a-z]+$/i, '').replace(/_/g, ' '))}</span></button>`).join('');
  $$('.sample').forEach((b) => b.onclick = async () => {
    const n = b.dataset.name;
    const blob = await (await fetch('/samples/' + encodeURIComponent(n))).blob();
    S.sample = n; markSample(); setPhoto(blob, n);
  });
}
function markSample() { $$('.sample').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.name === S.sample))); }

/* ------------------------------------------------------------------ photo and box */
function toBlob(img) {
  const k = Math.min(1, 1280 / img.naturalWidth);
  const c = document.createElement('canvas');
  c.width = Math.round(img.naturalWidth * k); c.height = Math.round(img.naturalHeight * k);
  c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
  return new Promise((res) => c.toBlob(res, 'image/jpeg', 0.9));
}

function setPhoto(file, name) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = async () => {
    S.img = img; S.name = name; S.blob = await toBlob(img); URL.revokeObjectURL(url);
    S.box = DEFAULT_BOX.slice(); S.res = null;
    $('#empty').hidden = true;
    $('#hint').textContent = 'Drag on the photo to box the whole bunch. The grade updates when you let go.';
    $('#btn-add').disabled = true;
    setView('box'); layoutCanvas(); drawCanvas(); analyse();
  };
  img.onerror = () => showError('That file could not be opened as a photo. Try a JPG or PNG.');
  img.src = url;
}

function layoutCanvas() {
  const cv = $('#cv'), w = $('#stage').clientWidth || 600, dpr = Math.min(window.devicePixelRatio || 1, 2);
  const ratio = S.img ? S.img.naturalHeight / S.img.naturalWidth : 0.6;
  cv.width = Math.round(w * dpr); cv.height = Math.round(w * ratio * dpr);
}

function drawCanvas() {
  const cv = $('#cv'), ctx = cv.getContext('2d'), W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  if (!S.img) return;
  ctx.drawImage(S.img, 0, 0, W, H);
  const [x0, y0, x1, y1] = S.box;
  ctx.fillStyle = 'rgba(12, 32, 22, .58)';
  ctx.fillRect(0, 0, W, y0 * H); ctx.fillRect(0, y1 * H, W, H - y1 * H);
  ctx.fillRect(0, y0 * H, x0 * W, (y1 - y0) * H); ctx.fillRect(x1 * W, y0 * H, W - x1 * W, (y1 - y0) * H);
  ctx.strokeStyle = '#FFB27A'; ctx.lineWidth = Math.max(2, W / 320);
  ctx.strokeRect(x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H);
}

function bindCanvas() {
  const cv = $('#cv');
  let drag = null;
  const at = (e) => { const r = cv.getBoundingClientRect(); return [clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height)]; };
  cv.addEventListener('pointerdown', (e) => { if (!S.img) return; cv.setPointerCapture(e.pointerId); drag = { start: at(e), prev: S.box.slice() }; });
  cv.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const [x, y] = at(e), [sx, sy] = drag.start;
    S.box = [Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)];
    drawCanvas();
  });
  const end = (go) => {
    if (!drag) return;
    const [x0, y0, x1, y1] = S.box;
    if (x1 - x0 < 0.05 || y1 - y0 < 0.05) S.box = drag.prev;
    drag = null; drawCanvas(); if (go) analyse();
  };
  cv.addEventListener('pointerup', () => end(true));
  cv.addEventListener('pointercancel', () => end(false));
  cv.addEventListener('keydown', (e) => {
    if (!S.img || !e.key.startsWith('Arrow')) return;
    e.preventDefault();
    const st = 0.02, dx = e.key === 'ArrowRight' ? st : e.key === 'ArrowLeft' ? -st : 0, dy = e.key === 'ArrowDown' ? st : e.key === 'ArrowUp' ? -st : 0;
    let [x0, y0, x1, y1] = S.box;
    if (e.shiftKey) { x1 = clamp(x1 + dx, x0 + 0.05); y1 = clamp(y1 + dy, y0 + 0.05); }
    else {
      if (x0 + dx >= 0 && x1 + dx <= 1) { x0 += dx; x1 += dx; }
      if (y0 + dy >= 0 && y1 + dy <= 1) { y0 += dy; y1 += dy; }
    }
    S.box = [x0, y0, x1, y1]; drawCanvas(); schedule();
  });
}

/* ------------------------------------------------------------------ analysis */
let timer = null;
function schedule() { clearTimeout(timer); timer = setTimeout(analyse, 300); }

function showError(msg) { const el = $('#error'); el.textContent = msg; el.hidden = !msg; }

async function analyse() {
  if (!S.blob) return;
  const id = ++S.seq;
  $('#busy').hidden = false; showError('');
  const fd = new FormData();
  fd.append('image', S.blob, S.name || 'photo.jpg');
  fd.append('box', S.box.map((v) => v.toFixed(4)).join(','));
  fd.append('opts', JSON.stringify(S.opts));
  fd.append('thr', JSON.stringify(S.thr));
  try {
    const r = await fetch('/api/analyse', { method: 'POST', body: fd });
    const j = await r.json();
    if (id !== S.seq) return;
    if (!j.ok) { showError(j.error || 'The photo could not be graded.'); return; }
    S.res = j; renderSlip(j); renderView();
  } catch {
    if (id === S.seq) showError('RipeCheck could not reach its server. Is it still running?');
  } finally {
    if (id === S.seq) $('#busy').hidden = true;
  }
}

function renderSlip(j) {
  const g = $('#grade'), key = GRADE_KEY[j.grade] || 'none';
  const changed = g.dataset.grade !== key || g.textContent !== j.grade;
  g.textContent = j.grade; g.dataset.grade = key;
  if (changed) { g.classList.remove('stamp'); void g.offsetWidth; g.classList.add('stamp'); }
  $('#advice').textContent = j.advice;
  const lean = $('#lean');
  if (j.grade === 'Uncertain') {
    lean.hidden = false;
    lean.textContent = j.leaning ? `Leaning ${j.leaning.toLowerCase()}, but the margin is only ${pct(j.margin)}.` : 'No grading rule fits these colours.';
  } else if (j.grade === 'Not gradable') { lean.hidden = true; }
  else { lean.hidden = false; lean.textContent = `Margin ${pct(j.margin)} past the nearest rule boundary.`; }

  const gradable = j.grade !== 'Not gradable';
  $('#ruler').hidden = !(j.features.valid_ratio > 0);
  const L = S.cfg.legend, order = ['dark', 'orange', 'red', 'other'], label = { dark: 'Dark', orange: 'Orange', red: 'Red', other: 'Other' };
  $('#bar').innerHTML = order.map((k) => `<i style="width:${(j.composition[k] * 100).toFixed(2)}%;background:${L[k]}" title="${label[k]} ${pct(j.composition[k])}"></i>`).join('');
  $('#legend').innerHTML = order.map((k) => `<li><span class="sw" style="background:${L[k]}"></span>${label[k]} ${pct(j.composition[k])}</li>`).join('');

  $('#why').hidden = !gradable;
  $('#rules').innerHTML = j.rules.map((r) => `<li class="rule ${r.fired ? 'fired' : ''}">
    <div class="rule-head"><span>${esc(r.label)}</span><span>${pct(r.value)} against at least ${pct(r.threshold)}</span></div>
    <div class="meter"><i style="width:${clamp(r.value) * 100}%"></i><b style="left:${clamp(r.threshold) * 100}%"></b></div>
    <p>${esc(r.note)} ${r.fired ? 'This rule decided the grade.' : r.met ? 'Met, but an earlier rule applied.' : 'Not met.'}</p></li>`).join('');

  $('#checks-wrap').hidden = !j.gates.length;
  $('#checks').innerHTML = j.gates.map((c) => `<li class="${c.ok ? 'ok' : 'bad'}"><span></span><div>${esc(c.label)}
    <small>${c.ok ? '' : esc(c.fix) + ' '}${(c.value * (c.name === 'light' || c.name === 'sharp' ? 1 : 100)).toFixed(c.name === 'light' || c.name === 'sharp' ? 0 : 0)}${c.name === 'light' || c.name === 'sharp' ? '' : '%'}, needs ${esc(c.limit)}</small></div></li>`).join('');
  $('#btn-add').disabled = !gradable;
}

/* ------------------------------------------------------------------ views of the result */
function setView(v) {
  S.view = v;
  $$('#viewtabs [role=tab]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.view === v)));
  renderView();
}

function renderView() {
  const box = $('#viewbox'), cv = $('#cv'), v = S.view, r = S.res;
  cv.hidden = v !== 'box'; box.hidden = v === 'box';
  if (v === 'box') { layoutCanvas(); drawCanvas(); return; }
  if (!r) { box.innerHTML = '<p style="color:#BFD0C4;padding:24px">Grade a photo first.</p>'; return; }
  const im = r.images, L = S.cfg.legend;
  if (v === 'outline') box.innerHTML = `<img src="${im.overlay}" alt="Bunch outline on the cropped photo">`;
  else if (v === 'edges') box.innerHTML = `<img src="${im.edges}" alt="Canny edge map of the cropped photo">`;
  else if (v === 'colours') {
    box.innerHTML = `<img src="${im.colormap}" alt="Colour map of the bunch"><ul class="legend" style="background:#FBFCF8;margin:0;padding:10px 12px">${['dark', 'orange', 'red', 'other']
      .map((k) => `<li><span class="sw" style="background:${L[k]}"></span>${k[0].toUpperCase() + k.slice(1)} ${pct(r.composition[k])}</li>`).join('')}</ul>`;
  } else if (v === 'compare') {
    box.innerHTML = `<div class="wipe" id="wipe"><img src="${im.crop}" alt="Your cropped photo"><img class="top-img" src="${im.overlay}" alt="With the bunch outline">
      <span class="handle"></span><span class="cap l">Outline</span><span class="cap r">Your photo</span>
      <input type="range" min="0" max="100" value="50" aria-label="Slide to compare the photo and its outline"></div>`;
    const w = $('#wipe'); $('input', w).oninput = (e) => w.style.setProperty('--cut', e.target.value + '%');
  } else if (v === 'steps') {
    box.innerHTML = `<div class="steps">${STEPS.map(([k, t, d]) => `<figure><img src="${im[k]}" alt="${esc(t)}"><figcaption><b>${esc(t)}</b>${esc(d)}</figcaption></figure>`).join('')}</div>`;
  }
}

/* ------------------------------------------------------------------ views of one bunch */
function addView() {
  if (!S.res || S.res.grade === 'Not gradable') return;
  S.views.push({ name: S.name || 'photo', grade: S.res.grade, features: S.res.features, gates_ok: S.res.gates_ok });
  renderViews(); fuse();
  const b = $('#btn-add'); b.textContent = 'Added. Add another side, or open Views of a bunch';
  setTimeout(() => { b.textContent = 'Add as a view of this bunch'; }, 2200);
}

function renderViews() {
  const n = S.views.length, badge = $('#views-count');
  badge.hidden = !n; badge.textContent = n;
  $('#viewlist').innerHTML = S.views.map((v, i) => `<li><span>${esc(v.name)}</span><span class="chip" data-grade="${GRADE_KEY[v.grade] || ''}">${esc(v.grade)}</span>
    <button class="linkish" type="button" data-i="${i}" aria-label="Remove view ${i + 1}">Remove</button></li>`).join('');
  $('#views-empty').hidden = n > 0; $('#btn-clear-views').hidden = !n;
  if (!n) renderFused();
}

async function fuse() {
  if (!S.views.length) { S.fused = null; renderFused(); return; }
  try {
    const r = await fetch('/api/fuse', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ views: S.views.map((v) => ({ features: v.features, gates_ok: v.gates_ok })), thr: S.thr }) });
    const j = await r.json();
    S.fused = j.ok ? j : null;
  } catch { S.fused = null; }
  renderFused();
}

function renderFused(note) {
  const f = S.fused, g = $('#fused-grade');
  g.textContent = f ? f.grade : 'No views'; g.dataset.grade = f ? (GRADE_KEY[f.grade] || 'none') : 'none';
  $('#fused-advice').textContent = f ? f.advice : 'Add a view to see the combined grade.';
  const votes = f ? Object.entries(f.votes || {}).map(([k, v]) => `${v} ${k.toLowerCase()}`).join(', ') : '';
  $('#fused-votes').textContent = note || (f ? `Combined from ${f.used} of ${S.views.length} view${S.views.length > 1 ? 's' : ''}. Single-view grades: ${votes || 'none'}.` : '');
  $('#btn-save').disabled = !f || f.grade === 'Not gradable';
}

function saveBunch(e) {
  e.preventDefault();
  if (!S.fused) return;
  const id = ($('#block-id').value.trim() || 'A1').toUpperCase();
  S.blocks.push({ block: id, grade: S.fused.grade, views: S.views.length, time: new Date().toISOString() });
  store.set('ripecheck.blocks', S.blocks);
  S.views = []; S.fused = null; renderViews(); renderFused(`Saved to block ${id}. Ready for the next bunch.`); renderBlocks();
}

/* ------------------------------------------------------------------ block planner */
function blockStats() {
  const m = {};
  S.blocks.forEach((b) => {
    const s = (m[b.block] = m[b.block] || { block: b.block, unripe: 0, ripe: 0, overripe: 0, unsure: 0, n: 0 });
    const k = GRADE_KEY[b.grade]; s.n++;
    if (k === 'unripe' || k === 'ripe' || k === 'overripe') s[k]++; else s.unsure++;
  });
  const share = (s) => { const g = s.unripe + s.ripe + s.overripe; return g ? s.ripe / g : 0; };
  return Object.values(m).sort((a, b) => b.overripe - a.overripe || share(b) - share(a) || a.block.localeCompare(b.block, undefined, { numeric: true }));
}
const planFor = (s) => s.overripe ? 'Harvest first' : s.ripe ? 'Harvest' : s.unripe ? 'Wait' : 'Re-check';

function renderBlocks() {
  const rows = blockStats(), has = rows.length > 0;
  $('#blocks-empty').hidden = has; $('#block-table').hidden = !has;
  $('#btn-export').disabled = !has; $('#btn-clear-blocks').disabled = !has;
  const L = S.cfg ? S.cfg.legend : {};
  $('#tiles').innerHTML = rows.map((s) => `<div class="tile"><h3>${esc(s.block)}</h3>
    <div class="mini" role="img" aria-label="${s.unripe} unripe, ${s.ripe} ripe, ${s.overripe} overripe, ${s.unsure} unsure">
      <i style="width:${s.unripe / s.n * 100}%;background:#3B2552"></i><i style="width:${s.ripe / s.n * 100}%;background:#E2661B"></i><i style="width:${s.overripe / s.n * 100}%;background:#A3261D"></i><i style="width:${s.unsure / s.n * 100}%;background:#9FB0A4"></i></div>
    <p>${s.n} bunch${s.n > 1 ? 'es' : ''}. ${planFor(s)}.</p></div>`).join('');
  $('#block-table tbody').innerHTML = rows.map((s, i) => `<tr><td>${i + 1}</td><td>${esc(s.block)}</td><td>${s.n}</td><td>${s.unripe}</td><td>${s.ripe}</td><td>${s.overripe}</td><td>${s.unsure}</td><td>${planFor(s)}</td></tr>`).join('');
}

function exportCsv() {
  const rows = blockStats();
  const csv = ['priority,block,bunches,unripe,ripe,overripe,unsure,plan'].concat(rows.map((s, i) => [i + 1, s.block, s.n, s.unripe, s.ripe, s.overripe, s.unsure, planFor(s)].join(','))).join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' })); a.download = 'ripecheck_blocks.csv';
  document.body.appendChild(a); a.click(); a.remove();
}

/* ------------------------------------------------------------------ test a set */
function batchLabel() {
  const n = S.batchFiles.length;
  $('#batch-count').textContent = n ? `${n} photo${n > 1 ? 's' : ''} chosen` : 'No photos chosen';
  $('#btn-batch').disabled = !n;
}

async function useSamplesForBatch() {
  const names = await (await fetch('/api/samples')).json();
  S.batchFiles = await Promise.all(names.map(async (n) => new File([await (await fetch('/samples/' + encodeURIComponent(n))).blob()], n, { type: 'image/jpeg' })));
  batchLabel();
}

async function runBatch() {
  const err = $('#batch-error'); err.hidden = true;
  const btn = $('#btn-batch'); btn.disabled = true; btn.textContent = 'Running';
  const fd = new FormData();
  S.batchFiles.forEach((f) => fd.append('images', f, f.name));
  fd.append('opts', JSON.stringify(S.opts)); fd.append('thr', JSON.stringify(S.thr));
  try {
    const j = await (await fetch('/api/batch', { method: 'POST', body: fd })).json();
    if (!j.ok) throw new Error(j.error);
    renderBatch(j);
  } catch (e) {
    err.hidden = false; err.textContent = e.message || 'The test could not run.';
  } finally { btn.textContent = 'Run the test'; btn.disabled = !S.batchFiles.length; }
}

function renderBatch(j) {
  $('#batch-out').hidden = false;
  const f = (v, p = true) => (v == null ? 'n/a' : p ? (v * 100).toFixed(1) + '%' : v.toFixed(3));
  const s = j.summary;
  $('#kpis').innerHTML = s ? `<div><b>${s.n}</b><span>photos scored</span></div><div><b>${f(s.macro_f1, false)}</b><span>macro F1</span></div>
    <div><b>${f(s.accuracy_all)}</b><span>accuracy, all photos</span></div><div><b>${f(s.accuracy_graded)}</b><span>accuracy, graded only</span></div>
    <div><b>${f(s.coverage)}</b><span>received a grade</span></div>` : '<p class="empty-note">No file names start with unripe_, ripe_ or overripe_, so nothing was scored.</p>';
  const labels = ['unripe', 'ripe', 'overripe'], outs = [...labels, 'uncertain', 'not gradable'];
  $('#per-class').innerHTML = s ? `<thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Photos</th></tr></thead><tbody>${labels.map((c) => { const m = s.per_class[c];
    return `<tr><td>${c}</td><td>${f(m.precision, false)}</td><td>${f(m.recall, false)}</td><td>${f(m.f1, false)}</td><td>${m.support}</td></tr>`; }).join('')}</tbody>` : '';
  $('#confusion').innerHTML = s ? `<thead><tr><th>True class, then result</th>${outs.map((o) => `<th>${o}</th>`).join('')}</tr></thead><tbody>${labels.map((t) =>
    `<tr><td>${t}</td>${outs.map((o) => `<td>${s.confusion[t][o]}</td>`).join('')}</tr>`).join('')}</tbody>` : '';
  $('#batch-rows').innerHTML = `<thead><tr><th>Photo</th><th>True</th><th>Result</th><th>Margin</th><th>Checks failed</th></tr></thead><tbody>${j.rows.map((r) =>
    `<tr><td>${esc(r.file)}</td><td>${r.label || 'not scored'}</td><td>${esc(r.result)}</td><td>${r.margin == null ? '' : r.margin}</td><td>${esc(r.note || '')}</td></tr>`).join('')}</tbody>`;
}

init();
})();
