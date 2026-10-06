/* WT 控制台 — core: helpers, formatting, theme, tabs, status pill. Modules training.js / replay.js hang off window.RLD. */
(function () {
  'use strict';
  const RLD = window.RLD = { modules: {} };

  // ---------------------------------------------------------------- dom helpers
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  function el(tag, props, ...kids) {
    const n = document.createElement(tag);
    if (props) {
      for (const k in props) {
        const v = props[k];
        if (v == null || v === false) continue;
        if (k === 'class') n.className = v;
        else if (k === 'text') n.textContent = v;
        else if (k === 'html') n.innerHTML = v;            // only ever called with static or escaped strings
        else if (k === 'style') { if (typeof v === 'string') n.style.cssText = v; else Object.assign(n.style, v); }
        else if (k === 'on') { for (const e in v) n.addEventListener(e, v[e]); }
        else if (k === 'dataset') Object.assign(n.dataset, v);
        else if (k in n && k !== 'list') { try { n[k] = v; } catch (e) { n.setAttribute(k, v); } }
        else n.setAttribute(k, v === true ? '' : v);
      }
    }
    const add = (c) => {
      if (c == null || c === false) return;
      if (Array.isArray(c)) c.forEach(add);
      else n.appendChild(c.nodeType ? c : document.createTextNode(String(c)));
    };
    kids.forEach(add);
    return n;
  }
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  function svgEl(tag, attrs) {
    const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const k in attrs || {}) n.setAttribute(k, attrs[k]);
    return n;
  }

  // ---------------------------------------------------------------- storage / api / events
  const store = {
    get(k, d) { try { const v = localStorage.getItem('rld.' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('rld.' + k, JSON.stringify(v)); } catch (e) { /* private mode */ } },
  };
  async function api(path, opts) {
    const r = await fetch(path, opts);
    let j = null;
    try { j = await r.json(); } catch (e) { /* not json */ }
    if (!r.ok) throw new Error((j && j.error) || r.statusText || ('HTTP ' + r.status));
    return j;
  }
  const handlers = {};
  RLD.on = (evt, fn) => { (handlers[evt] = handlers[evt] || []).push(fn); };
  RLD.emit = (evt, ...a) => (handlers[evt] || []).forEach((f) => { try { f(...a); } catch (e) { console.error(e); } });

  // ---------------------------------------------------------------- formatting
  const nf = new Intl.NumberFormat('en-US');
  const fmt = {
    int: (n) => (n == null || !isFinite(n) ? '—' : nf.format(Math.round(n))),
    compact(n, d) {
      if (n == null || !isFinite(n)) return '—';
      const a = Math.abs(n);
      if (a >= 1e9) return (n / 1e9).toFixed(d == null ? 2 : d) + 'G';
      if (a >= 1e6) return (n / 1e6).toFixed(d == null ? 2 : d) + 'M';
      if (a >= 1e4) return (n / 1e3).toFixed(d == null ? 1 : d) + 'k';
      return nf.format(Math.round(n));
    },
    /** adaptive precision for metrics */
    num(v, d) {
      if (v == null || !isFinite(v)) return '—';
      const a = Math.abs(v);
      if (d != null) return v.toFixed(d);
      if (a === 0) return '0';
      if (a >= 1e5) return v.toExponential(2);
      if (a >= 1000) return nf.format(Math.round(v));
      if (a >= 100) return v.toFixed(1);
      if (a >= 1) return v.toFixed(3);
      if (a >= 0.01) return v.toFixed(4);
      if (a >= 0.0001) return v.toFixed(5);
      return v.toExponential(2);
    },
    pct: (v, d) => (v == null || !isFinite(v) ? '—' : (v * 100).toFixed(d == null ? 1 : d) + '%'),
    dur(s) {
      if (s == null || !isFinite(s) || s < 0) return '—';
      if (s < 60) return Math.round(s) + ' 秒';
      if (s < 3600) return Math.floor(s / 60) + ' 分 ' + Math.round(s % 60) + ' 秒';
      if (s < 86400) return Math.floor(s / 3600) + ' 小时 ' + Math.floor((s % 3600) / 60) + ' 分';
      return Math.floor(s / 86400) + ' 天 ' + Math.floor((s % 86400) / 3600) + ' 小时';
    },
    ago(s) {
      if (s == null || !isFinite(s)) return '—';
      s = Math.max(0, s);
      if (s < 5) return '刚刚';
      if (s < 60) return Math.round(s) + ' 秒前';
      if (s < 3600) return Math.floor(s / 60) + ' 分钟前';
      if (s < 86400) return Math.floor(s / 3600) + ' 小时前';
      return Math.floor(s / 86400) + ' 天前';
    },
    clock(ts) {
      if (ts == null) return '—';
      const d = new Date(ts * 1000);
      const p = (n) => String(n).padStart(2, '0');
      return p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds());
    },
    mmss(t, tenths) {
      t = Math.max(0, t || 0);
      const m = Math.floor(t / 60);
      const s = t - m * 60;
      const p = (n) => String(n).padStart(2, '0');
      return tenths ? p(m) + ':' + (s < 10 ? '0' : '') + s.toFixed(1) : p(m) + ':' + p(Math.floor(s));
    },
    bytes(n) {
      if (n == null) return '—';
      if (n < 1024) return n + ' B';
      if (n < 1048576) return (n / 1024).toFixed(1) + ' KB';
      if (n < 1073741824) return (n / 1048576).toFixed(1) + ' MB';
      return (n / 1073741824).toFixed(2) + ' GB';
    },
    /** "5M", "2.5e6", "500k", "1,000,000" -> number (NaN when not a number) */
    parseCount(s) {
      s = String(s || '').trim().replace(/[,\s_]/g, '').toLowerCase();
      const m = s.match(/^([\d.]+(?:e[+-]?\d+)?)([kmg万亿]?)$/);
      if (!m) return NaN;
      const mult = { '': 1, k: 1e3, m: 1e6, g: 1e9, '万': 1e4, '亿': 1e8 }[m[2]];
      return parseFloat(m[1]) * mult;
    },
  };

  // ---------------------------------------------------------------- colour helpers
  const colorCache = new Map();
  function parseColor(c) {
    if (colorCache.has(c)) return colorCache.get(c);
    let rgb = [128, 128, 128];
    let m = /^#([0-9a-f]{6})$/i.exec(c.trim());
    if (m) rgb = [parseInt(m[1].slice(0, 2), 16), parseInt(m[1].slice(2, 4), 16), parseInt(m[1].slice(4, 6), 16)];
    else if ((m = /^#([0-9a-f]{3})$/i.exec(c.trim()))) rgb = m[1].split('').map((x) => parseInt(x + x, 16));
    else if ((m = /rgba?\(([^)]+)\)/.exec(c))) rgb = m[1].split(',').slice(0, 3).map((x) => parseFloat(x));
    colorCache.set(c, rgb);
    return rgb;
  }
  const rgba = (c, a) => { const r = parseColor(c); return 'rgba(' + r[0] + ',' + r[1] + ',' + r[2] + ',' + a + ')'; };

  // ---------------------------------------------------------------- theme
  const THEME_VARS = ['bg', 'surface', 'surface-2', 'surface-3', 'border', 'border-2', 'text', 'text-2', 'muted', 'faint', 'accent', 'team0', 'team1',
    'kill', 'ok', 'warn', 'missile', 'grid', 'grid-strong', 'map-bg-1', 'map-bg-2', 'map-out', 'pos', 'neg',
    'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'];
  RLD.C = {};
  const Theme = {
    pref: store.get('theme', 'auto'),
    effective() {
      if (this.pref === 'light' || this.pref === 'dark') return this.pref;
      return matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    },
    apply() {
      const root = document.documentElement;
      if (this.pref === 'auto') root.removeAttribute('data-theme'); else root.setAttribute('data-theme', this.pref);
      this.refresh();
    },
    refresh() {
      const cs = getComputedStyle(document.documentElement);
      THEME_VARS.forEach((v) => { RLD.C[v.replace(/-(\w)/g, (_, c) => c.toUpperCase())] = cs.getPropertyValue('--' + v).trim(); });
      RLD.C.palette = [1, 2, 3, 4, 5, 6, 7, 8].map((i) => RLD.C['p' + i]);
      RLD.C.mono = cs.getPropertyValue('--mono').trim();
      RLD.C.font = cs.getPropertyValue('--font').trim();
      RLD.emit('theme');
    },
    toggle() {
      this.pref = this.effective() === 'dark' ? 'light' : 'dark';
      store.set('theme', this.pref);
      this.apply();
    },
  };
  RLD.theme = Theme;

  // ---------------------------------------------------------------- toast + live pill
  let toastTimer = null;
  RLD.toast = (msg, ms) => {
    const t = $('#toast');
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, ms || 2600);
  };
  RLD.setLive = (state, text, title) => {
    const p = $('#livePill');
    p.className = 'pill ' + (state || '');
    $('#liveText').textContent = text;
    p.title = title || '';
  };

  // ---------------------------------------------------------------- tabs
  const tabs = { current: null };
  RLD.showTab = (name, fromHash) => {
    if (!['train', 'replay'].includes(name)) name = 'train';
    if (tabs.current === name) return;
    tabs.current = name;
    $$('.tab').forEach((b) => { const on = b.dataset.tab === name; b.classList.toggle('active', on); b.setAttribute('aria-selected', on); });
    $$('.panel').forEach((p) => { p.hidden = p.dataset.panel !== name; });
    $$('[data-for]').forEach((n) => { n.hidden = n.dataset.for !== name; });
    store.set('tab', name);
    if (!fromHash && location.hash !== '#' + name) history.replaceState(null, '', '#' + name);
    RLD.emit('tab', name);
  };
  RLD.tab = () => tabs.current;

  // ---------------------------------------------------------------- start
  RLD.start = () => {
    window.RLD.$ = $; window.RLD.$$ = $$; window.RLD.el = el; window.RLD.esc = esc; window.RLD.svgEl = svgEl;
    Theme.apply();
    matchMedia('(prefers-color-scheme: light)').addEventListener('change', () => { if (Theme.pref === 'auto') Theme.refresh(); });
    $('#themeBtn').addEventListener('click', () => Theme.toggle());
    $$('.tab').forEach((b) => b.addEventListener('click', () => RLD.showTab(b.dataset.tab)));
    window.addEventListener('hashchange', () => RLD.showTab(location.hash.replace('#', ''), true));
    document.addEventListener('visibilitychange', () => RLD.emit('visibility', !document.hidden));
    Object.keys(RLD.modules).forEach((k) => { try { RLD.modules[k].init(); } catch (e) { console.error('init ' + k, e); } });
    const start = (location.hash || '').replace('#', '') || store.get('tab', 'train');
    RLD.showTab(start, true);
  };

  Object.assign(RLD, { $, $$, el, esc, svgEl, store, api, fmt, rgba, parseColor });
})();
