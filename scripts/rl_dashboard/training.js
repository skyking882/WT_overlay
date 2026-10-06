/* WT 控制台 — 训练 tab: KPI cards, Chart.js charts, per-aircraft table, BC report, log tail, run picker, auto refresh. */
(function () {
  'use strict';
  const R = window.RLD;
  const { $, $$, el, esc, fmt, store, api, rgba, svgEl } = R;
  const REFRESH_S = 10;
  const DASHES = [[], [7, 4], [2, 3], [9, 3, 2, 3]];

  const T = {
    runs: [], sel: [], primary: null, data: {}, skew: 0, loaded: false, paused: false, countdown: REFRESH_S, busy: false,
    targets: {}, opt: { xaxis: 'decisions', smooth: 'auto', window: 20, head: '', sel: {}, sort: { key: 'decisions', dir: -1 } },
    charts: {}, kpi: {}, bcChart: null, bcSig: '', logSig: '', metaSig: '', lastError: null,
  };

  // ------------------------------------------------------------------ colours
  const col = (k) => (R.C[k] || k);
  const runColor = (id) => {
    const i = Math.max(0, T.runs.findIndex((r) => r.id === id));
    return R.C.palette[i % R.C.palette.length];
  };
  const runLabel = (id) => (T.runs.find((r) => r.id === id) || {}).label || id;
  const multi = () => T.sel.length > 1;

  // ------------------------------------------------------------------ chart definitions
  const per100 = (k) => (r) => { const e = r.events && r.events[k]; const d = r.decisions_in_round; return e == null || !d ? null : (e / d) * 100; };
  const EVENTS = [['launch', '发射', 'p1'], ['kill', '击杀', 'kill'], ['death', '阵亡', 'p4'], ['assist', '助攻', 'p6']];
  const tm = (f) => (r) => { const t = r.time; if (!t) return null; const v = f(t); return v == null || !isFinite(v) ? null : Math.max(0, v); };
  const outShare = (k) => (r) => { const o = r.outcomes; if (!o) return null; const n = (o.win || 0) + (o.loss || 0) + (o.trade || 0) + (o.none || 0); return n ? (o[k] || 0) / n : null; };
  const CHARTS = [
    { id: 'winrate', title: '胜率', sub: '击落对方且存活 · S1 过关线 60%', ref: [{ y: 0.6 }], min0: true, max1: true, fmtVal: (v) => fmt.pct(v, 1),
      series: [{ key: 'v', label: '胜率', color: 'p3', get: (r) => r.win_rate }] },
    { id: 'exchange', title: '交换比', sub: '击落 / 被击落 · S1 过关线 1.5', ref: [{ y: 1.5 }, { y: 1 }], min0: true,
      series: [{ key: 'v', label: '交换比', color: 'p1', get: (r) => r.exchange }] },
    { id: 'outcomes', title: '每局结果', sub: '胜 · 负 · 同归于尽 · 超时', min0: true, max1: true, fmtVal: (v) => fmt.pct(v, 1), multiDefault: ['win', 'loss'],
      series: [{ key: 'win', label: '胜', color: 'p3', get: outShare('win') }, { key: 'loss', label: '负', color: 'kill', get: outShare('loss') },
               { key: 'trade', label: '同归于尽', color: 'p6', get: outShare('trade') }, { key: 'none', label: '超时', color: 'p8', get: outShare('none') }] },
    { id: 'reward', title: '每决策奖励', sub: 'reward / decision', ref: [{ y: 0 }],
      series: [{ key: 'v', label: '每决策奖励', color: 'p3', get: (r) => r.reward_per_decision }] },
    { id: 'return', title: '回合回报', sub: 'episode return · mean', ref: [{ y: 0 }],
      series: [{ key: 'v', label: '回合平均回报', color: 'p1', get: (r) => r.episode_return_mean }] },
    { id: 'events', title: '事件 / 100 决策', sub: 'launch · kill · death · assist', multiDefault: ['kill'],
      series: EVENTS.map(([k, l, c]) => ({ key: k, label: l, color: c, get: per100(k) })) },
    { id: 'entropy', title: '策略熵', sub: 'entropy', headKey: 'entropy_head',
      series: [{ key: 'all', label: '整体', color: 'p4', get: (r) => r.entropy }] },
    { id: 'kl', title: 'KL 散度', sub: '行为策略 · BC 参考', headKey: 'kl_ref_head', refCfg: 'target_kl', min0: true,
      series: [{ key: 'target', label: '目标 KL', color: 'p1', get: (r) => r.kl_target },
               { key: 'ref', label: 'BC 参考 KL', color: 'p6', get: (r) => r.kl_ref }] },
    { id: 'clip', title: '裁剪比例', sub: 'clip fraction', min0: true,
      series: [{ key: 'v', label: '裁剪比例', color: 'p2', get: (r) => r.clip_frac }] },
    { id: 'vloss', title: '价值损失', sub: 'value loss', min0: true,
      series: [{ key: 'v', label: '价值损失', color: 'kill', get: (r) => r.value_loss }] },
    { id: 'ev', title: '解释方差', sub: 'explained variance', ref: [{ y: 0 }, { y: 1 }], max1: true,
      series: [{ key: 'v', label: '解释方差', color: 'p3', get: (r) => r.explained_variance }] },
    { id: 'valid', title: '有效决策比例', sub: 'valid fraction', min0: true, fmtVal: (v) => fmt.pct(v, 1),
      series: [{ key: 'v', label: '有效比例', color: 'p8', get: (r) => r.valid_fraction }] },
    { id: 'sched', title: '调度系数', sub: '熵系数 · KL β', min0: true,
      series: [{ key: 'ent', label: '熵系数', color: 'p4', get: (r) => r.ent_coef }, { key: 'beta', label: 'KL β', color: 'p6', get: (r) => r.kl_beta }] },
    { id: 'grad', title: '梯度范数', sub: 'actor · critic', min0: true,
      series: [{ key: 'a', label: 'actor', color: 'p1', get: (r) => r.grad_norm_actor }, { key: 'c', label: 'critic', color: 'p2', get: (r) => r.grad_norm_critic }] },
    { id: 'time', title: '每轮耗时', sub: '采样 = sample − 推理 · 仅主要运行', kind: 'stack', primaryOnly: true,
      fmtVal: (v) => fmt.num(v, 2) + ' s',
      series: [
        { key: 'env', label: '采样 (环境)', color: 'p1', get: tm((t) => (t.sample == null ? null : t.sample - (t.inference || 0))) },
        { key: 'inf', label: '推理', color: 'p4', get: tm((t) => t.inference) },
        { key: 'post', label: '后处理', color: 'p6', get: tm((t) => t.postpass) },
        { key: 'upd', label: '更新', color: 'p3', get: tm((t) => t.update) },
      ] },
  ];

  // ------------------------------------------------------------------ small utils
  const isNum = (v) => typeof v === 'number' && isFinite(v);
  const xOf = (r) => (T.opt.xaxis === 'round' ? r.round : r.decisions_total);
  const tickFmt = (v) => (Math.abs(v) >= 1000 ? fmt.compact(v, 1) : String(+v.toPrecision(3)));
  const axisX = (v) => (T.opt.xaxis === 'round' ? String(Math.round(v)) : (Math.abs(v) >= 1e6 ? +(v / 1e6).toFixed(2) + 'M' : Math.abs(v) >= 1e3 ? +(v / 1e3).toFixed(1) + 'k' : String(Math.round(v))));
  const smoothWin = (n) => {
    const s = T.opt.smooth;
    if (s === 'auto') return Math.max(1, Math.min(15, Math.round(n / 20)));
    return Math.max(1, parseInt(s, 10) || 1);
  };
  function movavg(pts, w) {
    const out = [];
    let sum = 0;
    const win = [];
    for (const p of pts) {
      win.push(p.y); sum += p.y;
      if (win.length > w) sum -= win.shift();
      out.push({ x: p.x, y: sum / win.length, raw: p.y, r: p.r });
    }
    return out;
  }
  const trimStr = (s, n) => (s.length > n ? s.slice(0, n - 1) + '…' : s);
  function saveOpt() { store.set('topt', T.opt); }

  // ------------------------------------------------------------------ chart plugins
  function nearestPt(pts, x) {
    let lo = 0, hi = pts.length - 1;
    if (hi < 0) return null;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (pts[m].x < x) lo = m; else hi = m; }
    return Math.abs(pts[lo].x - x) <= Math.abs(pts[hi].x - x) ? pts[lo] : pts[hi];
  }
  function collectHits(chart) {
    const hv = chart.$hv;
    if (!hv) return null;
    const xs = chart.scales.x, ys = chart.scales.y;
    if (chart.config.type === 'bar') {
      const n = chart.data.labels.length;
      if (!n) return null;
      const idx = Math.min(n - 1, Math.max(0, Math.round(xs.getValueForPixel(hv.x))));
      const items = chart.data.datasets.map((ds) => ({ ds, v: ds.data[idx], idx })).filter((i) => i.v != null);
      return { bar: true, idx, items, px: xs.getPixelForValue(idx) };
    }
    const xv = xs.getValueForPixel(hv.x);
    const items = [];
    for (const ds of chart.data.datasets) {
      if (ds.$raw || !ds.$pts || !ds.$pts.length) continue;
      const p = nearestPt(ds.$pts, xv);
      if (!p) continue;
      const px = xs.getPixelForValue(p.x);
      if (Math.abs(px - hv.x) > 64) continue;
      items.push({ ds, p, px, py: ys.getPixelForValue(p.y) });
    }
    return { bar: false, xv, items, px: items.length ? items[0].px : hv.x };
  }
  const crossPlugin = {
    id: 'rldCross',
    afterEvent(chart, args) {
      const e = args.event, a = chart.chartArea;
      if (!a || !e) return;
      const inside = e.type !== 'mouseout' && e.x != null && e.x >= a.left && e.x <= a.right && e.y >= a.top && e.y <= a.bottom;
      if (inside) { chart.$hv = { x: e.x, y: e.y }; args.changed = true; }
      else if (chart.$hv) { chart.$hv = null; args.changed = true; }
    },
    afterDatasetsDraw(chart) {
      const a = chart.chartArea;
      const refs = chart.$refs || [];
      const ctx = chart.ctx;
      ctx.save();
      for (const rf of refs) {
        const y = chart.scales.y.getPixelForValue(rf.y);
        if (y < a.top - 1 || y > a.bottom + 1) continue;
        ctx.strokeStyle = rf.color || R.C.gridStrong;
        ctx.lineWidth = 1;
        ctx.setLineDash(rf.dash || [4, 4]);
        ctx.beginPath(); ctx.moveTo(a.left, y + .5); ctx.lineTo(a.right, y + .5); ctx.stroke();
        if (rf.label) {
          ctx.setLineDash([]);
          ctx.fillStyle = R.C.faint;
          ctx.font = '10px ' + R.C.mono;
          ctx.textAlign = 'right';
          ctx.fillText(rf.label, a.right - 2, y - 4);
        }
      }
      ctx.restore();
      const hit = collectHits(chart);
      chart.$hit = hit;
      if (!hit) return;
      ctx.save();
      if (hit.bar) {
        const w = (chart.scales.x.getPixelForValue(1) - chart.scales.x.getPixelForValue(0)) || 6;
        ctx.fillStyle = rgba(R.C.text, 0.07);
        ctx.fillRect(hit.px - w / 2, a.top, w, a.bottom - a.top);
      } else if (hit.items.length) {
        ctx.strokeStyle = R.C.gridStrong; ctx.lineWidth = 1; ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(Math.round(hit.px) + .5, a.top); ctx.lineTo(Math.round(hit.px) + .5, a.bottom); ctx.stroke();
        for (const it of hit.items) {
          ctx.beginPath(); ctx.arc(it.px, it.py, 3.6, 0, 6.2832);
          ctx.fillStyle = it.ds.borderColor; ctx.fill();
          ctx.lineWidth = 1.6; ctx.strokeStyle = R.C.surface; ctx.stroke();
        }
      }
      ctx.restore();
    },
    afterDraw(chart) { paintTip(chart); },
  };

  function paintTip(chart) {
    const tip = chart.canvas.parentNode.querySelector('.chart-tip');
    if (!tip) return;
    const hit = chart.$hit;
    const def = chart.$def;
    if (!hit || !hit.items.length) { tip.hidden = true; return; }
    const f = def.fmtVal || fmt.num;
    let head, rows = [];
    if (hit.bar) {
      const rounds = chart.$rounds || [];
      const r = rounds[hit.idx];
      const total = hit.items.reduce((s, i) => s + (i.v || 0), 0);
      head = '第 ' + chart.data.labels[hit.idx] + ' 轮' + (r && r.decisions_total != null ? ' · ' + fmt.compact(r.decisions_total) + ' 决策' : '');
      rows = hit.items.map((i) => row(i.ds.backgroundColor, i.ds.label, f(i.v)));
      rows.push(row('transparent', '合计', f(total), null, true));
    } else {
      head = T.opt.xaxis === 'round' ? '第 ' + Math.round(hit.xv) + ' 轮' : '决策数 ' + fmt.int(hit.xv);
      rows = hit.items.map((i) => {
        const showRaw = i.p.raw != null && Math.abs(i.p.raw - i.p.y) > 1e-12;
        return row(i.ds.borderColor, i.ds.label, f(i.p.y), showRaw ? '原始 ' + f(i.p.raw) : null, false, multi() ? 'R' + i.p.r : null);
      });
    }
    tip.innerHTML = '<div class="tip-h">' + esc(head) + '</div>' + rows.join('');
    tip.hidden = false;
    const hv = chart.$hv, cv = chart.canvas;
    const boxW = cv.parentNode.clientWidth;
    let left = cv.offsetLeft + hv.x + 16;
    if (left + tip.offsetWidth > boxW - 4) left = cv.offsetLeft + hv.x - tip.offsetWidth - 16;
    tip.style.left = Math.max(4, left) + 'px';
    tip.style.top = Math.max(4, cv.offsetTop + hv.y - tip.offsetHeight / 2) + 'px';
    function row(color, name, val, raw, bold, rd) {
      return '<div class="tip-r"><i class="sw" style="background:' + esc(color) + '"></i><span class="k">' + esc(trimStr(name, 34)) + '</span>' +
        (raw ? '<span class="raw">' + esc(raw) + '</span>' : '') + (rd ? '<span class="rd">' + esc(rd) + '</span>' : '') +
        '<span class="v"' + (bold ? ' style="font-weight:600"' : '') + '>' + esc(val) + '</span></div>';
    }
  }

  // ------------------------------------------------------------------ chart cards
  function activeSet(def) {
    const keys = def.series.map((s) => s.key);
    const sel = T.opt.sel[def.id];
    if (Array.isArray(sel)) { const s = new Set(sel.filter((k) => keys.includes(k))); if (s.size) return s; }
    if (multi()) return new Set(def.multiDefault || [keys[0]]);
    return new Set(keys);
  }
  /** line style of a series when several runs are overlaid: the 1st shown series is solid, the next dashed, dotted ... */
  function dashIndex(def, key) {
    const act = activeSet(def);
    const shown = seriesOf(def).filter((x) => x.isHead || act.has(x.key));
    const i = shown.findIndex((x) => x.key === key);
    return shown.length > 1 && i > 0 ? i % 4 : 0;
  }
  function seriesOf(def) {
    const list = def.series.slice();
    if (def.headKey && T.opt.head) {
      const hk = def.headKey, h = T.opt.head;
      list.push({ key: 'head', label: '头 · ' + h, color: 'p5', get: (r) => (r[hk] ? r[hk][h] : null), isHead: true });
    }
    return list;
  }

  function buildCards() {
    const host = $('#charts');
    host.innerHTML = '';
    for (const def of CHARTS) {
      const canvas = el('canvas');
      const tip = el('div', { class: 'chart-tip', hidden: true });
      const skel = el('div', { class: 'skel', style: 'position:absolute;inset:10px 14px 16px 10px' });
      const box = el('div', { class: 'chart-box' + (def.kind === 'stack' ? '' : '') }, canvas, tip, skel);
      const legend = el('div', { class: 'legend' });
      const card = el('div', { class: 'card chart-card', style: def.wide ? 'grid-column: 1 / -1;' : '' },
        el('div', { class: 'card-h' }, el('h3', { text: def.title }), el('span', { class: 'sub', text: def.sub })),
        legend, box);
      host.appendChild(card);
      def.canvas = canvas; def.legend = legend; def.skel = skel; def.box = box; def.card = card;
    }
  }

  function updateLegend(def) {
    const lg = def.legend;
    lg.innerHTML = '';
    const act = activeSet(def);
    const one = !multi();
    if (def.series.length > 1) {
      def.series.forEach((s, i) => {
        const on = act.has(s.key);
        const di = !one && on ? dashIndex(def, s.key) : 0;
        const sample = el('span', { class: 'lg-sample' + (di === 1 || di === 3 ? ' dash' : di === 2 ? ' dot' : ''), style: { color: one ? col(s.color) : R.C.muted } });
        const b = el('button', { class: 'chip-btn' + (on ? ' on' : ''), title: s.label, on: { click: () => {
          const cur = new Set(activeSet(def));
          if (cur.has(s.key)) { if (cur.size > 1) cur.delete(s.key); } else cur.add(s.key);
          T.opt.sel[def.id] = def.series.map((x) => x.key).filter((k) => cur.has(k));
          saveOpt(); updateChart(def); updateLegend(def);
        } } }, sample, s.label);
        if (on && one) b.style.color = col(s.color);
        lg.appendChild(b);
      });
    }
    if (def.headKey) {
      const sel = el('select', { title: '叠加某个动作头的曲线', 'aria-label': '动作头', on: { change: (e) => {
        T.opt.head = e.target.value; saveOpt(); CHARTS.filter((d) => d.headKey).forEach((d) => { updateChart(d); updateLegend(d); });
      } } }, el('option', { value: '', text: '动作头: 无' }));
      for (const h of T.heads || []) sel.appendChild(el('option', { value: h, text: h }));
      sel.value = T.opt.head || '';
      lg.appendChild(sel);
    }
    lg.style.display = lg.children.length ? '' : 'none';
    lg.style.paddingBottom = lg.children.length ? '0' : '0';
  }

  function makeChart(def) {
    if (!window.Chart) return null;
    const C = R.C;
    const stack = def.kind === 'stack';
    const mono = { family: C.mono.split(',')[0].replace(/["']/g, '').trim() + ', monospace', size: 10.5 };
    const opts = {
      responsive: true, maintainAspectRatio: false, parsing: !stack ? false : undefined, normalized: !stack, spanGaps: true,
      animation: T.animateNext ? { duration: 700, easing: 'easeOutCubic' } : false,
      layout: { padding: { top: 6, right: 6 } },
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      elements: { point: { radius: 0, hoverRadius: 0 }, line: { borderWidth: 2, tension: 0.2, borderJoinStyle: 'round' }, bar: { borderRadius: 0 } },
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: stack
          ? { type: 'category', stacked: true, grid: { display: false }, border: { color: C.border }, ticks: { color: C.muted, font: mono, maxTicksLimit: 9, autoSkip: true, maxRotation: 0 } }
          : { type: 'linear', grid: { display: false }, border: { color: C.border }, ticks: { color: C.muted, font: mono, maxTicksLimit: 6, maxRotation: 0, callback: (v) => axisX(v) } },
        y: { stacked: stack, grid: { color: C.grid, drawTicks: false }, border: { display: false }, beginAtZero: !!(def.min0 || stack),
             grace: stack ? 0 : '8%', ticks: { color: C.muted, font: mono, maxTicksLimit: 5, padding: 8, callback: (v) => tickFmt(v) } },
      },
    };
    if (def.max1) opts.scales.y.suggestedMax = 1;
    const chart = new Chart(def.canvas.getContext('2d'), { type: stack ? 'bar' : 'line', data: { labels: [], datasets: [] }, options: opts, plugins: [crossPlugin] });
    chart.$def = def;
    return chart;
  }

  function datasetsFor(def) {
    const sets = [];
    const runs = def.primaryOnly ? [T.primary] : T.sel;
    const act = activeSet(def);
    const series = seriesOf(def);
    const m = multi() && !def.primaryOnly;
    for (const rid of runs) {
      const d = T.data[rid];
      if (!d || !d.m) continue;
      const rounds = d.m.rounds;
      series.forEach((s, si) => {
        if (!s.isHead && !act.has(s.key)) return;
        const color = m ? runColor(rid) : col(s.color);
        const dash = m ? DASHES[dashIndex(def, s.key)] : [];
        const pts = [];
        for (const r of rounds) { const y = s.get(r); if (isNum(y)) pts.push({ x: xOf(r), y, r: r.round }); }
        const w = smoothWin(pts.length);
        const sm = w > 1 ? movavg(pts, w) : pts.map((p) => ({ x: p.x, y: p.y, raw: p.y, r: p.r }));
        const label = (multi() && !def.primaryOnly ? runLabel(rid) + ' · ' : '') + s.label;
        if (w > 1) sets.push({ label: label + ' (原始)', data: pts, borderColor: rgba(color, 0.26), borderWidth: 1, borderDash: dash, $raw: true, pointRadius: 0 });
        sets.push({ label, data: sm, borderColor: color, backgroundColor: color, borderWidth: 2, borderDash: dash, $pts: sm, pointRadius: 0 });
      });
    }
    return sets;
  }

  function stackData(def) {
    const d = T.data[T.primary];
    const rounds = d && d.m ? d.m.rounds : [];
    const labels = rounds.map((r) => String(r.round));
    const datasets = def.series.map((s) => ({
      label: s.label, data: rounds.map((r) => { const v = s.get(r); return isNum(v) ? v : null; }),
      backgroundColor: col(s.color), borderWidth: 0, barPercentage: 0.94, categoryPercentage: 1, maxBarThickness: 18,
    }));
    return { labels, datasets, rounds };
  }

  function refLines(def) {
    const out = (def.ref || []).map((r) => ({ y: r.y, color: R.C.gridStrong, dash: [3, 4] }));
    if (def.refCfg) {
      const cfg = (T.data[T.primary] || {}).m && T.data[T.primary].m.config;
      if (cfg && isNum(cfg[def.refCfg])) out.push({ y: cfg[def.refCfg], label: '目标 ' + cfg[def.refCfg], color: rgba(R.C.warn, 0.55), dash: [5, 4] });
    }
    return out;
  }

  function updateChart(def) {
    if (!def.chart) {
      def.chart = makeChart(def);
      if (!def.chart) { def.skel.textContent = '无法加载 Chart.js (cdnjs.cloudflare.com) — 图表不可用'; def.skel.classList.remove('skel'); def.skel.style.cssText = 'position:absolute;inset:0;display:grid;place-items:center;color:var(--muted);font-size:12.5px;text-align:center;padding:20px'; return; }
    }
    def.skel.style.display = 'none';
    const ch = def.chart;
    if (def.kind === 'stack') {
      const s = stackData(def);
      ch.data.labels = s.labels; ch.data.datasets = s.datasets; ch.$rounds = s.rounds;
    } else {
      ch.data.datasets = datasetsFor(def);
    }
    ch.$refs = refLines(def);
    ch.update('none');
  }

  function updateAllCharts() { CHARTS.forEach(updateChart); }
  function rebuildCharts(animate) {
    T.animateNext = !!animate;
    for (const def of CHARTS) { if (def.chart) { def.chart.destroy(); def.chart = null; } }
    updateAllCharts();
    CHARTS.forEach(updateLegend);
    T.animateNext = false;
  }

  // ------------------------------------------------------------------ KPI cards
  const KPI_DEFS = [
    { id: 'round', label: '训练轮次' },
    { id: 'dec', label: '总决策数' },
    { id: 'rate', label: '决策 / 小时' },
    { id: 'eta', label: '预计完成' },
    { id: 'rew', label: '每决策奖励' },
    { id: 'ev', label: '解释方差' },
  ];
  function buildKPIs() {
    const host = $('#kpis');
    host.innerHTML = '';
    for (const k of KPI_DEFS) {
      const v = el('div', { class: 'kpi-v' }, el('span', { class: 'v', text: '—' }), el('small'));
      const d = el('div', { class: 'kpi-d' });
      const spark = k.id === 'eta' ? null : svgEl('svg', { class: 'spark', viewBox: '0 0 100 30', preserveAspectRatio: 'none' });
      const card = el('div', { class: 'card kpi', id: 'kpi-' + k.id }, el('div', { class: 'kpi-l' }, el('span', { text: k.label })), v, d, spark);
      if (spark) {
        const gid = 'sg-' + k.id;
        spark.innerHTML = '<defs><linearGradient id="' + gid + '" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="currentColor" stop-opacity=".28"/><stop offset="1" stop-color="currentColor" stop-opacity="0"/></linearGradient></defs>' +
          '<path class="a" fill="url(#' + gid + ')" stroke="none"/><path class="l" stroke="currentColor"/>';
      }
      host.appendChild(card);
      T.kpi[k.id] = { card, v: v.firstChild, small: v.lastChild, d, spark };
    }
    // eta card extras
    const e = T.kpi.eta;
    e.card.querySelector('.kpi-v').classList.add('mid');
    e.bar = el('div', { class: 'progress' }, el('i'));
    e.row = el('div', { class: 'target-row' });
    e.input = el('input', { class: 'num wide', type: 'text', placeholder: '目标决策数', 'aria-label': '目标决策数', spellcheck: false, on: {
      change: () => setTarget(e.input.value), keydown: (ev) => { if (ev.key === 'Enter') { e.input.blur(); } } } });
    e.hint = el('button', { class: 'linklike', hidden: true });
    e.row.append(e.input, e.hint);
    e.card.append(e.row, e.bar);
  }
  function setSpark(sp, ys, color) {
    if (!sp) return;
    sp.style.color = color;
    const v = ys.filter(isNum);
    const pa = sp.querySelector('.a'), pl = sp.querySelector('.l');
    if (v.length < 2) { pa.setAttribute('d', ''); pl.setAttribute('d', ''); return; }
    let lo = Math.min(...v), hi = Math.max(...v);
    if (hi - lo < 1e-12) { hi += 1; lo -= 1; }
    const n = v.length;
    const pts = v.map((y, i) => [(i / (n - 1)) * 100, 27 - ((y - lo) / (hi - lo)) * 24]);
    const d = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(2) + ' ' + p[1].toFixed(2)).join(' ');
    pl.setAttribute('d', d);
    pa.setAttribute('d', d + ' L100 30 L0 30 Z');
  }
  function deltaChip(cur, prev, o) {
    o = o || {};
    if (!isNum(cur) || !isNum(prev)) return null;
    const diff = cur - prev;
    let txt;
    if (o.pct) txt = (prev ? (diff / Math.abs(prev)) * 100 : 0);
    else txt = diff;
    const dir = Math.abs(diff) < 1e-12 ? 0 : diff > 0 ? 1 : -1;
    const cls = o.neutral || dir === 0 ? '' : ((dir > 0) === (o.upGood !== false) ? ' up' : ' down');
    const arrow = dir > 0 ? '▲' : dir < 0 ? '▼' : '•';
    const s = o.pct ? (txt > 0 ? '+' : '') + txt.toFixed(1) + '%' : (diff > 0 ? '+' : '') + (o.int ? fmt.int(diff) : fmt.num(diff, o.dec));
    return el('span', { class: 'delta' + cls, title: '相对上一轮' }, arrow + ' ' + s);
  }
  function instantRates(rs) {
    const out = [];
    for (let i = 1; i < rs.length; i++) {
      const a = rs[i - 1], b = rs[i];
      const dt = (b.timestamp != null && a.timestamp != null) ? b.timestamp - a.timestamp : (b.time && b.time.round);
      const dd = b.decisions_total - a.decisions_total;
      out.push(dt > 0 && dt < 3600 ? (dd / dt) * 3600 : null);
    }
    return out;
  }
  function renderKPIs() {
    const d = T.data[T.primary];
    const rs = d && d.m ? d.m.rounds : [];
    const st = d && d.m ? d.m.status : null;
    const last = rs[rs.length - 1], prev = rs[rs.length - 2];
    const K = T.kpi;
    const color = (i) => R.C.palette[i];
    const setKpi = (id, value, small, deltaEl, sparkYs, c, subEl) => {
      const k = K[id];
      k.v.textContent = value; k.small.textContent = small || '';
      k.d.innerHTML = '';
      if (deltaEl) k.d.appendChild(deltaEl);
      if (subEl) k.d.appendChild(subEl);
      if (sparkYs) setSpark(k.spark, sparkYs, c);
    };
    const sub = (t) => el('span', { text: t });
    const tail = (f, n) => rs.slice(-n).map(f);
    if (!last) {
      KPI_DEFS.forEach((k) => setKpi(k.id, '—', '', null, [], color(0), sub(st && st.state === 'waiting' ? '等待首轮数据' : '')));
      K.eta.input.value = ''; K.eta.bar.firstChild.style.width = '0%';
      return;
    }
    const rt = last.time && last.time.round;
    setKpi('round', fmt.int(last.round), st.rounds_total ? '/ ' + fmt.int(st.rounds_total) : '',
      prev ? el('span', { class: 'delta', title: '较上一轮' }, '+' + (last.round - prev.round)) : null,
      tail((r) => (r.time && r.time.round) || null, 60), color(0), sub(rt ? '本轮 ' + fmt.num(rt, 2) + ' s' : ''));
    setKpi('dec', fmt.compact(last.decisions_total), '', last.decisions_in_round != null ? el('span', { class: 'delta', title: '本轮新增决策' }, '+' + fmt.int(last.decisions_in_round)) : null,
      tail((r) => r.decisions_total, 60), color(2), sub(fmt.int(last.decisions_total)));
    const rates = instantRates(rs);
    const cur = rates[rates.length - 1], pr = rates[rates.length - 2];
    setKpi('rate', st.rate_per_hour == null ? '—' : fmt.compact(st.rate_per_hour, 1), '/ 小时', deltaChip(cur, pr, { pct: true, neutral: true }),
      rates.slice(-60), color(3), sub('近 ' + (st.rate_rounds || 0) + ' 轮'));
    const rw = last.reward_per_decision, rwp = prev && prev.reward_per_decision;
    setKpi('rew', fmt.num(rw, 4), '', deltaChip(rw, rwp, { dec: 4 }), tail((r) => r.reward_per_decision, 60), color(2));
    const ev = last.explained_variance, evp = prev && prev.explained_variance;
    setKpi('ev', fmt.num(ev, 3), '', deltaChip(ev, evp, { dec: 3 }), tail((r) => r.explained_variance, 60), color(7));
    // ETA
    const tgt = T.targets[T.primary];
    const e = K.eta;
    const vEl = e.v, sm = e.small;
    if (document.activeElement !== e.input) e.input.value = tgt ? fmt.compact(tgt, 2) : '';
    e.d.innerHTML = '';
    if (tgt && st.eta_s != null) {
      vEl.textContent = st.eta_s === 0 ? '已达成' : fmt.dur(st.eta_s); sm.textContent = '';
      e.d.appendChild(sub('完成 ' + fmt.pct(st.progress, 1) + ' · ' + fmt.compact(st.decisions_total) + ' / ' + fmt.compact(tgt)));
      e.bar.firstChild.style.width = ((st.progress || 0) * 100).toFixed(1) + '%';
      if (st.eta_s > 0) e.d.appendChild(sub('≈ ' + new Date((Date.now() + st.eta_s * 1000)).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })));
    } else if (tgt) {
      vEl.textContent = '—'; sm.textContent = '';
      e.d.appendChild(sub('速率未知'));
      e.bar.firstChild.style.width = ((st.progress || 0) * 100).toFixed(1) + '%';
    } else {
      vEl.textContent = '—'; sm.textContent = '';
      e.d.appendChild(sub('设定目标决策数后估算'));
      e.bar.firstChild.style.width = '0%';
    }
    if (!tgt && st.suggested_target) {
      e.hint.hidden = false;
      e.hint.textContent = '按配置 ≈ ' + fmt.compact(st.suggested_target, 1);
      e.hint.title = '配置的 ' + st.rounds_total + ' 轮 × 最近每轮决策数';
      e.hint.onclick = () => setTarget(String(st.suggested_target));
    } else e.hint.hidden = true;
  }
  async function setTarget(text) {
    const v = fmt.parseCount(text);
    if (text.trim() === '') delete T.targets[T.primary];
    else if (isFinite(v) && v > 0) T.targets[T.primary] = v;
    else { R.toast('无法识别的数字，例如 5M、2.5e6、500k'); return; }
    store.set('targets', T.targets);
    await loadMetrics(T.primary).catch(() => {});
    renderKPIs();
  }

  // ------------------------------------------------------------------ per-aircraft table
  const AC_COLS = [
    { k: 'name', label: '机型', left: true },
    { k: 'decisions', label: '决策数' },
    { k: 'share', label: '占比' },
    { k: 'rpd', label: '每决策奖励' },
    { k: 'episodes', label: '回合数' },
    { k: 'ret', label: '平均回报' },
  ];
  function renderAircraft() {
    const host = $('#aircraftTable');
    const d = T.data[T.primary];
    const rs = d && d.m ? d.m.rounds : [];
    const last = rs[rs.length - 1];
    const pa = last && last.per_aircraft;
    $('#aircraftSub').textContent = last ? '第 ' + last.round + ' 轮 · ' + runLabel(T.primary) : '';
    if (!pa || !Object.keys(pa).length) { host.innerHTML = '<div class="ev-empty">最近一轮没有分机型数据</div>'; return; }
    const total = Object.values(pa).reduce((s, v) => s + (v.decisions || 0), 0) || 1;
    const rows = Object.entries(pa).map(([name, v]) => ({ name, decisions: v.decisions || 0, share: (v.decisions || 0) / total, rpd: v.reward_per_decision, episodes: v.episodes || 0, ret: v.episode_return_mean }));
    const { key, dir } = T.opt.sort;
    rows.sort((a, b) => {
      const x = a[key], y = b[key];
      if (key === 'name') return dir * String(x).localeCompare(String(y));
      return dir * ((isNum(x) ? x : -Infinity) - (isNum(y) ? y : -Infinity));
    });
    const maxD = Math.max(...rows.map((r) => r.decisions), 1);
    const maxR = Math.max(...rows.map((r) => Math.abs(r.rpd || 0)), 1e-9);
    const table = el('table', { class: 'tbl' });
    const tr = el('tr');
    AC_COLS.forEach((c) => tr.appendChild(el('th', { class: c.left ? '' : '', dataset: { k: c.k }, on: { click: () => {
      T.opt.sort = { key: c.k, dir: T.opt.sort.key === c.k ? -T.opt.sort.dir : (c.k === 'name' ? 1 : -1) }; saveOpt(); renderAircraft();
    } } }, c.label, el('span', { class: 'ar', text: key === c.k ? (dir > 0 ? '↑' : '↓') : '' }))));
    table.appendChild(el('thead', null, tr));
    const tb = el('tbody');
    for (const r of rows) {
      tb.appendChild(el('tr', null,
        el('td', { text: r.name }),
        el('td', null, el('i', { class: 'bar', style: { width: (r.decisions / maxD * 100) + '%' } }), el('span', { text: fmt.int(r.decisions) })),
        el('td', { text: fmt.pct(r.share, 1) }),
        el('td', null, isNum(r.rpd) ? el('i', { class: 'bar ' + (r.rpd >= 0 ? 'pos' : 'neg'), style: { width: (Math.abs(r.rpd) / maxR * 100) + '%' } }) : null,
          el('span', { class: isNum(r.rpd) ? (r.rpd >= 0 ? 'pos-t' : 'neg-t') : '', text: fmt.num(r.rpd, 4) })),
        el('td', { text: fmt.int(r.episodes) }),
        el('td', { text: isNum(r.ret) ? fmt.num(r.ret, 2) : '—' })));
    }
    table.appendChild(tb);
    host.innerHTML = '';
    host.appendChild(table);
  }

  // ------------------------------------------------------------------ BC panel
  function renderBC() {
    const d = T.data[T.primary];
    const body = $('#bcBody');
    const rep = d && d.bc && d.bc.report;
    const sub = $('#bcSub');
    if (!rep) {
      sub.textContent = '';
      if (T.bcChart) { T.bcChart.destroy(); T.bcChart = null; }
      T.bcSig = '';
      body.innerHTML = '<div class="ev-empty">该运行还没有 bc_report.json</div>';
      return;
    }
    const sig = JSON.stringify([T.primary, rep.best_epoch, rep.best_val_loss, (rep.history || []).length, R.C.surface]);
    if (sig === T.bcSig) return;
    T.bcSig = sig;
    const hist = rep.history || [];
    const fv = rep.final_val || (hist.length ? hist[hist.length - 1].val : null) || {};
    const fire = fv.fire || {};
    sub.textContent = '最佳 epoch ' + rep.best_epoch + ' / ' + hist.length;
    const stat = (l, v, s) => '<div class="stat"><div class="l">' + esc(l) + '</div><div class="v">' + esc(v) + '</div>' + (s ? '<div class="s">' + esc(s) + '</div>' : '') + '</div>';
    const ds = rep.dataset || {};
    let html = '<div class="bc-grid"><div class="bc-stats">' +
      stat('最佳验证损失', fmt.num(rep.best_val_loss, 3), 'epoch ' + rep.best_epoch) +
      stat('开火精确率', fmt.pct(fire.precision, 1), 'TP ' + (fire.tp ?? '—') + ' · FP ' + (fire.fp ?? '—')) +
      stat('开火召回率', fmt.pct(fire.recall, 1), 'FN ' + (fire.fn ?? '—') + ' · TN ' + (fire.tn ?? '—')) +
      stat('BC 样本', fmt.compact(ds.decisions), (ds.trajectories ?? '—') + ' 轨迹 · ' + (ds.episodes ?? '—') + ' 回合') +
      '</div>';
    html += '<div class="bc-sec"><h4>损失曲线</h4><div class="bc-chart"><canvas id="bcCanvas"></canvas></div></div>';
    // heads
    const hl = fv.head_loss || {}, ha = fv.head_acc || {};
    const names = Object.keys(hl).sort((a, b) => hl[b] - hl[a]);
    if (names.length) {
      const mx = Math.max(...names.map((n) => hl[n]), 1e-9);
      html += '<div class="bc-sec"><h4>各动作头 (验证集)</h4><div class="headrow hd"><span>动作头</span><span></span><span style="text-align:right">损失</span><span style="text-align:right">准确率</span></div>' +
        names.map((n) => '<div class="headrow"><span class="n" title="' + esc(n) + '">' + esc(n) + '</span><span class="b"><i style="width:' + (hl[n] / mx * 100).toFixed(1) + '%"></i></span><span class="v">' + fmt.num(hl[n], 3) + '</span><span class="v acc">' + (ha[n] == null ? '—' : fmt.pct(ha[n], 0)) + '</span></div>').join('') + '</div>';
    }
    // flying
    const fl = rep.flying;
    if (fl && fl.policy && fl.script) {
      const rows = [['每决策奖励', fl.policy.reward_per_decision, fl.script.reward_per_decision, 4]];
      for (const [k, l] of [['launch', '发射 / 100'], ['kill', '击杀 / 100'], ['assist', '助攻 / 100'], ['death', '阵亡 / 100']]) {
        rows.push([l, (fl.policy.events_per_100_decisions || {})[k], (fl.script.events_per_100_decisions || {})[k], 2]);
      }
      html += '<div class="bc-sec"><h4>自主飞行 vs 脚本</h4><table class="fly"><thead><tr><th></th><th>脚本</th><th>BC 策略</th><th>差值</th></tr></thead><tbody>' +
        rows.map(([l, p, s, dd]) => { const df = isNum(p) && isNum(s) ? p - s : null;
          return '<tr><td>' + l + '</td><td>' + fmt.num(s, dd) + '</td><td>' + fmt.num(p, dd) + '</td><td class="' + (df == null ? '' : df >= 0 ? 'pos-t' : 'neg-t') + '">' + (df == null ? '—' : (df >= 0 ? '+' : '') + fmt.num(df, dd)) + '</td></tr>'; }).join('') +
        '</tbody></table><div class="s" style="font-size:11px;color:var(--faint);margin-top:6px">决策数: 脚本 ' + fmt.int(fl.script.decisions) + ' · 策略 ' + fmt.int(fl.policy.decisions) + '</div></div>';
    }
    html += '</div>';
    body.innerHTML = html;
    if (T.bcChart) { T.bcChart.destroy(); T.bcChart = null; }
    const cv = $('#bcCanvas');
    if (cv && hist.length) {
      T.bcChart = lossChart(cv, hist.map((h) => ({ epoch: h.epoch, train: h.train_loss, val: h.val && h.val.loss })), rep.best_epoch);
    }
  }

  /** train / validation loss per BC epoch (used by the report panel and by the log-derived stage view) */
  function lossChart(cv, pts, best) {
    if (!window.Chart) return null;
    const C = R.C;
    const mono = { family: C.mono.split(',')[0].replace(/["']/g, '').trim() + ', monospace', size: 10.5 };
    const num = (v) => (isNum(v) ? v : null);
    return new Chart(cv.getContext('2d'), {
      type: 'line',
      data: { datasets: [
        { label: '训练损失', data: pts.map((h) => ({ x: h.epoch, y: num(h.train) })), borderColor: C.p1, backgroundColor: C.p1, borderWidth: 2, tension: 0.25, pointRadius: 2.5, pointBackgroundColor: C.p1 },
        { label: '验证损失', data: pts.map((h) => ({ x: h.epoch, y: num(h.val) })), borderColor: C.p2, backgroundColor: C.p2, borderWidth: 2, tension: 0.25, pointRadius: pts.map((h) => (h.epoch === best ? 5 : 2.5)), pointBackgroundColor: C.p2 },
      ] },
      options: { responsive: true, maintainAspectRatio: false, parsing: false, animation: false, spanGaps: true, layout: { padding: { right: 6, top: 4 } },
        interaction: { mode: 'index', intersect: false },
        scales: { x: { type: 'linear', grid: { display: false }, border: { color: C.border }, ticks: { color: C.muted, font: mono, precision: 0, maxTicksLimit: 8 } },
                  y: { grid: { color: C.grid, drawTicks: false }, border: { display: false }, ticks: { color: C.muted, font: mono, maxTicksLimit: 4, padding: 6, callback: (v) => tickFmt(v) } } },
        plugins: { legend: { display: true, align: 'end', labels: { color: C.muted, boxWidth: 10, boxHeight: 2, usePointStyle: false, font: { size: 11 } } },
                   tooltip: { backgroundColor: C.surface, titleColor: C.muted, bodyColor: C.text, borderColor: C.border2, borderWidth: 1, padding: 9, cornerRadius: 9,
                              titleFont: mono, bodyFont: mono, callbacks: { title: (it) => 'epoch ' + it[0].parsed.x, label: (it) => ' ' + it.dataset.label + '  ' + fmt.num(it.parsed.y, 3) } } } },
    });
  }

  // ------------------------------------------------------------------ stage view (no PPO round yet: collecting BC data / BC training)
  const STAGE_TEXT = {
    collect: ['采集 BC 数据', '正在用脚本策略跑对局，采集行为克隆所需的示范数据。采集完成后自动进入 BC 训练。'],
    bc: ['BC 阶段 · 行为克隆训练', '用示范数据做行为克隆，得到 PPO 的初始策略和参考策略；训练结束后写出 bc_report.json 并进入 PPO。'],
    ppo_pending: ['PPO 启动中', 'BC 已完成，正在载入 BC 策略并启动 PPO。metrics.jsonl 会在第一轮结束后出现，图表随之出现。'],
    waiting: ['等待数据', '这个运行目录里还没有 train.log 或 metrics.jsonl。'],
  };
  const DS_LABELS = [
    ['decisions', '决策数'], ['trajectories', '轨迹'], ['episodes', '回合'], ['scenarios', '场景'], ['train_trajectories', '训练轨迹'],
    ['val_trajectories', '验证轨迹'], ['fire_steps', '开火步'], ['maneuver_switch_events', '机动切换'], ['vertical_switch_events', '垂直切换'],
    ['maneuver_ref_switch_events', '机动参照切换'], ['chaff_drop_steps', '投放干扰步'], ['free_look_entries', '自由视角'],
    ['mean_trajectory_return', '平均轨迹回报'], ['mean_trajectory_length', '平均轨迹长度'],
  ];
  function renderStage() {
    const d = T.data[T.primary];
    const host = $('#stageBody');
    const sg = d && d.stage;
    if (!sg) { host.innerHTML = ''; return; }
    const sig = JSON.stringify([T.primary, sg.stage, sg.parsed, sg.targets, R.C.surface]);
    if (sig === T.stageSig) return;
    T.stageSig = sig;
    if (T.stageChart) { T.stageChart.destroy(); T.stageChart = null; }
    const P = sg.parsed, tg = sg.targets;
    const [title, desc] = STAGE_TEXT[sg.stage] || [sg.label, ''];
    const run = T.runs.find((r) => r.id === T.primary) || {};
    // --- steppers
    const stepInfo = {
      collect: P.collected != null ? (tg.collect_decisions ? fmt.compact(P.collected, 1) + ' / ' + fmt.compact(tg.collect_decisions, 1) : fmt.compact(P.collected, 1)) + ' 决策' : '',
      bc: P.epochs.length ? 'epoch ' + P.epochs.length + (tg.max_epochs ? ' / ' + tg.max_epochs : '') : '',
      ppo: sg.stage === 'ppo_pending' ? '启动中…' : '',
    };
    const names = { collect: '采集示范数据', bc: 'BC 训练', ppo: 'PPO 训练' };
    const steps = sg.steps.map((st, i) => '<li class="step ' + st.state + '"><span class="dotc">' + (st.state === 'done' ? '✓' : i + 1) + '</span><span class="nm">' + names[st.id] +
      '</span><span class="sb">' + esc(stepInfo[st.id] || (st.state === 'todo' ? '未开始' : '')) + '</span></li>').join('');
    let html = '<div class="card stage-card"><div class="stage-top"><div><div class="eyebrow">当前阶段</div><h2>' + esc(title) + '</h2><p>' + esc(desc) + '</p></div>' +
      '<div class="stage-meta">' + (P.last_clock ? '<span>日志最后一行 <b>' + esc(P.last_clock) + '</b></span>' : '') +
      (P.workers ? '<span>workers <b>' + esc(P.workers) + '</b></span>' : '') +
      (run.kind === 'remote' ? '<span>来源 <b>' + esc(run.path) + '</b> (rsync)</span>' : '') + '</div></div>' +
      '<ol class="stepper">' + steps + '</ol></div>';
    // --- collection + dataset
    const total = P.collected, target = tg.collect_decisions;
    const frac = total != null && target ? Math.min(total / target, 1) : (sg.stage === 'collect' ? 0 : 1);
    html += '<div class="grid two"><div class="card"><div class="card-h"><h3>数据采集</h3><span class="sub">' + (P.n_shards ? P.n_shards + ' 个分片' : '') + '</span></div><div class="card-b">';
    if (total == null) html += '<div class="hint">还没有采集记录。</div>';
    else {
      html += '<div class="prog-row"><div class="big">' + fmt.int(total) + '<small> ' + (target ? '/ ' + fmt.int(target) + ' 决策' : '决策') + '</small></div>' +
        '<div class="rt">' + (target ? fmt.pct(Math.min(total / target, 1), 1) + ' · ' : '') + (P.collect_rate_per_hour ? '约 ' + fmt.compact(P.collect_rate_per_hour, 1) + ' 决策 / 小时' : '') +
        (tg.collect_eta_s ? '<br>预计还需 ' + fmt.dur(tg.collect_eta_s) : '') + '</div></div>' +
        '<div class="bigprog"><i style="width:' + (frac * 100).toFixed(1) + '%"></i></div>';
      html += '<div class="shards">' + P.shards.slice(-16).map((x) => '<span class="shard" title="累计 ' + fmt.int(x.total) + '">#' + x.shard + ' +' + fmt.compact(x.decisions, 1) + '</span>').join('') + '</div>';
    }
    html += '</div></div>';
    html += '<div class="card"><div class="card-h"><h3>数据集统计</h3><span class="sub">' + (P.dataset ? 'bc dataset' : '') + '</span></div><div class="card-b">';
    if (!P.dataset) html += '<div class="hint">采集结束、BC 开始时 train.log 会输出数据集统计。</div>';
    else {
      const known = new Set(DS_LABELS.map((x) => x[0]));
      const items = DS_LABELS.filter(([k]) => P.dataset[k] != null).concat(Object.keys(P.dataset).filter((k) => !known.has(k) && typeof P.dataset[k] !== 'object').map((k) => [k, k]));
      html += '<div class="tiles">' + items.map(([k, l]) => { const v = P.dataset[k]; return '<div class="stat"><div class="l">' + esc(l) + '</div><div class="v">' + (Number.isInteger(v) ? fmt.compact(v, 1) : fmt.num(v, 2)) + '</div></div>'; }).join('') + '</div>';
    }
    html += '</div></div></div>';
    // --- BC epochs (from the log)
    const eps = P.epochs;
    html += '<div class="card"><div class="card-h"><h3>BC 训练</h3><span class="sub">' + (eps.length ? 'epoch ' + eps.length + (tg.max_epochs ? ' / ' + tg.max_epochs : '') +
      (tg.bc_eta_max_s ? ' · 最多还需 ' + fmt.dur(tg.bc_eta_max_s) : '') + (P.done ? ' · 已完成, 最佳 epoch ' + P.done.best_epoch : '') : '') + '</span></div><div class="card-b">';
    if (!eps.length) html += '<div class="hint">' + (sg.stage === 'collect' ? 'BC 训练还没有开始。' : '还没有 epoch 记录。') + '</div>';
    else {
      const best = P.done ? P.done.best_epoch : (eps.filter((e) => isNum(e.val)).sort((a, b) => a.val - b.val)[0] || {}).epoch;
      html += '<div class="bc-chart" style="height:190px"><canvas id="stageCanvas"></canvas></div><table class="epoch-tbl"><thead><tr><th>epoch</th><th>训练损失</th><th>验证损失</th><th>较上一轮</th><th>patience</th></tr></thead><tbody>' +
        eps.slice(-12).map((e, i, arr) => { const prev = i ? arr[i - 1] : (eps.length > arr.length ? eps[eps.length - arr.length - 1] : null); const dv = prev && isNum(e.val) && isNum(prev.val) ? e.val - prev.val : null;
          return '<tr class="' + (e.epoch === best ? 'best' : '') + '"><td>' + e.epoch + '</td><td>' + fmt.num(e.train, 4) + '</td><td>' + fmt.num(e.val, 4) + '</td><td class="' + (dv == null ? '' : dv <= 0 ? 'pos-t' : 'neg-t') + '">' + (dv == null ? '—' : (dv > 0 ? '+' : '') + fmt.num(dv, 4)) + '</td><td>' + e.bad + (tg.patience ? ' / ' + tg.patience : '') + '</td></tr>'; }).join('') + '</tbody></table>';
    }
    html += '</div></div>';
    host.innerHTML = html;
    const cv = $('#stageCanvas');
    if (cv && eps.length) T.stageChart = lossChart(cv, eps, P.done ? P.done.best_epoch : null);
  }

  function applyMode() {
    const stage = T.stageMode;
    if (!stage) T.stageSig = '';
    $('#stageBody').hidden = !stage;
    $('#kpis').hidden = stage; $('#charts').hidden = stage;
    $('#cardAircraft').hidden = stage;
    const d = T.data[T.primary];
    $('#cardBC').hidden = stage && !(d && d.bc && d.bc.report);
  }

  // ------------------------------------------------------------------ log + meta
  function renderLog() {
    const d = T.data[T.primary];
    const box = $('#logBox');
    const lg = d && d.log;
    const sub = $('#logSub');
    if (!lg || !lg.file) {
      sub.textContent = '';
      box.innerHTML = '<div class="ev-empty">该运行目录里没有 train.log</div>';
      T.logSig = '';
      return;
    }
    sub.textContent = lg.file + ' · ' + fmt.bytes(lg.size) + (lg.mtime ? ' · ' + fmt.ago(Date.now() / 1000 + T.skew - lg.mtime) : '');
    const sig = lg.size + ':' + lg.lines.length + ':' + (lg.lines[lg.lines.length - 1] || '');
    if (sig === T.logSig) return;
    T.logSig = sig;
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    const frag = document.createDocumentFragment();
    for (const line of lg.lines) {
      const n = el('span', { class: 'ln' });
      const m = /^(\[\d\d:\d\d:\d\d\])\s?(.*)$/.exec(line);
      const body = m ? m[2] : line;
      if (m) n.appendChild(el('span', { class: 'ts', text: m[1] + ' ' }));
      const cls = /traceback|error|exception|failed|nan\b/i.test(body) ? 'err' : /warn/i.test(body) ? 'warn' : /^round \d+/.test(body) ? 'rd' : /checkpoint|done|finished/i.test(body) ? 'ok' : /^(workers|run dir|bc )/.test(body) ? 'dim' : '';
      n.appendChild(cls ? el('span', { class: cls, text: body }) : document.createTextNode(body));
      frag.appendChild(n);
    }
    box.innerHTML = '';
    box.appendChild(frag);
    if (atBottom || !T.logShown) box.scrollTop = box.scrollHeight;
    T.logShown = true;
  }
  function flatten(o, pre, out) {
    for (const k of Object.keys(o)) {
      const v = o[k], key = pre ? pre + '.' + k : k;
      if (v && typeof v === 'object' && !Array.isArray(v)) flatten(v, key, out); else out.push([key, Array.isArray(v) ? JSON.stringify(v) : String(v)]);
    }
    return out;
  }
  function renderMeta() {
    const d = T.data[T.primary];
    const body = $('#metaBody');
    if (!d) return;
    const ck = (d.ckpts && d.ckpts.ckpts) || [];
    const cfg = d.cfg && d.cfg.config;
    const sig = JSON.stringify([T.primary, ck.map((c) => c.name + c.size), cfg && cfg.name]);
    $('#metaSub').textContent = ck.length ? ck.length + ' 个检查点' : '';
    if (sig === T.metaSig) return;
    T.metaSig = sig;
    let html = '';
    if (ck.length) {
      html += '<div class="ckpts">' + ck.slice(-12).map((c) => '<span class="ckpt" title="' + esc(c.name) + '"><b>R' + (c.round ?? '?') + '</b> · ' + fmt.bytes(c.size) + '</span>').join('') + '</div>';
    } else {
      html += '<div class="ev-empty" style="padding:6px 0 12px;text-align:left">' + (d.ckpts && d.ckpts.remote ? '远端检查点列表尚未取得 (rsync --list-only)' : 'ppo/ 下还没有检查点') + '</div>';
    }
    if (cfg) {
      html += '<details class="cfg"><summary>配置 · ' + esc(cfg.name || '') + '</summary><div class="kv">' +
        flatten(cfg, '', []).map(([k, v]) => '<span class="k">' + esc(k) + '</span><span class="v">' + esc(v) + '</span>').join('') + '</div></details>';
    }
    body.innerHTML = html;
  }

  // ------------------------------------------------------------------ run picker + legend
  function renderPicker() {
    const btn = $('#runPickerBtn');
    const prim = T.runs.find((r) => r.id === T.primary);
    $('#runPickerLabel').textContent = prim ? prim.label + (T.sel.length > 1 ? '  +' + (T.sel.length - 1) : '') : (T.runs.length ? '选择训练运行' : '没有训练运行');
    const dot = $('#runPickerDot');
    dot.style.background = prim ? runColor(prim.id) : 'var(--faint)';
    const pop = $('#runPickerPop');
    if (pop.hidden) return;
    drawPickerPop();
  }
  function drawPickerPop() {
    const pop = $('#runPickerPop');
    pop.innerHTML = '';
    pop.appendChild(el('div', { class: 'pk-head' }, el('span', { text: '训练运行 · 可叠加对比' }), el('span', { text: T.sel.length + ' / ' + T.runs.length })));
    for (const r of T.runs) {
      const on = T.sel.includes(r.id);
      const cb = el('input', { type: 'checkbox', checked: on, 'aria-label': '叠加 ' + r.label });
      const meta = r.round != null
        ? 'R' + r.round + ' · ' + fmt.compact(r.decisions_total || 0) + ' 决策' + (r.age_s != null ? ' · ' + fmt.ago(r.age_s + (Date.now() / 1000 + T.skew - (T.runsAt || 0))) : '')
        : (r.stage_label || '无数据') + (r.log_mtime != null ? ' · 日志 ' + fmt.ago(Date.now() / 1000 + T.skew - r.log_mtime) : '');
      const row = el('div', { class: 'pk-row', on: { click: (e) => { if (e.target.tagName === 'BUTTON') return; if (e.target !== cb) cb.checked = !cb.checked; toggleRun(r.id, cb.checked); } } },
        cb,
        el('div', { class: 'pk-name' }, el('i', { class: 'swatch', style: { background: runColor(r.id) } }), el('span', { class: 't', text: r.label, title: r.path }), r.kind === 'remote' ? el('span', { class: 'tag remote', text: '远端' }) : null),
        el('button', { class: 'pk-primary' + (r.id === T.primary ? ' on' : ''), title: '用于 KPI、分机型、BC、日志面板', text: r.id === T.primary ? '主要' : '设为主要', on: { click: (e) => { e.stopPropagation(); setPrimary(r.id); } } }),
        el('div', { class: 'pk-meta', text: meta }));
      pop.appendChild(row);
    }
    const remotes = T.runs.filter((r) => r.sync);
    if (remotes.length) {
      const foot = el('div', { class: 'pk-foot' });
      for (const r of remotes) {
        const s = r.sync;
        foot.appendChild(el('div', null, '⇅ ' + r.path + ' — ' + (s.error ? '失败: ' + s.error : s.last_ok ? '已同步 ' + fmt.ago(Date.now() / 1000 + T.skew - s.last_ok) : s.syncing ? '同步中…' : '等待首次同步')));
      }
      foot.appendChild(el('button', { class: 'btn small', text: '立即同步', on: { click: syncNow } }));
      pop.appendChild(foot);
    }
    if (!T.runs.length) pop.appendChild(el('div', { class: 'ev-empty', text: '没有找到训练运行' }));
  }
  function toggleRun(id, on) {
    const set = new Set(T.sel);
    if (on) set.add(id); else if (set.size > 1) set.delete(id);
    T.sel = T.runs.map((r) => r.id).filter((x) => set.has(x));
    if (!T.sel.includes(T.primary)) T.primary = T.sel[0];
    persistSel(); refresh(false);
  }
  function setPrimary(id) {
    if (!T.sel.includes(id)) T.sel.push(id);
    T.primary = id; persistSel(); refresh(false);
  }
  function persistSel() { store.set('sel', T.sel); store.set('primary', T.primary); }
  function renderLegend() {
    const host = $('#runLegend');
    host.innerHTML = '';
    for (const id of T.sel) {
      const r = T.runs.find((x) => x.id === id);
      if (!r) continue;
      host.appendChild(el('span', { class: 'run-chip' + (id === T.primary ? ' primary' : ''), title: r.path },
        el('i', { class: 'swatch', style: { background: runColor(id) } }), el('span', { class: 't', text: r.label }),
        el('em', { text: (r.round != null ? 'R' + r.round : (r.stage_label || '无数据')) + (r.last_timestamp ? ' · ' + fmt.clock(r.last_timestamp) : r.log_mtime ? ' · ' + fmt.clock(r.log_mtime) : '') + (id === T.primary && T.sel.length > 1 ? ' · 主要' : '') })));
    }
  }
  async function syncNow() {
    try { await api('/api/sync', { method: 'POST' }); R.toast('已请求立即同步'); setTimeout(() => refresh(false), 2500); } catch (e) { R.toast('同步请求失败: ' + e.message); }
  }

  // ------------------------------------------------------------------ pill + sync chip
  function updatePill() {
    const prim = T.runs.find((r) => r.id === T.primary);
    const chip = $('#syncChip');
    const remotes = T.runs.filter((r) => r.sync);
    if (remotes.length) {
      chip.hidden = false;
      const bad = remotes.filter((r) => r.sync.error);
      const lastOk = Math.max(...remotes.map((r) => r.sync.last_ok || 0));
      chip.className = 'chip chip-sync ' + (bad.length ? 'bad' : 'ok');
      chip.style.cursor = 'pointer';
      chip.textContent = bad.length ? '同步失败' : lastOk ? '远端同步 · ' + fmt.ago(Date.now() / 1000 + T.skew - lastOk) : '远端同步中…';
      chip.title = remotes.map((r) => r.path + ': ' + (r.sync.error || (r.sync.last_ok ? '正常' : '等待首次同步'))).join('\n') + '\n点击立即同步';
      chip.onclick = syncNow;
    } else chip.hidden = true;
    if (T.lastError) { R.setLive('error', '连接中断', T.lastError); return; }
    if (!prim) { R.setLive('', T.loaded ? '没有训练运行' : '连接中'); return; }
    const d = T.data[prim.id];
    const st = d && d.m && d.m.status;
    if (prim.sync && prim.sync.error && (!st || st.state !== 'live')) { R.setLive('error', '同步失败', prim.sync.error); return; }
    if (!st || st.state === 'waiting') {
      const lm = prim.log_mtime, la = lm != null ? Date.now() / 1000 + T.skew - lm : null;
      R.setLive(la != null && la < 180 ? 'live' : la != null ? 'stale' : '', (prim.stage_label || '等待数据') + (la != null ? ' · 日志 ' + fmt.ago(la) : ''), prim.path);
      return;
    }
    const age = st.last_timestamp != null ? Date.now() / 1000 + T.skew - st.last_timestamp : null;
    const live = st.state === 'live' && age != null && age < Math.max(60, 2.5 * (st.median_round_s || 0));
    R.setLive(live ? 'live' : 'stale', (live ? '实时' : '已停滞') + ' · ' + fmt.ago(age), '最近一轮 ' + fmt.clock(st.last_timestamp) + ' · 第 ' + st.round + ' 轮');
  }

  // ------------------------------------------------------------------ data loading
  async function loadMetrics(id) {
    const q = new URLSearchParams({ window: String(T.opt.window) });
    if (T.targets[id]) q.set('target', String(T.targets[id]));
    const m = await api('/api/run/' + id + '/metrics?' + q);
    (T.data[id] = T.data[id] || {}).m = m;
    return m;
  }
  async function loadRun(id, primary) {
    const jobs = [loadMetrics(id)];
    const d = (T.data[id] = T.data[id] || {});
    if (primary) {
      jobs.push(api('/api/run/' + id + '/bc').then((x) => { d.bc = x; }));
      jobs.push(api('/api/run/' + id + '/log?lines=250').then((x) => { d.log = x; }));
      jobs.push(api('/api/run/' + id + '/ckpts').then((x) => { d.ckpts = x; }));
      jobs.push(api('/api/run/' + id + '/config').then((x) => { d.cfg = x; }));
    }
    await Promise.all(jobs);
    if (primary) {
      const none = !(d.m && d.m.rounds.length);
      d.stage = none ? await api('/api/run/' + id + '/stage') : null;
    }
  }
  function reconcileSelection(def) {
    const ids = T.runs.map((r) => r.id);
    T.sel = T.sel.filter((i) => ids.includes(i));
    if (!T.sel.length && ids.length) {
      const best = ids.includes(def) ? def : T.runs.slice().sort((a, b) => (b.mtime || 0) - (a.mtime || 0))[0].id;
      T.sel = [best];
      T.primary = best;
    }
    if (!T.sel.includes(T.primary)) T.primary = T.sel[0] || null;
  }
  function headsOfPrimary() {
    const d = T.data[T.primary];
    const rs = d && d.m ? d.m.rounds : [];
    const set = new Set();
    for (let i = rs.length - 1; i >= Math.max(0, rs.length - 5); i--) {
      Object.keys(rs[i].entropy_head || {}).forEach((k) => set.add(k));
      Object.keys(rs[i].kl_ref_head || {}).forEach((k) => set.add(k));
    }
    return Array.from(set).sort();
  }

  function showEmpty(kind) {
    const e = $('#trainEmpty'), body = $('#trainBody');
    if (!kind) { e.hidden = true; body.hidden = false; return; }
    body.hidden = true; e.hidden = false;
    const glyph = '<svg class="glyph" viewBox="0 0 56 56" fill="none"><circle cx="28" cy="28" r="22" stroke="var(--border-2)" stroke-width="1.5" stroke-dasharray="3 4"/><circle cx="28" cy="28" r="12" stroke="var(--border-2)" stroke-width="1.5"/><path d="M28 17l5 14-5-3-5 3z" fill="var(--accent)" opacity=".9"/></svg>';
    if (kind === 'none') {
      e.innerHTML = glyph + '<h2>还没有训练运行</h2><p>用 <b>--run DIR</b> 指向本地运行目录，或用 <b>--remote HOST:PATH</b> 同步远端运行。可以先跑一个几秒钟的冒烟训练：</p>' +
        '<code>python3 -m rl.train run --config smoke --run-dir outputs/rl_runs/smoke_demo</code>';
    } else if (kind === 'waiting') {
      const r = T.runs.find((x) => x.id === T.primary) || {};
      e.innerHTML = glyph + '<h2>等待 metrics.jsonl</h2><p>' + esc(r.label || '') + ' 目前没有 PPO 轮次记录' +
        (r.stage === 'bc' ? '（BC 阶段已完成，PPO 尚未开始）。' : r.stage === 'collect' ? '（正在采集 BC 数据）。' : '。') + '页面每 10 秒自动刷新。</p>' +
        (r.kind === 'remote' ? '<p>远端运行通过 rsync 每 30 秒同步一次，首次同步可能需要几秒。</p>' : '');
    }
  }

  async function refresh(first) {
    if (T.busy) return;
    T.busy = true;
    try {
      const resp = await api('/api/runs');
      T.runs = resp.runs; T.runsAt = resp.now; T.skew = resp.now - Date.now() / 1000;
      reconcileSelection(resp.default);
      persistSel();
      renderPicker(); renderLegend();
      if (!T.runs.length) { showEmpty('none'); T.lastError = null; updatePill(); T.loaded = true; return; }
      await Promise.all(T.sel.map((id) => loadRun(id, id === T.primary)));
      T.lastError = null; T.loaded = true;
      const d = T.data[T.primary];
      const hasRounds = d && d.m && d.m.rounds.length;
      T.stageMode = !hasRounds;
      const waiting = T.stageMode && d && d.stage && d.stage.stage === 'waiting' && !(d.bc && d.bc.report);
      if (waiting) showEmpty('waiting'); else showEmpty(null);
      applyMode();
      T.heads = headsOfPrimary();
      renderAll(first);
    } catch (e) {
      T.lastError = e.message || String(e);
      console.warn('refresh failed', e);
    } finally {
      T.busy = false;
      T.countdown = REFRESH_S;
      updatePill(); updateRefreshText();
    }
  }
  function renderAll(first) {
    if (T.stageMode) { renderStage(); renderBC(); renderLog(); renderMeta(); return; }
    renderKPIs();
    if (first || !CHARTS[0].chart) { CHARTS.forEach(updateLegend); T.animateNext = true; updateAllCharts(); T.animateNext = false; }
    else { updateAllCharts(); CHARTS.forEach((d) => { if (d.headKey || d.series.length > 1) updateLegend(d); }); }
    renderAircraft(); renderBC(); renderLog(); renderMeta();
  }
  function updateRefreshText() {
    const t = $('#refreshText');
    t.textContent = T.paused ? '自动刷新已暂停' : (T.busy ? '正在刷新…' : '每 ' + REFRESH_S + ' 秒刷新 · ' + T.countdown + 's');
    $('#btnPause').classList.toggle('on', T.paused);
  }

  // ------------------------------------------------------------------ init
  RLD.modules.training = {
    init() {
      const saved = store.get('topt', {});
      T.opt = Object.assign(T.opt, saved);
      T.opt.sel = T.opt.sel || {};
      T.opt.sort = T.opt.sort || { key: 'decisions', dir: -1 };
      T.targets = store.get('targets', {});
      T.sel = store.get('sel', []);
      T.primary = store.get('primary', null);
      T.paused = store.get('paused', false);
      buildKPIs(); buildCards();
      // toolbar controls
      const segX = $('#ctlX');
      const syncX = () => $$('button', segX).forEach((b) => b.classList.toggle('on', b.dataset.v === T.opt.xaxis));
      $$('button', segX).forEach((b) => b.addEventListener('click', () => { T.opt.xaxis = b.dataset.v; saveOpt(); syncX(); updateAllCharts(); }));
      syncX();
      const sm = $('#ctlSmooth'); sm.value = String(T.opt.smooth);
      sm.addEventListener('change', () => { T.opt.smooth = sm.value; saveOpt(); updateAllCharts(); });
      const win = $('#ctlWindow'); win.value = T.opt.window;
      win.addEventListener('change', async () => {
        const v = Math.max(1, Math.min(2000, parseInt(win.value, 10) || 20));
        win.value = v; T.opt.window = v; saveOpt();
        if (T.primary) { await loadMetrics(T.primary).catch(() => {}); renderKPIs(); }
      });
      $('#btnPause').addEventListener('click', () => { T.paused = !T.paused; store.set('paused', T.paused); updateRefreshText(); });
      $('#btnRefresh').addEventListener('click', () => { refresh(false); });
      // picker popover
      const pbtn = $('#runPickerBtn'), pop = $('#runPickerPop');
      pbtn.addEventListener('click', (e) => { e.stopPropagation(); pop.hidden = !pop.hidden; pbtn.setAttribute('aria-expanded', String(!pop.hidden)); if (!pop.hidden) drawPickerPop(); });
      document.addEventListener('click', (e) => { if (!pop.hidden && !pop.contains(e.target)) { pop.hidden = true; pbtn.setAttribute('aria-expanded', 'false'); } });
      document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !pop.hidden) { pop.hidden = true; pbtn.setAttribute('aria-expanded', 'false'); } });
      R.on('theme', () => { if (T.loaded) { T.stageSig = ''; if (T.stageMode) renderStage(); else rebuildCharts(false); T.bcSig = ''; renderBC(); renderKPIs(); renderLegend(); renderPicker(); } });
      R.on('tab', (name) => { if (name === 'train') { refresh(!T.loaded); } });
      R.on('visibility', (vis) => { if (vis && R.tab() === 'train') refresh(false); });
      setInterval(() => {
        if (R.tab() !== 'train' || document.hidden) return;
        if (!T.paused && !T.busy) { T.countdown -= 1; if (T.countdown <= 0) { refresh(false); return; } }
        updateRefreshText(); updatePill();
      }, 1000);
    },
  };
})();
