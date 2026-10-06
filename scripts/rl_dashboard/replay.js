/* WT 控制台 — 回放 tab: top-down canvas player, scrubber with event ticks, altitude chart, aircraft list, event log. */
(function () {
  'use strict';
  const R = window.RLD;
  const { $, $$, el, esc, fmt, store, api, rgba } = R;

  const TEAM_ZH = ['蓝队', '橙队'];
  const SPEEDS = [1, 2, 4, 8, 16, 32];
  const CHAFF_LIFE_S = 15;
  const MISSILE_TRAIL_S = 6;
  const KINDS = {
    launch: '发射', kill: '击杀', death: '阵亡', assist: '助攻', rwr: '告警', seeker_on: '导引头', datalink_lost: '数据链',
    missile_end: '导弹结束', chaff: '干扰弹', phase: '阶段', track_lost: '丢跟踪', end: '结束', missile_error: '错误',
  };
  const KIND_DEFAULT_ON = { launch: 1, kill: 1, death: 1, assist: 1, rwr: 1, end: 1, missile_error: 1 };
  const PHASE_ZH = { climb: '爬升', advance: '推进', suppress: '压制', evade: '规避', crawl: '低空潜行', popup: '跃升', round2: '第二轮', recommit: '重新切入', rush: '突进', home: '返航', '': '—' };
  const CAUSE_ZH = { missile: '导弹击落', crash: '坠毁', out_of_bounds: '出界' };
  const RESULT_ZH = { fuse: '近炸命中', target_dead: '目标已毁', lifetime: '燃尽失效', ground: '触地', error: '出错' };
  const END_ZH = { annihilation: '一方全灭', mutual_annihilation: '同归于尽', time_limit: '时间到', stalemate: '僵局' };
  const DL_ZH = { seeker_track: '导引头已截获', track_lost: '载机丢失跟踪', shooter_dead: '载机阵亡' };

  const P = {
    list: [], cur: null, data: null, loading: false, err: null,
    t: 0, tEnd: 0, playing: false, speed: 4, last: 0, dirty: true,
    view: { cx: 0, cy: 0, zoom: 1, tcx: 0, tcy: 0, tz: 1 }, half: 64000,
    sel: null, follow: false, hover: null, trails: true, aliveOnly: false,
    planes: [], byId: new Map(), missiles: [], events: [], kinds: Object.assign({}, KIND_DEFAULT_ON),
    w: 0, h: 0, dpr: 1, screen: [], screenM: [], hz: 1,
    rows: new Map(), listT: -1, listDirty: true, fEvents: [], fTimes: [], logShown: 0, logSig: '',
    scrubDirty: true, altDirty: true, scrubHover: null, altHover: null, loaded: false, drag: null,
  };

  // ------------------------------------------------------------------ geometry / interpolation
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  function idxAt(arr, t) {            // largest i with arr[i] <= t (0 when t is before the first sample)
    const n = arr.length;
    if (t <= arr[0]) return 0;
    if (t >= arr[n - 1]) return n - 1;
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (arr[m] <= t) lo = m; else hi = m; }
    return lo;
  }
  function lowerBound(arr, t) { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < t) lo = m + 1; else hi = m; } return lo; }
  function upperBound(arr, t) { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= t) lo = m + 1; else hi = m; } return lo; }
  function sample(tr, t, out) {
    const n = tr.t.length;
    if (!n) return false;
    const i = idxAt(tr.t, t), j = Math.min(i + 1, n - 1);
    const t0 = tr.t[i], t1 = tr.t[j];
    const f = t1 > t0 ? clamp((t - t0) / (t1 - t0), 0, 1) : 0;
    out.x = tr.x[i] + (tr.x[j] - tr.x[i]) * f;
    out.y = tr.y[i] + (tr.y[j] - tr.y[i]) * f;
    out.z = tr.z[i] + (tr.z[j] - tr.z[i]) * f;
    const dh = ((tr.h[j] - tr.h[i] + 540) % 360) - 180;
    out.h = tr.h[i] + dh * f;
    out.i = i;
    if (tr.vx) { out.vx = tr.vx[i] + (tr.vx[j] - tr.vx[i]) * f; out.vy = tr.vy[i] + (tr.vy[j] - tr.vy[i]) * f; out.vz = tr.vz[i] + (tr.vz[j] - tr.vz[i]) * f; out.m = tr.m[i]; out.c = tr.c[i]; out.p = tr.p[i]; }
    else { out.s = tr.s[i]; out.d = tr.d[i]; }
    return true;
  }
  const alive = (pl, t) => pl.deathT == null || t < pl.deathT;
  const planeName = (pl) => pl.aircraft + ' #' + pl.id;
  const tcol = (team) => (team === 0 ? R.C.team0 : R.C.team1);
  const nameHtml = (id) => {
    const pl = P.byId.get(id);
    if (!pl) return id == null ? '—' : '#' + id;
    return '<span class="t' + pl.team + '">' + esc(pl.aircraft) + '<span class="m"> #' + pl.id + '</span></span>';
  };
  const phaseZh = (name) => (name in PHASE_ZH ? PHASE_ZH[name] : name || '—');

  // ------------------------------------------------------------------ data preparation
  function prepare(d) {
    P.half = d.header.map_half_m || 64000;
    P.tEnd = d.t_end || 0;
    P.phases = d.phases || [];
    P.planes = []; P.byId = new Map();
    for (const m of d.header.planes) {
      const tr = d.planes[String(m.id)];
      if (!tr) continue;
      const pl = { id: m.id, team: m.team, aircraft: m.aircraft, name: m.name, archetype: m.archetype, skill: m.skill, missile: m.missile,
        missiles0: m.missiles, chaff0: m.chaff, script: m.script, tr, deathT: null, deathCause: null, cur: {}, maxZ: 0 };
      for (let i = 0; i < tr.z.length; i++) if (tr.z[i] > pl.maxZ) pl.maxZ = tr.z[i];
      P.planes.push(pl); P.byId.set(pl.id, pl);
    }
    P.planes.sort((a, b) => a.team - b.team || a.id - b.id);
    P.teamN = [0, 0];
    P.planes.forEach((p) => { P.teamN[p.team]++; });
    P.events = d.events.filter((e) => e && typeof e.t === 'number');
    P.events.sort((a, b) => a.t - b.t);
    P.byKind = {};
    for (const e of P.events) {
      (P.byKind[e.kind] = P.byKind[e.kind] || []).push(e);
      if (e.kind === 'death') { const pl = P.byId.get(e.plane); if (pl) { pl.deathT = e.t; pl.deathCause = e.cause; } }
    }
    for (const k of ['kill', 'launch', 'rwr', 'chaff', 'death']) { P.byKind[k] = P.byKind[k] || []; P.byKind[k + 'T'] = P.byKind[k].map((e) => e.t); }
    P.launchOf = new Map();
    (P.byKind.launch || []).forEach((e) => P.launchOf.set(e.uid, e));
    const endOf = new Map();
    (P.byKind.missile_end || []).forEach((e) => endOf.set(e.uid, e));
    P.missiles = d.missiles.map((m) => ({ uid: m.uid, owner: m.owner, target: m.target, tr: m, cur: {}, t0: m.t[0], tEnd: endOf.has(m.uid) ? endOf.get(m.uid).t : m.t[m.t.length - 1], end: endOf.get(m.uid) }));
    // spawn zones from the first sample of every aircraft
    P.spawn = [0, 1].map((team) => {
      const ps = P.planes.filter((p) => p.team === team);
      if (!ps.length) return null;
      const cx = ps.reduce((s, p) => s + p.tr.x[0], 0) / ps.length, cy = ps.reduce((s, p) => s + p.tr.y[0], 0) / ps.length;
      const r = Math.max(...ps.map((p) => Math.hypot(p.tr.x[0] - cx, p.tr.y[0] - cy)), 2000);
      return { cx, cy, r };
    });
    P.maxZ = Math.max(2000, ...P.planes.map((p) => p.maxZ));
    P.altMax = Math.ceil(P.maxZ / 2000) * 2000;
    // alive step lines for the scrubber
    P.aliveSteps = [0, 1].map((team) => {
      const ds = P.planes.filter((p) => p.team === team && p.deathT != null).map((p) => p.deathT).sort((a, b) => a - b);
      const n = P.teamN[team] || 1;
      const pts = [[0, 1]];
      ds.forEach((t, i) => { pts.push([t, 1 - i / n]); pts.push([t, 1 - (i + 1) / n]); });
      pts.push([P.tEnd, pts[pts.length - 1][1]]);
      return pts;
    });
    P.missileType = {};
    P.logSig = ''; P.logShown = 0; P.listT = -1; P.listDirty = true; P.scrubDirty = true; P.altDirty = true;
    rebuildFilteredEvents();
  }

  function rebuildFilteredEvents() {
    P.fEvents = P.events.filter((e) => P.kinds[e.kind] || (e.kind === 'rwr' && P.kinds.rwr));
    P.fTimes = P.fEvents.map((e) => e.t);
    P.logShown = 0; P.logSig = '';
  }

  // ------------------------------------------------------------------ list / loading
  async function loadList(keepCur) {
    try {
      const r = await api('/api/replays');
      P.list = r.replays; P.dirs = r.dirs;
    } catch (e) { P.list = []; P.err = e.message; }
    renderSelect();
    if (!P.list.length) { showEmpty(true); return; }
    showEmpty(false);
    let id = keepCur && P.cur && P.list.some((x) => x.id === P.cur) ? P.cur : store.get('replay', null);
    if (!P.list.some((x) => x.id === id)) {
      // prefer a mid-sized replay: opens quickly yet shows a real fight
      const pick = P.list.slice().sort((a, b) => Math.abs((a.summary.planes || 0) - 8) - Math.abs((b.summary.planes || 0) - 8))[0];
      id = pick.id;
    }
    $('#rpSelect').value = id;
    if (id !== P.cur || !P.data) await loadReplay(id);
  }
  function showEmpty(on) {
    $('#rpEmpty').hidden = !on; $('#replayBody').hidden = on; $('#rpChips').hidden = on;
    if (on) {
      const dirs = (P.dirs || []).map((d) => '<code>' + esc(d) + '</code>').join(' ');
      $('#rpEmpty').innerHTML = '<svg class="glyph" viewBox="0 0 56 56" fill="none"><rect x="9" y="9" width="38" height="38" rx="6" stroke="var(--border-2)" stroke-width="1.5" stroke-dasharray="3 4"/><path d="M24 20l14 8-14 8z" fill="var(--accent)" opacity=".9"/></svg>' +
        '<h2>没有找到对战回放</h2><p>扫描的目录: ' + (dirs || '(未配置)') + '</p><p>用下面的命令生成一场 2v2，或用 <b>--replays DIR</b> 指向别的目录：</p>' +
        '<code>python3 scripts/run_engagement.py --mode 2v2 --seed 1 --out outputs/engagements/skirmish_2v2.jsonl</code>';
    }
  }
  function renderSelect() {
    const sel = $('#rpSelect');
    sel.innerHTML = '';
    const groups = new Map();
    for (const it of P.list) {
      if (!groups.has(it.group)) groups.set(it.group, []);
      groups.get(it.group).push(it);
    }
    for (const [g, items] of groups) {
      const og = el('optgroup', { label: g });
      for (const it of items) {
        const s = it.summary || {};
        const t = s.teams ? s.teams[0] + 'v' + s.teams[1] : s.planes + ' 架';
        og.appendChild(el('option', { value: it.id, text: it.name + ' · ' + t + (s.duration_s != null ? ' · ' + fmt.mmss(s.duration_s) : '') + ' · ' + fmt.bytes(it.size) + (s.complete ? '' : ' · 未结束') }));
      }
      sel.appendChild(og);
    }
  }
  function overlay(html) {
    const o = $('#mapOverlay');
    o.classList.toggle('show', !!html);
    o.innerHTML = html || '';
  }
  async function loadReplay(id, keepState) {
    if (P.loading) return;
    P.loading = true; P.cur = id; store.set('replay', id);
    const item = P.list.find((x) => x.id === id);
    overlay('<div><div class="spinner"></div><div>正在载入 ' + esc(item ? item.name : '') + ' …</div><div class="s" style="color:var(--faint);font-size:12px;margin-top:4px">' + (item ? fmt.bytes(item.size) : '') + ' · 服务器降采样到 ' + P.hz + ' Hz</div></div>');
    const keep = keepState ? { t: P.t, sel: P.sel, follow: P.follow, view: Object.assign({}, P.view) } : null;
    try {
      const t0 = performance.now();
      const d = await api('/api/replay/' + id + '?hz=' + P.hz);
      P.data = d; P.err = null;
      prepare(d);
      P.t = 0; P.playing = false; P.sel = null; P.follow = false;
      Object.assign(P.view, { cx: 0, cy: 0, zoom: 1, tcx: 0, tcy: 0, tz: 1 });
      if (keep) { P.t = Math.min(keep.t, P.tEnd); if (keep.sel != null && P.byId.has(keep.sel)) { P.sel = keep.sel; P.follow = keep.follow; } Object.assign(P.view, keep.view); }
      P.loadMs = performance.now() - t0;
      buildChips(); buildPlaneList(); buildFilters(); syncTransport(); layoutAll();
      overlay('');
      P.dirty = P.scrubDirty = P.altDirty = true;
    } catch (e) {
      P.data = null;
      overlay('<div><div style="font-weight:600;color:var(--text);margin-bottom:6px">回放载入失败</div><div style="max-width:420px">' + esc(e.message) + '</div><div style="margin-top:12px"><button class="btn" id="rpRetry">重试</button></div></div>');
      const b = $('#rpRetry'); if (b) b.onclick = () => loadReplay(id);
    } finally { P.loading = false; }
  }

  function buildChips() {
    const d = P.data, s = (P.list.find((x) => x.id === P.cur) || {}).summary || {};
    const host = $('#rpChips');
    host.innerHTML = '';
    const chip = (html) => host.appendChild(el('span', { class: 'info-chip', html }));
    const n = P.teamN;
    chip('<span class="t0">蓝 <b>' + n[0] + '</b></span> v <span class="t1"><b>' + n[1] + '</b> 橙</span>');
    chip('时长 <b>' + fmt.mmss(P.tEnd) + '</b>');
    const end = d.end;
    chip(end ? '结束: <b>' + esc(END_ZH[end.reason] || end.reason) + '</b>' : '<b style="color:var(--warn)">未结束</b>');
    chip('击杀 <b>' + (P.byKind.kill || []).length + '</b> · 发射 <b>' + (P.byKind.launch || []).length + '</b>');
    chip('种子 <b>' + esc(d.header.seed) + '</b>');
    chip('<span style="color:var(--muted)">' + d.sample_hz + ' Hz · ' + fmt.int(d.frames_kept) + ' / ' + fmt.int(d.frames_total) + ' 帧</span>');
    if (d.bad_lines) chip('<span style="color:var(--warn)">' + d.bad_lines + ' 行损坏已跳过</span>');
  }

  // ------------------------------------------------------------------ plane list
  function buildPlaneList() {
    const host = $('#planeList');
    host.innerHTML = '';
    P.rows = new Map();
    for (const team of [0, 1]) {
      const ps = P.planes.filter((p) => p.team === team);
      if (!ps.length) continue;
      const head = el('div', { class: 'pteam' }, el('i', { style: { background: tcol(team) } }), el('span', { text: TEAM_ZH[team] }), el('span', { class: 'cnt', text: '' }));
      host.appendChild(head);
      ps.forEach((p) => {
        const ph = el('span', { class: 'phase', text: '—' });
        const ms = el('span', { class: 'ms' });
        const row = el('div', { class: 'prow', dataset: { id: p.id }, on: {
          click: () => selectPlane(P.sel === p.id ? null : p.id, true),
          mouseenter: () => { P.hover = p.id; P.dirty = true; }, mouseleave: () => { if (P.hover === p.id) { P.hover = null; P.dirty = true; } },
        } },
          el('i', { class: 'bar', style: { background: tcol(p.team) } }),
          el('div', null, el('div', { class: 'nm' }, el('span', { text: p.aircraft, title: p.aircraft }), el('em', { text: '#' + p.id })),
            el('div', { class: 'ds', text: [p.archetype, p.skill].filter(Boolean).join(' · ') || '—', title: p.missile ? '导弹 ' + p.missile : '' })),
          el('div', { class: 'rt' }, ph, ms));
        host.appendChild(row);
        P.rows.set(p.id, { row, ph, ms, cnt: head.lastChild, team, last: '' });
      });
      P.rows.set('head' + team, { cnt: head.lastChild });
    }
    P.listDirty = true;
  }
  function updatePlaneList(t) {
    if (!P.data) return;
    if (!P.listDirty && Math.abs(t - P.listT) < 0.09) return;
    P.listT = t; P.listDirty = false;
    const cnt = [0, 0];
    for (const p of P.planes) {
      const r = P.rows.get(p.id);
      if (!r) continue;
      const a = alive(p, t);
      if (a) cnt[p.team]++;
      sample(p.tr, t, p.cur);
      const key = (a ? '1' : '0') + (a ? p.cur.p + ':' + p.cur.m + ':' + p.cur.c : fmt.mmss(p.deathT));
      if (r.last !== key) {
        r.last = key;
        r.row.classList.toggle('dead', !a);
        r.ph.textContent = a ? phaseZh(P.phases[p.cur.p]) : '阵亡 ' + fmt.mmss(p.deathT);
        r.ph.title = a ? (P.phases[p.cur.p] || '') : (CAUSE_ZH[p.deathCause] || p.deathCause || '');
        r.ms.innerHTML = a ? '导弹 <b>' + p.cur.m + '</b>' + (p.missiles0 != null ? '/' + p.missiles0 : '') + ' · 干扰 <b>' + p.cur.c + '</b>' : '';
      }
      r.row.classList.toggle('sel', P.sel === p.id);
      r.row.style.display = P.aliveOnly && !a ? 'none' : '';
    }
    for (const team of [0, 1]) { const h = P.rows.get('head' + team); if (h) h.cnt.textContent = cnt[team] + ' / ' + P.teamN[team] + ' 存活'; }
    P.alive = cnt;
  }
  function selectPlane(id, follow) {
    P.sel = id;
    P.follow = id != null && !!follow;
    if (P.follow) {
      const pl = P.byId.get(id);
      if (pl && P.view.tz < 3) P.view.tz = 4;
      if (pl && !alive(pl, P.t)) P.follow = false;
    } else if (id == null) { P.view.tz = Math.min(P.view.tz, P.view.zoom); }
    P.listDirty = true; P.altDirty = true; P.dirty = true;
    syncTransport();
    if (id != null) { const r = P.rows.get(id); if (r) r.row.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }
  }

  // ------------------------------------------------------------------ event log
  function evText(e) {
    const km = (m) => (m / 1000).toFixed(1) + ' km';
    switch (e.kind) {
      case 'launch': return nameHtml(e.shooter) + ' 发射 <span class="m">' + esc(e.missile || '') + '</span> → ' + nameHtml(e.target) + ' <span class="m">· ' + km(e.range_m) + ' · 余 ' + e.left + '</span>';
      case 'kill': return e.killer == null ? nameHtml(e.victim) + ' ' + esc(CAUSE_ZH[e.cause] || e.cause) : nameHtml(e.killer) + ' 击杀 ' + nameHtml(e.victim) + ' <span class="m">· 导弹 #' + e.uid + '</span>';
      case 'death': return nameHtml(e.plane) + ' 阵亡 <span class="m">· ' + esc(CAUSE_ZH[e.cause] || e.cause) + '</span>';
      case 'assist': return nameHtml(e.plane) + ' 获得助攻 <span class="m">· ' + nameHtml(e.victim) + '</span>';
      case 'rwr': return nameHtml(e.plane) + (e.warning === 'missile' ? ' 导弹来袭告警' : ' 被雷达锁定') + ' <span class="m">· 方位 ' + Math.round(e.bearing_deg) + '°</span>';
      case 'seeker_on': return '导弹 #' + e.uid + ' 导引头开机 → ' + nameHtml(e.target) + ' <span class="m">· ' + km(e.range_m) + '</span>';
      case 'datalink_lost': return '导弹 #' + e.uid + ' 数据链中断 <span class="m">· ' + esc(DL_ZH[e.reason] || e.reason) + '</span>';
      case 'missile_end': return '导弹 #' + e.uid + ' ' + esc(RESULT_ZH[e.result] || e.result) + ' <span class="m">· 脱靶 ' + e.miss_m + ' m · 飞行 ' + e.flight_s + ' s</span>';
      case 'chaff': return nameHtml(e.plane) + ' 投放干扰 ×' + e.n + ' <span class="m">· 余 ' + e.left + '</span>';
      case 'phase': return nameHtml(e.plane) + ' <span class="m">' + esc(phaseZh(e.frm)) + ' → </span>' + esc(phaseZh(e.to));
      case 'track_lost': return nameHtml(e.plane) + ' 丢失对 ' + nameHtml(e.target) + ' 的跟踪' + (e.supporting ? ' <span class="m">· 影响制导</span>' : '');
      case 'end': return '对局结束 · ' + esc(END_ZH[e.reason] || e.reason);
      default: return esc(e.kind) + ' <span class="m">' + esc(JSON.stringify(e).slice(0, 80)) + '</span>';
    }
  }
  function evRow(e) {
    return el('div', { class: 'ev', dataset: { t: e.t }, on: { click: () => seek(e.t) } },
      el('span', { class: 'tm', text: fmt.mmss(e.t) }),
      el('span', { class: 'kd ' + e.kind, text: KINDS[e.kind] || e.kind }),
      el('span', { class: 'tx', html: evText(e) }));
  }
  function updateLog(t) {
    const log = $('#evLog');
    const n = upperBound(P.fTimes, t);
    const sig = P.logSig;
    $('#evCount').textContent = n + ' / ' + P.fEvents.length;
    if (sig !== 'ok' || n < P.logShown) { log.innerHTML = ''; P.logShown = 0; P.logSig = 'ok'; }
    if (n === P.logShown) { if (!log.children.length) log.innerHTML = '<div class="ev-empty">时间尚未到第一个事件</div>'; return; }
    if (log.firstChild && log.firstChild.className === 'ev-empty') log.innerHTML = '';
    const stick = log.scrollHeight - log.scrollTop - log.clientHeight < 36;
    const from = Math.max(P.logShown, n - 400);
    if (from > P.logShown) log.innerHTML = '';
    const frag = document.createDocumentFragment();
    for (let i = from; i < n; i++) frag.appendChild(evRow(P.fEvents[i]));
    log.appendChild(frag);
    while (log.children.length > 420) log.removeChild(log.firstChild);
    P.logShown = n;
    if (stick || P.playing) log.scrollTop = log.scrollHeight;
  }
  function buildFilters() {
    const host = $('#evFilters');
    host.innerHTML = '';
    const counts = {};
    P.events.forEach((e) => { counts[e.kind] = (counts[e.kind] || 0) + 1; });
    for (const k of Object.keys(KINDS)) {
      if (!counts[k] || k === 'end') continue;
      host.appendChild(el('button', { class: 'chip-btn' + (P.kinds[k] ? ' on' : ''), title: counts[k] + ' 条', text: KINDS[k], on: { click: (ev) => {
        P.kinds[k] = !P.kinds[k]; ev.currentTarget.classList.toggle('on', !!P.kinds[k]); store.set('rpKinds', P.kinds);
        rebuildFilteredEvents(); updateLog(P.t);
      } } }));
    }
  }

  // ------------------------------------------------------------------ transport
  const ICON_PLAY = '<svg viewBox="0 0 16 16" class="ico" style="width:18px;height:18px"><path d="M5 3.2v9.6L13 8z" fill="currentColor"/></svg>';
  const ICON_PAUSE = '<svg viewBox="0 0 16 16" class="ico" style="width:18px;height:18px"><path d="M5.2 3.2v9.6M10.8 3.2v9.6" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg>';
  function syncTransport() {
    $('#tpPlay').innerHTML = P.playing ? ICON_PAUSE : ICON_PLAY;
    $$('#tpSpeed button').forEach((b) => b.classList.toggle('on', +b.dataset.v === P.speed));
    $('#tpTrails').classList.toggle('on', P.trails);
    $('#tpFollow').classList.toggle('on', P.follow);
    $('#tpFollow').disabled = P.sel == null;
    $('#plAliveOnly').classList.toggle('on', P.aliveOnly);
    $('#tpEnd').textContent = '/ ' + fmt.mmss(P.tEnd);
    updateHudView();
  }
  function togglePlay(force) {
    if (!P.data) return;
    const on = force == null ? !P.playing : force;
    if (on && P.t >= P.tEnd - 1e-3) P.t = 0;
    P.playing = on; P.last = performance.now(); syncTransport();
  }
  function seek(t) { P.t = clamp(t, 0, P.tEnd); P.dirty = true; P.scrubDirty = false; }
  function setSpeed(s) { P.speed = clamp(s, 1, 32); store.set('rpSpeed', P.speed); syncTransport(); P.dirty = true; }
  function jumpKill(dir) {
    const ts = P.byKind.killT || [];
    if (!ts.length) return;
    if (dir > 0) { const i = upperBound(ts, P.t + 0.05); if (i < ts.length) seek(ts[i]); }
    else { const i = lowerBound(ts, P.t - 0.6) - 1; if (i >= 0) seek(ts[i]); else seek(0); }
  }
  function resetView() { Object.assign(P.view, { cx: 0, cy: 0, zoom: 1, tcx: 0, tcy: 0, tz: 1 }); P.follow = false; syncTransport(); P.dirty = true; }
  function updateHudView() {
    const host = $('#hudView');
    host.innerHTML = '';
    if (P.follow && P.sel != null) {
      const pl = P.byId.get(P.sel);
      host.appendChild(el('button', { class: 'chip-btn on', title: '取消跟随 (F)', text: '跟随 ' + (pl ? planeName(pl) : '') + '  ✕', on: { click: () => { P.follow = false; syncTransport(); } } }));
    }
    if (P.view.zoom > 1.05 || P.view.tz > 1.05 || P.view.cx !== 0 || P.view.cy !== 0) {
      host.appendChild(el('button', { class: 'chip-btn', title: '重置视图 (R / 双击)', text: '全图', on: { click: resetView } }));
    }
  }

  // ------------------------------------------------------------------ canvas setup
  let mapCv, mapCtx, scrubCv, scrubCtx, altCv, altCtx, scrubCache, altCache;
  function sizeCanvas(cv, wrap) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(wrap.clientWidth)), h = Math.max(1, Math.round(wrap.clientHeight));
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    return { w, h, dpr };
  }
  function layoutAll() {
    if (!mapCv) return;
    const m = sizeCanvas(mapCv, $('#mapStage')); P.w = m.w; P.h = m.h; P.dpr = m.dpr;
    const s = sizeCanvas(scrubCv, $('#scrubWrap')); P.sw = s.w; P.sh = s.h; P.sdpr = s.dpr;
    const a = sizeCanvas(altCv, $('#altWrap')); P.aw = a.w; P.ah = a.h; P.adpr = a.dpr;
    P.scrubDirty = P.altDirty = P.dirty = true;
  }

  // ------------------------------------------------------------------ map drawing
  const scale = () => (Math.min(P.w, P.h) / (2 * P.half)) * 0.94 * P.view.zoom;
  const SX = (x, s) => P.w / 2 + (x - P.view.cx) * s;
  const SY = (y, s) => P.h / 2 - (y - P.view.cy) * s;
  function haloText(ctx, text, x, y, color, halo) {
    ctx.lineWidth = 3; ctx.strokeStyle = halo; ctx.lineJoin = 'round';
    ctx.strokeText(text, x, y); ctx.fillStyle = color; ctx.fillText(text, x, y);
  }
  function drawGrid(ctx, s) {
    const C = R.C, w = P.w, h = P.h, half = P.half;
    // outside shade + bounds
    const x0 = SX(-half, s), x1 = SX(half, s), y0 = SY(half, s), y1 = SY(-half, s);
    ctx.fillStyle = C.mapOut;
    ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.rect(x0, y0, x1 - x0, y1 - y0); ctx.fill('evenodd');
    ctx.fillStyle = rgba(C.text, 0.018); ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    // grid
    const steps = [1000, 2000, 5000, 10000, 20000, 50000];
    let step = steps[steps.length - 1];
    for (const c of steps) if (c * s >= 52) { step = c; break; }
    ctx.save();
    ctx.beginPath(); ctx.rect(Math.max(0, x0), Math.max(0, y0), Math.min(w, x1) - Math.max(0, x0), Math.min(h, y1) - Math.max(0, y0)); ctx.clip();
    const vx0 = P.view.cx - w / 2 / s, vx1 = P.view.cx + w / 2 / s, vy0 = P.view.cy - h / 2 / s, vy1 = P.view.cy + h / 2 / s;
    ctx.lineWidth = 1;
    for (let k = Math.ceil(vx0 / step); k * step <= vx1; k++) {
      const x = Math.round(SX(k * step, s)) + 0.5, major = k === 0 || (k * step) % (step * 5) === 0;
      ctx.strokeStyle = major ? C.gridStrong : C.grid;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let k = Math.ceil(vy0 / step); k * step <= vy1; k++) {
      const y = Math.round(SY(k * step, s)) + 0.5, major = k === 0 || (k * step) % (step * 5) === 0;
      ctx.strokeStyle = major ? C.gridStrong : C.grid;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    ctx.restore();
    // bounds frame
    ctx.save();
    ctx.strokeStyle = rgba(C.accent, 0.5); ctx.lineWidth = 1.25; ctx.setLineDash([6, 5]);
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0, y1 - y0);
    ctx.restore();
    // axis labels (every grid line; anchored to the map bounds, or to the canvas edge when the bounds are off screen)
    ctx.font = '500 10px ' + C.mono; ctx.textBaseline = 'alphabetic';
    const halo = rgba(C.mapBg2, 0.9);
    const ly = clamp(y1 + 14, 12, h - 6), lx = x0 - 6;
    ctx.textAlign = 'center';
    for (let k = Math.ceil(vx0 / step); k * step <= vx1; k++) {
      if (Math.abs(k * step) > half) continue;
      const x = SX(k * step, s); if (x < 16 || x > w - 16) continue;
      haloText(ctx, String(k * step / 1000), x, ly, C.faint, halo);
    }
    for (let k = Math.ceil(vy0 / step); k * step <= vy1; k++) {
      if (Math.abs(k * step) > half) continue;
      const y = SY(k * step, s); if (y < 14 || y > h - 22) continue;
      if (lx > 22) { ctx.textAlign = 'right'; haloText(ctx, String(k * step / 1000), lx, y + 3.5, C.faint, halo); }
      else { ctx.textAlign = 'left'; haloText(ctx, String(k * step / 1000), 6, y + 3.5, C.faint, halo); }
    }
    ctx.textAlign = 'right';
    haloText(ctx, 'km', w - 8, h - 6, C.muted, halo);
    // compass
    ctx.textAlign = 'center'; ctx.font = '600 10px ' + C.font;
    haloText(ctx, '北 ↑', w - 26, 20, C.muted, halo);
  }
  function drawSpawn(ctx, s) {
    const C = R.C;
    ctx.font = '600 11px ' + C.font; ctx.textAlign = 'center';
    P.spawn.forEach((z, team) => {
      if (!z) return;
      const x = SX(z.cx, s), y = SY(z.cy, s), r = (z.r + 2500) * s;
      const g = ctx.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0, rgba(tcol(team), 0.13)); g.addColorStop(1, rgba(tcol(team), 0));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill();
      ctx.save(); ctx.strokeStyle = rgba(tcol(team), 0.28); ctx.setLineDash([3, 5]); ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, r * 0.86, 0, 6.2832); ctx.stroke(); ctx.restore();
      const up = z.cy >= 0;
      const ly = clamp(up ? y - r * 0.86 - 7 : y + r * 0.86 + 15, 14, P.h - 24);
      haloText(ctx, TEAM_ZH[team] + '出生区', clamp(x, 50, P.w - 50), ly, rgba(tcol(team), 0.92), rgba(C.mapBg2, 0.85));
    });
  }
  function bucketTrail(ctx, pts, color, a0, width) {
    // pts: [[x, y, u]] newest first, u = age fraction 0..1. Strokes in alpha buckets (one path per bucket).
    const NB = 6;
    for (let b = 0; b < NB; b++) {
      const lo = b / NB, hi = (b + 1) / NB;
      ctx.beginPath(); let any = false;
      for (let i = 0; i + 1 < pts.length; i++) {
        const u = pts[i + 1][2];
        if (u < lo || u >= hi) continue;
        ctx.moveTo(pts[i][0], pts[i][1]); ctx.lineTo(pts[i + 1][0], pts[i + 1][1]); any = true;
      }
      if (!any) continue;
      ctx.strokeStyle = rgba(color, a0 * Math.pow(1 - (b + 0.5) / NB, 1.35));
      ctx.lineWidth = width; ctx.stroke();
    }
  }
  function trailPoints(tr, t, cur, span, s) {
    const pts = [[SX(cur.x, s), SY(cur.y, s), 0]];
    let i = idxAt(tr.t, t);
    if (tr.t[i] > t) i--;
    for (let k = i; k >= 0; k--) {
      const age = t - tr.t[k];
      if (age > span) { pts.push([SX(tr.x[k], s), SY(tr.y[k], s), 1]); break; }
      if (age <= 0) continue;
      pts.push([SX(tr.x[k], s), SY(tr.y[k], s), age / span]);
    }
    return pts;
  }
  function chevron(ctx, x, y, hdg, sz, color, glow, ring) {
    ctx.save();
    ctx.translate(x, y); ctx.rotate((hdg * Math.PI) / 180);
    ctx.beginPath(); ctx.moveTo(0, -sz); ctx.lineTo(sz * 0.66, sz * 0.84); ctx.lineTo(0, sz * 0.44); ctx.lineTo(-sz * 0.66, sz * 0.84); ctx.closePath();
    ctx.shadowColor = color; ctx.shadowBlur = glow; ctx.fillStyle = color; ctx.fill();
    ctx.shadowBlur = 0; ctx.lineWidth = 1; ctx.strokeStyle = rgba('#ffffff', ring ? 0.9 : 0.5); ctx.stroke();
    ctx.restore();
  }
  function ringFx(ctx, x, y, u, r0, r1, color, a) {
    if (u < 0 || u > 1) return;
    ctx.beginPath(); ctx.arc(x, y, r0 + (r1 - r0) * (1 - Math.pow(1 - u, 2)), 0, 6.2832);
    ctx.strokeStyle = rgba(color, a * (1 - u)); ctx.lineWidth = 1.6 * (1 - u * 0.5) + 0.4; ctx.stroke();
  }
  function drawEffects(ctx, s, t) {
    const C = R.C, sp = Math.max(1, P.speed);
    // launch flashes
    const dl = 0.9 * sp;
    for (let i = lowerBound(P.byKind.launchT, t - dl); i < P.byKind.launch.length; i++) {
      const e = P.byKind.launch[i]; if (e.t > t) break;
      if (e.x == null) continue;
      const u = (t - e.t) / dl, x = SX(e.x, s), y = SY(e.y, s);
      ringFx(ctx, x, y, u, 4, 18, C.missile, 0.9);
    }
    // RWR pulses on the warned aircraft
    const dr = 1.5 * sp;
    for (let i = lowerBound(P.byKind.rwrT, t - dr); i < P.byKind.rwr.length; i++) {
      const e = P.byKind.rwr[i]; if (e.t > t) break;
      const pl = P.byId.get(e.plane); if (!pl || !alive(pl, t)) continue;
      sample(pl.tr, t, pl.cur);
      const u = (t - e.t) / dr, x = SX(pl.cur.x, s), y = SY(pl.cur.y, s);
      const col = e.warning === 'missile' ? C.kill : C.warn;
      ringFx(ctx, x, y, u, 8, 26, col, 0.8); ringFx(ctx, x, y, clamp(u - 0.25, -1, 1), 8, 26, col, 0.5);
    }
    // kill bursts
    const dk = 1.9 * sp;
    for (let i = lowerBound(P.byKind.killT, t - dk); i < P.byKind.kill.length; i++) {
      const e = P.byKind.kill[i]; if (e.t > t) break;
      if (e.x == null) continue;
      const u = (t - e.t) / dk, x = SX(e.x, s), y = SY(e.y, s);
      ctx.save();
      ringFx(ctx, x, y, u, 3, 38, C.kill, 1);
      ringFx(ctx, x, y, clamp(u * 1.4 - 0.2, -1, 1), 2, 22, '#ffffff', 0.7);
      const n = 10;
      ctx.lineCap = 'round';
      for (let k = 0; k < n; k++) {
        const a = (k / n) * 6.2832 + (e.uid || 0), r0 = 6 + 26 * u, r1 = r0 + 9 * (1 - u);
        ctx.strokeStyle = rgba(k % 2 ? C.kill : C.warn, 0.9 * (1 - u)); ctx.lineWidth = 1.7;
        ctx.beginPath(); ctx.moveTo(x + Math.cos(a) * r0, y + Math.sin(a) * r0); ctx.lineTo(x + Math.cos(a) * r1, y + Math.sin(a) * r1); ctx.stroke();
      }
      if (u < 0.35) { ctx.fillStyle = rgba('#ffffff', 0.9 * (1 - u / 0.35)); ctx.beginPath(); ctx.arc(x, y, 5 * (1 - u / 0.35) + 2, 0, 6.2832); ctx.fill(); }
      ctx.restore();
    }
  }
  function hash01(n) { const x = Math.sin(n * 12.9898) * 43758.5453; return x - Math.floor(x); }
  function drawChaff(ctx, s, t) {
    const C = R.C, evs = P.byKind.chaff;
    for (let i = lowerBound(P.byKind.chaffT, t - CHAFF_LIFE_S); i < evs.length; i++) {
      const e = evs[i]; if (e.t > t) break;
      if (e.x == null) continue;
      const age = t - e.t, k = age / CHAFF_LIFE_S;
      const n = Math.min(e.n || 1, 6);
      for (let j = 0; j < n; j++) {
        const a = hash01(i * 7 + j) * 6.2832, rr = (60 + 340 * Math.sqrt(k)) * (0.4 + hash01(i * 13 + j * 3));
        const x = SX(e.x + Math.cos(a) * rr, s), y = SY(e.y + Math.sin(a) * rr, s);
        ctx.fillStyle = rgba(C.text2, 0.8 * (1 - k) * (1 - k * 0.4));
        ctx.beginPath(); ctx.arc(x, y, 1.5, 0, 6.2832); ctx.fill();
      }
    }
  }
  function drawMissiles(ctx, s, t) {
    const C = R.C;
    P.screenM = [];
    const sel = P.sel;
    for (const m of P.missiles) {
      if (t < m.t0 || t > m.tEnd + 0.05) continue;
      const c = m.cur;
      sample(m.tr, t, c);
      const owner = P.byId.get(m.owner), team = owner ? owner.team : 0, tc = tcol(team);
      const x = SX(c.x, s), y = SY(c.y, s);
      if (x < -40 || y < -40 || x > P.w + 40 || y > P.h + 40) continue;
      P.screenM.push({ m, x, y });
      // thin trail
      const pts = trailPoints(m.tr, t, c, MISSILE_TRAIL_S, s);
      bucketTrail(ctx, pts, tc, 0.75, 1.1);
      // link to the target when the selected aircraft is involved
      if (sel != null && (m.owner === sel || m.target === sel)) {
        const tg = P.byId.get(m.target);
        if (tg && alive(tg, t)) {
          sample(tg.tr, t, tg.cur);
          ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = rgba(tc, 0.6); ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(SX(tg.cur.x, s), SY(tg.cur.y, s)); ctx.stroke(); ctx.restore();
        }
      }
      // bright streak
      const h = (c.h * Math.PI) / 180, dx = Math.sin(h), dy = -Math.cos(h), L = 11;
      ctx.save();
      ctx.lineCap = 'round';
      ctx.shadowColor = C.missile; ctx.shadowBlur = 8;
      ctx.strokeStyle = C.missile; ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.moveTo(x - dx * L, y - dy * L); ctx.lineTo(x, y); ctx.stroke();
      ctx.shadowBlur = 0; ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x - dx * L * 0.55, y - dy * L * 0.55); ctx.lineTo(x, y); ctx.stroke();
      if (c.s) { ctx.strokeStyle = rgba(C.kill, 0.85); ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(x, y, 4.6, 0, 6.2832); ctx.stroke(); }
      ctx.restore();
    }
  }
  function drawPlanes(ctx, s, t) {
    const C = R.C;
    P.screen = [];
    const z = P.view.zoom;
    const sz = 6.5 + Math.min(4, Math.log2(z + 1) * 1.6);
    const glow = 9;
    // trails
    if (P.trails) {
      const span = Math.max(40, P.speed * 5);
      for (const pl of P.planes) {
        if (!alive(pl, t)) continue;
        sample(pl.tr, t, pl.cur);
        const isSel = P.sel === pl.id, dim = P.sel != null && !isSel;
        const pts = trailPoints(pl.tr, t, pl.cur, span, s);
        bucketTrail(ctx, pts, tcol(pl.team), (isSel ? 0.95 : dim ? 0.2 : 0.62), isSel ? 2 : 1.3);
      }
    }
    drawChaff(ctx, s, t);
    drawMissiles(ctx, s, t);
    // wrecks
    for (const e of P.byKind.death) {
      if (e.t > t || e.x == null) continue;
      const pl = P.byId.get(e.plane);
      const x = SX(e.x, s), y = SY(e.y, s), a = clamp(0.2 + (t - e.t) / 3, 0, 1) * 0.6;
      ctx.strokeStyle = rgba(pl ? tcol(pl.team) : C.kill, a); ctx.lineWidth = 1.6; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(x - 3.5, y - 3.5); ctx.lineTo(x + 3.5, y + 3.5); ctx.moveTo(x + 3.5, y - 3.5); ctx.lineTo(x - 3.5, y + 3.5); ctx.stroke();
    }
    // aircraft
    for (const pl of P.planes) {
      if (!alive(pl, t)) continue;
      const c = pl.cur;
      sample(pl.tr, t, c);
      const x = SX(c.x, s), y = SY(c.y, s);
      if (x < -30 || y < -30 || x > P.w + 30 || y > P.h + 30) continue;
      const isSel = P.sel === pl.id, isHov = P.hover === pl.id;
      const dim = P.sel != null && !isSel;
      const color = tcol(pl.team);
      ctx.globalAlpha = dim ? 0.55 : 1;
      chevron(ctx, x, y, c.h, isSel ? sz * 1.3 : sz, color, isSel ? 16 : glow, isSel);
      ctx.globalAlpha = 1;
      if (isSel || isHov) {
        ctx.beginPath(); ctx.arc(x, y, sz + (isSel ? 8 : 6), 0, 6.2832);
        ctx.strokeStyle = rgba(isSel ? '#ffffff' : color, isSel ? 0.85 : 0.7); ctx.lineWidth = isSel ? 1.5 : 1.2; ctx.stroke();
      }
      if (z >= 2.4 || isSel || isHov) {
        ctx.font = '500 10px ' + C.mono; ctx.textAlign = 'left';
        haloText(ctx, '#' + pl.id + (isSel ? '  ' + (c.z / 1000).toFixed(1) + ' km' : ''), x + sz + 4, y - sz * 0.4, isSel ? C.text : C.muted, rgba(C.mapBg2, 0.9));
      }
      P.screen.push({ pl, x, y, r: sz + 6 });
    }
  }
  function drawScaleBar(ctx, s) {
    const C = R.C;
    const opts = [1000, 2000, 5000, 10000, 20000, 50000];
    let L = opts[0];
    for (const o of opts) { L = o; if (o * s >= 70) break; }
    const px = L * s, x1 = P.w - 14, x0 = x1 - px, y = P.h - 34;
    ctx.save();
    ctx.strokeStyle = C.muted; ctx.lineWidth = 1.5; ctx.lineCap = 'butt';
    ctx.beginPath(); ctx.moveTo(x0, y - 4); ctx.lineTo(x0, y); ctx.lineTo(x1, y); ctx.lineTo(x1, y - 4); ctx.stroke();
    ctx.font = '500 10px ' + C.mono; ctx.textAlign = 'right';
    haloText(ctx, (L / 1000) + ' km', x1, y - 8, C.muted, rgba(C.mapBg2, 0.9));
    ctx.restore();
  }
  function drawMap() {
    const ctx = mapCtx;
    if (!ctx || !P.data) return;
    ctx.setTransform(P.dpr, 0, 0, P.dpr, 0, 0);
    ctx.clearRect(0, 0, P.w, P.h);
    const s = scale(), t = P.t;
    drawGrid(ctx, s);
    drawSpawn(ctx, s);
    drawPlanes(ctx, s, t);
    drawEffects(ctx, s, t);
    drawScaleBar(ctx, s);
  }

  // ------------------------------------------------------------------ HUD + tooltip
  let hudSig = '';
  function updateHud() {
    const t = P.t;
    const a = P.alive || [0, 0];
    const sig = fmt.mmss(t, true) + '|' + a[0] + '|' + a[1];
    if (sig === hudSig) return;
    hudSig = sig;
    $('#hudTime').innerHTML = '<div class="hud-time">' + fmt.mmss(t, true) + '<small>/ ' + fmt.mmss(P.tEnd) + '</small></div>' +
      '<div class="hud-teams"><span class="team-pill"><i style="background:' + R.C.team0 + '"></i>蓝 <b>' + a[0] + '</b><span style="color:var(--faint)">/' + P.teamN[0] + '</span></span>' +
      '<span class="team-pill"><i style="background:' + R.C.team1 + '"></i>橙 <b>' + a[1] + '</b><span style="color:var(--faint)">/' + P.teamN[1] + '</span></span></div>';
    $('#tpNow').textContent = fmt.mmss(t, true);
  }
  function buildLegend() {
    $('#hudLegend').innerHTML =
      '<span class="lgi"><svg viewBox="0 0 12 12"><path d="M6 1l3.6 9.2L6 8 2.4 10.2z" fill="currentColor"/></svg>飞机</span>' +
      '<span class="lgi"><svg viewBox="0 0 12 12"><path d="M1.5 10.5L9 3" stroke="' + 'var(--missile)' + '" stroke-width="2.2" stroke-linecap="round"/></svg>导弹</span>' +
      '<span class="lgi"><svg viewBox="0 0 12 12"><circle cx="3" cy="6" r="1.2" fill="currentColor"/><circle cx="7" cy="4" r="1" fill="currentColor"/><circle cx="9" cy="8" r="1" fill="currentColor"/></svg>干扰弹</span>' +
      '<span class="lgi"><svg viewBox="0 0 12 12"><path d="M3 3l6 6M9 3l-6 6" stroke="var(--kill)" stroke-width="1.6" stroke-linecap="round"/></svg>阵亡</span>';
  }
  function hitTest(mx, my) {
    let best = null, bd = 1e9;
    for (const o of P.screen) { const d = Math.hypot(o.x - mx, o.y - my); if (d < o.r + 4 && d < bd) { bd = d; best = { kind: 'plane', o }; } }
    if (best) return best;
    for (const o of P.screenM) { const d = Math.hypot(o.x - mx, o.y - my); if (d < 10 && d < bd) { bd = d; best = { kind: 'missile', o }; } }
    return best;
  }
  function showTip(hit, mx, my) {
    const tip = $('#mapTip');
    if (!hit) { tip.hidden = true; return; }
    let html;
    if (hit.kind === 'plane') {
      const pl = hit.o.pl, c = pl.cur;
      const sp = Math.hypot(c.vx || 0, c.vy || 0, c.vz || 0);
      html = '<div class="hd"><i style="background:' + tcol(pl.team) + '"></i>' + esc(pl.aircraft) + ' <span style="color:var(--muted);font-family:var(--mono);font-weight:400">#' + pl.id + '</span></div>' +
        '<div class="row"><span>' + TEAM_ZH[pl.team] + ' · ' + esc(pl.archetype || '—') + ' · ' + esc(pl.skill || '—') + '</span></div>' +
        '<div class="row"><span>阶段</span><b>' + esc(phaseZh(P.phases[c.p])) + '</b></div>' +
        '<div class="row"><span>高度</span><b>' + fmt.int(c.z) + ' m</b></div>' +
        '<div class="row"><span>速度</span><b>' + fmt.int(sp * 3.6) + ' km/h</b></div>' +
        '<div class="row"><span>航向</span><b>' + Math.round(((c.h % 360) + 360) % 360) + '°</b></div>' +
        '<div class="row"><span>导弹 / 干扰弹</span><b>' + c.m + (pl.missiles0 != null ? '/' + pl.missiles0 : '') + ' · ' + c.c + '</b></div>' +
        (pl.missile ? '<div class="row"><span>挂载</span><b>' + esc(pl.missile) + '</b></div>' : '');
    } else {
      const m = hit.o.m, c = m.cur, L = P.launchOf.get(m.uid);
      html = '<div class="hd"><i style="background:var(--missile)"></i>导弹 #' + m.uid + (L ? ' <span style="color:var(--muted);font-weight:400">' + esc(L.missile) + '</span>' : '') + '</div>' +
        '<div class="row"><span>发射</span><b>' + nameHtml(m.owner).replace(/<[^>]+>/g, '') + '</b></div>' +
        '<div class="row"><span>目标</span><b>' + nameHtml(m.target).replace(/<[^>]+>/g, '') + '</b></div>' +
        '<div class="row"><span>飞行时间</span><b>' + (P.t - m.t0 + (L ? 0 : 0)).toFixed(1) + ' s</b></div>' +
        '<div class="row"><span>高度</span><b>' + fmt.int(c.z) + ' m</b></div>' +
        '<div class="row"><span>导引头 / 数据链</span><b>' + (c.s ? '开' : '关') + ' / ' + (c.d ? '通' : '断') + '</b></div>';
    }
    tip.innerHTML = html; tip.hidden = false;
    const W = P.w, tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = mx + 18, top = my + 14;
    if (left + tw > W - 6) left = mx - tw - 18;
    if (top + th > P.h - 6) top = my - th - 14;
    tip.style.left = Math.max(6, left) + 'px'; tip.style.top = Math.max(6, top) + 'px';
  }

  // ------------------------------------------------------------------ scrubber
  const SC = { top: 6, areaH: 22, tickTop: 32, tickBot: 44, axisY: 48 };
  function xOfT(t, w, pad) { return pad + (t / (P.tEnd || 1)) * (w - 2 * pad); }
  function tOfX(x, w, pad) { return clamp(((x - pad) / (w - 2 * pad)) * (P.tEnd || 1), 0, P.tEnd); }
  function buildScrubCache() {
    const dpr = P.sdpr, w = P.sw, h = P.sh, C = R.C;
    scrubCache = scrubCache || document.createElement('canvas');
    scrubCache.width = Math.round(w * dpr); scrubCache.height = Math.round(h * dpr);
    const g = scrubCache.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const pad = 8;
    // track
    g.fillStyle = C.surface2; g.strokeStyle = C.border;
    roundRect(g, pad - 2, SC.top - 2, w - 2 * pad + 4, SC.areaH + 4, 6); g.fill();
    // alive step areas
    [0, 1].forEach((team) => {
      const pts = P.aliveSteps[team];
      if (!pts) return;
      g.beginPath();
      pts.forEach((p, i) => { const x = xOfT(p[0], w, pad), y = SC.top + SC.areaH - p[1] * SC.areaH; if (i) g.lineTo(x, y); else g.moveTo(x, y); });
      g.strokeStyle = rgba(tcol(team), 0.85); g.lineWidth = 1.4; g.lineJoin = 'round'; g.stroke();
      g.lineTo(xOfT(P.tEnd, w, pad), SC.top + SC.areaH); g.lineTo(xOfT(0, w, pad), SC.top + SC.areaH); g.closePath();
      g.fillStyle = rgba(tcol(team), 0.10); g.fill();
    });
    // time axis
    g.strokeStyle = C.gridStrong; g.lineWidth = 1;
    g.beginPath(); g.moveTo(pad, SC.axisY + 0.5); g.lineTo(w - pad, SC.axisY + 0.5); g.stroke();
    const dur = P.tEnd || 1;
    const steps = [10, 15, 30, 60, 120, 300, 600, 1200];
    let st = steps[steps.length - 1];
    for (const c of steps) if ((c / dur) * (w - 2 * pad) >= 62) { st = c; break; }
    g.font = '500 10px ' + C.mono; g.textAlign = 'center'; g.fillStyle = C.faint;
    for (let t = 0; t <= dur + 1e-6; t += st) {
      const x = xOfT(t, w, pad);
      g.strokeStyle = C.gridStrong; g.beginPath(); g.moveTo(x + 0.5, SC.axisY - 3); g.lineTo(x + 0.5, SC.axisY + 3); g.stroke();
      g.textAlign = t === 0 ? 'left' : 'center';
      g.fillText(fmt.mmss(t), t === 0 ? x - 2 : x, h - 1);
    }
    // event ticks
    g.lineCap = 'round';
    P.byKind.rwr.forEach((e) => {
      if (e.warning !== 'missile') return;
      g.fillStyle = rgba(C.warn, 0.55); g.beginPath(); g.arc(xOfT(e.t, w, pad), SC.tickBot + 1.5, 1.6, 0, 6.2832); g.fill();
    });
    P.byKind.launch.forEach((e) => {
      const x = xOfT(e.t, w, pad);
      g.strokeStyle = rgba(C.missile, 0.85); g.lineWidth = 1.6; g.beginPath(); g.moveTo(x, SC.tickBot - 7); g.lineTo(x, SC.tickBot); g.stroke();
    });
    P.byKind.kill.forEach((e) => {
      const x = xOfT(e.t, w, pad);
      g.strokeStyle = C.kill; g.lineWidth = 2.2; g.beginPath(); g.moveTo(x, SC.tickTop - 2); g.lineTo(x, SC.tickBot); g.stroke();
      g.fillStyle = C.kill; g.beginPath(); g.arc(x, SC.tickTop - 3, 2.8, 0, 6.2832); g.fill();
    });
  }
  function roundRect(g, x, y, w, h, r) {
    g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
  }
  function drawScrub() {
    if (!P.data || !scrubCtx) return;
    if (P.scrubDirty || !scrubCache) { buildScrubCache(); P.scrubDirty = false; }
    const ctx = scrubCtx, w = P.sw, h = P.sh, C = R.C, pad = 8;
    ctx.setTransform(P.sdpr, 0, 0, P.sdpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(scrubCache, 0, 0, w, h);
    const x = xOfT(P.t, w, pad);
    ctx.fillStyle = rgba(C.accent, 0.12); ctx.fillRect(pad, SC.top, x - pad, SC.areaH);
    if (P.scrubHover != null) {
      const hx = xOfT(P.scrubHover, w, pad);
      ctx.strokeStyle = rgba(C.text, 0.35); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(hx + 0.5, SC.top - 2); ctx.lineTo(hx + 0.5, SC.axisY); ctx.stroke();
    }
    ctx.strokeStyle = C.text; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(x, SC.top - 3); ctx.lineTo(x, SC.axisY + 1); ctx.stroke();
    ctx.fillStyle = C.text; ctx.beginPath(); ctx.moveTo(x - 5, 0); ctx.lineTo(x + 5, 0); ctx.lineTo(x, 6); ctx.closePath(); ctx.fill();
  }
  function nearestEvent(t, pxTol) {
    const w = P.sw, pad = 8, tol = (pxTol / (w - 2 * pad)) * P.tEnd;
    let best = null, bd = tol;
    for (const k of ['kill', 'launch']) {
      for (const e of P.byKind[k]) { const d = Math.abs(e.t - t); if (d < bd) { bd = d; best = e; } }
    }
    return best;
  }
  function bindScrub() {
    const wrap = $('#scrubWrap'), tip = $('#scrubTip');
    let dragging = false;
    const pos = (e) => { const r = wrap.getBoundingClientRect(); return e.clientX - r.left; };
    wrap.addEventListener('pointerdown', (e) => {
      if (!P.data) return;
      dragging = true; wrap.setPointerCapture(e.pointerId);
      const t = tOfX(pos(e), P.sw, 8), ev = nearestEvent(t, 6);
      seek(ev ? ev.t : t);
    });
    wrap.addEventListener('pointermove', (e) => {
      if (!P.data) return;
      const x = pos(e), t = tOfX(x, P.sw, 8);
      P.scrubHover = t; P.dirty = true;
      if (dragging) seek(t);
      const ev = nearestEvent(t, 6);
      tip.hidden = false;
      tip.innerHTML = fmt.mmss(t) + (ev ? ' · <span style="color:' + (ev.kind === 'kill' ? 'var(--kill)' : 'var(--missile)') + '">' + KINDS[ev.kind] + '</span>' : '');
      tip.style.left = clamp(x + 10, 0, P.sw - tip.offsetWidth) + 'px'; tip.style.top = '-30px';
    });
    const end = () => { dragging = false; };
    wrap.addEventListener('pointerup', end); wrap.addEventListener('pointercancel', end);
    wrap.addEventListener('pointerleave', () => { P.scrubHover = null; tip.hidden = true; P.dirty = true; });
  }

  // ------------------------------------------------------------------ altitude chart
  const AL = { l: 34, r: 8, t: 8, b: 16 };
  function buildAltCache() {
    const dpr = P.adpr, w = P.aw, h = P.ah, C = R.C;
    altCache = altCache || document.createElement('canvas');
    altCache.width = Math.round(w * dpr); altCache.height = Math.round(h * dpr);
    const g = altCache.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const X = (t) => AL.l + (t / (P.tEnd || 1)) * (w - AL.l - AL.r), Y = (z) => h - AL.b - (z / P.altMax) * (h - AL.t - AL.b);
    g.font = '500 10px ' + C.mono;
    const stepKm = P.altMax <= 8000 ? 2 : P.altMax <= 16000 ? 4 : 5;
    for (let z = 0; z <= P.altMax + 1; z += stepKm * 1000) {
      const y = Math.round(Y(z)) + 0.5;
      g.strokeStyle = z === 0 ? C.gridStrong : C.grid; g.lineWidth = 1; g.beginPath(); g.moveTo(AL.l, y); g.lineTo(w - AL.r, y); g.stroke();
      g.fillStyle = C.faint; g.textAlign = 'right'; g.fillText(String(z / 1000), AL.l - 6, y + 3.5);
    }
    const dur = P.tEnd || 1, steps = [30, 60, 120, 300, 600, 1200];
    let st = steps[steps.length - 1];
    for (const c of steps) if ((c / dur) * (w - AL.l - AL.r) >= 54) { st = c; break; }
    g.textAlign = 'center';
    for (let t = 0; t <= dur + 1e-6; t += st) { g.fillStyle = C.faint; g.fillText(fmt.mmss(t), clamp(X(t), AL.l + 8, w - 12), h - 3); }
    const drawLine = (pl, color, width) => {
      const tr = pl.tr;
      g.beginPath();
      for (let i = 0; i < tr.t.length; i++) { const x = X(tr.t[i]), y = Y(tr.z[i]); if (i) g.lineTo(x, y); else g.moveTo(x, y); }
      g.strokeStyle = color; g.lineWidth = width; g.lineJoin = 'round'; g.stroke();
    };
    const sel = P.sel;
    for (const pl of P.planes) { if (pl.id === sel) continue; drawLine(pl, rgba(tcol(pl.team), sel != null ? 0.2 : 0.5), 1); }
    if (sel != null && P.byId.has(sel)) { const pl = P.byId.get(sel); drawLine(pl, tcol(pl.team), 2.2); }
    // death marks
    for (const pl of P.planes) {
      if (pl.deathT == null) continue;
      const x = X(pl.deathT), tr = pl.tr, y = Y(tr.z[tr.z.length - 1]);
      g.strokeStyle = rgba(pl.id === sel || sel == null ? C.kill : C.faint, 0.8); g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(x - 2.5, y - 2.5); g.lineTo(x + 2.5, y + 2.5); g.moveTo(x + 2.5, y - 2.5); g.lineTo(x - 2.5, y + 2.5); g.stroke();
    }
    altCache.$X = X; altCache.$Y = Y;
  }
  function drawAlt() {
    if (!P.data || !altCtx) return;
    if (P.altDirty || !altCache) { buildAltCache(); P.altDirty = false; }
    const ctx = altCtx, w = P.aw, h = P.ah, C = R.C;
    ctx.setTransform(P.adpr, 0, 0, P.adpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(altCache, 0, 0, w, h);
    const X = altCache.$X, Y = altCache.$Y, x = X(P.t);
    ctx.strokeStyle = rgba(C.text, 0.8); ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(x + 0.5, AL.t - 2); ctx.lineTo(x + 0.5, h - AL.b); ctx.stroke();
    for (const pl of P.planes) {
      if (!alive(pl, P.t)) continue;
      sample(pl.tr, P.t, pl.cur);
      ctx.fillStyle = tcol(pl.team); ctx.globalAlpha = P.sel != null && P.sel !== pl.id ? 0.4 : 1;
      ctx.beginPath(); ctx.arc(x, Y(pl.cur.z), P.sel === pl.id ? 3.4 : 2.2, 0, 6.2832); ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (P.altHover != null) { const hx = X(P.altHover); ctx.strokeStyle = rgba(C.text, 0.25); ctx.beginPath(); ctx.moveTo(hx + 0.5, AL.t); ctx.lineTo(hx + 0.5, h - AL.b); ctx.stroke(); }
    const sp = $('#altSub');
    const selPl = P.sel != null ? P.byId.get(P.sel) : null;
    const txt = selPl ? planeName(selPl) + (alive(selPl, P.t) ? ' · ' + (selPl.cur.z / 1000).toFixed(2) + ' km' : ' · 已阵亡') : '高度 (km)';
    if (sp.textContent !== txt) sp.textContent = txt;
  }
  function bindAlt() {
    const wrap = $('#altWrap'), tip = $('#altTip');
    let dragging = false;
    const tAt = (e) => { const r = wrap.getBoundingClientRect(); const x = e.clientX - r.left; return clamp(((x - AL.l) / (P.aw - AL.l - AL.r)) * P.tEnd, 0, P.tEnd); };
    wrap.addEventListener('pointerdown', (e) => { if (!P.data) return; dragging = true; wrap.setPointerCapture(e.pointerId); seek(tAt(e)); });
    wrap.addEventListener('pointermove', (e) => {
      if (!P.data) return;
      const t = tAt(e); P.altHover = t; P.dirty = true;
      if (dragging) seek(t);
      tip.hidden = false; tip.textContent = fmt.mmss(t);
      const r = wrap.getBoundingClientRect();
      tip.style.left = clamp(e.clientX - r.left + 10, 0, P.aw - 50) + 'px'; tip.style.top = '4px';
    });
    wrap.addEventListener('pointerup', () => { dragging = false; });
    wrap.addEventListener('pointerleave', () => { P.altHover = null; tip.hidden = true; P.dirty = true; });
  }

  // ------------------------------------------------------------------ map interaction
  function bindMap() {
    const stage = $('#mapStage');
    let down = null;
    const rel = (e) => { const r = stage.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    stage.addEventListener('pointerdown', (e) => {
      if (e.target.closest('.hud-tr') || !P.data) return;
      const [x, y] = rel(e);
      down = { x, y, cx: P.view.cx, cy: P.view.cy, moved: false, id: e.pointerId };
      stage.setPointerCapture(e.pointerId);
    });
    stage.addEventListener('pointermove', (e) => {
      if (!P.data) return;
      const [x, y] = rel(e);
      if (down) {
        const dx = x - down.x, dy = y - down.y;
        if (!down.moved && Math.hypot(dx, dy) > 4) { down.moved = true; stage.classList.add('dragging'); P.follow = false; syncTransport(); }
        if (down.moved) {
          const s = scale();
          P.view.cx = clamp(down.cx - dx / s, -P.half, P.half); P.view.cy = clamp(down.cy + dy / s, -P.half, P.half);
          P.view.tcx = P.view.cx; P.view.tcy = P.view.cy; P.dirty = true; updateHudView();
          $('#mapTip').hidden = true;
          return;
        }
      }
      P.mouse = [x, y]; P.dirty = true;
    });
    const up = (e) => {
      if (!down) return;
      const wasMove = down.moved;
      stage.classList.remove('dragging');
      const [x, y] = rel(e);
      down = null;
      if (wasMove) return;
      const hit = hitTest(x, y);
      if (hit && hit.kind === 'plane') selectPlane(hit.o.pl.id === P.sel ? null : hit.o.pl.id, true);
      else if (!hit) selectPlane(null);
    };
    stage.addEventListener('pointerup', up);
    stage.addEventListener('pointercancel', () => { down = null; stage.classList.remove('dragging'); });
    stage.addEventListener('pointerleave', () => { P.mouse = null; $('#mapTip').hidden = true; stage.classList.remove('hovering'); if (P.hover != null && !P.rowHover) { P.hover = null; } P.dirty = true; });
    stage.addEventListener('dblclick', (e) => { if (!e.target.closest('.hud-tr')) resetView(); });
    stage.addEventListener('wheel', (e) => {
      if (!P.data) return;
      e.preventDefault();
      const [x, y] = rel(e);
      const s0 = scale();
      const wx = P.view.cx + (x - P.w / 2) / s0, wy = P.view.cy - (y - P.h / 2) / s0;
      const f = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0016));
      const z = clamp(P.view.zoom * f, 1, 60);
      P.view.zoom = P.view.tz = z;
      const s1 = scale();
      if (P.follow) { P.view.tcx = P.view.cx; P.view.tcy = P.view.cy; }
      else {
        P.view.cx = clamp(wx - (x - P.w / 2) / s1, -P.half, P.half); P.view.cy = clamp(wy + (y - P.h / 2) / s1, -P.half, P.half);
        P.view.tcx = P.view.cx; P.view.tcy = P.view.cy;
      }
      P.dirty = true; updateHudView();
    }, { passive: false });
  }
  function updateHover() {
    // hover under the mouse (run after drawing so that screen positions are current)
    const stage = $('#mapStage');
    if (!P.mouse) return;
    const hit = hitTest(P.mouse[0], P.mouse[1]);
    const hovId = hit && hit.kind === 'plane' ? hit.o.pl.id : null;
    if (hovId !== P.hover && !P.rowHover) { P.hover = hovId; }
    stage.classList.toggle('hovering', !!hit);
    showTip(hit, P.mouse[0], P.mouse[1]);
  }

  // ------------------------------------------------------------------ frame loop
  function viewStep(dt) {
    const V = P.view;
    if (P.follow && P.sel != null) {
      const pl = P.byId.get(P.sel);
      if (pl && alive(pl, P.t)) { sample(pl.tr, P.t, pl.cur); V.tcx = pl.cur.x; V.tcy = pl.cur.y; }
      else { P.follow = false; syncTransport(); }
    }
    const k = 1 - Math.exp(-dt * 9);
    const dz = V.tz - V.zoom, dx = V.tcx - V.cx, dy = V.tcy - V.cy;
    if (P.follow || Math.abs(dz) > 1e-3) {
      V.zoom += dz * k;
      if (P.follow) { V.cx += dx * k; V.cy += dy * k; if (Math.abs(dx) + Math.abs(dy) > 5 || Math.abs(dz) > 1e-3) P.dirty = true; if (P.playing) P.dirty = true; }
      else if (Math.abs(dz) > 1e-3) P.dirty = true;
    }
    if (Math.abs(dz) <= 1e-3) V.zoom = V.tz;
  }
  function frame(ts) {
    requestAnimationFrame(frame);
    if (R.tab() !== 'replay' || !P.data) { P.last = ts; return; }
    const dt = Math.min(0.1, Math.max(0, (ts - P.last) / 1000));
    P.last = ts;
    if (P.playing) {
      P.t += dt * P.speed;
      if (P.t >= P.tEnd) { P.t = P.tEnd; P.playing = false; syncTransport(); }
      P.dirty = true;
    }
    viewStep(dt);
    if (P.dirty) { P.dirty = false; render(); }
  }
  function render() {
    const t0 = performance.now();
    updatePlaneList(P.t);
    const t1 = performance.now();
    drawMap(); updateHover();
    const t2 = performance.now();
    updateHud(); drawScrub(); drawAlt(); updateLog(P.t);
    const t3 = performance.now();
    P.prof = { list: t1 - t0, map: t2 - t1, rest: t3 - t2 };
    return P.prof;
  }

  // ------------------------------------------------------------------ keyboard
  function onKey(e) {
    if (R.tab() !== 'replay' || !P.data) return;
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select' || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    const big = e.shiftKey;
    let used = true;
    if (k === ' ' || k === 'Spacebar') togglePlay();
    else if (k === 'ArrowLeft') seek(P.t - (big ? 30 : 5));
    else if (k === 'ArrowRight') seek(P.t + (big ? 30 : 5));
    else if (k === ',' || k === '<') seek(P.t - 1);
    else if (k === '.' || k === '>') seek(P.t + 1);
    else if (k === 'ArrowUp' || k === ']') setSpeed(SPEEDS[Math.min(SPEEDS.length - 1, SPEEDS.indexOf(P.speed) + 1)]);
    else if (k === 'ArrowDown' || k === '[') setSpeed(SPEEDS[Math.max(0, SPEEDS.indexOf(P.speed) - 1)]);
    else if (/^[1-6]$/.test(k)) setSpeed(SPEEDS[+k - 1]);
    else if (k === 'Home') seek(0);
    else if (k === 'End') seek(P.tEnd);
    else if (k === 'k' || k === 'K') jumpKill(1);
    else if (k === 'j' || k === 'J') jumpKill(-1);
    else if (k === 'f' || k === 'F') { if (P.sel != null) { P.follow = !P.follow; if (P.follow && P.view.tz < 3) P.view.tz = 4; syncTransport(); P.dirty = true; } }
    else if (k === 't' || k === 'T') { P.trails = !P.trails; store.set('rpTrails', P.trails); syncTransport(); P.dirty = true; }
    else if (k === 'r' || k === 'R') resetView();
    else if (k === 'Escape') { if (!$('#kbdHelp').hidden) $('#kbdHelp').hidden = true; else selectPlane(null); }
    else if (k === '?') toggleHelp();
    else used = false;
    if (used) e.preventDefault();
  }
  function toggleHelp() {
    const h = $('#kbdHelp');
    if (!h.hidden) { h.hidden = true; return; }
    const rows = [['空格', '播放 / 暂停'], ['← →', '后退 / 前进 5 秒 (Shift: 30 秒)'], [', .', '后退 / 前进 1 秒'], ['↑ ↓  [ ]', '加速 / 减速'], ['1 – 6', '速度 1× … 32×'], ['J K', '上一次 / 下一次击杀'],
      ['Home End', '跳到开头 / 结尾'], ['F', '跟随所选飞机'], ['T', '显示 / 隐藏轨迹'], ['R', '重置视图'], ['Esc', '取消选择'], ['滚轮 / 拖动', '缩放 / 平移地图']];
    h.innerHTML = rows.map((r) => '<div>' + r[0].split(' ').filter(Boolean).map((k) => '<kbd>' + esc(k) + '</kbd>').join('') + ' ' + esc(r[1]) + '</div>').join('');
    h.hidden = false;
  }

  // ------------------------------------------------------------------ init
  RLD.modules.replay = {
    state: P,
    render,
    init() {
      P.speed = store.get('rpSpeed', 4);
      P.trails = store.get('rpTrails', true);
      P.hz = store.get('rpHz', 1);
      P.kinds = Object.assign({}, KIND_DEFAULT_ON, store.get('rpKinds', {}));
      mapCv = $('#mapCanvas'); mapCtx = mapCv.getContext('2d');
      scrubCv = $('#scrubCanvas'); scrubCtx = scrubCv.getContext('2d');
      altCv = $('#altCanvas'); altCtx = altCv.getContext('2d');
      buildLegend();
      $('#tpSpeed').innerHTML = '';
      SPEEDS.forEach((s) => $('#tpSpeed').appendChild(el('button', { dataset: { v: s }, text: s + '×', title: '速度 ' + s + '×', on: { click: () => setSpeed(s) } })));
      $('#tpPlay').addEventListener('click', () => togglePlay());
      $('#tpBack').addEventListener('click', () => seek(P.t - 5));
      $('#tpFwd').addEventListener('click', () => seek(P.t + 5));
      $('#tpPrevKill').addEventListener('click', () => jumpKill(-1));
      $('#tpNextKill').addEventListener('click', () => jumpKill(1));
      $('#tpTrails').addEventListener('click', () => { P.trails = !P.trails; store.set('rpTrails', P.trails); syncTransport(); P.dirty = true; });
      $('#tpFollow').addEventListener('click', () => { if (P.sel != null) { P.follow = !P.follow; if (P.follow && P.view.tz < 3) P.view.tz = 4; syncTransport(); P.dirty = true; } });
      $('#tpHelp').addEventListener('click', toggleHelp);
      $('#plAliveOnly').addEventListener('click', () => { P.aliveOnly = !P.aliveOnly; P.listDirty = true; syncTransport(); P.dirty = true; });
      $('#rpSelect').addEventListener('change', (e) => { P.playing = false; loadReplay(e.target.value); });
      $('#rpReload').addEventListener('click', () => loadList(true));
      const hz = $('#rpHz'); hz.value = String(P.hz);
      hz.addEventListener('change', () => { P.hz = parseFloat(hz.value); store.set('rpHz', P.hz); if (P.cur) loadReplay(P.cur, true); });
      $('#planeList').addEventListener('mouseenter', () => { P.rowHover = true; });
      $('#planeList').addEventListener('mouseleave', () => { P.rowHover = false; });
      bindMap(); bindScrub(); bindAlt();
      document.addEventListener('keydown', onKey);
      $('#transport').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) b.blur(); });
      const ro = new ResizeObserver(() => { if (R.tab() === 'replay') layoutAll(); });
      ro.observe($('#mapStage')); ro.observe($('#scrubWrap')); ro.observe($('#altWrap'));
      window.addEventListener('resize', () => { if (R.tab() === 'replay') layoutAll(); });
      R.on('theme', () => { hudSig = ''; P.scrubDirty = P.altDirty = P.dirty = true; buildLegend(); });
      R.on('tab', (name) => {
        if (name === 'replay') {
          if (!P.loaded) { P.loaded = true; loadList(false); }
          setTimeout(layoutAll, 0);
          P.dirty = true;
        } else { P.playing = false; syncTransport(); }
      });
      syncTransport();
      requestAnimationFrame(frame);
    },
  };
})();
