/* WT 控制台 — 训练指标按局类型拆分 (vs_script / self_play / history) from metrics.jsonl records, older records included.
   Pure functions: the page loads this after app.js (window.RLD.metrics); the tests load it with node (module.exports). */
(function (root) {
  'use strict';
  const KIND_SCRIPT = 'vs_script';
  const KIND_SELF = 'self_play';
  const KIND_HIST = 'history';        // the current policy against frozen past policies (league opponents)
  const OUTCOMES = ['win', 'loss', 'trade', 'none'];
  const isNum = (v) => typeof v === 'number' && isFinite(v);
  const num = (v) => (isNum(v) ? v : null);
  const byKindOf = (r) => (r && r.outcomes_by_kind && typeof r.outcomes_by_kind === 'object' ? r.outcomes_by_kind : null);

  /** kinds a record has data for. A record without outcomes_by_kind is from before self-play: all vs-script. */
  function kindsOf(r) {
    const bk = byKindOf(r);
    return bk ? Object.keys(bk) : [KIND_SCRIPT];
  }

  /** mean episode return of one kind, or null. episode_return_by_kind when the record has it; in older records the
   *  mixed episode_return_mean only when that kind was the only one in the round. */
  function kindReturn(r, kind) {
    if (!r) return null;
    const erk = r.episode_return_by_kind;
    if (erk && typeof erk === 'object') return num(erk[kind]);
    const kinds = kindsOf(r);
    return kinds.length === 1 && kinds[0] === kind ? num(r.episode_return_mean) : null;
  }

  /** the counts of one kind: outcomes_by_kind[kind]; before outcomes_by_kind the top-level (all vs-script) keys */
  function rawCounts(r, kind) {
    const bk = byKindOf(r);
    if (bk) return bk[kind] || null;
    if (kind !== KIND_SCRIPT || !r.outcomes || typeof r.outcomes !== 'object') return null;
    return Object.assign({}, r.outcomes, { win_rate: r.win_rate, exchange: r.exchange });
  }

  /**
   * Per-kind stats of one record, or null when the record has nothing for that kind:
   * win / loss / trade / none / kills / deaths, n (agent-episodes with an outcome), episodes, decisions, win_rate,
   * exchange, trade_rate, timeouts, timeout_rate, none_timeout / none_other (null where not split: older records),
   * split, ret (mean episode return). Cached on the record.
   */
  function kindStats(r, kind) {
    if (!r) return null;
    const cache = r.$kinds || (r.$kinds = {});
    if (Object.prototype.hasOwnProperty.call(cache, kind)) return cache[kind];
    const o = rawCounts(r, kind);
    const ret = kindReturn(r, kind);
    let s = null;
    if (o || ret != null) {
      const c = o || {};
      s = { ret };
      for (const k of OUTCOMES.concat(['kills', 'deaths'])) s[k] = num(c[k]);
      const have = OUTCOMES.every((k) => s[k] != null);
      s.n = have ? OUTCOMES.reduce((a, k) => a + s[k], 0) : null;
      s.episodes = num(c.episodes) != null ? c.episodes : s.n;
      if (s.episodes == null && kindsOf(r).length === 1 && kindsOf(r)[0] === kind) s.episodes = num(r.episodes_finished);
      s.decisions = num(c.decisions);
      s.win_rate = num(c.win_rate) != null ? c.win_rate : (s.n ? s.win / s.n : null);
      s.exchange = num(c.exchange) != null ? c.exchange : (s.deaths ? s.kills / s.deaths : null);
      s.trade_rate = num(c.trade_rate) != null ? c.trade_rate : (s.n ? s.trade / s.n : null);
      s.timeouts = num(c.timeouts);
      s.timeout_rate = num(c.timeout_rate);
      s.none_timeout = num(c.none_timeout);
      s.none_other = num(c.none_other);
      s.split = s.none_timeout != null && s.none_other != null;
    }
    cache[kind] = s;
    return s;
  }
  /** share of agent-episodes of a kind with an outcome: win, loss, trade, none_timeout, none_other; 'none' only for
   *  records where none is not split (older runs) */
  function outcomeShare(r, kind, key) {
    const s = kindStats(r, kind);
    if (!s || !s.n) return null;
    if (key === 'none_timeout' || key === 'none_other') return s.split ? s[key] / s.n : null;
    if (key === 'none') return s.split ? null : s.none / s.n;
    return s[key] == null ? null : s[key] / s.n;
  }

  /** the last `rounds` records of a kind pooled (sum of counts, rates from the sums; return weighted by episodes) */
  function pooled(records, kind, rounds) {
    const rs = (records || []).slice(-Math.max(1, rounds || 10));
    const t = { rounds: 0, episodes: 0, n: 0, win: 0, loss: 0, trade: 0, none: 0, kills: 0, deaths: 0,
      retSum: 0, retN: 0, timeouts: 0, timeoutN: 0, noneT: 0, noneO: 0, splitN: 0, unsplit: 0 };
    for (const r of rs) {
      const s = kindStats(r, kind);
      if (!s) continue;
      t.rounds += 1;
      if (s.n != null) {
        t.n += s.n;
        for (const k of OUTCOMES) t[k] += s[k];
        t.kills += s.kills || 0; t.deaths += s.deaths || 0;
        if (s.split) { t.noneT += s.none_timeout; t.noneO += s.none_other; t.splitN += s.n; } else t.unsplit += s.n;
      }
      if (s.episodes != null) t.episodes += s.episodes;
      if (s.ret != null && s.episodes) { t.retSum += s.ret * s.episodes; t.retN += s.episodes; }
      if (s.timeouts != null && s.episodes) { t.timeouts += s.timeouts; t.timeoutN += s.episodes; }
    }
    if (!t.rounds) return null;
    return {
      rounds: t.rounds, episodes: t.episodes, n: t.n,
      win_rate: t.n ? t.win / t.n : null,
      exchange: t.deaths ? t.kills / t.deaths : null,
      kills: t.kills, deaths: t.deaths,
      trade_rate: t.n ? t.trade / t.n : null,
      ret: t.retN ? t.retSum / t.retN : null,
      timeout_rate: t.timeoutN ? t.timeouts / t.timeoutN : null,
      none_rate: t.n ? t.none / t.n : null,
      none_timeout_rate: t.splitN && !t.unsplit ? t.noneT / t.splitN : null,
      none_other_rate: t.splitN && !t.unsplit ? t.noneO / t.splitN : null,
    };
  }

  /** value RMS error in reward units: value_rmse, or sqrt(value_loss) * value_scale for records written before it */
  function valueRmse(r) {
    if (!r) return null;
    if (isNum(r.value_rmse)) return r.value_rmse;
    return isNum(r.value_loss) && r.value_loss >= 0 && isNum(r.value_scale) ? Math.sqrt(r.value_loss) * r.value_scale : null;
  }

  /** whether any record has data for the kind */
  function hasKind(records, kind) {
    return (records || []).some((r) => kindStats(r, kind) != null);
  }

  /** [key, value] of the scalar sampler statistics a record passes through (sampler_stats; dicts flattened a.b) */
  function samplerExtras(r) {
    const out = [];
    const st = r && r.sampler_stats;
    if (!st || typeof st !== 'object') return out;
    for (const k of Object.keys(st).sort()) {
      const v = st[k];
      if (isNum(v) || typeof v === 'boolean') out.push([k, v]);
      else if (v && typeof v === 'object') for (const j of Object.keys(v).sort()) if (isNum(v[j])) out.push([k + '.' + j, v[j]]);
    }
    return out;
  }

  /** ppo.head_kl entry of a record for one head: kl (round mean), coef (used this round), coef_next, target; or null */
  function headKl(r, head, key) {
    const d = r && r.head_kl && r.head_kl[head];
    return d ? num(d[key]) : null;
  }

  const api = { KIND_SCRIPT, KIND_SELF, KIND_HIST, headKl, kindsOf, kindReturn, kindStats, outcomeShare, pooled, valueRmse, hasKind, samplerExtras };
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root && root.RLD) root.RLD.metrics = api;
})(typeof window !== 'undefined' ? window : null);
