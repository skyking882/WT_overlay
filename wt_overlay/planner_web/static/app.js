"use strict";

const $ = (id) => document.getElementById(id);
const state = { aircraft: [], byLabel: new Map(), byId: new Map(), radar: [], busy: false };
const SVG = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

function showAlert(message) {
  const box = $("alert");
  box.textContent = message || "";
  box.classList.toggle("hidden", !message);
}

function remember(values) {
  try { localStorage.setItem("wt-opening", JSON.stringify(values)); } catch (e) { /* private window */ }
}

function recall() {
  try { return JSON.parse(localStorage.getItem("wt-opening") || "{}") || {}; } catch (e) { return {}; }
}

// ISA sound speed, as the server's wt_overlay.fm.atmosphere (for the TAS hint only).
function soundSpeed(h) {
  const t = h < 11000 ? 288.15 - 0.0065 * h : 216.65;
  return Math.sqrt(1.4 * 287.05287 * t);
}

function label(a) {
  return a.name_en && a.name_en !== a.name ? `${a.name} · ${a.name_en}` : a.name;
}

function findAircraft(text) {
  const t = (text || "").trim();
  return state.byLabel.get(t) || state.byId.get(t) || state.aircraft.find((a) => a.name === t || a.name_en === t);
}

function fillMissiles(aircraft, wanted) {
  const select = $("missile");
  select.replaceChildren();
  const own = aircraft ? aircraft.missiles : [];
  if (own.length) {
    const group = el("optgroup", { label: "本机可挂的雷达弹" });
    own.forEach((m) => group.append(el("option", { value: m.id, "data-count": m.count }, `${m.name}（最多 ${m.count} 枚）`)));
    select.append(group);
  }
  const others = el("optgroup", { label: own.length ? "其他雷达弹" : "雷达弹" });
  state.radar.filter((m) => !own.some((o) => o.id === m.id)).forEach((m) => others.append(el("option", { value: m.id }, m.name)));
  select.append(others);
  if (wanted && [...select.options].some((o) => o.value === wanted)) select.value = wanted;
  const note = $("missile-note");
  note.textContent = aircraft && !own.length ? "这架飞机挂的雷达弹在 missile_sim 里没有模型（例如 MICA），只能先用别的雷达弹代算。" : "";
  note.classList.toggle("hidden", !note.textContent);
}

function onAircraft(keepMissile) {
  const aircraft = findAircraft($("aircraft").value);
  $("aircraft-br").textContent = aircraft ? `BR ${aircraft.br}` : "";
  fillMissiles(aircraft, keepMissile);
  const option = $("missile").selectedOptions[0];
  if (!keepMissile && option && option.dataset.count) $("missiles").value = Math.min(6, Number(option.dataset.count));
}

function updateHelp() {
  const h = Number($("altitude").value), m = Number($("mach").value);
  $("state-help").textContent = Number.isFinite(h) && Number.isFinite(m)
    ? `真空速约 ${Math.round(m * soundSpeed(h) * 3.6)} km/h（标准大气）。起始状态就当作出生瞬间：敌机同时、同高、同速出生。` : "";
}

async function loadOptions() {
  const response = await fetch("/api/options");
  const data = await response.json();
  if (!data.ok) throw new Error(data.error ? data.error.message : "读取机型失败");
  state.aircraft = data.aircraft;
  state.radar = data.radar_missiles;
  const list = $("aircraft-list");
  data.aircraft.forEach((a) => {
    state.byLabel.set(label(a), a);
    state.byId.set(a.id, a);
    list.append(el("option", { value: label(a) }, `BR ${a.br}`));
  });
  const saved = recall();
  const enemy = { ...data.enemy, ...(saved.enemy || {}) };
  $("distance").value = enemy.distance_km;
  $("enemy-fuel").value = enemy.fuel_percent;
  $("enemy-climb").value = enemy.climb_deg;
  $("enemy-level").value = enemy.level_m;
  const enemyAircraft = state.byId.get(enemy.aircraft);
  $("enemy").value = enemyAircraft ? label(enemyAircraft) : enemy.aircraft;
  const own = state.byId.get(saved.aircraft) || state.byId.get("ef_2000_typhoon_aesa") || data.aircraft[0];
  $("aircraft").value = label(own);
  for (const [key, id] of [["altitude_m", "altitude"], ["mach", "mach"], ["fuel_percent", "fuel"], ["missiles", "missiles"]]) {
    if (saved[key] !== undefined) $(id).value = saved[key];
  }
  onAircraft(saved.missile);
  updateHelp();
}

function payload() {
  const aircraft = findAircraft($("aircraft").value);
  const enemy = findAircraft($("enemy").value);
  if (!aircraft) throw new Error("请从列表里选一个机型");
  if (!enemy) throw new Error("请从列表里选一个敌机机型");
  return {
    aircraft: aircraft.id, missile: $("missile").value,
    altitude_m: Number($("altitude").value), mach: Number($("mach").value),
    fuel_percent: Number($("fuel").value), missiles: Number($("missiles").value),
    enemy: {
      aircraft: enemy.id, distance_km: Number($("distance").value), fuel_percent: Number($("enemy-fuel").value),
      climb_deg: Number($("enemy-climb").value), level_m: Number($("enemy-level").value),
    },
  };
}

function setView(view) {
  for (const id of ["empty", "busy", "result"]) $(id).classList.toggle("hidden", id !== view);
}

async function submit(event) {
  event.preventDefault();
  if (state.busy) return;
  showAlert("");
  let body;
  try { body = payload(); } catch (e) { showAlert(e.message); return; }
  remember(body);
  state.busy = true;
  $("submit").disabled = true;
  setView("busy");
  const started = performance.now();
  const timer = setInterval(() => {
    $("busy-detail").textContent = `已用 ${((performance.now() - started) / 1000).toFixed(0)} 秒。搜索爬升角、发射时刻和发射俯仰角，每个候选都是一次完整的导弹仿真。`;
  }, 500);
  try {
    const response = await fetch("/api/plan", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error ? data.error.message : "计算失败");
    render(data);
  } catch (e) {
    setView("empty");
    showAlert(e.message);
  } finally {
    clearInterval(timer);
    state.busy = false;
    $("submit").disabled = false;
  }
}

function fmt(seconds) {
  return seconds.toFixed(seconds < 10 ? 1 : 0);
}

function render(data) {
  const plan = data.plan, inputs = data.inputs;
  if (!plan.available) {
    setView("empty");
    showAlert(`这个起始状态打不到：${plan.reason}`);
    return;
  }
  $("hit-time").textContent = plan.hit_s.toFixed(1);
  const facts = $("facts");
  facts.replaceChildren();
  for (const [k, v] of [
    ["发射距离", `${(plan.launch_range_m / 1000).toFixed(1)} km`],
    ["导弹飞行", `${plan.flight_s.toFixed(1)} s`],
    ["本机总质量", `${Math.round(inputs.mass_kg).toLocaleString()} kg`],
    ["计算用时", `${data.wall_s.toFixed(1)} s`],
  ]) {
    const box = el("div");
    box.append(el("dt", {}, k), el("dd", {}, v));
    facts.append(box);
  }
  $("window").textContent = `发射窗口 ${fmt(plan.window_s[0])}–${fmt(plan.window_s[1])} s：在这段时间里发射，命中最多晚 1 秒`;
  const steps = $("steps");
  steps.replaceChildren();
  data.steps.forEach((step, i) => {
    const kind = step.title === "发射" ? "fire" : step.title === "命中" ? "hit" : "";
    const li = el("li", kind ? { class: kind } : {});
    const t = el("div", { class: "t" }, fmt(step.t));
    t.append(el("small", {}, " s"));
    const body = el("div");
    body.append(el("strong", {}, step.title), el("p", {}, step.detail));
    li.append(t, body);
    steps.append(li);
  });
  drawChart(data);
  $("assumptions").textContent =
    `${inputs.aircraft_name} 带 ${inputs.missiles} 枚 ${inputs.missile_name}，内油 ${Math.round(inputs.fuel_kg)} kg；` +
    `敌机 ${inputs.enemy_name}（内油 ${Math.round(inputs.enemy_fuel_percent)}%）在 ${(inputs.distance_m / 1000).toFixed(0)} km 外同时出生，` +
    `${inputs.enemy_climb_deg.toFixed(0)}° 全加力爬到 ${inputs.enemy_level_m.toFixed(0)} m 后平飞，直飞朝你、不反应。` +
    "飞行按各自 FM、导弹按 missile_sim；数据链全程不断，结果是“对方完全不躲”时的最快命中。";
  setView("result");
}

function drawChart(data) {
  const box = $("chart");
  box.replaceChildren();
  const tr = data.trace, plan = data.plan;
  const W = 900, H = 340, L = 52, R = 18, T = 18, B = 40;
  const all = [...tr.own, ...tr.missile, ...tr.enemy];
  const xMax = Math.max(data.inputs.distance_m, ...all.map((p) => p[1])) / 1000;
  const yMax = Math.max(12, Math.ceil(Math.max(...all.map((p) => p[2])) / 1000) + 1);
  const sx = (x) => L + (x / 1000) / xMax * (W - L - R);
  const sy = (h) => H - B - (h / 1000) / yMax * (H - T - B);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "侧视图：横轴距离，纵轴高度" });
  for (let k = 0; k <= yMax; k += 2) {
    root.append(svg("line", { class: "grid", x1: L, x2: W - R, y1: sy(k * 1000), y2: sy(k * 1000) }));
    root.append(svg("text", { class: "axis-label", x: L - 8, y: sy(k * 1000) + 4, "text-anchor": "end" }, `${k} km`));
  }
  const step = xMax > 120 ? 20 : 10;
  for (let k = 0; k <= xMax + 1e-6; k += step) {
    root.append(svg("line", { class: "grid", x1: sx(k * 1000), x2: sx(k * 1000), y1: T, y2: H - B }));
    root.append(svg("text", { class: "axis-label", x: sx(k * 1000), y: H - B + 18, "text-anchor": "middle" }, `${k}`));
  }
  root.append(svg("text", { class: "axis-label", x: W - R, y: H - 6, "text-anchor": "end" }, "距你的出生点 / km"));
  const path = (points, cls) => svg("path", { class: cls, d: points.map((p, i) => `${i ? "L" : "M"}${sx(p[1]).toFixed(1)},${sy(p[2]).toFixed(1)}`).join("") });
  root.append(path(tr.enemy, "enemy"), path(tr.own, "own"), path(tr.missile, "missile"));
  const marker = (p, color, text, dy) => {
    root.append(svg("circle", { class: "marker", cx: sx(p[1]), cy: sy(p[2]), r: 5.5, fill: color }));
    root.append(svg("text", { class: "marker-label", x: sx(p[1]) + 8, y: sy(p[2]) + dy }, text));
  };
  marker(tr.own[0], "var(--cyan)", "出生", 16);
  marker(tr.launch, "var(--accent)", `发射 ${fmt(plan.launch_s)} s`, 18);
  marker(tr.end, "var(--gold)", `命中 ${fmt(plan.hit_s)} s`, -10);
  box.append(root);
}

$("form").addEventListener("submit", submit);
$("aircraft").addEventListener("change", () => onAircraft());
$("missile").addEventListener("change", () => {
  const option = $("missile").selectedOptions[0];
  if (option && option.dataset.count) $("missiles").value = Math.min(6, Number(option.dataset.count));
});
$("altitude").addEventListener("input", updateHelp);
$("mach").addEventListener("input", updateHelp);
loadOptions().catch((e) => showAlert(e.message));
