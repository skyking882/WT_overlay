# 沙盘 War Thunder 回放支持 — 技术规格

状态：待实现（2026-10-08）。实现者：读完本文件再动手；不清楚的地方按"约束"一节取最保守的做法，并在交付说明里列出。

## 目标

在 `python -m wt_overlay.sandbox` 的网页沙盘里加载 War Thunder 回放，两种模式：

1. **原样回放（WT 轨迹）**：完全按回放记录的位置 / 速度 / 事件播放，不经过我们的任何模型。用于复盘。
2. **接管推演（我们的动力学）**：从回放的某一时刻 `t_fork` 起，用户选定的飞机改由我们的飞行模型 + 脚本 / 人工 / 键盘驾驶控制，其余飞机继续钉在 WT 真实轨迹上；回放里真实发射过的导弹由我们的导弹模型重新飞。用于回答"那一刻换个打法会怎样"。

## 输入数据

- 回放 JSONL：`outputs/engagements/wt_real/*.jsonl`，由 `scripts/wt_replay_import.py` 从 inspector 导出生成（格式见该脚本 docstring）。沙盘自己的 `outputs/sandbox/run-*/replay.jsonl` 是同一行格式（header / frame / event / end），原样回放模式也要能放。
- 行格式（WT 导入）：
  - `header`：`planes[]`（`id, team, aircraft, name, archetype(=玩家名), skill("AI" = 无主 AI 单位), missile, missiles, chaff("?"), wt{player, samples, first_s, last_s, max_gap_s, ...}}`），`plane_columns = [id,x,y,z,vx,vy,vz,heading_deg,missiles,chaff,phase]`，`missile_columns = [uid,owner,target,x,y,z,vx,vy,vz,heading_deg,age_s,seeker,datalink]`，`map_half_m`，`frame_dt_s=0.25`，`time_limit_s`，`source{kind:"wt_replay", ...}`。
  - `frame`：`{"type":"frame","t":..,"planes":[row...],"missiles":[row...]}`，约 4 Hz，只含当时存在的单位。
  - `event`：`launch{uid,shooter,target,missile,...}`、`missile_end{uid,outcome,...}`、`kill`、`death{plane,cause,...}`、`damage`、`chaff`、`flare`、`end`。
- 导弹 id 可能带 `_default` 后缀（如 `us_aim_120d_default`）：去掉后缀再查 `default_library().info`。
- 单个文件 4–10 MB、约 2400 帧、最多约 60 个单位，可整体读入内存。

## 一、原样回放模式

### 服务端（`wt_overlay/sandbox.py`）

- 新增 `ReplayTrack`（可放在新文件 `wt_overlay/sandbox_replay.py`）：读 JSONL，按 plane id 建时间序列（t, pos, vel, heading, missiles, chaff）；导弹按 uid 建序列；事件列表按时间排序。
  - `state_at(ident, t)`：用位置 + 速度做三次 Hermite 插值（不是线性），速度线性插值；超出该机首末样本或在 `max_gap` 大于 2 s 的空洞中返回 None（不显示）。
  - 姿态：俯仰由速度矢量求；滚转由航向变化率按协调转弯估算 `roll = atan(V·ω / g)`，限幅 ±85°，在 snapshot 里标 `attitude_source: "estimated"`。
- `SandboxSession` 新增状态 `playback`：
  - 命令 `load_replay{file}`（只允许列表里的文件名，不接受任意路径）、`play`、`pause`、`seek{t}`、`speed`（复用现有倍率）、`unload`。
  - 播放钟由墙钟 × 倍率推进，到尾自动暂停。
  - `_publish` 在 playback 下用 `ReplayTrack` 生成与现有 `analysis_snapshot` **同结构**的快照（planes / missiles / events / time_s），外加 `replay{file, duration_s, t, markers[]}`，`markers` 是发射 / 击杀 / 死亡事件的时间和简短文字（给时间轴用）。plane 额外字段：`player`（玩家名）、`ai`（bool）。
  - 死亡、离开回放的飞机：死亡时刻后 `alive=false`，保留残骸位置 10 s 后不再显示。
- 新接口 `GET /api/replays`：列出 `outputs/engagements/wt_real/*.jsonl` 和 `outputs/sandbox/run-*/replay.jsonl`，返回 `[{file, kind: "wt"|"sandbox", duration_s, players, units, mtime}]`，按 mtime 倒序；只读 header 和最后一行，别整文件解析。
- 安全：沿用现有 Host / Origin / token 校验；`file` 只能是列表里返回的相对名。

### 前端（`wt_overlay/sandbox_assets/`）

- 顶栏加模式切换：`沙盘布置` / `WT 回放`。回放模式下左栏换成：回放文件列表（显示日期、时长、人数），"我的飞机"下拉（按玩家名，默认上次选择，存 localStorage，取不到不报错），"显示 AI 单位"开关（默认关）。
- 地图下方加时间轴：可拖动 seek，带发射（小三角）/ 击杀（叉）标记，悬停显示文字；播放 / 暂停 / 倍率复用工具栏。
- 渲染复用现有 `draw()`：标签用玩家名（过长截断 14 字）+ 机型；"我的飞机"用白色描边强调；AI 单位半透明、小号。
- 事件栏显示回放事件（发射、击杀、死亡、导弹结束 outcome）。
- 右栏所选飞机信息照常；回放模式下隐藏所有编辑和命令控件。

## 二、接管推演模式

### 流程

1. 原样回放中暂停在 `t_fork`，点"从此刻接管"。弹出面板：
   - 每架**当时存活**的玩家飞机一行：控制方式 `WT 轨迹`（默认）/ `自主脚本` / `人工导航` / `亲自驾驶`；"我的飞机"默认 `亲自驾驶`。
   - 不能接管的飞机（机型没有飞行模型或装备数据）只能选 `WT 轨迹`，并说明原因。
   - 选项"包含 AI 单位"（默认关；开了也只能 `WT 轨迹`）。
2. 点"开始推演"：服务端建世界并跑（见下），前端进入现有的推演界面（暂停 / 倍率 / 命令 / 驾驶全部可用）。

### 建世界（服务端）

- 世界起点 `t0 = min(t_fork, 在 t_fork 时仍"有效"的导弹的最早发射时刻)`；`t0` 到 `t_fork` 之间所有飞机都钉在 WT 轨迹上（等于回放本身），只是让在途导弹由我们的模型从真实发射状态开始飞。
  - （实现后修订，2026-10-08）"有效" = 尚未过最近点（导入的 `missile_end.t_cpa`，缺失时用导弹结束时刻）且发射不早于 `t_fork - 60 s`（`REFLY_MAX_AGE_S`）。WT 里脱靶的导弹会继续飞约 2 分钟直到自毁，按"仍在飞"算会把起点拉回 100 s 以上；超出窗口的导弹记为 `replay_shot_skipped`。
  - （实现后修订）`t0`→接管这一段全速推进（不受倍率限制），所选飞机释放后的第一个 tick 自动暂停，用户在静止状态下接手。
- 每架入选飞机建 `PlaneSpec`：机型 = 回放 `aircraft`；位置 / 速度 = `t0` 时刻插值；导弹 = 该机在回放里发射过的、`modelled_missiles` 覆盖的第一种（去 `_default` 后缀），数量 = `t0` 时帧里的 `missiles` 列；箔条未知时不设（按种子抽样），并在 scenario 的 notes 里写明。
- 世界坐标：回放 ENU 原样使用，`map_half_m` 取 header 值；队伍 = 回放 team。每队上限从 16 提到 24（`MAX_TEAM`），超出时只取离"我的飞机"最近的 24 架并说明。
- **钉轨飞机**：新增 `TrackedAircraft(Aircraft)`（放 `sandbox_replay.py`），`step()` 不积分，而是把 `ReplayTrack` 在当前仿真时刻 `t0 + eng.time` 的 Hermite 插值状态通过 `Aircraft._commit` 写入（保证 `state_at` / ring buffer / `min_altitude_m` 照常工作），`normal` 用估算滚转构造，`aoa_deg`、`engine_percent` 取合理常数。
  - 轨迹结束（离开回放、死亡）后：如果回放里是被击落 / 坠毁，在该时刻按回放结果让它死亡（`eng` 里走正常的 death 事件，cause 带 `"replay"` 标记）；如果我们的导弹先打中它，以我们为准，此后不再钉轨。
  - 钉轨飞机的控制器：雷达 TWS（与现有 manual 模式一样 `Action(radar=RadarCommand(mode="tws"))`），不自行发射，不放箔条（回放里的箔条事件：按记录时刻调用 `Action(chaff=1)` 重放，箔条数不足时跳过）。
- **释放**：在 `t_fork`，被选为非 `WT 轨迹` 的飞机切换为普通物理积分（同一 `TrackedAircraft` 对象 `release()` 后走父类 `step()`，或换回 `Aircraft`——选不会丢掉 ring buffer 的方案），控制器换成 `SandboxPilot`（auto / manual / pilot），状态连续（位置、速度、法向不跳变）。
- **导弹重飞**：回放 `launch` 事件中，发射时刻 ∈ `[t0, ∞)` 且射手仍在钉轨的：在该时刻 `eng.fire(shooter, target)` 发射（临时把 `shooter.missile_id` 设成事件里的导弹 id，发射后恢复；`missiles` 计数照扣）。导弹不在库里（AIM-7、红外弹等）或目标不在世界里：跳过，记一条 `replay_shot_skipped` 事件写明原因。已释放飞机不再按回放发射（由脚本 / 人决定）。
- 回放里的 `kill` / `death` 只作用于仍钉轨的飞机（见上）；已释放飞机的生死完全由我们的模型决定。
- 对照：snapshot 里给已释放飞机附 `ghost`：该机在回放中 `[t_fork, t_fork+120 s]` 的真实轨迹点（每 1 s 一个），前端画成同队颜色的淡虚线，标"WT 实际"。

### 输出

- 与现有沙盘相同：`scenario.json`（含 `replay_source{file, sha256, t_fork, t0, released[], skipped_shots[]}`）、`replay.jsonl`、`result.json`。

## 约束

- 不改 `flight.py` / `engagement.py` 现有行为：新功能全部 opt-in，默认沙盘（不加载回放）的结果逐位不变。确需在 engagement 加钩子时，只加可选参数 / 新方法，并在交付说明里列出。
- 物理步长不变（`SUBSTEP_S = 1/48`），不为提速改步长。
- 回放数据里的推断字段（导弹目标 `target_basis`、outcome）原样展示其来源，不当成确定事实。
- 性能：30 架飞机 + 数枚导弹时记录实际倍率（`actual_speed`），在交付说明里给出这台机器上的数字和配置；不要为提速删模型。
- 不在 sharedhost 上跑任何东西；本地即可。

## 测试（`tests/test_sandbox_replay.py`，标准库 unittest，和 `tests/test_sandbox.py` 同风格）

1. `ReplayTrack` 在样本点上精确等于记录值；样本之间插值连续；空洞 / 首末之外返回 None。
2. 原样回放：用一个小型合成 JSONL（测试里生成，2 架飞机 + 1 枚导弹 + 1 个击杀事件），`load_replay` → `seek` → 快照里位置等于插值、击杀后 `alive=false`。
3. 钉轨：全部 `WT 轨迹` 时，跑到 `t_fork+10 s`，每架飞机位置与回放插值误差 < 1 m。
4. 释放：一架飞机在 `t_fork` 释放后，`t_fork` 前后一个 tick 的速度变化 < 5 m/s（无跳变），之后由 `FlightCommand` / `KeyboardCommand` 控制。
5. 导弹重飞：合成回放里一枚已建模导弹在 `t0` 后发射，世界里在同一时刻出现一枚同射手同目标的导弹；未建模导弹产生 `replay_shot_skipped`。
6. 真实文件冒烟测试（文件不存在时 skip）：`outputs/engagements/wt_real/` 里任一文件能 `load_replay`，并在 `t_fork = 120 s`、释放一架飞机的情况下推进 10 s 不报错。
7. 现有 `tests/test_sandbox.py` 全部通过。

运行：`PYTHONPATH=. /Users/skyking/Documents/missle_sim/.venv/bin/python -m pytest -q tests/test_sandbox.py tests/test_sandbox_replay.py`（本仓库 `.venv` 没装 pytest）。

## 交付说明需包含

改了哪些文件、新增的 engagement 钩子（如有）、测试结果原文、30 架规模的实际倍率、所有偏离本规格的地方及原因。
