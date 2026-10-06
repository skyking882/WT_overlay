# 技术文档：missile_sim 导引头搜索扫描（SRC）与重新捕获

## 背景

本文档描述需要在 `/Users/skyking/Documents/missle_sim`（注意拼写 "missle"）中完成的改动。

该仓库用纯 Python 复现《战争雷霆》（War Thunder）里空空导弹的游戏内模型，参数来自游戏客户端文件。兄弟仓库 `/Users/skyking/Documents/WT_overlay` 用它做多机交战环境，训练游戏内辅助策略（`WT_overlay/docs/rl_design.md`）。

### 问题：丢锁后几乎总能重新锁上

现在的雷达导引头，在搜索状态（第一次开机或丢锁后）只要目标满足下面三条，当拍就算探测到：

- 在导弹机头 `lockAngleMax`（R-77-1 为 55°）以内；
- 在距离限制内；
- 通过距离门、多普勒门和杂波 notch 判定。

结果是：

- **甩掉锁以后，目标做什么都会被重新锁上。** 目标只有一直贴着 notch 侧飞才能逃掉。
- **惯导准不准没有区别。** 无论是 GNSS 的 AIM-120D（漂移 0），还是漂移 2 m/s 的弹，结果一样。

2026-10-06 的实测（`WT_overlay/scripts/bench_seeker_reacquire.py notch`）用下视发射，载机 8 km，目标 2–3.5 km、1000 km/h、最大 7 g。目标先侧飞 9 s 进入 notch 甩锁，再做五种动作之一。三种弹合计：

| 甩锁之后 | 局数 | 丢锁 | 重新锁上 | 命中 |
|---|---|---|---|---|
| 回到原航向直飞 | 34 | 34 | 32 | 32 |
| 转 45° | 34 | 34 | 32 | 32 |
| 爬升 20° | 34 | 34 | 32 | 32 |
| 俯冲 15° | 34 | 34 | 32 | 32 |
| 一直侧飞保持 notch | 34 | 34 | 0 | 0 |

`weave` 实验用 AIM-120C-5 和 R-77-1，规避在无规避命中前 12 s 开始；括号里是 6 s 开始。结果：

| 动作 | 命中率 |
|---|---|
| 不规避 | 84%（84%） |
| 持续往一侧三九 | 12%（50%） |
| 左右来回变线 | 72–78% |
| 上下 ±25° 来回变线 | 72–75% |

持续三九脱靶的原因绝大多数是 `angle_gate`。

交战环境的录像统计（WT_overlay，240 局 1v1）也是同一个结论：

| 数据链情况 | 命中率 |
|---|---|
| 开机前已靠惯导飞了 5–15 s | 24–33% |
| 数据链一直连到导引头接管 | 21% |

### 游戏里的行为（C 级，玩家报告，2026-10-06）

- 导引头的 SRC 是**扫描搜索**，不是 55° 内立刻锁定。
- 甩锁后继续**平飞直线**的飞机，常被按惯导飞来的弹近炸或命中。例子是 JAS 39 平飞被打。GNSS 弹（如 AIM-120D）概率最高。
- 所以玩家甩锁后要**变线**，左右或上下都可以。

### 影响

训练出来的策略学不到"甩锁后要变线"，也低估了 GNSS 弹的威胁。

## 仓库规则（必读：`missle_sim/CLAUDE.md`）

- 不要改控制节拍。surface runtime 是 1/48 s。
- 新机制一律**可选**。不启用时，所有现有结果**逐位不变**（浮点数精确相等）。
- 仿真内核 `src/aim120_model/` 只用 Python 标准库。
- 涉及游戏 blkx 字段含义的主张，要登记到 `docs/BLKX_EVIDENCE_LEDGER.md` 并标证据等级。**本任务需要登记**，见最后一节。
- 工作区有用户未提交的改动：不要提交，不要回退、格式化或"顺手清理"无关代码。
- 测试：`.venv/bin/python -m pytest`。改动前已有约 33 个失败。**改动前先记录失败测试的完整列表**，改完后失败列表必须完全一致，新测试全部通过。

## 现状（代码位置）

| 位置 | 内容 |
|---|---|
| `src/aim120_model/radar_seeker.py:83` | `RadarSeekerObserver` |
| `radar_seeker.py:286` | `update(...)`：无诱饵时的探测与跟踪 |
| `radar_seeker.py:318-321` | 离轴角按**导弹机体前向**计算：`forward = body_axes_for_state(missile_state).forward` |
| `radar_seeker.py:339-358` | `track_hold`（上次探测后 `prolongation_time_max_s` 以内）与 `searching = not track_hold` |
| `radar_seeker.py:358-365` | 搜索时用 `lock_range_m`、`lock_angle_max`；跟踪时用 `receiver_range_max_m`、`angle_max` |
| `radar_seeker.py:149` | `beam_half_rad`（`angleHalfSens`），目前只用于杂波（第 454 行）和诱饵分辨（第 548、556 行），**不参与对目标的探测** |
| `radar_seeker.py:459`、`:501` | 诱饵路径：`_reflector_check`、`_update_with_decoys`。目标 RCS 只在这条路径里用到 |
| `src/aim120_model/observation.py:302` | `SensorTrackProvider`：雷达、数据链、惯导三者组合 |
| `observation.py:462` | `update`：先雷达，再数据链，再惯导。雷达开着时惯导解标为 `INS_SEARCH`（显示 "INS+SRC"）；雷达一有跟踪就 `inertial.reset(radar_solution)` |
| `observation.py:60` | `InertialTrackPropagator`：按最后一次解的速度匀速外推，叠加 `inertialNavigationDriftSpeed` 漂移 |
| `src/aim120_model/surface_runtime.py:249` | 引信条件。`require_seeker_lock` 时要求导引头跟踪过一次 |
| `surface_runtime.py:336`、`:386` | `simulate_surface`、`_configure_runtime`：可选项都从这里接入 |
| `surface_runtime.py`（约 416 行起） | 逐步推进入口 `create_surface_missile`。WT_overlay 交战环境用的就是它，新选项**必须**也能从这里传入 |
| `src/aim120_model/public_api.py:320` | 公共 `simulate(...)` 及选项校验 |

导弹文件里有、内核没用到的字段：`rocket.guidance.lockTimeOut`（R-77-1：0.75 s）、`warmUpTime`（0.3 s）。

## 游戏数据（R-77-1 示例，`gamedata/weapons/rocketguns/su_r_77_1.blkx`，数值 A 级）

| 字段 | 值 |
|---|---|
| `radarSeeker.transmitter.antenna.angleHalfSens` / `receiver.antenna.angleHalfSens` | 15° |
| `...antenna.sideLobesSensitivity` | −30 dB |
| `radarSeeker.sideLobesAttenuation` | −30 |
| `radarSeeker.lockAngleMax` / `angleMax` | 55° / 55° |
| `radarSeeker.rateMax` / `angleGateRate` | 60°/s / 30°/s |
| `radarSeeker.receiver.rcs` / `range` / `rangeMax` | 2 m² / 16000 m / 25000 m |
| `radarSeeker.prolongationTimeMax` | 1 s |
| `distGate.distGateSearchRange` / `dopplerSpeedGate.dopplerSpeedGateSearchRange` | 5000 m / 300 m/s |
| `guidance.lockDistance` / `lockTimeOut` / `breakLockMaxTime` | 16000 m / 0.75 s / 100 s |
| `guidance.inertialGuidance.inertialNavigationDriftSpeed` | 2 m/s（AIM-120D 为 0） |

**没有的：导引头搜索扫描的图案、范围和周期。** 飞机雷达文件有 `scanPatterns`，导弹的 `radarSeeker` 没有。用户也不知道这些值。所以扫描参数一律是 D 级、可配置，默认值只是起点，要靠游戏内实验校准（见"校准"一节）。

## 需求

新增可选项 `seeker_search`（字典，缺省 `None` 表示现状不变），在 `simulate(...)`、`simulate_surface(...)`、`create_surface_missile(...)` 三处都能传入。

### 1. 天线指向（搜索时）

- 搜索状态下（第一次开机，或跟踪丢失超过 `prolongation_time_max_s` 后），天线中心指向**导引头可用的目标估计方向**：
  - 数据链连着时，用数据链解；
  - 否则用惯导解（`InertialTrackPropagator`，丢锁时已用最后一次雷达解重置）；
  - 都没有时，用导弹机体前向。
- 扫描图案以这个方向为中心。中心本身仍受 `lockAngleMax` 限制：超出时，夹到机头 `lockAngleMax` 锥面上最接近的方向。

### 2. 扫描

- 波束中心在指向方向周围，按扫描图案周期性移动。
- 可配置（D）：
  - `scan_half_angle_deg`：扫描区半角，默认 20°，不超过 `lockAngleMax`；
  - `frame_time_s`：扫完一遍的时间，默认 1.5 s；
  - `pattern`：`"spiral"` 或 `"raster"`，默认 `"spiral"`。
- 扫描相位每次进入搜索时从中心开始。
- 跟踪状态下天线跟着目标走，与现状一致，不扫描。

### 3. 天线方向图与探测距离

- 双程增益：`G(θ) = G_t(θ) · G_r(θ)`，`θ` 是波束中心到目标视线的夹角。
  - 单程方向图：主瓣按 `angleHalfSens` 为半功率半角（`θ = angleHalfSens` 时为 −3 dB），可用高斯近似 `exp(−ln2·(θ/angleHalfSens)²)`；
  - 主瓣以外取 `sideLobesSensitivity`（dB）作为下限。
  - 方向图形状是 D 级，阈值字段是 A 级。
- 探测距离按雷达方程四次方根缩放：`R_det(θ) = receiver.range · (σ / receiver.rcs)^(1/4) · G(θ)^(1/4)`。
  - `σ` 是目标 RCS。现有非诱饵路径没有目标 RCS 输入，需要沿用诱饵路径的 `target_rcs_m2` 接口（缺省 1 m²）。WT_overlay 会按飞机传入。
  - 上限仍是 `lock_range_m` / `receiver_range_max_m`。
- 只有 `range ≤ R_det(θ)`，并且通过现有的全部门限（距离门、多普勒门、杂波 notch、角度和角速率门），才算这一拍探测到。

### 4. 锁定

- 连续探测到 `dwell_s`（默认取 `lockTimeOut`，0.75 s；该字段的语义是 D 级）后才转入跟踪。
- 跟踪之后的逻辑与现状一致。中途漏一拍，按现有 `prolongation` 处理。

### 5. 诱饵（箔条）

诱饵路径（`_update_with_decoys`）用同一套指向、扫描和方向图。目标和每个诱饵各自按自己的 `θ` 和 RCS 算探测。

### 6. 记录

- `summary` 增加：`first_lock_time_s`、`lock_loss_count`、`reacquire_times_s`（列表）。
- 每拍样本增加：`seeker_scan_offset_deg`（波束中心离指向中心的角度）、`seeker_beam_target_deg`（波束中心离目标的角度）。

## 验收

1. **默认逐位不变**：不传 `seeker_search` 时，全部现有测试结果和失败列表与改动前完全一致。
2. **新单元测试**：
   - 方向图在 `θ = angleHalfSens` 处双程为 −6 dB，主瓣外为旁瓣下限；
   - 扫描中心跟随惯导解；
   - `dwell_s` 生效；
   - 指向中心超过 `lockAngleMax` 时被夹住；
   - 不传选项时 `R_det` 不参与判定。
3. **行为验收**：用 `WT_overlay/scripts/bench_seeker_reacquire.py all --sim-options '{"seeker_search": {}}'`（用 missile_sim 的 venv 运行），与不加选项的基线对比。下面是期望的**方向**，具体数值等校准：
   - notch 实验里，"回到原航向直飞"重新锁上或命中的比例明显高于"转 45°/爬升/俯冲"；
   - 后三者的命中率明显低于现在的 94%；
   - 同条件下 AIM-120D（漂移 0）对"回到原航向直飞"的命中率不低于 AIM-120C-5；
   - 不机动目标（weave 实验的 `none`）的命中率与基线相比不应明显下降，因为数据链或惯导指向准确时，目标就在扫描中心；
   - 在报告中写明各项命中率、重新锁上的比例和平均重新捕获时间。
4. 把基线和启用后的 bench 输出贴在实现报告里。

## 校准（D 参数，需要游戏内实验）

用户不知道扫描参数，建议在自定义战斗里用队友对打来测：

1. **直飞时的重新锁定时间**：用 notch 甩锁后立刻回到平飞直线，记录导弹是否重新锁上、隔多久。
2. **变线逃脱需要的偏离量**：甩锁后分别转 15°、30°、45°，或爬升、俯冲约 1–2 km，记录是否还会被锁上或命中。
3. **GNSS 与非 GNSS**：同样做法分别用 AIM-120D 和 120C-5 测，看直飞时被惯导命中的差别。

结果用来设定 `scan_half_angle_deg`、`frame_time_s`、方向图形状和 `dwell_s`。在此之前，WT_overlay 只在单独的对照训练里启用这个选项，不作为默认规则。

## 证据台账（`docs/BLKX_EVIDENCE_LEDGER.md` 新增一节）

| 字段 | 模型中的语义 | 等级 |
|---|---|---|
| `radarSeeker.*.antenna.angleHalfSens` | 搜索和探测用的单程半功率半角 | 数值 A，语义 C（字段名直解） |
| `radarSeeker.*.antenna.sideLobesSensitivity` | 主瓣外单程增益下限 | 数值 A，语义 C |
| `radarSeeker.receiver.rcs` / `range` | 参考 RCS 下的探测距离，雷达方程缩放 | 数值 A，缩放 D |
| （无字段）扫描图案、半角、周期 | 见需求第 2 节 | D（用户 2026-10-06：SRC 是扫描，具体不知道） |
| `guidance.lockTimeOut` | 搜索到锁定所需的持续探测时间 | 数值 A，语义 D |
| （无字段）天线指向估计位置 | 数据链或惯导解 | C（玩家报告：甩锁后直飞会被惯导打中，变线可逃） |

判别实验：见"校准"一节。

## WT_overlay 侧（实现完成后由 WT_overlay 接入，不属于本任务）

- `MatchEnv` 配置 `seeker_search`，传到交战的发射选项（`wt_overlay/engagement.py` 的 `LAUNCH_OPTIONS`，第 66 行）。
- 每架飞机的 RCS 作为 `target_rcs_m2` 传给导弹（环境里已有 `rcs_m2`）。
- 先做对照训练，看策略是否学会"甩锁后变线"，再决定是否作为默认规则。
