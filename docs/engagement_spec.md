# 技术文档：交战环境（里程碑 A）

## 背景

WT_overlay 是《战争雷霆》（War Thunder，商业电子游戏）的玩家辅助工具。本文档描述的是它离线训练环境里的**游戏对局模拟器**：用游戏文件复现的飞机、导弹、雷达和 RWR，在一个共用时钟上跑完一场游戏内的空战，用来训练和评估游戏内提示。

整体设计见 `docs/rl_design.md`。所有导弹、雷达名称都指游戏内单位，参数来自游戏客户端文件。

本任务是里程碑 A 的主体：把已有部件串成一场可以连续运行的对局，加上打法脚本，产出回放、轨迹图和实测吞吐。

## 规则

- 不提交。
- **不改 `/Users/skyking/Documents/missle_sim`**，只通过它的公共接口使用（加入 `sys.path`，做法同 `scripts/escape_window.py` 的 `_init`）。
- **不改 `wt_overlay/ui.py`、`wt_overlay/lowalt.py`、`scripts/build_lowalt_window.py`、`tests/test_lowalt.py`**：另一个会话正在改这些文件。
- `wt_overlay/` 下的运行时模块只用纯 Python，不用 numpy，要能在 PyPy（`/opt/homebrew/bin/pypy3.11`）下运行。
- 画图脚本可以用 `outputs/.mlenv/bin/python`（里面有 matplotlib）。
- 测试用 unittest，跑 `python3 -m unittest discover -s tests`，现有测试必须全部继续通过。
- 所有凭推断定下的规则都标 D 级，并追加到 `docs/rl_design.md` 第 8 节的待校准表。

## 已有部件（先读）

| 部件 | 位置 | 要点 |
|---|---|---|
| 导弹（可逐步推进） | missle_sim `aim120_model.public_api.create_surface_missile` | 坐标系 x / y 向上 / z，航向 0 指向 +x，正航向转向 +z；时间从发射时刻 0 开始；`step()` 推进 1/48 s；属性 `time_s`、`done`、`event`、`state`；`result()`。目标对象要实现 `state_at(t) -> aim120_model.target.TargetState(position, velocity)`，可选 `observe_missile(t, pos, vel)`；必须**先推进飞机、再推进导弹**，并能插值查询节拍端点和步内时刻。`launcher_support(t, truth) -> str`：返回 "" 表示仍在支援，第一次返回非空原因后数据链永久断开。箔条：目标对象提供 `decoys_at(t) -> [aim120_model.chaff.Reflector]` 和 `rcs_m2`，参考 `aim120_model/chaff.py` 的 `ChaffingTarget`。 |
| 导弹参数集 | missle_sim 的 profile catalog | 加载方式见 `scripts/escape_window.py:_init`。挂载里的导弹编号没有对应 profile 时，用 `wt_overlay.pk.ALIASES` 回退（例如 su_rvv_sd → su_r_77_1）。 |
| 飞行模型 | `wt_overlay/turn.py` 的 `ManeuverModel`；`wt_overlay/escape.py` 的 `FMEvader._fly` | `_fly` 已经实现：按限定滚转速率转升力矢量、迎角响应、发动机响应、减速板，节拍 1/48 s（`SUBSTEP_S`）。坐标转换用 `escape.to_enu` / `escape.from_enu`。 |
| 雷达和 RWR | `wt_overlay/sensors.py` | ENU 坐标，地面 z = 0；航向是从正北顺时针的罗盘角。`RadarSensor`、`RwrSensor`、`OwnState`、`TargetTruth`、`Emission`；policy 可见的输出和真值 id 分开存放。 |
| 机体装备 | `wt_overlay/units.py`、`data/units/` | `units.load().equipment[aircraft]`：雷达、RWR、MLWS、导弹及数量上限、干扰弹数。 |
| 对局模型 | `data/match/top_tier.json` | BR 14.3–14.7 的 19 架飞机、出场频率、打法先验、出生参数、脚本参数。 |
| 命中概率与射程 | `wt_overlay/pk.py`、`wt_overlay/offense.py` | 脚本判断"进入 Rmax 的 80–100%"等条件时使用。 |

## 要做的东西

### 1. `wt_overlay/flight.py`：可控制的飞机

把 `FMEvader._fly` 的物理改成一个由指令驱动的飞机类。

- **指令** `FlightCommand`：
  - 期望速度方向。可以给 ENU 方向向量，也可以给罗盘航向加爬升角，或者给目标高度。
  - 速度要求：目标速度或油门。
  - 过载上限、是否允许用减速板。
- **每个节拍 1/48 s**：保存位置、速度、法向、迎角、发动机、减速板，并保留历史，供导弹查询和插值。
- **撞地**（z ≤ 0）算死亡。
- **质量**：空重 × U(1.15, 1.45)，油耗不计（D）。
- **不得改变 `FMEvader` 的结果**。如果把公共物理抽成函数共用，必须保证 `tests/test_escape.py` 全部通过，且 FMEvader 的输出逐位不变。

### 2. `wt_overlay/engagement.py`：交战管理器

**世界**：ENU 坐标，平地。地图边界可配置，默认 128 km 见方，即 ±64 km（C，论坛：顶级空战地图约 128×128 km，引擎上限约 130 km）。

**实体**：
- 飞机：队伍、机型、飞行模型、装备、导弹种类和剩余数量、干扰弹、雷达、RWR、是否有 MAW、存活状态。
- 导弹：missile_sim 实例、发射者、目标、世界发射时刻。
- 箔条包：投放时刻、位置、速度。

**每个节拍的顺序**：
1. 控制器决策。脚本约每 0.5 s 决策一次，两次之间保持指令。
2. 推进所有飞机。
3. 推进所有导弹。
4. 结算事件：引信触发即击落（D），记击落和助攻；撞地；导弹寿命或距离耗尽后移除。
5. 更新传感器：每架飞机的雷达和 RWR。

**导弹的目标代理**（每枚导弹一个）：
- `state_at(t)`：把世界时间 = 发射时刻 + t 处的飞机历史插值结果，换到 missile_sim 坐标系。
- `decoys_at(t)`：返回该飞机投放的箔条，换坐标系。
- `rcs_m2 = 1`：箔条 RCS 按相对比例给。
- `observe_missile`：记录导弹位置，供目标的观测使用。

**发射**：
- 只能对雷达里有航迹的目标发射（TWS 航迹或 STT），且导弹有剩余。两次发射至少间隔 1 s（D）。
- 发射状态：位置和速度取飞机当前值。俯仰、航向取速度方向，迎角不计，标 D。
- 选项与以前的数据生成一致：`observation_mode='sensor_track'`、`loft=True`、`clutter_model='look_down_angle'`、`clutter_min_depression_deg=2`、`cw_on_clear_beam=True`、`require_seeker_lock=True`。
- 发射方支援用 `launcher_support` 回调：发射者活着，且它的雷达当前仍对该目标保有航迹（TWS 航迹或 STT，含外推期间），就返回 ""。否则返回 "shooter_dead" 或 "track_lost"。按真值 id 在内部匹配。

**箔条**：
- 每架飞机的干扰弹数取 `equipment.countermeasures`，其中箔条占 50–100%（分装发射器可选比例，D）。
- 每局每架飞机抽一个 `rcs_ratio`，取 0.5、1、2 之一，与命中概率模型一致。
- 指令可以要求投放 n 包。

**每架飞机的观测**（只放玩家能得到的信息，真值另存）：
- 本机状态；
- 雷达画面；
- RWR 画面：敌方雷达照射，以及敌方导弹导引头开机后的照射；
- MAW：有 MLWS 的飞机，在导弹动力段内、约 10 km 内告警（D）；
- 动力段尾焰：导弹动力段内，按每架飞机抽一个概率看到（默认 0.5，与命中概率模型一致，D）；
- 目视：约 8 km 内的敌机给出方位（D）；
- 地图：队友位置；被任一队友雷达或目视发现的敌机，给出平面坐标和时间（D）。

**结束条件**：一方全灭；或到时间上限（默认 900 s）；或者双方相距很远且空中没有导弹超过 60 s。

**确定性**：每局一个随机种子；同一种子必须得到完全相同的回放。

**回放**：
- 每 0.25 s 写一行 JSONL：所有飞机和导弹的位置、速度、航向、剩余弹药，以及打法脚本当前所处的阶段。
- 另外记录事件：发射、击落、死亡、箔条、RWR 告警、航迹丢失、数据链断开、阶段切换。

### 3. `wt_overlay/archetypes.py`：打法脚本

按 `docs/rl_design.md` 第 2 节和 `data/match/top_tier.json` 实现五种原型：左飞、右飞、中路、爬虫、冲脸。

- **只用观测做决策**，不看真值。调试可以留一个开关。
- **阶段**：
  1. 爬升侧移；
  2. 可选：打一发压制弹（Rmax 的 80–100%）；
  3. 规避：往外侧三九或掉头，可往下扎，放箔条；
  4. RWR 清空 3–10 s 后回切；
  5. 第二轮：20–30 km 内发射。
  然后循环 3–5。
- **队伍坐标**：前方指向敌方出生点，左侧是前方逆时针转 90°。"外侧"指本侧翼方向：左飞往左、右飞往右、中路往左或回家。
- **反应**：
  - 对新威胁的反应有延迟，对数正态分布，中位数 1.5 s。
  - 技术水平"普通"：85% 会反应，其中 60% 选对动作；"高手"总是反应且总是选对。选错时从动作库里随机挑一个。
- **射程判断**：用 `wt_overlay.pk` 的模型，或者 `offense.OffenseAdvisor` 的 Rmax 线，按高度和速度分箱缓存。敌机型号未知时，用 `pk.Assumption()` 的默认值。
- 其余参数（爬升高度、侧移角度、压制弹概率、打完是否支援、箔条放法、第二轮距离、爬虫高度、中路回家的概率、歼-16 与苏-30SM2 中路提早左转）都从 JSON 读取。

### 4. `wt_overlay/match.py`：对局生成器

- 按 `aircraft_frequency.weights` 抽 32 架飞机，分成两队各 16 架。
- **出生**：两条出生线相距 90–110 km，横向散开约 ±15 km（D）；高度 2–3 km；速度 0.8–1.2 马赫；航向朝向敌方。
- 每架按其机型组的先验分配打法（Dirichlet 浮动）。
- 技术水平：普通占 80%，高手占 20%（D）。
- 挂载：在已建模的主动弹里选一种，数量取上限（上限可能偏高，D）。
- 也要能生成小规模场景：指定机型、打法和距离的 1 对 1、2 对 2。

### 5. 脚本

- `scripts/run_engagement.py`：
  - 参数：`--mode 1v1|2v2|16v16`、`--seed`、`--aircraft`、`--archetypes`、`--out`、`--profile`；
  - 运行一局，输出回放和统计：击落、死亡、发射数、各阶段用时、耗时分解（飞机、导弹、传感器、脚本）。
- `scripts/plot_engagement.py`：读回放，画俯视轨迹图和高度–时间图。飞机按队伍着色，导弹画细线，事件打标记。

## 测试

新增 `tests/test_flight.py` 和 `tests/test_engagement.py`，至少覆盖：

- **飞机跟随指令**：转到指定航向、爬到指定高度、跟到指定速度，过载和滚转速率不超限。
- **坐标换算一致**：目标平飞时，用目标代理跑出的导弹结果，与 missile_sim 旧接口在同一几何下的结果接近：引信结果相同，飞行时间相差 0.1 s 以内。
- **1 对 1 迎头**：发射后，目标在导引头开机时收到 RWR 导弹告警；规避方向是外侧；箔条数量减少；击落时记到发射者名下。
- **载机支援中断**：发射方转出雷达范围或丢失航迹后，导弹数据链断开。
- **确定性**：同一种子回放完全一致。
- **16 对 16 冒烟测试**：跑 120 s 不出异常，数值全部有限。

## 性能

分别在 CPython 和 PyPy 下实测 1v1、4v4、16v16 每墙钟秒能推进多少仿真秒，并给出耗时分解：飞机、导弹、雷达、RWR、脚本。

注意：雷达耗时随雷达类型、扫描范围、行数和模式变化，之前 `scripts/bench_sensors.py` 的数字只是 F-16C 和台风、全部目标都在探测距离内的一组测试情况。这次要在真实对局里按机型和模式分别统计。

## 交付

1. 回放文件和轨迹图：
   - 1 对 1：苏-30SM2 对 F-15C 金鹰；
   - 1 对 1：台风 AESA 对歼-16；
   - 一场 2 对 2；
   - 一场由生成器产生的 16 对 16。
2. 吞吐表，含 CPython 和 PyPy。
3. 新增的 D 级假设清单，已追加到 `docs/rl_design.md` 第 8 节。
4. 改动文件清单和测试结果。
5. 遇到但没解决的问题。例如某条规则会让结果明显不合理时，列出来，不要自行"调参"掩盖。
