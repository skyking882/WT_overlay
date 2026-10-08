# 技术文档：回机场补弹、保 KD 与误伤惩罚（WT_overlay 交战环境，4v4 起用）

## 背景

4v4 训练（`~/rl_runs/s2_4v4`，2026-10-07）发现：

- 超时局里活着的飞机导弹全是 0。每架 12 枚打掉 8–9 枚，模拟里没有红外弹和机炮，打光就只能干耗到时间结束。
- 误伤击落在上升，但现在只有被击落的一方扣分（阵亡 −2），开火的一方不受罚。

### 游戏里的行为（C 级，用户 2026-10-07）

- 主动弹打光会**回机场降落、补弹、再起飞**。一局总时长 **25 分钟**。
- 允许"打不过就撤回机场停住"（保 KD）。停在地上到结束**不扣分，也不加分**（0）。用户原想给 +0.5，但 4v4 里每架的平均战斗收益接近 0，加分会让开局直接回去降落比打更划算（挂机），所以定为 0。
- **机场就在本队出生点**（出生线中点，即 `Pilot.home_xy`），在地面上。
- 从**落地到补完弹再起飞约 20 秒**，不含进近。
- **不模拟机场防空**。地面上的飞机不能被探测、锁定或击中。
- **误伤击落**：开火的一方扣 0.5。不算最严重的问题，但要让 AI 注意开火。

## 要求

全部是可选项。不设时与改动前逐位一致：同一 seed 的观测、奖励、事件、回放都相同。现有测试必须不改就能通过。

### 1. MatchEnv 配置

- `airfield`：字典（`{}` 用默认值），键：
  - `approach_m`（默认 21000；原 15000，2026-10-07 改，见第 9 节"进近时间"）：进近半径，见第 2 节；
  - `approach_alt_m`（默认 300）：进近目标高度；
  - `approach_ias_kmh`（默认 450）：进近目标表速；
  - `land_radius_m`（默认 2500）：落地判定的水平半径；
  - `land_max_alt_m`（默认 600）：落地判定的最高高度；
  - `land_max_ias_kmh`（默认 550）：落地判定的最高表速；
  - `turnaround_s`（默认 20）：落地到可以再起飞的时间，期间补满导弹和干扰弹；
  - `takeoff_alt_m`（默认 100）、`takeoff_ias_kmh`（默认 350）：再起飞时的初始状态；
  - `all_grounded_end_s`（默认 30）：见第 5 节。
- `friendly_fire_reward`：数，默认不设（等于 0）。4v4 用 −0.5。
- 4v4 训练会同时把 `time_limit_s` 设为 1500（25 分钟），这部分只是配置值，不用改代码。

上面的默认值是估计（D 级），实现时可以根据飞行模型能否飞出来做调整。调整的值和理由写进本文档"实现记录"一节。

### 2. 回家与进近（`intent.py` 的 `IntentExecutor`，脚本和策略共用）

意图里的 `maneuver == 10` 原本是"回家"：飞向 `home_xy`，12 km 内盘旋。设了 `airfield` 时改为：

- 离机场超过 `approach_m`：照旧飞向机场。
- 在 `approach_m` 以内：直飞机场，目标高度改为 `approach_alt_m`，目标表速改为 `approach_ias_kmh`（收油门减速）。
- 满足落地条件（第 3 节）就由环境转入地面状态。

不设 `airfield` 时，`maneuver == 10` 的行为不变。

### 3. 地面状态（`engagement.py`）

飞机的状态分为空中和地面两种。

**落地条件**（每个 tick 检查）：

- 选了"回家"，即本步执行的意图 `maneuver == 10`；
- 距本队机场水平距离 ≤ `land_radius_m`；
- 高度 ≤ `land_max_alt_m`；
- 表速 ≤ `land_max_ias_kmh`。

满足后飞机转入地面状态：位置放在机场（z = 0），速度为 0，雷达关闭，发事件 `landing`。

**地面上**：

- 对所有传感器不可见：
  - 不出现在任何雷达、RWR（它也不发射）、目视或 MAW 里；
  - 不能被选为目标；
  - 正在追它的导弹失去目标：按现有"目标丢失 / 自毁"路径处理，不改 missile_sim。
- 不参与坠地、出界、超速判定。
- 它在空中的导弹照常飞，但失去数据链支持，和发射机关雷达的情况一样。
- 落地满 `turnaround_s` 后，导弹和干扰弹补满到本局开始时的挂载数，发事件 `rearm`。

**再起飞**：补完后，只要该机本步的意图 `maneuver != 10`，就再起飞：

- 出现在机场上空 `takeoff_alt_m`，表速 `takeoff_ias_kmh`；
- 航向朝向敌方出生点（`team_forward`）；
- 恢复空中状态，飞行模型状态合理重置；
- 发事件 `takeoff`。

补完后一直选 `maneuver == 10` 就一直停在地上（保 KD）。

### 4. 脚本（`archetypes.py`）

- 脚本没弹时进入 `home` 阶段，经 `from_flight_action` 转成 `maneuver = 10`，所以会按第 2–3 节自动进近、落地、补弹。
- 补完弹后要能离开 `home` 阶段再起飞，重新接敌（例如进入 `recommit` 或 `advance`，按现有阶段逻辑选最合理的）。
- `middle` 的 `go_home` 原型行为保持原意：它本来就是回家不打。实现时说明它在机场会怎么做。

### 5. 结束与奖励（`rl_env.py`、`engagement.py`）

- **全灭**：一队没有活着的飞机才算全灭。地面上的飞机算活着。
- **全员落地提前结束**：所有活着的飞机都在地面上，且没有在飞的导弹，持续 `all_grounded_end_s` 后提前结束。结果按超时处理：`reason` 用新值 `all_grounded`，但 `info["time_limit"]` 照旧为真，因为训练端按超时处理 bootstrap / timeout_reward 的逻辑见下一条。
- **超时扣分**：时间到（或全员落地提前结束）时，`timeout_reward` 只给仍在空中的策略飞机。在地面上的飞机拿 0。
- **误伤惩罚**：`friendly_fire_reward` 给误伤击落队友的开火者。
  - 用 `engagement` 的 `friendly_fire` 事件里的 `killer`。
  - 开火者已阵亡时走 `late_rewards`，和击落 +1 的迟到战果同一路径。
  - tallies 不变：误伤不算击落。
- **新事件**：`landing`、`takeoff`、`rearm`。`info["events"]` 里加上它们的计数，但只在设了 `airfield` 时才加。

### 6. 观测与动作屏蔽（`rl_observation.py`）

- 不新增输入维度，也不新增动作头，这样能从现有 4v4 检查点直接继续训练。
- 地面上本机状态自然体现为：高度 0、速度 0、离家距离约 0；补完弹时导弹数变满。
- 地面上屏蔽武器头（不能开火）和干扰弹头。其他头照常可选，选什么都不影响地面状态，只有第 3 节的 `maneuver` 规则起作用。
- 队友在地面上时，`friend` 实体照常给出（它在地图上）。敌方看不到它。
- critic 用的真值实体里要能区分地面状态。如果真值字段里没有现成的位置，至少保证高度 0、速度 0 一致。

### 7. 回放、快照与评估

- **回放**：plane 行的 `phase` 列在地面上写 `ground`。`landing`、`takeoff`、`rearm` 写进事件日志。
- **快照**：`MatchEnv.snapshot()` / `restore()` 要包含地面状态和补弹计时。
- **评估**（`rl/eval_replay.py` 的团队统计 `_team_game` / `team_tally` / `summarise_teams`）：
  - 增加受测飞机的 `landings`、`rearms` 和 `grounded_at_end`（比赛结束时在地面上的架数）；
  - 结果判定里地面上的飞机算活着；
  - 1v1 输出不变。

### 8. 测试

`tests/test_rl_env.py`、`tests/test_engagement.py`（或现有对应文件）、`rl/tests/test_eval_replay.py`：

- 不设 `airfield` / `friendly_fire_reward` 时，现有测试全部通过，逐位不变。
- 一架飞机选 `maneuver = 10`，能在合理时间内进近并落地；至少测两种机型，例如 F-15C 金鹰和苏-30SM2。
- 落地后不可见：敌方雷达航迹消失，追它的导弹失去目标。
- `turnaround_s` 后弹药补满；之后选别的 `maneuver` 就起飞，一直选 10 就留在地面。
- 时间到时，地面上的策略飞机拿 0，空中的拿 `timeout_reward`。
- 全员落地 30 s 后提前结束。
- 误伤击落时开火者拿 `friendly_fire_reward`，开火者已阵亡时走 `late_rewards`。
- 脚本打光导弹会回去补弹、再起飞、再接敌：跑一局 2v2 脚本局，检查事件序列。
- 快照 / 恢复包含地面状态。

### 9. 实现记录

（2026-10-07。代码：`engagement.py`（`AIRFIELD`、`airfield_settings`、`Engagement._airfield/_land/_takeoff`）、`intent.py`、`archetypes.py`、`rl_env.py`、`rl_observation.py`、`sensors.py`（`RadarSensor.forget`）、`match.py`、`rl/eval_replay.py`。）

**默认值**：全部沿用第 1 节，没有调整：飞行模型都飞得出来（下面的实测）。4v4 用的配置：`{"airfield": {}, "friendly_fire_reward": -0.5, "time_limit_s": 1500}`。

**进近 / 落地实测**：MatchEnv 1v1，策略执行路径（follow，默认延迟、5° 误差、5% 拒绝），放在离本队机场 40 km、8 km 高、0.9 Ma，从这一步起一直选 `maneuver = 10`（速度头 0：全油门），记到 `landing` 的时间（两个种子）：

| 机型 | 朝机场飞 | 背对机场（要先掉头） |
|---|---|---|
| F-15C 金鹰 | 121 / 121 s | 141 / 147 s |
| 苏-30SM2 | 118 / 119 s | 139 / 143 s |
| 歼-10C | 119 / 121 s | 145 / 155 s |
| 台风 AESA | 112 / 113 s | 134 / 141 s |

- 都是第一次飞到就落地，落地时高 240–420 m、表速 549–550 km/h（最后满足的总是表速条件）、离机场 245–740 m。15 km 进近（8 km 高、最大俯冲 30°）约 55–75 s；进近目标表速 450 km/h 在落地前到不了，它只起"收油门 + 减速板"的作用。
- `policy_ground_floor: false`、`policy_deck_m: 10` 时时间相同（最低 315 m，地面保护从未起作用）。
- 起飞：机型池全部 19 种、质量系数 1.15 和 1.45，从 100 m、350 km/h 按执行器的最大 30° 爬向 11 km，速度从未低于起飞速度（98 m/s），60 s 后在 4.7–8.5 km。

**4v4 脚本局**（MatchEnv，`team_size` 4，`top_tier_s1_far.json`，`time_limit_s` 1500，`policy_ids` 0–3 由各自脚本决策，`airfield {}`，`friendly_fire_reward -0.5`，`timeout_reward -0.5`）：

- 种子 1–10 中 8 局在 513–832 s 一方全灭，期间没有降落；种子 5、7 有降落。
- 种子 7，6 号机（歼-16，middle，1 队）：175 s 进入 home → 404 s `landing` → 424 s `rearm`、home→advance → 426 s `takeoff` → 634–905 s 发射 10 枚 → 910 s 再进入 home → 1241 s `landing` → 1261 s `rearm` → 1262 s `takeoff`。整局 1474 s 全灭结束（同种子不开 airfield 782 s 结束）。
- 种子 5：3 号机 1026 s 落地、1046 s 补弹、1047 s 起飞，1266、1388 s 再发射；7 号机（台风，middle）409 s 进入 home，1085 s 落地、1105 s 补弹、1108 s 起飞，1258–1387 s 发射 4 枚；2 号机 1180 s 落地、1201 s 起飞。到时间时没有飞机在地面上。每次地面时间 20.9–22.9 s（20 s 周转 + 执行器延迟）。
- 脚本从进入 home 到落地 229–676 s（1 队记有 phase 事件的三次）。最后 15 km 进近约 75 s，其余是脚本回家的飞法：home 阶段照样防御（被追时 evade、俯冲到低空再回 home，可反复多次），回 home 后先爬回 `HOME_ALT_M`（6 km，垂直头量化成 8 km）才在 15 km 处开始下降。没有改这部分。
- 墙钟：种子 5（两局都到 1500 s）214.5 s，不开 airfield 203.0 s（+5.7%，多出的是额外的导弹：发射 45 对 39）；种子 7：229.6 s（1474 s 的局）对 158.9 s（782 s 的局）。

**吞吐**：同种子、没人回家的局两种设置逐位相同，只差判定开销：种子 1 每步 80.59 对 80.46 ms，种子 2 94.49 对 94.35 ms（+0.15%）。不设选项时与改动前逐位一致（同一 seed 的观测、奖励、info、引擎日志和回放的哈希对比：1v1 脚本 / 随机动作、4v4 脚本含 policy_count/自博弈/egocentric/timeout_reward、4v4 随机动作、独立 3v3 引擎回放）。

**进近时间（2026-10-07 修订：`approach_m` 默认 15 km → 21 km）**：用户要进近（不含飞到机场附近的路程）约 90 s；上面的 55–75 s 是 15 km 时的数。只改默认半径，进近飞法没动。测法：MatchEnv 1v1，两架各在本队机场一侧 40 km、8 km 高、0.9 Ma，朝机场飞，每步 `maneuver = 10`；记从第一次进入 `approach_m` 到 `landing` 的时间。"策略"= 执行路径 follow（默认延迟、5° 误差、5% 拒绝）、速度头 0（全加力）；"脚本"= 该机没有导弹（一开始就是 home 阶段）、由脚本按自己的巡航速度飞。每格是 F-15C 金鹰、苏-30SM2、歼-10C、台风 AESA 各两个种子共 8 次的范围（均值）：

| `approach_m` | 策略（全加力飞来） | 脚本（巡航飞来，过半径时 0.86–1.14 Ma） |
|---|---|---|
| 15 km（原默认） | 57–61 s（58） | 69–77 s（72） |
| 20 km | 69–77 s（72） | 82–95 s（88） |
| **21 km（新默认）** | **76–86 s（79）** | **90–103 s（96）** |
| 22 km | 83–94 s（87） | — |
| 22.5 km | 88–98 s（92） | 101–115 s（108） |
| 24 km | 100–111 s（104） | — |

- 进场速度决定时长：同样 22.5 km，从半径外 1.5 km、0.9 Ma 直接进场的策略飞机要 101–110 s（不先加速）。取 21 km 是两种飞法的折中（合计均值约 88 s）；只按策略飞机定到 90 s 要 22.5 km，那时脚本约 107 s。
- 20 km 以上时都在 `land_radius_m`（2.5 km）边上落地，落地时高约 300 m、表速 440–550 km/h：已经降到进近高度、减到进近速度，最后几公里是 450 km/h 平飞。
- 4v4 脚本局里脚本从进入 home 到落地的总时间（上面的 229–676 s）主要是回家路上的防御和爬升，不是进近。
- 设了 `fuel`（docs/fuel_spec.md）时补弹同时加油到开局载油，`rearm` 事件多 `fuel_kg`。

**实现要点与判断**：

1. 落地和起飞都看执行器正在执行的意图（`IntentExecutor.executed`，带人的反应延迟和 5% 拒绝）：设了 airfield 时执行器每 tick 把 `plane.want_home` 设为 `executed.maneuver == 10`，规则在 `Engagement._settle` 里检查。
2. 地面上飞行模型冻结（不步进）；`plane.own` 是机场 (x, y, 0)、速度 0、航向 = 起飞航向（朝敌方出生点）。起飞时用同一个模型新建 `Aircraft`（t0 = 起飞时刻、满油门、机翼水平，故障计数延续），雷达回到 TWS。执行器在起飞后第一个 tick 重新瞄准：航向从起飞航向算起，高度目标沿用落地前的（从未设过时为起飞高度）。
3. 不可见：落地时敌方雷达直接删掉对它的航迹、扫描点和 STT（`RadarSensor.forget`，记 `track_lost`），并从对方地图标记里删除；目视、目标框、尾迹、MAW、RWR 都不再有它；不能对它发射。它自己雷达关闭，RWR、MAW、目视、导弹标记也都关（"停在地上"）；仍有队友标记、队友发现的敌机标记和自己的导弹。
4. 追它的导弹（第 3 节"失去目标"）：走现有"目标已死"的路径。导弹继续飞向落地点（固定点、速度 0，与飞向残骸相同）；`retarget_dead`（默认开）时每 tick 尝试重新捕获导引头视场内的空中飞机（含友机）。丢失目标的导弹不能杀伤（在落地点 fuse 也不算击落），以 ground / lifetime / max_distance 或这种空 fuse 结束；`retarget_dead` 关时立即结束，`result = target_lost`。落地时对它的 `missile_hist` 结束（不再计入助攻窗口）。注意：刚发射就丢目标的导弹可能重新捕获发射机自己（现有 `_retarget` 的行为，未改）。
5. 它自己在空中的导弹：落地时 `tracked` 清空，`LauncherSupport` 返回 `track_lost`，数据链断开，与关雷达相同。
6. 动作屏蔽：地面上屏蔽武器头和干扰弹头。选自由视角时 maneuver 头只能为 0，所以补完弹后选自由视角也等于起飞。
7. 观测：本机高度 0、速度 0、速度向量 0、俯仰/滚转/迎角 0、过载 1、发动机 0、雷达 off。critic 真值里地面飞机是（机场, 0）、速度 0、发动机 0、过载 1；没有用空余列加新标志。
8. 进近：`approach_m` 以内高度目标 `approach_alt_m`、速度目标为 `approach_ias_kmh` 在当前高度的真空速，允许减速板，覆盖 vertical / speed 头；`approach_m` 以外直接飞向机场（设了 airfield 时不再在 12 km 内盘旋）。
9. 脚本：补完弹后 home → advance（left / right / middle）、crawl（crawler）、rush（rusher），起飞后直接朝敌人去，不再侧翼爬升或打压制弹。middle 的 `go_home`：代码里它只改变 evade 中 drag 的方向（朝本方而不是背离威胁），那时 phase 是 evade、maneuver ≠ 10，所以飞到机场上空也不降落、照样飞过去；它和别的脚本一样只在打光导弹（home 阶段）时降落、补弹、再起飞。
10. 结束：全员落地计时从"最后一枚导弹结束且所有活着的飞机都在地面"起算；`reason = all_grounded`，`rl_env` 按 time_limit 处理（不设 `timeout_reward` 时 `info["timeout"]` 为真、可 bootstrap；设了时 `info["time_limit"]` 为真、地面上的策略飞机拿 0）。机场相距 80–100 km（小于僵持规则的 120 km），停在机场不会触发 stalemate。
11. 回放 / 汇总（只在设了 airfield 时）：header 多 `airfield`（设置值 + `bases`：每架 [x, y, 起飞航向]）；帧里地面飞机 phase = `ground`、位置（机场, 0）、速度 0；事件 `landing`（含落地时高度、表速、距离）、`rearm`、`takeoff`；`summary()` 每架多 `grounded / landings / rearms / takeoffs`；`Result.phase_time_s` 多 `ground`。
12. 评估：团队统计行多 `landings`、`rearms`、`grounded_at_end`，`summarise_teams` 给总数，团队考试结果多 `policy_landings / policy_rearms / grounded_at_end`；只在环境有 airfield 时出现，1v1 输出不变。

**未解决 / 注意**：

- `time_limit_s` 改为 1500 后，本机观测第一维 `time = t / 900` 在 900 s 以后大于 1，若现检查点训练时 `time_limit_s` ≤ 900 则是没见过的取值。按第 1 节只改配置，没有改代码。
- 两架同时起飞会在同一点出现（没有碰撞模型）。
- `spawn_layout` 可以把出生线（机场）放到离地图边 8 km 处，那时脚本的边界守卫（maneuver 7）可能打断进近；实测里没有出现。
