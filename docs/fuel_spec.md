# 技术文档：奖励防滥射、燃油与回场进近时间（WT_overlay 交战环境，可选项）

（2026-10-07。代码：`wt_overlay/rl_env.py`（配置检查、`step` 奖励）、`wt_overlay/engagement.py`（`FUEL`、`fuel_settings`、`assist_rule_setting`、`Engagement._kill/_retarget/_flameouts/_fuel_tank/missile_mass`、`AIRFIELD`）、`wt_overlay/flight.py`（`FuelData`、`read_fuel`、`fuel_data`、`FuelTank`、`Flameout`）；测试 `tests/test_fuel_reward.py`。）

## 背景

4v4 实测：助攻数约等于击落数（每次击落 0.93–1.00 个助攻），助攻占正奖励的 22%，每次击落要发射 11–19 枚。导弹不花钱，对已经有人在打的目标补射期望收益为正。另外，目标已死的导弹会改追导引头视场里最近的飞机（不分敌我），打中了照样给开火者记满额击落。

模拟里没有燃油：质量是空重 × U(1.15, 1.45)，不消耗，加力不要钱。

## 要求

全部是可选项。不设（或设为 `None`）时与改动前逐位一致：同一 seed 的观测、奖励、事件、回放都相同。现有测试不改就能通过。actor 的观测宽度不变（远程 4v4 要从现有检查点直接接着训），新量只放在原始本机观测 `OwnObs` 上。

## 1. 奖励（MatchEnv 配置）

| 键 | 默认 | 含义 |
|---|---|---|
| `assist_rule` | `None`（现有规则） | `"first_shot"`：见下 |
| `launch_reward` | `None`（= 0） | 策略飞机每发射一枚加这个值，例如 −0.05 |
| `retarget_kill_reward` | `None`（= +1） | 改追过目标的导弹打下的击落改记这个值，例如 0.5 |

- **`assist_rule: "first_shot"`**（`Engagement._kill`）：助攻（+0.3，没有 `assist_reward` 键，数值不变）只给这样的队友：他指向受害者的导弹**早于**击落导弹指向受害者（发射时刻；改追来的导弹按改追时刻；直接调用 `_kill` 不带导弹对象时按击落时刻），并且在飞或结束不到 20 s；每对（开火者，受害者）整局最多一次（`Engagement.assisted`）。`assist` 事件只为记了分的助攻发，`Plane.assists`、`info["events"]["assist"]` 和评估指标的口径一致。不设时规则不变，`assisted` 一直是空集。
  - `pending_credit()` 没有改：它仍按原来的 20 s 窗口判断"可能还有助攻"，在 `first_shot` 下是偏宽的估计（多等几步，不会漏账）。
- **`launch_reward`**：`launch` 事件的 `shooter` 在 `policy_ids` 里才加，加在发射那一步（开火者在本步已死时走 `late_rewards`，实际不会出现）。脚本机不受影响。
- **`retarget_kill_reward`**：`Engagement._retarget` 把改追过的导弹 uid 记进 `Engagement.retargeted_uids`（不改事件内容）；`kill` 事件的 `uid` 在里面时，开火者得到这个值而不是 +1。
  - 目标降落（airfield）后改追的也算。
  - 迟到战果（开火者已阵亡，`info["late_rewards"]`）同样用这个值。
  - `info["tallies"]` 照旧把它算一次击落，`kill` 事件计数不变：胜负、交换比统计不受影响，只有奖励变小。
  - 改追打到队友仍是误伤（`friendly_fire_reward`），与本键无关。

## 2. 燃油（`fuel`）

配置 `fuel: {}` 用默认值，键：

| 键 | 默认 | 含义 |
|---|---|---|
| `fraction` | `[0.45, 1.0]` | 每架飞机开局载油 = 油箱容量 × U(lo, hi)，每架单独抽（游戏里玩家自选载油，D） |
| `bingo` | `0.15` | 剩油 / 开局载油低于它时 `OwnObs.bingo` 为真（给脚本回家用，见 2.7） |
| `tanks` | `"max"` | 油箱容量：`"max"` = `Mass.MaxFuelMass0`；`"internal"` = 不带 external 标记的油箱之和 |

不设 `fuel` 时质量仍是空重 × `mass_factor`（match 照旧抽这个随机数，开了 fuel 也照抽、只是不用，所以其他随机数序列不变），不耗油，加力不要钱。

### 2.1 数据与单位

FM 文件（`data/fm/aircraft/*.blkx`，A）：`Mass.EmptyMass`、`Mass.MaxFuelMass0`、`Mass.Parts.tank*_capacity / tank*_external`，每个发动机类型 `EngineType*.Main` 的 `FuelConsumptionOnIdle / OnHalfThr / OnFullThr / OnWEP`。`flight.read_fuel` 按 `fm.StaticModel` 同样的顺序列发动机（前飞时关着的升力发动机不算）。

**单位：比油耗，kg 燃油 / (kgf 推力 · h)（D，推断）。** 依据：

1. 同一 `Main` 块里的推力表（`ThrustMax`）是 kgf（`fm/engine.py`）。
2. 数值大小与战斗机涡扇的军用推力比油耗同量级（约 0.65–0.9 kg/(kgf·h)）：F-15 / F-16A / 台风 / 阵风一族 0.74、F-16C Block 50 / F-14 / F-2 一族 0.69（狂风 0.8）、苏-27 / 苏-30 / 歼-10 / 米格-29 一族 0.88、F/A-18 / 鹰狮 0.8。
3. F-16C Block 50 一族的 idle 0.93 大于 half 0.69：作为每小时绝对流量（kg/h）这不可能（慢车比半油门烧得多），作为比油耗则合理（低推力时比油耗升高）。
4. `ConsumptionOmegaMax` 总等于 `FuelConsumptionOnFullThr`。

WEP 值按文件原样用：多数机型 1.05–1.15；苏-27 / 歼-10 / 米格-29 一族 0.88、F/A-18 / 鹰狮 0.8，与军用值相同（加力只按推力多烧）；F-16C Block 50 / 40、F-14、F-2、狂风一族 2.75。差 2.5–3 倍，是游戏文件的数，不是本实现的选择。

**`MaxFuelMass0` 包括副油箱。** 例：F-15C 金鹰 15,795 kg = 机内 6,103 kg + 3 × 1,791 + 2 × 2,158（`tank*_external` 为真）；F-16C Block 50 6,398 = 机内 3,305 + 外挂 3,088。游戏里副油箱是挂载项，不挂时这部分油没有。按用户原话默认用 `MaxFuelMass0`（`tanks: "max"`）；`tanks: "internal"` 只用机内油箱（苏-30SM2 两者相同，9,400 kg）。建议 4v4 用 `"internal"`，见 2.8 的质量对比。

### 2.2 载油与质量

- 开局载油：`Random(f"{seed}:fuel:{ident}").uniform(*fraction)` × 容量，自己的随机数，不影响其他抽签。
- 质量 = FM 空重 + 燃油 + 导弹（`fraction` 取代 `mass_factor`）。导弹质量取 missile_sim 档案的 `geometry.initial_mass_kg`（AIM-120A 156.5 kg、AIM-120C-5 161.5、R-77-1 190、PL-12 198），档案没有时用 `MISSILE_MASS_KG` = 150 kg（D）。挂架、机炮弹、飞行员、滑油不计。
- 发射一枚，质量减去这枚导弹；补弹时加回。
- 飞行模型的质量每差 `FUEL_MASS_STEP_KG` = 10 kg 才更新一次（`ManeuverModel` 的迎角缓存键里没有质量，所以每次更新清空迎角缓存；10 kg 在 1 万 kg 以上的飞机上是 0.1% 以内，D）。`OwnObs.mass_kg` 给的是精确值。

### 2.3 消耗（`flight.FuelTank.burn`，每 tick 在 `Aircraft._commit` 里）

- 流量 = Σ 各发动机：比油耗(油门) × 该发动机当前推力（kgf）/ 3600，kg/s。
- 油门用本 tick 结束时的发动机状态 `engine_percent`（已含发动机响应延迟），推力用同一状态的高度、真空速（`ManeuverModel._thrusts` + `JetEngine.blend`，与受力计算同一推力律）。
- 比油耗随油门插值：0% = idle，50% = half，100% = full（军用），100% 以上在加力段（到该发动机 WEP 模式的油门，通常 110%）从 full 线性过渡到 WEP，与推力从军用过渡到全加力的规律相同（D）。
- 推力表外（这一 tick 按弹道下落）不耗油。停在地上不耗油（发动机关着，D）。

### 2.4 耗尽（flameout）

- 油耗尽时推力为 0：飞行模型换成 `flight.Flameout`（`ManeuverModel` 的子类，`forces_at_aoa` 推力恒为 0，升力、阻力不变），升力、操纵照常，飞机滑翔，最后坠地（死亡原因 `crash`）或滑翔到机场落地。
- 事件 `flameout`（`plane`、`altitude_m`、`speed_mps`、`ias_kmh`），每次耗尽一次；`info["events"]["flameout"]`（全体飞机，只在设了 fuel 时有这个键）。

### 2.5 机场（airfield）

补弹（落地满 `turnaround_s`）时同时加油到开局载油、导弹质量加回，发动机恢复（`Flameout` 换回 `ManeuverModel`）；`rearm` 事件多 `fuel_kg`。起飞时新建的 `Aircraft` 沿用同一个油箱对象。

### 2.6 观测、回放、快照

- `OwnObs`（原始本机观测）新增 `fuel_kg`、`fuel_fraction`（剩油 / 开局载油）、`mass_kg`（空重 + 燃油 + 导弹）、`bingo`。不设 fuel 时为 `None / None / None / False`。**没有加进 actor 的 `own_vector`**（宽度不变；由另一位负责）。
- 回放：header 的 `plane_columns` 末尾加 `fuel_kg`，每帧 plane 行末尾是剩油（kg，0.1 kg）；header 多 `fuel`（设置值 + `loads`：每架 [开局载油, 油箱容量]），`planes[*].mass_kg` 是含油的开局质量。不设 fuel 时 header、帧都不变。
- 快照 / 恢复：油箱在 `Aircraft.fuel` 上，随 `MatchEnv.snapshot()` 深拷贝；`Flameout` 是模块级类，可拷贝也可 pickle。

### 2.7 脚本（bingo）

`OwnObs.bingo` 已经给出，但脚本的阶段逻辑在 `archetypes.py`（不归本改动）。需要的改动（`Pilot.decide`，三处，同一个条件 `own.missiles <= 0` 换成 `own.missiles <= 0 or own.bingo`）：

```python
if self.phase == "home" and own.missiles > 0 and not own.bingo:          # 补完弹 / 加完油才离开 home
if (own.missiles <= 0 or own.bingo) and self.phase not in ("home", "evade") and self.support_from is None:
self._set_phase("home" if own.missiles <= 0 or own.bingo else {...}.get(p.archetype, "recommit"), now)   # 规避结束
```

不设 fuel 时 `own.bingo` 恒为 False，行为不变。设了 airfield 时 bingo 的脚本回去落地、加油、再起飞；没设 airfield 时它回出生点上空盘旋，直到耗尽滑翔坠地。

### 2.8 每机型数据（`tanks: "max"` / `"internal"`；稳态，平飞，未计爬升与机动）

| 机型 | 空重 | 满油 max / internal | 质量（45% / 100%，含导弹） max；internal | 原来 1.15–1.45 × 空重 |
|---|---|---|---|---|
| F-15C 金鹰（8 × AIM-120A） | 13,470 | 15,795 / 6,103 | 21,830 / 30,517；17,468 / 20,825 | 15,490–19,532 |
| 苏-30SM2（12 × R-77-1） | 18,800 | 9,400 / 9,400 | 25,310 / 30,480 | 21,620–27,260 |
| F-16C Block 50（6 × AIM-120A） | 9,031 | 6,398 / 3,305 | 12,849 / 16,368；11,457 / 13,275 | 10,386–13,095 |
| 歼-10C（8 × PL-12） | 9,045 | 5,945 / 2,900 | 13,304 / 16,574；11,934 / 13,529 | 10,402–13,115 |
| 台风 AESA（8 × AIM-120B） | 11,220 | 7,110 / 4,650 | 15,671 / 19,582；14,564 / 17,122 | 12,903–16,269 |

流量（kg/s，全机）与满油可飞时间（分钟，max；括号内 internal）：

| 机型 | 8 km、0.9 Ma 军用 | 8 km、0.9 Ma 加力 | 1 km、0.8 Ma 军用 | 1 km、0.8 Ma 加力 |
|---|---|---|---|---|
| F-15C 金鹰 | 1.14 → 230 (89) | 4.07 → 65 (25) | 2.37 → 111 (43) | 7.08 → 37 (14) |
| 苏-30SM2 | 2.04 → 77 | 4.21 → 37 | 3.49 → 45 | 7.00 → 22 |
| F-16C Block 50 | 0.58 → 185 (95) | 5.93 → 18 (9.3) | 1.22 → 88 (45) | 11.18 → 9.5 (4.9) |
| 歼-10C | 1.06 → 94 (46) | 1.88 → 53 (26) | 1.80 → 55 (27) | 3.51 → 28 (14) |
| 台风 AESA | 1.29 → 92 (60) | 3.71 → 32 (21) | 2.19 → 54 (35) | 6.01 → 20 (13) |

慢车约 0.01–0.03 kg/s，85%（脚本巡航）约为军用的 60–70%。策略的速度头 0 就是 110%（全加力）。

## 3. 回场进近（`airfield.approach_m` 默认 21 km，原 15 km）

用户（2026-10-07）：进近（不含飞到机场附近的路程）要约 90 s，原来最后 15 km 只要 55–75 s。只改了默认半径，进近飞法（`intent.py`）没动。实测见 `docs/airfield_rearm_spec.md` 第 9 节：21 km 时策略（全加力进场）76–86 s，脚本（巡航进场）90–103 s。

## 4. 测试（`tests/test_fuel_reward.py`）

- 键不设与设为 `None`：2v2 脚本局的观测、奖励、info、引擎日志、原始本机观测逐位相同；独立引擎回放逐行相同；坏值报错。
- `first_shot`：先开火的队友得一次助攻，击落导弹之后才开火的不得，同一对不得第二次；不设时两者都得。
- `launch_reward`：每枚策略飞机的发射记一次，脚本机没有；不设时为 0。
- `retarget_kill_reward`：改追导弹的击落记 0.5（含迟到战果），tallies 记击落；直接击落仍是 +1；不设时为 +1。
- 燃油：开局质量 = 空重 + 油 + 导弹；随消耗减小；发射减去导弹质量；加力比军用烧得快（3 种机型），慢车比军用少；耗尽后推力为 0、滑翔、仍能转弯；`flameout` 事件只发一次；bingo 标志；补弹加满并恢复发动机，起飞后沿用油箱；快照 / 恢复；回放列。
- 进近：两架（F-15C 金鹰、苏-30SM2）从 40 km 全加力回家，进入进近半径到落地 70–100 s。

## 5. 实现记录与未解决

- 逐位一致的核对：另做了一份把本改动全部去掉的副本，与当前代码在默认配置下对比哈希（观测、奖励、info、引擎日志、原始本机观测；回放逐行）：2v2 脚本局、4v4 脚本局（policy_count、自博弈、timeout_reward）、4v4 随机动作、独立 3v3 引擎回放，全部相同。airfield 局因默认 `approach_m` 改了会不同（预期）。
- 吞吐（估计，没有做整局对比）：`FuelTank.burn` 单独测每次约 8 µs（CPython，机器负载高时），4v4 每步 8 架 × 20 tick 约 1.3 ms，相对每步约 80 ms 约 +1.6%；另有每 10 kg 一次的迎角缓存清空。不设 fuel 时每 tick 只多一次 `self.fuel is not None` 判断。
- 加力比油耗两族差 2.5 倍（2.1 节）：F-16C Block 50 带 45% 的 max 油（2,879 kg）全加力在 8 km 约 8 分钟、1 km 约 4 分钟就耗尽；策略默认全加力，开 fuel 后这一族会大量 flameout，别的机型不会。要不要修正这个游戏数据是用户的决定。
- `tanks: "max"` 把副油箱算进去，F-15C 满油 30.5 t，比原来的随机质量上限 19.5 t 重得多。
- 质量更新有 10 kg 的台阶；结构过载上限（`Structure.limits`）随质量变化。
- 目标已死的导弹刚离轨时可能改追发射机自己并击落它（`Engagement._retarget`，已有行为，未改；测试里用 `retarget_dead=False` 避开）。`retarget_kill_reward` 只降低改追击落的奖励，不改这个行为。
