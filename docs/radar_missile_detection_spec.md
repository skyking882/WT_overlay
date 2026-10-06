# 技术文档：机载雷达探测来袭导弹（WT_overlay 交战环境）

## 背景

WT_overlay 的交战环境（`wt_overlay/engagement.py`、`wt_overlay/sensors.py`）里，机载雷达只探测敌方飞机。导弹从不出现在雷达上：`engagement.py:1068-1075` 只把飞机放进 `truths`，再交给 `RadarSensor.update`。

所以策略（以及脚本）只能从这几处知道有弹来：

- RWR 告警：导弹导引头开机照射之后；
- 导弹逼近告警（MAW），有这种设备的飞机才有；
- 在视野里看到尾焰或烟迹，或者导弹标记。

### 游戏里的行为（C 级，用户 2026-10-06）

- **机扫雷达也能看到导弹**，不只是 AESA。
- 没有 NCTR（目标类型识别）时，分不出哪个是导弹、哪个是飞机，只能靠接近率区分。B 显上目标速度线越长，接近率越大。
- 有 NCTR 的雷达大多能直接识别导弹，图标和飞机不同。
- 大约 **70 km** 就能看到导弹。
- 导弹能被锁定，甚至能用导弹去打导弹（"对弹"）。**但现阶段不让策略学对弹**，必须有开关控制。

### 游戏数据（A 级）

顶级房 19 种飞机的雷达文件都有目标类型表 `targetTypeId`，其中一项是：

```
{"name": "hud/rocket", "targetPropulsion": {"type": "rocket"}}
```

即按火箭推进识别为导弹。机扫的 CAPTOR-M、N011M、N035E 也有，不是 AESA 专有。

文件里**没有**导弹被照射时的 RCS，也没有对导弹的专门探测距离。导弹文件里的 `radarSeeker.receiver.rcs` 是它自己导引头的参考 RCS，不是它被照射时的 RCS。

## 规则

- 新机制一律**可选**：`MatchEnv` 配置 `radar_sees_missiles`（缺省 `False`），传到 `Engagement(..., radar_sees_missiles=...)`。不启用时，所有现有结果**逐位不变**。
- 先记录 `.venv` 下 `tests/` 和 `rl/tests` 的现有结果（目前全部通过），改完后全部通过，并新增测试。
- 不提交，不改无关代码。

## 需求

### 1. 雷达目标里加入在飞的导弹

- `engagement.py` 构造 `truths` 的地方（约第 1068 行），给每架飞机的雷达加上**敌方在飞导弹**的 `TargetTruth`，即 `self.missiles` 里 `not m.done`、发射者是对方队伍的。
  - 位置、速度用导弹当前状态（`m.pos_enu`、`m.vel_enu`）；
  - RCS 用新常量 `MISSILE_RCS_M2`（D）。
- 默认值按"典型顶级房雷达约 70 km 探测到导弹"标定：雷达探测距离与 RCS 的四次方根成正比，`MISSILE_RCS_M2 = σ_ref·(70 km / R_ref)⁴`。以 N035E 或 APG-63(V)3 的搜索波形为参考，在报告里写明取值和依据。
- 导弹沿用现有的全部探测规则：扫描体积、波束、多普勒滤波、主瓣杂波 notch、TWS 和 STT 逻辑。
- 导弹也占用 TWS 航迹表的名额，和飞机一样受 `targets_max` 限制。这是有意的：真实情况下导弹航迹会挤占列表。

### 2. 识别（NCTR）

- 雷达文件有 `targetTypeId`，且其中含 `targetPropulsion.type == "rocket"` 时，视为能识别导弹（A）。顶级房全部满足。
  - 由 `scripts/import_datamine_units.py` 读出，存入 `data/units/radars.json` 的新字段 `identifies_missiles`。
- 能识别时，导弹航迹标为导弹类型：`Entity.aircraft = "missile"`，或新增等价字段。
  - 识别距离沿用现有 NCTR 的规则；如果目前没有，就设一个 D 级常量，与飞机类型识别一致。
- 不能识别时，导弹航迹与飞机航迹外观相同，只能从接近率、速度区分。观测里本来就有这两项（`closure`、`speed`）。

### 3. 观测

- 导弹航迹以现有 `kind='radar'` 实体进入观测。
- 识别出的导弹在 `aircraft_type` 位用一个保留值（例如 `-1` 或单独的类别位），不和飞机目录序号冲突。
- **实体宽度不变**，方便从现有存档接着训。

### 4. 动作约束：现阶段不允许对弹（重要）

- 新增配置 `allow_missile_targets`（缺省 `False`）。为 `False` 时：
  - `masks_for`（`wt_overlay/rl_observation.py:222`）里，`target` 指针和 `weapon` 头**排除**所有导弹航迹，**包括没识别出类型的**。也就是用真值判断，不让策略通过"没识别出来"绕过去；
  - 雷达的 STT 也不能锁导弹航迹。
- 脚本飞行员：不把导弹航迹当攻击目标，但可以当威胁信息用。比如看到高速接近的航迹就提前规避，这一项可选。
- 以后放开对弹，只改这个开关，并单独评估。

### 5. 记录

- 回放帧里导弹被哪些雷达跟踪：每个导弹记录 `tracked_by` 列表，看板上能显示"被雷达发现"。
- `info["events"]` 新增 `radar_missile_tracks`：本步新建的导弹航迹数。

## 验收

1. 不启用时逐位不变，现有测试全部通过。
2. 新测试：
   - 一枚迎头导弹在参考雷达的扫描范围和多普勒通带内，约 70 km 处（按标定）开始被探测；
   - 有 NCTR 时标为导弹，没有时不标；
   - `allow_missile_targets=False` 时，导弹航迹在 `target` 和 `weapon` 掩码里不可选，STT 不锁导弹；
   - 导弹占用 TWS 名额。
3. 行为检查：用 `rl.eval_replay --stats` 跑现有策略，打开与不打开各一组，确认对局正常、没有对弹、没有非法动作。可以预期策略会更早规避，但现有存档没学过这些输入，不要求胜率变化。

## 校准（D 参数）

- `MISSILE_RCS_M2`：用户说约 70 km 能看到。建议在游戏里用几种雷达测一下首次出现导弹航迹的距离：迎头、侧向、导弹在发动机工作中与已熄火。
- NCTR 识别导弹的距离：同样在游戏里看图标什么时候从未知变成导弹。
