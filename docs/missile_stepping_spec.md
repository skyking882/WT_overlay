# 技术文档：missile_sim 可逐步推进的导弹仿真

## 背景

本文档描述需要在 `/Users/skyking/Documents/missle_sim`（注意拼写 "missle"）中完成的改动。

该仓库用纯 Python 复现《战争雷霆》（War Thunder，商业电子游戏）里空空导弹的游戏内飞行模型，参数全部来自游戏客户端文件，见仓库 `CLAUDE.md` 的"项目性质"一节。它被兄弟仓库 `/Users/skyking/Documents/WT_overlay` 用来给玩家计算游戏内提示。

WT_overlay 正在搭建一个多机交战的游戏环境：16 对 16、多枚导弹同时在飞、双方都由飞行模型驱动，用来训练和评估游戏内辅助策略（设计见 `WT_overlay/docs/rl_design.md`）。目前 missile_sim 的游戏导弹仿真有三个限制，阻碍它接入这个环境：

1. **只能一口气跑完。** `SurfaceRuntime.simulate()` 是一个完整的 while 循环，没法和其他物体共用一个时钟逐步推进。
2. **发射速度绑在机头方向上。** 速度等于机体 x 轴乘发射速度，无法表达转弯中或带迎角发射时速度方向与机头不一致的情况。
3. **发射方雷达按发射时速度直线外推。** 发射后载机掉头、侧转，都不影响数据链是否中断。

## 仓库规则（必读：`missle_sim/CLAUDE.md`）

- 不要改控制节拍。surface runtime 是 1/48 s，构造时会检查；不要为提速改积分器。
- 新机制一律**可选**，不启用时，所有现有结果**逐位不变**（浮点数精确相等）。
- 仿真内核 `src/aim120_model/` 只用 Python 标准库。
- 涉及游戏 blkx 字段含义的主张，要登记到 `docs/BLKX_EVIDENCE_LEDGER.md` 并标证据等级。本任务是接口和机制改动，预计不需要登记。
- 工作区有大量用户未提交的改动：不要提交，不要回退、格式化或"顺手清理"无关代码，只改任务需要的文件。
- 测试：`.venv/bin/python -m pytest`（系统 python 没有 pytest）。改动前已有约 33 个失败（进行中的工作）。**改动前先记录失败测试的完整列表**，改完后失败列表必须完全一致，新测试全部通过。

## 现状（代码位置）

| 位置 | 内容 |
|---|---|
| `src/aim120_model/surface_runtime.py:44` | `class SurfaceRuntime(EffectiveSurfaceGame)` |
| `surface_runtime.py:93-101` | 构造 `creation`：位置 `[0, launch_altitude, 0]`，速度 `scale(axes(q)[0], launch_speed)`（第 97 行），四元数来自 `launch_pitch_deg` 和 `launch_heading_deg` |
| `surface_runtime.py:175` | `simulate(end_time, early_miss_s)`：主循环。每个节拍依次执行：`target.observe_missile`（如有）→ `truth = target.state_at(time)` → `provider.update` → 记一行 → 判断终止（引信、触地、最大距离、提前判脱靶、到时）→ `update_control` → 按发动机分段切分子步积分 → `event_candidates(..., target.state_at(next_time), ...)` 处理步内事件 |
| `surface_runtime.py:215-227` | 可选的"提前判定脱靶"捷径（`early_miss_s`） |
| `surface_runtime.py:257` | `simulate_surface(...)`：校验场景、建 runtime、应用可选项（clutter_model、cw_on_clear_beam、multipath_gain、launcher_radar_gimbal_deg、require_seeker_lock），调用 `simulate`，汇总 |
| `src/aim120_model/observation.py:382` | `SensorTrackProvider.enable_launcher_radar(position, velocity, gimbal_deg)`：载机按发射速度直线飞 |
| `observation.py:394` | `_launcher_loses_target`：目标出了载机雷达的范围锥，或处于载机视角的杂波 notch，返回原因 |
| `observation.py:411` | `_refresh_datalink_state`：一旦返回原因，数据链永久断开 |
| `src/aim120_model/public_api.py:320` | 公共入口 `simulate(...)`，含 `launcher_radar_gimbal_deg`、`require_seeker_lock` 的校验 |
| `src/aim120_model/events.py:80` | `event_candidates` |

## 需求

### 1. 可逐步推进，原接口结果逐位不变

把 `SurfaceRuntime.simulate` 的循环拆开，建议接口：

```python
runtime.begin(end_time, early_miss_s=None)   # 初始化循环状态
runtime.step()            # 推进一个控制节拍，或推进到终止事件
runtime.time_s            # 当前时间（发射后秒数）
runtime.done              # 是否已终止
runtime.event             # 终止事件类型（与现有 event_type 一致）
runtime.state             # 当前导弹状态（位置、速度、姿态等）
runtime.result()          # 与现在 simulate() 的返回值完全相同
```

`simulate()` 改成 `begin` + 循环 `step` + `result` 的薄封装。

**验收（金标准比对）**：
- **改动前**，用公共入口跑约 20 个有代表性的场景，保存完整输出（summary 加全部逐行样本），存到仓库外（例如 `/private/tmp/...`），不要往仓库里加大文件。场景覆盖：
  - 多个导弹：missiles/ 下的 us_aim_120c_5、cn_pl12、su_r_77_1、swd_rb99；
  - 观测模式：ideal_truth 和 sensor_track；
  - 开、关爬升弹道；
  - 杂波：`clutter_model='look_down_angle'` 配 `clutter_min_depression_deg=2`，以及 `cw_on_clear_beam`；
  - `launcher_radar_gimbal_deg=60`、`require_seeker_lock=True`、`early_miss_s=2`；
  - 转弯目标（场景支持的话）；箔条（`aim120_model.chaff.chaffing_factory`，方便的话）。
- **改动后**重跑，逐字段**精确相等**。
- 留一个精简版作为永久测试：在几个场景上，`simulate()` 与手动 `begin/step/result` 的结果精确相等。

### 2. 发射状态与机头方向分开（可选）

新增从真实发射瞬间状态创建导弹的入口，例如：

```python
create_surface_missile(
    profile, *,
    launch_position_m,        # 仿真坐标系 (x, y 向上, z)
    launch_velocity_mps,      # 速度矢量，可与机头方向不一致
    launch_pitch_deg, launch_heading_deg,   # 机体姿态（至少俯仰、航向）
    target,                   # 见第 3 条
    observation_mode='sensor_track', loft=True,
    clutter_model=None, clutter_min_depression_deg=None, cw_on_clear_beam=False,
    launcher_support=None,    # 见第 4 条
    require_seeker_lock=False,
    end_time_s=None,
) -> SurfaceRuntime           # 已 begin，可直接 step()
```

要求：
- 走同一套 `SurfaceRuntime` 代码路径。
- 速度沿机头、其他条件相同时，结果与旧接口逐位相同（写测试）。
- 文档写清坐标系，以及导弹时间从发射时刻 0 开始；世界时间由调用方换算。

### 3. 目标接口（共用时钟的约定）

目标是任何实现以下方法的对象：
- `state_at(time_s) -> aim120_model.target.TargetState`
- 可选的 `observe_missile(time_s, position, velocity)`

`step()` 从 t 推进到 t+dt 时需要目标在 t 和 t+dt 两个时刻的状态，因为 `event_candidates` 会调用 `state_at(next_time)`。所以调用方必须**先推进飞机、再推进导弹**。把这条约定写进 docstring，并在测试里写一个保存历史的目标代理，证明按共用时钟逐步推进可行。

### 4. 发射方支援状态改为回调（可选）

现有的直线外推保留不变，另外接受一个回调：

```python
launcher_support(time_s, target_truth) -> str   # "" 表示仍在支援；非空字符串为中断原因
```

语义与 `_launcher_loses_target` 相同：第一次返回原因后，数据链永久断开。载机雷达是否还跟踪着目标，由 WT_overlay 的雷达模型判断，再通过这个回调告诉导弹。现有 `launcher_radar_gimbal_deg` 的行为必须完全不变。

### 5. 多枚导弹同时推进

同一进程里交替推进多个实例，彼此不能有共享的可变状态。要检查的地方：模块级缓存或全局变量、被原地修改的 profile 字典等。

**测试**：两个不同导弹交替 `step()`，结果分别与单独运行时精确相等。

### 6. 提前判脱靶只在调用方传入 `early_miss_s` 时生效

连续交战不使用这个捷径，因为双方之后可能回头。

### 7. 性能

测一个实例每次 `step()` 的耗时（微秒），分别用 CPython（`.venv/bin/python`）和 PyPy（`/opt/homebrew/bin/pypy3.11`，内核是纯 Python，应该能跑）。

## 交付

- 接口说明：函数、参数、约定。
- 改动文件清单，每个附一句原因。
- 金标准比对结果。
- 改动前后的失败测试列表，应一致。
- 新增测试。
- 耗时数据。

## 不在本任务范围

- WT_overlay 侧的交战管理器、雷达和 RWR 模型（另有工作在做：`WT_overlay/wt_overlay/sensors.py`）。
- 毁伤模型（目前引信触发即算击落）。
- 任何导弹参数或游戏字段语义的修改。
