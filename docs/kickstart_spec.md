# 技术文档：教师模仿项（kickstart）与开局爬升教师（可选项）

（2026-10-07；分档系数与自适应 2026-10-08。代码：`wt_overlay/rl_env.py`（`TEACHERS`、`teacher_settings`、`missile_threat`、`climb_label`、`MatchEnv.observe/step`）、`rl/worker.py`（`_labels`、StepResult `"teacher"`）、`rl/rollout.py`（`Sampler.pending_teacher/_store_label`）、`rl/buffer.py`（`RoundBuffer.teacher/teacher_name/teacher_names`）、`rl/ppo.py`（`kickstart_ce`、`PPOTrainer.kickstart_coef/_kickstart_metrics`）、`rl/config.py`（`PPOCfg.kickstart`、`kickstart_spec`）、`rl/train.py`（`summary_line`）；测试 `tests/test_rl_env.py` 的 `TeacherTests`、`rl/tests/test_kickstart.py`。）

## 背景

4v4 里策略出生后一直停在约 2.5 km（1v1 能爬到约 11 km，看得见队友时就不爬）。对照实验：强制开局爬到 8 km，对固定顶级脚本胜率 60% → 78%（60 局配对，现行规则）。vertical 头的熵正常（约 0.25），但 `execution.vertical_mode: "angle"` 下爬到 8 km 要把同一个爬升选项连续保持 1–2 分钟，随机探索碰不到。用户批准用一个逐轮衰减的模仿项把开局爬升教给策略。以后同一机制还要带 "push" 教师（大离轴首发 + 快速补射，见 `docs/analysis_offboresight_push.md`），所以做成通用形式：教师名 → 每个决策若干个头的标签。

## 要求

全部是可选项。环境的 `teacher` 和训练的 `ppo.kickstart` 都不设时与改动前逐位一致：观测、奖励、事件、info、随机数序列，采样的动作 / log-prob / 奖励，PPO 更新后的参数，训练器状态（检查点不多存东西）。网络输入和输出头不变，现有检查点直接续训。

## 1. 环境：`teacher`（MatchEnv 配置）

`teacher: {教师名: 设置}`。现在只有 `climb`（`{}` 或 `None` 用默认值）：

| 键 | 默认 | 含义 |
|---|---|---|
| `target_m` | 8000 | 爬升目标高度 |
| `until_s` | 150 | 比赛时间早于它才给标签 |
| `steep_below_m` | 2000 | 角度模式：比目标低这么多以上用陡爬 |
| `min_speed_mps` | 250 | 低于这个速度不给（两档爬升在 250 m/s 以下本来就不爬） |

未知教师名、未知键、非数值、负数、`target_m ≤ 300`、`until_s = 0` 报错。

每个策略控制且不在 `frozen_ids`（历史局的冻结方）里的 agent，每次决策（`observe()` 里掩码算完之后；只读，不改状态、不抽随机数）条件全部满足时给一个标签：

- 条件：
  - 比赛时间 < `until_s`；高度 < `target_m` − 300 m（`CLIMB_DEADBAND_M`，与脚本的平飞死区相同）；速度 ≥ `min_speed_mps`；不在地面（airfield）。
  - 没有感知到导弹（`missile_threat`）：RWR 导弹告警、MAW（只在执行器 `maw_entities` 为真时算，actor 看不到 MAW 时不用它）、导弹尾焰、导弹标记，或实体列表里有这些（包括 `entity_memory_s` 的记忆实体、雷达识别为导弹的航迹）。导弹标记也包括队友的导弹：观测里分不出敌我，与执行器释放保持的规则一致。
  - vertical 头能重新选：该 agent 当前掩码 `masks["vertical"][0]`（view_mode 0 的两行）全为真，即不在 2 s 保持里。
- 标签：角度模式比目标低 `steep_below_m` 以上为 1（+25°），否则 2（+10°）；高度模式为 2（8 km 档；`target_m` 更接近 11 km 时为 1）。
- 输出：`info["teacher"] = {aid: {"vertical": 选项, "name": "climb"}}`，是本步返回的观测（下一次决策用的那个）的标签，只列有标签的 agent，都没有时为 `{}`。`reset()` 后的标签在 `env.teacher_labels`（同格式）。
- 配了多个教师时按 `TEACHERS` 的顺序，第一个条件满足的给标签（每个决策最多一个教师）；一个标签可以有多个头。
- 不设 `teacher`：info 里没有这个键，环境也没有 `teacher_labels` 属性。

实测（本地 `s2_4v4` 的 env 配置，`policy_count [4, 4]`、`time_limit_s 200`，6 局，前 150 s 内的决策）：

| 动作 | 有标签 | 不给的原因（占全部决策） |
|---|---|---|
| 脚本动作（脚本自己会爬） | 33% | 已在 7.7 km 以上 39%、慢于 250 m/s 13%、保持 9%、威胁 6% |
| 脚本动作，vertical 强制平飞（相当于现在停在 2.5 km 的策略） | 56%（全是 +25°） | 威胁 25%、保持 18%、慢 1% |

占整轮决策的比例还要乘上 150 s / 平均局长。

## 2. 传输与缓冲区

- worker：StepResult 多一个可选键 `"teacher"` `{aid: {头: 选项, "name": 名}}`。局继续时取 `info["teacher"]` 里属于 `obs` 的 agent；新局（包括 `init`）取 reset 后的 `env.teacher_labels`。没有标签时没有这个键。
- 采样端：每条流的待决观测带着它的标签（`pending_teacher`），决策那一步写进 `buf.teacher[步, 头]`（int16，−1 = 无）和 `buf.teacher_name[步]`（int8，`buf.teacher_names` 的下标，−1 = 无）。冻结方的流丢掉标签（它们的步本来就不存）；burn-in 前缀、padding、settle 扩展的空位都是 −1。头名不在 `spec.HEAD_NAMES` 或选项不是整数时报错。
- 标签不影响采样：同种子下动作、log-prob、奖励与没有标签时相同（测试）。

## 3. PPO：`ppo.kickstart`

`ppo.kickstart = {"coef": 0.5, "decay_rounds": 40, "start_round": None}`；不设（`None`）为关，`{}` 全用默认值。`Config.validate` 和 `PPOTrainer` 都检查（`kickstart_spec`）：coef 为 ≥ 0 的有限数，decay_rounds 为 ≥ 1 的整数，start_round 为 `None` 或 ≥ 0 的整数；另外只允许可选的 `tiers`（3.2，代替 coef）和 `adapt`（3.3）。

- 损失：actor loss += coef_t × 小批次里有合法标签的有效步上 Σ_被标注的头 −log π(标签 | 状态, 已采样的前面各头) 的平均。合法 = 标签在存储的掩码里、按这一步实际采样的 view_mode、maneuver_ref 等前面的头选出的那一行里为真（`HeadOut.eff`），不合法的跳过。所以 vertical 标签只在策略采样了 view_mode 0 的步上起作用（view_mode 1/2 时 vertical 只有一个选项）。只有一个合法选项的头 −log π = 0，没有梯度。
- 系数：coef_t = coef × max(0, 1 − (round − start) / decay_rounds)，start 之前为 0。`start_round` 为 `None` 时取第一次以这个配置更新的轮次，存在训练器状态 `kickstart_start` 里，续训接着衰减；配置里给了 `start_round` 时以配置为准。关掉期间状态里不存，再打开从当时的轮次重新开始。critic 预热轮也计入。
- coef_t = 0 时不加这一项（参数与关时一样），指标照算。缓冲区有标签而没配 kickstart 时只记指标（`coef` 为 `None`），更新不变。
- PPO 的裁剪只作用于 PPO 项，kickstart 项不裁。

### 3.1 会不会被 KL 控制挡住：会

- kickstart 项和 PPO 项在同一个 actor loss 里，按小批次一起做或一起不做：
  - `kl_mode: "skip"`（s2_4v4 用的）：每个小批次更新前算联合动作对行为策略的 k3 KL，超过 `target_kl_skip` 的小批次整个跳过，kickstart 项一起丢；已做小批次的平均 KL 超过 `target_kl` 后本轮 actor 停止，余下的也没有了。
  - `kl_mode: "stop"`：第一个超过 `target_kl` 的小批次停掉本轮 actor。
- kickstart 自己就推高 KL：把有标签步上 vertical 的分布拉向标签，就是让新策略离开行为策略。所以每轮能拉多少由 KL 预算决定，并且占用 PPO 本身的预算。粗算：有标签步上 P(标签) 从 0.2 到 0.4，这些步上 KL ≈ 0.09，标签占 15% 时联合 KL ≈ 0.014，与 s2_4v4 的 `target_kl 0.03` 同一量级。KL 是对每轮的行为策略算的，不跨轮累积：被挡的是每轮的步长，不是方向，多轮下来照样能拉过去。后几个 epoch、以及标签集中的小批次（开局片段）最容易超过 `target_kl_skip`。
- `ppo.head_kl`（现在只在 weapon 上）不约束 vertical，不直接挡；共享躯干的变化会让 weapon 的 KL 变大，它的自适应系数可能上升。不要把被标注的头放进 `head_kl`（两者正好相反）。
- 熵奖励（包括 `ent_floor`）把 vertical 往均匀推，与 kickstart 反向，但 coef 0.5 远大于熵系数（0.003–0.01 × 头倍数）。
- 怎么看：`kickstart.applied_share`（有标签的步进入真正做了更新的小批次的比例，按 epochs 计，1 = 全用上）、`kickstart.kl_skipped_minibatches`（被跳过的、含标签的小批次数），以及 `kl_skipped_minibatches`、`actor_stopped_at_minibatch`、`kl_target_head.vertical`。`applied_share` 长期很低时调低 coef 或调高 `target_kl_skip`。

### 3.2 分档系数（`tiers`，2026-10-08，可选）

`ppo.kickstart.tiers = [[系数, 比例], ...]`，例如 `[[0.1, 0.10], [0.25, 0.25], [0.4, 0.65]]`：10% 的 actor 小批次用 0.1，25% 用 0.25，其余 65% 用 0.4。它代替单一的 `coef`（两个都写报错）；不写 `tiers` 时与只有 `coef` 时完全相同（逐位，测试对照了改动前的代码）。

- 检查：非空列表，每项是两个数，系数 ≥ 0，比例 ≥ 0，比例之和为 1（±1e-6）。
- 抽签：每个 actor 小批次（critic 预热轮除外；被 KL 跳过的、actor 停止之后的小批次也抽，所以抽签序列不受 KL 事件影响）按当前比例抽一档，用训练器自己的一个专用生成器 `ks_gen`（种子 `run.seed + 29`，状态存在训练器状态 `kickstart_gen` 里，续训接着抽）。不用打乱小批次的 `self.gen`：开不开分档，小批次顺序都一样，只有一档 `[[c, 1.0]]` 时与 `coef: c` 逐位相同。
- 这个小批次实际用的系数 = 该档系数 × s × decay_t。decay_t 就是原来的线性衰减（start 轮为 1，`decay_rounds` 轮内降到 0），s 见 3.3（不开 adapt 时为 1）。系数为 0 的档不加这一项。
- 指标里的 `coef` 是本轮的期望系数 Σ 比例 × 档系数 × s × decay_t。

### 3.3 自适应（`adapt`，2026-10-08，可选，第一版）

`ppo.kickstart.adapt = {"skip_target": 0.1, "step": 0.8, "min_scale": 0.2, "max_scale": 1.0}`（`{}` 用这些默认值；不写为关）。检查：skip_target、step 在 (0, 1)，min_scale > 0，max_scale ≥ min_scale。

用户以后要让"比例"和"系数范围"都自适应，这里先做一个简单版本，规则预计还要改：

- 一个倍数 s 乘在每一档系数上（没有 tiers 时乘在 `coef` 上），起点 1.0（夹在 [min_scale, max_scale] 里），存在训练器状态 `kickstart_scale`。
- 每轮更新后，若本轮 kickstart 起作用（期望系数 > 0、有标签、不是 critic 预热轮），算 skip_share = 被 KL 控制挡掉的 actor 小批次 / 本轮全部 actor 小批次。挡掉 = `kl_mode "skip"` 跳过的，加上 actor 停止的那个小批次及其后的所有小批次。所有小批次都算，不只是含标签的（kickstart 改的是共享参数，所有小批次的 KL 都受影响）。
  - skip_share > skip_target：s ×= step；并把 0.05 的比例从系数最高的档挪到系数最低的档。
  - skip_share < skip_target / 2：s ÷= step；并把 0.05 挪回去（从最低档到最高档）。
  - 介于两者之间：不变。
  - s 夹在 [min_scale, max_scale]（默认上限 1.0：配置的系数就是上限）。挪出的一档至少留 0.05（不够 0.05 的只挪到 0.05 为止，已经低于 0.05 的不挪），总和保持 1；中间各档不动；最高、最低系数相同时不挪。
- 当前比例存在训练器状态 `kickstart_shares`，连同它们所属的配置档位；续训时只有配置的 `tiers`（系数和比例）没变才恢复调整后的比例，改了配置就从新配置的比例开始。s 只要开着 adapt 就恢复（夹在新的范围里）。
- 衰减结束（decay_t = 0）后不再调整。
- 已知粗糙之处（待用户细化）：只看跳过率，不看 agree/ce 的进展；中间档不参与；挪动步长固定 0.05；s 和比例同时动，作用会叠加。

## 4. 指标与日志

`metrics.jsonl` 每轮 `kickstart`（配了 kickstart 或缓冲区有标签时才有）：

| 键 | 含义 |
|---|---|
| `coef` | 本轮用的 coef_t；没配 kickstart 时为 `None` |
| `start_round` | 衰减起点 |
| `labelled_steps`、`label_share` | 本轮有标签的有效步数、占有效步的比例 |
| `agree` | 有标签的步里，采样的动作在所有被标注的头上都等于标签的比例（行为策略） |
| `ce` | 做了更新的小批次（所有 epoch）里合法标签步的平均 −log π(标签)，按当时的参数 |
| `legal_share` | 这些小批次里有标签的步中标签合法的比例（主要看 view_mode） |
| `applied_share`、`kl_skipped_minibatches` | 见 3.1 |
| `by_teacher` | `{教师名: {labelled_steps, agree}}` |
| `decay`、`shares`（tiers） | 本轮的 decay_t、本轮用的比例 |
| `tiers`（tiers） | 每档 `{coef（配置的档系数）, share（本轮比例）, drawn, applied, skipped, stopped, ce}`：抽到的小批次数，其中做了更新的、被 KL 跳过的、因 actor 停止没做的，及做了更新的小批次里合法标签步的平均 −log π |
| `scale`、`adapt`（adapt） | 本轮用的 s；`{skip_share（本轮未起作用时 None）, scale_next, shares_next}` |

train.log 每轮一行（只在配了 kickstart 时）行尾加 `ks <coef_t> lab <label_share> agree <agree> ce <ce> skip <kl_skipped_minibatches>`；开 adapt 时 `ks <coef_t> s<s> lab ...`；有 tiers 时 `ks t[0.1:10% 0.25:25% 0.4:65%] s1.00 d0.95 lab ...`（档系数:本轮比例、s、decay_t）。

## 5. 用法

续训的运行目录里 `config.json` 加两处（`--set` 改不了 `env.config` 里的单个键；不给 `--config` 时训练读运行目录的 `config.json`，worker 用其中的 `env.config` 建环境）：

```json
"env":  {"config": {"teacher": {"climb": {"target_m": 8000, "until_s": 150, "steep_below_m": 2000, "min_speed_mps": 250}}}},
"ppo":  {"kickstart": {"coef": 0.5, "decay_rounds": 40, "start_round": null}}
```

`ppo.kickstart` 也可以用 `--set "ppo.kickstart={'coef': 0.5, 'decay_rounds': 40}"`；分档加自适应：`--set "ppo.kickstart={'tiers': [[0.1, 0.10], [0.25, 0.25], [0.4, 0.65]], 'decay_rounds': 40, 'adapt': {'skip_target': 0.1}}"`。不写 `start_round` 时衰减起点取检查点里的 `kickstart_start`（从单一系数换成分档时也一样，衰减接着走）。看 `kickstart.agree`（应上升）、`ce`（下降）、`label_share`（策略爬上去后下降）、`applied_share`、`action_freq_active.vertical` 和对脚本胜率；衰减到 0 以后 `agree` 仍在算，用来看爬升是否保持。

## 6. 局限

- 只在策略采样 view_mode 0 的步上教；4v4 里 view_mode 1/2 用得多时 `legal_share` 低，学得慢。
- 只教 7.7 km 以下、开局 150 s 内往上爬；到高度后不再给标签，是否保持高度看回报。
- 队友导弹的导弹标记也算威胁，队友开火早时少给一些标签。
- 1v1 局（`team_size_mix`）、自博弈和历史局的当前策略一方同样给标签（自博弈两方都是当前策略，都给）。
