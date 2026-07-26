# Mini3 FB 与 Mini3-BeyondMimic 域随机化、地形配置对比

## 1. 结论

当前 UFO 的 Mini3 FB 训练配置与 `mini3_lab` 中注册为 `Mini3-BeyondMimic` 的 task **不一致**。

- 两边的基础仿真频率一致：physics `500 Hz`、control/policy `50 Hz`、decimation `10`。
- 域随机化并非完全不同：push 的时间间隔与水平速度范围、base COM 的作用对象和执行时机、三组 actuator command delay 是一致的。
- 主要差异集中在 contact friction/material、mass、COM 范围、默认关节偏置、PD gain、joint friction/armature、motor strength、IMU delay 和 reset 状态扰动。
- 地形差异很大：UFO 当前实际创建纯平面且没有 terrain curriculum；BeyondMimic 默认创建 wave/random-rough 高度场并启用 tracking-success curriculum。
- 当前源码与已有 Mini3 FB checkpoint 的 IMU delay 配置也不相同：当前源码关闭，最近一份完整训练快照开启。

按本文表格的比较粒度统计：

| 范围 | 完全一致 | 参数/时机不一致 | 仅 UFO 有 | 仅 BeyondMimic 有 |
| --- | ---: | ---: | ---: | ---: |
| 有效域随机化与 sim-to-real 扰动，共 20 项 | 6 | 9 | 2 | 3 |
| 有效地形设置，共 8 项 | 1 | 7 | 0 | 0 |

统计中的“一致”要求对象、范围、启用状态和主要执行时机都相同。仅仅机制相似但范围不同，会记为“不一致”。UFO 平面配置中未生效的 rough-terrain 数值不计为有效地形一致项。

## 2. 对比范围与配置来源

检查日期：2026-07-26。

UFO：

- revision：`9e58f409911797f6d9582a17503d3fb00caddf48`
- Mini3 robot/training override：`configs/robots/mini3.yaml`
- 共享 domain-rand：`humanoidverse/config/domain_rand/domain_rand.yaml`
- 共享 terrain：`humanoidverse/config/terrain/terrain_locomotion_plane.yaml`
- MJLab 环境构建和事件映射：`humanoidverse/agents/envs/humanoidverse_mjlab.py`
- FB agent 本身不改 terrain/domain-rand；`--agent fb` 与其他 agent 共用上述环境配置。

BeyondMimic：

- repository：`/home/amax/Desktop/robot/mini3_lab`
- revision：`e830ddee1451f5a03eee3f832ca4bb2529c38172`
- task id `Mini3-BeyondMimic` 注册到 `Mini3BeyondMimicEnvCfg`
- Mini3 配置：`robolab/tasks/manager_based/beyondmimic/mini3_beyondmimic_env_cfg.py`
- 继承的 event 和 plane material：`robolab/tasks/manager_based/beyondmimic/beyondmimic_env_cfg.py`
- actuator delay：`robolab/assets/robots/roboparty.py` 中的 `MINI3_IMPLICIT_CFG`
- terrain curriculum：`robolab/tasks/manager_based/beyondmimic/mdp/curriculums.py`

本报告比较的是 task id `Mini3-BeyondMimic`，不是 `Mini3-BeyondMimic-RealMotor`。标准 task 使用 `MINI3_IMPLICIT_CFG`；UFO 当前则启用了自己的 Mini3 real-motor chain。本文只把和域随机化直接相关的 actuator delay 纳入逐项统计，基础 actuator dynamics 的差异不计入表中。

检查时本机没有正在运行的 `humanoidverse.train` 进程，因此“当前 UFO 配置”以当前源码为准。最近一份包含完整配置的 Mini3 FB 训练记录为：

```text
runs/ufo_fb_lafan1_mini3_real_motor_finetune_selfcollision_new/config.json
```

该快照来自 `/home/wzk/UFO`，不能替代当前 checkout，但可用于判断已有 checkpoint 当时实际使用的参数。

域随机化表假定训练没有传入 `--disable-dr`。最近完整快照中的 `disable_domain_randomization=false` 符合这个假定；如果启动时使用 `--disable-dr`，UFO 表中由 `domain_rand` 控制的 event 和 command-delay randomization 会被关闭。

本文把 actuator/IMU delay 和 reset state perturbation 一并视作 sim-to-real 扰动进行统计。Observation noise 单独说明但不计入 20 项 domain-rand 统计，reward 和 motion sampling algorithm 不在本报告范围内。

## 3. 共同的仿真时基

| 项目 | UFO Mini3 FB | Mini3-BeyondMimic | 结果 |
| --- | --- | --- | --- |
| Physics step | `1 / 500 s = 0.002 s` | `sim.dt = 0.002 s` | 一致 |
| Control decimation | `10` | `10` | 一致 |
| Policy/control rate | `50 Hz` | `50 Hz` | 一致 |

因此，两边用“physics step”表示的 delay 可以直接按 `2 ms/step` 比较，不需要换算不同的仿真频率。

## 4. 域随机化详细对比

状态说明：

- **一致**：有效行为和数值一致。
- **不一致**：机制相似，但范围、对象、启用状态或采样时机不同。
- **UFO only / BM only**：只在一边有效。

| # | 比较项 | UFO Mini3 FB 当前有效配置 | Mini3-BeyondMimic 当前有效配置 | 状态 |
| ---: | --- | --- | --- | --- |
| 1 | Push interval | `[1, 3] s` | `[1, 3] s` | **一致** |
| 2 | Push 水平速度 x/y | x/y 均为 `[-0.5, 0.5] m/s` | x/y 均为 `[-0.5, 0.5] m/s` | **一致** |
| 3 | Push z/角速度 | z 不扰动；roll/pitch/yaw 均为 `[-0.5, 0.5] rad/s` | z 为 `[-0.2, 0.2] m/s`；roll/pitch `[-0.52, 0.52]`，yaw `[-0.78, 0.78] rad/s` | **不一致** |
| 4 | Base COM 对象与时机 | `waist_yaw_link`，startup | `waist_yaw_link`，startup | **一致** |
| 5 | Base COM 范围 | x/y `[-0.03, 0.03] m`，z `[-0.02, 0.02] m` | x `[-0.03, 0.10] m`，y `[-0.05, 0.05] m`，z `[-0.08, 0.08] m` | **不一致** |
| 6 | Robot contact friction/material | startup；全部 robot geom 的 MuJoCo friction axis 0 取绝对值 `[0.5, 2.2]` | startup；全部 robot body 的 static friction `[0.6, 1.6]`、dynamic friction `[0.6, 1.2]`、restitution `[0, 0.5]`，64 buckets | **不一致** |
| 7 | Link mass scale | startup；全部 robot body 乘 `[0.98, 1.02]` | startup；仅 `left_.*_link`、`right_.*_link` 乘 `[0.8, 1.2]` | **不一致** |
| 8 | Torso/base 附加质量 | `randomize_base_mass=False`，当前 MJLab 路径也没有独立实现 | startup；`waist_yaw_link` 加 `[-1.5, 1.5] kg` | **BM only** |
| 9 | Default joint position offset | 每个 episode reset；全部关节加 `[-0.02, 0.02] rad` 的 model/default offset | startup；全部关节 default position 加 `[-0.07, 0.07] rad` | **不一致** |
| 10 | 非脚踝 PD gain | 每个 episode reset；全部关节 Kp/Kd 均乘 `[0.75, 1.25]` | startup；非脚踝 Kp/Kd 均乘 `[0.8, 1.2]` | **不一致** |
| 11 | 脚踝 PD gain | 每个 episode reset；Kp/Kd 均乘 `[0.75, 1.25]` | startup；Kp 乘 `[0.8, 1.2]`，Kd 乘 `[0.3, 5.0]` | **不一致** |
| 12 | Motor strength/torque capacity | 每个 episode reset；连续/峰值 torque capacity 乘 `[0.9, 1.1]` | 没有对应 event | **UFO only** |
| 13 | Joint friction | 固定使用 robot YAML nominal 值，没有 joint-friction DR | startup；非脚踝乘 `[0.7, 1.3]`，脚踝乘 `[0.5, 2.0]` | **BM only** |
| 14 | Joint armature | 固定使用 robot YAML nominal 值，没有 armature DR | startup；非脚踝乘 `[0.7, 1.3]`，脚踝乘 `[0.5, 1.5]` | **BM only** |
| 15 | 4340P command delay | 固定 `4` physics steps，即 `8 ms` | 固定 `4` physics steps，即 `8 ms` | **一致** |
| 16 | Ankle command delay | 每个环境 reset 采样 `3–5` physics steps，即 `6–10 ms`；四个 ankle joints 共用 actuator group delay | `DelayedPDActuatorCfg(min_delay=3, max_delay=5)`，即 `6–10 ms` | **一致** |
| 17 | Arm command delay | `0` physics steps | implicit arm actuator，无 delay | **一致** |
| 18 | Policy IMU delay | 当前源码 `enabled=false`；保留但不生效的范围是 `[8, 26] ms` | `enabled=true`；每环境、每 reset 采样 `[8, 26] ms`，physics-step buffer 并插值 | **不一致** |
| 19 | Motion reset pose/velocity/joint perturbation | `noise_to_initial_level=0`，常规 Gaussian root/joint reset noise 实际关闭；只保留 reference root z `+0.01 m` | 每次 reset：root x/y `±0.05 m`、z `+[0.02, 0.04] m`、yaw `±0.2 rad`；linear vel 各轴 `±0.1 m/s`；angular roll/pitch `±0.2`、yaw `±0.3 rad/s`；joint pos `±0.03 rad` | **不一致** |
| 20 | Lie-down reset | 以 `0.3` 概率将 root height 置 `0.35 m` 并侧转 `±90°` | 没有对应 reset 分支 | **UFO only** |

### 4.1 完全一致的域随机化项

共 6 项：

1. Push interval：`1–3 s`
2. Push x/y 水平速度范围：`±0.5 m/s`
3. Base COM 作用对象和 startup 时机：`waist_yaw_link`
4. 4340P command delay：`4 steps`
5. Ankle command delay：`3–5 steps`
6. Arm command delay：`0 step`

两边虽然 command delay 一致，但 actuator 本体不一致：UFO 使用 real-motor chain，标准 `Mini3-BeyondMimic` 使用 PACE/delayed/implicit actuator 组合。

### 4.2 不一致的域随机化项

共 9 项：

- Push 的 z 与角速度范围
- Base COM 数值范围
- Robot contact friction/material 的参数化方式与范围
- Link mass 的范围与 body 作用范围
- Default joint position offset 的范围与采样时机
- 非脚踝 PD gain 的范围与采样时机
- 脚踝 PD gain，尤其 Kd 的范围与采样时机
- 当前源码的 IMU delay 启用状态
- Motion reset 的 pose/velocity/joint perturbation

### 4.3 仅一侧存在的项

UFO only，共 2 项：

- Motor strength/torque capacity `[0.9, 1.1]`
- `30%` lie-down reset

BeyondMimic only，共 3 项：

- Torso mass additive randomization `[-1.5, 1.5] kg`
- Joint friction scaling
- Joint armature scaling

### 4.4 当前源码与最近训练快照的 IMU 差异

当前源码：

```yaml
training:
  imu_delay:
    enabled: false
    time_range_s: [0.008, 0.026]
    randomize_on_reset: true
    interpolate: true
```

最近 Mini3 FB 完整训练快照：

```json
"imu_delay": {
  "enabled": true,
  "time_range_s": [0.008, 0.026],
  "randomize_on_reset": true,
  "interpolate": true
}
```

因此：

- 对“当前 HEAD 源码”而言，IMU delay 与 BeyondMimic 不一致。
- 对该历史 checkpoint 的训练配置而言，IMU delay 的启用状态、范围、reset 重采样和插值方式与 BeyondMimic 一致。

### 4.5 未启用或未接线的字段

- UFO 的 `randomize_torque_rfi`、`randomize_rfi_lim` 当前均为 `False`，MJLab 环境构建器也没有为其创建 event。
- UFO 配置了 `push_robot_recovery_time=2.0`，但当前 MJLab push event 没有读取这个字段，因此它不构成实际的 recovery gate。
- BeyondMimic 的 `base_upward_force` 被显式设为 `None`；注释中的 force curriculum 不生效。

### 4.6 Observation noise（不计入域随机化统计）

两边都只对 policy/actor observation 加噪，critic/privileged observation 保持干净，但数值并不完全相同：

| Observation | UFO noise range（scale 前） | BeyondMimic noise range（scale 前） | 结果 |
| --- | --- | --- | --- |
| Base angular velocity | `±0.2`，随后乘 observation scale `0.25` | `±0.1`，scale `1.0` | 不一致 |
| Projected gravity | `±0.05` | `±0.05` | 一致 |
| Joint position | `±0.01 rad` | `±0.01 rad` | 一致 |
| Joint velocity | `±0.5`，scale `1.0` | `±0.2`，随后乘 scale `0.1` | 不一致 |

当前 UFO 的 observation-noise curriculum 为关闭状态；BeyondMimic 也没有对应的 observation-noise curriculum。

## 5. 地形详细对比

### 5.1 有效地形设置

| # | 比较项 | UFO Mini3 FB 当前有效配置 | Mini3-BeyondMimic 当前有效配置 | 状态 |
| ---: | --- | --- | --- | --- |
| 1 | Terrain type | MJLab builder 硬编码 `TerrainEntityCfg(terrain_type="plane")` | `use_terrain=True`，改为 `terrain_type="generator"` | **不一致** |
| 2 | Terrain curriculum 开关 | `curriculum=False`，且运行时没有 terrain curriculum term | `TerrainGeneratorCfg(curriculum=True)`，并安装 `terrain_levels_tracking` | **不一致** |
| 3 | Terrain 类型/占比 | 无有效 sub-terrain，只有平面 | wave `0.6` + random rough `0.4` | **不一致** |
| 4 | 有效高度范围 | 全平面，高度变化为 `0` | wave amplitude `0–15 mm`；random rough noise `-5–10 mm`，step `5 mm` | **不一致** |
| 5 | 有效水平分辨率 | 平面不使用 heightfield horizontal scale；YAML 中未生效值为 `0.1 m` | `0.05 m` | **不一致** |
| 6 | Border | 平面不使用 generator border；YAML 中未生效值为 `40 m` | generator border `20 m`；每种子地形 border `0.25 m` | **不一致** |
| 7 | Environment origin/spacing | plane grid，`env_spacing=5.0 m` | terrain patches 分配；scene `env_spacing=2.5 m` 作为基础设置 | **不一致** |
| 8 | Policy height scan/height observation | `measure_heights=False`，无 height scan observation | 未配置 height scanner，policy observation 中没有 terrain height | **一致** |

有效地形设置共 8 项：1 项一致，7 项不一致。

### 5.2 BeyondMimic curriculum 行为

BeyondMimic 的地形不是只有 `curriculum=True` 开关；它还在每次 episode reset 时根据 tracking 结果调整 level：

- episode 正常 time-out：视为成功，移到更难的 terrain level；
- episode 提前 termination：视为失败，移到更简单的 terrain level；
- 共 `10` 个 row/difficulty levels；
- wave 难度从近似平面递增到约 `15 mm` amplitude；
- random-rough generator 本身不随 difficulty 改变。

UFO 当前没有对应的 terrain-level 更新逻辑。

### 5.3 UFO YAML 中“数值相同但实际未生效”的项

UFO 的 `terrain_locomotion_plane.yaml` 仍保留 rough-terrain 字段，其中有一些数值看起来与 BeyondMimic 相同：

| 字段 | UFO plane YAML | BeyondMimic generator | 是否算有效一致 |
| --- | ---: | ---: | --- |
| patch length/width | `8 m × 8 m` | `8 m × 8 m` | 否；UFO runtime 是 plane |
| rows | `10` | `10` | 否；UFO runtime 不生成 row |
| cols | `20` | `20` | 否；UFO runtime 不生成 column |
| vertical scale | `0.005 m` | `0.005 m` | 否；UFO plane 没有 heightfield |

这些字段不能据此判断两边地形一致，因为 UFO 的 MJLab builder 最终只传入：

```python
TerrainEntityCfg(
    terrain_type="plane",
    env_spacing=float(config.env_spacing),
)
```

`terrain_locomotion_plane.yaml` 中的 static/dynamic friction、restitution、rows、cols、scale、terrain types 和 proportions 都没有被这个 builder 映射到运行时 terrain entity。

## 6. 关键源码证据

UFO：

- `configs/robots/mini3.yaml`
  - domain-rand override：约第 68–80 行
  - 当前 IMU delay：约第 89–97 行
  - actuator delay groups：约第 223–259 行
- `humanoidverse/config/domain_rand/domain_rand.yaml`
  - 共享 push、COM、mass、friction、default-DOF 配置
- `humanoidverse/config/terrain/terrain_locomotion_plane.yaml`
  - `mesh_type='plane'`、`curriculum=False`
- `humanoidverse/agents/envs/humanoidverse_mjlab.py`
  - domain-rand event 映射：约第 987–1067 行
  - runtime plane：约第 1069–1075 行
  - per-reset default-DOF offset：约第 1405–1412、1664–1669 行
  - reset state perturbation：约第 1675–1720 行

BeyondMimic：

- `mini3_beyondmimic_env_cfg.py`
  - terrain generator：约第 50–75 行
  - Mini3 events：约第 135–196 行
  - IMU delay 与 `use_terrain=True`：约第 297–326 行
  - motion reset ranges：约第 361–378 行
- `beyondmimic_env_cfg.py`
  - 继承的 material/mass/gain/joint-parameter events：约第 192–268 行
  - base plane material：约第 85–100 行
  - physics/control timing：约第 410–419 行
- `roboparty.py`
  - `MINI3_IMPLICIT_CFG`：约第 1338–1441 行
- `mdp/curriculums.py`
  - `terrain_levels_tracking`：约第 21–55 行

## 7. 如果目标是让两边对齐

优先级建议如下：

1. 先明确要对齐当前 `Mini3-BeyondMimic` 还是 `Mini3-BeyondMimic-RealMotor`；当前 UFO 的 actuator dynamics 更接近后者，但本报告指定的标准 task 使用前者。
2. 地形必须在 UFO 的 MJLab builder 中真正接入 generator 和 curriculum；只修改 `terrain_locomotion_plane.yaml` 的 rows/proportions 不会生效。
3. 决定 domain parameters 是 startup 固定一整个训练进程，还是每 episode reset 重采样。当前两边在 default joint offset 和 PD gain 上的时机不同，单纯复制 range 仍不会等价。
4. 若要复现最近 checkpoint 的训练条件，应显式恢复 IMU delay；若以当前 HEAD 为准，则应把 BeyondMimic 的 IMU delay 也关闭后再做公平对照。
5. 决定是否保留 UFO 独有的 motor-strength 和 lie-down randomization，以及是否补齐 BeyondMimic 独有的 torso mass、joint friction 和 armature randomization。
