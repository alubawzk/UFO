# UFO（Mini3）与原始 BFM-Zero 一致性审计

审计日期：2026-07-15

## 1. 审计范围与比较基线

本报告比较以下两个工作区的实际代码与有效配置，而不是只比较同名 YAML：

- 当前工程：`/home/amax/Desktop/robot/UFO`，提交 `093f76347835a185062c02e692b6fa68924f886c`
- 原始工程：`/home/amax/Desktop/robot/BFM-Zero`，提交 `b87916f52d3d9e6eeba484f5e80851a235191837`

原始 BFM 工作区包含额外的 Mini3 资产和模型文件，但本报告的 BFM 基线取其原始 `train_bfm_zero()`、IsaacSim 环境、G1 配置和 LaFAN 数据链。当前工程基线取下面这条 Mini3 命令对应的有效配置：

```bash
CUDA_VISIBLE_DEVICES=0 ./run_train.sh \
  --agent fb \
  --robot-config configs/robots/mini3.yaml \
  --data-manifest configs/data/mini3_pkl.yaml \
  --gpu-ids single \
  --smoke \
  --work-dir runs/ufo_fb_mini3_smoke
```

需要特别说明：当前命令中的 `--agent fb` 实际构建的仍是原始 BFM 使用的 `FBcprAuxAgent`，不是代码目录中较早的纯 `FBAgent`。

一致性标记：

- **一致**：默认行为和关键参数相同，或单卡下执行逻辑等价。
- **结构一致**：算法意图相同，但机器人、仿真后端或数值参数不同。
- **不一致**：默认行为、训练信号或能力不同。
- **扩展**：UFO 新增、原始 BFM 不具备的能力。

## 2. 总结

| 维度 | 结论 | 关键判断 |
|---|---|---|
| 数据加载及使用 | **结构一致，但当前大数据链存在 P0 级不可扩展点** | 运动库、参考状态插值、expert trajectory 和优先采样思路来自 BFM；UFO 新增 manifest、多源配比和目录懒索引。但 expert loader 与正式 evaluation 仍强制全量物化 16.2 万条运动。 |
| Rewards | **公式高度一致，Mini3 参数有意不同** | 环境 raw reward 公式基本逐项复刻；真正驱动 `FBcprAuxAgent` 的 auxiliary reward 中，Mini3 改了 `action_rate` 和 `ankle_roll`。环境 penalty curriculum 不作用于 auxiliary critic。 |
| 域随机化 | **部分一致，参数和实际作用方式差异明显** | push、COM、质量、摩擦、默认关节偏置保留；Mini3 开启 PD gain、3–6 step 控制延迟和新增 motor strength。摩擦后端语义不同，原 BFM 的 RFI/base-mass 能力未迁移。 |
| Reset | **主流程一致，Mini3 少了 30% lie-down reset** | 都从随机 motion/time 重置 root 与关节，初始噪声当前均为 0，并清空动作和历史；原始 BFM 有 `lie_down_init=True, prob=0.3`，Mini3 当前未启用。 |
| Terminate | **默认训练近似一致，能力不一致** | 两边默认均只按 10 s timeout 结束，不启用跌倒、接触、运动偏离等 early termination；UFO 当前只创建 timeout term，其他开关必须为 false。timeout 还有 1 个控制步的边界差异。 |
| 模型训练 | **单卡核心算法高度一致，运行预算不同** | 网络模型文件与大部分 agent 逻辑直接沿用 BFM；架构和优化器超参数一致。UFO 总步数减半，checkpoint/eval 更频繁，支持多卡；Mini3 输入输出维度随本体变化。 |

当前不能把“代码结构一致”理解为“训练结果可复现原始 BFM”。G1→Mini3、IsaacSim→MJLab/MuJoCo-Warp、数据规模、接触/摩擦语义和 reset 分布均会改变训练分布。

## 3. P0 风险：Mini3 expert 数据仍被全量物化

这是本次审计最重要的结果。

### 3.1 原始 BFM

原始数据 `lafan_29dof_10s-clipped.pkl` 为单个约 196 MiB 的 joblib 字典：

- 862 条 motion clip
- 每条 300 帧、30 Hz
- 共 258,600 个源数据帧
- expert loader 将所有 motion 转成 50 Hz expert trajectory，并一次性构建 `TrajectoryDictBuffer`

这个“全量加载”设计对 862 条 LaFAN clip 尚可接受。

### 3.2 当前 Mini3

Mini3 转换索引记录：

- 162,044 条 motion clip，按最长 10 s 切分，索引总时长对应平均约 6.12 s/clip
- 总时长 991,006.108 s
- 118,925,490 个源数据帧
- 数据目录约 24 GiB

manifest 只检查目录索引和少量样本，MotionLib 初次也只按环境数加载一个 motion batch，因此表面上是懒加载；但正式进入训练前，`Workspace.train_online()` 会调用：

```python
load_expert_trajectories_from_motion_lib(...)
```

该函数首先执行不带 `max_num_seqs` 的：

```python
env._motion_lib.load_motions_for_training()
```

随后遍历 `_num_unique_motions`，把每条 motion 的全部 expert observation 拼到一个位于 `buffer_device="cuda"` 的 `TrajectoryDictBuffer`。按 50 Hz 估算约有 49,550,305 个 expert step。Mini3 每步只计算以下三项就有 427 个 `float32`：

- `state`: 48
- `last_action`: 21
- `privileged_state`: 358（运行时实际值；Hydra 配置错误声明为 357，见 4.4）

仅这三项的理论最低存储约为：

```text
49,550,305 × 427 × 4 bytes ≈ 78.82 GiB
```

这还没有包含 motion library 的 body position/rotation/velocity 中间张量、终止标记、索引和临时拷贝。对日志中的 31 GiB GPU，不可能按现有方式完整物化。

正式 tracking evaluation 也会把 `_num_unique_motions` 全部加载，并逐条评估；因此即使绕过 expert buffer，默认 prioritization/evaluation 路径仍不可扩展到当前数据量。

`--smoke` 只关闭 evaluation/prioritization，没有关闭 `load_isaac_expert_data`。因此修好当前 CUDA 架构问题后，smoke 也会继续走全量 expert loader；它不会自动把 162,044 条 motion 缩成小样本。

**结论：数据文件转换已经完成不等于当前 Mini3 数据能够按原 BFM 的全量 expert 使用方式开始正式训练。必须先把 expert sampling 和 evaluation 改为流式、分片或受控子集。**

相关代码：

- UFO：`humanoidverse/agents/envs/expert_motion_loader.py:14-129`
- UFO：`humanoidverse/training/workspace.py:556-597`
- UFO：`humanoidverse/agents/evaluations/humanoidverse_mjlab.py:316-365, 401-404`
- BFM：`humanoidverse/agents/envs/humanoidverse_isaac.py:39-166`

## 4. 数据加载以及使用方式

### 4.1 数据入口

| 项目 | 原始 BFM | 当前 UFO Mini3 | 一致性 |
|---|---|---|---|
| 入口 | `lafan_tail_path='humanoidverse/data/lafan_29dof_10s-clipped.pkl'` | `configs/data/mini3_pkl.yaml` → per-motion directory | 结构一致，入口扩展 |
| 数据组织 | 单个 joblib dict | 目录中每条 motion 一个 pkl，带完整索引 | 扩展 |
| 本体 | G1 29-DoF | Mini3 21-DoF | 本体差异 |
| 训练 clip | 862 条、约 10 s | 162,044 条、最长 10 s、平均约 6.12 s | 规模严重不同 |
| 数据混合 | 单源 | 支持 manifest 多源及归一化权重 | 扩展 |
| 本体校验 | 无 manifest 本体匹配检查 | manifest 中的 `robot_config` 必须与 CLI 配置兼容 | 扩展 |

### 4.2 MotionLib 核心语义

以下部分保持一致：

- `MotionLibRobot` 文件在两个工程中逐字节相同。
- motion ID 和起始时间都按概率采样。
- `get_motion_state()` 根据各 motion 的 fps 做时间插值。
- 环境 reset 从参考 motion 取得 root pose/velocity、DOF position/velocity。
- actor 使用本体 proprioception，critic/Backward map 使用 `max_local_self`。
- expert trajectory 末帧标为 `truncated=True`，`terminated=False`。
- `seq_length=8` 的 expert slice 进入 FB/CPR/auxiliary 更新。

UFO 对 `motion_lib_base.py` 做了三类扩展：

1. 支持单文件、目录和多个数据源。
2. 多源情况下先固定 source-level probability mass，再在源内做 priority sampling，防止 tracking priority 改掉数据集配比。
3. 若 motion 记录提供 `dof_pos`，直接使用 retarget 后的本体关节角并用有限差分生成 `dof_vel`。Mini3 当前数据走这条路径；原始 BFM 主要从 skeleton/pose 数据生成 DOF 状态。

### 4.3 Observation 使用差异

`max_local_self` 的几何构造函数保持相同：heading-local body position、6D rotation、linear velocity、angular velocity，再按本体 body 数决定维度。

| 维度 | 原始 G1 | 当前 Mini3 |
|---|---:|---:|
| Action / DOF | 29 | 21 |
| `state` | 64 | 48 |
| `last_action` | 29 | 21 |
| `history_actor` | 372 | 276 |
| `privileged_state` / `max_local_self`（Hydra 声明/日志） | 462 | 357 |
| `privileged_state` / `max_local_self`（运行时实际） | 463 | 358 |

UFO 还修正了原始 expert loader 中 base angular velocity 的坐标系/缩放不一致：原始 loader 直接使用 world-frame angular velocity，而在线 actor observation 使用 body-frame angular velocity 并乘 `obs_scales.base_ang_vel=0.25`；UFO 的 `reference_base_ang_vel()` 按在线 observation 方式处理。该项是合理修正，但不属于逐值复刻。

### 4.4 `max_local_self` 维度元数据存在共同的 off-by-one

两边的 observation 函数都在 `root_height_obs=true` 时输出 1 维 root height，并输出：

```text
3 × (body_count - 1) + 6 × body_count + 3 × body_count + 3 × body_count
= 15 × body_count - 3
```

加上 root height 后，运行时总维度应为 `15 × body_count - 2`。但共享的 Hydra 公式写成：

```yaml
(3 + 6 + 3 + 3) * body_count + 1 - 3 - 1
```

即 `15 × body_count - 3`，多减了 1。由相同 observation 函数在 CPU 上直接构造张量校验：

- 原始 G1：30 个 robot body + 1 个 extend body，配置声明 462，运行时为 463。
- 当前 Mini3：24 个 body，配置声明 357，运行时为 358。

这是 UFO 从 BFM 继承的共同问题，因此属于“缺陷一致”，而不是 Mini3 新引入的差异。当前 vector-env 的 Gym observation space 是从实际 reset tensor 推导的，所以 agent 模型按 358 维构建；`history_actor` 也不包含 `max_local_self`，当前主路径不一定立即报 shape 错误。但预处理日志、`algo_obs_dim_dict` 和任何直接信任 `obs_dims.max_local_self` 的工具都会少报 1 维，应修正公式并增加维度断言。

### 4.5 优先采样

原始 BFM 和 UFO 正式训练默认都开启 tracking-evaluation 驱动的 priority：

- EMD 先裁剪到 `[0.5, 2.0]`
- 再乘 `2.0`
- `prioritization_mode='exp'`，即使用 `2 ** priority`
- 更新 expert buffer 和 MotionLib 的采样概率

差异是 UFO 的多源实现保留 source-level mix；原始 BFM 对全体 motion 统一归一化。当前 Mini3 只有一个 source，二者在数学上退化为相同分布。

`--smoke` 会关闭 evaluation 和 prioritization，因此 smoke 不能验证该链路。

## 5. Rewards 设计和参数设置

### 5.1 必须区分两套 reward

BFM/UFO 同时存在两套权重：

1. `rewards.reward_scales.*`：环境标量 reward，用于 rollout 输出和日志。
2. `agent.aux_rewards_scaling`：`FBcprAuxAgent` 将 raw `aux_rewards` 加权后训练 auxiliary critic 的实际权重。

原始和当前 agent 都不直接用环境标量 reward 训练 auxiliary critic。环境 penalty curriculum 只缩放第 1 套，不缩放传给 agent 的 raw auxiliary terms。因此判断训练信号时应优先看第 2 套。

### 5.2 Reward 公式

下列 raw term 在 UFO 中保持了 BFM 的定义：

| Term | 核心定义 |
|---|---|
| `penalty_torques` | `sum(torque²)` |
| `penalty_action_rate` | `sum((last_action - action)²)` |
| `limits_dof_pos` | 超出 soft position limit 的线性总量 |
| `limits_dof_vel` | 超出 soft velocity limit 的量，单关节最大裁到 1 |
| `limits_torque` | 超出 soft torque limit 的线性总量 |
| `penalty_undesired_contact` | 任一指定 body 的接触力绝对值超过 1 时为 1 |
| `penalty_ankle_roll` | 左右 ankle-roll position 的平方和 |
| `penalty_feet_ori` | 足部接触时，足部局部重力 XY 分量的范数 |
| `penalty_slippage` | 足部接触时的 foot linear velocity norm |
| `feet_heading_alignment` | 左右足 heading 与 root heading 的绝对角差之和 |

公式一致不代表数值分布一致。BFM 的接触和 actuator signal 来自 PhysX/IsaacSim，UFO 来自 MJLab/MuJoCo-Warp；接触聚合、摩擦模型和 actuator force 的数值分布仍需实测校准。

### 5.3 环境标量 reward 的有效参数

表中数值为乘 `dt=0.02` 前的 YAML scale。

| Term | 原始 BFM 有效值 | 当前 Mini3 有效值 | 结论 |
|---|---:|---:|---|
| `penalty_torques` | `-0.000001` | `-0.000001` | 一致 |
| `penalty_undesired_contact` | `-1.0` | `-1.0` | 一致，body 列表随本体变化 |
| `penalty_action_rate` | `-0.5` | `-0.2` | 不一致，Mini3 减弱 |
| `penalty_ankle_roll` | `-0.5` | `-1.0` | 不一致，Mini3 加强 |
| `penalty_feet_ori` | `-0.1` | `-0.1` | 一致 |
| `feet_heading_alignment` | `-0.1` | `-0.1` | 一致 |
| `penalty_slippage` | `-1.0` | `-1.0` | 一致 |
| `limits_dof_pos` | `-10.0` | `-10.0` | 一致，物理 limit 随本体变化 |
| `limits_dof_vel` | `-5.0` | `-5.0` | 一致，物理 limit 随本体变化 |
| `limits_torque` | `-5.0` | `-5.0` | 一致，物理 limit 随本体变化 |

### 5.4 FB auxiliary critic 的实际权重

| Term | 原始 BFM | 当前 Mini3 | 结论 |
|---|---:|---:|---|
| `penalty_torques` | `0.0` | `0.0` | 一致，当前不进入 auxiliary scalar |
| `penalty_action_rate` | `-0.1` | `-0.2` | 不一致，Mini3 加强 2 倍 |
| `limits_dof_pos` | `-10.0` | `-10.0` | 一致 |
| `limits_torque` | `0.0` | `0.0` | 一致，当前不进入 auxiliary scalar |
| `penalty_undesired_contact` | `-1.0` | `-1.0` | 一致，body 语义不同 |
| `penalty_feet_ori` | `-0.4` | `-0.4` | 一致 |
| `penalty_ankle_roll` | `-4.0` | `-1.0` | 不一致，Mini3 减弱到 1/4 |
| `penalty_slippage` | `-2.0` | `-2.0` | 一致 |

两边都使用 `RewardNormalizer(translate=False, scale=True)` 归一化合成后的 auxiliary reward。

### 5.5 Penalty curriculum

原始 BFM 的有效配置不是 reward YAML 中的 `false`，而是 experiment 配置覆盖后的：

```yaml
reward_penalty_curriculum: true
reward_initial_penalty_scale: 0.10
reward_min_penalty_scale: 0.0
reward_max_penalty_scale: 1.0
reward_penalty_level_down_threshold: 40
reward_penalty_level_up_threshold: 42
reward_penalty_degree: 0.00001
```

当前 Mini3 明确覆盖为 `reward_penalty_curriculum=false`。这意味着环境 scalar penalty 从一开始就按完整 YAML scale 计算，而不是从 `0.1` 倍开始；但 raw `aux_rewards` 和 `agent.aux_rewards_scaling` 不经过这个 curriculum，所以对当前 FB auxiliary critic 没有直接作用。

## 6. 域随机化设计和参数设置

控制频率两边均为 50 Hz（physics 200 Hz、decimation 4）。

| 随机项 | 原始 BFM 有效配置/实现 | 当前 Mini3 有效配置/实现 | 一致性 |
|---|---|---|---|
| Push | 开启；1–3 s；XY `±0.5 m/s`；RPY `±0.5 rad/s` | 相同范围；MJLab interval event | 结构一致 |
| Base COM | 开启；torso；XYZ 均 `[-0.02, 0.02] m`；startup | 开启；`waist_yaw_link`；X/Y `[-0.03,0.03]`，Z `[-0.02,0.02]`；startup | 参数/本体不同 |
| Link mass | 开启；all bodies；scale `[0.95,1.05]`；startup | 开启；all bodies；scale `[0.98,1.02]`；startup | Mini3 更窄 |
| Friction | 开启；static/dynamic material `[0.5,1.25]`；startup | 开启；geom friction axis 0 absolute `[0.5,2.2]`；startup | 数值和物理语义不同 |
| PD gain | 默认关闭；预留 Kp/Kd `[0.75,1.25]` | 开启；Kp/Kd 各自逐 env/逐 joint scale `[0.75,1.25]`，每次 reset 重采样 | 默认不一致 |
| Motor strength | 无 | 开启；DC motor continuous/peak torque scale `[0.9,1.1]`，每次 reset 重采样 | UFO 扩展 |
| Control delay | 默认关闭；预留 `[0,1]` step | 开启；`[3,6]` step，即约 60–120 ms，每次 reset 重采样 | 默认和参数不一致 |
| Default DOF offset | 开启；逐 env/逐 joint `[-0.02,0.02] rad` | 相同 | 一致 |
| Torque RFI | 默认关闭；通用环境有 `rfi_lim=0.1` 计算，但默认 IsaacSim implicit-actuator 路径没有把该扰动注入实际 actuator | 配置字段保留但 MJLab bridge 未实现 | 默认结果一致；两边都不能仅靠开 YAML 获得预期动力学扰动 |
| RFI limit scale | 默认关闭；通用环境有 `[0.5,1.5]` 计算，但同样受 implicit-actuator 路径限制 | 配置字段保留但未实现 | 默认结果一致；能力均不完整 |
| Base mass | 默认关闭 | 配置字段保留但未实现 | 默认一致，能力缺失 |

### 6.1 PD gain 的一个重要实现差异

原始 BFM 默认使用 IsaacSim implicit actuator。虽然通用环境代码能随机 `_kp_scale/_kd_scale`，IsaacSim 分支实际发送的是 joint position target，implicit actuator 的 stiffness/damping 没有用这两个 scale 重写。因此原始默认关闭时没有问题，但简单把原始 `randomize_pd_gain` 改成 true，并不能保证真实动力学增益被随机化。

UFO 的 MJLab 版本调用 `mjlab_dr.pd_gains`，会重写 actuator gain；当前 Mini3 的 PD DR 是实际生效的，不只是 reward 中的 torque estimate 改变。

### 6.2 摩擦不能只按同名参数判断一致

原始 IsaacSim 同时采样 static/dynamic rigid-body material friction；UFO 当前只对 MuJoCo geom friction 的 axis 0（sliding friction）做 absolute sampling，torsional/rolling 分量不变。因此 `[0.5,1.25]` 即使写成相同数值，也不是严格相同的动力学分布。

### 6.3 Disable 开关

UFO 的 `--disable-dr` 会关闭当前已支持的 push、PD、motor strength、COM、mass、friction、RFI 字段和 default-DOF offset。

原始 BFM 的 `disable_domain_randomization` 中写的是 `randomize_push_robots=False`，而有效配置字段名是 `push_robots`；因此原始 disable 开关存在不能关闭 push 的字段名问题。UFO 同时写两个字段，修正了这个问题。

Observation noise 与 dynamics DR 分开控制。两边 `bfm_zero_obs.yaml` 当前相同；`--disable-dr` 不会自动等价于 `--disable-obs-noise`。

## 7. Reset 设置

### 7.1 默认 reset 主流程

两边的训练 reset 主流程一致：

1. 找出 timeout/terminated 的 env。
2. 清理 action、last action、episode length 和 history。
3. 执行 per-episode DR（UFO 中 PD/motor strength 为 MJLab reset event，control delay/default offset 由 wrapper 重采样）。
4. 重新采样 motion ID 和随机 motion start time。
5. 从 MotionLib 取得该时刻的 root pose/velocity 和 DOF position/velocity。
6. 写入仿真状态并刷新派生量。

### 7.2 有效参数对比

| 项目 | 原始 BFM | 当前 Mini3 | 一致性 |
|---|---|---|---|
| `noise_to_initial_level` | `0` | `0` | 一致，所有 `init_noise_scale` 当前实际不生效 |
| 随机 motion start | 开启 | 开启 | 一致 |
| `resample_motion_when_training` | `false` | `false` | 一致，不按 150 s 周期主动换 motion batch |
| Lie-down init | `true`，概率 `0.3` | `false`，概率 `0.0` | **不一致** |
| Default DOF offset DR | `[-0.02,0.02]` | `[-0.02,0.02]` | 一致 |
| Action/history 清零 | 是 | 是 | 一致 |
| Control-delay queue 清零 | 原始功能存在但默认 delay 关闭 | 是，且重新采样 3–6 step | 结构一致、默认不同 |
| 指定 target-state reset | 支持，tracking evaluation 使用 | 支持，tracking evaluation 使用 | 一致 |
| `reset_to_default_pose` API | 支持 | 参数被忽略 | 能力缺失，但训练主流程不使用 |

原始 BFM 的 30% lie-down reset 是一个显著训练分布，不应被当成普通初始化细节。是否给 Mini3 恢复它要根据 Mini3 起身能力、碰撞几何和目标技能决定；如果目标是复现 BFM 的 recovery 能力，则当前配置不一致。

## 8. Terminate 设置

### 8.1 默认行为

两边的有效 YAML 都将以下 early termination 设为 false：

- contact
- gravity/orientation
- low base height
- motion end
- motion far
- close to DOF position limit
- close to DOF velocity limit
- close to torque limit

所以默认训练中，摔倒、非足端接触、偏离参考 motion 或 motion 播放结束都不会提前结束 episode。唯一结束条件是 10 s timeout，并映射为 Gymnasium `truncated=True`、`terminated=False`。

### 8.2 能力差异

原始 BFM 实现了上述 termination 逻辑及 motion-far curriculum；当前 MJLab bridge 在构建时断言所有开关必须为 false，只向 MJLab 注册：

```python
"time_out": TerminationTermCfg(func=time_out, time_out=True)
```

因此默认行为近似一致，但如果以后打开跌倒、接触、limit 或 motion tracking termination，UFO 当前不是“参数未调”，而是“功能尚未迁移”，会在构建阶段失败。

### 8.3 Timeout 的 1-step 差异

两边控制周期都是 0.02 s，10 s 对应 500 个控制步：

- 原始 BFM：`episode_length_buf > max_episode_length`，从 0 开始计数时通常在第 501 步 timeout。
- 当前 MJLab：`episode_length_buf >= max_episode_length`，在第 500 步 timeout。

默认 episode 长度因此相差约 20 ms。对训练影响通常小，但它不是严格逐步一致。

## 9. 模型训练设置

### 9.1 核心模型与算法

以下 UFO 文件与 BFM 对应文件逐字节相同：

- `humanoidverse/agents/fb/model.py`
- `humanoidverse/agents/fb_cpr/model.py`
- `humanoidverse/agents/fb_cpr_aux/model.py`
- `humanoidverse/agents/nn_models.py`
- `humanoidverse/agents/normalizers.py`
- `humanoidverse/agents/nn_filters.py`
- `humanoidverse/agents/buffers/trajectory.py`
- `humanoidverse/utils/motion_lib/motion_lib_robot.py`

三个 agent 文件的核心更新逻辑也保持一致；UFO 只在 backward 后增加 `average_gradients()`。单卡未初始化 distributed group 时该函数立即返回，因此单卡更新语义与原始 BFM 一致。

### 9.2 网络结构

| 模块 | 原始 BFM 与当前 UFO |
|---|---|
| Latent `z_dim` | 256，`norm_z=True` |
| Forward map | residual，hidden 2048，6 hidden layers，2 embedding layers，ensemble 2 |
| Backward map | hidden 256，1 layer，normalized output |
| Actor | residual，hidden 2048，6 hidden layers，2 embedding layers |
| Critic | residual，hidden 2048，6 hidden layers，2 embedding layers，ensemble 2 |
| Discriminator | hidden 1024，3 layers |
| Auxiliary critic | residual，hidden 2048，6 hidden layers，2 embedding layers，ensemble 2 |
| Sequence length | 8 |
| Actor std | 0.05 |
| AMP | false |
| Compile | 单卡 true |
| CUDA graphs | false |

网络宽度和深度一致，但第一层和 actor 输出尺寸会根据 observation/action space 自动变化，所以 G1 checkpoint 不能直接视为 Mini3 21-DoF 模型。

### 9.3 优化器和算法超参数

当前默认 `lr_scale=1.0`、`clip_grad_norm=0.0` 时，下列参数与原始 BFM 相同：

| 参数 | 值 |
|---|---:|
| `lr_f` / `lr_actor` / `lr_critic` / `lr_aux_critic` | `3e-4` |
| `lr_b` / `lr_discriminator` | `1e-5` |
| Batch size | 1024 |
| Discount | 0.98 |
| FB target tau | 0.01 |
| Critic target tau | 0.005 |
| Ortho coefficient | 100.0 |
| Actor/critic/aux pessimism | 0.5 |
| FB pessimism | 0.0 |
| Stddev clip | 0.3 |
| Train goal ratio | 0.2 |
| Expert ASM ratio | 0.6 |
| Relabel ratio | 0.8 |
| Discriminator gradient penalty | 10.0 |
| Auxiliary regularization | 0.02 |
| Main regularization | 0.05 |
| Latent update interval | 100 step |
| Z buffer size | 8192 |
| Expert rollout length / ratio | 250 / 0.5 |

### 9.4 训练运行参数

| 参数 | 原始 BFM | UFO 正式默认 | 当前 `--smoke` | 结论 |
|---|---:|---:|---:|---|
| Seed | 4728 | 4728 | 4728 | 一致 |
| Envs / GPU | 1024 | 1024 | 16 | smoke 缩小 |
| Global env steps | 384,000,000 | 192,000,000 | 2,048 | 正式预算减半 |
| Seed env steps | 10,240 | 10,240 | 10,240 | 一致 |
| Log interval | 384,000 | 384,000 | 384,000 | 一致（字段虽名为 `log_every_updates`，实际按 env step 检查） |
| Update trigger | 每 1,024 local env steps | 每 1,024 local env steps | 同左 | 单卡一致 |
| Agent updates / trigger | 16 | 16 | 16，但不会触发 | 配置一致 |
| Online replay capacity | 5,120,000 | 5,120,000 / rank | 同左 | 单卡一致 |
| Trajectory buffer | true | true | true | 一致 |
| Checkpoint interval | 9,600,000 | 3,200,000 | 不会触发 | UFO 更频繁 |
| Eval interval | 9,600,000 | 3,200,000 | evaluation 关闭 | UFO 更频繁 |
| Prioritization | true | true | false | smoke 不覆盖正式链路 |
| Buffer device | CUDA | CUDA | CUDA | 一致 |
| W&B | false | CLI 默认 false | false | 一致 |

当前 smoke 只有 2,048 env steps，小于 10,240 seed steps，所以不会执行任何 optimizer update。它只能验证环境创建、reset、step、replay 写入等路径，不能验证模型 loss/backward/update，也不能验证 evaluation/prioritization。

### 9.5 UFO 的训练扩展

UFO 新增：

- `--agent fb/tech`
- 多 GPU、rank-local replay shard、gradient averaging 和全局 step accounting
- `--lr-scale`、`--clip-grad-norm`、`--num-agent-updates`
- manifest、多源数据固定配比
- 每机器人 auxiliary reward override
- non-finite 检查、分布式 checkpoint 状态

单卡默认不改变 BFM 核心更新公式；多卡是新增路径，不能用“与原始 BFM 完全一致”描述。

## 10. 优先级建议

### P0：开始正式 Mini3 训练前必须处理

1. 把 expert trajectory 改成流式/分片采样或有界 cache，禁止对 162,044 条 motion 调用全量 `load_motions_for_training()`。
2. 把 tracking evaluation 改成按 chunk 加载和释放，或为正式训练配置固定、可复现的 validation 子集。
3. 解决当前 RTX 5090 D `sm_120` 与已安装 PyTorch CUDA wheel 不兼容的问题；现有 smoke 日志在创建 CUDA tensor 时已经失败，尚未验证 MJLab step。

### P1：决定要“复刻 BFM”还是“适配 Mini3”

1. 明确是否恢复 `lie_down_init=true, prob=0.3`。它影响 recovery 分布，不应无意识遗漏。
2. 保留两套命名清楚的配置：
   - `bfm_fidelity`：尽量复刻原始 BFM 分布，用于回归。
   - `mini3_sim2real`：保留 Mini3 的 PD、delay、motor-strength、摩擦和 reward 调整。
3. 若要做 BFM fidelity 对照，agent auxiliary 权重应恢复 `action_rate=-0.1`、`ankle_roll=-4.0`；仅修改环境 `reward_scales` 不能完成该对照。
4. 给 DR 增加 reset 后参数统计，确认 PD、delay、motor strength、COM、mass 和 friction 的实际样本范围。

### P2：功能一致性补齐

1. 迁移 contact、gravity、height、motion-end、motion-far 和 joint/torque-limit termination，或从 UFO 配置中删除这些目前不可开启的假开关。
2. 决定是否对齐 500/501 step timeout 边界。
3. 若后续需要 torque RFI、RFI-limit 或 base-mass DR，应先实现 MJLab 消费代码，不能只改 YAML。
4. 用相同短 motion、关闭 DR/noise 的条件，对 IsaacSim 与 MJLab 的 observation、raw reward、reset state 做逐步数值回归；接触与 actuator force 需要单独设容差。
5. 修正共享的 `max_local_self` Hydra 维度公式，并在环境构建时断言配置维度等于运行时 tensor 维度。

## 11. 主要证据位置

### 原始 BFM

- 训练 preset 与运行参数：`/home/amax/Desktop/robot/BFM-Zero/humanoidverse/train.py:587-721`
- BFM experiment：`/home/amax/Desktop/robot/BFM-Zero/humanoidverse/config/exp/bfm_zero/bfm_zero.yaml`
- Reward 配置：`/home/amax/Desktop/robot/BFM-Zero/humanoidverse/config/rewards/reward_bfm_zero.yaml`
- DR 配置：`/home/amax/Desktop/robot/BFM-Zero/humanoidverse/config/domain_rand/domain_rand.yaml`
- Reward/DR/reset/timeout：`/home/amax/Desktop/robot/BFM-Zero/humanoidverse/envs/legged_base_task/legged_robot_base.py`
- Motion reset/termination：`/home/amax/Desktop/robot/BFM-Zero/humanoidverse/envs/legged_robot_motions/legged_robot_motions.py`
- IsaacSim startup DR：`/home/amax/Desktop/robot/BFM-Zero/humanoidverse/simulator/isaacsim/isaacsim.py:115-197`
- Expert loader 与 adapter：`/home/amax/Desktop/robot/BFM-Zero/humanoidverse/agents/envs/humanoidverse_isaac.py`

### 当前 UFO

- CLI 与有效 TrainConfig：`humanoidverse/train.py:37-249, 405-533`
- FB preset：`humanoidverse/agents/presets/fb.py`
- Mini3 配置：`configs/robots/mini3.yaml`
- Mini3 manifest：`configs/data/mini3_pkl.yaml`
- Manifest 处理：`humanoidverse/utils/motion_data/manifest.py`
- 多源/目录 MotionLib：`humanoidverse/utils/motion_lib/motion_lib_base.py`
- Expert loader：`humanoidverse/agents/envs/expert_motion_loader.py`
- MJLab reward、DR、reset、termination adapter：`humanoidverse/agents/envs/humanoidverse_mjlab.py`
- Training workspace：`humanoidverse/training/workspace.py`
- Tracking evaluation：`humanoidverse/agents/evaluations/humanoidverse_mjlab.py`
