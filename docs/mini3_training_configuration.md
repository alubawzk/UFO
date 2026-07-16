# Mini3 训练与配置调优指南

本文整理 Mini3 在 UFO 中的训练准备、配置加载关系、reward、域随机化、smoke training 和正式训练流程。内容针对当前 MJLab 训练路径。

## 1. 当前训练准备状态

截至 2026-07-15，当前工作区已具备启动 Mini3 训练所需的基础文件：

- Robot YAML：`configs/robots/mini3.yaml`
- Hydra robot config：`humanoidverse/config/robot/mini3/mini3_auto.yaml`
- MuJoCo XML：`humanoidverse/data/robots/mini3_mjlab/mini3.xml`
- 训练数据 manifest：`configs/data/mini3_pkl.yaml`
- 转换后的训练数据：`humanoidverse/data/mini3_pkl_ufo/`

已完成的静态检查：

- 数据转换状态为 `complete`，共有 162,044 个训练片段，约占用 24 GB。
- 抽样数据的 `dof_pos` 为 21 维，`pose_aa` 为 24 body。
- Mini3 XML 可由 MuJoCo 加载，`nq/nv/nu == 28/27/21`。
- Mini3 robot config 显示 24 个机器人 body 和 21 维 action。
- 21 项 Mini3 相关回归测试通过。
- 当前机器为单张约 32 GB 显存的 RTX 5090 D，可先进行单卡 smoke training。

仍需通过运行时验证的项目：

- 零位站姿是否穿地、自碰撞或立即失稳。
- PD、力矩限制和速度限制是否适合当前 Mini3 版本与控制频率。
- 左右脚 contact、slippage、feet orientation 和 ankle-roll 语义是否正确。
- smoke training 是否无 NaN、shape mismatch 和 CUDA OOM。

`configs/robots/mini3.yaml` 当前仍标记为 `metadata.review_status: draft`，因此完成静态检查不等于已经完成动力学标定。

## 2. 配置加载关系

训练入口是 `run_train.sh`，实际调用 `humanoidverse/train.py`。当前训练固定组合 `humanoidverse/config/exp/bfm_zero/bfm_zero.yaml`，该实验配置继续加载：

- `humanoidverse/config/rewards/reward_bfm_zero.yaml`
- `humanoidverse/config/domain_rand/domain_rand.yaml`
- `humanoidverse/config/obs/bfm_zero_obs.yaml`
- `humanoidverse/config/terrain/terrain_locomotion_plane.yaml`
- robot Hydra config

Mini3 的实际配置顺序可以概括为：

1. Hydra 加载 `exp/bfm_zero/bfm_zero.yaml` 及其共享 reward、domain randomization、observation 和 terrain 配置。
2. `configs/robots/mini3.yaml` 中的 `training.hydra_robot: mini3/mini3_auto` 选择 Mini3 Hydra robot config。
3. `training.hydra_overrides` 覆盖共享 Hydra 参数。
4. Robot YAML 中的 init state、PD、actuator、关节限制、语义字段以及 agent override 注入最终训练配置。
5. `--disable-dr` 和 `--disable-obs-noise` 等 CLI 开关在最后强制关闭对应功能。

因此，Mini3 专属的环境 reward 和域随机化调参放在 `training.hydra_overrides` 中；FB auxiliary reward 权重放在 `training.agent.fb.aux_rewards_scaling` 中。直接修改共享 YAML 或 FB preset 会同时影响 G1 和其他机器人。

## 3. 常用配置位置

| 调整目标 | 文件 | 作用范围 |
| --- | --- | --- |
| Mini3 专属 Hydra override | `configs/robots/mini3.yaml` | 所有使用该 Mini3 robot config 的训练 |
| Mini3 专属 FB auxiliary reward | `configs/robots/mini3.yaml` | `training.agent.fb.aux_rewards_scaling` |
| 共享环境 reward | `humanoidverse/config/rewards/reward_bfm_zero.yaml` | 所有使用 `bfm_zero` 实验配置的机器人 |
| 共享域随机化 | `humanoidverse/config/domain_rand/domain_rand.yaml` | 所有使用该 domain rand config 的机器人 |
| 观测噪声 | `humanoidverse/config/obs/bfm_zero_obs.yaml` | actor/critic observation |
| FB auxiliary reward 默认值 | `humanoidverse/agents/presets/fb.py` | 所有 `--agent fb` 训练 |
| TeCH auxiliary reward 权重 | `humanoidverse/agents/presets/tldr.py` | 所有 `--agent tech` 训练 |
| Reward 计算实现 | `humanoidverse/agents/envs/humanoidverse_mjlab.py::_compute_reward()` | 当前 MJLab 环境 |
| 域随机化事件实现 | `humanoidverse/agents/envs/humanoidverse_mjlab.py::make_mjlab_ufo_env_cfg()` | 当前 MJLab 环境 |

`humanoidverse/config/robot/mini3/mini3_auto.yaml` 是 Hydra 结构配置。一般不应在这里维护 reward 或域随机化；Mini3 XML 和 `configs/robots/mini3.yaml` 才是机器人结构、关节顺序和控制参数的主要来源。

## 4. Mini3 专属 Hydra override

当前 `configs/robots/mini3.yaml` 使用以下 override：

```yaml
training:
  hydra_robot: mini3/mini3_auto
  hydra_overrides:
    # Simulation/control frequency
    - "simulator.config.sim.fps=500"
    - "simulator.config.sim.control_decimation=10"

    # Environment reward
    - "rewards.reward_scales.penalty_action_rate=-0.2"
    - "rewards.reward_scales.penalty_ankle_roll=-1.0"
    - "rewards.reward_penalty_curriculum=false"

    # Domain randomization
    - "domain_rand.push_robots=true"
    - "domain_rand.randomize_base_com=true"
    - "domain_rand.base_com_range.x=[-0.03,0.03]"
    - "domain_rand.base_com_range.y=[-0.03,0.03]"
    - "domain_rand.link_mass_range=[0.98,1.02]"
    - "domain_rand.friction_range=[0.5,2.2]"
    - "domain_rand.randomize_pd_gain=true"
    - "domain_rand.kp_range=[0.75,1.25]"
    - "domain_rand.kd_range=[0.75,1.25]"
    - "domain_rand.randomize_ctrl_delay=true"
    - "domain_rand.randomize_motor_strength=true"
    - "domain_rand.motor_strength_range=[0.9,1.1]"
  agent:
    fb:
      aux_rewards_scaling:
        penalty_action_rate: -0.2
        penalty_ankle_roll: -1.0
```

这些 override 已通过 Hydra 合成检查。最终有效值包括：

- `penalty_action_rate = -0.2`
- `penalty_ankle_roll = -1.0`
- physics 为 500 Hz，policy/control update 为 50 Hz（decimation 10）
- reward penalty curriculum 关闭
- push 开启
- base COM：x/y 均为 `[-0.03, 0.03]`
- base COM 的 z 未被覆盖，继续继承共享值 `[-0.02, 0.02]`
- link mass scale 为 `[0.98, 1.02]`
- friction 为 `[0.5, 2.2]`
- 每次 episode reset 独立采样 Kp/Kd 乘法系数，范围均为 `[0.75, 1.25]`
- delay 按 actuator 组配置：4340P 固定 4 个仿真步（8 ms），脚踝每次 reset 在 3–5 步（6–10 ms）间采样，手臂为 0；FIFO 每个 physics substep 前进一步，不按 50 Hz policy step 计数
- IMU delay 每个环境在 reset 时从 8–26 ms 均匀采样；`base_ang_vel` 与 `projected_gravity` 共用一个 500 Hz physics-step 环形缓冲，并对非整数延迟步插值。该延迟只进入 actor/history actor，privileged state 和 reward 仍读取实时状态
- 每次 episode reset 将 DC motor 连续/峰值力矩能力同步缩放至 nominal 的 `[0.9, 1.1]`
- FB auxiliary critic 实际使用 `penalty_action_rate=-0.2`、`penalty_ankle_roll=-1.0`

如果需要同时维护多组实验参数，建议复制一份 Robot YAML，例如 `configs/robots/mini3_train_v2.yaml`，保持 `name: mini3` 和同一个 `xml_path`，只修改 `training.hydra_overrides`。启动时显式传入该文件，并使用新的 `--work-dir`。

## 5. Reward 配置

### 5.1 环境 reward

环境 reward 的共享默认值位于：

```text
humanoidverse/config/rewards/reward_bfm_zero.yaml
```

当前环境实现可以计算以下项：

- `penalty_torques`
- `penalty_action_rate`
- `limits_dof_pos`
- `limits_dof_vel`
- `limits_torque`
- `penalty_undesired_contact`
- `penalty_ankle_roll`
- `penalty_feet_ori`
- `penalty_slippage`
- `feet_heading_alignment`

`reward_scales` 中的非零项会组成环境返回的 scalar reward。环境内部会再将每个 scale 乘以仿真控制步长 `dt`。

Penalty curriculum 由以下字段控制：

```yaml
rewards:
  reward_penalty_curriculum: false
  reward_initial_penalty_scale: 0.10
  reward_min_penalty_scale: 0.0
  reward_max_penalty_scale: 1.0
  reward_penalty_level_down_threshold: 40
  reward_penalty_level_up_threshold: 42
  reward_penalty_degree: 0.000003
  reward_penalty_reward_names:
    - penalty_torques
    - penalty_action_rate
    - limits_dof_pos
    - limits_torque
    - penalty_slippage
    - penalty_undesired_contact
    - penalty_ankle_roll
    - penalty_feet_ori
```

只有同时出现在 `reward_penalty_reward_names` 中的 reward 才会被 penalty curriculum 缩放。

### 5.2 FB/TeCH auxiliary reward

UFO 的 FB 和 TeCH agent 还有一套独立的 auxiliary reward 权重。`humanoidverse/agents/presets/fb.py` 提供 FB 默认值，Robot YAML 可以用 `training.agent.fb.aux_rewards_scaling` 做局部覆盖：

```python
aux_rewards_scaling = {
    "penalty_action_rate": -0.1,
    "penalty_feet_ori": -0.4,
    "penalty_ankle_roll": -4.0,
    "limits_dof_pos": -10.0,
    "penalty_slippage": -2.0,
    "penalty_undesired_contact": -1.0,
    "penalty_torques": 0.0,
    "limits_torque": 0.0,
}
```

Mini3 当前把 action-rate 从默认 `-0.1` 覆盖为 `-0.2`，把 ankle-roll 从默认 `-4.0` 覆盖为 `-1.0`，其余项继续继承 FB preset。启动日志会打印最终的完整 `aux_rewards_scaling`。

需要区分：

- `reward_bfm_zero.yaml` 或 Hydra override 控制环境返回的 scalar reward、reward curriculum 和相关日志。
- `fb.py` 默认值与 Robot YAML 的 `training.agent.fb.aux_rewards_scaling` 合成后，控制 FB auxiliary critic 使用的加权辅助奖励。
- `tldr.py` 中的对应配置控制 TeCH auxiliary reward。

因此，如果目的是改变 FB 实际学习时的辅助约束强度，应修改 Robot YAML 的 `training.agent.fb.aux_rewards_scaling`；仅修改 `rewards.reward_scales` 不会改变 FB auxiliary critic。`--cartwheel-aux-safe` 会继续排除不适合撑地技能的接触和足部姿态项，优先级高于 Robot YAML。

### 5.3 添加新的 reward

仅在 YAML 中加入一个新名字不会自动产生 reward。完整步骤是：

1. 在 `HumanoidVerseMjlabCore._compute_reward()` 中计算新的逐环境张量，并写入 `aux`。
2. 如果它需要进入环境 scalar reward，在 `reward_scales` 中加入权重。
3. 如果 FB/TeCH auxiliary critic 需要使用它，在对应 agent preset 的 `aux_rewards` 和 `aux_rewards_scaling` 中注册。
4. 添加数值、shape、机器人语义和异常输入测试。

Mini3 没有 G1 的 wrist/hand body；新增上肢 reward 时不能直接假设 G1 body 语义。

## 6. 域随机化配置

共享配置位于：

```text
humanoidverse/config/domain_rand/domain_rand.yaml
```

当前 MJLab 路径实际实现的域随机化如下：

| 配置 | 当前支持 | 执行方式 |
| --- | --- | --- |
| `push_robots` | 是 | 按 `push_interval_s` 周期施加随机基座速度 |
| `randomize_base_com` | 是 | 启动时对 torso COM 添加偏移 |
| `randomize_link_mass` | 是 | 启动时缩放全部机器人 body mass |
| `randomize_friction` | 是 | 启动时随机化机器人 geom 的第一维摩擦系数 |
| `randomize_default_dof_pos` | 是 | reset 时加入默认关节位置偏移 |
| `randomize_pd_gain` | 是 | 每次 episode reset 按环境和关节独立缩放 nominal Kp/Kd |
| `randomize_motor_strength` | 是 | 每次 reset 同步缩放 DC motor 连续和峰值力矩能力 |
| `randomize_ctrl_delay` | 是 | 启用 actuator 分组仿真步延迟；脚踝每次 reset 按环境采样，action FIFO 每个 physics substep 前进一步 |
| `randomize_torque_rfi` / `randomize_rfi_lim` | 否 | YAML 有字段，但当前 MJLab 路径未实现 |
| `randomize_base_mass` | 否 | YAML 有字段，但没有独立 MJLab 实现 |

PD gain 随机化使用 MJLab 的 `dr.pd_gains`，`kp_range` 和 `kd_range` 都是相对 Robot YAML nominal gain 的乘法系数，而不是绝对 Kp/Kd。每次 reset 都从 nominal gain 重新采样，因此不会在多个 episode 之间累乘。

Motor strength 随机化同样从 nominal actuator 参数重新采样，并用同一系数缩放 `effort_limit` 与 `saturation_effort`，保持 DC motor 连续力矩和峰值力矩一致。它随机化的是最大输出能力，不会额外缩放未饱和区间内的 PD 输出。

控制延迟使用逐环境 action FIFO。策略观测和 action-rate reward 仍记录策略刚发出的 command，MJLab actuator 接收延迟后的 command；reset 会清空该环境的历史，避免跨 episode 泄漏动作。

不要仅根据 YAML 中存在某个字段就判断它已经生效。对其余未实现项，需要在 `make_mjlab_ufo_env_cfg()` 或环境 step/reset 路径中补充实现和测试。

### 6.1 一次关闭域随机化

调试、基线对比或 smoke 时可以使用：

```bash
--disable-dr
```

该开关在 Hydra compose 之后执行，会强制关闭当前支持的域随机化，优先级高于 `training.hydra_overrides`。

观测噪声与域随机化是两个独立开关。关闭观测噪声使用：

```bash
--disable-obs-noise
```

## 7. Smoke training

每次修改 robot config、reward、域随机化或 agent auxiliary reward 后，都应使用新的工作目录运行 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 ./run_train.sh \
  --agent fb \
  --robot-config configs/robots/mini3.yaml \
  --data-manifest configs/data/mini3_pkl.yaml \
  --gpu-ids single \
  --smoke \
  --work-dir runs/ufo_fb_mini3_reward_v1_smoke
```

如果需要先排除域随机化和观测噪声的影响：

```bash
CUDA_VISIBLE_DEVICES=0 ./run_train.sh \
  --agent fb \
  --robot-config configs/robots/mini3.yaml \
  --data-manifest configs/data/mini3_pkl.yaml \
  --gpu-ids single \
  --smoke \
  --disable-dr \
  --disable-obs-noise \
  --work-dir runs/ufo_fb_mini3_deterministic_smoke
```

启动日志应确认：

- `robot=mini3`
- XML 指向 `mini3_mjlab/mini3.xml`
- actuator count 和 action dim 为 21
- joint order 与 Mini3 21 个 control joint 完全一致
- actuator source 为 `yaml`
- kp、kd、effort、velocity、armature 和 friction 符合预期
- observation 中没有残留 29-DOF 维度
- `aux_rewards_scaling` 显示 Mini3 的最终 FB 权重
- EventManager reset terms 包含 `random_pd_gains` 和 `random_motor_strength`
- 无 NaN、shape mismatch、脚底穿透、持续异常接触力或明显高频抖动

当前 Hydra 静态合成结果为：

- action dim：21
- actor observation dim：702
- critic observation dim：357

## 8. 正式训练

Smoke 通过后，可以启动单卡 FB 训练：

```bash
CUDA_VISIBLE_DEVICES=0 ./run_train.sh \
  --agent fb \
  --robot-config configs/robots/mini3.yaml \
  --data-manifest configs/data/mini3_pkl.yaml \
  --gpu-ids single \
  --work-dir runs/ufo_fb_mini3
```

当前默认值包括：

- `--num-envs 1024`
- `--num-env-steps 192000000`
- `--buffer-size 5120000`
- `--checkpoint-every-steps 3200000`
- FB 的 `--update-z-every-step 100`

正式训练前可以根据 smoke 的显存和吞吐结果调整 `--num-envs` 和 `--buffer-size`。

如需 W&B：

```bash
uv run wandb login
```

然后在训练命令中增加：

```bash
--use-wandb --wandb-run-name ufo_fb_mini3
```

## 9. 实验和 checkpoint 管理

- 每组配置使用独立的 `--work-dir`。
- 不要用修改后的 reward/域随机化参数直接复用旧实验目录，训练入口可能自动恢复已有 checkpoint 和 replay buffer。
- 建议在目录名中记录 agent、robot 和实验版本，例如 `runs/ufo_fb_mini3_dr_v2`。
- 保存每次实验最终使用的 Robot YAML、agent preset diff、启动命令和 smoke 日志。
- Mini3 checkpoint 不能与 G1 或其他 action/observation 维度不同的机器人互换。

## 10. 后续推理注意事项

- 训练使用约 10 秒裁剪片段。
- Tracking inference 建议使用未裁剪的 Mini3 full motion sequence。
- 推理必须使用 Mini3 自己训练的 checkpoint、Mini3 robot config 和 Mini3 motion。
- ONNX metadata 应记录 `robot_name=mini3`、`num_dof=21` 和完整 joint order。
- 实机部署还需要单独完成限幅、急停、低增益和吊装测试；训练完成不代表已经具备安全实机部署条件。
