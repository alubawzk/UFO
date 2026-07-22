# PICO 数据进入 BFM Policy 的链路

## 核心结论

PICO 人体动作不会直接作为 BFM policy 的 observation。系统先将 PICO/XR 人体姿态重定向为 G1 姿态，再使用 BFM 的 backward encoder 将目标姿态编码成 256 维 latent `z`。最终 policy 执行的是：

```text
PICO 人体姿态 → GMR 重定向 → G1 目标姿态 → Backward Map → z
机器人传感器状态 ───────────────────────────────────────┐
                                                        ├→ π(a | s, z) → 29 维关节动作
z ──────────────────────────────────────────────────────┘
```

因此，PICO 决定“机器人要表现什么技能”，机器人本体观测描述“机器人当前处于什么状态”。PICO 数据在推理时只通过 `z` 条件影响 policy，不会在线更新 policy 权重。

> 当前工作区检出的是 `main` 训练分支；PICO 实时遥操作实现位于 `origin/deploy` 分支。

## 1. PICO 姿态重定向为 G1 姿态

入口是 `origin/deploy:scripts/teleop/xrobot_teleop_to_pose_zmq_server.py`：

1. XRoboToolkit 回调接收 PICO 全身跟踪姿态。
2. `_body_poses_to_pose_dict()` 完成坐标系转换。
3. `GeneralMotionRetargeting(src_human="xrobot", tgt_robot="unitree_g1")` 将人体姿态重定向到 G1。
4. 输出 36 维 G1 `qpos`：

```text
root_pos[3] + root_quat_wxyz[4] + dof_pos[29]
```

遥操作桥通过 ZMQ 请求/应答通道 `28701/28702` 发送 `root_pos`、`root_quat` 和 `dof_pos`。短时缓存和插值用于降低抖动。

## 2. G1 目标姿态编码为 latent z

`origin/deploy:scripts/realtime/realtime_z_server.py` 以 50 Hz 获取最新 G1 姿态。`OnlineZInferer.step()` 执行以下处理：

- 通过 MuJoCo FK 计算全身刚体位置和旋转；
- 通过相邻帧差分计算 `dof_vel`、`body_vel` 和 `body_ang_vel`；
- 构造 backward observation；
- 调用 `backward_encoder.onnx` 得到 256 维 `z`；
- 通过 `tcp://*:28711` 发布 `float32` 字节流。

Backward Map 的 `state` 为 64 维：

| 字段 | 维度 |
| --- | ---: |
| `dof_pos - default_dof_pos` | 29 |
| `dof_vel` | 29 |
| `projected_gravity` | 3 |
| `root_ang_vel` | 3 |

另一个输入 `privileged_state` 包含全身局部位置、6D 旋转、线速度和角速度。默认 FB 配置的 Backward Map 只使用 `state + privileged_state`，见 [`agents/presets/fb.py`](../humanoidverse/agents/presets/fb.py)；`last_action` 虽然保留在 ONNX 接口中，但默认不会进入 B 网络。Backward encoder 同时包含 observation normalizer、Backward Map 和 `project_z`，见 [`export/backward_encoder.py`](../humanoidverse/export/backward_encoder.py)。

## 3. z 与机器人观测一起进入 policy

`origin/deploy:rl_policy/ufo_policy.py` 订阅 `28711`，验证 `z` 必须是 256 个有限的 `float32`。`config/exp/tracking/teleop.yaml` 默认使用最近 3 帧、`gamma=0.8` 做加权平滑，再将结果归一化到 `ctx_norm_ref=16`。

Policy ONNX 的实际输入为：

| 输入 | 维度 |
| --- | ---: |
| 当前关节位置、速度、重力、角速度和上一动作 | 93 |
| 4 帧本体观测历史 | 372 |
| PICO 目标对应的 `z` | 256 |
| **合计** | **721** |

```python
actor_input = np.concatenate([robot_observation, z], axis=-1)  # [1, 721]
action = policy_onnx(actor_input)                              # [1, 29]
```

训练侧 actor 使用 `state + last_action + history_actor + z`，ONNX 导出时也按相同顺序拆分输入，见 [`utils/helpers.py`](../humanoidverse/utils/helpers.py)。输出经过 `[-1, 1]` 裁剪、`action_rescale`、逐关节 action scale、默认关节角叠加、关节限位和 slew-rate 限制，最终形成 G1 的 `q_target`。

## 4. PICO 按键是独立控制通道

PICO 控制器按键不进入神经网络。`28703` 通道负责 `z` 的 follow/freeze；可选的 `28704` PUB 通道控制 policy 初始化、启用、停止和复位。按键只改变运行状态机和安全逻辑，不属于 721 维 actor 输入。

## 5. 分支间的观测对齐风险

当前 `main` 在构造 backward observation 时将根角速度按 `obs_scales.base_ang_vel=0.25` 缩放，见 [`utils/helpers.py`](../humanoidverse/utils/helpers.py) 和 [`utils/reference_observations.py`](../humanoidverse/utils/reference_observations.py)。当前 `origin/deploy` 的 realtime `z` server 则直接使用未缩放的 `root_ang_vel`。该差异会影响 Backward Map 生成的 `z`。

部署前应使用 `origin/deploy:tools/check_realtime_z_obs_alignment.py`，用同一段离线动作和同一个 `backward_encoder.onnx` 对比离线/实时的 `state`、`privileged_state` 与 `z`，确认预处理、四元数顺序、角速度坐标系和缩放完全一致。
