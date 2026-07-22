# PICO 4 → JOYIn → Mini3 → UFO MuJoCo 遥操

## 1. 目标与实现状态

本项目新增了独立入口：

```text
humanoidverse/mujoco_pico_teleop.py
```

它完成以下实时链路：

```text
PICO 4 / Sonic ZMQ
        ↓
Sonic SMPL pose + root orientation
        ↓
标准 SMPL-X 参数（pose_body/root_orient/trans/betas）
        ↓
JOYIn SMPL-X body model + get_smplx_data()
        ↓
本地 JOYIn-Retarget：smplx_to_mini3
        ↓
Mini3 reference qpos
        ↓
在线 MuJoCo FK、速度差分、backward observation
        ↓
UFO Backward Map → latent z
        ↓
UFO policy + Mini3 MuJoCo controller
```

新的入口复用了 `mujoco_tracking_inference.py` 中的 checkpoint 加载、Mini3 MuJoCo 控制器、电机模型和 viewer，但没有修改原文件：

```text
humanoidverse/mujoco_tracking_inference.py
```

## 2. 支持的数据源

### 2.1 PICO 实时数据

脚本订阅 Sonic PICO manager 发布的 `pose` ZMQ 消息，默认地址为：

```text
tcp://127.0.0.1:5556
```

使用的主要字段为：

| 字段 | 含义 |
| --- | --- |
| `smpl_pose` | 21 个 SMPL-X body joint 的局部 axis-angle |
| `smpl_joints` | 24 个 Sonic 局部关节位置，仅用于协议校验，不作为 JOYIn 重定向输入 |
| `body_quat_w` | SMPL 根节点四元数，顺序为 `wxyz` |
| `timestamp_monotonic` | 可选的采样时间戳，用于实时速度差分 |

收到滑动窗口消息时，脚本使用窗口中的最新一帧。

### 2.2 PICO v2 离线数据

脚本也可以直接读取 `/home/amax/Desktop/v2` 中的录制文件：

```text
/home/amax/Desktop/v2/walking_v2.npz
/home/amax/Desktop/v2/running_v2.npz
/home/amax/Desktop/v2/pickup_v2.npz
```

离线模式使用 `sonic_smpl_pose` 和 `sonic_smpl_anchor_orientation` 重建标准 SMPL-X 参数；`sonic_smpl_joints` 不直接参与重定向。如果 NPZ 包含 `body_pos_w`，默认会恢复录制动作的根节点平移，因此 walking 和 running 不只是原地摆腿。

## 3. 安装遥操依赖

在 UFO 仓库根目录执行：

```bash
cd /home/amax/Desktop/robot/UFO
uv sync --extra pico-teleop
```

`pico-teleop` extra 包含：

- `mink`
- `qpsolvers[daqp]`
- `pyzmq`
- `loop-rate-limiters`
- `smplx`

JOYIn 源码不会复制到 UFO 中，运行时默认从下面的本地工程导入：

```text
/home/amax/Desktop/robot/JOYIn-Retarget
```

可通过 `--joyin-root` 指定其他路径。

## 4. 离线测试

以下命令使用 `runs/Revise_torque_limit` 中的 Mini3 checkpoint。若要测试其他 checkpoint，请替换 `--model-folder`。

### 4.1 Walking

```bash
cd /home/amax/Desktop/robot/UFO

.venv/bin/python -m humanoidverse.mujoco_pico_teleop \
  --model-folder runs/Revise_torque_limit \
  --pico-npz /home/amax/Desktop/v2/walking_v2.npz \
  --device cuda:0 \
  --loop
```

### 4.2 Running

```bash
.venv/bin/python -m humanoidverse.mujoco_pico_teleop \
  --model-folder runs/Revise_torque_limit \
  --pico-npz /home/amax/Desktop/v2/running_v2.npz \
  --device cuda:0 \
  --loop
```

### 4.3 Pickup

```bash
.venv/bin/python -m humanoidverse.mujoco_pico_teleop \
  --model-folder runs/Revise_torque_limit \
  --pico-npz /home/amax/Desktop/v2/pickup_v2.npz \
  --device cuda:0 \
  --loop
```

### 4.4 无界面快速检查

下面的命令适合检查数据、JOYIn、backward encoder 和 policy 是否能完整运行：

```bash
.venv/bin/python -m humanoidverse.mujoco_pico_teleop \
  --model-folder runs/Revise_torque_limit \
  --pico-npz /home/amax/Desktop/v2/walking_v2.npz \
  --device cpu \
  --headless \
  --max-steps 10 \
  --realtime false \
  --enable-real-motor false \
  --disable-action-delay \
  --disable-imu-delay \
  --log-every 1
```

## 5. 实时 PICO 遥操

下面两条是当前 PICO 实时遥操使用的完整命令。需要分别在两个终端中运行，并且先启动终端 1，看到 PICO/Sonic 服务开始工作后再启动终端 2。

### 5.1 启动 PICO/Sonic 服务

在第一个终端中进入 GR00T-WholeBodyControl：

```bash
cd /home/amax/Desktop/robot/GR00T-WholeBodyControl
source .venv_teleop/bin/activate

python gear_sonic/scripts/pico_manager_thread_server.py \
  --port 5556 \
  --target_fps 50 \
  --num_frames_to_send 5
```

这条命令使用脚本默认的单线程 pose streaming 模式，并通过 `tcp://127.0.0.1:5556` 发布最近 5 帧、目标频率为 50 Hz 的 Sonic pose 数据。

需要观察 PICO 骨架时，可以增加：

```text
--vis_vr3pt --vis_smpl
```

启动服务后：

1. 戴好 PICO 头显、控制器和身体追踪器。
2. 保持标定姿势。
3. 按 `A+B+X+Y` 完成初始化和标定。
4. 按 `A+X` 进入 POSE 模式。
5. POSE 模式开始后，manager 才会持续发布 `pose` 数据。

### 5.2 启动 Mini3 UFO MuJoCo 遥操

在第二个终端执行当前带首帧自动落地校正的命令：

```bash
cd /home/amax/Desktop/robot/UFO
source .venv/bin/activate

python -m humanoidverse.mujoco_pico_teleop \
  --model-folder runs/Revise_torque_limit \
  --pico-endpoint tcp://127.0.0.1:5556 \
  --device cuda:0 \
  --connect-timeout-ms 60000 \
  --auto-ground-retarget-reference true \
  --retarget-ground-height 0.0 \
  --root-z-offset 0.02
```

其中自动落地会根据第一帧 Mini3 reference 的足底最低点计算固定 z offset；`--root-z-offset 0.02` 会在自动落地后把实际仿真机器人额外抬高 `2 cm`。若不需要额外抬高，将它改为 `0.0`。

如果 PICO streamer 运行在另一台机器上，将地址改为：

```text
--pico-endpoint tcp://<PICO-streamer-IP>:5556
```

## 6. JOYIn 重定向处理

脚本严格复用本地 JOYIn 的 SMPL-X 数据入口和 Mini3 重定向器：

```python
GeneralMotionRetargeting(
    src_human="smplx",
    tgt_robot="mini3",
)
```

每帧按下面的顺序处理：

1. `smpl_pose[21,3]` 直接作为标准 SMPL-X 的 63 维 `pose_body`。
2. 对 Sonic 已移除 SMPL base rotation 的 `body_quat_w` 做逆变换，恢复标准 z-up `root_orient`。
3. 离线数据使用录制根位移作为 `trans`；实时 Sonic 未发布 pelvis 位移，因此显式使用零 `trans`。
4. Sonic 未发布人体形状，因此显式使用 neutral SMPL-X 的 16 维零 `betas`，默认人体高度按 JOYIn 公式得到 `1.66 m`；可用 `--actual-human-height` 覆盖重定向缩放。
5. 调用 JOYIn 工程中的 `SMPLX_NEUTRAL.pkl` body model 做前向计算，再调用 JOYIn 的 `get_smplx_data()` 生成全局关节位置和旋转。Sonic 的 `smpl_joints` 不参与这一步。
6. 将 JOYIn `get_smplx_data()` 的原生输出直接送入 `GeneralMotionRetargeting.retarget()`，由 `smplx_to_mini3.json` 生成 Mini3 qpos。
7. 按关节名称将 JOYIn qpos 映射到 UFO checkpoint 的 21 个 Mini3 control joints，不依赖裸索引顺序。
8. 默认使用 `offset_to_ground=True`，让脚部目标保持在地面附近。
9. 启动时对首帧 Mini3 qpos 做一次 MuJoCo 前向运动学，计算左右脚碰撞 mesh 的最低顶点，得到固定的 `z_offset = ground_z - lowest_foot_z`。之后所有 reference 根高度都加同一个 offset，不做逐帧抖动式校正。

## 7. Mini3 reference 到 UFO latent z

JOYIn 每帧输出：

```text
root_pos[3] + root_quat_wxyz[4] + dof_pos[21]
```

在线 reference encoder 使用 UFO 的 Mini3 XML 做 FK，并生成：

- 21 维相对关节位置；
- 21 维关节速度；
- 3 维 projected gravity；
- 3 维根角速度；
- 357 维全身 privileged state。

Mini3 backward observation 的 `state` 共 48 维：

```text
21 dof_pos + 21 dof_vel + 3 gravity + 3 root_ang_vel
```

根角速度通过训练侧的 `reference_base_ang_vel()` 处理，因此会使用 checkpoint 中保存的 `obs_scales.base_ang_vel`，避免实时和离线预处理不一致。

随后执行：

```python
z = policy.project_z(policy.backward_map(backward_observation))
```

默认对最近 3 帧 latent 使用 `gamma=0.8` 的加权平滑。若 checkpoint 启用了 `norm_z`，平滑后会重新归一化到 `sqrt(z_dim)`；当前 256 维 FB latent 的目标范数为 16。

## 8. MuJoCo Viewer 操作

Viewer 默认同时显示两台 Mini3：

- 原材质机器人：UFO policy 实际控制的 MuJoCo 机器人；
- 青色半透明机器人：JOYIn 当前输出的 Mini3 reference qpos。

reference 的根部 XY 跟随实际机器人，并默认沿 Y 方向偏移 `1.0 m`，根高度、根姿态和 21 个关节角仍来自 JOYIn。这样可以在机器人发生漂移时继续并排比较目标姿态与策略执行结果。

| 按键 | 功能 |
| --- | --- |
| `Space` | 暂停或继续 |
| `R` | 使用最新 Mini3 reference 重置仿真 |
| `F` | 开关相机跟随 |
| `C` | 显示接触点和接触力 |
| `Q` | 退出 |

## 9. 常用参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--pico-endpoint` | `tcp://127.0.0.1:5556` | 实时 Sonic 地址 |
| `--joyin-root` | `/home/amax/Desktop/robot/JOYIn-Retarget` | 本地 JOYIn 工程 |
| `--smplx-device` | 与 `--device` 相同 | SMPL-X body model 运行设备；显存不足时可设为 `cpu` |
| `--show-retarget-reference` | `true` | 在实际机器人旁显示青色 Mini3 reference |
| `--retarget-visual-lateral-offset` | `1.0` | reference 相对实际机器人的 Y 方向偏移，单位 m |
| `--retarget-visual-alpha` | `0.45` | reference 透明度，范围 `(0,1]` |
| `--auto-ground-retarget-reference` | `true` | 首帧根据 Mini3 两脚碰撞 mesh 计算固定 z offset，并应用到所有 reference |
| `--retarget-ground-height` | `0.0` | reference 足底目标地面高度，单位 m |
| `--root-z-offset` | `0.0` | reference 落地后仅对实际仿真机器人的额外根高度调整，单位 m |
| `--physics-hz` | `500` | 必须与 checkpoint 一致 |
| `--policy-hz` | `50` | 必须与 checkpoint 一致 |
| `--latent-window` | `3` | latent 平滑窗口 |
| `--latent-gamma` | `0.8` | 越大越重视旧帧 |
| `--max-pico-stale-seconds` | `0.5` | 实时输入超时保护 |
| `--joyin-offset-to-ground` | `true` | JOYIn 逐帧地面对齐 |
| `--enable-recorded-root-motion` | `true` | 离线 NPZ 恢复根平移 |
| `--enable-real-motor` | `true` | 启用 Mini3 电机响应模型 |
| `--headless` | `false` | 不启动 MuJoCo viewer |

查看完整参数：

```bash
.venv/bin/python -m humanoidverse.mujoco_pico_teleop --help
```

## 10. 验证结果

已完成以下检查：

- walking、running、pickup 共 1390 帧全部通过 JOYIn；
- JOYIn 输出的 root pose 和 21 个关节均无 NaN/Inf；
- 首帧 Mini3 左右脚碰撞 mesh 的最低点用于一次性 z 标定，校正后的首帧最低点与 `z=0` 对齐；
- CPU 上包含 SMPL-X body model 前向计算的 JOYIn 重定向平均约 `8.06 ms/帧`，99 分位约 `19.76 ms`；运行时完整编码耗时以日志中的 `retarget=...ms` 为准；
- walking 和 running 均使用真实 Mini3 checkpoint 完成端到端 MuJoCo rollout；
- 在线 latent 范数稳定为 `16`；
- PICO/Sonic 协议、6D 旋转、标准 SMPL-X 参数恢复、NPZ 回放、backward observation 和 latent 平滑单测全部通过；
- `mujoco_tracking_inference.py` 保持未修改。

运行相关单测：

```bash
.venv/bin/python -m unittest \
  tests.test_convert_pico_motion_clip \
  tests.test_mujoco_pico_teleop \
  -v
```

## 11. 当前限制

### 11.1 实时根平移

当前 Sonic `pose` 消息提供的是 root-local SMPL skeleton，没有提供 PICO 用户在房间中的绝对 pelvis 平移。因此：

- 实时模式可以跟踪全身姿态和迈步模式；
- 实时模式不能从现有消息恢复 room-scale 绝对根位移；
- 离线 walking/running 会从 `body_pos_w` 恢复根位移。

如果实时模式也需要绝对根平移，需要扩展 `pico_manager_thread_server.py` 的 `pose` 协议，额外发布经过坐标转换的 pelvis/root translation。

### 11.2 实时人体形状

当前 Sonic `pose` 消息没有发布 SMPL-X `betas`。实时模式使用 JOYIn neutral 模型的 16 维零 `betas`。若需要匹配具体操作者体型，应扩展协议发送标定后的 `betas`，或增加本地标定文件。

### 11.3 硬件验证

离线 PICO 数据、JOYIn、UFO checkpoint 和 MuJoCo 已完成端到端验证。真实 PICO 头显的最终延迟、丢帧和追踪质量仍需要在 PICO/Sonic 服务实际运行时测试。
