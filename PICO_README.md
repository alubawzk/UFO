# PICO Motion Clips v2

用 PICO VR 录制的全身动作片段（`.npz`），可直接给 `run_motion_replay_service.py` 回放，作为无硬件时的假设备输入。

来源标记：`source = pico_motion_clip`  
坐标版本：`pico_position_axes_version = 3`（统一 Pico 坐标约定；低于 3 的旧 clip 不会自动校正，建议重录）

## 文件一览

| 文件 | 帧数 | 标称时长 | 内容 |
|------|------|----------|------|
| `pickup_v2.npz` | 466 | ~9.3 s | 拾取 |
| `running_v2.npz` | 462 | ~9.2 s | 跑步 |
| `walking_v2.npz` | 462 | ~9.2 s | 走路 |

三者 schema 相同，标称 **50 fps**（`dt = 0.02`）。

## 快速回放

在仓库根目录激活 `teleop` 环境后：

```powershell
.\teleop\activate.ps1
python .\teleop\run_motion_replay_service.py `
  .\teleop\demo_data\pico_motion_clips\v2\pickup_v2.npz `
  --port 5555
```

可选参数：

- `--no-loop`：播完停止（默认循环）
- `--port`：ZMQ PUB 端口（默认 `5555`）

另一终端再接 FlowMimic / Sonic 等策略，`--device-source tcp://127.0.0.1:5555`。完整命令见 [`teleop/README.md`](../../../README.md) 第 4 节。

可替换为 `walking_v2.npz` 或 `running_v2.npz`。

## 数据字段

用 `numpy` 加载：

```python
import numpy as np

data = np.load("pickup_v2.npz", allow_pickle=True)
print(list(data.keys()))
print(data["body_pos_w"].shape)   # (T, 5, 3)
print(data["body_names"])
```

### 核心轨迹（回放主输入）

| Key | Shape / Type | 说明 |
|-----|--------------|------|
| `body_pos_w` | `(T, 5, 3)` float32 | 世界系位置 |
| `body_quat_w` | `(T, 5, 4)` float32 | 世界系四元数，**wxyz** |
| `body_names` | `(5,)` str | 与第二维顺序一致 |
| `fps` | scalar float32 | 标称帧率，v2 为 `50` |
| `dt` | scalar float32 | `1 / fps` |

`body_names` 固定为稀疏跟踪点（对应 FlowMimic `feet_hands_vr`）：

```text
pelvis
left_ankle_roll_link
right_ankle_roll_link
left_wrist_yaw_link
right_wrist_yaw_link
```

运行时 `MotionReplayDevice` 会把 `body_quat_w` 的 wxyz 转成 `BodyState` 的 xyzw。

### 元数据

| Key | 值 | 说明 |
|-----|-----|------|
| `source` | `pico_motion_clip` | 标识为 teleop 录制的 PICO clip |
| `pico_position_axes_version` | `3` | 坐标轴约定版本 |
| `body_state_frame` | `g1_robotics_zup_v1` | 稀疏 body 位姿坐标系 |
| `pico_body_joints_frame` | `xrobotoolkit_raw_v1` | 原始 Pico 关节坐标系 |
| `sonic_smpl_frame` | `sonic_root_local_v1` | Sonic SMPL 坐标系 |
| `timestamp_monotonic` | `(T,)` float64 | 录制时 monotonic 时间戳 |

### 附加流（Sonic / 原始 Pico）

| Key | Shape | 说明 |
|-----|-------|------|
| `pico_body_joints` | `(T, 24, 7)` | 原始 Pico 24 关节（xyz + quat） |
| `sonic_smpl_pose` | `(T, 21, 3)` | Sonic SMPL 姿态 |
| `sonic_smpl_joints` | `(T, 24, 3)` | Sonic SMPL 关节位置 |
| `sonic_smpl_anchor_orientation` | `(T, 6)` | 锚点 6D 朝向 |
| `sonic_smpl_wrist_joint_pos` | `(T, 6)` | 腕关节位置 |
| `pico_body_timestamp_ns` | `(T,)` int64 | 设备时间戳；本批 v2 为全 0 |

回放时优先级：

1. 若有 `sonic_smpl_joints` → 走 Sonic SMPL 路径  
2. 否则若有 `pico_body_joints` → 走原始 Pico 路径  
3. 否则从 G1 body FK 合成近似 SMPL（仅 smoke test）

## 自己检查一份 NPZ

```python
import numpy as np

path = "teleop/demo_data/pico_motion_clips/v2/pickup_v2.npz"
with np.load(path, allow_pickle=True) as d:
    T, B, _ = d["body_pos_w"].shape
    assert B == len(d["body_names"])
    assert d["body_quat_w"].shape == (T, B, 4)
    assert str(d["source"]) == "pico_motion_clip"
    assert int(d["pico_position_axes_version"]) == 3
    print(f"ok: {T} frames, bodies={list(d['body_names'])}, fps={float(d['fps'])}")
```

## 录制新 clip

需要 WSL2 + PICO 环境（见 `teleop/README.md`）。录制时建议打开 Sonic SMPL，以便写出与本目录 v2 相同的附加字段：

```bash
python teleop/run_pico_service.py \
  --no-waist \
  --pico-calibration teleop/configs/pico_g1_calibration.json \
  --robot-urdf assets/unitree_description/urdf/g1/main.urdf \
  --publish-sonic-smpl \
  --record-motion-clip teleop/demo_data/pico_motion_clips/my_clip.npz \
  --record-duration-sec 10
```

录制前默认倒计时 3 秒。完成后在 Windows 上回放：

```powershell
python .\teleop\run_motion_replay_service.py `
  .\teleop\demo_data\pico_motion_clips\my_clip.npz `
  --port 5555
```

写出逻辑见 `teleop/devices/recording.py`（`MotionClipRecorder`）。

## 注意事项

- 本目录 clip **不含** `joint_pos`；回放依赖 `body_pos_w` / `body_quat_w` 与 Sonic/Pico extras。
- `body_quat_w` 存 **wxyz**；不要直接当 xyzw 用。
- 旧版（`pico_position_axes_version < 3`）坐标约定不同，请用当前 `run_pico_service.py` 重录。
- 策略侧若使用 `feet_hands_vr`，应与上述 5 个 `body_names` 对齐。
