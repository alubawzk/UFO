# PICO sim2sim standalone runtime

本目录把 PICO 4 → Sonic → SMPL-X body → JOYIn → Mini3 所需的代码和安装逻辑
收进 UFO 仓库。完成一次安装后，运行过程不依赖本机的
`GR00T-WholeBodyControl` 或 `JOYIn-Retarget`。

## 目录

```text
pico_sim2sim/
├── sonic_server.py             # PICO/XRT → Sonic 兼容 ZMQ publisher
├── sonic_protocol.py           # 1280-byte header 二进制协议
├── smplx_model.py              # 真实 SMPL-X body model 适配器
├── smplx/
│   └── SMPLX_NEUTRAL.pkl       # 真实 neutral 模型（Git LFS）
├── joyin/
│   ├── retarget.py             # JOYIn 两阶段 Mink GMR
│   ├── smplx_to_mini3.json     # JOYIn IK mapping
│   └── mini3_ik.xml            # 无 mesh 的 Mini3 IK 模型
├── xrobotoolkit_binding/       # 最小 XRT Python binding 源码
├── install.sh                  # PC 环境一键安装
├── doctor.py                   # 安装检查
└── third_party/                # 来源、修改说明和许可证
```

## 安装

支持 x86_64 Ubuntu 22.04 和 24.04：

```bash
bash pico_sim2sim/install.sh
.venv/bin/python -m pico_sim2sim.doctor
```

安装脚本需要 `sudo` 安装系统包和 XRoboToolkit PC Service，也需要网络下载
官方 v1.0.0 源码和 deb。生成目录 `.build/`、`.downloads/`、`native/` 已加入
本目录的 `.gitignore`。系统没有 `uv` 时，脚本会把固定版本的 `uv` 安装到
`pico_sim2sim/.tools/`，不会要求预先配置 Python 3.10；`uv sync` 会按项目声明
准备 Python 和 `.venv`。

`SMPLX_NEUTRAL.pkl` 约 519 MiB，使用 Git LFS 管理。安装脚本会安装
`git-lfs` 并检查 clone 后取得的是实际模型而不是 LFS pointer。该模型受
SMPL-X 自身许可约束；推送或分发仓库前请确认你的使用方式符合许可。

UFO policy checkpoint 不属于 Gear Sonic/JOYIn 环境。当前
`runs/Revise_torque_limit` 约 3.3 GB，且 `runs/` 被仓库忽略；新电脑还需要复制
该目录，或者使用后续发布的 checkpoint 下载地址。`doctor` 会检查它是否齐全。

PICO 设备端需另行安装官方 `XRoboToolkit-PICO-1.1.1.apk`，打开应用、连接 PC
IP，并按应用提示完成身体追踪器标定。

## 实时运行

终端 1：

```bash
source .venv/bin/activate
python -m pico_sim2sim.sonic_server \
  --port 5556 \
  --target-fps 50 \
  --num-frames-to-send 5
```

终端 2：

```bash
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

## 离线检查

离线运行不需要 XRoboToolkit 原生组件：

```bash
uv sync --extra pico-teleop
.venv/bin/python -m humanoidverse.mujoco_pico_teleop \
  --model-folder runs/Revise_torque_limit \
  --pico-npz /path/to/walking_v2.npz \
  --device cpu \
  --headless \
  --max-steps 10 \
  --realtime false \
  --enable-real-motor false \
  --disable-action-delay \
  --disable-imu-delay
```

完整设计、参数和限制见 `docs/PICO_MINI3_MUJOCO_TELEOP.md`。
