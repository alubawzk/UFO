# BFM-Zero Actor ONNX 与 Mini3 Latent 前瞻对比

本文整理 BFM-Zero 官方 `deploy` 分支中的两部分实现：

1. Actor ONNX 的输入、输出与网络结构。
2. 官方 tracking latent 前瞻与当前 Mini3 部署的差异。

## 1. Actor ONNX 输入与网络结构

### 1.1 模型基本信息

对本地官方模型
`/home/amax/Desktop/robot/BFM-Zero/model/exported/FBcprAuxModel.onnx`
的 ONNX 图检查结果如下：

| 项目 | 内容 |
|---|---|
| 输入名称 | `actor_obs` |
| 输入形状 | `[1, 721]` |
| 输出名称 | `action` |
| 输出形状 | `[1, 29]` |
| Batch | 固定为 1 |
| 参数量 | 31,907,075 |
| 文件大小 | 约 122.6 MB |

该 ONNX 只包含观测归一化和 Actor，不包含 backward map、forward
map、critic 或 discriminator。

### 1.2 输入组成

`actor_obs` 由 465 维机器人观测和 256 维 latent `z` 拼接而成：

| 输入部分 | 维度 | 具体内容 |
|---|---:|---|
| `state` | 64 | 关节位置偏差 29、关节速度 29、重力投影 3、机身角速度 3 |
| `last_action` | 29 | 上一步经过 `action_rescale=5` 后的策略动作 |
| `history_actor` | 372 | 4 帧动作、角速度、关节位置、关节速度和重力投影 |
| `z` | 256 | tracking、reward 或 goal latent |
| 合计 | 721 | `64 + 29 + 372 + 256` |

其中：

```text
state =
    dof_pos_minus_default[29]
  + dof_vel[29]
  + projected_gravity[3]
  + base_ang_vel[3]
```

历史观测的布局不是逐帧交错，而是按观测类型分块：

```text
history_actor =
    action_history[4 × 29]
  + base_ang_vel_history[4 × 3]
  + dof_pos_history[4 × 29]
  + dof_vel_history[4 × 29]
  + projected_gravity_history[4 × 3]
```

每个历史块内部均按 `newest → oldest` 排列。

官方配置中：

- 关节位置、关节速度、重力投影和动作的手工 scale 为 `1.0`。
- 当前和历史机身角速度的 scale 为 `0.25`。
- ONNX 内部还包含 `state`、`last_action` 和 `history_actor` 的 running
  mean/variance 标准化。
- `z` 不经过上述 observation normalizer，应在输入前保持训练时约定的
  范数；256 维 `z` 的目标范数是 `sqrt(256)=16`。

### 1.3 Actor 网络结构

Actor 不是 Transformer，而是双分支残差 MLP：

```text
上下文分支：
    [proprio(465), z(256)]
        721 → 2048 → 1024

状态分支：
    proprio(465)
        465 → 2048 → 1024

两分支拼接：
    1024 + 1024 → 2048
        → 6 个 2048 维残差 MLP block
        → LayerNorm
        → Linear(2048, 29)
        → Tanh
        → action[29]
```

网络中的非线性主要为 Mish；ONNX 图中表现为
`x * tanh(softplus(x))`。

### 1.4 动作解释

Actor 输出经过如下处理后转换为关节位置目标：

```text
raw_action   = clip(actor_output, -1, 1)
policy_action = 5 × raw_action
q_des[j]     = q_default[j] + policy_action[j] × action_scale[j]
q_des[j]     = clip(q_des[j], joint_lower[j], joint_upper[j])
```

下一控制周期输入 Actor 的 `last_action` 是 `policy_action`，即乘以 5
之后、乘以各关节 `action_scale` 之前的动作。

相关官方源码：

- [Actor 部署入口](https://github.com/LeCAR-Lab/BFM-Zero/blob/deploy/rl_policy/bfm_zero.py)
- [观测与动作配置](https://github.com/LeCAR-Lab/BFM-Zero/blob/deploy/config/policy/motivo_newG1.yaml)
- [观测构造](https://github.com/LeCAR-Lab/BFM-Zero/blob/deploy/rl_policy/observations/bfm_zero.py)

## 2. 官方部署与 Mini3 Latent 前瞻差异

### 2.1 BFM-Zero 官方 tracking

官方 tracking 配置为：

```yaml
gamma: 0.8
window_size: 3
```

部署端在第 `t` 个控制周期读取：

```python
window = ctx[t:t + 3]
```

因此参与计算的是：

```text
z_t, z_{t+1}, z_{t+2}
```

加权和重新归一化可以写为：

```text
z_actor(t) = normalize_to_16(
    z_t + 0.8 × z_{t+1} + 0.64 × z_{t+2}
)
```

这属于未来前瞻，不是过去帧平滑。控制频率为 50 Hz 时：

- 最远使用未来 2 帧，即未来 40 ms 的 latent。
- 加权中心约为未来 0.852 帧，即约 17 ms。

官方部署不在线运行 backward encoder，而是加载预先计算好的
`[sequence_length, 256]` latent 序列。

相关官方源码：

- [Tracking 配置](https://github.com/LeCAR-Lab/BFM-Zero/blob/deploy/config/exp/tracking/walking.yaml)
- [Tracking 窗口计算](https://github.com/LeCAR-Lab/BFM-Zero/blob/deploy/rl_policy/bfm_zero.py)

### 2.2 当前 Mini3 部署

当前 Mini3 部署首先用在线 backward ONNX 将最新参考状态编码为
`raw_z`，然后将其插入 `latent_history_` 的队首：

```text
latent_history_ =
    z_t, z_{t-1}, z_{t-2}
```

随后计算：

```text
z_actor(t) = normalize_to_16(
    z_t + 0.8 × z_{t-1} + 0.64 × z_{t-2}
)
```

这属于因果的过去帧平滑，会抑制抖动，但也会引入动作滞后。

对应实现位置：

- [Mini3 latent 平滑](/home/amax/Desktop/robot/RZ_mini3_deploy/src/inference/src/teleop_interface.cpp#L174)
- [Mini3 latent 配置](/home/amax/Desktop/robot/RZ_mini3_deploy/src/inference/config/inference_teleop.yaml#L78)
- [参考状态发布端](/home/amax/Desktop/robot/RZ_mini3_deploy/scripts/teleop_backward_obs_publisher.py#L124)

### 2.3 两者对比

| 项目 | BFM-Zero 官方 tracking | 当前 Mini3 teleop |
|---|---|---|
| `z` 的产生位置 | 离线预计算 | 在线 backward ONNX |
| 窗口内容 | `z_t,z_{t+1},z_{t+2}` | `z_t,z_{t-1},z_{t-2}` |
| 窗口方向 | 未来前瞻 | 过去平滑 |
| 权重 | `1,0.8,0.64` | `1,0.8,0.64` |
| 平滑后范数 | 16 | 16 |
| 是否需要未来数据 | 是 | 否 |
| 主要效果 | 提供动作趋势信息 | 降低噪声，但增加延迟 |

### 2.4 对 Mini3 的影响

#### 离线动作或 NPZ 回放

离线数据已知未来参考状态，因此应使用和官方相同的未来窗口：

```text
z_t, z_{t+1}, z_{t+2}
```

比较稳妥的实现方式是：

1. 预先对完整参考动作计算逐帧 `z`。
2. 在第 `t` 帧使用 `z[t:t+3]` 加权。
3. 将当前 C++ 过去帧 latent smoother 设为 `window=1`，避免重复平滑。

#### 实时 PICO 遥操

实时遥操无法直接获得未来两帧人体姿态，因此不能无延迟地复现官方
tracking 前瞻。当前过去帧平滑是一个因果近似，但会降低响应速度和跟踪
精度。

建议至少进行以下 A/B 测试：

```text
A: latent_window=1
B: latent_window=3, gamma=0.8
```

如果 `window=1` 的快速动作跟踪明显更准，而抖动仍可接受，说明过去窗口
引入的滞后是当前跟踪误差的重要来源。

如果实时遥操也需要类似官方的未来前瞻，可选方案包括：

1. 引入 2 帧缓冲延迟，以延迟后的时间点为控制目标。
2. 对人体参考状态或 latent 做短时预测。
3. 训练时加入同样的因果延迟和平滑，使策略适应部署链路。

当前 ROS2 发布端和订阅端都采用 `KeepLast(1)`，只保存最新一帧参考状态。
因此，对于离线回放，若要使用未来窗口，需要在发布端预计算 latent 序列
或扩展消息格式；仅修改 C++ 中的权重顺序无法获得未来数据。
