# RZ_mini3_deploy 修改清单

目标工程：

```text
/home/amax/Desktop/robot/RZ_mini3_deploy
```

核心修改只有两项：

1. 实时 PICO 遥操默认只使用当前 latent，避免过去三帧平滑带来的滞后；
2. 离线 NPZ 动作单独实现当前帧加未来两帧的 latent 前瞻。

---

## 1. 这些内容保持不动

当前 Mini3 Actor 输入组织是正确的，不要为了修改 latent 前瞻而改动。

```text
actor_obs =
    state(48)
  + last_action(21)
  + history_actor(276)
  + z(256)
  = 601
```

保持以下配置：

```yaml
teleop_state_dim: 48
teleop_last_action_dim: 21
teleop_privileged_state_dim: 358
teleop_history_actor_dim: 276
teleop_actor_obs_dim: 601
teleop_z_dim: 256
teleop_num_actions: 21
```

保持 `state` 顺序：

```text
[dof_pos, dof_vel, projected_gravity, base_ang_vel]
```

保持 `history_actor` 顺序：

```text
[
    actions_history,
    base_ang_vel_history,
    dof_pos_history,
    dof_vel_history,
    projected_gravity_history,
]
```

保持 history 时序：

```text
先 query 历史
→ 组成 actor_obs
→ 再 push 当前帧
```

保持当前动作处理：

```text
current_action = clip(raw_action × 8, -8, 8)
target = default_angle + current_action × 0.25
```

---

## 2. 需要修改的文件

| 文件 | 修改内容 |
|---|---|
| `src/inference/config/inference_teleop.yaml` | 增加 latent mode；实时默认 `window=1`；增加离线前瞻配置 |
| `src/inference/include/teleop_interface.hpp` | 增加 mode 配置和三种 latent 更新接口 |
| `src/inference/src/teleop_interface.cpp` | 拆分当前帧、过去平滑、未来前瞻；抽出统一归一化 |
| `src/inference/src/ros_interface.cpp` | 读取新增 YAML 配置 |
| `scripts/teleop_backward_obs_publisher.py` | 离线 NPZ 不再只提供当前帧，需要提供完整 latent 序列或未来窗口 |
| `src/inference/src/inference_node.cpp` | 实时单帧和离线前瞻使用不同数据入口 |
| 建议新增 `scripts/precompute_teleop_latents.py` | 将离线 NPZ 预计算为 `[T,256]` latent 序列 |

---

## 3. 修改 inference_teleop.yaml

文件：

```text
src/inference/config/inference_teleop.yaml
```

建议改为：

```yaml
# live_current / live_causal_smooth / offline_lookahead
teleop_latent_mode: "live_current"

# 实时模式
teleop_latent_window: 1
teleop_latent_gamma: 0.8
teleop_latent_norm_z: true

# 离线动作模式
offline_latent_window: 3
offline_latent_gamma: 0.8
```

三种模式：

```text
live_current:
    只使用 z[t]

live_causal_smooth:
    使用 z[t], z[t-1], ...

offline_lookahead:
    使用 z[t], z[t+1], ...
```

实时 PICO 首先使用：

```yaml
teleop_latent_mode: "live_current"
teleop_latent_window: 1
```

这是当前最值得先验证的修改。

---

## 4. 修改 teleop_interface.hpp

文件：

```text
src/inference/include/teleop_interface.hpp
```

在 `TeleopConfig` 中增加：

```cpp
std::string latent_mode = "live_current";
int offline_latent_window = 3;
float offline_latent_gamma = 0.8f;
```

将当前单一的：

```cpp
void smoothLatent(const std::vector<float>& raw_z);
```

拆分为：

```cpp
void setCurrentLatent(const std::vector<float>& raw_z);

void smoothCausalLatent(const std::vector<float>& raw_z);

void setLookaheadLatents(
    const std::vector<std::vector<float>>& future_zs);

void normalizeLatent(std::vector<float>& z) const;
```

含义：

```text
setCurrentLatent:
    实时模式，只使用当前 z

smoothCausalLatent:
    当前已有的过去 latent 平滑

setLookaheadLatents:
    离线模式，接收 z[t], z[t+1], z[t+2]

normalizeLatent:
    将 256 维 z 归一化到范数 16
```

---

## 5. 修改 teleop_interface.cpp

文件：

```text
src/inference/src/teleop_interface.cpp
```

### 5.1 抽出统一归一化

把当前 `smoothLatent()` 内的 norm 逻辑抽成 `normalizeLatent()`。

必须满足：

```text
z_dim = 256
||z||₂ = sqrt(256) = 16
```

当前帧、过去平滑和未来前瞻最终都调用同一个归一化函数。

### 5.2 当前帧模式

`setCurrentLatent()` 应执行：

```text
current_z_ = raw_z
→ normalizeLatent(current_z_)
→ has_latent_ = true
```

它不读写 `latent_history_`。

### 5.3 过去平滑模式

将当前 `smoothLatent()` 重命名为：

```cpp
smoothCausalLatent()
```

它继续处理：

```text
z[t], z[t-1], z[t-2]
```

该函数只能用于 `live_causal_smooth`，不能用于离线前瞻。

### 5.4 离线前瞻模式

新增 `setLookaheadLatents()`，输入：

```text
z[t], z[t+1], z[t+2]
```

当 `window=3`、`gamma=0.8` 时计算：

\[
\bar z_t =
\frac{z_t+0.8z_{t+1}+0.64z_{t+2}}
     {1+0.8+0.64}
\]

然后：

\[
z_t^{cmd}=16\frac{\bar z_t}{\|\bar z_t\|_2}
\]

该函数不能使用 `latent_history_`，未来 latent 必须由调用方显式传入。

### 5.5 修改 teleopUpdateLatentFromFlat

当前代码在 backward ONNX 得到 `raw_z` 后，无条件调用过去平滑。

应改成：

```cpp
if (cfg_.latent_mode == "live_current") {
    setCurrentLatent(raw_z);
} else if (cfg_.latent_mode == "live_causal_smooth") {
    smoothCausalLatent(raw_z);
} else if (cfg_.latent_mode == "offline_lookahead") {
    // 离线模式必须走 future window 接口。
    return false;
}
```

切换模式或重置 teleop 时继续清空：

```text
latent_history_
current_z_
has_latent_
```

---

## 6. 修改 ros_interface.cpp

文件：

```text
src/inference/src/ros_interface.cpp
```

在读取 teleop YAML 的位置增加：

```text
teleop_latent_mode
offline_latent_window
offline_latent_gamma
```

启动日志应打印：

```text
latent_mode
teleop_latent_window
teleop_latent_gamma
offline_latent_window
offline_latent_gamma
latent_norm_z
```

这样可以确认实际运行配置不是默认值或旧配置。

---

## 7. 修改离线 NPZ 数据路径

当前文件：

```text
scripts/teleop_backward_obs_publisher.py
```

当前行为只发布当前 reference frame：

```text
reference[t]
→ backward observation[t]
→ 发布
```

因此 C++ 只能算出：

```text
z[t]
```

无法得到：

```text
z[t+1], z[t+2]
```

### 推荐方案

新增：

```text
scripts/precompute_teleop_latents.py
```

执行：

```text
离线 NPZ
→ 逐帧构造 backward observation
→ backward ONNX
→ 保存 z_sequence.npy
```

输出：

```text
shape = [T, 256]
```

离线播放第 `t` 帧时读取：

```text
z_sequence[t:t+3]
```

并传给：

```cpp
setLookaheadLatents(future_zs)
```

序列末尾处理：

```text
t = T-2：使用 2 帧并重新计算权重和
t = T-1：使用 1 帧
```

不要固定除以 `1+0.8+0.64`。

### 避免重复平均

二选一：

```text
方案 A：
    文件保存逐帧 raw z
    C++ 执行 future average

方案 B：
    预计算脚本已经执行 future average
    C++ 直接使用单个 z，window=1
```

不能在预计算和部署端各平均一次。

---

## 8. 修改 inference_node.cpp

文件：

```text
src/inference/src/inference_node.cpp
```

当前：

```text
/teleop_backward_obs
→ 缓存最新单帧
→ teleopUpdateLatentFromFlat()
```

保留它用于实时 PICO。

离线模式应增加单独入口，例如：

```text
/teleop_tracking_latents
```

数据应明确包含：

```text
frame_index
window_count
z[window_count][256]
```

离线回调调用：

```cpp
setLookaheadLatents()
```

不要把未来 latent 窗口塞进现有单帧
`teleop_backward_obs_`，否则实时和离线消息语义会混在一起。

---

## 9. 推荐实施顺序

### 第一步：验证实时延迟

只修改：

```yaml
teleop_latent_window: 1
```

比较 `window=1` 和当前 `window=3` 的遥操效果。

### 第二步：拆分 latent mode

修改：

```text
inference_teleop.yaml
teleop_interface.hpp
teleop_interface.cpp
ros_interface.cpp
```

完成：

```text
live_current
live_causal_smooth
offline_lookahead
```

### 第三步：增加离线前瞻

修改或新增：

```text
scripts/precompute_teleop_latents.py
scripts/teleop_backward_obs_publisher.py
inference_node.cpp
```

实现：

```text
NPZ
→ z_sequence[T,256]
→ z[t:t+3]
→ future weighted average
→ Actor
```

---

## 10. 验证清单

### Actor

- [ ] 输入形状为 `[1,601]`；
- [ ] 输出形状为 `[1,21]`；
- [ ] state 顺序未改变；
- [ ] history 顺序未改变；
- [ ] history 仍然先 query、后 push；
- [ ] 动作缩放未改变。

### Latent

- [ ] backward 输出形状为 `[1,256]`；
- [ ] `||z||₂` 接近 16；
- [ ] 实时默认只使用当前 `z[t]`；
- [ ] causal 模式只使用当前和过去 latent；
- [ ] 离线模式使用 `z[t:t+3]`；
- [ ] 离线模式不读取 `latent_history_`；
- [ ] 序列末尾正确处理短窗口；
- [ ] 没有重复执行窗口平均。

### 数值一致性

建议检查：

```text
| ||z||₂ - 16 | < 1e-4
actor_obs max_abs_error < 1e-5
Python/C++ actor output max_abs_error < 1e-4
```

最终使用同一段离线数据比较 UFO Python 与 RZ C++ 的：

```text
state
last_action
history_actor
z
actor_obs
actor output
```

