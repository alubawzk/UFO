# Mini3 有界 MotionLib 与按需 Expert Sequence 设计草案

状态：**仅供审查，尚未修改训练代码**

日期：2026-07-15

目标平台：8 × 48 GiB RTX 4090，单机多 GPU 训练

## 1. 待确认的核心决策

本方案在不切换 Tracking PPO、不改变 FB/CPR/Aux 训练目标的前提下，将当前全量 expert 物化替换为：

> 全量 metadata/index + 每 rank 有界 MotionLib cache + 按需 ExpertSequenceSampler

建议第一版采用以下参数作为服务器 profiling 起点，而不是未经测量的最终最优值：

| 参数 | 建议起点 | 说明 |
|---|---:|---|
| GPU 数量 | 8 | 一进程一卡，保留现有 distributed gradient averaging |
| Envs / GPU | 512 | 稳定后再测 1024 |
| Motion cache / GPU | 1024 个 motion slot | 稳定后测试 2048、4096 |
| Expert batch / GPU | 1024 | 保持当前 FB 配置 |
| Expert sequence length | 8 | 保持当前 FB 配置 |
| Priority uniform floor | 0.05 | 只用于 cache 内 source 条件 motion 采样，避免高 priority 垄断 |
| Cache refresh 间隔 | 1000 个环境 control step | 每个环境约 20 s 仿真时间；需 profiling 500/1000/2000 |
| Cache 首版替换方式 | 全量替换当前 cache | 逻辑简单；部分替换放到性能优化阶段 |
| Online replay / GPU | 640,000 transitions | 八卡合计 5,120,000，与原始单卡总容量同量级 |
| 常规 validation subset | 4096 条按 source 分层的固定 motion | 快速、可复现；完整评估另走分块流程 |
| 目标显存峰值 | 不高于 42 GiB/GPU | 至少预留约 6 GiB 给瞬时分配和后续调参 |

需要审查确认：

1. 是否接受第一版在 cache refresh 后 reset 全部环境。
2. 是否接受八卡将 `--buffer-size` 从每卡 5,120,000 调整到每卡 640,000。
3. 是否以 1024 motions/GPU、1000 control steps 为首轮 profiling 起点。
4. 是否接受常规训练只评估固定 validation subset，完整数据集采用低频、分块评估。
5. 是否接受 expert sampler 显式保持 manifest source mix；这是对 legacy expert buffer 的有意修复，不是严格分布等价。
6. 是否接受第一版保留 legacy 合法 start 的 off-by-one 行为用于数值回归，后续再单独修复。

## 2. 约束与非目标

### 2.1 必须保持

- Agent 继续使用 `FBcprAuxAgent`。
- 每次 update 继续同时采样 online replay 和 expert sequence。
- `seq_length=8` expert latent 编码保持不变。
- CPR discriminator 继续使用 expert observation 作为正样本。
- `expert_asm_ratio=0.6` 保持不变。
- expert-conditioned rollout、FB、critic 和 auxiliary critic 更新公式保持不变。
- manifest 多数据源的 source-level mix 权重保持不变。
- tracking evaluation 产生的 priority 仍能影响 expert 和 cache 内在线 motion 采样。

这里的 source-level mix 是新正式路径的目标契约。当前 `TrajectoryDictBuffer` 按 motion 均匀初始化，source mass 实际随各 source 的 motion 数量变化；后续 priority 也按全局 motion 归一化。因此，新方案保持 manifest source mix 属于有意修复。当前 Mini3 manifest 只有一个 source，这个差异在 Mini3 上不可见，但多源训练必须记录为行为变更。

### 2.2 本次不做

- 不切换成 SONIC/Tracking PPO。
- 不修改 actor、Forward map、Backward map、critic 或 discriminator 网络结构。
- 不修改 reward、域随机化、reset 或 termination。
- 不把全量数据复制到每张 GPU。
- 不在每次 optimizer update 中读取磁盘 PKL 或执行完整 FK。
- 第一版不实现复杂的 GPU 双 cache；先使用单 GPU cache。

## 3. 当前问题

当前 `Workspace.train_online()` 调用：

```python
load_expert_trajectories_from_motion_lib(...)
```

loader 内部无上限调用：

```python
env._motion_lib.load_motions_for_training()
```

随后遍历全部 `_num_unique_motions`，将所有 expert observation 拼接为 CUDA `TrajectoryDictBuffer`。

Mini3 当前索引包含以下聚合统计：

- 162,044 条 motion
- `sum(frame_count / fps) = 991,006.108 s`
- 据此粗略估算，按 50 Hz 约 49,550,305 个 expert sample
- expert 每步最低 427 个 `float32`：`state=48`、`last_action=21`、运行时 `privileged_state=358`

仅这三项的最低显存约为：

```text
49,550,305 × 427 × 4 bytes ≈ 78.82 GiB
```

该 78.82 GiB 只是用于说明 OOM 量级的粗估。MotionLib 的单条精确时长是 `(frame_count - 1) / fps`，而不是 index 当前累计的 `frame_count / fps`；精确 expert sample 总数还需要逐 motion 计算 `ceil(motion_length / env.dt)`。现有 index 没有逐 motion metadata，无法直接给出精确总数。

当前多卡逻辑会在每个 rank 各自调用同一 loader。8 张 48 GiB GPU 不会自动合并显存，因此按上述粗估，现状是每张 GPU 都尝试构建约 78.82 GiB 的 expert 表示，而不是自动除以 8。

该估算还未包含：

- MotionLib body position/rotation/velocity 张量
- online replay
- 模型、梯度和 optimizer state
- MJLab/Warp 环境
- 拼接和 cache refresh 瞬时张量

## 4. 目标架构

```text
                     CPU / host
┌────────────────────────────────────────────────────────────┐
│ GlobalMotionIndex                                          │
│ relative_path、source_id、fps、frame_count、motion_length、ID │
│                                                            │
│ GlobalPriorityTable                                        │
│ global_id → raw tracking priority；采样时按 source 归一化    │
│                                                            │
│ Rank-aware Cache Planner                                   │
│ 按 rank × source 直接生成有界、尽量无重复的 ID 列表          │
└───────────────────────────┬────────────────────────────────┘
                            │ selected global motion IDs
                            ▼
                       GPU / each rank
┌────────────────────────────────────────────────────────────┐
│ BoundedMotionLibCache                                      │
│ 只保存当前 rank 的 1024（可调）个 motion 的派生运动学张量    │
│                                                            │
│ global_motion_id ↔ cache_local_id                          │
│                                                            │
│ ┌──────────────────────┐  ┌──────────────────────────────┐ │
│ │ Online MJLab envs    │  │ ExpertSequenceSampler        │ │
│ │ reset/reference 查询 │  │ 查询连续 L+1 个 50 Hz 时间点 │ │
│ └──────────────────────┘  └──────────────┬───────────────┘ │
└───────────────────────────────────────────┼────────────────┘
                                            ▼
                              FB / CPR / Aux agent update
```

设计重点：

- 全量数据只保留轻量 metadata/index 和 priority。
- 当前 cache 中的 motion 派生运动学张量常驻 GPU。
- expert sampler 每次只生成本次 update 需要的 batch。
- expert sequence 按 `env.dt` 的 control-time grid 生成，通过 `get_motion_state()` 对源数据帧插值；不得直接把源数据连续帧当作 50 Hz sample。
- expert sampler 不拥有覆盖整个 cache 的第二份永久 observation storage。
- Agent 继续通过 `replay_buffer["expert_slicer"].sample(...)` 获取数据。

## 5. 全量索引与全局 ID

现有 per-motion directory index：

```text
humanoidverse/data/mini3_pkl_ufo/_ufo_per_motion_index.json
```

当前只包含 motion 数量、总帧数、`sum(frame_count / fps)` 等聚合统计，没有逐 motion 的 `relative_path/fps/frame_count/motion_length`。因此不能直接用它构建本方案要求的 `GlobalMotionIndex`。

正式训练必须复用升级后的逐 motion index 和 manifest 解析结果，不在启动时重新扫描、反序列化全部 PKL。`convert_mini3_pkl.py` 在生成每个输出 clip 时已经知道 path、fps 和 frame count，必须同步写出这些 metadata。对于已经生成的 162,044 条 Mini3 motion，提供一次性 index-only 迁移/重建命令；该命令允许离线扫描全部输出 PKL，但训练启动路径不允许这样做。

每条 motion 至少需要：

```text
global_motion_id
source_id
motion_key
relative_path
fps
frame_count
motion_length
```

绝对 resolved path 由运行时的 source root 与 `relative_path` 组合得到，不作为 global ID 或 index hash 的一部分。

converter 生成的单 source index 保存稳定的 source-local 记录顺序和 metadata；它不独立决定跨 manifest 的 `global_motion_id`。manifest loader 按第 5.1 节组合各 source index 后派生连续 global ID，并生成覆盖 manifest 与所有 source index 的 composite hash。

字段语义固定为：

```text
motion_length = (frame_count - 1) / fps
```

如果仍需保留 `frame_count / fps` 口径的转换统计，必须使用不同字段名，例如 `sample_coverage_seconds`，不得把它命名为 MotionLib `duration`。50 Hz expert sample 数是给定 `control_dt` 后的派生值，不作为与控制频率无关的原始 metadata：

```text
expert_sample_count(control_dt) = ceil(motion_length / control_dt)
```

### 5.1 稳定 global ID 顺序

`global_motion_id` 的顺序契约为：

1. 按 manifest 中 dataset/source 的声明顺序。
2. per-motion directory 内按规范化 POSIX relative path 排序，不按绝对路径排序。
3. 聚合 PKL 内按 `motion_key` 排序。
4. 按上述顺序从 0 连续编号。

当前 manifest 传入的是 source path 列表，MotionLib 的 mixed-source directory 分支已经使用排序后的 glob；单字符串 directory 分支仍使用未排序 glob。无论当前调用恰好走哪个分支，都不能把实现细节当作持久 ID 契约。新 index 必须显式保存排序后的记录，并以 index 顺序作为唯一 global ID 来源。

### 5.2 Index schema、hash 与迁移

要求：

- `global_motion_id` 在训练、cache refresh、evaluation 和 resume 后稳定不变。
- 多数据源时，cache 组成和 expert 采样都必须保持 manifest 的 source-level probability mass。
- 全局 metadata 默认放 CPU；只有当前 cache 映射和采样所需的小张量放 GPU。
- 索引缺少 `fps/frame_count` 时在训练启动前报错，不允许训练时逐文件补扫全部数据。
- index 带明确 schema version；旧的聚合-only v1 index 不得被 on-demand 模式静默接受。
- index hash 覆盖 schema version、manifest source 顺序、source 名称/权重、relative path、motion key、fps 和 frame count；绝对数据根目录不进入 hash。
- 迁移命令原子写入新 index，并校验记录数、source 内 relative path 唯一性、`(source_id, motion_key)` 唯一性、fps 有限且大于 0、frame count 大于 0；无法满足任一 sequence 请求的一帧 motion 保留在 index 中，但不会进入 eligible cache 配额。
- converter 新生成数据与一次性迁移生成的数据必须得到相同的排序、global ID 和 index hash。

计划的一次性迁移入口为：

```bash
uv run python -m humanoidverse.tools.convert_mini3_pkl \
  --output-dir humanoidverse/data/mini3_pkl_ufo \
  --manifest configs/data/mini3_pkl.yaml \
  --rebuild-index-only
```

`--rebuild-index-only` 不重新转换或覆盖 motion PKL；它离线逐个反序列化现有输出文件以提取必要 metadata，写临时 index，完成全量校验后原子替换正式 index。迁移中断时保留旧 index，on-demand 训练继续拒绝旧 schema。

## 6. Bounded MotionLib cache

### 6.1 新增确定性 ID 加载接口

当前 `load_motions_for_training(max_num_seqs=...)` 负责内部随机采样。为支持多卡无重复分片和精确恢复，新增等价于下面的接口：

```python
motion_lib.load_motion_ids_for_training(global_motion_ids: Tensor) -> None
```

该接口必须：

1. 接收明确的 dataset-global ID 列表。
2. 只读取这些 motion 文件。
3. 执行现有 retarget/FK/interpolation 和派生速度逻辑。
4. 构建 cache-local 连续张量。
5. 保存 `_curr_motion_ids`，表示 `cache_local_id → global_motion_id`。
6. 保存每条已加载 motion 的源 `frame_count/fps/motion_length`；不同调用按 `control_dt` 推导 expert sample 数，并针对请求的 `seq_length + 1` 动态判断是否可采样。
7. 不修改全局 index 和全局 priority。

现有随机加载接口继续保留给兼容路径和普通环境使用，但 Mini3 正式训练不得调用无参数的全量加载。

### 6.2 Cache 大小

第一轮固定：

```text
cache_size_per_rank = min(configured_cache_size, dataset_size_for_rank)
configured_cache_size = 1024
```

完成显存 profiling 后按以下顺序测试：

```text
1024 → 2048 → 4096 motions/GPU
```

不直接假设每 rank 能装下全数据集的八分之一。即使 expert observation 不再物化，MotionLib 仍保存全身运动学张量，并与 replay、模型、仿真和临时内存竞争显存。

## 7. ExpertSequenceSampler 契约

### 7.1 对 Agent 保持兼容

新 sampler 对外保留当前最关键的接口：

```python
sample(batch_size: int, seq_length: int | None = None) -> dict
```

返回结构保持：

```python
{
    "observation": {
        "state": ...,              # [B, 48]
        "last_action": ...,        # [B, 21]
        "privileged_state": ...,   # [B, 358]
    },
    "next": {
        "observation": {
            "state": ...,
            "last_action": ...,
            "privileged_state": ...,
        }
    },
}
```

因此下面这些 agent 文件原则上不改训练公式：

- `humanoidverse/agents/fb/agent.py`
- `humanoidverse/agents/fb_cpr/agent.py`
- `humanoidverse/agents/fb_cpr_aux/agent.py`

### 7.2 Sequence 采样语义

普通 agent update 使用 `batch_size=1024, seq_length=8`：

```text
num_sequences = 1024 / 8 = 128
```

每条 sequence 先选择一个 cache motion，再选择一个 50 Hz control-grid start index，并查询连续 `8 + 1` 个时间点：

```text
t0 ... t7     → observation
t1 ... t8     → next.observation
```

这里的 `t0 ... t8` 不是源 PKL 的连续帧。对于源 `frame_count=N`、`fps=f`、控制间隔 `control_dt=env.dt` 的 motion，必须使用与当前 `expert_motion_loader.py` 相同的时间语义：

```text
motion_length = (N - 1) / f
expert_sample_count K = ceil(motion_length / control_dt)
eligible = K >= L + 1

times = (start + arange(L + 1)) * control_dt
motion_state = motion_lib.get_motion_state(cache_local_motion_ids, times)
```

`expert_sample_count` 必须由 legacy loader、全局 metadata eligibility、cache planner 和新 sampler 共用同一个 helper 计算。第一版 helper 保留当前 MotionLib `float32 motion_length / env.dt` 后取 `ceil` 的数值语义，避免 CPU float64 metadata 与 GPU float32 在整除边界产生一个 sample 的差异；精确整除、略小于和略大于整除点都要做回归测试。

不能直接复用当前 `MotionLibBase.get_motion_num_steps()`，因为它按 `frame_count * sim_fps / fps` 计算，与 loader 基于 `(frame_count - 1) / fps` 的语义不同；实现时应让它调用同一个 helper 或新增语义明确的方法。

`get_motion_state()` 负责在 MotionLib 保存的源数据帧之间插值。禁止用 `source_start + arange(L + 1)` 直接索引源帧，否则约 120 Hz 的 Mini3 数据会被错误地当作 50 Hz 数据，改变 sequence 时间尺度。

按预期的合法窗口定义，start 范围应为：

```text
0 <= start <= K - L - 1
```

但是当前 `TrajectoryDictBuffer.get_idxs()` 在 `K > L + 1` 时实际只生成 `0 ... K-L-2`，排除了最后一个合法 start；`K == L + 1` 时固定为 0。为满足第一版 legacy 分布回归，首版 on-demand sampler 明确保留这个 off-by-one 兼容行为，并增加回归测试。修复为完整合法范围必须作为单独行为变更提交，不能在本次内静默改变。

必须保证：

- batch size 能被 sequence length 整除。
- sequence 不跨 motion 边界。
- eligibility 按 50 Hz `expert_sample_count` 判断，而不是按源 `frame_count` 判断。
- flatten 顺序与当前 `TrajectoryDictBuffer.sample()` 一致，使 `encode_expert()` 的 reshape 语义不变。
- 第一版复刻 legacy sampler 的 control-time grid 和 start-index 分布；priority/source mix 的有意变化见第 8 节。

此外，当前 expert-conditioned rollout 会调用：

```python
sample(batch_dim * 250, seq_length=250)
```

因此 sampler 不能只支持固定的 8 个 control sample。它必须针对每次请求动态建立 eligible-motion mask：

- 普通 update 要求 motion 至少有 9 个 50 Hz expert sample。
- expert-conditioned rollout 要求 motion 至少有 251 个 50 Hz expert sample。
- 短 motion 只从当前长 sequence 请求中排除，不从普通 8-sample 训练中永久删除。
- cache planner 应按上述时间公式保证当前 cache 中存在足够多的 251-sample eligible motion，避免 expert rollout 长期集中到极少数 clip。
- 128K 量级的临时 expert rollout batch 必须分块构造 observation/Backward-map，或通过 profiling 证明一次处理的显存峰值安全。

### 7.3 Expert observation 构造

现场生成逻辑复用当前 `expert_motion_loader.py` 的数学定义：

```text
state = [
    reference_dof_pos - default_dof_pos,
    reference_dof_vel,
    projected_gravity,
    reference_base_ang_vel,
]

last_action = zeros_like(reference_dof_pos)

privileged_state = compute_humanoid_observations_max(...)
```

要求：

- 使用 `reference_base_ang_vel()`，保持 body-frame 和 `obs_scales.base_ang_vel` 语义。
- quaternion、body 顺序、DOF 顺序与在线 observation 一致。
- 全部时间点构造、`get_motion_state()` 批量查询/插值和 observation 变换在 GPU 上完成。
- 禁止 Python 逐 motion、逐 sequence 或逐帧循环。
- 新 sampler 输出维度以运行时 tensor 为准，并断言 Mini3 `privileged_state=358`。

### 7.4 Priority 接口

现有 agent 只调用 `sample()`；priority 更新主要由 workspace 使用。新 sampler 提供：

```python
set_global_priorities(global_ids, priorities)
loaded_global_motion_ids
file_names
```

不再要求 sampler 为全量 162,044 条 motion 建立 `TrajectoryDictBuffer.priorities`。全局 priority 由独立 registry 保存，sampler 只获取当前 cache 对应的归一化权重。

第一版只在 cache 内 motion 采样阶段使用 priority，cache inclusion 本身不再使用同一 priority。对当前请求先应用 eligible mask，再按 source 分层：

```text
P(source=s) = manifest_source_weight[s]

within source s:
priority_prob[i] = priority[i] / sum(priority[eligible_in_s])
P(i | s, current_cache, eligible) =
    (1 - uniform_floor) * priority_prob[i]
    + uniform_floor / eligible_count_in_s
```

priority 必须有限且非负；某个 source 的 eligible priority 总和为 0 时，该 source 内回退为均匀分布。validation subset 的部分更新只覆盖对应 global IDs，未评估 motion 保留原值。

对于配置声明必须支持的 sequence length（首版为 8 和 250），如果某个 source 在全数据集中存在对应 eligible motion，cache planner 必须保证每 rank cache 至少包含一条；对于其他临时请求，如果当前 cache 某个 source 没有 eligible motion，只能在其余 eligible source 间重新归一化 source mass，并记录指标/警告。如果某个 source 在全数据集中本来就没有对应 eligible motion，同样记录为数据约束，而不能伪称保持了原 source mass。

## 8. 多 GPU cache 分配

八卡下不能让 8 个 rank 独立地从相同分布随意抽取，否则会增加重复和共享磁盘随机读取。也不能只生成一个满足全局 source 比例的 ID 列表后任意切给各 rank：每个 rank 会在自己的 cache 内重新归一化并贡献相同数量的 expert batch，任意切分会改变全局 source mass 和 priority mass。

同一个 priority 如果先用于 cache inclusion，进入 cache 后又用于 motion 采样，则边际概率近似为 `inclusion_probability(i) × priority(i)`；当 inclusion probability 也近似正比于 priority 时会形成近似二次加权。第一版禁止这种双重使用。

第一版建议：

1. rank 0 根据 manifest source mix，为每个 `rank × source` 计算 cache slot 配额。每个正权重且存在 eligible motion 的 source 在 cache 大小允许时每 rank 至少分配一个 slot；余数使用确定性的 largest-remainder/轮转规则分配。
2. 在每个 source 内对 cache inclusion 做 priority-blind 均匀无放回采样，并在数据量允许时跨 rank 去重，总计选出：

   ```text
   world_size × cache_size = 8 × 1024 = 8192 slots
   ```

3. 直接生成分层后的 per-rank ID 列表并 broadcast/scatter，不再对全局列表任意切片。
4. 每个 rank 只加载自己的 ID。
5. expert 和在线 motion 在 cache 内先按 manifest source mass 选择 source。expert sampler 仅在该 source 的当前 eligible motion 内按第 7.4 节的 `priority + uniform floor` 采样；在线环境 resample 不需要 sequence eligibility mask，在该 source 的全部已加载 motion 内使用同一 priority 语义。

采样必须满足：

- 每 rank 以及等 batch 聚合后的全局 source-level mix 都不被 priority 改写。
- priority 只改变 source 内 motion 分布。
- 至少保留一个可配置的 uniform floor，避免困难样本垄断 cache。
- cache inclusion 第一版不使用 priority，避免与 cache 内 priority 采样形成二次加权。
- 小数据源不足以无放回填满 per-rank 配额时允许有放回，并记录 source/rank 级重复率。
- checkpoint 保存 cache generation 和每 rank 的 global ID 列表。

由于任一时刻只覆盖全数据的一小部分，新路径不可能在单个 cache generation 内严格复现全量 legacy expert 分布；验收目标是跨多个 refresh generation 的边际分布在统计容差内接近“source mass 固定、source 内按 priority”的目标。若后续确实需要 priority-weighted cache inclusion，必须同时设计 inclusion-probability 校正并独立验证，不能直接叠加同一 priority。

## 9. Cache refresh

### 9.1 第一版刷新时机

新增按“每个环境经历的 control step”计数的配置：

```text
motion_cache_refresh_control_steps = 1000
```

该计数不随 GPU 数和 `num_envs_per_rank` 改变。第一版在一个完整 vector-env step 和全部 agent update 完成后执行：

```text
完成当前 env step
→ 完成本轮 16 次 agent update
→ 保存必要统计
→ 同步所有 rank
→ 加载新 cache
→ 重建 global/local ID 映射
→ reset 全部环境
→ 清理并强制重建 expert-conditioned rollout 的 `tracking_z`
→ 继续 rollout
```

禁止在 episode 或 agent update 中途覆盖 MotionLib 张量。

### 9.2 第一版全量替换

首版每次替换当前 rank 的整个 cache，原因是 MotionLib 使用连续拼接张量，部分插入/删除会显著增加索引和内存管理复杂度。

后续若 profiling 显示分布跳变或 I/O 过高，再增加：

```text
retain_ratio = 0.25
replace_ratio = 0.75
```

### 9.3 Online replay 非平稳性

cache refresh 后，online replay 仍包含旧 cache 产生的 transition，而 expert sampler 使用新 cache。这不会破坏张量接口，但会增加 expert 分布的非平稳性。

缓解措施：

- cache 初始不少于 1024 motions/GPU。
- 八卡 cache 尽量无重复，增加每一代全局覆盖。
- 保留 uniform floor。
- refresh 不要过于频繁。
- 记录 cache generation，并按 generation 监控 discriminator loss。
- 若仍不稳定，再考虑小型、有界的跨 cache expert reservoir；不在第一版实现。

## 10. Tracking evaluation 与 priority

当前 evaluation 和 priority 假设 expert buffer 覆盖全部 motion。新方案拆成两类评估。

### 10.1 共享 MotionLib 的统一恢复事务

当前 MJLab evaluation 与训练共享同一个环境和 MotionLib。任何会加载 evaluation motion IDs 的评估——包括常规 4096 subset 和完整评估——都会覆盖 rank 0 的训练 cache，必须使用同一恢复事务，不能只在完整评估中恢复：

```text
训练 rollout 边界，将评估前 transition 标记为 truncated
→ 所有 rank 到达评估控制点；非 rank 0 在约定的 collective 等待
→ rank 0 保存训练 cache global IDs 和 cache generation
→ rank 0 按 eval_chunk_size 分块加载、评估、释放
→ rank 0 用明确 global IDs 恢复原训练 cache 和 global/local 映射
→ rank 0 在 finally 中 broadcast 评估状态和已评估 priority payload
→ 任一步失败则所有 rank 一致停止；成功才继续
→ 所有 rank reset 各自训练环境
→ 清除并强制重建 rollout context、tracking_z 和 history context
→ 所有 rank 完成最终 barrier
→ 继续训练
```

恢复不得调用会随机重采样或可能因 `all_motions_loaded` 提前返回的无参数 `load_motions_for_training()`；必须调用确定性 ID 加载接口，并断言恢复后的 global ID 列表、顺序、cache generation 和 global/local 映射与评估前一致。异常和提前退出也必须通过 `try/finally` 恢复训练 cache，并保证 rank 0 始终向其他 rank 发布成功/失败状态，避免其他 rank 永久阻塞；恢复失败时所有 rank 一致停止训练，不允许带着 evaluation cache 继续 rollout。

reset 后 `step_count=0` 当前通常会间接触发 context 重采样，但实现不能依赖这个隐式行为。workspace/agent 应提供明确的 rollout-context invalidation 路径。

如果 evaluation 使用完全独立的环境和 MotionLib，则不覆盖训练 cache，可以跳过上述 cache 保存/恢复；仍需保持模型 mode、priority broadcast 和 rank 同步正确。

### 10.2 常规训练评估

- 从 manifest 按 source 分层、固定生成 4096 条 validation motion ID。
- validation ID 和 seed 写入 run config/checkpoint。
- 每次 eval 对该 subset 分块处理。
- 只更新已评估 motion 的 priority，未评估项保留旧值或初始值。
- 即使 validation subset 只有 4096 条，也必须执行第 10.1 节的共享 cache 恢复事务。

### 10.3 完整评估

- 仅在里程碑 checkpoint 或独立离线任务执行。
- 每次按 `eval_chunk_size` 加载一块，评估后释放，再加载下一块。
- rank 0 执行时其他训练 rank 等待；更优的长期方案是独立评估作业。
- 使用第 10.1 节的同一恢复事务，不定义另一套恢复路径。

### 10.4 Priority 公式

保持当前公式：

```text
emd_clipped = clip(emd, 0.5, 2.0)
scaled = emd_clipped × 2.0
priority = 2 ** scaled
```

多源数据按第 7.4 和第 8 节在每个 source 内归一化，不改变 manifest source mass。需要明确记录：这与 legacy expert buffer 的全局 motion 归一化不同，是新正式路径的有意修复。

## 11. Checkpoint 与恢复

checkpoint 新增保存：

```text
global priority table
每 rank 当前 cache global IDs
cache generation
cache refresh control-step counter
sampler RNG state
cache planner RNG state
validation subset IDs
index schema version 和 manifest/index hash
```

不保存 MotionLib 派生 GPU 张量。Resume 时根据 global IDs 从磁盘重建当前 cache，然后恢复 RNG 和计数器。

恢复要求：

- 相同 world size 时精确恢复每 rank cache。
- world size 改变时重新规划 cache，但保留全局 priority；日志明确标记非 bitwise-resume。
- manifest/index hash 不一致时拒绝恢复，防止 global ID 指向错误文件。

## 12. CLI / 配置草案

计划新增：

```text
--expert-data-mode {legacy,on-demand}
--motion-cache-size 1024
--motion-cache-refresh-control-steps 1000
--expert-priority-uniform-floor 0.05
--validation-motion-limit 4096
--eval-motion-chunk-size 1024
--cache-rank-disjoint
```

建议语义：

- `on-demand`：正式 Mini3 路径，禁止全量 expert materialization。
- `legacy`：仅用于小数据集数值回归，不允许在 Mini3 manifest 上无显式确认运行。
- `motion-cache-size=0` 非法；不使用 0 表示无限，避免误触全量加载。
- `expert-priority-uniform-floor` 必须在 `[0, 1]`；只作用于 cache 内 source 条件采样，不作用于 cache inclusion。
- multi-GPU 默认启用 rank-disjoint cache planning。

八卡首轮命令草案：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent fb \
  --robot-config configs/robots/mini3.yaml \
  --data-manifest configs/data/mini3_pkl.yaml \
  --gpu-ids all \
  --num-envs 512 \
  --buffer-size 640000 \
  --expert-data-mode on-demand \
  --motion-cache-size 1024 \
  --motion-cache-refresh-control-steps 1000 \
  --expert-priority-uniform-floor 0.05 \
  --validation-motion-limit 4096 \
  --eval-motion-chunk-size 1024 \
  --cache-rank-disjoint \
  --work-dir runs/ufo_fb_mini3_8gpu
```

完成 512 env/GPU 稳定性测试后，再单独测试：

```text
num_envs:       512 → 1024
motion cache:  1024 → 2048 → 4096
refresh steps: 1000 → 500 / 2000
```

每次只改变一个变量。

## 13. 文件级改动清单

| 文件 | 计划改动 |
|---|---|
| `humanoidverse/agents/envs/expert_sequence_sampler.py` | 新增按需 sampler；实现 sequence ID/control-time 采样、GPU 批量 `get_motion_state()` 查询/插值、expert observation 构造和接口兼容。 |
| `humanoidverse/agents/envs/expert_motion_loader.py` | 提取可复用的批量 observation 构造函数；保留 legacy loader 仅用于小数据回归，并增加全量加载保护。 |
| `humanoidverse/utils/motion_lib/motion_lib_base.py` | 新增按明确 global ID 加载 cache 的接口；暴露 global/local ID 映射；统一 `(N-1)/fps` expert sample count helper；禁止正式路径的无上限加载。 |
| `humanoidverse/tools/convert_mini3_pkl.py` | converter 输出逐 motion metadata/index；区分 MotionLib `motion_length=(N-1)/fps` 与转换 coverage 统计；提供/接入 index-only 迁移流程。 |
| `humanoidverse/utils/motion_data/manifest.py` | 只从升级后的 index 暴露稳定 global ID、source、fps、frame count、motion length 和 index hash；on-demand 启动不重复扫描 motion 文件。 |
| `humanoidverse/training/workspace.py` | 构建新 sampler；替代全量 expert buffer；加入安全 refresh、所有共享评估的事务式 cache 恢复、rollout-context invalidation、rank 同步、checkpoint state 和 priority registry。 |
| `humanoidverse/agents/evaluations/humanoidverse_mjlab.py` | evaluation 改为固定 subset/分块加载；结果携带 global motion ID；通过明确 ID 恢复共享训练 cache。 |
| `humanoidverse/train.py` | 增加上述 CLI 和 TrainConfig 字段；日志输出每 rank cache/replay/eval 参数。 |
| `humanoidverse/config/obs/bfm_zero_obs.yaml` | 修正 `max_local_self` 声明少 1 维的问题，并增加运行时维度断言。 |
| `tests/test_convert_mini3_pkl.py` | 增加逐 motion metadata、motion-length/coverage 语义、稳定排序、index-only 迁移与 converter/migration hash 等价测试。 |
| `tests/test_expert_sequence_sampler.py` | 新增不同源 fps 下的 50 Hz 插值时间网格、eligibility、legacy start off-by-one、形状、连续性、数值等价和 priority 测试。 |
| `tests/test_motion_cache_planner.py` | 新增 rank × source 分层、跨 rank 去重、priority 不参与 inclusion、跨 refresh 边际分布、确定性和 checkpoint 恢复测试。 |
| `tests/test_motion_manifest.py` 或现有相关测试 | 增加 index schema/hash、稳定 global ID、缺失逐 motion metadata 的失败测试，并断言训练启动不扫描 PKL。 |
| `tests/test_tracking_evaluation_cache_restore.py` | 新增常规/完整评估成功和异常路径的 cache ID/映射恢复、环境 reset、context invalidation 与 rank 同步测试。 |

不计划修改 FB/CPR/Aux 的 loss 公式。若 agent 文件发生修改，只允许是类型/接口兼容，不改变数值路径。

## 14. 实现阶段

### 阶段 A：逐 motion index 与数值等价 sampler

1. 升级 converter index schema，并实现 Mini3 现有输出目录的一次性 index-only 迁移。
2. 验证 converter 直出和迁移路径得到相同 global ID 顺序与 index hash。
3. 在原始 862 条 LaFAN 小数据上同时构建 legacy buffer 和新 sampler。
4. 固定 motion ID、50 Hz start index，比较两者经 `get_motion_state()` 插值后的输出。
5. 验证 `state`、`last_action`、`privileged_state`、current/next 最大误差。
6. 验证 `encode_expert()` 得到的 `z_expert` 等价。
7. 验证 legacy start off-by-one 兼容分布，暂不修复。
8. 暂不启用 cache refresh。

### 阶段 B：有界 cache 与八卡分配

1. 增加 global ID 加载。
2. 增加 rank × source 分层、priority-blind inclusion 的 cache planner。
3. 接入 workspace 和安全全环境 reset。
4. 确认每 rank 不再调用全量 loader。
5. 接入 checkpoint/resume。

### 阶段 C：分块 evaluation

1. 固定 validation subset。
2. 分块 evaluation。
3. 全局 priority registry。
4. 常规和完整评估统一执行事务式训练 cache 恢复。
5. reset 环境并显式清除 context/tracking_z/history context。
6. 覆盖评估异常时的 `try/finally` 恢复路径。

### 阶段 D：性能优化

只有 profiling 证明必要时再实现：

- pinned CPU prefetch cache
- rank 间错峰磁盘读取
- PKL 数据 shard 打包
- 部分 cache replacement
- 小型跨 cache expert reservoir
- 当前 shard 的可选 expert observation cache

## 15. 测试与验收标准

### 15.1 单元测试

- `batch_size % seq_length != 0` 时失败。
- 对 60/120/非整数 fps 的源 motion，生成与 legacy loader 相同的 50 Hz 时间点并通过 `get_motion_state()` 插值。
- eligibility 使用 `ceil(((frame_count - 1) / fps) / control_dt)`，不使用源 `frame_count` 直接判断。
- 不采样 50 Hz expert sample 数小于 `seq_length + 1` 的 motion。
- sequence 不跨 motion 边界。
- current/next 恰好相差一个 `control_dt`。
- flatten 后每组 8 个 control sample 保持连续。
- 精确整除和非整除 `motion_length / control_dt` 的边界与当前 loader 一致。
- 第一版复现 legacy 最后一个合法 start 被排除的 off-by-one 行为；未来修复时该测试随独立变更更新。
- `seq_length=250` 时只使用至少 251 个 50 Hz expert sample 的 motion，并保持连续。
- 大 expert-conditioned rollout batch 可以分块处理且结果与非分块版本等价。
- Mini3 expert shape 为 `state=48`、`last_action=21`、`privileged_state=358`。
- 固定输入时与 legacy loader 数值等价。
- priority 高的 motion 在 cache 内统计采样频率更高，但 cache inclusion 频率不随 priority 改变。
- 每 rank source mix 和多 rank 等 batch 聚合后的 source mix 都在统计容差内保持 manifest 权重。
- 跨多个 refresh generation 的 motion 边际分布不存在近似 priority 二次加权。
- 多 rank cache 在数据量足够时没有 global ID 重复。
- checkpoint/resume 恢复 cache IDs、priority 和 RNG。
- converter 直出与 index-only 迁移得到相同 global ID 顺序和 index hash。
- on-demand 启动缺少逐 motion metadata 时立即失败，metadata 完整时不扫描/反序列化 motion PKL。
- 常规 validation 和完整评估完成后恢复完全相同的训练 cache IDs/顺序/映射并清除 rollout context；评估抛异常时同样恢复或停止训练。

### 15.2 集成测试

1. 单 GPU、小 cache、16 env smoke：至少越过环境创建、expert sample 和一次 optimizer update。
2. 单 GPU、Mini3 完整 index：确认不会遍历/加载全部 motion 内容。
3. 两 GPU：确认 cache 分片、梯度同步和 checkpoint。
4. 两 GPU、共享环境常规 validation：确认 rank 0 分块覆盖后恢复原 cache，所有 rank reset/context invalidation 后继续训练。
5. 八 GPU、512 env/GPU：运行至少 2 次 cache refresh。
6. 八 GPU、1024 env/GPU：在 512 方案稳定后测试。

当前 `--smoke` 只有 2048 env steps且小于 seed steps，不能验证 optimizer update。需要新增可控测试参数或专用 integration 命令，确保至少实际执行一次 `agent.update()`。

### 15.3 显存和性能验收

每 rank 记录：

```text
torch.cuda.max_memory_allocated
torch.cuda.max_memory_reserved
MotionLib cache 加载时间
ExpertSequenceSampler.sample 时间
expert observation 构造时间
env step FPS
agent update 时间
cache refresh 总停顿
PKL read throughput
cache global-ID 重复率
per-rank/source cache slot 数与 expert sample mass
跨 refresh generation 的 priority 边际采样频率
```

首轮目标：

- 任一 rank 峰值显存不超过 42 GiB。
- 不出现完整 162,044 motion 的 CUDA tensor。
- 稳态 sampler 开销目标低于 agent update 时间的 5%。
- 含 cache refresh 的摊销吞吐下降目标低于 10%。
- 八卡 cache global ID 在可无放回采样时无重复。
- 训练 loss 在 cache refresh 前后无 non-finite 或数量级突变。

性能比例是验收目标，不是实现前保证；若不达标，先 profiling，再启用阶段 D 优化。

### 15.4 远程服务器预检

正式 smoke 前逐 rank 验证：

- `nvidia-smi` 确认 8 张卡均报告预期的约 48 GiB 可用显存。
- PyTorch 能在 Ada `sm_89` 上实际创建并执行 CUDA tensor/kernel，不能只检查 `torch.cuda.is_available()`。
- NCCL 能完成 8 rank all-reduce，并记录 PCIe 拓扑和带宽。
- motion 数据位于服务器本地 NVMe；若只能使用共享存储，先测 8 rank 并发随机读取吞吐。
- 每张 GPU 在训练启动前没有其他进程占用显存。

本地 RTX 5090 D 的 `sm_120` 报错与远程 4090 不是同一个架构问题，但远程环境仍必须独立验证，不直接复制当前本地虚拟环境。

## 16. 风险与回退

| 风险 | 缓解 |
|---|---|
| Cache refresh 同步 I/O 停顿 | 第一版测量；后续 pinned CPU 异步预取、PKL shard。 |
| 新旧 cache 导致 discriminator 分布跳变 | 增大 cache、降低刷新频率、uniform floor；必要时部分替换。 |
| 多 rank 重复数据 | rank 0 统一规划并按 rank 分配 global IDs。 |
| 共享存储被 8 rank 随机读取打满 | 使用服务器本地 NVMe、预取和错峰读取。 |
| Priority 只覆盖 validation subset | 未评估 motion 保留初始/历史值；低频完整分块评估。 |
| Priority 同时用于 cache inclusion 和 cache 内采样导致二次加权 | 第一版 inclusion 在 source 内均匀，仅在 cache 内使用 priority；跨 refresh 做边际分布统计。 |
| 全局 source 配额任意切 rank 后被各 rank 重新归一化 | planner 直接生成 rank × source 分层配额，并记录每 rank 的 source sample mass。 |
| 常规/完整评估覆盖共享训练 cache | 两类评估统一执行确定性 ID 恢复、reset、context invalidation 和 rank barrier；异常路径用 `try/finally`。 |
| Resume 后 global ID 漂移 | 保存并校验 manifest/index hash。 |
| 旧 Mini3 index 只有聚合统计 | 正式训练前运行一次 index-only 迁移；on-demand 模式拒绝聚合-only index。 |
| On-demand observation 计算过慢 | 先向量化/compile；必要时增加有界 shard-local observation cache。 |
| 新 sampler 与 legacy 数值不一致 | 小数据双路径逐 batch 对照，不通过则不进入正式训练。 |

保留 `legacy` 模式作为小数据回归和紧急回退，但 Mini3 正式命令必须显式使用 `on-demand`，且 legacy 路径检测到数据规模超过安全阈值时应直接拒绝启动。

## 17. 审查通过后的实施边界

审查通过后，第一轮实现只包含：

1. 升级后的逐 motion index、converter 输出和现有 Mini3 index-only 迁移。
2. 按 50 Hz control-time grid 插值、数值等价的按需 ExpertSequenceSampler。
3. 每 rank 有界 MotionLib cache 和明确 global ID 加载。
4. 同步式、全量 cache refresh 及安全 reset/context invalidation。
5. rank × source 分层且 inclusion 不使用 priority 的 cache ID 分配。
6. 固定 validation subset、分块 evaluation 和所有共享评估的事务式训练 cache 恢复。
7. checkpoint/resume 必要状态。
8. 单元测试、单卡与八卡 smoke/profile 日志。

异步预取、部分替换、PKL shard 和 expert reservoir 不与第一版混在一起；只有第一版 profiling 证明需要时再分别实现。
