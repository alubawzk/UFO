# Mini3 接入 UFO 的适配清单

本文针对当前仓库中的 `humanoidverse/data/robots/mini3_mjlab/mini3.xml`，说明为了让 Mini3 进入 UFO 的 RobotState 数据导入、MJLab 训练、tracking inference 和后续部署流程，还需要修改或新增哪些内容。

本文只记录适配工作，不代表自动生成的参数可以直接用于大规模训练或实机。PD、力矩限制、默认姿态、碰撞和奖励参数必须经过仿真与硬件规格复核。

> 实施状态（2026-07-14）：XML 路径/default pose、RobotSpec、Hydra 配置、MotionLib actuator/freejoint 兼容、CSV 转换器和结构回归测试已完成；真实 CSV 的抽样转换及 manifest/cache 构建已通过，完整数据转换、Mini3 MJLab smoke、PD/碰撞调参与部署工作尚未完成。

## 1. 当前模型检查结论

导入时有两份内容相同的机器人 XML；本次适配只修改训练用的第二份：

- `humanoidverse/data/robots/mini3/mjcf/mini3.xml`
- `humanoidverse/data/robots/mini3_mjlab/mini3.xml`

建议以 `mini3_mjlab/mini3.xml` 作为 UFO/MJLab 的唯一训练模型，避免两份 XML 后续产生参数漂移。

使用原始目录中可正常解析的同一份模型检查后，结构为：

| 项目 | 当前值 | 说明 |
| --- | ---: | --- |
| `nq` | 28 | floating base 7 + 21 个 hinge joint |
| `nv` | 27 | floating base 6 + 21 个 hinge DOF |
| `nu` | 21 | policy/action 维度应为 21 |
| 非 world body 数 | 24 | `pose_aa` 应为 `[T, 24, 3]` |
| 执行器数 | 21 | 每个控制关节一个 motor |
| 总质量 | 约 12.5566 kg | 需要和真实机器人版本核对 |
| MuJoCo keyframe 数 | 1 | 已增加 28 维 `stand` keyframe |
| 根 body | `base_link` | freejoint 名为 `floating_base` |

XML 的 model 名称已由 `mini3_27` 改为 `mini3`。原名称中的 `27` 与 `nv=27` 一致，不代表有 27 个 policy action。UFO 中以下字段均已设置为 21：

```yaml
robot:
  dof_obs_size: 21
  actions_dim: 21
```

当前 `qpos0` 是：

- 根位置 `[0, 0, 0.46305]`；
- 根四元数在 MuJoCo 中为 `wxyz=[1, 0, 0, 0]`；
- 21 个关节角全部为 0；
- 零姿态下 head link 高度约为 `0.8007 m`；
- 左右 ankle-roll link 高度约为 `0.028 m`。

这些值只能作为初始检查结果，仍需确认零姿态是否稳定、脚底是否穿透地面、质心是否位于支撑区域内。

## 2. P0：先修复 XML 的 mesh 路径

导入时 `mini3_mjlab/mini3.xml` 无法从它所在的位置直接加载，原 mesh 路径是：

```xml
<mesh name="base_link_mj.STL" file="../meshes/base_link_mj.STL" />
```

但实际 mesh 位于：

```text
humanoidverse/data/robots/mini3_mjlab/meshes/
```

因此 `../meshes/` 会错误地解析到 `humanoidverse/data/robots/meshes/`。

现已使用 `compiler.meshdir` 统一管理路径，把：

```xml
<compiler angle="radian" />
```

改为：

```xml
<compiler angle="radian" meshdir="meshes" />
```

同时已把所有 mesh 的 `file` 改成不带目录的文件名：

```xml
<mesh name="base_link_mj.STL" file="base_link_mj.STL" />
```

而不是仅把 `../meshes/` 改成 `meshes/`。使用 `compiler.meshdir` 还有一个额外好处：reward inference 将 XML 复制到输出目录时，会把 `meshdir` 转成绝对路径；散落在每个 `<mesh file="...">` 中的相对路径不会被当前复制逻辑自动修正。

修复后先检查：

```bash
uv run python -m humanoidverse.tools.robot_inspect \
  --xml humanoidverse/data/robots/mini3_mjlab/mini3.xml \
  --name mini3 \
  --out /tmp/mini3_robot_check.yaml
```

预期摘要至少包含：

```text
nq/nv/nu: 28/27/21
bodies: 24
joints: 22
actuators: 21
freejoint: floating_base on body base_link
```

### XML 还需要人工确认的项目

- 保留 robot-only 的 `mini3.xml` 作为训练 asset，不要把带 floor 的 `scene.xml` 配为 `xml_path`；MJLab 会自行创建 terrain。
- 保留 21 个 motor，因为 `RobotSpec.control_joint_names` 默认从 XML actuator 顺序推导。
- 当前 XML motor 没有 `forcerange`、`ctrlrange` 和 velocity limit；不要依赖 XML 自动得到真实电机限制，相关值应在 Robot YAML 中明确配置。
- 当前训练构建器会删除 XML motor，再按 Robot YAML 创建 position-PD actuator，所以不要为适配而加入 `actuatorfrcrange`；MJLab 路径会显式拒绝包含该字段的 XML。
- 检查 collision mesh 是否过于复杂。当前双脚使用 STL mesh 直接碰撞，若接触抖动或性能差，应换成经过验证的简化 box/capsule/convex collision geom。
- 检查脚底摩擦、self-collision 和已有的 `<exclude>`。当前只显式排除了 `base_link/waist_yaw_link` 与 `waist_yaw_link/head_link` 两对接触。
- 已增加并验证 `<keyframe><key name="stand" .../></keyframe>`，使默认站立姿态成为模型的一部分；keyframe 的 `qpos` 为 28 维，并与 YAML 的零关节默认姿态一致。动态稳定性仍需在 MJLab 中验证。
- XML 中的 5 个 base site sensor 对当前训练不是必需项；可以保留，但部署侧不能假设训练代码读取了这些 sensor。

## 3. P0：生成并维护两层 Mini3 配置

UFO 对新机器人使用两层配置：

1. `configs/robots/mini3.yaml`：RobotSpec、数据格式和训练参数的 source of truth；
2. `humanoidverse/config/robot/mini3/mini3_auto.yaml`：Hydra 环境配置。

修复 XML 路径后，可以用已有 URDF 辅助生成草稿：

```bash
uv run python -m humanoidverse.tools.robot_inspect \
  --xml humanoidverse/data/robots/mini3_mjlab/mini3.xml \
  --urdf humanoidverse/data/robots/mini3/urdf/mini3.urdf \
  --name mini3 \
  --out configs/robots/mini3.yaml \
  --hydra-out humanoidverse/config/robot/mini3/mini3_auto.yaml
```

当前 XML 与 URDF 的控制关节名称一致，不需要 `--urdf-joint-name-map`。URDF 中存在真实的 effort/velocity 候选值，而 XML actuator 没有这些限制，因此 Mini3 适配中建议使用 URDF 生成初始草稿，然后逐项复核。

### 3.1 必须固定的 21DOF 顺序

Robot YAML 中建议使用显式顺序，不要长期依赖隐式的 `all_actuated`：

```yaml
control_joints:
  mode: explicit
  names:
    - left_hip_pitch_joint
    - left_hip_roll_joint
    - left_hip_yaw_joint
    - left_knee_pitch_joint
    - left_ankle_pitch_joint
    - left_ankle_roll_joint
    - right_hip_pitch_joint
    - right_hip_roll_joint
    - right_hip_yaw_joint
    - right_knee_pitch_joint
    - right_ankle_pitch_joint
    - right_ankle_roll_joint
    - waist_yaw_joint
    - left_shoulder_pitch_joint
    - left_shoulder_roll_joint
    - left_shoulder_yaw_joint
    - left_elbow_pitch_joint
    - right_shoulder_pitch_joint
    - right_shoulder_roll_joint
    - right_shoulder_yaw_joint
    - right_elbow_pitch_joint
```

这个顺序必须同时用于：

- RobotState CSV/NPZ 的 `dof_pos`；
- policy action 输出；
- default joint angles；
- effort/velocity limit 数组；
- PD 和 actuator 参数；
- 实机部署时的 joint mapping。

### 3.2 RobotSpec 语义建议

以下值适合作为第一版配置，但必须人工确认：

```yaml
name: mini3
xml_path: humanoidverse/data/robots/mini3_mjlab/mini3.xml
base_body: base_link
root_quat_order: xyzw
coordinate_system: z_up
dof_unit: rad

feet:
  - left_ankle_roll_link
  - right_ankle_roll_link

hands: []

key_bodies:
  - base_link
  - waist_yaw_link
  - left_ankle_roll_link
  - right_ankle_roll_link
  - left_elbow_pitch_link
  - right_elbow_pitch_link
```

特别注意：`training.semantics.contact_bodies` 应当恰好按“左脚、右脚”排列：

```yaml
training:
  semantics:
    contact_bodies:
      - left_ankle_roll_link
      - right_ankle_roll_link
    torso_name: waist_yaw_link
    left_ankle_dof_names:
      - left_ankle_pitch_joint
      - left_ankle_roll_joint
    right_ankle_dof_names:
      - right_ankle_pitch_joint
      - right_ankle_roll_joint
```

不能把 `left_ankle_pitch_link` 和 `left_ankle_roll_link` 当成两只脚。当前 MJLab 辅助奖励直接把 `contact_bodies[0]` 和 `[1]` 当作左、右脚，并使用 ankle DOF 列表的第二项作为 ankle-roll。

`undesired_contact_bodies` 不应完全接受自动猜测。根据当前具有 collision geom 的 body，至少人工评估这些候选项：

- `base_link`
- `waist_yaw_link`
- `head_link`
- 左右 hip-roll link
- 左右 knee-pitch link
- 左右 shoulder-roll link
- 左右 elbow-pitch link

是否惩罚膝、肘接触取决于动作数据是否包含跪地、侧手翻等技能，不能一刀切。

### 3.3 默认姿态和初始高度

当前 XML 没有 stand/default/home/init keyframe，自动工具会使用 `model.qpos0`，也就是 base 高度 `0.46305 m` 和全零关节角。

需要完成以下检查：

- 全零腿姿态是否是 Mini3 的真实可站立姿态；
- 膝关节和踝关节是否需要预弯；
- 双脚 collision 最低点是否刚好接触地面；
- 初始姿态是否在全部 joint range 内；
- `training.init_state.default_joint_angles` 是否包含全部 21 个关节；
- XML stand keyframe、Robot YAML 默认角和 motion 数据的零位定义是否一致。

如果使用非零站立角，优先把它写入 XML 的 `stand` keyframe，再重新生成草稿并复核 YAML，避免 XML、训练配置和部署默认姿态分别维护三套数值。

### 3.4 电机、PD 和 action scale

当前 URDF 给出的候选硬件限制为：

| 关节组 | effort | velocity |
| --- | ---: | ---: |
| hip、knee、waist | 27 | 10 |
| ankle pitch/roll | 25 | 45 |
| shoulder、elbow | 12.5 | 45 |

单位应与厂商定义再次确认，通常分别是 `N·m` 和 `rad/s`。

Robot YAML 必须使用：

```yaml
training:
  actuator:
    source: yaml
```

不要使用 `g1_mode15`，那是 Unitree G1 专用电机模型。`training.actuator.joints` 中的每一个 Mini3 控制关节都必须提供：

- `effort_limit`
- `velocity_limit`
- `armature`
- `friction`
- 可选的物理 `damping`

当前 XML 已给出 armature，并在 ankle 和 arm 关节上给出 frictionloss/damping。URDF 当前没有 `<dynamics>`，所以 URDF 主要补充 effort 和 velocity，不会自动给出可靠的 PD gains。

当前配置已按用户提供的 21 维 Mini3 `kps/kds` 参考数组逐关节填写，不再使用按名称套用的通用模板。仍需结合 Mini3 电机、减速比、控制频率和实机控制器验证，并一起检查：

- `training.control.stiffness` 和 `damping`；
- `action_scale`、`action_clip_value`、`normalize_action_to`；
- `training.control.effort_limit` 与 `training.actuator.joints.*.effort_limit` 是否一致；
- `action_rescale` 后每个关节的实际目标角范围；
- domain randomization 的 PD、质量、摩擦和延迟范围。

注意：当前 MJLab actuator 直接读取 `training.actuator.joints.*.effort_limit`。不要认为只调小 `effort_limit_scale` 就一定会降低 MJLab 电机饱和力矩；需要直接核对 actuator joint 参数和启动日志。

### 3.5 Hydra 配置应达到的结构

生成的 `mini3_auto.yaml` 至少应满足：

```yaml
robot:
  num_bodies: 24
  dof_obs_size: 21
  actions_dim: 21
  lower_body_actions_dim: 12
  upper_body_actions_dim: 9
  num_feet: 2
  left_foot_name: left_ankle_roll_link
  right_foot_name: right_ankle_roll_link
  foot_name: ankle_roll_link
  knee_name: knee_pitch_link
  torso_name: waist_yaw_link
```

同时检查：

- `body_names` 包含 `base_link`、固定的 `imu_link/head_link` 和其余运动 body，共 24 个；
- `dof_pos_lower_limit_list`、`dof_pos_upper_limit_list`、velocity 和 effort 数组都严格为 21 项；
- `robot.motion.asset.assetRoot` 指向 `mini3_mjlab/`；
- `robot.motion.asset.assetFileName` 是 `mini3.xml`；
- `robot.motion.extend_config` 为空，`nums_extend_bodies` 为 0，除非以后确实增加虚拟 tracking body；
- `randomize_link_body_names` 不包含不存在的 G1 body；
- 配置中没有残留 `pelvis`、`torso_link`、`left_knee_link`、wrist joint 或 `g1_29dof`。

## 4. P0：准备 Mini3 专属 RobotState 动作数据

G1 的 LaFAN pkl、29DOF CSV 和 G1 checkpoint 都不能用于 Mini3。动作必须先 retarget 到这份 Mini3 XML 的 21 个控制关节。

当前 `humanoidverse/data/csv` 中的数据已经 retarget 到 Mini3，原始格式为：

```text
Frame,
root_translateX,root_translateY,root_translateZ,
root_rotateX,root_rotateY,root_rotateZ,
left_hip_pitch_joint_dof,...,right_elbow_pitch_joint_dof
```

已确认的源数据约定为：

- 采样率为 `120 FPS`；
- root position 为厘米，转换时乘 `0.01` 得到米；
- root Euler 和 21 个关节角均为 degree；
- root Euler 顺序为 intrinsic `XYZ`；SciPy 中必须使用大写 `"XYZ"`，小写 `"xyz"` 表示 extrinsic，不能混用；
- 输出 root quaternion 为 `xyzw`，21 个关节角为 radian；
- 每一帧 root pose 和 21 个 DOF 必须属于同一时刻；
- 数据已经适配到 Mini3，UFO 不负责 retarget；
- `_M.csv` 是镜像动作，可按训练集需求保留或使用 `--exclude-mirrored` 排除。

仓库中的转换器会逐文件处理、检查字段/连续帧/有限值/XML 关节限位，并原子写入标准 RobotState NPZ：

```bash
.venv/bin/python -m humanoidverse.tools.convert_mini3_csv \
  --input humanoidverse/data/csv \
  --output-dir /tmp/mini3_npz_smoke \
  --sample-files 20 \
  --verify-output
```

默认参数已经固定为 `--fps 120 --position-scale 0.01 --euler-order XYZ`。若输出文件已经存在，默认跳过，因此中断后可以重跑续转；只有确认需要覆盖时才加 `--overwrite`。输出目录是扁平 NPZ 目录，源文件 stem 冲突时转换器会拒绝继续。

先生成一个可控规模的数据集，而不是直接把全部原始数据放进同一个训练 cache：

```bash
.venv/bin/python -m humanoidverse.tools.convert_mini3_csv \
  --input humanoidverse/data/csv \
  --output-dir /path/to/mini3_robot_state_npz_2k \
  --sample-files 2000 \
  --seed 4728 \
  --exclude-mirrored \
  --verify-output
```

如确实需要转换全部文件，去掉 `--sample-files` 和 `--exclude-mirrored`。原始目录约 13.5 万个文件、上亿帧，不建议把全量结果一次构建成单个内存/GPU 训练 cache，应按动作类别或多个 manifest dataset 分片。

无表头格式虽然也能用，但要注意：Mini3 无表头无 time 恰好也是 28 列，和 MuJoCo `qpos` 宽度相同；两者四元数顺序不同：

- RobotState CSV：`px py pz qx qy qz qw q0 ... q20`
- MuJoCo qpos：`px py pz qw qx qy qz q0 ... q20`

不能直接把 MuJoCo qpos 原样当作无表头 RobotState CSV。

转换后先检查数据：

```bash
.venv/bin/python -m humanoidverse.tools.data_inspect \
  --robot configs/robots/mini3.yaml \
  --source /path/to/mini3_robot_state_npz_2k \
  --format robot_state_npz
```

也可以直接用 MuJoCo 交互回放单个转换结果。播放器按 NPZ 中的 FPS 运动学回放 root pose 和 21 个关节，不施加重力、PD 或接触动力学，因此画面反映的是数据本身：

```bash
.venv/bin/python -m humanoidverse.tools.play_robot_state_npz \
  --npz humanoidverse/data/mini3_robot_state_npz/Idle_Left_001__A017.npz
```

常用按键：空格暂停/继续，左右方向键逐帧，上下方向键前后跳一秒，`R` 回到起点，`[`/`]` 调整倍速，`L` 切换循环，`C` 显示/隐藏碰撞体，`Q` 退出。远程机器没有图形桌面时可先只做数据、qpos 和 MuJoCo FK 检查：

```bash
.venv/bin/python -m humanoidverse.tools.play_robot_state_npz \
  --npz /path/to/motion.npz \
  --check-only \
  --print-joint-ranges
```

如果要检查转换前的原始 CSV，可直接在内存中按 120 FPS、cm→m、intrinsic `XYZ`、degree→radian 解释并播放，不会生成新的 NPZ：

```bash
.venv/bin/python -m humanoidverse.tools.play_mini3_csv \
  --csv humanoidverse/data/csv/220705/Idle_Left_001__A017.csv
```

还可以逐帧比较转换前后的 root position、root orientation 和 21 个 DOF；只有误差在浮点容差内才会打开播放器：

```bash
.venv/bin/python -m humanoidverse.tools.play_mini3_csv \
  --csv humanoidverse/data/csv/220705/Idle_Left_001__A017.csv \
  --compare-npz humanoidverse/data/mini3_robot_state_npz/Idle_Left_001__A017.npz
```

如果要直接检查 flat PKL，可使用 PKL 播放器。它直接读取 `root_rot`，不会从 CSV 欧拉角重新计算四元数；当前 Mini3 PKL 的四元数分量顺序为 `xyzw`（默认值）：

```bash
.venv/bin/python -m humanoidverse.tools.play_mini3_pkl \
  --pkl humanoidverse/data/pkl/210531/jump_and_land_heavy_001__A001_M.pkl
```

无图形桌面时可只校验 PKL 字段、关节限位、MuJoCo `qpos` 映射和前向运动学：

```bash
.venv/bin/python -m humanoidverse.tools.play_mini3_pkl \
  --pkl humanoidverse/data/pkl/210531/jump_and_land_heavy_001__A001_M.pkl \
  --check-only \
  --print-joint-ranges
```

确认 PKL 根姿态正确后，可以直接从 flat PKL 生成训练格式，不再经过 CSV 或 NPZ：

```bash
.venv/bin/python -m humanoidverse.tools.convert_mini3_pkl \
  --input humanoidverse/data/pkl \
  --output-dir humanoidverse/data/mini3_pkl_ufo \
  --manifest configs/data/mini3_pkl.yaml
```

转换器直接读取每个源文件的 `root_pos`、`root_rot`、`dof_pos` 和 `fps`。`root_rot` 默认按 `xyzw` 解释，并仅转换成 MotionLib 要求的根节点 axis-angle，不会使用 CSV 欧拉角。输出采用每个约 10 秒片段一个 UFO PKL 的目录布局，避免启动训练时将全量数据先合并成一个超大 Python 字典；不足 1 秒的尾段会丢弃。同一条命令可以安全续跑，已有输出会被复用。

转换完成后直接训练：

```bash
./run_train.sh \
  --agent fb \
  --data-manifest configs/data/mini3_pkl.yaml \
  --gpu-ids single \
  --work-dir runs/ufo_fb_mini3
```

生成 manifest 和 UFO pkl cache：

```bash
.venv/bin/python -m humanoidverse.tools.data_build \
  --robot configs/robots/mini3.yaml \
  --source /path/to/mini3_robot_state_npz_2k \
  --format robot_state_npz \
  --name mini3_motion \
  --clip-seconds 10 \
  --out configs/data/mini3_motion.yaml \
  --rebuild-cache
```

建议继续使用 RobotState 导入器生成的 pkl，因为其中会保留显式的 21 维 `dof_pos`。Mini3 包含固定的 `imu_link` 和 `head_link`；仅含 `pose_aa`、不含显式 `dof_pos` 的旧式 pkl 应额外验证 MotionLib 的固定 body/DOF 映射。

## 5. P1：训练前的 smoke 顺序

### 5.1 配置和数据静态检查

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'

uv run python -m humanoidverse.tools.data_inspect \
  --robot configs/robots/mini3.yaml \
  --source /path/to/mini3_robot_state_npz_2k \
  --format robot_state_npz
```

建议增加 Mini3 回归测试，至少断言：

- XML 可以加载；
- `nq/nv/nu == 28/27/21`；
- control joint 顺序等于本文的 21DOF 顺序；
- body 数为 24；
- `load_robot_training_spec()` 可以加载 Mini3 YAML；
- effort/velocity/default pose 数组均为 21 项；
- 一小段 RobotState fixture 可以转换并进入 MotionLib。

### 5.2 单卡 smoke training

```bash
./run_train.sh \
  --agent fb \
  --robot-config configs/robots/mini3.yaml \
  --data-manifest configs/data/mini3_motion.yaml \
  --gpu-ids single \
  --smoke \
  --work-dir /tmp/ufo_smoke_mini3
```

manifest 已声明相同的 `robot_config` 时，`--robot-config` 可以省略；首次接入建议保留，让兼容性检查尽早报错。

启动日志必须核对：

- robot 为 `mini3`；
- XML 路径为 `mini3_mjlab/mini3.xml`；
- actuator count 和 action dim 都是 21；
- joint order 与本文一致；
- actuator source 是 `yaml`；
- 打印的 kp、kd、effort、velocity、armature 和 friction 符合预期；
- observation、action、replay buffer 中没有残留 29DOF 维度；
- reset 后无 NaN、爆炸、脚底穿透或持续大接触力；
- 左右脚 contact、slippage、feet orientation 和 ankle-roll penalty 对应正确 body/DOF。

在 smoke 稳定之前，不要开启大规模环境数、长训练、强 domain randomization 或实机导出。

## 6. P1：推理适配范围

### Tracking inference

tracking 路径已经支持 robot config，但必须使用 Mini3 自己训练的 checkpoint、Mini3 robot config 和 Mini3 full motion：

```bash
uv run python -m humanoidverse.tracking_inference \
  --model-folder runs/ufo_fb_mini3 \
  --robot-config configs/robots/mini3.yaml \
  --data-path /path/to/mini3_full_motion.pkl \
  --device cuda:0 \
  --headless \
  --save-mp4 \
  --motion-list 0
```

### Goal inference

非 G1 机器人必须准备 Mini3 专属 goal JSON，并保证其中 DOF 状态为 21 维、完整 qpos 状态为 28 维：

```text
--goal-json /path/to/mini3_goal_frames.json
```

### Reward inference

当前非 G1 路径只开放 root/locomotion 类任务。涉及 G1 wrist、hand、sit/crouch 等 body 语义的默认任务不能直接用于 Mini3。Mini3 没有 wrist/hand body，如需上肢任务，需要单独实现 Mini3 的 body 语义和 reward。

另外，reward relabel 会把 XML 写入运行目录，因此第 2 节推荐的 `compiler.meshdir` 修复是必要的。

## 7. P2：当前代码中需要重点观察的兼容性风险

最小 RobotState 训练链路预期不需要先修改算法代码，但 smoke 失败时优先检查以下位置：

1. `humanoidverse/utils/motion_lib/torch_humanoid_batch.py`
   - 旧代码按 actuator 的 `name` 与 joint name 比较；Mini3 actuator 名带 `_ctrl`，会打印 21 个 unmatched 名称。
   - 本次已改为读取 motor 的 `joint` 属性，并同时兼容 `<freejoint>` 与 `<joint type="free">`。
   - Mini3 有固定 `imu_link/head_link`，旧式不含 `dof_pos` 的 motion pkl 可能触发错误的 pose-to-DOF fallback；RobotState 生成的 pkl 会保留正确的 `dof_pos`。

2. `humanoidverse/agents/evaluations/humanoidverse_mjlab.py`
   - 文件中仍保留 G1 body 名称和旧的固定切片辅助代码；当前主要 tracking path 使用动态 MotionLib 状态，但修改 evaluation 流程时不要重新启用这些 G1 固定映射。

3. `humanoidverse/envs/g1_env_helper/rewards.py`
   - 包含大量 `pelvis`、`torso_link`、wrist 和 G1 身高常量。
   - 这些不能作为 Mini3 通用 reward；非 G1 reward inference 当前只应使用已开放的 locomotion 子集。

4. `humanoidverse/agents/envs/humanoidverse_mjlab.py`
   - 当前辅助奖励无条件假设至少两只脚；contact body 顺序错误会让左右脚奖励直接算错。
   - `penalty_ankle_roll` 启用时，左右 ankle DOF 列表必须至少各含 pitch、roll 两项，并把 roll 放在第二项。

如果这些兼容性点在 smoke 中确实触发，应增加 Mini3 回归测试后再修改代码，不建议通过伪造 G1 body 名称来绕过。

## 8. P2：实机部署还需要单独完成的内容

`main` 分支的仿真适配不等于实机部署完成。部署侧至少还要确认：

- ONNX 输出 21 维 action 的顺序；
- 实机 SDK joint index 与本文 21DOF 顺序的映射；
- 编码器零位、方向符号、弧度单位；
- policy observation 的 DOF、IMU、角速度、重力投影和历史帧顺序；
- 控制频率、仿真 decimation 与实机循环频率；
- position target 的 action scale 和 default pose；
- 每关节 position/velocity/torque 限制；
- 急停、超时、通信丢包、姿态异常和跌倒保护；
- 仿真 actuator 与实机底层 PD/FOC 控制的职责边界。

必须使用 Mini3 checkpoint 导出的 ONNX 和 metadata，不能使用 G1 checkpoint 或只替换输出层。

## 9. 文件级修改清单

### 必须修改或创建

- [x] 修改 `humanoidverse/data/robots/mini3_mjlab/mini3.xml` 的 mesh 路径。
- [x] 在 XML/YAML 中确定并统一第一版 stand pose（动态稳定性待 smoke 验证）。
- [x] 创建 `configs/robots/mini3.yaml` 第一版（已填入参考 PD，动态稳定性和碰撞参数仍待校准）。
- [x] 创建并完成结构复核 `humanoidverse/config/robot/mini3/mini3_auto.yaml`。
- [x] 接入 Mini3 专属原始 CSV，并提供标准 RobotState NPZ 转换器（全量转换仍待执行）。
- [ ] 创建 `configs/data/mini3_motion.yaml`。
- [x] 增加 Mini3 XML、RobotSpec 和 Hydra 配置回归测试。
- [x] 增加 Mini3 CSV 单位、intrinsic XYZ、关节顺序、限位和 NPZ 读取回归测试。

### 按功能选做

- [ ] 创建 Mini3 goal JSON。
- [ ] 扩展 Mini3 上肢/姿态 reward。
- [x] 修正 MotionLib 对 actuator name 和 freejoint 写法的遗留假设。
- [ ] 如需兼容不含显式 `dof_pos` 的旧 pkl，再处理固定 body 的 pose-to-DOF fallback。
- [ ] 在 deploy 分支实现 Mini3 观测/action/runtime 映射。

### 不需要为了 Mini3 直接修改

- `humanoidverse/train.py` 的 G1 默认路径：只要命令或 manifest 显式传入 Mini3 robot config 即可。
- G1 的 robot YAML、Hydra YAML 和 29DOF 数据。
- XML 中现有的 base site sensor，除非部署或自定义 reward 明确需要不同 sensor。
- URDF runtime 路径：URDF 仅用于生成配置草稿和核对硬件参数，训练运行时仍使用 MuJoCo XML。

## 10. 完成标准

只有同时满足以下条件，才建议认为 Mini3 的最小训练适配完成：

- [x] `mini3_mjlab/mini3.xml` 可以在当前位置加载全部 mesh。
- [x] RobotSpec 显示 21 个 control joint，顺序固定且无重复。
- [x] Hydra 配置显示 24 body、21 action、两只脚。
- [ ] stand pose 无穿透、无明显自碰撞，并能在 PD 下稳定保持。
- [ ] RobotState 数据是 Mini3 21DOF，四元数、单位和关节方向正确。
- [ ] data inspect/build 成功，MotionLib 输出 DOF 维度为 21。
- [ ] 单卡 smoke training 可以完成且没有 NaN/shape mismatch。
- [ ] tracking inference 可以用 Mini3 XML 渲染专家和 policy rollout。
- [ ] 导出的 ONNX metadata 记录 `robot_name=mini3`、`num_dof=21` 和完整 joint order。
- [ ] 实机部署另行通过限幅、急停和低增益吊装测试。
