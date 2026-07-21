"""Pure MuJoCo sim-to-sim tracking inference for UFO policies.

Unlike :mod:`humanoidverse.tracking_inference`, this entrypoint owns the
``mujoco.MjModel`` and advances its dynamics with ``mujoco.mj_step``.  It is
intended for local controller debugging: physics runs at the robot training
rate, the policy runs at the configured decimation, and the interactive viewer
shows contacts and contact forces from that same simulation instance.  Its
default Mini3 torque path also mirrors the motor-side current-loop response,
T-N envelopes, KT calibration, and parallel-ankle mapping from mini3_lab's
``sim2sim_mini3_bm.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils._pytree import tree_map

from humanoidverse.agents.envs.humanoidverse_mjlab import _compose_humanoidverse_config
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.utils.helpers import get_backward_observation
from humanoidverse.utils.mini3_real_motor import Mini3RealMotorModel
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from humanoidverse.utils.robot_spec import load_robot_training_spec, resolve_robot_config_path

DEFAULT_ROBOT_CONFIG = "configs/robots/mini3.yaml"


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(name, nargs="?", const=True, default=default, type=_str2bool, help=help_text)


def _quat_rotate_inverse_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / max(float(np.linalg.norm(quat)), 1.0e-12)
    w, x, y, z = quat
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation.T @ np.asarray(vector, dtype=np.float64)


def _match_joint_value(joint_name: str, values: dict[str, Any]) -> float:
    if joint_name in values:
        return float(values[joint_name])
    for pattern, value in values.items():
        if str(pattern) in joint_name:
            return float(value)
    raise KeyError(f"No value configured for joint {joint_name!r}")


def _saved_robot_training(run_config: dict[str, Any], local_robot_config: Path) -> dict[str, Any]:
    env_config = run_config.get("env")
    if not isinstance(env_config, dict):
        raise ValueError(
            "The run root config.json must contain an 'env' mapping. Do not use checkpoint/model/config.json as the run config."
        )
    saved = env_config.get("robot_training")
    if not isinstance(saved, dict):
        raise ValueError("Run config does not contain env.robot_training; this checkpoint cannot be reconstructed safely")

    result = copy.deepcopy(saved)
    local = load_robot_training_spec(local_robot_config)
    saved_robot = result.get("robot")
    if not isinstance(saved_robot, dict):
        raise ValueError("Run config env.robot_training.robot must be a mapping")
    saved_joints = [str(name) for name in saved_robot.get("control_joint_names", [])]
    local_joints = list(local.robot.control_joint_names)
    if saved_joints != local_joints:
        raise ValueError(f"Checkpoint/local robot joint order mismatch: saved={saved_joints}, local={local_joints}")
    saved_name = str(saved_robot.get("name", ""))
    if saved_name and saved_name != local.robot.name:
        raise ValueError(f"Checkpoint robot {saved_name!r} does not match local robot {local.robot.name!r}")

    # Preserve all controller parameters captured at training start, replacing
    # only paths that may refer to another machine.
    saved_robot["xml_path"] = str(Path(local.robot.xml_path).expanduser().resolve())
    result["config_path"] = str(local.config_path)
    result.setdefault("action_rescale", True)  # Legacy checkpoints predate the explicit field.
    return result


def _compose_motion_config(
    run_config: dict[str, Any],
    *,
    data_path: Path,
    robot_training: dict[str, Any],
) -> Any:
    env_config = run_config["env"]
    hv_config, _ = _compose_humanoidverse_config(
        num_envs=1,
        relative_config_path=str(env_config.get("relative_config_path", "exp/bfm_zero/bfm_zero")),
        hydra_overrides=list(env_config.get("hydra_overrides", [])),
        lafan_tail_path=str(data_path),
        data_mix_weights=None,
        disable_obs_noise=True,
        disable_domain_randomization=True,
        max_episode_length_s=None,
        root_height_obs=bool(env_config.get("root_height_obs", False)),
        robot_training=robot_training,
    )
    return hv_config


@dataclass(frozen=True)
class JointLayout:
    names: tuple[str, ...]
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    actuator_adr: np.ndarray


def _joint_layout(model: mujoco.MjModel, joint_names: list[str]) -> JointLayout:
    qpos_adr: list[int] = []
    dof_adr: list[int] = []
    actuator_adr: list[int] = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint {joint_name!r}")
        qpos_adr.append(int(model.jnt_qposadr[joint_id]))
        dof_adr.append(int(model.jnt_dofadr[joint_id]))
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if matches.size != 1:
            raise ValueError(f"Expected one actuator for joint {joint_name!r}, found {matches.tolist()}")
        actuator_adr.append(int(matches[0]))
    return JointLayout(
        names=tuple(joint_names),
        qpos_adr=np.asarray(qpos_adr, dtype=np.int32),
        dof_adr=np.asarray(dof_adr, dtype=np.int32),
        actuator_adr=np.asarray(actuator_adr, dtype=np.int32),
    )


def _build_mujoco_model(
    xml_path: Path,
    robot_training: dict[str, Any],
    *,
    physics_hz: float,
    ground_friction: float,
) -> tuple[mujoco.MjModel, JointLayout]:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    spec.worldbody.add_geom(
        name="sim2sim_ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.01],
        rgba=[0.35, 0.37, 0.40, 1.0],
        contype=1,
        conaffinity=1,
        friction=[float(ground_friction), 0.005, 0.0001],
    )
    spec.worldbody.add_light(
        name="sim2sim_key_light",
        pos=[0.0, -3.0, 4.0],
        dir=[0.2, 0.5, -1.0],
        diffuse=[0.8, 0.8, 0.8],
        ambient=[0.35, 0.35, 0.35],
        specular=[0.1, 0.1, 0.1],
    )
    model = spec.compile()
    model.opt.timestep = 1.0 / float(physics_hz)

    joint_names = [str(name) for name in robot_training["robot"]["control_joint_names"]]
    layout = _joint_layout(model, joint_names)
    actuator_joints = robot_training.get("actuator", {}).get("joints", {})
    if not isinstance(actuator_joints, dict):
        raise ValueError("Pure MuJoCo inference currently requires training.actuator.source=yaml with per-joint parameters")
    for index, joint_name in enumerate(joint_names):
        params = actuator_joints.get(joint_name)
        if not isinstance(params, dict):
            raise ValueError(f"Missing saved actuator parameters for joint {joint_name!r}")
        dof_id = layout.dof_adr[index]
        model.dof_armature[dof_id] = float(params["armature"])
        model.dof_frictionloss[dof_id] = float(params["friction"])
        model.dof_damping[dof_id] = float(params.get("viscous_friction", params.get("damping", 0.0)))
    return model, layout


class PhysicsStepActionDelay:
    def __init__(self, delay_steps: np.ndarray):
        self.delay_steps = np.asarray(delay_steps, dtype=np.int64)
        self.history = np.zeros((int(self.delay_steps.max(initial=0)) + 1, self.delay_steps.size), dtype=np.float64)
        self.cursor = 0

    def reset(self) -> None:
        self.history.fill(0.0)
        self.cursor = 0

    def push(self, action: np.ndarray) -> np.ndarray:
        self.cursor = (self.cursor + 1) % self.history.shape[0]
        self.history[self.cursor] = action
        indices = np.remainder(self.cursor - self.delay_steps, self.history.shape[0])
        return self.history[indices, np.arange(self.delay_steps.size)].copy()


class PhysicsStepImuDelay:
    def __init__(self, physics_dt: float, delay_seconds: float, max_delay_seconds: float, *, interpolate: bool):
        self.physics_dt = float(physics_dt)
        self.delay_seconds = float(delay_seconds)
        self.interpolate = bool(interpolate)
        self.history = np.zeros((math.ceil(float(max_delay_seconds) / self.physics_dt) + 2, 6), dtype=np.float64)
        self.write_idx = 0

    def reset(self, current_imu: np.ndarray) -> None:
        self.history[:] = np.asarray(current_imu, dtype=np.float64)
        self.write_idx = 0

    def record(self, current_imu: np.ndarray) -> None:
        self.history[self.write_idx] = current_imu
        self.write_idx = (self.write_idx + 1) % self.history.shape[0]

    def _read_steps(self, steps: int) -> np.ndarray:
        steps = int(np.clip(steps, 0, self.history.shape[0] - 1))
        return self.history[(self.write_idx - 1 - steps) % self.history.shape[0]]

    def read(self) -> np.ndarray:
        delay_steps = self.delay_seconds / self.physics_dt
        if not self.interpolate:
            return self._read_steps(int(round(delay_steps))).copy()
        lower = math.floor(delay_steps)
        alpha = delay_steps - lower
        return ((1.0 - alpha) * self._read_steps(lower) + alpha * self._read_steps(lower + 1)).copy()


class ActorHistory:
    def __init__(self, history_lengths: dict[str, int], obs_dims: dict[str, int]):
        self.lengths = {str(key): int(value) for key, value in history_lengths.items()}
        self.buffers = {key: np.zeros((length, int(obs_dims[key])), dtype=np.float32) for key, length in self.lengths.items()}

    def reset(self) -> None:
        for buffer in self.buffers.values():
            buffer.fill(0.0)

    def query_flat(self) -> np.ndarray:
        return np.concatenate([self.buffers[key][: self.lengths[key]].reshape(-1) for key in sorted(self.lengths)])

    def add(self, values: dict[str, np.ndarray]) -> None:
        for key in self.lengths:
            buffer = self.buffers[key]
            buffer[1:] = buffer[:-1]
            buffer[0] = np.asarray(values[key], dtype=np.float32)


def _sample_action_delays(robot_training: dict[str, Any], rng: np.random.Generator, *, enabled: bool) -> np.ndarray:
    joint_names = [str(name) for name in robot_training["robot"]["control_joint_names"]]
    delays = np.zeros(len(joint_names), dtype=np.int64)
    if not enabled:
        return delays
    groups = robot_training.get("actuator", {}).get("delay_groups", {})
    if not isinstance(groups, dict):
        return delays
    joint_index = {name: index for index, name in enumerate(joint_names)}
    for group_name, group in groups.items():
        lower, upper = int(group["min_delay"]), int(group["max_delay"])
        sampled = int(rng.integers(lower, upper + 1)) if lower < upper else lower
        for joint_name in group["joint_names"]:
            if joint_name not in joint_index:
                raise ValueError(f"Delay group {group_name!r} contains unknown joint {joint_name!r}")
            delays[joint_index[joint_name]] = sampled
    return delays


def _dc_motor_clip(torque: np.ndarray, velocity: np.ndarray, effort: np.ndarray, velocity_limit: np.ndarray) -> np.ndarray:
    # Matches MJLab DcMotorActuator when saturation_effort == effort_limit.
    clipped_velocity = np.clip(velocity, -2.0 * velocity_limit, 2.0 * velocity_limit)
    upper = np.minimum(effort * (1.0 - clipped_velocity / velocity_limit), effort)
    lower = np.maximum(effort * (-1.0 - clipped_velocity / velocity_limit), -effort)
    return np.clip(torque, lower, upper)


class PureMujocoController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        layout: JointLayout,
        robot_training: dict[str, Any],
        hv_config: Any,
        *,
        action_delay_steps: np.ndarray,
        imu_delay_seconds: float,
        imu_delay_enabled: bool,
        imu_delay_interpolate: bool,
        real_motor_enabled: bool,
        tn_torque_limit_enabled: bool,
        tn_limit_after_response: bool,
        torque_response_enabled: bool,
        kt_output_model_enabled: bool,
        torque_response_kp: float,
        torque_response_ki: float,
        torque_response_plant_tau_s: float,
        torque_response_delay_steps: float,
        ankle_motor_torque_limit: float,
    ):
        self.model = model
        self.data = data
        self.layout = layout
        self.robot_training = robot_training
        self.joint_count = len(layout.names)
        self.default_pos = np.asarray([float(robot_training["default_joint_angles"][name]) for name in layout.names], dtype=np.float64)
        self.kp = np.asarray([_match_joint_value(name, robot_training["stiffness"]) for name in layout.names])
        self.kd = np.asarray([_match_joint_value(name, robot_training["damping"]) for name in layout.names])
        actuator = robot_training["actuator"]["joints"]
        self.effort = np.asarray([float(actuator[name]["effort_limit"]) for name in layout.names])
        self.velocity_limit = np.asarray([float(actuator[name]["velocity_limit"]) for name in layout.names])
        self.action_scale = np.full(self.joint_count, float(robot_training["action_scale"]), dtype=np.float64)
        if bool(robot_training.get("action_rescale", True)):
            configured_effort = np.asarray(robot_training["effort_limits"], dtype=np.float64)
            self.action_scale *= configured_effort / self.kp
        self.action_clip = float(robot_training["action_clip_value"])
        self.normalize_action_to = float(robot_training["normalize_action_to"])
        self.current_action = np.zeros(self.joint_count, dtype=np.float64)
        self.applied_action = np.zeros_like(self.current_action)
        self.target_pos = self.default_pos.copy()
        self.torque = np.zeros_like(self.current_action)
        self.action_delay = PhysicsStepActionDelay(action_delay_steps)
        self.real_motor = (
            Mini3RealMotorModel(
                tuple(layout.names),
                self.kp,
                self.kd,
                self.effort,
                dt=float(model.opt.timestep),
                response_enabled=torque_response_enabled,
                tn_enabled=tn_torque_limit_enabled,
                tn_limit_after_response=tn_limit_after_response,
                kt_enabled=kt_output_model_enabled,
                response_kp=torque_response_kp,
                response_ki=torque_response_ki,
                response_plant_tau_s=torque_response_plant_tau_s,
                response_delay_steps=torque_response_delay_steps,
                ankle_motor_torque_limit=ankle_motor_torque_limit,
            )
            if real_motor_enabled
            else None
        )

        obs_scales = OmegaConf.to_container(hv_config.obs.obs_scales, resolve=True)
        self.obs_scales = {str(key): float(value) for key, value in obs_scales.items()}
        history_lengths = OmegaConf.to_container(hv_config.obs.obs_auxiliary["history_actor"], resolve=True)
        raw_obs_dims = OmegaConf.to_container(hv_config.obs.obs_dims, resolve=True)
        if isinstance(raw_obs_dims, dict):
            obs_dims = {str(key): int(value) for key, value in raw_obs_dims.items()}
        else:
            obs_dims = {str(key): int(value) for item in raw_obs_dims for key, value in item.items()}
        self.history = ActorHistory(history_lengths, obs_dims)

        imu_cfg = robot_training.get("imu_delay", {})
        max_delay = float((imu_cfg.get("time_range_s") or [0.0, 0.0])[1])
        self.imu_delay = (
            PhysicsStepImuDelay(
                float(model.opt.timestep),
                imu_delay_seconds,
                max_delay,
                interpolate=imu_delay_interpolate,
            )
            if imu_delay_enabled and max_delay > 0.0
            else None
        )

    def current_imu(self) -> np.ndarray:
        root_quat_wxyz = self.data.qpos[3:7]
        base_ang_vel = self.data.qvel[3:6]
        projected_gravity = _quat_rotate_inverse_wxyz(root_quat_wxyz, np.array([0.0, 0.0, -1.0]))
        return np.concatenate([base_ang_vel, projected_gravity])

    def reset(self) -> None:
        self.current_action.fill(0.0)
        self.applied_action.fill(0.0)
        self.target_pos[:] = self.default_pos
        self.torque.fill(0.0)
        self.action_delay.reset()
        if self.real_motor is not None:
            self.real_motor.reset()
        self.history.reset()
        if self.imu_delay is not None:
            self.imu_delay.reset(self.current_imu())

    def set_policy_action(self, actor_action: np.ndarray) -> None:
        actor_action = np.asarray(actor_action, dtype=np.float64).reshape(-1)
        if actor_action.size != self.joint_count:
            raise ValueError(f"Expected actor action size {self.joint_count}, got {actor_action.size}")
        self.current_action = np.clip(actor_action * self.normalize_action_to, -self.action_clip, self.action_clip)

    def observation(self, device: str) -> dict[str, torch.Tensor]:
        joint_pos = self.data.qpos[self.layout.qpos_adr]
        joint_vel = self.data.qvel[self.layout.dof_adr]
        imu = self.current_imu() if self.imu_delay is None else self.imu_delay.read()
        values = {
            "actions": self.current_action * self.obs_scales.get("actions", 1.0),
            "base_ang_vel": imu[:3] * self.obs_scales.get("base_ang_vel", 1.0),
            "dof_pos": (joint_pos - self.default_pos) * self.obs_scales.get("dof_pos", 1.0),
            "dof_vel": joint_vel * self.obs_scales.get("dof_vel", 1.0),
            "projected_gravity": imu[3:] * self.obs_scales.get("projected_gravity", 1.0),
        }
        history_actor = self.history.query_flat() * self.obs_scales.get("history_actor", 1.0)
        state = np.concatenate([values["dof_pos"], values["dof_vel"], values["projected_gravity"], values["base_ang_vel"]])
        observation = {
            "state": torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0),
            "last_action": torch.as_tensor(values["actions"], device=device, dtype=torch.float32).unsqueeze(0),
            "history_actor": torch.as_tensor(history_actor, device=device, dtype=torch.float32).unsqueeze(0),
        }
        self.history.add(values)
        return observation

    def physics_step(self) -> None:
        self.applied_action = self.action_delay.push(self.current_action)
        self.target_pos = self.default_pos + self.applied_action * self.action_scale
        joint_pos = self.data.qpos[self.layout.qpos_adr]
        joint_vel = self.data.qvel[self.layout.dof_adr]
        raw_torque = self.kp * (self.target_pos - joint_pos) - self.kd * joint_vel
        if self.real_motor is None:
            self.torque = _dc_motor_clip(raw_torque, joint_vel, self.effort, self.velocity_limit)
        else:
            self.torque = self.real_motor.compute(self.target_pos, joint_pos, joint_vel)
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.layout.actuator_adr] = self.torque
        mujoco.mj_step(self.model, self.data)
        if self.imu_delay is not None:
            self.imu_delay.record(self.current_imu())


class DebugViewer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, *, distance: float, azimuth: float, elevation: float):
        import mujoco.viewer

        self.paused = False
        self.follow = True
        self.reset_requested = False
        self.quit_requested = False
        self.show_contacts = False
        self.contact_flags_dirty = False

        def key_callback(keycode: int) -> None:
            if keycode == 32:  # Space
                self.paused = not self.paused
                print(f"[viewer] {'paused' if self.paused else 'running'}", flush=True)
            elif keycode in (70, 102):  # F
                self.follow = not self.follow
                print(f"[viewer] camera_follow={self.follow}", flush=True)
            elif keycode in (82, 114):  # R
                self.reset_requested = True
            elif keycode in (67, 99):  # C
                self.show_contacts = not self.show_contacts
                self.contact_flags_dirty = True
                print(f"[viewer] show_contact_forces={self.show_contacts}", flush=True)
            elif keycode in (81, 113):  # Q
                self.quit_requested = True

        self.model = model
        self.data = data
        self.viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
        with self.viewer.lock():
            self.viewer.cam.distance = float(distance)
            self.viewer.cam.azimuth = float(azimuth)
            self.viewer.cam.elevation = float(elevation)
        self.viewer.sync()
        print("[viewer] Controls: Space pause | R reset | F camera follow | C contact forces | Q quit", flush=True)

    def _apply_contact_flags(self) -> None:
        if not self.contact_flags_dirty:
            return
        with self.viewer.lock():
            enabled = int(self.show_contacts)
            self.viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_CONTACTPOINT)] = enabled
            self.viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_CONTACTFORCE)] = enabled
        self.contact_flags_dirty = False

    def running(self) -> bool:
        return not self.quit_requested and self.viewer.is_running()

    def wait_if_paused(self) -> bool:
        while self.running() and self.paused and not self.reset_requested:
            self._apply_contact_flags()
            self.viewer.sync()
            time.sleep(0.01)
        return self.running()

    def consume_reset(self) -> bool:
        requested = self.reset_requested
        self.reset_requested = False
        return requested

    def sync(self, *, step: int, torque: np.ndarray) -> None:
        self._apply_contact_flags()
        if self.follow:
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = [float(self.data.qpos[0]), float(self.data.qpos[1]), max(float(self.data.qpos[2]), 0.4)]
        self.viewer.set_texts(
            (
                None,
                None,
                "UFO pure MuJoCo sim2sim",
                f"step={step}  root_z={self.data.qpos[2]:.3f}  ncon={self.data.ncon}  max|tau|={np.max(np.abs(torque)):.2f}",
            )
        )
        self.viewer.sync()

    def close(self) -> None:
        self.viewer.close()


def _motion_reference(
    hv_config: Any,
    *,
    device: str,
    motion_id: int,
    policy_dt: float,
    root_height_obs: bool,
) -> tuple[MotionLibRobot, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    hv_config.robot.motion.step_dt = float(policy_dt)
    motion_lib = MotionLibRobot(hv_config.robot.motion, num_envs=1, device=device)
    motion_lib.load_all_motions()
    if motion_id < 0 or motion_id >= motion_lib.num_motions():
        raise IndexError(f"motion_id={motion_id} is outside [0, {motion_lib.num_motions() - 1}]")
    default_pos = torch.tensor(
        [float(hv_config.robot.init_state.default_joint_angles[name]) for name in hv_config.robot.dof_names],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    helper_env = SimpleNamespace(
        _motion_lib=motion_lib,
        dt=float(policy_dt),
        device=device,
        default_dof_pos=default_pos,
        use_contact_in_obs_max=bool(hv_config.get("use_contact_in_obs_max", False)),
        config=hv_config,
        gravity_vec=torch.tensor([[0.0, 0.0, -1.0]], device=device),
    )
    if helper_env.use_contact_in_obs_max:
        raise NotImplementedError("Pure MuJoCo inference does not yet support use_contact_in_obs_max=True")
    backward_obs, ref = get_backward_observation(helper_env, motion_id, use_root_height_obs=root_height_obs)
    return motion_lib, backward_obs, ref


@torch.no_grad()
def _tracking_z(model: torch.nn.Module, backward_obs: dict[str, torch.Tensor], device: str) -> torch.Tensor:
    obs = tree_map(lambda value: value[1:].to(device) if hasattr(value, "to") else value, backward_obs)
    return model.project_z(model.backward_map(obs))


def _initialize_from_reference(
    data: mujoco.MjData,
    layout: JointLayout,
    ref: dict[str, torch.Tensor],
    *,
    start_step: int,
    root_z_offset: float,
    zero_init_velocity: bool,
) -> None:
    index = min(max(int(start_step), 0), int(ref["dof_pos"].shape[0]) - 1)
    root_pos = ref["ref_body_pos"][index, 0].detach().cpu().numpy().astype(np.float64)
    root_xyzw = ref["ref_body_rots"][index, 0].detach().cpu().numpy().astype(np.float64)
    root_wxyz = root_xyzw[[3, 0, 1, 2]]
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:3] = root_pos
    data.qpos[2] += float(root_z_offset)
    data.qpos[3:7] = root_wxyz
    data.qpos[layout.qpos_adr] = ref["dof_pos"][index].detach().cpu().numpy()
    if not zero_init_velocity:
        root_vel = ref["ref_body_vels"][index, 0].detach().cpu().numpy()
        root_ang_vel_world = ref["ref_body_angular_vels"][index, 0].detach().cpu().numpy()
        data.qvel[:3] = root_vel
        data.qvel[3:6] = _quat_rotate_inverse_wxyz(root_wxyz, root_ang_vel_world)
        data.qvel[layout.dof_adr] = ref["ref_dof_vel"][index].detach().cpu().numpy()
    data.ctrl[:] = 0.0


def run(args: argparse.Namespace) -> None:
    model_folder = args.model_folder.expanduser().resolve()
    run_config_path = model_folder / "config.json"
    checkpoint_dir = model_folder / "checkpoint"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run config: {run_config_path}")
    if not (checkpoint_dir / "model" / "model.safetensors").exists():
        raise FileNotFoundError(f"Missing checkpoint model: {checkpoint_dir / 'model/model.safetensors'}")
    run_config = json.loads(run_config_path.read_text())

    robot_config = resolve_robot_config_path(args.robot_config)
    robot_training = _saved_robot_training(run_config, robot_config)
    xml_path = Path(robot_training["robot"]["xml_path"])
    data_path = args.data_path.expanduser().resolve()
    hv_config = _compose_motion_config(run_config, data_path=data_path, robot_training=robot_training)

    physics_hz = float(hv_config.simulator.config.sim.fps)
    decimation = int(hv_config.simulator.config.sim.control_decimation)
    policy_hz = physics_hz / decimation
    policy_dt = 1.0 / policy_hz
    if not math.isclose(physics_hz, args.physics_hz, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(f"Checkpoint physics_hz={physics_hz:g}, but --physics-hz={args.physics_hz:g}")
    if not math.isclose(policy_hz, args.policy_hz, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(f"Checkpoint policy_hz={policy_hz:g}, but --policy-hz={args.policy_hz:g}")

    torch_device = torch.device(args.device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
        load_device = "cuda"
    elif torch_device.type == "cpu":
        load_device = "cpu"
    else:
        raise ValueError(f"Unsupported device: {args.device}")
    policy = load_model_from_checkpoint_dir(checkpoint_dir, device=load_device).to(args.device).eval()

    motion_lib, backward_obs, ref = _motion_reference(
        hv_config,
        device=args.device,
        motion_id=args.motion_id,
        policy_dt=policy_dt,
        root_height_obs=bool(run_config["env"].get("root_height_obs", False)),
    )
    del motion_lib
    z = _tracking_z(policy, backward_obs, args.device)
    if args.start_step:
        z = z[args.start_step :]
    if args.max_steps is not None:
        z = z[: args.max_steps]
    if len(z) == 0:
        raise ValueError("Selected motion range contains no policy steps")
    output_dir = model_folder / "mujoco_tracking_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(z.detach().cpu().numpy(), output_dir / f"zs_{args.motion_id}.pkl")

    model, layout = _build_mujoco_model(
        xml_path,
        robot_training,
        physics_hz=physics_hz,
        ground_friction=args.ground_friction,
    )
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)
    delay_steps = _sample_action_delays(robot_training, rng, enabled=not args.disable_action_delay)
    imu_cfg = robot_training.get("imu_delay", {})
    imu_range = [float(value) for value in imu_cfg.get("time_range_s", [0.0, 0.0])]
    imu_enabled = bool(imu_cfg.get("enabled", False)) and not args.disable_imu_delay
    if not imu_enabled:
        imu_delay_seconds = 0.0
    elif imu_range[0] < imu_range[1]:
        imu_delay_seconds = float(rng.uniform(*imu_range))
    else:
        imu_delay_seconds = imu_range[0]
    controller = PureMujocoController(
        model,
        data,
        layout,
        robot_training,
        hv_config,
        action_delay_steps=delay_steps,
        imu_delay_seconds=imu_delay_seconds,
        imu_delay_enabled=imu_enabled,
        imu_delay_interpolate=bool(imu_cfg.get("interpolate", True)),
        real_motor_enabled=args.enable_real_motor,
        tn_torque_limit_enabled=args.enable_tn_torque_limit,
        tn_limit_after_response=args.tn_limit_after_response,
        torque_response_enabled=args.enable_torque_response,
        kt_output_model_enabled=args.enable_kt_output_model,
        torque_response_kp=args.torque_response_kp,
        torque_response_ki=args.torque_response_ki,
        torque_response_plant_tau_s=args.torque_response_plant_tau_ms / 1000.0,
        torque_response_delay_steps=args.torque_response_delay_steps,
        ankle_motor_torque_limit=args.ankle_motor_torque_limit,
    )

    viewer = (
        None
        if args.headless
        else DebugViewer(
            model,
            data,
            distance=args.camera_distance,
            azimuth=args.camera_azimuth,
            elevation=args.camera_elevation,
        )
    )
    print(
        "[sim2sim] "
        f"physics={physics_hz:g}Hz policy={policy_hz:g}Hz decimation={decimation} "
        f"action_delay_steps={delay_steps.tolist()} imu_delay_ms={imu_delay_seconds * 1000.0:.3f} "
        f"action_rescale={bool(robot_training.get('action_rescale', True))} "
        f"normalize_to={float(robot_training['normalize_action_to']):g}",
        flush=True,
    )
    if controller.real_motor is None:
        print("[sim2sim] Mini3 real-motor chain disabled; using the training-equivalent MJLab DC motor clip", flush=True)
    else:
        print("[sim2sim] Mini3 real-motor model enabled:", flush=True)
        for line in controller.real_motor.summary_lines():
            print(f"  {line}", flush=True)
        if args.enable_torque_response:
            print(
                "  current-loop response: "
                f"kp={args.torque_response_kp:g} ki={args.torque_response_ki:g} "
                f"plant_tau={args.torque_response_plant_tau_ms:g}ms "
                f"delay={args.torque_response_delay_steps:g} physics steps",
                flush=True,
            )

    def reset_simulation() -> dict[str, torch.Tensor]:
        mujoco.mj_resetData(model, data)
        _initialize_from_reference(
            data,
            layout,
            ref,
            start_step=args.start_step,
            root_z_offset=args.root_z_offset,
            zero_init_velocity=args.zero_init_velocity,
        )
        mujoco.mj_forward(model, data)
        controller.reset()
        return controller.observation(args.device)

    observation = reset_simulation()
    step = 0
    try:
        while True:
            if viewer is not None:
                if not viewer.wait_if_paused():
                    break
                if viewer.consume_reset():
                    observation = reset_simulation()
                    step = 0
            cycle_started = time.perf_counter()
            with torch.no_grad():
                action = policy.act(observation, z[step].unsqueeze(0), mean=True)[0].detach().cpu().numpy()
            controller.set_policy_action(action)
            for _ in range(decimation):
                controller.physics_step()
            observation = controller.observation(args.device)
            step += 1

            if viewer is not None:
                viewer.sync(step=step, torque=controller.torque)
            if step == 1 or step % args.log_every == 0 or step == len(z):
                print(
                    f"[sim2sim] step={step}/{len(z)} root_z={data.qpos[2]:.4f} ncon={data.ncon} "
                    f"max_torque={np.max(np.abs(controller.torque)):.3f}",
                    flush=True,
                )
            if step >= len(z):
                if args.loop:
                    observation = reset_simulation()
                    step = 0
                else:
                    break
            if args.realtime and viewer is not None:
                remaining = policy_dt - (time.perf_counter() - cycle_started)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        if viewer is not None:
            viewer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure MuJoCo sim2sim inference for UFO tracking policies.")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, default=Path(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--motion-id", type=int, default=0)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--physics-hz", type=float, default=500.0)
    parser.add_argument("--policy-hz", type=float, default=50.0)
    parser.add_argument("--ground-friction", type=float, default=1.0)
    parser.add_argument("--root-z-offset", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    _add_bool_arg(parser, "--headless", False, "Run without the interactive MuJoCo viewer.")
    _add_bool_arg(parser, "--loop", False, "Reset and replay the selected motion continuously.")
    _add_bool_arg(parser, "--realtime", True, "Rate-limit viewer rollout to policy_hz.")
    _add_bool_arg(parser, "--zero-init-velocity", False, "Zero root and joint velocities when resetting.")
    _add_bool_arg(parser, "--disable-action-delay", False, "Disable saved actuator physics-step delays.")
    _add_bool_arg(parser, "--disable-imu-delay", False, "Disable saved IMU delay.")
    _add_bool_arg(
        parser,
        "--enable-real-motor",
        True,
        "Use the Mini3 motor-space response, T-N, KT, and parallel-ankle torque chain.",
    )
    _add_bool_arg(parser, "--enable-tn-torque-limit", True, "Enable the calibrated 4340P/4310P T-N envelopes.")
    _add_bool_arg(parser, "--tn-limit-after-response", True, "Reapply T-N limits after the current-loop response.")
    _add_bool_arg(parser, "--enable-torque-response", True, "Enable the PI current-loop and first-order torque response.")
    _add_bool_arg(parser, "--enable-kt-output-model", True, "Enable the measured KT-feedback-to-output torque maps.")
    parser.add_argument("--torque-response-kp", type=float, default=0.0)
    parser.add_argument("--torque-response-ki", type=float, default=90.6769527429)
    parser.add_argument("--torque-response-plant-tau-ms", type=float, default=3.93417593548)
    parser.add_argument("--torque-response-delay-steps", type=float, default=1.0)
    parser.add_argument("--ankle-motor-torque-limit", type=float, default=12.5)
    args = parser.parse_args()
    if args.start_step < 0:
        parser.error("--start-step must be nonnegative")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.physics_hz <= 0.0 or args.policy_hz <= 0.0:
        parser.error("--physics-hz and --policy-hz must be positive")
    if args.ground_friction <= 0.0:
        parser.error("--ground-friction must be positive")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.torque_response_kp < 0.0 or args.torque_response_ki < 0.0:
        parser.error("--torque-response-kp and --torque-response-ki must be nonnegative")
    if args.torque_response_plant_tau_ms <= 0.0:
        parser.error("--torque-response-plant-tau-ms must be positive")
    if args.torque_response_delay_steps < 0.0:
        parser.error("--torque-response-delay-steps must be nonnegative")
    if args.ankle_motor_torque_limit <= 0.0:
        parser.error("--ankle-motor-torque-limit must be positive")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
