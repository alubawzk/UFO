"""PICO 4 -> JOYIn Mini3 -> UFO latent -> pure MuJoCo teleoperation.

This is an independent entry point.  It deliberately imports controller and
model-loading helpers from :mod:`humanoidverse.mujoco_tracking_inference`
without changing that offline tracking program.

Two PICO sources are supported:

* Live Sonic pose messages from ``pico_manager_thread_server.py`` (ZMQ port
  5556 by default).
* Recorded PICO v2 clips containing ``sonic_smpl_*`` arrays, such as
  ``walking_v2.npz`` and ``running_v2.npz``.

JOYIn-Retarget is loaded from ``--joyin-root`` at runtime.  Its
``smplx_to_mini3`` configuration converts each PICO/Sonic SMPL frame to a
Mini3 free-base qpos.  A separate Mini3 MuJoCo model then computes the same
body-space backward observation used during UFO training, and the policy
tracks the resulting latent in the pure MuJoCo controller.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.envs.motion_observations import compute_humanoid_observations_max
from humanoidverse.mujoco_tracking_inference import (
    DEFAULT_ROBOT_CONFIG,
    DebugViewer,
    JointLayout,
    PureMujocoController,
    _add_bool_arg,
    _build_mujoco_model,
    _compose_motion_config,
    _joint_layout,
    _sample_action_delays,
    _saved_robot_training,
)
from humanoidverse.utils.reference_observations import reference_base_ang_vel
from humanoidverse.utils.robot_spec import resolve_robot_config_path
from humanoidverse.utils.torch_utils import quat_rotate_inverse

DEFAULT_JOYIN_ROOT = Path("/home/amax/Desktop/robot/JOYIn-Retarget")
DEFAULT_PICO_ENDPOINT = "tcp://127.0.0.1:5556"
SONIC_HEADER_SIZE = 1280

# Sonic uses the standard SMPL/SMPL-X body-joint order for the first 24
# joints.  The 21 axis-angle entries are the local rotations for joints 1:22.
SMPL_BODY_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
JOYIN_REQUIRED_BODIES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
)


@dataclass(frozen=True)
class PicoSmplFrame:
    """One Sonic SMPL frame in the z-up, root-local convention."""

    smpl_pose: np.ndarray
    smpl_joints: np.ndarray
    root_quat_wxyz: np.ndarray
    root_translation: np.ndarray
    timestamp: float
    sequence_reset: bool = False


@dataclass(frozen=True)
class Mini3Reference:
    """Mini3 reference pose in UFO control-joint order."""

    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    dof_pos: np.ndarray


def _normalize_quaternion_wxyz(quaternion: np.ndarray, *, field: str) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError(f"{field} must contain four finite wxyz values, got {quaternion}")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-8:
        raise ValueError(f"{field} has zero norm")
    return quaternion / norm


def rotation_6d_columns_to_quat_wxyz(rotation_6d: np.ndarray) -> np.ndarray:
    """Invert Sonic's ``matrix[..., :2].reshape(6)`` representation."""

    columns = np.asarray(rotation_6d, dtype=np.float64).reshape(3, 2)
    first = columns[:, 0]
    second = columns[:, 1]
    first_norm = float(np.linalg.norm(first))
    if first_norm <= 1.0e-8:
        raise ValueError("Sonic anchor orientation has a degenerate first column")
    first = first / first_norm
    second = second - float(np.dot(first, second)) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm <= 1.0e-8:
        raise ValueError("Sonic anchor orientation has collinear columns")
    second = second / second_norm
    matrix = np.stack((first, second, np.cross(first, second)), axis=-1)
    return Rotation.from_matrix(matrix).as_quat(scalar_first=True)


def unpack_sonic_pose_message(message: bytes, *, topic: str = "pose") -> dict[str, Any]:
    """Decode the self-describing Sonic binary pose protocol."""

    topic_bytes = topic.encode("utf-8")
    header_start = len(topic_bytes)
    if not message.startswith(topic_bytes):
        raise ValueError(f"Sonic message does not start with topic {topic!r}")
    if len(message) < header_start + SONIC_HEADER_SIZE:
        raise ValueError(f"Sonic message is too short: {len(message)} bytes")
    raw_header = message[header_start : header_start + SONIC_HEADER_SIZE]
    raw_header = raw_header.split(b"\x00", 1)[0]
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid Sonic JSON header") from exc

    dtype_map = {
        "f32": np.dtype("<f4"),
        "f64": np.dtype("<f8"),
        "i32": np.dtype("<i4"),
        "i64": np.dtype("<i8"),
        "u8": np.dtype("u1"),
        "bool": np.dtype("?"),
    }
    result: dict[str, Any] = {
        "version": int(header.get("v", 0)),
        "endian": str(header.get("endian", "le")),
    }
    if result["endian"] != "le":
        raise ValueError(f"Unsupported Sonic endian={result['endian']!r}")
    offset = header_start + SONIC_HEADER_SIZE
    for field in header.get("fields", []):
        name = str(field["name"])
        dtype_name = str(field["dtype"])
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported Sonic dtype={dtype_name!r} for field={name!r}")
        dtype = dtype_map[dtype_name]
        shape = tuple(int(value) for value in field["shape"])
        count = int(np.prod(shape, dtype=np.int64))
        size = count * dtype.itemsize
        if offset + size > len(message):
            raise ValueError(f"Truncated Sonic payload while reading field={name!r}")
        result[name] = np.frombuffer(message, dtype=dtype, count=count, offset=offset).reshape(shape).copy()
        offset += size
    if offset != len(message):
        raise ValueError(f"Sonic message contains {len(message) - offset} trailing payload bytes")
    return result


def _validate_pico_frame(frame: PicoSmplFrame) -> PicoSmplFrame:
    pose = np.asarray(frame.smpl_pose, dtype=np.float64)
    joints = np.asarray(frame.smpl_joints, dtype=np.float64)
    translation = np.asarray(frame.root_translation, dtype=np.float64)
    if pose.shape != (21, 3):
        raise ValueError(f"smpl_pose must have shape (21, 3), got {pose.shape}")
    if joints.shape != (24, 3):
        raise ValueError(f"smpl_joints must have shape (24, 3), got {joints.shape}")
    if translation.shape != (3,):
        raise ValueError(f"root_translation must have shape (3,), got {translation.shape}")
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(joints)) or not np.all(np.isfinite(translation)):
        raise ValueError("PICO/Sonic frame contains non-finite values")
    return PicoSmplFrame(
        smpl_pose=pose,
        smpl_joints=joints,
        root_quat_wxyz=_normalize_quaternion_wxyz(frame.root_quat_wxyz, field="root_quat_wxyz"),
        root_translation=translation,
        timestamp=float(frame.timestamp),
        sequence_reset=bool(frame.sequence_reset),
    )


def sonic_frame_to_joyin_pose(
    frame: PicoSmplFrame,
    *,
    root_frame_alignment_wxyz: np.ndarray | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build JOYIn's SMPL-X global ``position, quaternion_wxyz`` mapping.

    Sonic publishes a z-up root orientation after removing the SMPL base
    rotation.  JOYIn's IK config expects the original SMPL root frame and
    applies its own per-body frame offsets.  ``root_frame_alignment_wxyz``
    restores that frame before global child rotations are accumulated.
    """

    frame = _validate_pico_frame(frame)
    if root_frame_alignment_wxyz is None:
        root_alignment = Rotation.identity()
    else:
        root_alignment = Rotation.from_quat(
            _normalize_quaternion_wxyz(root_frame_alignment_wxyz, field="JOYIn SMPL root alignment"),
            scalar_first=True,
        )
    global_rotations: list[Rotation] = [Rotation.identity() for _ in SMPL_BODY_NAMES]
    global_rotations[0] = Rotation.from_quat(frame.root_quat_wxyz, scalar_first=True) * root_alignment
    for joint_index in range(1, 22):
        parent = SMPL_PARENTS[joint_index]
        local_rotation = Rotation.from_rotvec(frame.smpl_pose[joint_index - 1])
        global_rotations[joint_index] = global_rotations[parent] * local_rotation

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    required = set(JOYIN_REQUIRED_BODIES)
    for joint_index, name in enumerate(SMPL_BODY_NAMES[:22]):
        if name not in required:
            continue
        position = frame.smpl_joints[joint_index] + frame.root_translation
        quaternion = global_rotations[joint_index].as_quat(scalar_first=True)
        result[name] = (position.copy(), quaternion.copy())
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"Cannot build JOYIn pose; missing SMPL-X bodies: {missing}")
    return result


class PicoNpzSource:
    """Read the Sonic fields saved in a PICO v2 motion clip."""

    is_live = False

    def __init__(self, path: Path, *, start_frame: int, loop: bool, root_motion: bool):
        self.path = path.expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"PICO NPZ does not exist: {self.path}")
        with np.load(self.path, allow_pickle=False) as data:
            required = {"sonic_smpl_pose", "sonic_smpl_joints", "sonic_smpl_anchor_orientation", "fps"}
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"PICO NPZ={self.path} is missing Sonic fields: {missing}")
            self.smpl_pose = np.asarray(data["sonic_smpl_pose"], dtype=np.float64)
            self.smpl_joints = np.asarray(data["sonic_smpl_joints"], dtype=np.float64)
            self.anchor_orientation = np.asarray(data["sonic_smpl_anchor_orientation"], dtype=np.float64)
            self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            self.timestamps = (
                np.asarray(data["timestamp_monotonic"], dtype=np.float64)
                if "timestamp_monotonic" in data
                else np.arange(self.smpl_pose.shape[0], dtype=np.float64) / self.fps
            )
            if root_motion and "body_pos_w" in data:
                sparse_root = np.asarray(data["body_pos_w"], dtype=np.float64)[:, 0]
                self.root_translation = sparse_root - sparse_root[0]
            else:
                self.root_translation = np.zeros((self.smpl_pose.shape[0], 3), dtype=np.float64)

        frame_count = self.smpl_pose.shape[0]
        if self.smpl_pose.shape != (frame_count, 21, 3):
            raise ValueError(f"sonic_smpl_pose must have shape (T, 21, 3), got {self.smpl_pose.shape}")
        if self.smpl_joints.shape != (frame_count, 24, 3):
            raise ValueError(f"sonic_smpl_joints must have shape (T, 24, 3), got {self.smpl_joints.shape}")
        if self.anchor_orientation.shape != (frame_count, 6):
            raise ValueError(f"sonic_smpl_anchor_orientation must have shape (T, 6), got {self.anchor_orientation.shape}")
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError(f"PICO NPZ fps must be positive, got {self.fps}")
        if start_frame < 0 or start_frame >= frame_count:
            raise IndexError(f"start_frame={start_frame} is outside [0, {frame_count - 1}]")
        self.frame_count = frame_count
        self.start_frame = int(start_frame)
        self.index = int(start_frame)
        self.loop = bool(loop)
        self.finished = False
        self._first = True

    def receive(self, _timeout_ms: int = 0) -> PicoSmplFrame | None:
        if self.finished:
            return None
        index = self.index
        reset = self._first or (index == self.start_frame and not self._first)
        frame = PicoSmplFrame(
            smpl_pose=self.smpl_pose[index],
            smpl_joints=self.smpl_joints[index],
            root_quat_wxyz=rotation_6d_columns_to_quat_wxyz(self.anchor_orientation[index]),
            root_translation=self.root_translation[index],
            timestamp=float(self.timestamps[index]),
            sequence_reset=reset,
        )
        self._first = False
        self.index += 1
        if self.index >= self.frame_count:
            if self.loop:
                self.index = self.start_frame
            else:
                self.finished = True
        return _validate_pico_frame(frame)

    def close(self) -> None:
        return None


class SonicZmqSource:
    """Subscribe to the live Sonic PICO ``pose`` topic."""

    is_live = True
    finished = False

    def __init__(self, endpoint: str):
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("Live PICO input requires pyzmq; install the project's pico-teleop extra") from exc
        self._zmq = zmq
        self.endpoint = str(endpoint)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"pose")
        self.socket.setsockopt(zmq.RCVHWM, 5)
        self.socket.connect(self.endpoint)
        self._first = True
        self.last_received_monotonic: float | None = None

    @staticmethod
    def _latest(array: np.ndarray, expected_tail: tuple[int, ...], field: str) -> np.ndarray:
        value = np.asarray(array)
        if value.shape == expected_tail:
            return value
        if value.ndim == len(expected_tail) + 1 and value.shape[1:] == expected_tail and value.shape[0] > 0:
            return value[-1]
        raise ValueError(f"Live Sonic field={field!r} must end in shape={expected_tail}, got {value.shape}")

    def receive(self, timeout_ms: int = 0) -> PicoSmplFrame | None:
        if self.socket.poll(timeout=max(0, int(timeout_ms))) == 0:
            return None
        message = self.socket.recv()
        while self.socket.poll(timeout=0):
            message = self.socket.recv()
        payload = unpack_sonic_pose_message(message)
        for field in ("smpl_pose", "smpl_joints", "body_quat_w"):
            if field not in payload:
                raise ValueError(f"Live Sonic pose message is missing field={field!r}")
        timestamp_array = payload.get("timestamp_monotonic")
        timestamp = (
            float(np.asarray(timestamp_array).reshape(-1)[-1])
            if timestamp_array is not None and np.asarray(timestamp_array).size
            else time.monotonic()
        )
        frame = PicoSmplFrame(
            smpl_pose=self._latest(payload["smpl_pose"], (21, 3), "smpl_pose"),
            smpl_joints=self._latest(payload["smpl_joints"], (24, 3), "smpl_joints"),
            root_quat_wxyz=self._latest(payload["body_quat_w"], (4,), "body_quat_w"),
            root_translation=np.zeros(3, dtype=np.float64),
            timestamp=timestamp,
            sequence_reset=self._first,
        )
        self._first = False
        self.last_received_monotonic = time.monotonic()
        return _validate_pico_frame(frame)

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


def _model_joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]


class JoyInMini3Retargeter:
    """Thin runtime adapter around the local JOYIn GMR implementation."""

    def __init__(
        self,
        joyin_root: Path,
        control_joint_names: list[str],
        *,
        actual_human_height: float | None,
        solver: str,
        damping: float,
        ik_dt: float,
        max_iter: int,
        posture_cost: float,
        velocity_limit: bool,
        offset_to_ground: bool,
        verbose: bool,
    ):
        self.joyin_root = joyin_root.expanduser().resolve()
        package = self.joyin_root / "general_motion_retargeting"
        if not package.is_dir():
            raise FileNotFoundError(f"JOYIn package not found under: {self.joyin_root}")
        joyin_path = str(self.joyin_root)
        if joyin_path not in sys.path:
            sys.path.insert(0, joyin_path)
        try:
            from general_motion_retargeting import GeneralMotionRetargeting
        except ImportError as exc:
            raise ImportError(
                "Cannot import local JOYIn-Retarget. Install the pico-teleop extra so mink and qpsolvers[daqp] are available"
            ) from exc

        self.retargeter = GeneralMotionRetargeting(
            src_human="smplx",
            tgt_robot="mini3",
            actual_human_height=actual_human_height,
            solver=solver,
            damping=float(damping),
            verbose=bool(verbose),
            use_velocity_limit=bool(velocity_limit),
            posture_cost=float(posture_cost),
            ik_dt=float(ik_dt),
            max_iter=int(max_iter),
        )
        joyin_names = _model_joint_names(self.retargeter.model)
        missing = sorted(set(control_joint_names).difference(joyin_names))
        if missing:
            raise ValueError(f"JOYIn Mini3 model is missing UFO control joints: {missing}")
        self.qpos_addresses = np.asarray(
            [
                int(self.retargeter.model.jnt_qposadr[mujoco.mj_name2id(self.retargeter.model, mujoco.mjtObj.mjOBJ_JOINT, name)])
                for name in control_joint_names
            ],
            dtype=np.int32,
        )
        if "pelvis" not in self.retargeter.rot_offsets1:
            raise ValueError("JOYIn smplx_to_mini3 config does not define the pelvis rotation offset")
        self.smpl_root_alignment_wxyz = self.retargeter.rot_offsets1["pelvis"].inv().as_quat(scalar_first=True)
        self.offset_to_ground = bool(offset_to_ground)

    def retarget(self, frame: PicoSmplFrame) -> Mini3Reference:
        human_pose = sonic_frame_to_joyin_pose(
            frame,
            root_frame_alignment_wxyz=self.smpl_root_alignment_wxyz,
        )
        qpos = np.asarray(
            self.retargeter.retarget(human_pose, offset_to_ground=self.offset_to_ground),
            dtype=np.float64,
        )
        if qpos.shape != (self.retargeter.model.nq,) or not np.all(np.isfinite(qpos)):
            raise ValueError(f"JOYIn returned invalid Mini3 qpos with shape={qpos.shape}")
        return Mini3Reference(
            root_pos=qpos[:3].copy(),
            root_quat_wxyz=_normalize_quaternion_wxyz(qpos[3:7], field="JOYIn root quaternion"),
            dof_pos=qpos[self.qpos_addresses].copy(),
        )


class OnlineMini3ReferenceEncoder:
    """Convert one Mini3 qpos at a time into the UFO backward observation."""

    def __init__(
        self,
        xml_path: Path,
        control_joint_names: list[str],
        body_names: list[str],
        default_joint_angles: dict[str, float],
        hv_config: Any,
        *,
        device: str,
        dt: float,
        root_height_obs: bool,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.layout = _joint_layout(self.model, control_joint_names)
        self.body_names = tuple(str(name) for name in body_names)
        self.body_ids = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.body_names],
            dtype=np.int32,
        )
        if np.any(self.body_ids < 0):
            missing = [name for name, body_id in zip(self.body_names, self.body_ids) if body_id < 0]
            raise ValueError(f"UFO Mini3 model is missing checkpoint bodies: {missing}")
        model_body_names = tuple(
            str(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)) for body_id in range(1, self.model.nbody)
        )
        if self.body_names != model_body_names:
            raise ValueError(
                "Checkpoint body order differs from the local Mini3 XML; backward observation would be misaligned: "
                f"checkpoint={self.body_names}, local={model_body_names}"
            )
        self.default_dof_pos = np.asarray([float(default_joint_angles[name]) for name in control_joint_names], dtype=np.float64)
        self.hv_config = hv_config
        self.device = str(device)
        self.dt = float(dt)
        self.root_height_obs = bool(root_height_obs)
        self.previous_qpos: np.ndarray | None = None
        self.previous_body_pos: np.ndarray | None = None
        self.previous_body_quat_xyzw: np.ndarray | None = None
        self.helper_env = SimpleNamespace(config=hv_config)
        self.gravity = torch.tensor([[0.0, 0.0, -1.0]], device=self.device, dtype=torch.float32)
        if bool(hv_config.get("use_contact_in_obs_max", False)):
            raise NotImplementedError("Online PICO inference does not support use_contact_in_obs_max=True")

    def reset(self) -> None:
        self.previous_qpos = None
        self.previous_body_pos = None
        self.previous_body_quat_xyzw = None

    def _qpos(self, reference: Mini3Reference) -> np.ndarray:
        qpos = np.zeros(self.model.nq, dtype=np.float64)
        qpos[:3] = reference.root_pos
        qpos[3:7] = reference.root_quat_wxyz
        if reference.dof_pos.shape != (len(self.layout.names),):
            raise ValueError(f"Mini3 reference has {reference.dof_pos.size} joints, expected {len(self.layout.names)}")
        qpos[self.layout.qpos_adr] = reference.dof_pos
        return qpos

    def backward_observation(
        self,
        reference: Mini3Reference,
        *,
        sample_dt: float | None = None,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        qpos = self._qpos(reference)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        body_pos = self.data.xpos[self.body_ids].copy()
        body_quat_wxyz = self.data.xquat[self.body_ids].copy()
        body_quat_xyzw = body_quat_wxyz[:, [1, 2, 3, 0]]

        if self.previous_qpos is None:
            dof_vel = np.zeros(len(self.layout.names), dtype=np.float64)
            body_vel = np.zeros_like(body_pos)
            body_ang_vel = np.zeros_like(body_pos)
        else:
            velocity_dt = self.dt if sample_dt is None else float(sample_dt)
            if not math.isfinite(velocity_dt) or velocity_dt <= 0.0:
                raise ValueError(f"Reference sample_dt must be finite and positive, got {velocity_dt}")
            qvel = np.zeros(self.model.nv, dtype=np.float64)
            mujoco.mj_differentiatePos(self.model, qvel, velocity_dt, self.previous_qpos, qpos)
            dof_vel = qvel[self.layout.dof_adr]
            body_vel = (body_pos - self.previous_body_pos) / velocity_dt
            previous_rot = Rotation.from_quat(self.previous_body_quat_xyzw)
            current_rot = Rotation.from_quat(body_quat_xyzw)
            body_ang_vel = (current_rot * previous_rot.inv()).as_rotvec() / velocity_dt

        self.previous_qpos = qpos.copy()
        self.previous_body_pos = body_pos.copy()
        self.previous_body_quat_xyzw = body_quat_xyzw.copy()

        ref_body_pos = torch.as_tensor(body_pos, device=self.device, dtype=torch.float32).unsqueeze(0)
        ref_body_rots = torch.as_tensor(body_quat_xyzw, device=self.device, dtype=torch.float32).unsqueeze(0)
        ref_body_vels = torch.as_tensor(body_vel, device=self.device, dtype=torch.float32).unsqueeze(0)
        ref_body_ang_vels = torch.as_tensor(body_ang_vel, device=self.device, dtype=torch.float32).unsqueeze(0)
        obs_dict = compute_humanoid_observations_max(
            ref_body_pos,
            ref_body_rots,
            ref_body_vels,
            ref_body_ang_vels,
            local_root_obs=True,
            root_height_obs=self.root_height_obs,
        )
        privileged_state = torch.cat(tuple(obs_dict.values()), dim=-1)
        if not bool(self.hv_config.obs.use_obs_filter):
            return privileged_state

        dof_pos = torch.as_tensor(reference.dof_pos, device=self.device, dtype=torch.float32).unsqueeze(0)
        default_dof_pos = torch.as_tensor(self.default_dof_pos, device=self.device, dtype=torch.float32).unsqueeze(0)
        ref_dof_pos = dof_pos - default_dof_pos
        ref_dof_vel = torch.as_tensor(dof_vel, device=self.device, dtype=torch.float32).unsqueeze(0)
        base_quat = ref_body_rots[:, 0]
        ref_ang_vel = reference_base_ang_vel(self.helper_env, base_quat, ref_body_ang_vels[:, 0])
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity, w_last=True)
        state = torch.cat((ref_dof_pos, ref_dof_vel, projected_gravity, ref_ang_vel), dim=-1)
        return {
            "state": state,
            "last_action": ref_dof_pos,
            "privileged_state": privileged_state,
        }


class LatentSmoother:
    def __init__(self, window: int, gamma: float, *, renormalize: bool):
        self.values: deque[torch.Tensor] = deque(maxlen=int(window))
        self.gamma = float(gamma)
        self.renormalize = bool(renormalize)

    def reset(self) -> None:
        self.values.clear()

    def update(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.detach()
        self.values.append(latent)
        newest_first = list(reversed(self.values))
        weights = torch.as_tensor(
            [self.gamma**index for index in range(len(newest_first))],
            device=latent.device,
            dtype=latent.dtype,
        )
        stacked = torch.stack(newest_first, dim=0)
        smoothed = (stacked * weights[:, None, None]).sum(dim=0) / weights.sum()
        if self.renormalize:
            norm = torch.linalg.norm(smoothed, dim=-1, keepdim=True).clamp_min(1.0e-8)
            smoothed = math.sqrt(smoothed.shape[-1]) * smoothed / norm
        return smoothed


def _initialize_from_live_reference(
    data: mujoco.MjData,
    layout: JointLayout,
    reference: Mini3Reference,
    *,
    root_z_offset: float,
) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:3] = reference.root_pos
    data.qpos[2] += float(root_z_offset)
    data.qpos[3:7] = reference.root_quat_wxyz
    data.qpos[layout.qpos_adr] = reference.dof_pos
    data.ctrl[:] = 0.0


def _load_policy(model_folder: Path, device: str) -> torch.nn.Module:
    checkpoint_dir = model_folder / "checkpoint"
    model_file = checkpoint_dir / "model" / "model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"Missing checkpoint model: {model_file}")
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
        load_device = "cuda"
    elif torch_device.type == "cpu":
        load_device = "cpu"
    else:
        raise ValueError(f"Unsupported device: {device}")
    return load_model_from_checkpoint_dir(checkpoint_dir, device=load_device).to(device).eval()


def run(args: argparse.Namespace) -> None:
    model_folder = args.model_folder.expanduser().resolve()
    run_config_path = model_folder / "config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run config: {run_config_path}")
    run_config = json.loads(run_config_path.read_text())
    robot_config = resolve_robot_config_path(args.robot_config)
    robot_training = _saved_robot_training(run_config, robot_config)
    xml_path = Path(robot_training["robot"]["xml_path"])

    placeholder_data = args.pico_npz if args.pico_npz is not None else args.joyin_root / "online_pico_placeholder.pkl"
    hv_config = _compose_motion_config(run_config, data_path=placeholder_data, robot_training=robot_training)
    physics_hz = float(hv_config.simulator.config.sim.fps)
    decimation = int(hv_config.simulator.config.sim.control_decimation)
    policy_hz = physics_hz / decimation
    policy_dt = 1.0 / policy_hz
    if not math.isclose(physics_hz, args.physics_hz, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(f"Checkpoint physics_hz={physics_hz:g}, but --physics-hz={args.physics_hz:g}")
    if not math.isclose(policy_hz, args.policy_hz, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(f"Checkpoint policy_hz={policy_hz:g}, but --policy-hz={args.policy_hz:g}")

    source: PicoNpzSource | SonicZmqSource
    if args.pico_npz is not None:
        source = PicoNpzSource(
            args.pico_npz,
            start_frame=args.start_frame,
            loop=args.loop,
            root_motion=args.enable_recorded_root_motion,
        )
        if not math.isclose(source.fps, policy_hz, rel_tol=0.0, abs_tol=1.0e-3):
            raise ValueError(f"PICO clip fps={source.fps:g} must match checkpoint policy_hz={policy_hz:g}")
    else:
        source = SonicZmqSource(args.pico_endpoint)

    joint_names = [str(name) for name in robot_training["robot"]["control_joint_names"]]
    retargeter = JoyInMini3Retargeter(
        args.joyin_root,
        joint_names,
        actual_human_height=args.actual_human_height,
        solver=args.joyin_solver,
        damping=args.joyin_damping,
        ik_dt=policy_dt,
        max_iter=args.joyin_max_iter,
        posture_cost=args.joyin_posture_cost,
        velocity_limit=args.joyin_velocity_limit,
        offset_to_ground=args.joyin_offset_to_ground,
        verbose=args.joyin_verbose,
    )
    policy = _load_policy(model_folder, args.device)
    body_names = [str(name) for name in robot_training["robot"]["body_names"]]
    reference_encoder = OnlineMini3ReferenceEncoder(
        xml_path,
        joint_names,
        body_names,
        robot_training["default_joint_angles"],
        hv_config,
        device=args.device,
        dt=policy_dt,
        root_height_obs=bool(run_config["env"].get("root_height_obs", False)),
    )
    norm_z = bool(getattr(getattr(policy.cfg, "archi", None), "norm_z", False))
    latent_smoother = LatentSmoother(args.latent_window, args.latent_gamma, renormalize=norm_z)

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
    source_name = args.pico_endpoint if source.is_live else str(source.path)
    print(
        "[pico-teleop] "
        f"source={source_name} joyin={retargeter.joyin_root} physics={physics_hz:g}Hz "
        f"policy={policy_hz:g}Hz latent_smoothing={args.latent_window}@{args.latent_gamma:g}",
        flush=True,
    )
    if source.is_live:
        print(f"[pico-teleop] waiting for first Sonic pose on {args.pico_endpoint} ...", flush=True)

    first_frame = source.receive(args.connect_timeout_ms)
    if first_frame is None:
        source.close()
        if viewer is not None:
            viewer.close()
        raise TimeoutError(f"No PICO pose received within {args.connect_timeout_ms} ms")
    first_reference = retargeter.retarget(first_frame)
    first_backward_obs = reference_encoder.backward_observation(first_reference)
    with torch.no_grad():
        current_z = latent_smoother.update(policy.project_z(policy.backward_map(first_backward_obs)))

    latest_reference = first_reference
    previous_frame_timestamp = float(first_frame.timestamp)

    def reset_simulation() -> dict[str, torch.Tensor]:
        mujoco.mj_resetData(model, data)
        _initialize_from_live_reference(
            data,
            layout,
            latest_reference,
            root_z_offset=args.root_z_offset,
        )
        mujoco.mj_forward(model, data)
        controller.reset()
        return controller.observation(args.device)

    observation = reset_simulation()
    step = 0
    last_live_frame_time = time.monotonic()
    retarget_ms = 0.0
    try:
        while True:
            if viewer is not None:
                if not viewer.wait_if_paused():
                    break
                if viewer.consume_reset():
                    observation = reset_simulation()
            cycle_started = time.perf_counter()

            frame = source.receive(0)
            if frame is not None:
                if frame.sequence_reset:
                    reference_encoder.reset()
                    latent_smoother.reset()
                    sample_dt = None
                elif not source.is_live:
                    sample_dt = policy_dt
                else:
                    measured_dt = float(frame.timestamp) - previous_frame_timestamp
                    sample_dt = measured_dt if 0.25 * policy_dt <= measured_dt <= 5.0 * policy_dt else policy_dt
                retarget_started = time.perf_counter()
                latest_reference = retargeter.retarget(frame)
                backward_obs = reference_encoder.backward_observation(latest_reference, sample_dt=sample_dt)
                with torch.no_grad():
                    current_z = latent_smoother.update(policy.project_z(policy.backward_map(backward_obs)))
                if frame.sequence_reset and not source.is_live:
                    observation = reset_simulation()
                retarget_ms = (time.perf_counter() - retarget_started) * 1000.0
                last_live_frame_time = time.monotonic()
                previous_frame_timestamp = float(frame.timestamp)
            elif source.finished:
                break
            elif source.is_live and time.monotonic() - last_live_frame_time > args.max_pico_stale_seconds:
                raise TimeoutError(f"PICO stream has been stale for more than {args.max_pico_stale_seconds:g}s; stopping simulation")

            with torch.no_grad():
                action = policy.act(observation, current_z, mean=True)[0].detach().cpu().numpy()
            controller.set_policy_action(action)
            for _ in range(decimation):
                controller.physics_step()
            observation = controller.observation(args.device)
            step += 1

            if viewer is not None:
                viewer.sync(step=step, torque=controller.torque)
            if step == 1 or step % args.log_every == 0:
                print(
                    f"[pico-teleop] step={step} root_z={data.qpos[2]:.4f} ncon={data.ncon} "
                    f"z_norm={torch.linalg.norm(current_z).item():.3f} retarget={retarget_ms:.2f}ms "
                    f"max_torque={np.max(np.abs(controller.torque)):.3f}",
                    flush=True,
                )
            if args.max_steps is not None and step >= args.max_steps:
                break
            if args.realtime:
                remaining = policy_dt - (time.perf_counter() - cycle_started)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        source.close()
        if viewer is not None:
            viewer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PICO 4 -> JOYIn Mini3 -> UFO pure MuJoCo teleoperation.")
    parser.add_argument("--model-folder", type=Path, required=True, help="UFO run directory containing config.json and checkpoint/.")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--pico-npz", type=Path, default=None, help="Recorded PICO v2 NPZ; omit for live ZMQ input.")
    source_group.add_argument("--pico-endpoint", default=DEFAULT_PICO_ENDPOINT, help="Live Sonic pose PUB endpoint.")
    parser.add_argument("--joyin-root", type=Path, default=DEFAULT_JOYIN_ROOT)
    parser.add_argument("--robot-config", type=Path, default=Path(DEFAULT_ROBOT_CONFIG))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--connect-timeout-ms", type=int, default=10000)
    parser.add_argument("--max-pico-stale-seconds", type=float, default=0.5)
    parser.add_argument("--physics-hz", type=float, default=500.0)
    parser.add_argument("--policy-hz", type=float, default=50.0)
    parser.add_argument("--ground-friction", type=float, default=1.0)
    parser.add_argument("--root-z-offset", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--actual-human-height", type=float, default=None)
    parser.add_argument("--joyin-solver", default="daqp")
    parser.add_argument("--joyin-damping", type=float, default=0.5)
    parser.add_argument("--joyin-max-iter", type=int, default=10)
    parser.add_argument("--joyin-posture-cost", type=float, default=0.0)
    parser.add_argument("--latent-window", type=int, default=3)
    parser.add_argument("--latent-gamma", type=float, default=0.8)
    _add_bool_arg(parser, "--headless", False, "Run without the interactive MuJoCo viewer.")
    _add_bool_arg(parser, "--loop", False, "Loop a recorded PICO NPZ.")
    _add_bool_arg(parser, "--realtime", True, "Rate-limit rollout to policy_hz.")
    _add_bool_arg(parser, "--enable-recorded-root-motion", True, "Restore root displacement from recorded body_pos_w.")
    _add_bool_arg(parser, "--joyin-offset-to-ground", True, "Ground each human frame before JOYIn IK.")
    _add_bool_arg(parser, "--joyin-velocity-limit", True, "Enable JOYIn IK velocity limits.")
    _add_bool_arg(parser, "--joyin-verbose", False, "Print JOYIn model details.")
    _add_bool_arg(parser, "--disable-action-delay", False, "Disable saved actuator physics-step delays.")
    _add_bool_arg(parser, "--disable-imu-delay", False, "Disable saved IMU delay.")
    _add_bool_arg(parser, "--enable-real-motor", True, "Use the Mini3 real-motor response chain.")
    _add_bool_arg(parser, "--enable-tn-torque-limit", True, "Enable calibrated Mini3 T-N envelopes.")
    _add_bool_arg(parser, "--tn-limit-after-response", True, "Reapply T-N limits after current-loop response.")
    _add_bool_arg(parser, "--enable-torque-response", True, "Enable PI current-loop and torque response.")
    _add_bool_arg(parser, "--enable-kt-output-model", True, "Enable measured KT output-torque maps.")
    parser.add_argument("--torque-response-kp", type=float, default=0.0)
    parser.add_argument("--torque-response-ki", type=float, default=90.6769527429)
    parser.add_argument("--torque-response-plant-tau-ms", type=float, default=3.93417593548)
    parser.add_argument("--torque-response-delay-steps", type=float, default=1.0)
    parser.add_argument("--ankle-motor-torque-limit", type=float, default=12.5)
    args = parser.parse_args()
    if args.start_frame < 0:
        parser.error("--start-frame must be nonnegative")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.connect_timeout_ms <= 0:
        parser.error("--connect-timeout-ms must be positive")
    if args.max_pico_stale_seconds <= 0.0:
        parser.error("--max-pico-stale-seconds must be positive")
    if args.physics_hz <= 0.0 or args.policy_hz <= 0.0:
        parser.error("--physics-hz and --policy-hz must be positive")
    if args.ground_friction <= 0.0:
        parser.error("--ground-friction must be positive")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.actual_human_height is not None and args.actual_human_height <= 0.0:
        parser.error("--actual-human-height must be positive")
    if args.joyin_damping <= 0.0 or args.joyin_max_iter <= 0 or args.joyin_posture_cost < 0.0:
        parser.error("JOYIn damping/max-iter must be positive and posture-cost nonnegative")
    if args.latent_window <= 0 or not 0.0 < args.latent_gamma <= 1.0:
        parser.error("--latent-window must be positive and --latent-gamma must be in (0, 1]")
    if args.torque_response_kp < 0.0 or args.torque_response_ki < 0.0:
        parser.error("--torque-response-kp and --torque-response-ki must be nonnegative")
    if args.torque_response_plant_tau_ms <= 0.0 or args.torque_response_delay_steps < 0.0:
        parser.error("Torque response tau must be positive and delay nonnegative")
    if args.ankle_motor_torque_limit <= 0.0:
        parser.error("--ankle-motor-torque-limit must be positive")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
