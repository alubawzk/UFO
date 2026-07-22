"""Retarget sparse PICO motion clips into UFO MotionLib PKLs.

PICO clips contain world poses for the pelvis, both feet, and both hands but
do not contain robot joint positions.  This tool uses damped least-squares
MuJoCo IK to infer the controlled joints of the target robot, then delegates
to the standard RobotState converter to build ``pose_aa`` and the remaining
MotionLib fields.

The retarget is necessarily approximate.  On Mini3, which has no wrist links,
the two wrist targets are mapped to points near the tips of the elbow links.
The generated metadata records that mapping and the measured endpoint errors.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mujoco
import numpy as np

from humanoidverse.utils.motion_data.robot_state import RobotStateMotion
from humanoidverse.utils.motion_data.robot_state_convert import robot_state_to_ufo_motion
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict
from humanoidverse.utils.robot_spec import RobotSpec, load_robot_spec, resolve_robot_config_path

PICO_BODY_NAMES = (
    "pelvis",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


@dataclass(frozen=True)
class PicoMotionClip:
    path: Path
    body_pos_w: np.ndarray
    body_quat_wxyz: np.ndarray
    fps: float
    position_axes_version: int
    body_state_frame: str


@dataclass(frozen=True)
class TrackingTarget:
    source_name: str
    target_body_name: str
    target_body_id: int
    local_point: np.ndarray
    position_weight: float
    orientation_scale: float


@dataclass(frozen=True)
class RetargetResult:
    motion: dict[str, Any]
    scale: float
    mean_position_error_m: dict[str, float]
    max_position_error_m: dict[str, float]
    mean_orientation_error_rad: dict[str, float]
    max_orientation_error_rad: dict[str, float]
    mean_iterations: float
    max_iterations: int


def discover_pico_clips(input_path: str | Path) -> list[Path]:
    source = Path(input_path).expanduser().resolve()
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(source.glob("*.npz"))
    else:
        raise FileNotFoundError(f"PICO input does not exist: {source}")
    if not paths:
        raise FileNotFoundError(f"No .npz PICO clips found under: {source}")
    return paths


def _scalar(data: Any, key: str, path: Path) -> Any:
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"PICO clip={path} field={key} must be scalar, got shape={value.shape}")
    return value.reshape(-1)[0]


def load_pico_clip(path: str | Path) -> PicoMotionClip:
    clip_path = Path(path).expanduser().resolve()
    required = {
        "body_pos_w",
        "body_quat_w",
        "body_names",
        "fps",
        "dt",
        "source",
        "pico_position_axes_version",
        "body_state_frame",
    }
    with np.load(clip_path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"PICO clip={clip_path} is missing fields: {missing}")

        source = str(_scalar(data, "source", clip_path))
        if source != "pico_motion_clip":
            raise ValueError(f"PICO clip={clip_path} has source={source!r}, expected 'pico_motion_clip'")
        axes_version = int(_scalar(data, "pico_position_axes_version", clip_path))
        if axes_version != 3:
            raise ValueError(f"PICO clip={clip_path} has unsupported pico_position_axes_version={axes_version}; expected 3")

        names = tuple(str(name) for name in np.asarray(data["body_names"]).tolist())
        if names != PICO_BODY_NAMES:
            raise ValueError(f"PICO clip={clip_path} body_names must be {list(PICO_BODY_NAMES)}, got {list(names)}")

        positions = np.asarray(data["body_pos_w"], dtype=np.float64)
        quaternions = np.asarray(data["body_quat_w"], dtype=np.float64)
        if positions.ndim != 3 or positions.shape[1:] != (len(PICO_BODY_NAMES), 3):
            raise ValueError(f"PICO clip={clip_path} body_pos_w must have shape [T, 5, 3], got {positions.shape}")
        if quaternions.shape != (positions.shape[0], len(PICO_BODY_NAMES), 4):
            raise ValueError(f"PICO clip={clip_path} body_quat_w must have shape [{positions.shape[0]}, 5, 4], got {quaternions.shape}")
        if positions.shape[0] < 2:
            raise ValueError(f"PICO clip={clip_path} must contain at least two frames")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(quaternions)):
            raise ValueError(f"PICO clip={clip_path} contains non-finite body poses")

        quat_norm = np.linalg.norm(quaternions, axis=-1, keepdims=True)
        if np.any(quat_norm <= 1.0e-8):
            raise ValueError(f"PICO clip={clip_path} contains zero-length body quaternions")
        quaternions = quaternions / quat_norm

        fps = float(_scalar(data, "fps", clip_path))
        dt = float(_scalar(data, "dt", clip_path))
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"PICO clip={clip_path} fps must be finite and positive, got {fps}")
        if not np.isfinite(dt) or dt <= 0.0 or not np.isclose(dt, 1.0 / fps, rtol=1.0e-5, atol=1.0e-8):
            raise ValueError(f"PICO clip={clip_path} dt={dt} does not match 1/fps={1.0 / fps}")

        return PicoMotionClip(
            path=clip_path,
            body_pos_w=positions,
            body_quat_wxyz=quaternions,
            fps=fps,
            position_axes_version=axes_version,
            body_state_frame=str(_scalar(data, "body_state_frame", clip_path)),
        )


def _object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Target robot model is missing {object_type.name.removeprefix('mjOBJ_').lower()} {name!r}")
    return int(object_id)


def _mini3_hand_point(model: mujoco.MjModel, side: str) -> tuple[str, np.ndarray]:
    body_name = f"{side}_elbow_pitch_link"
    geom_name = f"{side}_elbow_pitch_link_collision"
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if body_id < 0 or geom_id < 0 or int(model.geom_bodyid[geom_id]) != int(body_id):
        raise ValueError(f"Target robot has no {side}_wrist_yaw_link and no usable Mini3 fallback point {body_name}/{geom_name}")
    return body_name, np.asarray(model.geom_pos[geom_id], dtype=np.float64).copy()


def _tracking_targets(model: mujoco.MjModel, robot_spec: RobotSpec) -> list[TrackingTarget]:
    targets: list[TrackingTarget] = []
    for source_name in PICO_BODY_NAMES[1:3]:
        body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, source_name)
        targets.append(
            TrackingTarget(
                source_name=source_name,
                target_body_name=source_name,
                target_body_id=body_id,
                local_point=np.zeros(3, dtype=np.float64),
                position_weight=2.0,
                orientation_scale=1.0,
            )
        )

    for side, source_name in (("left", PICO_BODY_NAMES[3]), ("right", PICO_BODY_NAMES[4])):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, source_name)
        if body_id >= 0:
            target_body_name = source_name
            local_point = np.zeros(3, dtype=np.float64)
            orientation_scale = 1.0
        else:
            target_body_name, local_point = _mini3_hand_point(model, side)
            body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
            # A G1 wrist frame has no orientation-equivalent Mini3 link.  Its
            # position still drives the virtual hand point, but constraining
            # the elbow-link orientation would substantially degrade reach.
            orientation_scale = 0.0
        targets.append(
            TrackingTarget(
                source_name=source_name,
                target_body_name=target_body_name,
                target_body_id=body_id,
                local_point=local_point,
                position_weight=1.0,
                orientation_scale=orientation_scale,
            )
        )

    _object_id(model, mujoco.mjtObj.mjOBJ_BODY, robot_spec.base_body)
    return targets


def _quat_conjugate_wxyz(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float64).copy()
    result[1:] *= -1.0
    return result


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _quat_error_vector_wxyz(desired: np.ndarray, actual: np.ndarray) -> np.ndarray:
    delta = _quat_multiply_wxyz(desired, _quat_conjugate_wxyz(actual))
    if delta[0] < 0.0:
        delta *= -1.0
    vector_norm = float(np.linalg.norm(delta[1:]))
    if vector_norm <= 1.0e-10:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vector_norm, float(np.clip(delta[0], -1.0, 1.0)))
    return delta[1:] * (angle / vector_norm)


def _world_point(data: mujoco.MjData, target: TrackingTarget) -> np.ndarray:
    rotation = data.xmat[target.target_body_id].reshape(3, 3)
    return data.xpos[target.target_body_id] + rotation @ target.local_point


def _clip_control_joints(model: mujoco.MjModel, qpos: np.ndarray, joint_ids: np.ndarray, qpos_addresses: np.ndarray) -> None:
    for joint_id, qpos_address in zip(joint_ids, qpos_addresses):
        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)) or ""
            # Both Mini3 and G1 XMLs allow a small negative knee angle.  Sparse
            # position IK can enter that hyperextended branch and then cannot
            # leave it during a deep crouch.  Human/PICO motion uses the
            # positive-flexion branch, so retain a tiny bend and a non-zero
            # vertical Jacobian throughout the retarget.
            if "knee" in joint_name:
                lower = max(float(lower), 0.02)
            qpos[qpos_address] = np.clip(qpos[qpos_address], lower, upper)


def _weighted_residual(
    data: mujoco.MjData,
    targets: list[TrackingTarget],
    desired_positions: np.ndarray,
    desired_orientations: np.ndarray,
    orientation_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual: list[np.ndarray] = []
    position_errors: list[float] = []
    orientation_errors: list[float] = []
    for target_idx, target in enumerate(targets):
        position_error = desired_positions[target_idx] - _world_point(data, target)
        orientation_error = _quat_error_vector_wxyz(desired_orientations[target_idx], data.xquat[target.target_body_id])
        residual.append(np.sqrt(target.position_weight) * position_error)
        resolved_orientation_weight = orientation_weight * target.orientation_scale
        if resolved_orientation_weight > 0.0:
            residual.append(np.sqrt(resolved_orientation_weight) * orientation_error)
        position_errors.append(float(np.linalg.norm(position_error)))
        orientation_errors.append(float(np.linalg.norm(orientation_error)))
    return np.concatenate(residual), np.asarray(position_errors), np.asarray(orientation_errors)


def _solve_ik_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    targets: list[TrackingTarget],
    desired_positions: np.ndarray,
    desired_orientations: np.ndarray,
    joint_ids: np.ndarray,
    qpos_addresses: np.ndarray,
    dof_addresses: np.ndarray,
    reference_dof_pos: np.ndarray,
    *,
    max_iterations: int,
    position_tolerance: float,
    orientation_weight: float,
    damping: float,
    posture_weight: float,
    max_joint_step: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    identity = np.eye(len(qpos_addresses), dtype=np.float64)
    previous_cost = float("inf")
    stagnation_steps = 0
    iterations_used = 0

    for iteration in range(max_iterations):
        iterations_used = iteration + 1
        mujoco.mj_forward(model, data)
        residual, position_errors, orientation_errors = _weighted_residual(
            data, targets, desired_positions, desired_orientations, orientation_weight
        )
        posture_delta = reference_dof_pos - data.qpos[qpos_addresses]
        cost = float(residual @ residual + posture_weight * (posture_delta @ posture_delta))
        if float(np.max(position_errors)) <= position_tolerance:
            return position_errors, orientation_errors, iteration + 1
        if np.isfinite(previous_cost) and previous_cost - cost <= max(1.0e-12, previous_cost * 1.0e-7):
            stagnation_steps += 1
        else:
            stagnation_steps = 0
        if stagnation_steps >= 3:
            return position_errors, orientation_errors, iteration + 1
        previous_cost = cost

        jacobian_rows: list[np.ndarray] = []
        for target in targets:
            jacobian_pos = np.zeros((3, model.nv), dtype=np.float64)
            jacobian_rot = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jac(
                model,
                data,
                jacobian_pos,
                jacobian_rot,
                _world_point(data, target),
                target.target_body_id,
            )
            jacobian_rows.append(np.sqrt(target.position_weight) * jacobian_pos[:, dof_addresses])
            resolved_orientation_weight = orientation_weight * target.orientation_scale
            if resolved_orientation_weight > 0.0:
                jacobian_rows.append(np.sqrt(resolved_orientation_weight) * jacobian_rot[:, dof_addresses])
        jacobian = np.vstack(jacobian_rows)
        lhs = jacobian.T @ jacobian + (damping + posture_weight) * identity
        rhs = jacobian.T @ residual + posture_weight * posture_delta
        try:
            joint_step = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            joint_step = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        joint_step = np.clip(joint_step, -max_joint_step, max_joint_step)

        current_dof_pos = data.qpos[qpos_addresses].copy()
        accepted = False
        for step_scale in (1.0, 0.5, 0.25, 0.125):
            data.qpos[qpos_addresses] = current_dof_pos + step_scale * joint_step
            _clip_control_joints(model, data.qpos, joint_ids, qpos_addresses)
            mujoco.mj_forward(model, data)
            candidate_residual, _, _ = _weighted_residual(data, targets, desired_positions, desired_orientations, orientation_weight)
            candidate_posture = reference_dof_pos - data.qpos[qpos_addresses]
            candidate_cost = float(candidate_residual @ candidate_residual + posture_weight * (candidate_posture @ candidate_posture))
            if candidate_cost < cost:
                accepted = True
                break
        if not accepted:
            data.qpos[qpos_addresses] = current_dof_pos
            mujoco.mj_forward(model, data)
            break

    _, position_errors, orientation_errors = _weighted_residual(data, targets, desired_positions, desired_orientations, orientation_weight)
    return position_errors, orientation_errors, iterations_used


def _target_scale(
    clip: PicoMotionClip,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_body_id: int,
    targets: list[TrackingTarget],
) -> float:
    target_leg_lengths = [np.linalg.norm(_world_point(data, target) - data.xpos[base_body_id]) for target in targets[:2]]
    source_leg_lengths = np.linalg.norm(clip.body_pos_w[:, 1:3] - clip.body_pos_w[:, 0:1], axis=-1)
    source_extended_leg_length = float(np.percentile(source_leg_lengths, 95.0))
    if source_extended_leg_length <= 1.0e-8:
        raise ValueError(f"PICO clip={clip.path} has degenerate pelvis-to-foot distances")
    return float(np.mean(target_leg_lengths) / source_extended_leg_length)


def _nonsingular_leg_seed(robot_spec: RobotSpec, model: mujoco.MjModel, qpos_addresses: np.ndarray) -> np.ndarray:
    """Return a nearly straight pose whose knee has a usable vertical Jacobian."""

    seed = model.qpos0[qpos_addresses].astype(np.float64, copy=True)
    joint_index = {name: idx for idx, name in enumerate(robot_spec.control_joint_names)}
    for side in ("left", "right"):
        hip_name = f"{side}_hip_pitch_joint"
        knee_names = (f"{side}_knee_pitch_joint", f"{side}_knee_joint")
        ankle_name = f"{side}_ankle_pitch_joint"
        if hip_name in joint_index and ankle_name in joint_index:
            knee_name = next((name for name in knee_names if name in joint_index), None)
            if knee_name is not None:
                seed[joint_index[hip_name]] = -0.075
                seed[joint_index[knee_name]] = 0.15
                seed[joint_index[ankle_name]] = -0.075
    return seed


def _apply_geometric_leg_seed(
    dof_pos: np.ndarray,
    robot_spec: RobotSpec,
    desired_positions: np.ndarray,
    root_position: np.ndarray,
    maximum_leg_lengths: np.ndarray,
) -> None:
    """Seed the positive-flexion IK branch from pelvis-to-foot distance."""

    joint_index = {name: idx for idx, name in enumerate(robot_spec.control_joint_names)}
    for target_idx, side in enumerate(("left", "right")):
        knee_name = next(
            (name for name in (f"{side}_knee_pitch_joint", f"{side}_knee_joint") if name in joint_index),
            None,
        )
        hip_name = f"{side}_hip_pitch_joint"
        ankle_name = f"{side}_ankle_pitch_joint"
        if knee_name is None or hip_name not in joint_index or ankle_name not in joint_index:
            continue
        leg_length = float(np.linalg.norm(desired_positions[target_idx] - root_position))
        length_ratio = np.clip(leg_length / float(maximum_leg_lengths[target_idx]), 0.0, 1.0)
        knee_angle = max(0.02, float(2.0 * np.arccos(length_ratio)))
        dof_pos[joint_index[hip_name]] = -0.5 * knee_angle
        dof_pos[joint_index[knee_name]] = knee_angle
        dof_pos[joint_index[ankle_name]] = -0.5 * knee_angle


def retarget_pico_clip(
    clip: PicoMotionClip,
    robot_spec: RobotSpec,
    *,
    scale: float | None = None,
    max_iterations: int = 40,
    position_tolerance: float = 0.005,
    orientation_weight: float = 0.05,
    damping: float = 0.001,
    posture_weight: float = 0.0001,
    max_joint_step: float = 0.08,
) -> RetargetResult:
    for name, value in (
        ("position_tolerance", position_tolerance),
        ("damping", damping),
        ("max_joint_step", max_joint_step),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value}")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be positive, got {max_iterations}")
    if not np.isfinite(orientation_weight) or orientation_weight < 0.0:
        raise ValueError(f"orientation_weight must be finite and non-negative, got {orientation_weight}")
    if not np.isfinite(posture_weight) or posture_weight < 0.0:
        raise ValueError(f"posture_weight must be finite and non-negative, got {posture_weight}")

    model = mujoco.MjModel.from_xml_path(str(robot_spec.xml_file))
    data = mujoco.MjData(model)
    if robot_spec.free_joint is None:
        raise ValueError(f"Target robot={robot_spec.name} must have a free root joint")
    free_joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, robot_spec.free_joint)
    free_qpos_address = int(model.jnt_qposadr[free_joint_id])
    if free_qpos_address != 0:
        raise ValueError(f"Target robot={robot_spec.name} free root joint must start at qpos 0, got {free_qpos_address}")

    joint_ids = np.asarray([_object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in robot_spec.control_joint_names], dtype=np.int32)
    if any(model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE for joint_id in joint_ids):
        raise ValueError(f"PICO IK currently supports hinge control joints only; target robot={robot_spec.name}")
    qpos_addresses = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=np.int32)
    dof_addresses = np.asarray([model.jnt_dofadr[joint_id] for joint_id in joint_ids], dtype=np.int32)
    targets = _tracking_targets(model, robot_spec)
    base_body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, robot_spec.base_body)

    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    maximum_leg_lengths = np.asarray(
        [np.linalg.norm(_world_point(data, target) - data.xpos[base_body_id]) for target in targets[:2]],
        dtype=np.float64,
    )
    resolved_scale = _target_scale(clip, model, data, base_body_id, targets) if scale is None else float(scale)
    if not np.isfinite(resolved_scale) or resolved_scale <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {resolved_scale}")

    source_index = {name: idx for idx, name in enumerate(PICO_BODY_NAMES)}
    orientation_alignment: list[np.ndarray] = []
    for target in targets:
        source_quat = clip.body_quat_wxyz[0, source_index[target.source_name]]
        target_quat = data.xquat[target.target_body_id]
        orientation_alignment.append(_quat_multiply_wxyz(target_quat, _quat_conjugate_wxyz(source_quat)))

    frame_count = clip.body_pos_w.shape[0]
    root_pos = np.empty((frame_count, 3), dtype=np.float32)
    root_quat_wxyz = clip.body_quat_wxyz[:, 0].astype(np.float64, copy=True)
    dof_pos = np.empty((frame_count, len(qpos_addresses)), dtype=np.float32)
    position_errors = np.empty((frame_count, len(targets)), dtype=np.float64)
    orientation_errors = np.empty_like(position_errors)
    iterations = np.empty(frame_count, dtype=np.int32)

    target_root_origin = model.qpos0[:3].astype(np.float64, copy=True)
    source_root_origin = clip.body_pos_w[0, 0].copy()
    # A perfectly straight leg is a first-order singularity for vertical foot
    # motion.  Start from an equivalent, very slightly bent pose so crouching
    # clips choose physical knee flexion instead of the XML's small negative
    # knee range.
    previous_dof_pos = _nonsingular_leg_seed(robot_spec, model, qpos_addresses)
    for frame_idx in range(frame_count):
        current_root_pos = target_root_origin + resolved_scale * (clip.body_pos_w[frame_idx, 0] - source_root_origin)
        root_pos[frame_idx] = current_root_pos
        desired_positions = np.stack(
            [
                current_root_pos
                + resolved_scale
                * (
                    clip.body_pos_w[frame_idx, source_index[target.source_name]]
                    - clip.body_pos_w[frame_idx, source_index[PICO_BODY_NAMES[0]]]
                )
                for target in targets
            ]
        )
        frame_seed = previous_dof_pos.copy()
        _apply_geometric_leg_seed(
            frame_seed,
            robot_spec,
            desired_positions,
            current_root_pos,
            maximum_leg_lengths,
        )
        data.qpos[:] = model.qpos0
        data.qpos[:3] = current_root_pos
        data.qpos[3:7] = root_quat_wxyz[frame_idx]
        data.qpos[qpos_addresses] = frame_seed
        _clip_control_joints(model, data.qpos, joint_ids, qpos_addresses)

        desired_orientations = np.stack(
            [
                _quat_multiply_wxyz(
                    orientation_alignment[target_idx],
                    clip.body_quat_wxyz[frame_idx, source_index[target.source_name]],
                )
                for target_idx, target in enumerate(targets)
            ]
        )
        position_errors[frame_idx], orientation_errors[frame_idx], iterations[frame_idx] = _solve_ik_frame(
            model,
            data,
            targets,
            desired_positions,
            desired_orientations,
            joint_ids,
            qpos_addresses,
            dof_addresses,
            previous_dof_pos,
            max_iterations=max_iterations,
            position_tolerance=position_tolerance,
            orientation_weight=orientation_weight,
            damping=damping,
            posture_weight=posture_weight,
            max_joint_step=max_joint_step,
        )
        previous_dof_pos = data.qpos[qpos_addresses].copy()
        dof_pos[frame_idx] = previous_dof_pos

    if robot_spec.root_quat_order == "xyzw":
        root_quat = root_quat_wxyz[:, [1, 2, 3, 0]].astype(np.float32)
    elif robot_spec.root_quat_order == "wxyz":
        root_quat = root_quat_wxyz.astype(np.float32)
    else:
        raise ValueError(f"Unsupported target root_quat_order={robot_spec.root_quat_order!r}")

    target_mapping = {
        target.source_name: {
            "target_body": target.target_body_name,
            "target_local_point": target.local_point.tolist(),
            "orientation_scale": target.orientation_scale,
        }
        for target in targets
    }
    mean_position_error = {target.source_name: float(np.mean(position_errors[:, idx])) for idx, target in enumerate(targets)}
    max_position_error = {target.source_name: float(np.max(position_errors[:, idx])) for idx, target in enumerate(targets)}
    mean_orientation_error = {target.source_name: float(np.mean(orientation_errors[:, idx])) for idx, target in enumerate(targets)}
    max_orientation_error = {target.source_name: float(np.max(orientation_errors[:, idx])) for idx, target in enumerate(targets)}
    robot_state = RobotStateMotion(
        motion_key=clip.path.stem,
        root_pos=root_pos,
        root_quat=root_quat,
        dof_pos=dof_pos,
        fps=clip.fps,
        joint_names=list(robot_spec.control_joint_names),
        source="pico_motion_clip_ik_retarget",
        metadata={
            "approximate_retarget": True,
            "source_file": str(clip.path),
            "source_schema": "pico_motion_clip",
            "source_body_names": list(PICO_BODY_NAMES),
            "source_body_state_frame": clip.body_state_frame,
            "pico_position_axes_version": clip.position_axes_version,
            "target_mapping": target_mapping,
            "position_scale": resolved_scale,
            "orientation_weight": orientation_weight,
            "ik_mean_position_error_m": mean_position_error,
            "ik_max_position_error_m": max_position_error,
            "ik_mean_orientation_error_rad": mean_orientation_error,
            "ik_max_orientation_error_rad": max_orientation_error,
            "ik_mean_iterations": float(np.mean(iterations)),
            "ik_max_iterations": int(np.max(iterations)),
        },
    )
    motion = robot_state_to_ufo_motion(robot_state, robot_spec, source_name=str(clip.path))
    return RetargetResult(
        motion=motion,
        scale=resolved_scale,
        mean_position_error_m=mean_position_error,
        max_position_error_m=max_position_error,
        mean_orientation_error_rad=mean_orientation_error,
        max_orientation_error_rad=max_orientation_error,
        mean_iterations=float(np.mean(iterations)),
        max_iterations=int(np.max(iterations)),
    )


def convert_pico_clips(
    paths: list[Path],
    output_path: str | Path,
    robot_spec: RobotSpec,
    *,
    scale: float | None = None,
    max_iterations: int = 40,
    position_tolerance: float = 0.005,
    orientation_weight: float = 0.05,
    damping: float = 0.001,
    posture_weight: float = 0.0001,
    max_joint_step: float = 0.08,
    compression: int = 3,
    overwrite: bool = False,
) -> dict[str, RetargetResult]:
    if not paths:
        raise ValueError("paths must not be empty")
    if compression < 0 or compression > 9:
        raise ValueError(f"compression must be in [0, 9], got {compression}")
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".pkl":
        raise ValueError(f"Output must use the .pkl extension: {output}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite to replace it")

    results: dict[str, RetargetResult] = {}
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        clip = load_pico_clip(path)
        if clip.path.stem in records:
            raise ValueError(f"Duplicate PICO motion key={clip.path.stem!r}")
        result = retarget_pico_clip(
            clip,
            robot_spec,
            scale=scale,
            max_iterations=max_iterations,
            position_tolerance=position_tolerance,
            orientation_weight=orientation_weight,
            damping=damping,
            posture_weight=posture_weight,
            max_joint_step=max_joint_step,
        )
        records[clip.path.stem] = result.motion
        results[clip.path.stem] = result

    payload = validate_ufo_motion_dict(records, "pico_motion_clip_conversion")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(payload, temporary, compress=compression)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retarget sparse PICO clips into a UFO MotionLib PKL.")
    parser.add_argument("--input", type=Path, required=True, help="A PICO .npz file or directory containing .npz clips.")
    parser.add_argument("--output", type=Path, required=True, help="Combined MotionLib .pkl output path.")
    parser.add_argument("--robot-config", type=Path, default=Path("configs/robots/mini3.yaml"))
    parser.add_argument("--scale", type=float, default=None, help="Position scale; defaults to automatic pelvis-to-foot scaling.")
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--position-tolerance", type=float, default=0.005)
    parser.add_argument("--orientation-weight", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=0.001)
    parser.add_argument("--posture-weight", type=float, default=0.0001)
    parser.add_argument("--max-joint-step", type=float, default=0.08)
    parser.add_argument("--compression", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot_config = resolve_robot_config_path(args.robot_config)
    robot_spec = load_robot_spec(robot_config)
    paths = discover_pico_clips(args.input)
    results = convert_pico_clips(
        paths,
        args.output,
        robot_spec,
        scale=args.scale,
        max_iterations=args.max_iterations,
        position_tolerance=args.position_tolerance,
        orientation_weight=args.orientation_weight,
        damping=args.damping,
        posture_weight=args.posture_weight,
        max_joint_step=args.max_joint_step,
        compression=args.compression,
        overwrite=args.overwrite,
    )
    for motion_key, result in results.items():
        position_summary = ", ".join(f"{name}={error:.4f}m" for name, error in result.mean_position_error_m.items())
        print(
            f"[pico-convert] motion={motion_key} frames={result.motion['root_trans_offset'].shape[0]} "
            f"fps={result.motion['fps']:g} scale={result.scale:.6f} mean_position_error: {position_summary}"
        )
    print(f"[pico-convert] wrote {len(results)} motions to {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
