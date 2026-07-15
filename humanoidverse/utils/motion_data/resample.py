"""Time-grid resampling for UFO MotionLib motion records."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict
from humanoidverse.utils.motion_data.time_grid import (
    LEGACY_EXPERT_CONSUMER_CONTRACT,
    RESAMPLING_ALGORITHM,
    RESAMPLING_SCHEMA_VERSION,
    legacy_expert_sample_count,
)

_LINEAR_TIME_SERIES_FIELDS = frozenset({"root_trans_offset", "dof_pos", "action"})
_NEAREST_TIME_SERIES_FIELDS = frozenset({"contact", "contacts", "contact_binary"})
_ROTATION_TIME_SERIES_FIELDS = frozenset({"pose_aa", "root_quat"})
_KNOWN_TIME_SERIES_FIELDS = _LINEAR_TIME_SERIES_FIELDS | _NEAREST_TIME_SERIES_FIELDS | _ROTATION_TIME_SERIES_FIELDS


def _target_sample_times(source_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    """Return an exact target-fps grid which never extends past the source."""

    if source_frames <= 0:
        raise ValueError(f"source_frames must be positive, got {source_frames}")
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"source_fps must be finite and positive, got {source_fps}")
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError(f"target_fps must be finite and positive, got {target_fps}")

    duration = (source_frames - 1) / source_fps
    scaled_last_index = duration * target_fps
    nearest_integer = round(scaled_last_index)
    tolerance = 1.0e-9 * max(1.0, abs(scaled_last_index))
    if abs(scaled_last_index - nearest_integer) <= tolerance:
        scaled_last_index = float(nearest_integer)
    last_index = int(np.floor(scaled_last_index))
    return np.arange(last_index + 1, dtype=np.float64) / target_fps


def target_frame_count(source_frames: int, source_fps: float, target_fps: float) -> int:
    """Return the number of frames produced by the non-extrapolating target grid."""

    return int(_target_sample_times(source_frames, source_fps, target_fps).shape[0])


def _source_indices_and_blend(target_times: np.ndarray, source_frames: int, source_fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_positions = np.asarray(target_times, dtype=np.float64) * source_fps
    lower = np.floor(source_positions).astype(np.int64)
    lower = np.clip(lower, 0, source_frames - 1)
    upper = np.minimum(lower + 1, source_frames - 1)
    blend = np.clip(source_positions - lower, 0.0, 1.0)
    blend[lower == upper] = 0.0
    return lower, upper, blend


def _linear_resample(values: np.ndarray, lower: np.ndarray, upper: np.ndarray, blend: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    original_dtype = source.dtype
    weight = blend.reshape((blend.shape[0],) + (1,) * (source.ndim - 1))
    result = np.asarray(source[lower], dtype=np.float64) * (1.0 - weight) + np.asarray(source[upper], dtype=np.float64) * weight
    return result.astype(original_dtype, copy=False)


def _nearest_resample(values: np.ndarray, lower: np.ndarray, upper: np.ndarray, blend: np.ndarray) -> np.ndarray:
    indices = np.where(blend < 0.5, lower, upper)
    return np.asarray(values)[indices].copy()


def _normalize_quaternions_xyzw(quaternions: np.ndarray, field: str) -> np.ndarray:
    quat = np.asarray(quaternions, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f"{field} must end in quaternion dimension 4, got {quat.shape}")
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0.0):
        raise ValueError(f"{field} contains zero or non-finite quaternions")
    return quat / norm


def _slerp_quaternion_pairs(quat0: np.ndarray, quat1: np.ndarray, blend: np.ndarray, field: str) -> np.ndarray:
    q0 = _normalize_quaternions_xyzw(quat0, field)
    q1 = _normalize_quaternions_xyzw(quat1, field)
    dot = np.sum(q0 * q1, axis=-1)
    q1 = np.where((dot < 0.0)[..., None], -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)

    blend_view = blend.reshape((blend.shape[0],) + (1,) * (dot.ndim - 1))
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    safe_denominator = np.where(sin_theta > 1.0e-8, sin_theta, 1.0)
    weight0 = np.sin((1.0 - blend_view) * theta) / safe_denominator
    weight1 = np.sin(blend_view * theta) / safe_denominator
    spherical = weight0[..., None] * q0 + weight1[..., None] * q1
    linear = (1.0 - blend_view[..., None]) * q0 + blend_view[..., None] * q1
    result = np.where((dot > 0.9995)[..., None], linear, spherical)
    return _normalize_quaternions_xyzw(result, field)


def _quaternion_order(record: dict[str, Any]) -> str:
    metadata = record.get("metadata")
    order = metadata.get("root_quat_order", "xyzw") if isinstance(metadata, dict) else "xyzw"
    if order not in {"xyzw", "wxyz"}:
        raise ValueError(f"Unsupported root_quat_order={order!r}; expected 'xyzw' or 'wxyz'")
    return str(order)


def _to_xyzw(quaternions: np.ndarray, order: str) -> np.ndarray:
    quat = np.asarray(quaternions)
    return quat if order == "xyzw" else quat[..., [1, 2, 3, 0]]


def _from_xyzw(quaternions: np.ndarray, order: str) -> np.ndarray:
    quat = np.asarray(quaternions)
    return quat if order == "xyzw" else quat[..., [3, 0, 1, 2]]


def _resample_quaternions(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    blend: np.ndarray,
    *,
    field: str,
    order: str = "xyzw",
) -> np.ndarray:
    source = np.asarray(values)
    xyzw = _to_xyzw(source, order)
    result = _slerp_quaternion_pairs(xyzw[lower], xyzw[upper], blend, field)
    result = _from_xyzw(result, order)
    return result.astype(source.dtype, copy=False)


def _resample_axis_angle(values: np.ndarray, lower: np.ndarray, upper: np.ndarray, blend: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    source_quat = Rotation.from_rotvec(np.asarray(source, dtype=np.float64).reshape(-1, 3)).as_quat().reshape(*source.shape[:-1], 4)
    target_quat = _slerp_quaternion_pairs(source_quat[lower], source_quat[upper], blend, "pose_aa")
    target = Rotation.from_quat(target_quat.reshape(-1, 4)).as_rotvec().reshape(*target_quat.shape[:-1], 3)

    exact = (blend == 0.0) | (lower == upper)
    if np.any(exact):
        target[exact] = source[lower[exact]]
    return target.astype(source.dtype, copy=False)


def resample_ufo_motion_record(record: dict[str, Any], target_fps: float = 50.0, *, source_name: str = "resample") -> dict[str, Any]:
    """Resample explicitly supported synchronized fields onto a fixed FPS grid.

    Float arrays use linear interpolation, except ``root_quat`` and ``pose_aa``
    which use shortest-path quaternion interpolation. Integer, boolean, and
    other contact time-series arrays use nearest-neighbour selection. Ambiguous
    frame-shaped arrays are rejected. The target grid starts at zero and never
    extrapolates beyond the final source frame.
    """

    validated = validate_ufo_motion_dict({"motion": record}, source_name)["motion"]
    source_frames = int(np.asarray(validated["root_trans_offset"]).shape[0])
    if source_frames < 2:
        raise ValueError(f"Cannot resample a one-frame motion for expert training: source={source_name}")
    source_fps = float(np.asarray(validated["fps"]).reshape(-1)[0])
    target_fps = float(target_fps)
    target_times = _target_sample_times(source_frames, source_fps, target_fps)
    if target_times.shape[0] < 2:
        raise ValueError(
            f"Resampling source={source_name} would produce only one frame at target_fps={target_fps:g}; "
            "expert MotionLib requires at least two target frames"
        )
    lower, upper, blend = _source_indices_and_blend(target_times, source_frames, source_fps)
    quat_order = _quaternion_order(validated)

    for key in _KNOWN_TIME_SERIES_FIELDS.intersection(validated):
        value = validated[key]
        if not isinstance(value, np.ndarray) or value.ndim == 0 or value.shape[0] != source_frames:
            shape = getattr(value, "shape", None)
            raise ValueError(f"Known time-series field {key!r} must have first dimension {source_frames}, got {shape}")
        if key in (_LINEAR_TIME_SERIES_FIELDS | _ROTATION_TIME_SERIES_FIELDS) and not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"Interpolated time-series field {key!r} must use a floating dtype, got {value.dtype}")

    result: dict[str, Any] = {}
    for key, value in validated.items():
        if key == "fps":
            continue
        if key == "pose_aa":
            result[key] = _resample_axis_angle(value, lower, upper, blend)
        elif key == "root_quat":
            result[key] = _resample_quaternions(value, lower, upper, blend, field=key, order=quat_order)
        elif key in _LINEAR_TIME_SERIES_FIELDS:
            result[key] = _linear_resample(value, lower, upper, blend)
        elif key in _NEAREST_TIME_SERIES_FIELDS:
            result[key] = _nearest_resample(value, lower, upper, blend)
        elif isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == source_frames:
            raise ValueError(
                f"Unknown ndarray field {key!r} looks synchronized with the motion ({source_frames} frames); "
                "declare an explicit interpolation policy before resampling it"
            )
        else:
            result[key] = deepcopy(value)

    target_frames = int(target_times.shape[0])
    source_duration = (source_frames - 1) / source_fps
    target_duration = (target_frames - 1) / target_fps
    metadata = dict(result.get("metadata") or {})
    metadata["resampling"] = {
        "algorithm": RESAMPLING_ALGORITHM,
        "schema_version": RESAMPLING_SCHEMA_VERSION,
        "source_fps": source_fps,
        "target_fps": target_fps,
        "source_frame_count": source_frames,
        "target_frame_count": target_frames,
        "source_duration_seconds": source_duration,
        "target_duration_seconds": target_duration,
        "time_grid": "arange(target_frame_count) / target_fps",
        "endpoint_policy": "last target time must not exceed the final source-frame time",
        "consumer_contract": LEGACY_EXPERT_CONSUMER_CONTRACT,
        "expert_control_dt_seconds": 1.0 / target_fps,
        "legacy_expert_sample_count": legacy_expert_sample_count(source_frames, source_fps, 1.0 / target_fps),
        "translation_and_scalar_interpolation": "linear",
        "rotation_interpolation": "shortest-path quaternion slerp",
        "derived_state_semantics": "MotionLib recomputes FK and velocities at target_fps; values are not legacy numerically equivalent",
    }
    result["metadata"] = metadata
    result["fps"] = target_fps
    return validate_ufo_motion_dict({"motion": result}, source_name)["motion"]
