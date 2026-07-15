"""Shared control-time grid helpers for motion-data consumers."""

from __future__ import annotations

import math

import numpy as np

RESAMPLING_ALGORITHM = "ufo_motion_resample_v2"
RESAMPLING_SCHEMA_VERSION = 2
LEGACY_EXPERT_CONSUMER_CONTRACT = "legacy_expert_control_grid_v1"


def legacy_expert_sample_count_from_length(motion_length: float, control_dt: float) -> int:
    """Match the legacy loader's float32 ``ceil(length / control_dt)`` rule."""

    length = np.float32(motion_length)
    dt = np.float32(control_dt)
    if not np.isfinite(length) or length < 0.0:
        raise ValueError(f"motion_length must be finite and non-negative, got {motion_length}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"control_dt must be finite and positive, got {control_dt}")
    return int(math.ceil(float(np.float32(length / dt))))


def legacy_expert_sample_count(frame_count: int, fps: float, control_dt: float) -> int:
    """Return the legacy expert sample count for one uniformly sampled motion."""

    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    fps_value = float(fps)
    if not np.isfinite(fps_value) or fps_value <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    motion_length = (frame_count - 1) / fps_value
    return legacy_expert_sample_count_from_length(motion_length, control_dt)
