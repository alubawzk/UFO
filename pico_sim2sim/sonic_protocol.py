"""Gear Sonic compatible binary pose messages.

Wire layout: ``topic + 1280-byte JSON header + contiguous little-endian
arrays``.  Keeping this tiny codec local makes the PICO publisher independent
of the Gear Sonic repository.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np

HEADER_SIZE = 1280

_DTYPE_TO_NAME = {
    np.dtype("float32"): "f32",
    np.dtype("float64"): "f64",
    np.dtype("int32"): "i32",
    np.dtype("int64"): "i64",
    np.dtype("uint8"): "u8",
    np.dtype("bool"): "bool",
}


def pack_pose_message(
    pose_data: Mapping[str, np.ndarray],
    *,
    topic: str = "pose",
    version: int = 3,
) -> bytes:
    """Pack NumPy fields using the Sonic self-describing pose protocol."""

    fields: list[dict[str, object]] = []
    payload: list[bytes] = []
    for name, raw_value in pose_data.items():
        value = np.asarray(raw_value)
        dtype_name = _DTYPE_TO_NAME.get(value.dtype)
        if dtype_name is None:
            value = value.astype(np.float32)
            dtype_name = "f32"
        value = np.ascontiguousarray(value)
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))
        fields.append({"name": str(name), "dtype": dtype_name, "shape": list(value.shape)})
        payload.append(value.tobytes())

    header = {
        "v": int(version),
        "endian": "le",
        "count": 1,
        "fields": fields,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_bytes) > HEADER_SIZE:
        raise ValueError(f"Sonic header is too large: {len(header_bytes)} > {HEADER_SIZE}")
    return topic.encode("utf-8") + header_bytes.ljust(HEADER_SIZE, b"\x00") + b"".join(payload)
