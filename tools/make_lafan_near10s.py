#!/usr/bin/env python3
"""Build a near-10s RP1/LAFAN pickle from a full-length pickle.

Input format is the same as `lafan_data_50hz.pkl`:
  {motion_name: {"root_trans_offset": (T,3), "pose_aa": (T,J,3), "fps": int, ...}}

Default behavior (`--strategy fullsplit`) tries to avoid wasting motion:
- If a motion is long, split it into non-overlapping equal parts that cover all frames.
- Part length is usually close to `seconds`, and all frames are used.
- Example: 48s with target 10s -> 4 parts of ~12s.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create near-10s clipped LAFAN pickle.")
    p.add_argument("--input", type=str, required=True, help="Input full-length pkl path.")
    p.add_argument("--output", type=str, required=True, help="Output near-10s pkl path.")
    p.add_argument("--seconds", type=float, default=10.0, help="Target clip seconds per motion.")
    p.add_argument(
        "--strategy",
        type=str,
        choices=("fullsplit", "single"),
        default="fullsplit",
        help="fullsplit: cover full motion with near-target chunks; single: output one clip per motion.",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=("center", "head", "tail", "random"),
        default="center",
        help="Only for --strategy single: how to pick one clip when motion is longer than target.",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for --mode random.")
    p.add_argument(
        "--skip-short",
        action="store_true",
        help="Skip motions shorter than target length. Default keeps them unchanged.",
    )
    return p.parse_args()


def _pick_start(total_len: int, clip_len: int, mode: str, rng: np.random.Generator) -> int:
    if total_len <= clip_len:
        return 0
    if mode == "head":
        return 0
    if mode == "tail":
        return total_len - clip_len
    if mode == "center":
        return (total_len - clip_len) // 2
    # random
    return int(rng.integers(0, total_len - clip_len + 1))


def _clip_motion_dict(motion: dict[str, Any], start: int, end: int, total_len: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in motion.items():
        # Clip any time-major ndarray whose first dim equals total frame length.
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == total_len:
            out[k] = v[start:end].copy()
        else:
            out[k] = v
    return out


def _fullsplit_ranges(total_len: int, target_len: int) -> list[tuple[int, int]]:
    if total_len <= 0:
        return []
    if total_len <= target_len:
        return [(0, total_len)]

    # Keep each chunk >= target length and use all frames without overlap.
    num_parts = max(1, total_len // target_len)
    base = total_len // num_parts
    rem = total_len % num_parts

    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(num_parts):
        seg_len = base + (1 if i < rem else 0)
        end = start + seg_len
        ranges.append((start, end))
        start = end
    return ranges


def main() -> None:
    args = _parse_args()
    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_path} ...")
    data = joblib.load(in_path)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict pkl, got {type(data)} from {in_path}")
    print(f"Loaded {len(data)} motions.")

    rng = np.random.default_rng(args.seed)
    out: dict[str, Any] = {}
    n_skipped = 0
    n_total_parts = 0

    for name, motion in data.items():
        if not isinstance(motion, dict):
            out[name] = motion
            continue
        if "root_trans_offset" not in motion:
            out[name] = motion
            continue

        root = motion["root_trans_offset"]
        if not isinstance(root, np.ndarray) or root.ndim != 2:
            out[name] = motion
            continue

        fps = int(motion.get("fps", 50))
        clip_len = max(1, int(round(float(args.seconds) * float(fps))))
        total_len = int(root.shape[0])

        if total_len < clip_len and args.skip_short:
            n_skipped += 1
            continue

        if args.strategy == "single":
            start = _pick_start(total_len, clip_len, args.mode, rng)
            end = min(total_len, start + clip_len)
            out[name] = _clip_motion_dict(motion, start, end, total_len)
            n_total_parts += 1
        else:
            ranges = _fullsplit_ranges(total_len, clip_len)
            if len(ranges) == 1:
                out[name] = _clip_motion_dict(motion, ranges[0][0], ranges[0][1], total_len)
                n_total_parts += 1
            else:
                n_parts = len(ranges)
                for idx, (start, end) in enumerate(ranges, start=1):
                    part_name = f"{name}__part{idx:02d}of{n_parts:02d}"
                    out[part_name] = _clip_motion_dict(motion, start, end, total_len)
                    n_total_parts += 1

    print(f"Saving to {out_path} ...")
    joblib.dump(out, out_path)
    print(f"Saved near-{args.seconds:g}s data: {out_path}")
    print(
        f"Input motions: {len(data)}, output motions: {len(out)}, "
        f"total output parts: {n_total_parts}, skipped short: {n_skipped}"
    )


if __name__ == "__main__":
    main()
