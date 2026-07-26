"""Time-stretch every motion in one combined UFO PKL at multiple scales.

The output keeps the source FPS. A scale of 2 creates roughly twice as many
frames and therefore doubles the motion duration when a fixed-rate policy
consumes it. Translation and joint positions use linear interpolation, while
root quaternions and axis-angle rotations use shortest-path quaternion SLERP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from humanoidverse.utils.motion_data.resample import resample_ufo_motion_record
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict

DEFAULT_SCALES = (1.25, 1.5, 1.75, 2.0, 0.75, 0.5, 0.25)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _scale_tag(scale: float) -> str:
    return f"x{scale:g}".replace(".", "p")


def _write_joblib_atomic(path: Path, payload: dict[str, Any], *, compression: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(payload, temporary, compress=compression)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _frame_count(record: dict[str, Any]) -> int:
    return int(np.asarray(record["root_trans_offset"]).shape[0])


def _fps(record: dict[str, Any]) -> float:
    return float(np.asarray(record["fps"]).reshape(-1)[0])


def _motion_duration(record: dict[str, Any]) -> float:
    return (_frame_count(record) - 1) / _fps(record)


def resample_scales(
    input_path: str | Path,
    output_dir: str | Path,
    scales: list[float] | tuple[float, ...],
    *,
    compression: int = 3,
    overwrite: bool = False,
    verify_output: bool = False,
    log_every: int = 10,
) -> dict[str, Any]:
    source_path = Path(input_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Input UFO PKL does not exist: {source_path}")
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root == source_path.parent:
        raise ValueError("Output directory must differ from the input file directory")
    if compression < 0 or compression > 9:
        raise ValueError(f"compression must be in [0, 9], got {compression}")
    if log_every <= 0:
        raise ValueError(f"log_every must be positive, got {log_every}")

    normalized_scales = [float(scale) for scale in scales]
    if not normalized_scales:
        raise ValueError("At least one resampling scale is required")
    if any(not np.isfinite(scale) or scale <= 0.0 for scale in normalized_scales):
        raise ValueError(f"All scales must be finite and positive, got {normalized_scales}")
    tags = [_scale_tag(scale) for scale in normalized_scales]
    if len(set(tags)) != len(tags):
        raise ValueError(f"Scales produce duplicate output names: {normalized_scales}")

    source = validate_ufo_motion_dict(joblib.load(source_path), f"source:{source_path.name}")
    source_sha256 = _sha256_file(source_path)
    source_frames = sum(_frame_count(record) for record in source.values())
    source_duration = sum(_motion_duration(record) for record in source.values())
    source_fps_values = sorted({_fps(record) for record in source.values()})

    summary: dict[str, Any] = {
        "operation": "combined_ufo_multi_scale_time_stretch",
        "input_path": str(source_path),
        "input_sha256": source_sha256,
        "motion_count": len(source),
        "source_frames": source_frames,
        "source_duration_seconds": source_duration,
        "source_fps_values": source_fps_values,
        "outputs": [],
    }

    for scale, tag in zip(normalized_scales, tags):
        output_path = output_root / f"{source_path.stem}_{tag}.pkl"
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}; pass --overwrite to replace it")

        output: dict[str, Any] = {}
        for motion_index, (motion_key, source_record) in enumerate(source.items(), start=1):
            source_fps = _fps(source_record)
            interpolation_grid_fps = source_fps * scale
            output_record = resample_ufo_motion_record(
                source_record,
                interpolation_grid_fps,
                source_name=f"{source_path.name}:{motion_key}:{tag}",
            )
            metadata = dict(output_record.get("metadata") or {})
            interpolation_metadata = metadata.pop("resampling")
            output_record["fps"] = source_fps
            output_frames = _frame_count(output_record)
            output_duration = (output_frames - 1) / source_fps
            source_motion_duration = _motion_duration(source_record)
            metadata["time_stretch"] = {
                "algorithm": "ufo_motion_time_stretch_v1",
                "requested_duration_scale": scale,
                "realized_duration_scale": output_duration / source_motion_duration,
                "source_fps": source_fps,
                "output_fps": source_fps,
                "interpolation_grid_fps": interpolation_grid_fps,
                "source_frame_count": _frame_count(source_record),
                "output_frame_count": output_frames,
                "source_duration_seconds": source_motion_duration,
                "output_duration_seconds": output_duration,
                "endpoint_policy": interpolation_metadata["endpoint_policy"],
                "translation_and_scalar_interpolation": interpolation_metadata[
                    "translation_and_scalar_interpolation"
                ],
                "rotation_interpolation": interpolation_metadata["rotation_interpolation"],
                "source_combined_file": str(source_path),
                "source_combined_sha256": source_sha256,
            }
            output_record["metadata"] = metadata
            output[motion_key] = output_record
            if motion_index % log_every == 0 or motion_index == len(source):
                print(f"[{tag}] motions={motion_index}/{len(source)}", flush=True)

        validated_output = validate_ufo_motion_dict(output, f"output:{output_path.name}")
        _write_joblib_atomic(output_path, validated_output, compression=compression)
        if verify_output:
            verified = validate_ufo_motion_dict(joblib.load(output_path), f"verify:{output_path.name}")
            if list(verified) != list(source):
                raise ValueError(f"Verified output motion order differs from source: {output_path}")
        else:
            verified = validated_output

        target_frames = sum(_frame_count(record) for record in verified.values())
        target_duration = sum(_motion_duration(record) for record in verified.values())
        output_fps_values = sorted({_fps(record) for record in verified.values()})
        interpolation_grid_fps_values = sorted(
            {
                float(record["metadata"]["time_stretch"]["interpolation_grid_fps"])
                for record in verified.values()
            }
        )
        output_summary = {
            "duration_scale": scale,
            "tag": tag,
            "path": str(output_path),
            "motion_count": len(verified),
            "frames": target_frames,
            "duration_seconds": target_duration,
            "output_fps_values": output_fps_values,
            "interpolation_grid_fps_values": interpolation_grid_fps_values,
            "size_bytes": output_path.stat().st_size,
        }
        summary["outputs"].append(output_summary)
        print(
            f"[{tag}] wrote={output_path} frames={source_frames}->{target_frames} "
            f"duration={source_duration:.6f}s->{target_duration:.6f}s "
            f"output_fps={output_fps_values} interpolation_grid_fps={interpolation_grid_fps_values}",
            flush=True,
        )

    summary_path = output_root / "_resampling_summary.json"
    summary["summary_path"] = str(summary_path)
    _write_json_atomic(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Combined UFO MotionLib PKL containing one or more full motions.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=list(DEFAULT_SCALES),
        help="Motion-duration scales; output FPS remains unchanged.",
    )
    parser.add_argument("--compression", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-output", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args(argv)

    summary = resample_scales(
        args.input,
        args.output_dir,
        args.scales,
        compression=args.compression,
        overwrite=bool(args.overwrite),
        verify_output=bool(args.verify_output),
        log_every=args.log_every,
    )
    print(
        f"Finished multi-scale resampling: motions={summary['motion_count']} "
        f"outputs={len(summary['outputs'])}\n  summary={summary['summary_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
