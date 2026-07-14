"""Convert flat Mini3 PKLs into lazy per-motion UFO training PKLs.

Each source file must contain ``root_pos[T,3]``, ``root_rot[T,4]``,
``dof_pos[T,21]``, and ``fps``. Root rotation is read directly from
``root_rot``; no CSV Euler-angle reconstruction is performed.

The output directory contains one UFO PKL per approximately fixed-duration
training clip. This layout lets MotionLib index the full dataset while loading
only the motions sampled for the current environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from omegaconf import OmegaConf

from humanoidverse.utils.motion_data.clip import clip_ufo_motion_dict
from humanoidverse.utils.motion_data.manifest import PER_MOTION_DIRECTORY_INDEX
from humanoidverse.utils.motion_data.robot_state import RobotStateMotion
from humanoidverse.utils.motion_data.robot_state_convert import robot_state_to_ufo_motion
from humanoidverse.utils.motion_data.robot_state_readers import read_robot_state_pkl
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict
from humanoidverse.utils.robot_spec import RobotSpec, load_robot_spec

INDEX_FORMAT = "ufo_per_motion_directory_v1"


class Mini3PklConversionError(ValueError):
    """Raised when flat Mini3 PKL conversion cannot safely continue."""


@dataclass(frozen=True)
class DatasetConversionResult:
    source_files: int
    converted_source_files: int
    reused_motion_files: int
    written_motion_files: int
    skipped_short_files: int
    failed_source_files: int
    output_frames: int
    output_duration_seconds: float
    output_dir: Path
    manifest_path: Path


def discover_pkl_files(
    input_path: str | Path,
    *,
    exclude_mirrored: bool = False,
    max_files: int | None = None,
    sample_files: int | None = None,
    seed: int = 4728,
) -> list[Path]:
    """Recursively discover source flat PKLs with unique motion stems."""

    source = Path(input_path).expanduser().resolve()
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(source.rglob("*.pkl"))
    else:
        raise FileNotFoundError(f"Mini3 PKL input does not exist: {source}")
    if exclude_mirrored:
        paths = [path for path in paths if not path.stem.endswith("_M")]
    if not paths:
        raise FileNotFoundError(f"No Mini3 PKL files found under: {source}")
    if max_files is not None and sample_files is not None:
        raise ValueError("max_files and sample_files are mutually exclusive")
    for name, value in (("max_files", max_files), ("sample_files", sample_files)):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if max_files is not None:
        paths = paths[:max_files]
    elif sample_files is not None and sample_files < len(paths):
        paths = sorted(random.Random(seed).sample(paths, sample_files))

    path_by_stem: dict[str, Path] = {}
    collisions: list[tuple[str, Path, Path]] = []
    for path in paths:
        previous = path_by_stem.get(path.stem)
        if previous is not None:
            collisions.append((path.stem, previous, path))
        path_by_stem[path.stem] = path
    if collisions:
        details = [f"{stem}: {first} / {second}" for stem, first, second in collisions[:10]]
        raise Mini3PklConversionError("Selected PKL files have duplicate motion stems: " + "; ".join(details))
    return paths


def _validate_source_motion(
    motion: RobotStateMotion,
    robot_spec: RobotSpec,
    path: Path,
    *,
    quat_norm_tolerance: float,
    joint_limit_tolerance_degrees: float,
) -> None:
    if quat_norm_tolerance < 0.0 or not np.isfinite(quat_norm_tolerance):
        raise ValueError(f"quat_norm_tolerance must be finite and non-negative, got {quat_norm_tolerance}")
    quat_norm = np.linalg.norm(motion.root_quat, axis=1)
    max_norm_error = float(np.max(np.abs(quat_norm - 1.0)))
    if max_norm_error > quat_norm_tolerance:
        raise Mini3PklConversionError(
            f"PKL file={path} root_rot is not unit length: max |norm-1|={max_norm_error:.6g}, tolerance={quat_norm_tolerance:.6g}"
        )

    if joint_limit_tolerance_degrees < 0.0 or not np.isfinite(joint_limit_tolerance_degrees):
        raise ValueError(f"joint_limit_tolerance_degrees must be finite and non-negative, got {joint_limit_tolerance_degrees}")
    lower: list[float] = []
    upper: list[float] = []
    for joint_name in robot_spec.control_joint_names:
        joint_range = robot_spec.joint_ranges[joint_name]
        if joint_range is None:
            lower.append(-np.inf)
            upper.append(np.inf)
        else:
            lower.append(float(joint_range[0]))
            upper.append(float(joint_range[1]))
    lower_array = np.asarray(lower, dtype=np.float32)
    upper_array = np.asarray(upper, dtype=np.float32)
    tolerance = np.deg2rad(float(joint_limit_tolerance_degrees))
    invalid = (motion.dof_pos < lower_array[None, :] - tolerance) | (motion.dof_pos > upper_array[None, :] + tolerance)
    if np.any(invalid):
        frame_index, joint_index = (int(value) for value in np.argwhere(invalid)[0])
        joint_name = robot_spec.control_joint_names[joint_index]
        raise Mini3PklConversionError(
            f"PKL file={path} joint={joint_name} frame={frame_index} is outside the XML limit: "
            f"value={float(motion.dof_pos[frame_index, joint_index]):.8g} rad, "
            f"range=[{lower_array[joint_index]:.8g}, {upper_array[joint_index]:.8g}] rad"
        )


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _selection_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_motion_file(path: Path) -> dict[str, Any]:
    payload = joblib.load(path)
    validated = validate_ufo_motion_dict(payload, f"converted:{path.name}")
    if list(validated) != [path.stem]:
        raise Mini3PklConversionError(f"Converted file={path} must contain exactly one motion whose key matches the filename stem")
    return validated[path.stem]


def _write_motion_file(path: Path, record: dict[str, Any], *, compression: int) -> None:
    payload = validate_ufo_motion_dict({path.stem: record}, f"write:{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(payload, temporary, compress=compression)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_config(robot_config: str | Path, output_dir: Path, dataset_name: str) -> dict[str, Any]:
    return {
        "robot_config": _portable_path(Path(robot_config)),
        "datasets": [
            {
                "name": dataset_name,
                "format": "ufo_pkl",
                "train_path": _portable_path(output_dir),
                "storage": "per_motion_directory",
                "weight": 1.0,
            }
        ],
    }


def _write_manifest(path: Path, config: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        existing = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if existing != config:
            raise FileExistsError(f"Training manifest already exists with different content: {path}; use --overwrite")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        OmegaConf.save(OmegaConf.create(config), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def convert_pkl_dataset(
    paths: list[Path],
    output_dir: str | Path,
    manifest_path: str | Path,
    robot_spec: RobotSpec,
    *,
    robot_config: str | Path,
    dataset_name: str = "mini3_pkl",
    root_quat_order: str = "xyzw",
    fps_override: float | None = None,
    clip_seconds: float = 10.0,
    stride_seconds: float | None = None,
    keep_short: bool = True,
    min_clip_seconds: float = 1.0,
    quat_norm_tolerance: float = 1.0e-3,
    joint_limit_tolerance_degrees: float = 0.01,
    compression: int = 3,
    overwrite: bool = False,
    verify_output: bool = False,
    continue_on_error: bool = False,
    log_every: int = 100,
) -> DatasetConversionResult:
    """Stream flat source PKLs into a lazy per-motion UFO training directory."""

    if not paths:
        raise ValueError("paths must not be empty")
    if clip_seconds <= 0.0 or not np.isfinite(clip_seconds):
        raise ValueError(f"clip_seconds must be finite and positive, got {clip_seconds}")
    stride = clip_seconds if stride_seconds is None else float(stride_seconds)
    if stride <= 0.0 or not np.isfinite(stride):
        raise ValueError(f"stride_seconds must be finite and positive, got {stride}")
    if min_clip_seconds < 0.0 or not np.isfinite(min_clip_seconds):
        raise ValueError(f"min_clip_seconds must be finite and non-negative, got {min_clip_seconds}")
    if compression < 0 or compression > 9:
        raise ValueError(f"compression must be in [0, 9], got {compression}")
    if log_every <= 0:
        raise ValueError(f"log_every must be positive, got {log_every}")

    output = Path(output_dir).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / PER_MOTION_DIRECTORY_INDEX
    conversion_config = {
        "robot_name": robot_spec.name,
        "robot_config": _portable_path(Path(robot_config)),
        "source_files": len(paths),
        "source_selection_sha256": _selection_digest(paths),
        "root_rotation_source": "direct root_rot field",
        "root_quat_order": root_quat_order,
        "fps_override": fps_override,
        "clip_seconds": float(clip_seconds),
        "stride_seconds": stride,
        "keep_short": bool(keep_short),
        "min_clip_seconds": float(min_clip_seconds),
        "quat_norm_tolerance": float(quat_norm_tolerance),
        "joint_limit_tolerance_degrees": float(joint_limit_tolerance_degrees),
        "compression": int(compression),
    }
    if index_path.exists():
        existing_index = json.loads(index_path.read_text())
        if existing_index.get("conversion") != conversion_config:
            raise Mini3PklConversionError(
                f"Output directory={output} was created with different conversion options; use a new output directory"
            )
    elif any(output.glob("*.pkl")):
        raise Mini3PklConversionError(
            f"Output directory={output} contains PKLs but has no {PER_MOTION_DIRECTORY_INDEX}; use an empty directory"
        )

    index: dict[str, Any] = {
        "format": INDEX_FORMAT,
        "status": "in_progress",
        "conversion": conversion_config,
        "motion_files": 0,
    }
    _write_json_atomic(index_path, index)

    started = time.monotonic()
    converted_sources = 0
    written = 0
    reused = 0
    skipped_short = 0
    failed = 0
    output_frames = 0
    output_duration = 0.0
    expected_output_names: set[str] = set()
    failures: list[str] = []

    for source_index, source in enumerate(paths, start=1):
        source = source.expanduser().resolve()
        try:
            motions = read_robot_state_pkl(
                source,
                source_name="mini3_flat_pkl_conversion",
                robot_spec=robot_spec,
                fps=fps_override,
                root_quat_order=root_quat_order,
            )
            motion = motions[source.stem]
            _validate_source_motion(
                motion,
                robot_spec,
                source,
                quat_norm_tolerance=quat_norm_tolerance,
                joint_limit_tolerance_degrees=joint_limit_tolerance_degrees,
            )
            converted = robot_state_to_ufo_motion(motion, robot_spec, "mini3_flat_pkl_conversion")
            try:
                clips = clip_ufo_motion_dict(
                    {motion.motion_key: converted},
                    clip_seconds=clip_seconds,
                    stride_seconds=stride,
                    keep_short=keep_short,
                    min_clip_seconds=min_clip_seconds,
                    source_name=str(source),
                )
            except ValueError as exc:
                if "No motion clips were generated" not in str(exc):
                    raise
                clips = {}
            if not clips:
                skipped_short += 1
                converted_sources += 1
                continue

            if overwrite:
                expected_for_source = {f"{key}.pkl" for key in clips}
                for old_path in output.glob(f"{source.stem}__clip*.pkl"):
                    if old_path.name not in expected_for_source:
                        old_path.unlink()
            for motion_key, record in clips.items():
                output_path = output / f"{motion_key}.pkl"
                expected_output_names.add(output_path.name)
                frame_count = int(np.asarray(record["root_trans_offset"]).shape[0])
                motion_fps = float(record["fps"])
                output_frames += frame_count
                output_duration += frame_count / motion_fps
                if output_path.exists() and not overwrite:
                    if verify_output:
                        _verify_motion_file(output_path)
                    reused += 1
                else:
                    _write_motion_file(output_path, record, compression=compression)
                    if verify_output:
                        _verify_motion_file(output_path)
                    written += 1
            converted_sources += 1
        except Exception as exc:
            failed += 1
            failures.append(f"{source}: {type(exc).__name__}: {exc}")
            if not continue_on_error:
                index.update({"status": "failed", "failures": failures[:20]})
                _write_json_atomic(index_path, index)
                raise

        if source_index % log_every == 0 or source_index == len(paths):
            elapsed = time.monotonic() - started
            index.update(
                {
                    "status": "in_progress",
                    "processed_source_files": source_index,
                    "motion_files": written + reused,
                    "failed_source_files": failed,
                }
            )
            _write_json_atomic(index_path, index)
            print(
                f"progress={source_index}/{len(paths)} converted_sources={converted_sources} "
                f"written={written} reused={reused} skipped_short={skipped_short} failed={failed} "
                f"frames={output_frames} elapsed={elapsed:.1f}s",
                flush=True,
            )

    actual_output_names = {path.name for path in output.glob("*.pkl")}
    if failed:
        index.update({"status": "failed", "failures": failures[:20], "motion_files": len(actual_output_names)})
        _write_json_atomic(index_path, index)
        raise Mini3PklConversionError(f"Mini3 PKL conversion failed for {failed} source files; first error: {failures[0]}")
    if actual_output_names != expected_output_names:
        extra = sorted(actual_output_names - expected_output_names)[:10]
        missing = sorted(expected_output_names - actual_output_names)[:10]
        index.update({"status": "failed", "motion_files": len(actual_output_names)})
        _write_json_atomic(index_path, index)
        raise Mini3PklConversionError(f"Output directory has stale or missing motion files: extra={extra}, missing={missing}")

    elapsed = time.monotonic() - started
    index.update(
        {
            "status": "complete",
            "processed_source_files": len(paths),
            "converted_source_files": converted_sources,
            "skipped_short_files": skipped_short,
            "failed_source_files": 0,
            "motion_files": len(actual_output_names),
            "output_frames": output_frames,
            "output_duration_seconds": output_duration,
            "elapsed_seconds": elapsed,
        }
    )
    index.pop("failures", None)
    _write_json_atomic(index_path, index)
    manifest_config = _manifest_config(robot_config, output, dataset_name)
    _write_manifest(manifest, manifest_config, overwrite=overwrite)
    return DatasetConversionResult(
        source_files=len(paths),
        converted_source_files=converted_sources,
        reused_motion_files=reused,
        written_motion_files=written,
        skipped_short_files=skipped_short,
        failed_source_files=0,
        output_frames=output_frames,
        output_duration_seconds=output_duration,
        output_dir=output,
        manifest_path=manifest,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source flat PKL file or directory; directories are searched recursively.")
    parser.add_argument(
        "--output-dir",
        default="humanoidverse/data/mini3_pkl_ufo",
        help="Output directory containing one UFO PKL per training clip.",
    )
    parser.add_argument("--manifest", default="configs/data/mini3_pkl.yaml", help="Generated training manifest path.")
    parser.add_argument("--robot", default="configs/robots/mini3.yaml", help="Mini3 RobotSpec YAML path.")
    parser.add_argument("--dataset-name", default="mini3_pkl")
    parser.add_argument(
        "--quat-order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Component order stored in root_rot; current Mini3 PKLs use xyzw.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Optional override; by default use each PKL's fps field.")
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=None)
    parser.add_argument("--keep-short", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-clip-seconds", type=float, default=1.0)
    parser.add_argument("--quat-norm-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--joint-limit-tolerance-deg", type=float, default=0.01)
    parser.add_argument("--compression", type=int, default=3, help="joblib compression level in [0, 9].")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-files", type=int, default=None)
    selection.add_argument("--sample-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--exclude-mirrored", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-output", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args(argv)

    paths = discover_pkl_files(
        args.input,
        exclude_mirrored=bool(args.exclude_mirrored),
        max_files=args.max_files,
        sample_files=args.sample_files,
        seed=args.seed,
    )
    print(
        f"Mini3 flat PKL conversion: selected_files={len(paths)} output_dir={Path(args.output_dir).expanduser().resolve()} "
        f"root_rotation=direct:root_rot quat_order={args.quat_order} fps={'from-pkl' if args.fps is None else args.fps} "
        f"clip={args.clip_seconds:g}s stride={args.stride_seconds or args.clip_seconds:g}s",
        flush=True,
    )
    if args.dry_run:
        for path in paths[:10]:
            print(f"  {path}")
        if len(paths) > 10:
            print(f"  ... {len(paths) - 10} more files")
        return

    robot_spec = load_robot_spec(args.robot)
    result = convert_pkl_dataset(
        paths,
        args.output_dir,
        args.manifest,
        robot_spec,
        robot_config=args.robot,
        dataset_name=args.dataset_name,
        root_quat_order=args.quat_order,
        fps_override=args.fps,
        clip_seconds=args.clip_seconds,
        stride_seconds=args.stride_seconds,
        keep_short=bool(args.keep_short),
        min_clip_seconds=args.min_clip_seconds,
        quat_norm_tolerance=args.quat_norm_tolerance,
        joint_limit_tolerance_degrees=args.joint_limit_tolerance_deg,
        compression=args.compression,
        overwrite=bool(args.overwrite),
        verify_output=bool(args.verify_output),
        continue_on_error=bool(args.continue_on_error),
        log_every=args.log_every,
    )
    print(
        f"Finished Mini3 PKL conversion: sources={result.converted_source_files}/{result.source_files} "
        f"motions={result.written_motion_files + result.reused_motion_files} written={result.written_motion_files} "
        f"reused={result.reused_motion_files} skipped_short={result.skipped_short_files} "
        f"frames={result.output_frames} duration={result.output_duration_seconds:.3f}s\n"
        f"  output_dir={result.output_dir}\n  manifest={result.manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
