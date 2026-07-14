"""Convert retargeted Mini3 Euler/degree CSV files to RobotState NPZ files.

The source format handled here is:

    Frame,
    root_translateX, root_translateY, root_translateZ,
    root_rotateX, root_rotateY, root_rotateZ,
    <mini3_joint_name>_dof ...

Root translations are converted from centimeters to meters, intrinsic XYZ
Euler angles from degrees to xyzw quaternions, and joint angles from degrees to
radians. Files are processed one at a time so large source directories do not
need to fit in memory.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from humanoidverse.utils.motion_data.robot_state import RobotStateMotion, validate_robot_state_motion
from humanoidverse.utils.robot_spec import RobotSpec, load_robot_spec

FRAME_COLUMN = "Frame"
ROOT_POS_COLUMNS = ["root_translateX", "root_translateY", "root_translateZ"]
ROOT_EULER_COLUMNS = ["root_rotateX", "root_rotateY", "root_rotateZ"]


class Mini3CsvConversionError(ValueError):
    """Raised when a source CSV does not satisfy the Mini3 source contract."""


@dataclass(frozen=True)
class ConversionResult:
    source_path: Path
    output_path: Path
    frames: int
    clipped_joint_values: int
    skipped: bool = False


@dataclass(frozen=True)
class LoadedMini3CsvMotion:
    source_path: Path
    motion: RobotStateMotion
    frame: np.ndarray
    clipped_joint_values: int


def _read_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        try:
            header = [str(value).strip() for value in next(csv.reader(stream))]
        except StopIteration as exc:
            raise Mini3CsvConversionError(f"CSV file is empty: {path}") from exc
    if not header or not any(header):
        raise Mini3CsvConversionError(f"CSV header is empty: {path}")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise Mini3CsvConversionError(f"CSV file={path} has duplicate columns: {duplicates}")
    return header


def _load_numeric_rows(path: Path, expected_width: int) -> np.ndarray:
    try:
        values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    except ValueError as exc:
        raise Mini3CsvConversionError(f"CSV file={path} contains malformed or non-numeric rows: {exc}") from exc
    if values.size == 0 or values.shape[0] == 0:
        raise Mini3CsvConversionError(f"CSV file has no motion frames: {path}")
    if values.shape[1] != expected_width:
        raise Mini3CsvConversionError(f"CSV file={path} has {values.shape[1]} data columns, but its header has {expected_width}")
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))[0]
        raise Mini3CsvConversionError(f"CSV file={path} contains a non-finite value at data row={int(bad[0])}, column={int(bad[1])}")
    return values


def _required_column_indices(header: list[str], robot_spec: RobotSpec, path: Path) -> dict[str, int]:
    required = [
        FRAME_COLUMN,
        *ROOT_POS_COLUMNS,
        *ROOT_EULER_COLUMNS,
        *(f"{joint_name}_dof" for joint_name in robot_spec.control_joint_names),
    ]
    missing = [name for name in required if name not in header]
    if missing:
        raise Mini3CsvConversionError(f"CSV file={path} is missing required columns: {missing}")
    return {name: header.index(name) for name in required}


def _validate_frame_column(frame: np.ndarray, path: Path) -> np.ndarray:
    rounded = np.rint(frame)
    if not np.allclose(frame, rounded, atol=1.0e-6, rtol=0.0):
        index = int(np.flatnonzero(np.abs(frame - rounded) > 1.0e-6)[0])
        raise Mini3CsvConversionError(f"CSV file={path} has a non-integer Frame value at row={index}: {frame[index]}")
    frame_int = rounded.astype(np.int64)
    if frame_int.shape[0] > 1 and not np.all(np.diff(frame_int) == 1):
        index = int(np.flatnonzero(np.diff(frame_int) != 1)[0])
        raise Mini3CsvConversionError(
            f"CSV file={path} Frame must increase by one; rows {index}/{index + 1} contain {frame_int[index]}/{frame_int[index + 1]}"
        )
    return frame_int


def _continuous_xyzw_quaternions(euler_degrees: np.ndarray, euler_order: str, path: Path) -> np.ndarray:
    if len(euler_order) != 3 or not euler_order.isupper() or set(euler_order) != {"X", "Y", "Z"}:
        raise Mini3CsvConversionError(f"Euler order must be an intrinsic uppercase permutation of XYZ, got {euler_order!r} for {path}")
    try:
        quat = Rotation.from_euler(euler_order, euler_degrees, degrees=True).as_quat()
    except ValueError as exc:
        raise Mini3CsvConversionError(f"CSV file={path} has invalid Euler rotations: {exc}") from exc
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise Mini3CsvConversionError(f"CSV file={path} produced zero or non-finite root quaternions")
    quat = quat / norms

    # q and -q are the same orientation. Keep neighboring samples in the same
    # hemisphere so downstream angular-velocity estimates do not see sign jumps.
    if quat.shape[0] > 1:
        adjacent_sign = np.where(np.sum(quat[1:] * quat[:-1], axis=1) < 0.0, -1.0, 1.0)
        signs = np.concatenate([np.ones(1, dtype=np.float64), np.cumprod(adjacent_sign)])
        quat *= signs[:, None]
    return quat.astype(np.float32)


def _joint_limits(robot_spec: RobotSpec) -> tuple[np.ndarray, np.ndarray]:
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
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _validate_and_clip_joint_limits(
    dof_radians: np.ndarray,
    robot_spec: RobotSpec,
    path: Path,
    tolerance_degrees: float,
) -> tuple[np.ndarray, int]:
    if tolerance_degrees < 0.0 or not np.isfinite(tolerance_degrees):
        raise ValueError(f"joint_limit_tolerance_degrees must be finite and non-negative, got {tolerance_degrees}")
    lower, upper = _joint_limits(robot_spec)
    tolerance = np.deg2rad(float(tolerance_degrees))
    invalid = (dof_radians < lower[None, :] - tolerance) | (dof_radians > upper[None, :] + tolerance)
    if np.any(invalid):
        frame_idx, joint_idx = (int(value) for value in np.argwhere(invalid)[0])
        joint_name = robot_spec.control_joint_names[joint_idx]
        raise Mini3CsvConversionError(
            f"CSV file={path} joint={joint_name} at row={frame_idx} is outside the XML limit by more than "
            f"{tolerance_degrees} deg: value={np.rad2deg(dof_radians[frame_idx, joint_idx]):.8g} deg, "
            f"range=[{np.rad2deg(lower[joint_idx]):.8g}, {np.rad2deg(upper[joint_idx]):.8g}] deg"
        )
    needs_clip = (dof_radians < lower[None, :]) | (dof_radians > upper[None, :])
    clipped_count = int(np.count_nonzero(needs_clip))
    return np.clip(dof_radians, lower[None, :], upper[None, :]), clipped_count


def _verify_output(path: Path, expected_frames: int, expected_dofs: int, expected_fps: float) -> None:
    with np.load(path, allow_pickle=False) as data:
        missing = [name for name in ("root_pos", "root_quat", "dof_pos", "joint_names", "fps") if name not in data]
        if missing:
            raise Mini3CsvConversionError(f"Converted NPZ file={path} is missing fields: {missing}")
        expected_shapes = {
            "root_pos": (expected_frames, 3),
            "root_quat": (expected_frames, 4),
            "dof_pos": (expected_frames, expected_dofs),
        }
        for name, shape in expected_shapes.items():
            if data[name].shape != shape:
                raise Mini3CsvConversionError(f"Converted NPZ file={path} field={name} has shape {data[name].shape}, expected {shape}")
            if not np.isfinite(data[name]).all():
                raise Mini3CsvConversionError(f"Converted NPZ file={path} field={name} contains non-finite values")
        if not np.isclose(float(np.asarray(data["fps"]).reshape(-1)[0]), expected_fps):
            raise Mini3CsvConversionError(f"Converted NPZ file={path} has an unexpected fps")


def load_mini3_csv_motion(
    source_path: str | Path,
    robot_spec: RobotSpec,
    *,
    fps: float = 120.0,
    position_scale: float = 0.01,
    euler_order: str = "XYZ",
    joint_limit_tolerance_degrees: float = 0.01,
) -> LoadedMini3CsvMotion:
    """Load the confirmed raw Mini3 CSV format into normalized memory arrays."""

    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Mini3 source CSV does not exist: {source}")
    if source.suffix.lower() != ".csv":
        raise Mini3CsvConversionError(f"Mini3 source must be a .csv file: {source}")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    if not np.isfinite(position_scale) or position_scale <= 0.0:
        raise ValueError(f"position_scale must be finite and positive, got {position_scale}")

    header = _read_header(source)
    indices = _required_column_indices(header, robot_spec, source)
    raw = _load_numeric_rows(source, len(header))
    frame = _validate_frame_column(raw[:, indices[FRAME_COLUMN]], source)
    root_pos = raw[:, [indices[name] for name in ROOT_POS_COLUMNS]] * float(position_scale)
    root_euler = raw[:, [indices[name] for name in ROOT_EULER_COLUMNS]]
    root_quat = _continuous_xyzw_quaternions(root_euler, euler_order, source)
    dof_columns = [f"{joint_name}_dof" for joint_name in robot_spec.control_joint_names]
    dof_radians = np.deg2rad(raw[:, [indices[name] for name in dof_columns]])
    dof_radians, clipped_count = _validate_and_clip_joint_limits(
        dof_radians,
        robot_spec,
        source,
        tolerance_degrees=joint_limit_tolerance_degrees,
    )
    motion = validate_robot_state_motion(
        RobotStateMotion(
            motion_key=source.stem,
            root_pos=root_pos.astype(np.float32),
            root_quat=root_quat,
            dof_pos=dof_radians.astype(np.float32),
            fps=float(fps),
            joint_names=list(robot_spec.control_joint_names),
            source="mini3_raw_csv",
            metadata={
                "path": str(source),
                "reader": "mini3_raw_csv",
                "source_euler_order": f"intrinsic {euler_order}",
                "source_position_scale": float(position_scale),
            },
        ),
        robot_spec,
        source_name="mini3_raw_csv",
    )
    return LoadedMini3CsvMotion(
        source_path=source,
        motion=motion,
        frame=frame,
        clipped_joint_values=clipped_count,
    )


def convert_csv_file(
    source_path: str | Path,
    output_path: str | Path,
    robot_spec: RobotSpec,
    *,
    fps: float = 120.0,
    position_scale: float = 0.01,
    euler_order: str = "XYZ",
    joint_limit_tolerance_degrees: float = 0.01,
    compressed: bool = True,
    overwrite: bool = False,
    verify_output: bool = False,
) -> ConversionResult:
    """Convert one Mini3 source CSV into one normalized RobotState NPZ."""

    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Mini3 source CSV does not exist: {source}")
    if source.suffix.lower() != ".csv":
        raise Mini3CsvConversionError(f"Mini3 source must be a .csv file: {source}")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    if not np.isfinite(position_scale) or position_scale <= 0.0:
        raise ValueError(f"position_scale must be finite and positive, got {position_scale}")
    if output.exists() and not overwrite:
        return ConversionResult(source, output, frames=0, clipped_joint_values=0, skipped=True)

    loaded = load_mini3_csv_motion(
        source,
        robot_spec,
        fps=fps,
        position_scale=position_scale,
        euler_order=euler_order,
        joint_limit_tolerance_degrees=joint_limit_tolerance_degrees,
    )
    motion = loaded.motion
    frame = loaded.frame
    frame_count = int(motion.root_pos.shape[0])

    payload = {
        "root_pos": motion.root_pos,
        "root_quat": motion.root_quat,
        "dof_pos": motion.dof_pos,
        "joint_names": np.asarray(robot_spec.control_joint_names),
        "fps": np.asarray(float(fps), dtype=np.float32),
        "frame": frame,
        "time": (frame.astype(np.float64) / float(fps)).astype(np.float32),
        "motion_key": np.asarray(source.stem),
        "source_path": np.asarray(str(source)),
        "source_euler_order": np.asarray(f"intrinsic {euler_order}"),
        "source_position_scale": np.asarray(float(position_scale), dtype=np.float32),
        "robot_name": np.asarray(robot_spec.name),
        "root_quat_order": np.asarray("xyzw"),
        "coordinate_system": np.asarray(robot_spec.coordinate_system),
        "dof_unit": np.asarray("rad"),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    save = np.savez_compressed if compressed else np.savez
    try:
        with temp_path.open("wb") as stream:
            save(stream, **payload)
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)

    if verify_output:
        _verify_output(output, expected_frames=frame_count, expected_dofs=len(robot_spec.control_joint_names), expected_fps=fps)
    return ConversionResult(source, output, frames=frame_count, clipped_joint_values=loaded.clipped_joint_values)


def discover_csv_files(
    input_path: str | Path,
    *,
    exclude_mirrored: bool = False,
    max_files: int | None = None,
    sample_files: int | None = None,
    seed: int = 4728,
) -> list[Path]:
    """Discover and deterministically select source CSV files."""

    source = Path(input_path).expanduser().resolve()
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(source.rglob("*.csv"))
    else:
        raise FileNotFoundError(f"Mini3 CSV input does not exist: {source}")
    if exclude_mirrored:
        paths = [path for path in paths if not path.stem.endswith("_M")]
    if not paths:
        raise FileNotFoundError(f"No Mini3 CSV files found under: {source}")
    if max_files is not None and sample_files is not None:
        raise ValueError("max_files and sample_files are mutually exclusive")
    for name, value in (("max_files", max_files), ("sample_files", sample_files)):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if max_files is not None:
        paths = paths[:max_files]
    elif sample_files is not None and sample_files < len(paths):
        paths = sorted(random.Random(seed).sample(paths, sample_files))

    stems: dict[str, Path] = {}
    collisions: list[tuple[str, Path, Path]] = []
    for path in paths:
        previous = stems.get(path.stem)
        if previous is not None:
            collisions.append((path.stem, previous, path))
        stems[path.stem] = path
    if collisions:
        details = [f"{stem}: {first} / {second}" for stem, first, second in collisions[:10]]
        raise Mini3CsvConversionError(
            "Selected CSV files have duplicate stems and cannot be flattened into one NPZ directory: " + "; ".join(details)
        )
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source CSV file or directory; directories are searched recursively.")
    parser.add_argument("--output-dir", required=True, help="Flat output directory for RobotState NPZ files.")
    parser.add_argument("--robot", default="configs/robots/mini3.yaml", help="Mini3 RobotSpec YAML path.")
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--position-scale", type=float, default=0.01, help="Source translation multiplier; 0.01 converts cm to m.")
    parser.add_argument(
        "--euler-order",
        default="XYZ",
        help="Intrinsic Euler order. Uppercase XYZ is the confirmed source convention.",
    )
    parser.add_argument("--joint-limit-tolerance-deg", type=float, default=0.01)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-files", type=int, default=None, help="Convert the first N sorted files, useful for smoke checks.")
    selection.add_argument("--sample-files", type=int, default=None, help="Convert a deterministic random sample of N files.")
    parser.add_argument("--seed", type=int, default=4728)
    parser.add_argument("--exclude-mirrored", action="store_true", help="Skip source stems ending in _M.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--uncompressed", action="store_true", help="Use np.savez instead of np.savez_compressed.")
    parser.add_argument("--verify-output", action="store_true", help="Reopen and validate every written NPZ file.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args(argv)

    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    robot_spec = load_robot_spec(args.robot)
    paths = discover_csv_files(
        args.input,
        exclude_mirrored=bool(args.exclude_mirrored),
        max_files=args.max_files,
        sample_files=args.sample_files,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    print(
        f"Mini3 CSV conversion: selected_files={len(paths)} output_dir={output_dir} fps={args.fps:g} "
        f"euler=intrinsic-{args.euler_order} position_scale={args.position_scale:g}"
    )
    if args.dry_run:
        for path in paths[:10]:
            print(f"  {path} -> {output_dir / (path.stem + '.npz')}")
        if len(paths) > 10:
            print(f"  ... {len(paths) - 10} more files")
        return

    started = time.monotonic()
    converted = 0
    skipped = 0
    failed = 0
    total_frames = 0
    total_clipped = 0
    failures: list[str] = []
    for index, source in enumerate(paths, start=1):
        output = output_dir / f"{source.stem}.npz"
        try:
            result = convert_csv_file(
                source,
                output,
                robot_spec,
                fps=args.fps,
                position_scale=args.position_scale,
                euler_order=args.euler_order,
                joint_limit_tolerance_degrees=args.joint_limit_tolerance_deg,
                compressed=not args.uncompressed,
                overwrite=bool(args.overwrite),
                verify_output=bool(args.verify_output),
            )
            if result.skipped:
                skipped += 1
            else:
                converted += 1
                total_frames += result.frames
                total_clipped += result.clipped_joint_values
        except Exception as exc:
            failed += 1
            message = f"{source}: {type(exc).__name__}: {exc}"
            failures.append(message)
            if not args.continue_on_error:
                raise
        if index % args.log_every == 0 or index == len(paths):
            elapsed = time.monotonic() - started
            print(
                f"progress={index}/{len(paths)} converted={converted} skipped={skipped} failed={failed} "
                f"frames={total_frames} elapsed={elapsed:.1f}s"
            )

    elapsed = time.monotonic() - started
    print(
        f"Finished Mini3 CSV conversion: converted={converted} skipped={skipped} failed={failed} "
        f"frames={total_frames} duration={total_frames / args.fps:.3f}s "
        f"clipped_rounding_values={total_clipped} elapsed={elapsed:.1f}s"
    )
    if failures:
        print("Conversion failures:")
        for message in failures[:20]:
            print(f"  {message}")
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} more failures")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
