"""Offline-resample a per-motion UFO dataset to a fixed frame rate.

Example for the current Mini3 dataset::

    python -m humanoidverse.tools.resample_ufo_motion \
      --input humanoidverse/data/mini3_pkl_ufo \
      --output-dir humanoidverse/data/mini3_pkl_ufo_50fps \
      --target-fps 50 \
      --manifest configs/data/mini3_pkl_50fps.yaml \
      --robot-config configs/robots/mini3.yaml \
      --workers 8 --verify-output

The output remains one motion per PKL and can be consumed by the existing
``storage: per_motion_directory`` manifest path. Existing valid output files
are reused, so an interrupted conversion can be resumed with the same command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from omegaconf import OmegaConf

from humanoidverse.utils.motion_data.manifest import PER_MOTION_DIRECTORY_INDEX
from humanoidverse.utils.motion_data.resample import (
    LEGACY_EXPERT_CONSUMER_CONTRACT,
    RESAMPLING_ALGORITHM,
    RESAMPLING_SCHEMA_VERSION,
    resample_ufo_motion_record,
    target_frame_count,
)
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict

INDEX_FORMAT = "ufo_per_motion_directory_v2"
SUPPORTED_INPUT_INDEX_FORMATS = {"ufo_per_motion_directory_v1", INDEX_FORMAT}


class UfoResamplingError(ValueError):
    """Raised when a UFO dataset cannot be safely resampled."""


@dataclass(frozen=True)
class ResamplingResult:
    source_files: int
    written_motion_files: int
    reused_motion_files: int
    source_frames: int
    output_frames: int
    source_duration_seconds: float
    output_duration_seconds: float
    output_dir: Path
    manifest_path: Path | None


@dataclass(frozen=True)
class _FileResult:
    source_index: int
    metadata: dict[str, Any]
    reused: bool


def _load_and_validate_input_index(directory: Path, paths: list[Path]) -> dict[str, Any] | None:
    index_path = directory / PER_MOTION_DIRECTORY_INDEX
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UfoResamplingError(f"Input directory has an invalid {PER_MOTION_DIRECTORY_INDEX}: {index_path}") from exc
    if index.get("format") not in SUPPORTED_INPUT_INDEX_FORMATS:
        raise UfoResamplingError(f"Input index has unsupported format={index.get('format')!r}: {index_path}")
    if index.get("status") != "complete":
        raise UfoResamplingError(f"Input index must have status='complete', got {index.get('status')!r}: {index_path}")
    expected_count = int(index.get("motion_files", -1))
    if expected_count != len(paths):
        raise UfoResamplingError(f"Input directory count mismatch: index={expected_count}, files={len(paths)}")
    return index


def discover_ufo_motion_files(input_path: str | Path, *, max_files: int | None = None) -> list[Path]:
    source = Path(input_path).expanduser().resolve()
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(source.glob("*.pkl"))
        _load_and_validate_input_index(source, paths)
    else:
        raise FileNotFoundError(f"UFO input does not exist: {source}")
    if not paths:
        raise FileNotFoundError(f"No UFO PKL files found under: {source}")
    if max_files is not None:
        if max_files <= 0:
            raise ValueError(f"max_files must be positive, got {max_files}")
        paths = paths[:max_files]
    stems = [path.stem for path in paths]
    if len(set(stems)) != len(stems):
        raise UfoResamplingError("Selected UFO PKLs have duplicate file stems")
    return paths


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_digest(paths: list[Path], input_root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        try:
            identity = path.relative_to(input_root)
        except ValueError:
            identity = Path(path.name)
        stat = path.stat()
        digest.update(str(identity).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\n")
    index_path = input_root / PER_MOTION_DIRECTORY_INDEX
    if index_path.is_file():
        digest.update(hashlib.sha256(index_path.read_bytes()).digest())
    return digest.hexdigest()


def _motion_index_digest(motions: list[dict[str, Any]]) -> str:
    canonical = json.dumps(motions, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_single_motion(path: Path, *, label: str) -> tuple[str, dict[str, Any]]:
    payload = joblib.load(path)
    validated = validate_ufo_motion_dict(payload, f"{label}:{path.name}")
    if len(validated) != 1:
        raise UfoResamplingError(f"UFO file={path} must contain exactly one motion, got {len(validated)}")
    motion_key, record = next(iter(validated.items()))
    if motion_key != path.stem:
        raise UfoResamplingError(f"UFO file={path} motion key must match its filename stem, got {motion_key!r}")
    return motion_key, record


def _write_motion_file(path: Path, motion_key: str, record: dict[str, Any], *, compression: int) -> None:
    payload = validate_ufo_motion_dict({motion_key: record}, f"write:{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(payload, temporary, compress=compression)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _record_metadata(
    *,
    source_local_id: int,
    motion_key: str,
    output_path: Path,
    output_root: Path,
    source_content_sha256: str,
    source_record: dict[str, Any],
    output_record: dict[str, Any],
) -> dict[str, Any]:
    source_frames = int(np.asarray(source_record["root_trans_offset"]).shape[0])
    output_frames = int(np.asarray(output_record["root_trans_offset"]).shape[0])
    source_fps = float(np.asarray(source_record["fps"]).reshape(-1)[0])
    output_fps = float(np.asarray(output_record["fps"]).reshape(-1)[0])
    resampling = (output_record.get("metadata") or {}).get("resampling", {})
    return {
        "source_local_id": source_local_id,
        "motion_key": motion_key,
        "relative_path": output_path.relative_to(output_root).as_posix(),
        "fps": output_fps,
        "frame_count": output_frames,
        "motion_length": (output_frames - 1) / output_fps,
        "sample_coverage_seconds": output_frames / output_fps,
        "source_fps": source_fps,
        "source_frame_count": source_frames,
        "source_motion_length": (source_frames - 1) / source_fps,
        "source_content_sha256": source_content_sha256,
        "expert_control_dt_seconds": float(resampling["expert_control_dt_seconds"]),
        "legacy_expert_sample_count": int(resampling["legacy_expert_sample_count"]),
    }


def _resample_one(
    source_index: int,
    source_path: Path,
    output_path: Path,
    *,
    target_fps: float,
    compression: int,
    overwrite: bool,
    verify_output: bool,
) -> _FileResult:
    motion_key, source_record = _load_single_motion(source_path, label="source")
    source_content_sha256 = _sha256_file(source_path)
    source_frames = int(np.asarray(source_record["root_trans_offset"]).shape[0])
    source_fps = float(np.asarray(source_record["fps"]).reshape(-1)[0])
    expected_target_frames = target_frame_count(source_frames, source_fps, target_fps)
    reused = False
    if output_path.exists() and not overwrite:
        output_key, output_record = _load_single_motion(output_path, label="existing-output")
        if output_key != motion_key:
            raise UfoResamplingError(f"Existing output={output_path} contains motion {output_key!r}, expected {motion_key!r}")
        output_fps = float(np.asarray(output_record["fps"]).reshape(-1)[0])
        resampling = (output_record.get("metadata") or {}).get("resampling", {})
        output_frames = int(np.asarray(output_record["root_trans_offset"]).shape[0])
        reused = (
            np.isclose(output_fps, target_fps, rtol=0.0, atol=1.0e-12)
            and resampling.get("algorithm") == RESAMPLING_ALGORITHM
            and int(resampling.get("schema_version", -1)) == RESAMPLING_SCHEMA_VERSION
            and resampling.get("consumer_contract") == LEGACY_EXPERT_CONSUMER_CONTRACT
            and resampling.get("source_content_sha256") == source_content_sha256
            and int(resampling.get("source_frame_count", -1)) == source_frames
            and np.isclose(float(resampling.get("source_fps", np.nan)), source_fps, rtol=0.0, atol=1.0e-12)
            and int(resampling.get("target_frame_count", -1)) == expected_target_frames
            and output_frames == expected_target_frames
        )
    if not reused:
        output_record = resample_ufo_motion_record(source_record, target_fps, source_name=f"resample:{source_path.name}")
        output_record["metadata"]["resampling"]["source_content_sha256"] = source_content_sha256
        _write_motion_file(output_path, motion_key, output_record, compression=compression)
        if verify_output:
            output_key, output_record = _load_single_motion(output_path, label="verify-output")
            if output_key != motion_key:
                raise UfoResamplingError(f"Verified output={output_path} contains motion {output_key!r}, expected {motion_key!r}")

    metadata = _record_metadata(
        source_local_id=source_index,
        motion_key=motion_key,
        output_path=output_path,
        output_root=output_path.parent,
        source_content_sha256=source_content_sha256,
        source_record=source_record,
        output_record=output_record,
    )
    return _FileResult(source_index=source_index, metadata=metadata, reused=reused)


def _write_manifest(path: Path, *, robot_config: str | Path, output_dir: Path, dataset_name: str, overwrite: bool) -> None:
    config = {
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


def resample_ufo_dataset(
    paths: list[Path],
    output_dir: str | Path,
    *,
    target_fps: float = 50.0,
    compression: int = 3,
    workers: int = 1,
    overwrite: bool = False,
    verify_output: bool = False,
    continue_on_error: bool = False,
    log_every: int = 100,
    manifest_path: str | Path | None = None,
    robot_config: str | Path | None = None,
    dataset_name: str = "mini3_pkl_50fps",
) -> ResamplingResult:
    """Stream-resample UFO PKLs into a resumable per-motion directory."""

    if not paths:
        raise ValueError("paths must not be empty")
    target_fps = float(target_fps)
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError(f"target_fps must be finite and positive, got {target_fps}")
    if compression < 0 or compression > 9:
        raise ValueError(f"compression must be in [0, 9], got {compression}")
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    if log_every <= 0:
        raise ValueError(f"log_every must be positive, got {log_every}")
    if manifest_path is not None and robot_config is None:
        raise ValueError("robot_config is required when manifest_path is provided")

    resolved_paths = sorted((Path(path).expanduser().resolve() for path in paths), key=lambda path: path.as_posix())
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("paths must not contain duplicates")
    if len({path.parent for path in resolved_paths}) != 1:
        raise ValueError("All programmatic resampling inputs must come from the same directory")
    names = [path.name for path in resolved_paths]
    stems = [path.stem for path in resolved_paths]
    if len(set(names)) != len(names) or len(set(stems)) != len(stems):
        raise ValueError("Resampling inputs must have unique basenames and stems")
    input_root = resolved_paths[0].parent
    if (input_root / PER_MOTION_DIRECTORY_INDEX).is_file():
        input_files = sorted(input_root.glob("*.pkl"))
        _load_and_validate_input_index(input_root, input_files)
        input_names = {path.name for path in input_files}
        missing_from_input = [path.name for path in resolved_paths if path.name not in input_names]
        if missing_from_input:
            raise UfoResamplingError(f"Selected files are not declared by the input directory: {missing_from_input[:10]}")
    output = Path(output_dir).expanduser().resolve()
    if output == input_root:
        raise ValueError("output_dir must differ from the input directory; in-place resampling is not supported")
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / PER_MOTION_DIRECTORY_INDEX
    conversion_config = {
        "operation": "offline_ufo_resample",
        "algorithm": RESAMPLING_ALGORITHM,
        "resampling_schema_version": RESAMPLING_SCHEMA_VERSION,
        "consumer_contract": LEGACY_EXPERT_CONSUMER_CONTRACT,
        "source_path": _portable_path(input_root),
        "source_files": len(resolved_paths),
        "source_selection_sha256": _selection_digest(resolved_paths, input_root),
        "target_fps": target_fps,
        "endpoint_policy": "last target time must not exceed the final source-frame time",
        "compression": compression,
    }
    if index_path.exists():
        existing_index = json.loads(index_path.read_text())
        if existing_index.get("conversion") != conversion_config and not overwrite:
            raise UfoResamplingError(
                f"Output directory={output} was created with different options; use --overwrite or a new output directory"
            )
    elif any(output.glob("*.pkl")):
        raise UfoResamplingError(f"Output directory={output} contains PKLs but has no {PER_MOTION_DIRECTORY_INDEX}; use an empty directory")

    progress_index: dict[str, Any] = {
        "format": INDEX_FORMAT,
        "status": "in_progress",
        "conversion": conversion_config,
        "motion_files": 0,
    }
    _write_json_atomic(index_path, progress_index)

    started = time.monotonic()
    results: list[_FileResult | None] = [None] * len(resolved_paths)
    failures: list[str] = []
    processed = 0
    batch_size = max(1, workers * 4)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for batch_start in range(0, len(resolved_paths), batch_size):
            batch = list(enumerate(resolved_paths[batch_start : batch_start + batch_size], start=batch_start))
            future_to_source = {
                executor.submit(
                    _resample_one,
                    source_index,
                    source_path,
                    output / source_path.name,
                    target_fps=target_fps,
                    compression=compression,
                    overwrite=overwrite,
                    verify_output=verify_output,
                ): source_path
                for source_index, source_path in batch
            }
            for future in as_completed(future_to_source):
                source_path = future_to_source[future]
                try:
                    file_result = future.result()
                    results[file_result.source_index] = file_result
                except Exception as exc:
                    failures.append(f"{source_path}: {type(exc).__name__}: {exc}")
                    if not continue_on_error:
                        progress_index.update({"status": "failed", "failures": failures[:20], "processed_source_files": processed})
                        _write_json_atomic(index_path, progress_index)
                        raise
                finally:
                    processed += 1
                    if processed % log_every == 0 or processed == len(resolved_paths):
                        progress_index.update(
                            {
                                "processed_source_files": processed,
                                "motion_files": sum(item is not None for item in results),
                                "failed_source_files": len(failures),
                            }
                        )
                        _write_json_atomic(index_path, progress_index)
                        print(
                            f"progress={processed}/{len(resolved_paths)} completed={progress_index['motion_files']} "
                            f"failed={len(failures)} elapsed={time.monotonic() - started:.1f}s",
                            flush=True,
                        )

    if failures:
        progress_index.update({"status": "failed", "failures": failures[:20]})
        _write_json_atomic(index_path, progress_index)
        raise UfoResamplingError(f"UFO resampling failed for {len(failures)} files; first error: {failures[0]}")

    completed = [item for item in results if item is not None]
    if len(completed) != len(resolved_paths):
        raise RuntimeError(f"Internal error: completed {len(completed)} of {len(resolved_paths)} files")
    motions = [item.metadata for item in completed]
    expected_names = {path.name for path in resolved_paths}
    actual_names = {path.name for path in output.glob("*.pkl")}
    if actual_names != expected_names:
        extra = sorted(actual_names - expected_names)[:10]
        missing = sorted(expected_names - actual_names)[:10]
        raise UfoResamplingError(f"Output directory has stale or missing motion files: extra={extra}, missing={missing}")

    source_frames = sum(int(item["source_frame_count"]) for item in motions)
    output_frames = sum(int(item["frame_count"]) for item in motions)
    source_duration = sum(float(item["source_motion_length"]) for item in motions)
    output_duration = sum(float(item["motion_length"]) for item in motions)
    written = sum(not item.reused for item in completed)
    reused = sum(item.reused for item in completed)
    final_index = {
        "format": INDEX_FORMAT,
        "status": "complete",
        "conversion": conversion_config,
        "motion_files": len(motions),
        "source_frames": source_frames,
        "output_frames": output_frames,
        "source_motion_length_seconds": source_duration,
        "output_motion_length_seconds": output_duration,
        "output_sample_coverage_seconds": sum(float(item["sample_coverage_seconds"]) for item in motions),
        "duration_convention": "(frame_count - 1) / fps",
        "motion_index_sha256": _motion_index_digest(motions),
        "motions": motions,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json_atomic(index_path, final_index)

    manifest: Path | None = None
    if manifest_path is not None:
        manifest = Path(manifest_path).expanduser().resolve()
        _write_manifest(
            manifest,
            robot_config=robot_config,
            output_dir=output,
            dataset_name=dataset_name,
            overwrite=overwrite,
        )
    return ResamplingResult(
        source_files=len(resolved_paths),
        written_motion_files=written,
        reused_motion_files=reused,
        source_frames=source_frames,
        output_frames=output_frames,
        source_duration_seconds=source_duration,
        output_duration_seconds=output_duration,
        output_dir=output,
        manifest_path=manifest,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="One UFO PKL or a per-motion UFO directory.")
    parser.add_argument("--output-dir", required=True, help="Destination per-motion UFO directory; must differ from the input.")
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--compression", type=int, default=3, help="joblib compression level in [0, 9].")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-files", type=int, default=None, help="Process only the first N sorted files for a smoke test.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-output", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--manifest", default=None, help="Optional output training manifest YAML.")
    parser.add_argument("--robot-config", default="configs/robots/mini3.yaml")
    parser.add_argument("--dataset-name", default="mini3_pkl_50fps")
    args = parser.parse_args(argv)

    paths = discover_ufo_motion_files(args.input, max_files=args.max_files)
    print(
        f"UFO resampling: selected_files={len(paths)} target_fps={args.target_fps:g} workers={args.workers} "
        f"output_dir={Path(args.output_dir).expanduser().resolve()}",
        flush=True,
    )
    print(
        "Compatibility note: the legacy expert control-time sample count is preserved through resampling metadata, "
        "but MotionLib will recompute FK and velocities at the target FPS, so derived states are not numerically identical.",
        flush=True,
    )
    result = resample_ufo_dataset(
        paths,
        args.output_dir,
        target_fps=args.target_fps,
        compression=args.compression,
        workers=args.workers,
        overwrite=bool(args.overwrite),
        verify_output=bool(args.verify_output),
        continue_on_error=bool(args.continue_on_error),
        log_every=args.log_every,
        manifest_path=args.manifest,
        robot_config=args.robot_config if args.manifest is not None else None,
        dataset_name=args.dataset_name,
    )
    print(
        f"Finished UFO resampling: files={result.source_files} written={result.written_motion_files} reused={result.reused_motion_files} "
        f"frames={result.source_frames}->{result.output_frames} "
        f"duration={result.source_duration_seconds:.3f}s->{result.output_duration_seconds:.3f}s\n"
        f"  output_dir={result.output_dir}\n  manifest={result.manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
