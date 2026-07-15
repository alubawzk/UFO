"""Manifest-based weighted multi-source motion data composition."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from loguru import logger
from omegaconf import OmegaConf

from humanoidverse.utils.motion_data.adapters import (
    dump_ufo_pkl,
    expand_motion_paths,
    load_motion_data_by_format,
    load_ufo_pkl,
)
from humanoidverse.utils.motion_data.clip import clip_ufo_motion_dict
from humanoidverse.utils.motion_data.per_motion_index import (
    PER_MOTION_DIRECTORY_INDEX,
    PER_MOTION_DIRECTORY_INDEX_V2,
    load_per_motion_directory_index,
    v2_indexed_motion_paths,
)
from humanoidverse.utils.motion_data.schema import format_fps_distribution, validate_ufo_motion_dict
from humanoidverse.utils.robot_spec import RobotSpec, load_robot_spec


@dataclass(frozen=True)
class ManifestMotionData:
    manifest_path: Path
    cache_dir: Path
    train_data_paths: list[str]
    train_data_weights: list[float]
    inference_paths: dict[str, str]
    robot_config_path: str | None = None
    robot_name: str | None = None
    robot_xml_path: str | None = None


def _safe_dataset_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not safe:
        raise ValueError(f"Invalid dataset name for manifest cache: {name!r}")
    return safe


def _load_manifest(manifest_path: str | Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Motion data manifest does not exist: {path}")
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Motion data manifest must be a mapping: {path}")
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or len(datasets) == 0:
        raise ValueError(f"Motion data manifest must contain a non-empty datasets list: {path}")
    normalized: list[dict[str, Any]] = []
    dataset_names: set[str] = set()
    for idx, item in enumerate(datasets):
        if not isinstance(item, dict):
            raise ValueError(f"Motion data manifest dataset #{idx} must be a mapping")
        for field in ("name", "format", "weight"):
            if field not in item:
                raise ValueError(f"Motion data manifest dataset #{idx} is missing required field '{field}'")
        dataset_name = str(item["name"])
        if dataset_name in dataset_names:
            raise ValueError(f"Motion data manifest dataset name must be unique: {dataset_name!r}")
        dataset_names.add(dataset_name)
        has_train_path = "train_path" in item
        has_auto_build = "source_path" in item and "auto_build" in item
        if not has_train_path and not has_auto_build:
            raise ValueError(f"Motion data manifest dataset #{idx} must define either train_path or source_path + auto_build")
        if "source_path" in item and "auto_build" not in item:
            raise ValueError(f"Motion data manifest dataset #{idx} uses source_path but is missing auto_build")
        normalized.append(dict(item))
    return path, config, normalized


def _normalize_weights(datasets: list[dict[str, Any]]) -> list[float]:
    weights: list[float] = []
    for item in datasets:
        try:
            weight = float(item["weight"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Dataset {item.get('name')} has non-numeric weight={item.get('weight')!r}") from exc
        if not math.isfinite(weight):
            raise ValueError(f"Dataset {item.get('name')} weight must be finite, got {weight}")
        if weight < 0.0:
            raise ValueError(f"Dataset {item.get('name')} weight must be non-negative, got {weight}")
        weights.append(weight)
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Motion data manifest weights must sum to a positive value")
    return [weight / total for weight in weights]


def _default_cache_dir(manifest_path: Path, cache_root: str | Path | None = None) -> Path:
    if cache_root is None:
        return (Path.cwd() / "cache" / "motion_data" / manifest_path.stem).resolve()
    return (Path(cache_root).expanduser() / manifest_path.stem).resolve()


def _cache_path(cache_dir: Path, dataset_name: str, split: str) -> Path:
    return cache_dir / f"{_safe_dataset_name(dataset_name)}_{split}_ufo.pkl"


def _resolve_existing_path(raw_path: str | Path, manifest_dir: Path) -> Path:
    expanded = Path(str(raw_path)).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    for candidate in (manifest_dir / expanded, Path.cwd() / expanded):
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / expanded).resolve()


def _manifest_needs_robot_spec(datasets: list[dict[str, Any]]) -> bool:
    return any(str(dataset["format"]).startswith("robot_state_") for dataset in datasets)


def _load_manifest_robot_spec(config: dict[str, Any], manifest_path: Path, datasets: list[dict[str, Any]]) -> RobotSpec | None:
    if not _manifest_needs_robot_spec(datasets):
        return None
    robot_config = config.get("robot_config")
    if not robot_config:
        raise ValueError("Motion data manifest uses robot_state_* data but is missing top-level robot_config")
    robot_config_path = _resolve_existing_path(robot_config, manifest_path.parent)
    spec = load_robot_spec(robot_config_path)
    logger.info(f"[motion-manifest] robot={spec.name} xml={spec.xml_path}")
    return spec


def _manifest_robot_config_path(config: dict[str, Any], manifest_path: Path) -> Path | None:
    robot_config = config.get("robot_config")
    if not robot_config:
        return None
    return _resolve_existing_path(robot_config, manifest_path.parent)


def prepare_manifest_robot_config_path(manifest_path: str | Path) -> str | None:
    manifest_path, config, _datasets = _load_manifest(manifest_path)
    robot_config_path = _manifest_robot_config_path(config, manifest_path)
    return str(robot_config_path) if robot_config_path is not None else None


def _source_path_for_split(dataset: dict[str, Any], split: str, *, fallback_to_train: bool) -> tuple[Any, bool]:
    if split == "train":
        return dataset["train_path"], False
    key = f"{split}_path"
    if dataset.get(key):
        return dataset[key], False
    if fallback_to_train:
        logger.warning(f"Dataset {dataset['name']} has no {key}; falling back to train_path")
        return dataset["train_path"], True
    raise ValueError(f"Dataset {dataset['name']} has no {key}")


def _load_cache(cache_path: Path, source_name: str) -> dict[str, Any]:
    data = joblib.load(cache_path)
    return validate_ufo_motion_dict(data, source_name)


def _prepare_per_motion_directory(path_spec: Any, manifest_dir: Path, source_name: str) -> str:
    if not isinstance(path_spec, (str, Path)):
        raise ValueError(f"Dataset {source_name} storage=per_motion_directory requires one directory path")
    directory = _resolve_existing_path(path_spec, manifest_dir)
    if not directory.is_dir():
        raise ValueError(f"Dataset {source_name} per-motion UFO path is not a directory: {directory}")
    index_path = directory / PER_MOTION_DIRECTORY_INDEX
    if not index_path.is_file():
        raise ValueError(
            f"Dataset {source_name} is missing {PER_MOTION_DIRECTORY_INDEX}; build it with humanoidverse.tools.convert_mini3_pkl"
        )
    index = load_per_motion_directory_index(directory)
    if index is None:
        raise ValueError(f"Dataset {source_name} is missing {PER_MOTION_DIRECTORY_INDEX}: {directory}")
    index_format = index.get("format")
    expected_count = int(index.get("motion_files", -1))
    if index_format == PER_MOTION_DIRECTORY_INDEX_V2:
        motions = index["motions"]
        files = v2_indexed_motion_paths(directory, index)
        assert files is not None
        for motion in motions:
            motion_key = str(motion["motion_key"])
            fps = float(motion.get("fps", math.nan))
            frame_count = int(motion.get("frame_count", 0))
            motion_length = float(motion.get("motion_length", math.nan))
            expected_length = (frame_count - 1) / fps if fps > 0.0 else math.nan
            if not math.isfinite(fps) or fps <= 0.0 or frame_count <= 0:
                raise ValueError(f"Dataset {source_name} has invalid fps/frame_count for motion_key={motion_key!r}")
            if not math.isclose(motion_length, expected_length, rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(f"Dataset {source_name} has invalid motion_length for motion_key={motion_key!r}")
    else:
        files = sorted(directory.glob("*.pkl"))
        if not files or expected_count != len(files):
            raise ValueError(f"Dataset {source_name} per-motion directory count mismatch: index={expected_count}, files={len(files)}")

    sample_indices = sorted({0, len(files) // 2, len(files) - 1})
    for sample_index in sample_indices:
        path = files[sample_index]
        if not path.is_file():
            raise ValueError(f"Dataset {source_name} indexed motion file does not exist: {path}")
        data = joblib.load(path)
        validated = validate_ufo_motion_dict(data, f"{source_name}:{path.name}")
        if list(validated) != [path.stem]:
            raise ValueError(f"Per-motion UFO file={path} must contain exactly one record whose key matches the filename stem")
    logger.info(f"[motion-manifest] dataset={source_name} storage=per_motion_directory path={directory} motions={len(files)} cache=lazy")
    return str(directory)


def _prepare_dataset_path(
    dataset: dict[str, Any],
    *,
    manifest_dir: Path,
    cache_dir: Path,
    split: str,
    rebuild_cache: bool,
    fallback_to_train: bool,
    robot_spec: RobotSpec | None = None,
) -> str:
    dataset_name = str(dataset["name"])
    fmt = str(dataset["format"])
    path_spec, used_train_fallback = _source_path_for_split(dataset, split, fallback_to_train=fallback_to_train)
    source_name = f"{dataset_name}:{split}"
    fps = dataset.get("fps")

    storage = dataset.get("storage")
    if storage is not None and storage != "per_motion_directory":
        raise ValueError(f"Dataset {dataset_name} has unsupported storage={storage!r}")
    if storage == "per_motion_directory":
        if fmt != "ufo_pkl":
            raise ValueError(f"Dataset {dataset_name} storage=per_motion_directory requires format=ufo_pkl")
        return _prepare_per_motion_directory(path_spec, manifest_dir, source_name)

    if fmt == "ufo_pkl":
        paths = expand_motion_paths(path_spec, base_dir=manifest_dir, suffix=".pkl")
        data = load_ufo_pkl([str(path) for path in paths], source_name=source_name, base_dir=manifest_dir)
        if len(paths) == 1:
            output_path = paths[0].resolve()
            cache_text = "original pkl"
        else:
            output_path = _cache_path(cache_dir, dataset_name, split)
            dump_ufo_pkl(data, output_path, source_name)
            cache_text = str(output_path)
    else:
        output_path = _cache_path(cache_dir, dataset_name, split)
        if output_path.exists() and not rebuild_cache:
            data = _load_cache(output_path, source_name)
            cache_text = f"{output_path} (reused)"
        else:
            data = load_motion_data_by_format(
                fmt,
                path_spec,
                source_name=source_name,
                base_dir=manifest_dir,
                fps=fps,
                robot_spec=robot_spec,
                columns=dataset.get("columns"),
            )
            dump_ufo_pkl(data, output_path, source_name)
            cache_text = str(output_path)

    logger.info(
        "[motion-manifest] dataset={name} split={split} format={fmt} train_fallback={fallback} "
        "path={path} motions={count} fps={fps_dist} cache={cache}".format(
            name=dataset_name,
            split=split,
            fmt=fmt,
            fallback=used_train_fallback,
            path=path_spec,
            count=len(data),
            fps_dist=format_fps_distribution(data, source_name),
            cache=cache_text,
        )
    )
    return str(output_path)


def _auto_build_cache_paths(cache_dir: Path, dataset_name: str) -> tuple[Path, Path]:
    safe_name = _safe_dataset_name(dataset_name)
    return cache_dir / f"{safe_name}_full_ufo.pkl", cache_dir / f"{safe_name}_train_near10s_ufo.pkl"


def _prepare_auto_build_dataset(
    dataset: dict[str, Any],
    *,
    manifest_dir: Path,
    cache_dir: Path,
    rebuild_cache: bool,
    robot_spec: RobotSpec | None,
) -> tuple[str, str]:
    dataset_name = str(dataset["name"])
    fmt = str(dataset["format"])
    source_path = dataset["source_path"]
    source_name = f"{dataset_name}:auto_build"
    full_path, train_path = _auto_build_cache_paths(cache_dir, dataset_name)

    if full_path.exists() and train_path.exists() and not rebuild_cache:
        full_data = _load_cache(full_path, f"{source_name}:full")
        train_data = _load_cache(train_path, f"{source_name}:train")
        cache_text = "reused"
    else:
        full_data = load_motion_data_by_format(
            fmt,
            source_path,
            source_name=f"{source_name}:full",
            base_dir=manifest_dir,
            fps=dataset.get("fps"),
            robot_spec=robot_spec,
            columns=dataset.get("columns"),
        )
        auto_build = dict(dataset.get("auto_build") or {})
        train_clip_seconds = float(auto_build.get("train_clip_seconds", 10.0))
        clip_stride_seconds = float(auto_build.get("clip_stride_seconds", train_clip_seconds))
        train_data = clip_ufo_motion_dict(
            full_data,
            clip_seconds=train_clip_seconds,
            stride_seconds=clip_stride_seconds,
            keep_short=bool(auto_build.get("keep_short", True)),
            min_clip_seconds=float(auto_build.get("min_clip_seconds", 1.0)),
            source_name=f"{source_name}:train",
        )
        dump_ufo_pkl(full_data, full_path, f"{source_name}:full")
        dump_ufo_pkl(train_data, train_path, f"{source_name}:train")
        cache_text = "built"

    logger.info(
        "[motion-manifest] dataset={name} format={fmt} source_path={source_path} source_weight={weight} "
        "full_motions={full_count} train_clips={train_count} full_fps={full_fps} train_fps={train_fps} "
        "full_cache={full_cache} train_cache={train_cache} cache={cache}".format(
            name=dataset_name,
            fmt=fmt,
            source_path=source_path,
            weight=dataset.get("weight"),
            full_count=len(full_data),
            train_count=len(train_data),
            full_fps=format_fps_distribution(full_data, f"{source_name}:full"),
            train_fps=format_fps_distribution(train_data, f"{source_name}:train"),
            full_cache=full_path,
            train_cache=train_path,
            cache=cache_text,
        )
    )
    return str(full_path), str(train_path)


def prepare_motion_manifest(
    manifest_path: str | Path,
    *,
    rebuild_cache: bool = False,
    cache_root: str | Path | None = None,
) -> ManifestMotionData:
    manifest_path, config, datasets = _load_manifest(manifest_path)
    cache_dir = _default_cache_dir(manifest_path, cache_root)
    weights = _normalize_weights(datasets)
    robot_spec = _load_manifest_robot_spec(config, manifest_path, datasets)
    robot_config_path = _manifest_robot_config_path(config, manifest_path)

    train_paths: list[str] = []
    inference_paths: dict[str, str] = {}
    for dataset, weight in zip(datasets, weights):
        if "source_path" in dataset:
            full_path, train_path = _prepare_auto_build_dataset(
                dataset,
                manifest_dir=manifest_path.parent,
                cache_dir=cache_dir,
                rebuild_cache=rebuild_cache,
                robot_spec=robot_spec,
            )
            inference_paths[str(dataset["name"])] = full_path
        else:
            train_path = _prepare_dataset_path(
                dataset,
                manifest_dir=manifest_path.parent,
                cache_dir=cache_dir,
                split="train",
                rebuild_cache=rebuild_cache,
                fallback_to_train=False,
                robot_spec=robot_spec,
            )
            if dataset.get("inference_path"):
                inference_paths[str(dataset["name"])] = str(dataset["inference_path"])
        train_paths.append(train_path)
        logger.info(f"[motion-manifest] dataset={dataset['name']} normalized_weight={weight:.6f}")

    return ManifestMotionData(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        train_data_paths=train_paths,
        train_data_weights=weights,
        inference_paths=inference_paths,
        robot_config_path=str(robot_config_path) if robot_config_path is not None else None,
        robot_name=robot_spec.name if robot_spec is not None else None,
        robot_xml_path=robot_spec.xml_path if robot_spec is not None else None,
    )


def prepare_manifest_dataset_path(
    manifest_path: str | Path,
    dataset_name: str,
    *,
    split: str = "inference",
    rebuild_cache: bool = False,
    cache_root: str | Path | None = None,
) -> str:
    manifest_path, config, datasets = _load_manifest(manifest_path)
    cache_dir = _default_cache_dir(manifest_path, cache_root)
    robot_spec = _load_manifest_robot_spec(config, manifest_path, datasets)
    for dataset in datasets:
        if str(dataset["name"]) == dataset_name:
            if "source_path" in dataset:
                full_path, train_path = _prepare_auto_build_dataset(
                    dataset,
                    manifest_dir=manifest_path.parent,
                    cache_dir=cache_dir,
                    rebuild_cache=rebuild_cache,
                    robot_spec=robot_spec,
                )
                return train_path if split == "train" else full_path
            return _prepare_dataset_path(
                dataset,
                manifest_dir=manifest_path.parent,
                cache_dir=cache_dir,
                split=split,
                rebuild_cache=rebuild_cache,
                fallback_to_train=True,
                robot_spec=robot_spec,
            )
    names = [str(dataset["name"]) for dataset in datasets]
    raise ValueError(f"Dataset '{dataset_name}' was not found in motion manifest {manifest_path}. Available datasets: {names}")
