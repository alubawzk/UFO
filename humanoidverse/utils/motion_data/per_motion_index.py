"""Shared parsing for authoritative per-motion directory indexes."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

PER_MOTION_DIRECTORY_INDEX = "_ufo_per_motion_index.json"
PER_MOTION_DIRECTORY_INDEX_V1 = "ufo_per_motion_directory_v1"
PER_MOTION_DIRECTORY_INDEX_V2 = "ufo_per_motion_directory_v2"
SUPPORTED_PER_MOTION_INDEX_FORMATS = {PER_MOTION_DIRECTORY_INDEX_V1, PER_MOTION_DIRECTORY_INDEX_V2}


def load_per_motion_directory_index(directory: str | Path) -> dict[str, Any] | None:
    """Load and validate the common index envelope when one is present."""

    root = Path(directory)
    index_path = root / PER_MOTION_DIRECTORY_INDEX
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid per-motion directory index: {index_path}") from exc
    index_format = index.get("format")
    if index_format not in SUPPORTED_PER_MOTION_INDEX_FORMATS:
        raise ValueError(f"Unsupported per-motion directory index format={index_format!r}: {index_path}")
    if index.get("status") != "complete":
        raise ValueError(f"Per-motion directory index must have status='complete', got {index.get('status')!r}: {index_path}")
    return index


def v2_indexed_motion_paths(
    directory: str | Path,
    index: dict[str, Any] | None = None,
) -> list[Path] | None:
    """Return v2 motion paths in authoritative source-local order.

    A v1 index returns ``None`` so callers can use their legacy flat-directory
    fallback. Extra files on disk are intentionally ignored for v2 indexes.
    """

    root = Path(directory)
    index = load_per_motion_directory_index(root) if index is None else index
    if index is None or index.get("format") == PER_MOTION_DIRECTORY_INDEX_V1:
        return None
    if index.get("format") != PER_MOTION_DIRECTORY_INDEX_V2:
        raise ValueError(f"Unsupported per-motion directory index format={index.get('format')!r}: {root}")

    motions = index.get("motions")
    indexed_count = len(motions) if isinstance(motions, list) else -1
    expected_count = int(index.get("motion_files", -1))
    if not isinstance(motions, list) or not motions or expected_count != indexed_count:
        raise ValueError(f"v2 per-motion index count mismatch: index={expected_count}, motions={indexed_count}: {root}")

    relative_paths: list[str] = []
    motion_keys: set[str] = set()
    for source_local_id, motion in enumerate(motions):
        if not isinstance(motion, dict):
            raise ValueError(f"v2 per-motion index record #{source_local_id} must be a mapping: {root}")
        if int(motion.get("source_local_id", -1)) != source_local_id:
            raise ValueError(f"v2 source_local_id must be contiguous at record #{source_local_id}: {root}")

        relative_path = str(motion.get("relative_path", ""))
        pure_path = PurePosixPath(relative_path)
        if not relative_path or pure_path.is_absolute() or ".." in pure_path.parts or pure_path.as_posix() != relative_path:
            raise ValueError(f"Invalid v2 relative_path={relative_path!r}: {root}")
        motion_key = str(motion.get("motion_key", ""))
        if not motion_key or motion_key in motion_keys:
            raise ValueError(f"Missing or duplicate v2 motion_key={motion_key!r}: {root}")
        if pure_path.suffix != ".pkl" or pure_path.stem != motion_key:
            raise ValueError(f"v2 relative_path={relative_path!r} must be a PKL whose stem matches motion_key={motion_key!r}")

        relative_paths.append(relative_path)
        motion_keys.add(motion_key)

    if relative_paths != sorted(relative_paths) or len(set(relative_paths)) != len(relative_paths):
        raise ValueError(f"v2 relative paths must be unique and sorted: {root}")
    paths = [root.joinpath(*PurePosixPath(relative_path).parts) for relative_path in relative_paths]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"v2 per-motion index references missing motion files: {missing[:10]}")
    return paths
