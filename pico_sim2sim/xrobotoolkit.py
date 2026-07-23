"""Load the repository-local XRoboToolkit native extension."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load() -> ModuleType:
    native_dir = Path(__file__).resolve().parent / "native"
    candidates = sorted(native_dir.glob("xrobotoolkit_sdk*.so"))
    if not candidates:
        raise ImportError(f"XRoboToolkit native extension is missing under {native_dir}. Run: bash pico_sim2sim/install.sh")
    spec = importlib.util.spec_from_file_location("xrobotoolkit_sdk", candidates[0])
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load XRoboToolkit extension: {candidates[0]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
