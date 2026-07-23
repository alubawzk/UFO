"""Check whether the integrated PICO-to-Mini3 runtime is ready."""

from __future__ import annotations

import argparse
import importlib.util
import platform
from pathlib import Path

from .joyin.retarget import DEFAULT_IK_CONFIG, DEFAULT_ROBOT_XML
from .smplx_model import DEFAULT_SMPLX_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, default=Path("runs/Revise_torque_limit"))
    args = parser.parse_args()
    model_folder = args.model_folder.expanduser().resolve()
    root = Path(__file__).resolve().parent
    checks = {
        "Python 3.10": platform.python_version_tuple()[:2] == ("3", "10"),
        "JOYIn Mini3 XML": DEFAULT_ROBOT_XML.is_file(),
        "JOYIn IK config": DEFAULT_IK_CONFIG.is_file(),
        "SMPL-X neutral model": DEFAULT_SMPLX_MODEL.is_file() and DEFAULT_SMPLX_MODEL.stat().st_size > 100_000_000,
        "mink": importlib.util.find_spec("mink") is not None,
        "qpsolvers": importlib.util.find_spec("qpsolvers") is not None,
        "smplx": importlib.util.find_spec("smplx") is not None,
        "zmq": importlib.util.find_spec("zmq") is not None,
        "XR PC service": Path("/opt/apps/roboticsservice/runService.sh").is_file(),
        "XR Python extension": any((root / "native").glob("xrobotoolkit_sdk*.so")),
        "XR native library": (root / "native" / "libPXREARobotSDK.so").is_file(),
        "UFO run config": (model_folder / "config.json").is_file(),
        "UFO model weights": (model_folder / "checkpoint" / "model" / "model.safetensors").is_file(),
    }
    width = max(len(name) for name in checks)
    for name, passed in checks.items():
        print(f"{name:<{width}}  {'OK' if passed else 'MISSING'}", flush=True)
    missing = [name for name, passed in checks.items() if not passed]
    if missing:
        raise SystemExit("\nMissing: " + ", ".join(missing) + "\nRun: bash pico_sim2sim/install.sh")
    print("\nPICO sim2sim PC runtime is ready.")


if __name__ == "__main__":
    main()
