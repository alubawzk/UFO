"""Interactively play a flat Mini3 motion PKL with its stored root quaternion.

Expected fields are ``root_pos[T,3]``, ``root_rot[T,4]``,
``dof_pos[T,21]``, and ``fps``. The player does not reconstruct root rotation
from CSV Euler angles; it reads ``root_rot`` directly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from humanoidverse.tools.play_robot_state_npz import (
    apply_lie_down_reset_pose,
    build_qpos_trajectory,
    frame_bounds,
    load_playback_model,
    play_interactively,
    print_motion_summary,
    validate_forward_kinematics,
    validate_joint_limits,
)
from humanoidverse.utils.motion_data.robot_state import RobotStateMotion
from humanoidverse.utils.motion_data.robot_state_readers import read_robot_state_pkl
from humanoidverse.utils.robot_spec import RobotSpec, load_robot_spec


@dataclass(frozen=True)
class LoadedMini3PklMotion:
    source_path: Path
    motion: RobotStateMotion
    source_quat_order: str
    source_keys: tuple[str, ...]


def load_mini3_pkl_motion(
    path: str | Path,
    robot_spec: RobotSpec,
    *,
    quat_order: str = "xyzw",
    fps_override: float | None = None,
) -> LoadedMini3PklMotion:
    """Load one flat Mini3 PKL without deriving rotation from Euler angles."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Mini3 motion PKL does not exist: {resolved}")
    if resolved.suffix.lower() != ".pkl":
        raise ValueError(f"Expected a .pkl motion file, got: {resolved}")
    motions = read_robot_state_pkl(
        resolved,
        source_name="mini3_flat_pkl",
        robot_spec=robot_spec,
        fps=fps_override,
        root_quat_order=quat_order,
    )
    if len(motions) != 1:
        raise ValueError(f"Expected exactly one motion in Mini3 PKL={resolved}, got {len(motions)}")
    motion = next(iter(motions.values()))
    return LoadedMini3PklMotion(
        source_path=resolved,
        motion=motion,
        source_quat_order=quat_order,
        source_keys=tuple(str(key) for key in motion.metadata.get("source_keys", [])),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", required=True, help="One flat Mini3 motion PKL file.")
    parser.add_argument("--robot", default="configs/robots/mini3.yaml", help="Mini3 RobotSpec YAML path.")
    parser.add_argument("--scene", default=None, help="Optional playback XML. Defaults to scene.xml beside the robot XML.")
    parser.add_argument(
        "--quat-order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Component order of the stored root_rot field; this dataset uses xyzw.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Optional override; by default read fps directly from PKL.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive end frame; defaults to the full motion.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--follow-root", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument(
        "--preview-lie-down-reset",
        action="store_true",
        help="Apply the training reset's root rotation/height before playback (kinematic preview only).",
    )
    parser.add_argument("--lie-down-height", type=float, default=0.30, help="Root height used by --preview-lie-down-reset.")
    parser.add_argument(
        "--lie-down-angle-deg",
        type=float,
        choices=(-90.0, 90.0),
        default=-90.0,
        help="X-axis reset rotation used by --preview-lie-down-reset.",
    )
    parser.add_argument("--camera-distance", type=float, default=1.5)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--joint-limit-tolerance-deg", type=float, default=0.01)
    parser.add_argument("--validation-stride", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=120)
    parser.add_argument("--print-joint-ranges", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate PKL/qpos/FK without opening a window.")
    args = parser.parse_args(argv)

    if not np.isfinite(args.speed) or args.speed <= 0.0:
        parser.error("--speed must be finite and positive")
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")

    robot_spec = load_robot_spec(args.robot)
    loaded = load_mini3_pkl_motion(
        args.pkl,
        robot_spec,
        quat_order=args.quat_order,
        fps_override=args.fps,
    )
    motion = loaded.motion
    scene_path, model = load_playback_model(robot_spec, args.scene)
    start_frame, end_frame = frame_bounds(motion.root_pos.shape[0], args.start_frame, args.end_frame)
    validate_joint_limits(motion, robot_spec, tolerance=np.deg2rad(args.joint_limit_tolerance_deg))
    qpos = build_qpos_trajectory(motion, robot_spec, model)
    if args.preview_lie_down_reset:
        qpos = apply_lie_down_reset_pose(
            qpos,
            robot_spec,
            model,
            root_height=float(args.lie_down_height),
            angle_deg=float(args.lie_down_angle_deg),
        )
    validated_frames = validate_forward_kinematics(
        model,
        qpos,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=args.validation_stride,
    )
    print_motion_summary(
        loaded.source_path,
        scene_path,
        motion,
        model,
        source_kind="Mini3 flat PKL",
        start_frame=start_frame,
        end_frame=end_frame,
        validated_frames=validated_frames,
        print_joint_ranges=bool(args.print_joint_ranges),
    )
    print(
        f"  pkl_keys={list(loaded.source_keys)} root_rotation=direct:root_rot "
        f"source_quat_order={loaded.source_quat_order} fps_source={'override' if args.fps is not None else 'pkl'}"
    )
    if args.preview_lie_down_reset:
        print(
            f"  lie_down_reset_preview=true root_height={args.lie_down_height:g}m x_rotation={args.lie_down_angle_deg:g}deg mode=kinematic"
        )
    if args.check_only:
        print("Mini3 PKL check passed; root_rot was read directly and the viewer was not opened.")
        return

    try:
        play_interactively(
            model,
            qpos,
            motion,
            start_frame=start_frame,
            end_frame=end_frame,
            speed=float(args.speed),
            paused=bool(args.paused),
            loop=bool(args.loop),
            follow_root=bool(args.follow_root),
            show_collision=bool(args.show_collision),
            camera_distance=float(args.camera_distance),
            camera_azimuth=float(args.camera_azimuth),
            camera_elevation=float(args.camera_elevation),
            log_every=int(args.log_every),
        )
    except KeyboardInterrupt:
        print("Playback interrupted; viewer closed.")


if __name__ == "__main__":
    main()
