"""Interactively play a raw pre-conversion Mini3 CSV in MuJoCo.

The confirmed source interpretation is applied in memory only: 120 FPS,
centimeters to meters, intrinsic XYZ Euler degrees to root quaternion, and
joint degrees to radians. No NPZ file is created.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from humanoidverse.tools.convert_mini3_csv import load_mini3_csv_motion
from humanoidverse.tools.play_robot_state_npz import (
    build_qpos_trajectory,
    frame_bounds,
    load_npz_motion,
    load_playback_model,
    play_interactively,
    print_motion_summary,
    validate_forward_kinematics,
    validate_joint_limits,
)
from humanoidverse.utils.motion_data.robot_state import RobotStateMotion
from humanoidverse.utils.robot_spec import RobotSpec, load_robot_spec


def compare_csv_and_npz(
    csv_motion: RobotStateMotion,
    npz_path: str | Path,
    robot_spec: RobotSpec,
    model: mujoco.MjModel,
    *,
    root_tolerance: float = 1.0e-6,
    dof_tolerance: float = 1.0e-6,
    quat_angle_tolerance_degrees: float = 1.0e-4,
) -> None:
    """Require a converted NPZ to match the in-memory raw CSV interpretation."""

    resolved_npz, npz_motion = load_npz_motion(npz_path, robot_spec)
    if csv_motion.root_pos.shape != npz_motion.root_pos.shape or csv_motion.dof_pos.shape != npz_motion.dof_pos.shape:
        raise ValueError(
            f"CSV/NPZ shape mismatch: root={csv_motion.root_pos.shape}/{npz_motion.root_pos.shape}, "
            f"dof={csv_motion.dof_pos.shape}/{npz_motion.dof_pos.shape}"
        )
    if not np.isclose(csv_motion.fps, npz_motion.fps, atol=1.0e-6, rtol=0.0):
        raise ValueError(f"CSV/NPZ fps mismatch: {csv_motion.fps:g}/{npz_motion.fps:g}")

    root_error = float(np.max(np.abs(csv_motion.root_pos - npz_motion.root_pos)))
    dof_error = float(np.max(np.abs(csv_motion.dof_pos - npz_motion.dof_pos)))
    csv_qpos = build_qpos_trajectory(csv_motion, robot_spec, model)
    npz_qpos = build_qpos_trajectory(npz_motion, robot_spec, model)
    free_joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, robot_spec.free_joint))
    free_qpos_addr = int(model.jnt_qposadr[free_joint_id])
    csv_quat = csv_qpos[:, free_qpos_addr + 3 : free_qpos_addr + 7]
    npz_quat = npz_qpos[:, free_qpos_addr + 3 : free_qpos_addr + 7]
    dots = np.clip(np.abs(np.sum(csv_quat * npz_quat, axis=1)), 0.0, 1.0)
    quat_angle_error_degrees = float(np.rad2deg(2.0 * np.max(np.arccos(dots))))

    print(
        f"CSV/NPZ comparison: npz={resolved_npz} root_max_abs={root_error:.3g}m "
        f"dof_max_abs={dof_error:.3g}rad quat_max_angle={quat_angle_error_degrees:.3g}deg"
    )
    if root_error > root_tolerance:
        raise ValueError(f"CSV/NPZ root position mismatch exceeds {root_tolerance:g} m")
    if dof_error > dof_tolerance:
        raise ValueError(f"CSV/NPZ DOF mismatch exceeds {dof_tolerance:g} rad")
    if quat_angle_error_degrees > quat_angle_tolerance_degrees:
        raise ValueError(f"CSV/NPZ root orientation mismatch exceeds {quat_angle_tolerance_degrees:g} deg")
    print("CSV/NPZ comparison passed.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="One raw pre-conversion Mini3 CSV file.")
    parser.add_argument("--robot", default="configs/robots/mini3.yaml", help="Mini3 RobotSpec YAML path.")
    parser.add_argument("--scene", default=None, help="Optional playback XML. Defaults to scene.xml beside the robot XML.")
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--position-scale", type=float, default=0.01, help="0.01 converts source centimeters to meters.")
    parser.add_argument("--euler-order", default="XYZ", help="Intrinsic uppercase Euler order; confirmed source value is XYZ.")
    parser.add_argument("--joint-limit-tolerance-deg", type=float, default=0.01)
    parser.add_argument("--compare-npz", default=None, help="Optional converted NPZ to compare frame by frame before playback.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive end frame; defaults to the full motion.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--follow-root", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument("--camera-distance", type=float, default=1.5)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--validation-stride", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=120)
    parser.add_argument("--print-joint-ranges", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate/compare without opening a window.")
    args = parser.parse_args(argv)

    if not np.isfinite(args.speed) or args.speed <= 0.0:
        parser.error("--speed must be finite and positive")
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")

    robot_spec = load_robot_spec(args.robot)
    loaded = load_mini3_csv_motion(
        args.csv,
        robot_spec,
        fps=args.fps,
        position_scale=args.position_scale,
        euler_order=args.euler_order,
        joint_limit_tolerance_degrees=args.joint_limit_tolerance_deg,
    )
    motion = loaded.motion
    scene_path, model = load_playback_model(robot_spec, args.scene)
    start_frame, end_frame = frame_bounds(motion.root_pos.shape[0], args.start_frame, args.end_frame)
    validate_joint_limits(motion, robot_spec, tolerance=np.deg2rad(args.joint_limit_tolerance_deg))
    qpos = build_qpos_trajectory(motion, robot_spec, model)
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
        source_kind="Raw Mini3 CSV",
        start_frame=start_frame,
        end_frame=end_frame,
        validated_frames=validated_frames,
        print_joint_ranges=bool(args.print_joint_ranges),
    )
    print(
        f"  source_frame=[{int(loaded.frame[0])}, {int(loaded.frame[-1])}] "
        f"interpretation=fps:{motion.fps:g},cm_to_m:{args.position_scale:g},euler:intrinsic-{args.euler_order},deg_to_rad "
        f"clipped_rounding_values={loaded.clipped_joint_values}"
    )
    if args.compare_npz:
        compare_csv_and_npz(motion, args.compare_npz, robot_spec, model)
    if args.check_only:
        print("Raw Mini3 CSV check passed; no NPZ was written and the viewer was not opened.")
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
