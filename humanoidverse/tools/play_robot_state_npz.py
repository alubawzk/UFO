"""Interactively play a RobotState NPZ file with its MuJoCo robot model.

The player writes each recorded root pose and control-joint pose directly into
MuJoCo qpos and calls ``mj_forward``. It is intentionally kinematic: gravity,
PD control, and contact dynamics do not modify the recorded trajectory.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from humanoidverse.utils.motion_data.robot_state import RobotStateMotion
from humanoidverse.utils.motion_data.robot_state_readers import read_robot_state_npz
from humanoidverse.utils.robot_spec import RobotSpec, load_robot_spec


@dataclass
class PlayerState:
    frame: int
    paused: bool
    loop: bool
    speed: float
    show_collision: bool
    quit: bool = False


def load_npz_motion(path: str | Path, robot_spec: RobotSpec) -> tuple[Path, RobotStateMotion]:
    """Load one NPZ and normalize its DOF order against the RobotSpec."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"RobotState NPZ does not exist: {resolved}")
    if resolved.suffix.lower() != ".npz":
        raise ValueError(f"Expected a .npz motion file, got: {resolved}")
    motions = read_robot_state_npz(resolved, source_name="npz_player", robot_spec=robot_spec)
    if len(motions) != 1:
        raise ValueError(f"Expected one motion in {resolved}, got {len(motions)}")
    return resolved, next(iter(motions.values()))


def load_playback_model(robot_spec: RobotSpec, scene_path: str | Path | None = None) -> tuple[Path, mujoco.MjModel]:
    """Load an explicit scene or the robot sibling scene.xml when available."""

    robot_xml = Path(robot_spec.xml_path).expanduser().resolve()
    if scene_path is None:
        sibling_scene = robot_xml.with_name("scene.xml")
        resolved = sibling_scene if sibling_scene.is_file() else robot_xml
    else:
        resolved = Path(scene_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"MuJoCo playback XML does not exist: {resolved}")
    model = mujoco.MjModel.from_xml_path(str(resolved))
    if model.nq != robot_spec.nq:
        raise ValueError(f"Playback model={resolved} has nq={model.nq}, RobotSpec expects nq={robot_spec.nq}")
    return resolved, model


def _joint_id(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
    if joint_id < 0:
        raise ValueError(f"Playback model is missing joint={joint_name!r}")
    return joint_id


def _root_quat_wxyz(root_quat: np.ndarray, root_quat_order: str) -> np.ndarray:
    quat = np.asarray(root_quat, dtype=np.float64)
    if root_quat_order == "xyzw":
        wxyz = quat[:, [3, 0, 1, 2]].copy()
    elif root_quat_order == "wxyz":
        wxyz = quat.copy()
    else:
        raise ValueError(f"Unsupported RobotSpec root_quat_order={root_quat_order!r}")
    norms = np.linalg.norm(wxyz, axis=1, keepdims=True)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise ValueError("Motion contains zero or non-finite root quaternions")
    return wxyz / norms


def validate_joint_limits(motion: RobotStateMotion, robot_spec: RobotSpec, *, tolerance: float = 1.0e-5) -> None:
    """Reject poses outside the XML joint ranges before opening the viewer."""

    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError(f"Joint-limit tolerance must be finite and non-negative, got {tolerance}")
    for dof_index, joint_name in enumerate(robot_spec.control_joint_names):
        joint_range = robot_spec.joint_ranges[joint_name]
        if joint_range is None:
            continue
        values = motion.dof_pos[:, dof_index]
        lower, upper = (float(value) for value in joint_range)
        invalid = np.flatnonzero((values < lower - tolerance) | (values > upper + tolerance))
        if invalid.size:
            frame = int(invalid[0])
            raise ValueError(
                f"Motion joint={joint_name} frame={frame} value={float(values[frame]):.8g} rad is outside "
                f"XML range=[{lower:.8g}, {upper:.8g}] rad (tolerance={tolerance:g})"
            )


def build_qpos_trajectory(motion: RobotStateMotion, robot_spec: RobotSpec, model: mujoco.MjModel) -> np.ndarray:
    """Map normalized RobotState arrays into the playback model's qpos layout."""

    if robot_spec.free_joint is None:
        raise ValueError(f"Robot={robot_spec.name} has no free joint for RobotState root poses")
    free_joint_id = _joint_id(model, robot_spec.free_joint)
    if int(model.jnt_type[free_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(f"Robot root joint={robot_spec.free_joint!r} is not a MuJoCo free joint")
    free_qpos_addr = int(model.jnt_qposadr[free_joint_id])

    frame_count = int(motion.root_pos.shape[0])
    qpos = np.repeat(np.asarray(model.qpos0, dtype=np.float64)[None, :], frame_count, axis=0)
    qpos[:, free_qpos_addr : free_qpos_addr + 3] = np.asarray(motion.root_pos, dtype=np.float64)
    qpos[:, free_qpos_addr + 3 : free_qpos_addr + 7] = _root_quat_wxyz(motion.root_quat, robot_spec.root_quat_order)

    if motion.dof_pos.shape != (frame_count, len(robot_spec.control_joint_names)):
        raise ValueError(f"Motion dof_pos shape={motion.dof_pos.shape}, expected ({frame_count}, {len(robot_spec.control_joint_names)})")
    for dof_index, joint_name in enumerate(robot_spec.control_joint_names):
        joint_id = _joint_id(model, joint_name)
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError(f"Controlled joint={joint_name!r} must be hinge/slide for scalar dof_pos playback")
        qpos[:, int(model.jnt_qposadr[joint_id])] = motion.dof_pos[:, dof_index]

    if not np.isfinite(qpos).all():
        raise ValueError("Mapped MuJoCo qpos trajectory contains non-finite values")
    return qpos


def apply_lie_down_reset_pose(
    qpos: np.ndarray,
    robot_spec: RobotSpec,
    model: mujoco.MjModel,
    *,
    root_height: float,
    angle_deg: float,
) -> np.ndarray:
    """Apply the training reset's root-height and X-axis rotation to qpos frames."""

    if not np.isfinite(root_height) or root_height <= 0.0:
        raise ValueError(f"Lie-down root height must be finite and positive, got {root_height}")
    if not np.isfinite(angle_deg):
        raise ValueError(f"Lie-down angle must be finite, got {angle_deg}")
    if robot_spec.free_joint is None:
        raise ValueError(f"Robot={robot_spec.name} has no free joint for a lie-down reset pose")

    free_joint_id = _joint_id(model, robot_spec.free_joint)
    free_qpos_addr = int(model.jnt_qposadr[free_joint_id])
    transformed = np.asarray(qpos, dtype=np.float64).copy()
    transformed[:, free_qpos_addr + 2] = float(root_height)

    angle_rad = np.deg2rad(float(angle_deg))
    extra = np.array([np.cos(angle_rad / 2.0), np.sin(angle_rad / 2.0), 0.0, 0.0], dtype=np.float64)
    original = transformed[:, free_qpos_addr + 3 : free_qpos_addr + 7].copy()
    extra_vec = extra[1:]
    original_vec = original[:, 1:]
    rotated = np.empty_like(original)
    rotated[:, 0] = extra[0] * original[:, 0] - original_vec @ extra_vec
    rotated[:, 1:] = (
        extra[0] * original_vec
        + original[:, :1] * extra_vec[None, :]
        + np.cross(np.broadcast_to(extra_vec, original_vec.shape), original_vec)
    )
    rotated /= np.linalg.norm(rotated, axis=1, keepdims=True)
    transformed[:, free_qpos_addr + 3 : free_qpos_addr + 7] = rotated
    return transformed


def validate_forward_kinematics(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    stride: int = 100,
) -> int:
    """Run mj_forward on sampled frames and reject non-finite world poses."""

    if stride <= 0:
        raise ValueError(f"Validation stride must be positive, got {stride}")
    sampled = list(range(start_frame, end_frame, stride))
    if not sampled or sampled[-1] != end_frame - 1:
        sampled.append(end_frame - 1)
    data = mujoco.MjData(model)
    for frame in sampled:
        data.qpos[:] = qpos[frame]
        data.qvel[:] = 0.0
        if model.nu:
            data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.xpos).all() or not np.isfinite(data.xquat).all() or not np.isfinite(data.geom_xpos).all():
            raise ValueError(f"MuJoCo forward kinematics produced non-finite body/geom poses at frame={frame}")
    return len(sampled)


def frame_bounds(frame_count: int, start_frame: int, end_frame: int | None) -> tuple[int, int]:
    if start_frame < 0 or start_frame >= frame_count:
        raise ValueError(f"--start-frame must be in [0, {frame_count - 1}], got {start_frame}")
    end = frame_count if end_frame is None else int(end_frame)
    if end <= start_frame or end > frame_count:
        raise ValueError(f"--end-frame must be in [{start_frame + 1}, {frame_count}], got {end}")
    return int(start_frame), end


def print_motion_summary(
    source_path: Path,
    scene_path: Path,
    motion: RobotStateMotion,
    model: mujoco.MjModel,
    *,
    source_kind: str,
    start_frame: int,
    end_frame: int,
    validated_frames: int,
    print_joint_ranges: bool,
) -> None:
    duration = (end_frame - start_frame) / motion.fps
    root_min = np.min(motion.root_pos[start_frame:end_frame], axis=0)
    root_max = np.max(motion.root_pos[start_frame:end_frame], axis=0)
    quat_norm = np.linalg.norm(motion.root_quat[start_frame:end_frame], axis=1)
    print(
        f"{source_kind} ready: motion={motion.motion_key} frames={motion.root_pos.shape[0]} "
        f"range=[{start_frame}, {end_frame}) fps={motion.fps:g} duration={duration:.3f}s"
    )
    print(f"  source={source_path}")
    print(f"  scene={scene_path} nq/nv/nu={model.nq}/{model.nv}/{model.nu}")
    print(f"  root_xyz_min={root_min.tolist()} root_xyz_max={root_max.tolist()}")
    print(f"  root_quat_norm=[{float(quat_norm.min()):.8f}, {float(quat_norm.max()):.8f}]")
    print(f"  dofs={motion.dof_pos.shape[1]} sampled_fk_frames={validated_frames} joint_limit_violations=0")
    if print_joint_ranges:
        for index, joint_name in enumerate(motion.joint_names or []):
            values = motion.dof_pos[start_frame:end_frame, index]
            print(f"  joint[{index:02d}] {joint_name}: [{float(values.min()): .6f}, {float(values.max()): .6f}] rad")


def play_interactively(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    motion: RobotStateMotion,
    *,
    start_frame: int,
    end_frame: int,
    speed: float,
    paused: bool,
    loop: bool,
    follow_root: bool,
    show_collision: bool,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
    log_every: int,
) -> None:
    """Open the passive MuJoCo viewer and play the selected frame range."""

    import mujoco.viewer

    state = PlayerState(
        frame=start_frame,
        paused=paused,
        loop=loop,
        speed=speed,
        show_collision=show_collision,
    )
    state_lock = threading.Lock()
    one_second = max(1, int(round(motion.fps)))

    # GLFW key codes. Letter/bracket codes are ASCII; arrows are GLFW constants.
    key_space, key_r, key_l, key_c, key_q = 32, 82, 76, 67, 81
    key_left, key_right, key_up, key_down = 263, 262, 265, 264
    key_left_bracket, key_right_bracket = 91, 93

    def key_callback(keycode: int) -> None:
        message: str | None = None
        with state_lock:
            if keycode == key_space:
                state.paused = not state.paused
                message = "paused" if state.paused else "playing"
            elif keycode == key_r:
                state.frame = start_frame
                message = f"restart frame={state.frame}"
            elif keycode == key_l:
                state.loop = not state.loop
                message = f"loop={state.loop}"
            elif keycode == key_c:
                state.show_collision = not state.show_collision
                message = f"show_collision={state.show_collision}"
            elif keycode == key_q:
                state.quit = True
            elif keycode in (key_left, key_right, key_up, key_down):
                delta = {
                    key_left: -1,
                    key_right: 1,
                    key_up: one_second,
                    key_down: -one_second,
                }[keycode]
                state.frame = min(max(state.frame + delta, start_frame), end_frame - 1)
                state.paused = True
                message = f"seek frame={state.frame} time={(state.frame - start_frame) / motion.fps:.3f}s"
            elif keycode == key_left_bracket:
                state.speed = max(0.0625, state.speed * 0.5)
                message = f"speed={state.speed:g}x"
            elif keycode == key_right_bracket:
                state.speed = min(16.0, state.speed * 2.0)
                message = f"speed={state.speed:g}x"
        if message:
            print(f"[player] {message}", flush=True)

    data = mujoco.MjData(model)
    print("Controls: Space pause | Left/Right 1 frame | Up/Down 1 second | R restart")
    print("          [/] speed | L loop | C collision geoms | Q or viewer close to quit")
    last_frame = -1
    last_speed = speed
    last_collision = not show_collision
    next_tick = time.perf_counter() + 1.0 / (motion.fps * speed)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = float(camera_distance)
        viewer.cam.azimuth = float(camera_azimuth)
        viewer.cam.elevation = float(camera_elevation)
        while viewer.is_running():
            with state_lock:
                frame = state.frame
                is_paused = state.paused
                current_speed = state.speed
                current_collision = state.show_collision
                should_quit = state.quit
            if should_quit:
                break
            if frame != last_frame or current_collision != last_collision:
                with viewer.lock():
                    data.qpos[:] = qpos[frame]
                    data.qvel[:] = 0.0
                    if model.nu:
                        data.ctrl[:] = 0.0
                    mujoco.mj_forward(model, data)
                    viewer.opt.geomgroup[1] = int(current_collision)
                    viewer.opt.geomgroup[2] = 1
                    if follow_root:
                        viewer.cam.lookat[:] = [
                            float(motion.root_pos[frame, 0]),
                            float(motion.root_pos[frame, 1]),
                            max(float(motion.root_pos[frame, 2]), 0.4),
                        ]
                viewer.sync()
                if log_every > 0 and (frame == start_frame or frame == end_frame - 1 or frame % log_every == 0):
                    print(
                        f"[player] frame={frame}/{end_frame - 1} time={(frame - start_frame) / motion.fps:.3f}s speed={current_speed:g}x",
                        flush=True,
                    )
                last_frame = frame
                last_collision = current_collision

            now = time.perf_counter()
            if is_paused:
                next_tick = now
                time.sleep(0.01)
                continue
            if current_speed != last_speed:
                next_tick = now
                last_speed = current_speed
            interval = 1.0 / (motion.fps * current_speed)
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.01))
                continue

            with state_lock:
                if state.frame + 1 >= end_frame:
                    if state.loop:
                        state.frame = start_frame
                    else:
                        state.frame = end_frame - 1
                        state.paused = True
                else:
                    state.frame += 1
            next_tick += interval
            if now - next_tick > 4.0 * interval:
                next_tick = now


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="One converted RobotState NPZ file.")
    parser.add_argument("--robot", default="configs/robots/mini3.yaml", help="RobotSpec YAML path.")
    parser.add_argument("--scene", default=None, help="Optional playback XML. Defaults to scene.xml beside the robot XML.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive end frame; defaults to the full motion.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--paused", action="store_true", help="Open the viewer paused at the start frame.")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--follow-root", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument("--camera-distance", type=float, default=1.5)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--joint-limit-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--validation-stride", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=120)
    parser.add_argument("--print-joint-ranges", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate NPZ/qpos/FK without opening a window.")
    args = parser.parse_args(argv)

    if not np.isfinite(args.speed) or args.speed <= 0.0:
        parser.error("--speed must be finite and positive")
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")

    robot_spec = load_robot_spec(args.robot)
    npz_path, motion = load_npz_motion(args.npz, robot_spec)
    scene_path, model = load_playback_model(robot_spec, args.scene)
    start_frame, end_frame = frame_bounds(motion.root_pos.shape[0], args.start_frame, args.end_frame)
    validate_joint_limits(motion, robot_spec, tolerance=args.joint_limit_tolerance)
    qpos = build_qpos_trajectory(motion, robot_spec, model)
    validated_frames = validate_forward_kinematics(
        model,
        qpos,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=args.validation_stride,
    )
    print_motion_summary(
        npz_path,
        scene_path,
        motion,
        model,
        source_kind="RobotState NPZ",
        start_frame=start_frame,
        end_frame=end_frame,
        validated_frames=validated_frames,
        print_joint_ranges=bool(args.print_joint_ranges),
    )
    if args.check_only:
        print("RobotState NPZ check passed; viewer was not opened.")
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
