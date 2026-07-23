"""Minimal Gear Sonic compatible PICO body-pose publisher.

This module contains the exact live pose path consumed by
``humanoidverse.mujoco_pico_teleop``.  It intentionally excludes Gear
Sonic's G1 planner, hand IK, and VR visualization code.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .smplx_model import DEFAULT_SMPLX_MODEL, NeutralSmplxBodyModel
from .sonic_protocol import pack_pose_message

SMPL_PARENTS_24 = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22],
    dtype=np.int32,
)
SMPL_BASE_ROTATION = Rotation.from_quat(np.full(4, 0.5), scalar_first=True)
Y_UP_TO_Z_UP = Rotation.from_rotvec(np.asarray([np.pi / 2.0, 0.0, 0.0]))
XRT_JOINT_FRAME_CORRECTION = Rotation.from_euler("y", 180.0, degrees=True)
DEFAULT_SERVICE_SCRIPT = Path("/opt/apps/roboticsservice/runService.sh")


def body_poses_to_sonic(
    body_poses: np.ndarray,
    smplx_model: NeutralSmplxBodyModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert XRT's 24 global body transforms into Sonic body fields.

    XRT rows are ``xyz + xyzw`` in the PICO/SMPL Y-up convention.  The
    returned values are ``smpl_pose[21,3]``, root-local
    ``smpl_joints[24,3]``, and ``body_quat_w[4]``.
    """

    poses = np.asarray(body_poses, dtype=np.float64)
    if poses.shape != (24, 7) or not np.all(np.isfinite(poses)):
        raise ValueError(f"XRoboToolkit body poses must have finite shape (24, 7), got {poses.shape}")

    global_rotations = Rotation.from_quat(poses[:, 3:7]) * XRT_JOINT_FRAME_CORRECTION
    local_rotations: list[Rotation] = []
    for index, rotation in enumerate(global_rotations):
        parent = int(SMPL_PARENTS_24[index])
        local_rotations.append(rotation if parent < 0 else global_rotations[parent].inv() * rotation)

    local_rotvec = np.stack([rotation.as_rotvec() for rotation in local_rotations])
    smpl_pose = local_rotvec[1:22].astype(np.float32)
    root_z_up = Y_UP_TO_Z_UP * local_rotations[0]
    body_quaternion = (root_z_up * SMPL_BASE_ROTATION.inv()).as_quat(scalar_first=True)

    body_output = smplx_model.forward(
        root_orient=root_z_up.as_rotvec(),
        pose_body=smpl_pose,
        trans=np.zeros(3),
    )
    joints_local = Rotation.from_quat(body_quaternion, scalar_first=True).inv().apply(body_output.sonic_positions)
    return smpl_pose, joints_local.astype(np.float32), body_quaternion.astype(np.float32)


def _normalized_quaternion_lerp(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    second_value = -second if float(np.dot(first, second)) < 0.0 else second
    value = (1.0 - alpha) * first + alpha * second_value
    norm = float(np.linalg.norm(value))
    return first.copy() if norm <= 1.0e-12 else value / norm


def _interpolate_axis_angle(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    first_quaternions = Rotation.from_rotvec(first.reshape(-1, 3)).as_quat()
    second_quaternions = Rotation.from_rotvec(second.reshape(-1, 3)).as_quat()
    output = np.stack(
        [
            _normalized_quaternion_lerp(first_quaternions[index], second_quaternions[index], alpha)
            for index in range(first_quaternions.shape[0])
        ]
    )
    return Rotation.from_quat(output).as_rotvec().reshape(first.shape)


class PoseBuffer:
    """Timestamp-resample XRT frames and build Sonic sliding-window payloads."""

    def __init__(self, *, window: int, target_fps: float):
        if window < 1 or target_fps <= 0.0:
            raise ValueError("window and target_fps must be positive")
        self.window = int(window)
        self.step_ns = int(round(1.0e9 / float(target_fps)))
        self.poses: deque[np.ndarray] = deque(maxlen=self.window)
        self.joints: deque[np.ndarray] = deque(maxlen=self.window)
        self.quaternions: deque[np.ndarray] = deque(maxlen=self.window)
        self.frame_indices: deque[int] = deque(maxlen=self.window)
        self.previous: tuple[int, np.ndarray, np.ndarray, np.ndarray] | None = None
        self.next_target_ns: int | None = None
        self.frame_index = 0

    def update(
        self,
        *,
        timestamp_ns: int,
        smpl_pose: np.ndarray,
        smpl_joints: np.ndarray,
        body_quaternion: np.ndarray,
        timestamp_realtime: float,
        timestamp_monotonic: float,
        device_dt: float,
        device_fps: float,
    ) -> dict[str, np.ndarray] | None:
        current = (
            int(timestamp_ns),
            np.asarray(smpl_pose),
            np.asarray(smpl_joints),
            np.asarray(body_quaternion),
        )
        if self.previous is None:
            self.previous = current
            self.next_target_ns = current[0]
            return None
        previous_stamp, previous_pose, previous_joints, previous_quaternion = self.previous
        if current[0] <= previous_stamp:
            return None
        if self.next_target_ns is None:
            self.next_target_ns = previous_stamp + self.step_ns
        self.next_target_ns = max(self.next_target_ns, previous_stamp)
        if self.next_target_ns > current[0]:
            self.previous = current
            return None

        alpha = float(np.clip((self.next_target_ns - previous_stamp) / (current[0] - previous_stamp), 0.0, 1.0))
        pose = _interpolate_axis_angle(previous_pose, current[1], alpha).astype(np.float32)
        joints = ((1.0 - alpha) * previous_joints + alpha * current[2]).astype(np.float32)
        quaternion = _normalized_quaternion_lerp(previous_quaternion, current[3], alpha).astype(np.float32)
        self.poses.append(pose)
        self.joints.append(joints)
        self.quaternions.append(quaternion)
        self.frame_indices.append(self.frame_index)
        self.frame_index += 1
        self.next_target_ns += self.step_ns
        self.previous = current
        if len(self.poses) < self.window:
            return None
        return {
            "smpl_pose": np.stack(self.poses),
            "smpl_joints": np.stack(self.joints),
            "body_quat_w": np.stack(self.quaternions),
            "frame_index": np.asarray(self.frame_indices, dtype=np.int64),
            "pico_dt": np.asarray([device_dt], dtype=np.float32),
            "pico_fps": np.asarray([device_fps], dtype=np.float32),
            "timestamp_realtime": np.asarray([timestamp_realtime], dtype=np.float64),
            "timestamp_monotonic": np.asarray([timestamp_monotonic], dtype=np.float64),
        }


def _load_xrobotoolkit() -> Any:
    try:
        import xrobotoolkit_sdk as xrt
    except ImportError:
        from .xrobotoolkit import load

        xrt = load()
    return xrt


def _start_pc_service(service_script: Path) -> subprocess.Popen[bytes] | None:
    if not service_script.is_file():
        print(f"[pico-sonic] PC service script not found, assuming it is already running: {service_script}")
        return None
    print(f"[pico-sonic] starting XRoboToolkit PC service: {service_script}", flush=True)
    return subprocess.Popen(["bash", str(service_script)])


def run(args: argparse.Namespace) -> None:
    import zmq

    smplx_model = NeutralSmplxBodyModel(args.smplx_model, device=args.smplx_device)
    print(f"[pico-sonic] smplx_model={smplx_model.model_file} device={smplx_model.device}", flush=True)
    xrt = _load_xrobotoolkit()
    service_process = None
    if args.start_service:
        service_process = _start_pc_service(args.service_script)
        if service_process is not None:
            time.sleep(args.service_startup_seconds)
    xrt.init()
    deadline = None if args.body_timeout_seconds <= 0.0 else time.monotonic() + args.body_timeout_seconds
    print("[pico-sonic] waiting for PICO body tracking data ...", flush=True)
    while not xrt.is_body_data_available():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for PICO body data")
        time.sleep(0.1)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{args.port}")
    pose_buffer = PoseBuffer(window=args.num_frames_to_send, target_fps=args.target_fps)
    if args.record_dir is not None:
        args.record_dir.mkdir(parents=True, exist_ok=True)
    previous_stamp: int | None = None
    fps_ema = 0.0
    sent = 0
    print(
        f"[pico-sonic] publishing tcp://0.0.0.0:{args.port} at {args.target_fps:g}Hz with {args.num_frames_to_send} frames/message",
        flush=True,
    )
    try:
        while True:
            stamp = int(xrt.get_time_stamp_ns())
            if previous_stamp is not None and stamp == previous_stamp:
                time.sleep(0.0005)
                continue
            device_dt = 0.0 if previous_stamp is None else (stamp - previous_stamp) * 1.0e-9
            if device_dt > 0.0:
                instant_fps = 1.0 / device_dt
                fps_ema = instant_fps if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * instant_fps
            previous_stamp = stamp
            pose, joints, quaternion = body_poses_to_sonic(np.asarray(xrt.get_body_joints_pose()), smplx_model)
            payload = pose_buffer.update(
                timestamp_ns=stamp,
                smpl_pose=pose,
                smpl_joints=joints,
                body_quaternion=quaternion,
                timestamp_realtime=time.time(),
                timestamp_monotonic=time.monotonic(),
                device_dt=device_dt,
                device_fps=fps_ema,
            )
            if payload is None:
                continue
            socket.send(pack_pose_message(payload))
            if args.record_dir is not None:
                np.savez_compressed(args.record_dir / f"pose_{sent:06d}.npz", **payload)
            sent += 1
            if sent == 1 or sent % args.log_every == 0:
                print(f"[pico-sonic] sent={sent} pico_fps={fps_ema:.1f}", flush=True)
    except KeyboardInterrupt:
        print("\n[pico-sonic] stopped", flush=True)
    finally:
        socket.close(linger=0)
        context.term()
        close = getattr(xrt, "close", None)
        if callable(close):
            close()
        # The PC service is shared system state; do not kill it on publisher exit.
        del service_process


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--target-fps", "--target_fps", dest="target_fps", type=float, default=50.0)
    parser.add_argument(
        "--num-frames-to-send",
        "--num_frames_to_send",
        dest="num_frames_to_send",
        type=int,
        default=5,
    )
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--body-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--smplx-model", type=Path, default=DEFAULT_SMPLX_MODEL)
    parser.add_argument("--smplx-device", default="cpu")
    parser.add_argument("--service-script", type=Path, default=DEFAULT_SERVICE_SCRIPT)
    parser.add_argument("--service-startup-seconds", type=float, default=2.0)
    parser.add_argument(
        "--start-service",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start /opt/apps/roboticsservice/runService.sh before opening the SDK",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if args.target_fps <= 0.0 or args.num_frames_to_send <= 0:
        parser.error("--target-fps and --num-frames-to-send must be positive")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")
    if args.body_timeout_seconds < 0.0 or args.service_startup_seconds < 0.0:
        parser.error("timeout/startup durations must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
