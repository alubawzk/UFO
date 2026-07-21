"""Mini3 motor-side dynamics used by the pure MuJoCo sim-to-sim runner.

The constants and torque pipeline in this module mirror
``mini3_lab/robolab/scripts/mujoco/sim2sim_mini3_bm.py``.  Serial joints are
modelled directly in joint/motor space.  The four serial ankle coordinates are
converted to the Mini3 parallel-motor space before applying the current-loop
response, torque-speed envelope, and KT calibration, then converted back with
the ankle Jacobian transpose.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

MOTOR_SPECS: dict[str, dict[str, float]] = {
    "4340p": {
        "rated_torque": 9.0,
        "peak_torque": 27.0,
        "rated_speed_rpm": 36.0,
        "no_load_speed_rpm": 60.0,
    },
    "4310p": {
        "rated_torque": 3.5,
        "peak_torque": 12.5,
        "rated_speed_rpm": 120.0,
        "no_load_speed_rpm": 450.0,
    },
}

KT_OUTPUT_TABLES: dict[str, dict[str, tuple[float, ...]]] = {
    "4310p": {
        "feedback_tau_nm": (1.2, 2.35, 4.7, 7.12, 9.9, 13.5),
        "actual_tau_nm": (1.0, 2.0, 4.0, 6.0, 8.0, 10.0),
    },
    "4340p": {
        "feedback_tau_nm": (1.558, 3.158, 5.477, 8.324, 10.55, 13.121, 15.733, 18.509, 21.34, 24.786, 27.576),
        "actual_tau_nm": (1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0),
    },
}

ANKLE_PARAMS: dict[str, float] = {
    "l": 26.0,
    "lm": 30.0,
    "hl": 89.4,
    "hr": 148.3,
    "z0": 0.0,
    "d": 22.0,
    "df": 14.0,
    "zl": 89.0,
    "zr": 148.0,
}

ANKLE_JOINT_NAMES = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)


def mini3_motor_type(joint_name: str) -> str:
    """Return the physical Mini3 motor family used by a control joint."""

    if any(token in joint_name for token in ("hip_", "knee_", "waist_")):
        return "4340p"
    if any(token in joint_name for token in ("ankle_", "shoulder_", "elbow_")):
        return "4310p"
    raise ValueError(f"No Mini3 motor type mapping for joint {joint_name!r}")


class DelayLine:
    """Single-environment fractional-step delay with filled warm-up history."""

    def __init__(self, max_delay_steps: int, sample_shape: tuple[int, ...], dtype: Any = np.float64):
        self.max_delay_steps = int(max(0, max_delay_steps))
        self.sample_shape = tuple(sample_shape)
        self.dtype = dtype
        self.frames: deque[np.ndarray] = deque(maxlen=self.max_delay_steps + 1)

    def reset(self) -> None:
        self.frames.clear()

    def append(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=self.dtype).copy()
        if value.shape != self.sample_shape:
            raise ValueError(f"DelayLine expected shape {self.sample_shape}, got {value.shape}")
        if not self.frames:
            for _ in range(self.max_delay_steps + 1):
                self.frames.append(value.copy())
        else:
            self.frames.append(value)

    def read(self, delay_steps: int) -> np.ndarray:
        if not self.frames:
            return np.zeros(self.sample_shape, dtype=self.dtype)
        delay_steps = int(np.clip(delay_steps, 0, len(self.frames) - 1))
        return self.frames[-1 - delay_steps]

    def read_interpolated(self, delay_steps: float) -> np.ndarray:
        delay_steps = max(0.0, float(delay_steps))
        lower = int(math.floor(delay_steps))
        alpha = delay_steps - lower
        return ((1.0 - alpha) * self.read(lower) + alpha * self.read(lower + 1)).copy()


class TorqueCurrentLoopResponse:
    """Equivalent PI current loop, first-order torque plant, and output delay."""

    def __init__(
        self,
        num_channels: int,
        dt: float,
        *,
        kp: float = 0.0,
        ki: float = 90.6769527429,
        plant_tau_s: float = 0.00393417593548,
        delay_steps: float = 1.0,
    ):
        if num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if kp < 0.0 or ki < 0.0:
            raise ValueError(f"Current-loop gains must be nonnegative, got kp={kp}, ki={ki}")
        if plant_tau_s <= 0.0:
            raise ValueError(f"plant_tau_s must be positive, got {plant_tau_s}")
        if delay_steps < 0.0:
            raise ValueError(f"delay_steps must be nonnegative, got {delay_steps}")
        self.num_channels = int(num_channels)
        self.dt = float(dt)
        self.kp = float(kp)
        self.ki = float(ki)
        self.plant_tau_s = float(plant_tau_s)
        self.delay_steps = float(delay_steps)
        self.alpha = min(1.0, self.dt / self.plant_tau_s)
        self.tau_raw = np.zeros(self.num_channels, dtype=np.float64)
        self.integral = np.zeros(self.num_channels, dtype=np.float64)
        self.delay = DelayLine(math.ceil(self.delay_steps), (self.num_channels,))

    def reset(self) -> None:
        self.tau_raw.fill(0.0)
        self.integral.fill(0.0)
        self.delay.reset()

    def compute(self, torque_command: np.ndarray) -> np.ndarray:
        torque_command = np.nan_to_num(np.asarray(torque_command, dtype=np.float64))
        if torque_command.shape != (self.num_channels,):
            raise ValueError(f"Expected torque command shape {(self.num_channels,)}, got {torque_command.shape}")
        self.tau_raw[:] = np.nan_to_num(self.tau_raw)
        self.integral[:] = np.nan_to_num(self.integral)
        error = torque_command - self.tau_raw
        self.integral += error * self.dt
        current_loop_output = self.kp * error + self.ki * self.integral
        self.tau_raw += self.alpha * (current_loop_output - self.tau_raw)
        self.tau_raw[:] = np.nan_to_num(self.tau_raw)
        self.delay.append(self.tau_raw)
        return self.delay.read_interpolated(self.delay_steps)


class MotorTnLimit:
    """Four-quadrant motor torque-speed parallelogram limiter."""

    def __init__(
        self,
        *,
        name: str,
        rated_torque: float,
        peak_torque: float,
        rated_speed_rpm: float,
        no_load_speed_rpm: float,
        peak_speed_rpm: float | None = None,
    ):
        self.name = str(name)
        self.rated_torque = float(rated_torque)
        self.peak_torque = float(peak_torque)
        self.rated_speed_rpm = float(rated_speed_rpm)
        self.no_load_speed_rpm = float(no_load_speed_rpm)
        self.peak_speed_rpm = self.rated_speed_rpm if peak_speed_rpm is None else float(peak_speed_rpm)
        if self.rated_torque <= 0.0 or self.peak_torque < self.rated_torque:
            raise ValueError(f"{self.name}: invalid rated/peak torque")
        if self.rated_speed_rpm <= 0.0 or self.no_load_speed_rpm <= self.peak_speed_rpm:
            raise ValueError(f"{self.name}: invalid rated/peak/no-load speed")
        self.speed_drop_per_nm = (self.no_load_speed_rpm - self.peak_speed_rpm) / self.peak_torque

    def bounds(self, speed_rad_s: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        speed_rpm = np.asarray(speed_rad_s, dtype=np.float64) * 60.0 / (2.0 * math.pi)
        lower = np.maximum(-self.peak_torque, (-self.no_load_speed_rpm - speed_rpm) / self.speed_drop_per_nm)
        upper = np.minimum(self.peak_torque, (self.no_load_speed_rpm - speed_rpm) / self.speed_drop_per_nm)
        empty = lower > upper
        fallback = np.where(speed_rpm >= 0.0, lower, upper)
        return np.where(empty, fallback, lower), np.where(empty, fallback, upper)

    def clip(self, torque_command: np.ndarray | float, speed_rad_s: np.ndarray | float) -> np.ndarray:
        lower, upper = self.bounds(speed_rad_s)
        return np.clip(np.asarray(torque_command, dtype=np.float64), lower, upper)

    def summary(self) -> str:
        return (
            f"{self.name}: rated={self.rated_torque:g}Nm@{self.rated_speed_rpm:g}rpm "
            f"peak={self.peak_torque:g}Nm no-load={self.no_load_speed_rpm:g}rpm"
        )


class MotorKtOutputModel:
    """Map KT-feedback torque to calibrated measured output torque."""

    def __init__(self, *, name: str, feedback_tau_nm: tuple[float, ...], actual_tau_nm: tuple[float, ...]):
        self.name = str(name)
        feedback = np.asarray(feedback_tau_nm, dtype=np.float64)
        actual = np.asarray(actual_tau_nm, dtype=np.float64)
        if feedback.shape != actual.shape or feedback.ndim != 1 or feedback.size < 2:
            raise ValueError(f"{self.name}: invalid KT table shape")
        if feedback[0] > 0.0:
            feedback = np.concatenate(([0.0], feedback))
            actual = np.concatenate(([0.0], actual))
        if np.any(feedback < 0.0) or np.any(actual < 0.0):
            raise ValueError(f"{self.name}: KT table must contain nonnegative magnitudes")
        if np.any(np.diff(feedback) <= 0.0) or np.any(np.diff(actual) < 0.0):
            raise ValueError(f"{self.name}: KT table must be monotonic")
        self.feedback_tau_nm = feedback
        self.actual_tau_nm = actual
        self.max_feedback_tau_nm = float(feedback[-1])
        self.warned_clamp = False

    def map(self, feedback_torque: np.ndarray | float) -> np.ndarray:
        feedback_torque = np.asarray(feedback_torque, dtype=np.float64)
        magnitude = np.abs(feedback_torque)
        if np.any(magnitude > self.max_feedback_tau_nm) and not self.warned_clamp:
            print(
                f"[sim2sim] KT model {self.name} clamps |tau|={float(magnitude.max()):.3f}Nm "
                f"to calibrated maximum {self.max_feedback_tau_nm:.3f}Nm",
                flush=True,
            )
            self.warned_clamp = True
        output = np.interp(
            np.minimum(magnitude, self.max_feedback_tau_nm),
            self.feedback_tau_nm,
            self.actual_tau_nm,
        )
        return np.sign(feedback_torque) * output

    def summary(self) -> str:
        return f"{self.name}: feedback=0..{self.max_feedback_tau_nm:g}Nm -> actual=0..{float(self.actual_tau_nm[-1]):g}Nm"


def ankle_ik(
    d: float,
    df: float,
    zl: float,
    zr: float,
    link_l: float,
    lm: float,
    hl: float,
    hr: float,
    z0: float,
    roll: float,
    pitch: float,
) -> tuple[float, float]:
    """Return ``(tMR, tML)`` for the Mini3 parallel ankle."""

    cx, cy = math.cos(roll), math.cos(pitch)
    sx, sy = math.sin(roll), math.sin(pitch)
    eps = 1.0e-12
    al = -2.0 * d * (df * cx - z0 * sx)
    bl = 2.0 * d * (zl + link_l * sy - df * sx * cy - z0 * cx * cy)
    cl = (
        d**2
        + (lm - link_l * cy - df * sx * sy - z0 * cx * sy) ** 2
        + (df * cx - z0 * sx) ** 2
        + (zl + link_l * sy - df * sx * cy - z0 * cx * cy) ** 2
    )
    dl = hl**2 - cl
    rl = math.sqrt(al**2 + bl**2)
    ar = -2.0 * d * (df * cx + z0 * sx)
    br = -2.0 * d * (zr + link_l * sy + df * sx * cy - z0 * cx * cy)
    cr = (
        d**2
        + (lm - link_l * cy + df * sx * sy - z0 * cx * sy) ** 2
        + (df * cx + z0 * sx) ** 2
        + (zr + link_l * sy + df * sx * cy - z0 * cx * cy) ** 2
    )
    dr = hr**2 - cr
    rr = math.sqrt(ar**2 + br**2)
    tml = 0.0
    if not (rl + eps < abs(dl) or rl <= eps):
        tml = math.asin(np.clip(dl / max(rl, eps), -1.0, 1.0)) - math.atan2(al, bl)
    tmr = 0.0
    if not (rr + eps < abs(dr) or rr <= eps):
        tmr = math.pi - math.asin(np.clip(dr / max(rr, eps), -1.0, 1.0)) - math.atan2(ar, br) - 2.0 * math.pi
    return tmr, tml


def ankle_jacobian(*args: float, h: float = 1.0e-6) -> np.ndarray:
    """Return J mapping ``[roll_dot, pitch_dot]`` to ``[tML_dot, tMR_dot]``."""

    kinematic_args, roll, pitch = args[:-2], args[-2], args[-1]
    tmr_base, tml_base = ankle_ik(*kinematic_args, roll, pitch)
    tmr_roll, tml_roll = ankle_ik(*kinematic_args, roll + h, pitch)
    tmr_pitch, tml_pitch = ankle_ik(*kinematic_args, roll, pitch + h)
    return np.asarray(
        [
            [(tml_roll - tml_base) / h, (tml_pitch - tml_base) / h],
            [(tmr_roll - tmr_base) / h, (tmr_pitch - tmr_base) / h],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class AnkleSideState:
    pitch_index: int
    roll_index: int
    jacobian: np.ndarray
    motor_torque_tml_tmr: np.ndarray
    motor_velocity_tml_tmr: np.ndarray


class Mini3ParallelAnkle:
    """Convert Mini3 serial ankle commands to and from parallel motor space."""

    def __init__(
        self,
        joint_names: tuple[str, ...],
        *,
        parameters: dict[str, float] | None = None,
        default_motor_kd: float = 0.1,
        max_motor_kd: float = 3.5,
    ):
        self.parameters = dict(ANKLE_PARAMS if parameters is None else parameters)
        self.default_motor_kd = float(default_motor_kd)
        self.max_motor_kd = float(max_motor_kd)
        try:
            self.left_indices = (joint_names.index(ANKLE_JOINT_NAMES[0]), joint_names.index(ANKLE_JOINT_NAMES[1]))
            self.right_indices = (joint_names.index(ANKLE_JOINT_NAMES[2]), joint_names.index(ANKLE_JOINT_NAMES[3]))
        except ValueError as exc:
            raise ValueError(f"Mini3 real-motor model requires ankle joints {ANKLE_JOINT_NAMES}") from exc
        self.indices = np.asarray((*self.left_indices, *self.right_indices), dtype=np.int64)

    def _side_args(self, *, left: bool) -> tuple[float, ...]:
        p = self.parameters
        zl = p["zl"] + p["z0"]
        zr = p["zr"] + p["z0"]
        if left:
            return p["d"], p["df"], zl, zr, p["l"], p["lm"], p["hl"], p["hr"], p["z0"]
        return p["d"], p["df"], zr, zl, p["l"], p["lm"], p["hr"], p["hl"], p["z0"]

    def _motor_command_for_side(
        self,
        *,
        left: bool,
        target_pos: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> AnkleSideState:
        pitch_index, roll_index = self.left_indices if left else self.right_indices
        roll = float(joint_pos[roll_index])
        pitch = float(joint_pos[pitch_index])
        jacobian = ankle_jacobian(*self._side_args(left=left), roll, pitch)
        if not np.all(np.isfinite(jacobian)):
            return AnkleSideState(pitch_index, roll_index, np.eye(2), np.zeros(2), np.zeros(2))
        jacobian_inverse = np.linalg.pinv(jacobian)
        motor_velocity = jacobian @ np.asarray([joint_vel[roll_index], joint_vel[pitch_index]], dtype=np.float64)
        joint_damping = np.diag([float(kd[roll_index]), float(kd[pitch_index])])
        motor_damping = jacobian_inverse.T @ joint_damping @ jacobian_inverse
        diagonal_damping = np.diag(motor_damping)
        diagonal_damping = np.nan_to_num(diagonal_damping, nan=self.default_motor_kd)
        diagonal_damping = np.clip(diagonal_damping, 0.0, self.max_motor_kd)
        cross_damping = np.nan_to_num(motor_damping.copy())
        cross_damping[0, 0] = 0.0
        cross_damping[1, 1] = 0.0
        stiffness_torque = np.asarray(
            [
                kp[roll_index] * (target_pos[roll_index] - joint_pos[roll_index]),
                kp[pitch_index] * (target_pos[pitch_index] - joint_pos[pitch_index]),
            ],
            dtype=np.float64,
        )
        motor_torque = jacobian_inverse.T @ stiffness_torque - cross_damping @ motor_velocity
        motor_torque -= diagonal_damping * motor_velocity
        return AnkleSideState(pitch_index, roll_index, jacobian, motor_torque, motor_velocity)

    def motor_command(
        self,
        target_pos: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple[AnkleSideState, AnkleSideState]]:
        left = self._motor_command_for_side(left=True, target_pos=target_pos, joint_pos=joint_pos, joint_vel=joint_vel, kp=kp, kd=kd)
        right = self._motor_command_for_side(left=False, target_pos=target_pos, joint_pos=joint_pos, joint_vel=joint_vel, kp=kp, kd=kd)
        # Physical motor order: left tMR, left tML, right tML, right tMR.
        torque = np.asarray(
            [left.motor_torque_tml_tmr[1], left.motor_torque_tml_tmr[0], *right.motor_torque_tml_tmr],
            dtype=np.float64,
        )
        velocity = np.asarray(
            [left.motor_velocity_tml_tmr[1], left.motor_velocity_tml_tmr[0], *right.motor_velocity_tml_tmr],
            dtype=np.float64,
        )
        return torque, velocity, (left, right)

    def motor_to_joint_torque(
        self,
        motor_torque: np.ndarray,
        side_states: tuple[AnkleSideState, AnkleSideState],
    ) -> np.ndarray:
        motor_torque = np.asarray(motor_torque, dtype=np.float64)
        if motor_torque.shape != (4,):
            raise ValueError(f"Expected four ankle motor torques, got {motor_torque.shape}")
        result = np.zeros(4, dtype=np.float64)
        left, right = side_states
        left_joint_roll_pitch = left.jacobian.T @ motor_torque[[1, 0]]
        right_joint_roll_pitch = right.jacobian.T @ motor_torque[2:4]
        # Serial ankle order is left pitch/roll, right pitch/roll.
        result[0:2] = left_joint_roll_pitch[[1, 0]]
        result[2:4] = right_joint_roll_pitch[[1, 0]]
        return result


class Mini3RealMotorModel:
    """Apply the Mini3 sim2sim real-motor torque chain for one robot."""

    def __init__(
        self,
        joint_names: tuple[str, ...],
        kp: np.ndarray,
        kd: np.ndarray,
        effort_limit: np.ndarray,
        *,
        dt: float,
        response_enabled: bool = True,
        tn_enabled: bool = True,
        tn_limit_after_response: bool = True,
        kt_enabled: bool = True,
        response_kp: float = 0.0,
        response_ki: float = 90.6769527429,
        response_plant_tau_s: float = 0.00393417593548,
        response_delay_steps: float = 1.0,
        ankle_motor_torque_limit: float = 12.5,
    ):
        self.joint_names = tuple(joint_names)
        self.kp = np.asarray(kp, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)
        self.effort_limit = np.asarray(effort_limit, dtype=np.float64)
        expected_shape = (len(self.joint_names),)
        if any(array.shape != expected_shape for array in (self.kp, self.kd, self.effort_limit)):
            raise ValueError("Mini3 motor parameter arrays must match the joint count")
        if ankle_motor_torque_limit <= 0.0:
            raise ValueError("ankle_motor_torque_limit must be positive")
        self.response_enabled = bool(response_enabled)
        self.tn_enabled = bool(tn_enabled)
        self.tn_limit_after_response = bool(tn_limit_after_response)
        self.kt_enabled = bool(kt_enabled)
        self.ankle_motor_torque_limit = float(ankle_motor_torque_limit)
        self.ankle = Mini3ParallelAnkle(self.joint_names)
        ankle_set = set(self.ankle.indices.tolist())
        self.serial_indices = np.asarray([index for index in range(len(self.joint_names)) if index not in ankle_set], dtype=np.int64)
        self.serial_motor_types = tuple(mini3_motor_type(self.joint_names[index]) for index in self.serial_indices)
        self.tn_models = {name: MotorTnLimit(name=name, **spec) for name, spec in MOTOR_SPECS.items()}
        self.kt_models = {name: MotorKtOutputModel(name=name, **table) for name, table in KT_OUTPUT_TABLES.items()}
        response_kwargs = {
            "dt": dt,
            "kp": response_kp,
            "ki": response_ki,
            "plant_tau_s": response_plant_tau_s,
            "delay_steps": response_delay_steps,
        }
        self.serial_response = TorqueCurrentLoopResponse(len(self.serial_indices), **response_kwargs)
        self.ankle_response = TorqueCurrentLoopResponse(4, **response_kwargs)
        self.raw_pd_torque = np.zeros(expected_shape, dtype=np.float64)
        self.pre_response_torque = np.zeros(expected_shape, dtype=np.float64)
        self.response_torque = np.zeros(expected_shape, dtype=np.float64)
        self.applied_torque = np.zeros(expected_shape, dtype=np.float64)
        self.ankle_motor_velocity = np.zeros(4, dtype=np.float64)
        self.ankle_motor_torque = np.zeros(4, dtype=np.float64)

    def reset(self) -> None:
        self.serial_response.reset()
        self.ankle_response.reset()
        self.raw_pd_torque.fill(0.0)
        self.pre_response_torque.fill(0.0)
        self.response_torque.fill(0.0)
        self.applied_torque.fill(0.0)
        self.ankle_motor_velocity.fill(0.0)
        self.ankle_motor_torque.fill(0.0)

    def _clip_serial(self, torque: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        if not self.tn_enabled:
            return np.clip(torque, -self.effort_limit[self.serial_indices], self.effort_limit[self.serial_indices])
        output = np.asarray(torque, dtype=np.float64).copy()
        for local_index, motor_type in enumerate(self.serial_motor_types):
            output[local_index] = self.tn_models[motor_type].clip(output[local_index], velocity[local_index])
        return output

    def _map_serial_kt(self, torque: np.ndarray) -> np.ndarray:
        if not self.kt_enabled:
            return np.asarray(torque, dtype=np.float64)
        output = np.asarray(torque, dtype=np.float64).copy()
        for local_index, motor_type in enumerate(self.serial_motor_types):
            output[local_index] = self.kt_models[motor_type].map(output[local_index])
        return output

    def compute(self, target_pos: np.ndarray, joint_pos: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
        target_pos = np.asarray(target_pos, dtype=np.float64)
        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        joint_vel = np.asarray(joint_vel, dtype=np.float64)
        self.raw_pd_torque = self.kp * (target_pos - joint_pos) - self.kd * joint_vel

        serial_velocity = joint_vel[self.serial_indices]
        serial_command = self._clip_serial(self.raw_pd_torque[self.serial_indices], serial_velocity)
        self.pre_response_torque[self.serial_indices] = serial_command
        serial_response = self.serial_response.compute(serial_command) if self.response_enabled else serial_command
        self.response_torque[self.serial_indices] = serial_response
        if self.tn_enabled and self.tn_limit_after_response:
            serial_response = self._clip_serial(serial_response, serial_velocity)
        serial_output = self._map_serial_kt(serial_response)

        ankle_command, ankle_velocity, side_states = self.ankle.motor_command(target_pos, joint_pos, joint_vel, self.kp, self.kd)
        if self.tn_enabled:
            ankle_command = self.tn_models["4310p"].clip(ankle_command, ankle_velocity)
        else:
            ankle_command = np.clip(ankle_command, -self.ankle_motor_torque_limit, self.ankle_motor_torque_limit)
        ankle_response = self.ankle_response.compute(ankle_command) if self.response_enabled else ankle_command
        if self.tn_enabled and self.tn_limit_after_response:
            ankle_response = self.tn_models["4310p"].clip(ankle_response, ankle_velocity)
        ankle_output = self.kt_models["4310p"].map(ankle_response) if self.kt_enabled else ankle_response
        ankle_joint_output = self.ankle.motor_to_joint_torque(ankle_output, side_states)

        self.pre_response_torque[self.ankle.indices] = self.ankle.motor_to_joint_torque(ankle_command, side_states)
        self.response_torque[self.ankle.indices] = self.ankle.motor_to_joint_torque(ankle_response, side_states)
        self.applied_torque[self.serial_indices] = serial_output
        self.applied_torque[self.ankle.indices] = ankle_joint_output
        self.ankle_motor_velocity = ankle_velocity.copy()
        self.ankle_motor_torque = ankle_output.copy()
        self.applied_torque[:] = np.nan_to_num(self.applied_torque)
        return self.applied_torque.copy()

    def summary_lines(self) -> list[str]:
        lines = [
            "motor chain: PD -> pre-limit -> current-loop response -> post-TN -> KT -> MuJoCo",
            (f"response={self.response_enabled} TN={self.tn_enabled} post_response_TN={self.tn_limit_after_response} KT={self.kt_enabled}"),
        ]
        if self.tn_enabled:
            lines.extend(self.tn_models[name].summary() for name in sorted(self.tn_models))
        if self.kt_enabled:
            lines.extend(self.kt_models[name].summary() for name in sorted(self.kt_models))
        lines.append("mapping: hips/knees/waist=4340P; ankles/shoulders/elbows=4310P; ankles use motor-space J/J.T")
        return lines
