"""MJLab actuators for the Mini3 calibrated real-motor torque chain."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator import ActuatorCmd, IdealPdActuator, IdealPdActuatorCfg
from mjlab.entity import Entity

from humanoidverse.utils.mini3_real_motor import ANKLE_PARAMS, KT_OUTPUT_TABLES, MOTOR_SPECS


class _PhysicsStepTorqueDelay:
    """Fixed fractional physics-step delay with independent reset per environment."""

    def __init__(self, num_envs: int, num_channels: int, delay_steps: float, device: str):
        if delay_steps < 0.0:
            raise ValueError(f"delay_steps must be nonnegative, got {delay_steps}")
        self.delay_steps = float(delay_steps)
        self.capacity = math.ceil(self.delay_steps) + 1
        self.history = torch.zeros(num_envs, self.capacity, num_channels, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.write_index = 0

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.history[env_ids] = 0.0
        self.initialized[env_ids] = False

    def append_and_read(self, value: torch.Tensor) -> torch.Tensor:
        uninitialized = ~self.initialized
        if torch.any(uninitialized):
            self.history[uninitialized] = value[uninitialized, None, :]
            self.initialized[uninitialized] = True
        self.history[:, self.write_index] = value
        lower = int(math.floor(self.delay_steps))
        alpha = self.delay_steps - lower
        newer = self.history[:, (self.write_index - lower) % self.capacity]
        older = self.history[:, (self.write_index - lower - 1) % self.capacity]
        output = (1.0 - alpha) * newer + alpha * older
        self.write_index = (self.write_index + 1) % self.capacity
        return output


@dataclass(kw_only=True)
class Mini3RealMotorActuatorCfg(IdealPdActuatorCfg):
    """Configuration for serial Mini3 joints using the calibrated motor model."""

    motor_type: str
    stiffness_by_joint: dict[str, float] | None = None
    damping_by_joint: dict[str, float] | None = None
    torque_response_enabled: bool = True
    torque_response_kp: float = 0.0
    torque_response_ki: float = 90.6769527429
    torque_response_plant_tau_s: float = 0.00393417593548
    torque_response_delay_steps: float = 1.0
    tn_torque_limit_enabled: bool = True
    tn_limit_after_response: bool = True
    kt_output_model_enabled: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        motor_type = self.motor_type.lower()
        if motor_type not in MOTOR_SPECS:
            raise ValueError(f"Unsupported Mini3 motor type {self.motor_type!r}")
        if self.torque_response_kp < 0.0 or self.torque_response_ki < 0.0:
            raise ValueError("Torque response gains must be nonnegative")
        if self.torque_response_plant_tau_s <= 0.0:
            raise ValueError("torque_response_plant_tau_s must be positive")
        if self.torque_response_delay_steps < 0.0:
            raise ValueError("torque_response_delay_steps must be nonnegative")

    def build(self, entity: Entity, target_ids: list[int], target_names: list[str]) -> Mini3RealMotorActuator:
        return Mini3RealMotorActuator(self, entity, target_ids, target_names)


class Mini3RealMotorActuator(IdealPdActuator):
    """Vectorized Mini3 serial-joint PD, response, T-N and KT model."""

    cfg: Mini3RealMotorActuatorCfg

    def __init__(self, cfg: Mini3RealMotorActuatorCfg, entity: Entity, target_ids: list[int], target_names: list[str]):
        super().__init__(cfg, entity, target_ids, target_names)
        self.motor_type = cfg.motor_type.lower()
        self.physics_dt: float | None = None
        self.response_alpha: float | None = None
        self.torque_state: torch.Tensor | None = None
        self.torque_integral: torch.Tensor | None = None
        self.motor_strength: torch.Tensor | None = None
        self.response_delay: _PhysicsStepTorqueDelay | None = None
        self.kt_feedback: torch.Tensor | None = None
        self.kt_actual: torch.Tensor | None = None
        self.computed_effort: torch.Tensor | None = None
        self.pre_response_effort: torch.Tensor | None = None
        self.response_effort: torch.Tensor | None = None
        self.post_tn_effort: torch.Tensor | None = None
        self.applied_effort: torch.Tensor | None = None

    def initialize(self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        assert self.stiffness is not None
        assert self.damping is not None
        if self.cfg.stiffness_by_joint is not None:
            self.stiffness[:] = self._values_by_target(self.cfg.stiffness_by_joint, device)
        if self.cfg.damping_by_joint is not None:
            self.damping[:] = self._values_by_target(self.cfg.damping_by_joint, device)
        self.default_stiffness = self.stiffness.clone()
        self.default_damping = self.damping.clone()

        shape = (data.nworld, len(self.target_names))
        self.physics_dt = float(mj_model.opt.timestep)
        self.response_alpha = min(1.0, self.physics_dt / self.cfg.torque_response_plant_tau_s)
        self.torque_state = torch.zeros(shape, device=device)
        self.torque_integral = torch.zeros(shape, device=device)
        self.motor_strength = torch.ones(shape, device=device)
        self.response_delay = _PhysicsStepTorqueDelay(
            data.nworld,
            len(self.target_names),
            self.cfg.torque_response_delay_steps,
            device,
        )
        self.computed_effort = torch.zeros(shape, device=device)
        self.pre_response_effort = torch.zeros(shape, device=device)
        self.response_effort = torch.zeros(shape, device=device)
        self.post_tn_effort = torch.zeros(shape, device=device)
        self.applied_effort = torch.zeros(shape, device=device)
        table = KT_OUTPUT_TABLES[self.motor_type]
        self.kt_feedback = torch.tensor((0.0, *table["feedback_tau_nm"]), device=device)
        self.kt_actual = torch.tensor((0.0, *table["actual_tau_nm"]), device=device)

    def _values_by_target(self, values: dict[str, float], device: str) -> torch.Tensor:
        missing = [name for name in self.target_names if name not in values]
        if missing:
            raise ValueError(f"Mini3 actuator gain mapping is missing targets: {missing}")
        return torch.tensor([float(values[name]) for name in self.target_names], device=device).unsqueeze(0)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        for tensor in (
            self.torque_state,
            self.torque_integral,
            self.computed_effort,
            self.pre_response_effort,
            self.response_effort,
            self.post_tn_effort,
            self.applied_effort,
        ):
            if tensor is not None:
                tensor[env_ids] = 0.0
        if self.response_delay is not None:
            self.response_delay.reset(env_ids)

    def set_motor_strength(self, env_ids: torch.Tensor | slice, strength: torch.Tensor) -> None:
        assert self.motor_strength is not None
        if strength.ndim == 1:
            strength = strength.unsqueeze(-1)
        self.motor_strength[env_ids] = strength

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        assert self.stiffness is not None
        assert self.damping is not None
        assert self.computed_effort is not None
        assert self.pre_response_effort is not None
        assert self.response_effort is not None
        assert self.post_tn_effort is not None
        assert self.applied_effort is not None
        torque = self.stiffness * (cmd.position_target - cmd.pos)
        torque += self.damping * (cmd.velocity_target - cmd.vel)
        torque += cmd.effort_target
        self.computed_effort[:] = torque
        torque = self._pre_limit(torque, cmd.vel)
        self.pre_response_effort[:] = torque
        torque = self._current_loop_response(torque)
        self.response_effort[:] = torque
        if self.cfg.tn_torque_limit_enabled and self.cfg.tn_limit_after_response:
            torque = self._tn_clip(torque, cmd.vel)
        self.post_tn_effort[:] = torque
        if self.cfg.kt_output_model_enabled:
            torque = self._kt_map(torque)
        self.applied_effort[:] = torch.nan_to_num(torque)
        return self.applied_effort

    def _pre_limit(self, torque: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        assert self.force_limit is not None
        assert self.motor_strength is not None
        if self.cfg.tn_torque_limit_enabled:
            return self._tn_clip(torque, velocity)
        limit = self.force_limit * self.motor_strength
        return torch.clamp(torque, min=-limit, max=limit)

    def _current_loop_response(self, torque_command: torch.Tensor) -> torch.Tensor:
        if not self.cfg.torque_response_enabled:
            return torque_command
        assert self.torque_state is not None
        assert self.torque_integral is not None
        assert self.physics_dt is not None
        assert self.response_alpha is not None
        assert self.response_delay is not None
        torque_command = torch.nan_to_num(torque_command)
        self.torque_state[:] = torch.nan_to_num(self.torque_state)
        self.torque_integral[:] = torch.nan_to_num(self.torque_integral)
        error = torque_command - self.torque_state
        self.torque_integral += error * self.physics_dt
        loop_output = self.cfg.torque_response_kp * error + self.cfg.torque_response_ki * self.torque_integral
        self.torque_state += self.response_alpha * (loop_output - self.torque_state)
        self.torque_state[:] = torch.nan_to_num(self.torque_state)
        return self.response_delay.append_and_read(self.torque_state)

    def _tn_clip(self, torque: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        assert self.motor_strength is not None
        spec = MOTOR_SPECS[self.motor_type]
        peak_torque = float(spec["peak_torque"]) * self.motor_strength
        peak_speed_rpm = float(spec["rated_speed_rpm"])
        no_load_speed_rpm = float(spec["no_load_speed_rpm"])
        speed_drop_per_nm = (no_load_speed_rpm - peak_speed_rpm) / peak_torque
        speed_rpm = velocity * 60.0 / (2.0 * math.pi)
        lower = torch.maximum(-peak_torque, (-no_load_speed_rpm - speed_rpm) / speed_drop_per_nm)
        upper = torch.minimum(peak_torque, (no_load_speed_rpm - speed_rpm) / speed_drop_per_nm)
        empty = lower > upper
        fallback = torch.where(speed_rpm >= 0.0, lower, upper)
        lower = torch.where(empty, fallback, lower)
        upper = torch.where(empty, fallback, upper)
        return torch.clamp(torque, min=lower, max=upper)

    def _kt_map(self, feedback_torque: torch.Tensor) -> torch.Tensor:
        assert self.kt_feedback is not None
        assert self.kt_actual is not None
        magnitude = torch.clamp(torch.abs(feedback_torque), max=self.kt_feedback[-1])
        indices = torch.bucketize(magnitude.contiguous(), self.kt_feedback)
        indices = torch.clamp(indices, min=1, max=self.kt_feedback.numel() - 1)
        x0 = self.kt_feedback[indices - 1]
        x1 = self.kt_feedback[indices]
        y0 = self.kt_actual[indices - 1]
        y1 = self.kt_actual[indices]
        ratio = (magnitude - x0) / torch.clamp(x1 - x0, min=torch.finfo(feedback_torque.dtype).eps)
        return torch.sign(feedback_torque) * (y0 + ratio * (y1 - y0))


@dataclass(kw_only=True)
class Mini3ParallelAnkleRealMotorActuatorCfg(Mini3RealMotorActuatorCfg):
    """Configuration for the four Mini3 parallel ankle motors."""

    ankle_motor_torque_limit: float = 12.5
    ankle_parameters: dict[str, float] | None = None
    default_motor_kd: float = 0.1
    max_motor_kd: float = 3.5
    jacobian_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.motor_type.lower() != "4310p":
            raise ValueError("Mini3 parallel ankle must use the 4310P motor model")
        if self.ankle_motor_torque_limit <= 0.0:
            raise ValueError("ankle_motor_torque_limit must be positive")

    def build(self, entity: Entity, target_ids: list[int], target_names: list[str]) -> Mini3ParallelAnkleRealMotorActuator:
        return Mini3ParallelAnkleRealMotorActuator(self, entity, target_ids, target_names)


class Mini3ParallelAnkleRealMotorActuator(Mini3RealMotorActuator):
    """Apply response, T-N and KT in parallel-ankle motor space."""

    cfg: Mini3ParallelAnkleRealMotorActuatorCfg

    REQUIRED_JOINTS = (
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    )

    def __init__(
        self,
        cfg: Mini3ParallelAnkleRealMotorActuatorCfg,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ):
        super().__init__(cfg, entity, target_ids, target_names)
        missing = [name for name in self.REQUIRED_JOINTS if name not in target_names]
        if missing or len(target_names) != 4:
            raise ValueError(f"Parallel ankle actuator requires exactly {self.REQUIRED_JOINTS}, got {target_names}")
        self.local_index = {name: target_names.index(name) for name in self.REQUIRED_JOINTS}
        self.parameters = dict(ANKLE_PARAMS if cfg.ankle_parameters is None else cfg.ankle_parameters)
        self.eye2: torch.Tensor | None = None
        self.motor_velocity: torch.Tensor | None = None
        self.motor_command: torch.Tensor | None = None

    def initialize(self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        self.eye2 = torch.eye(2, device=device).unsqueeze(0)
        self.motor_velocity = torch.zeros(data.nworld, 4, device=device)
        self.motor_command = torch.zeros(data.nworld, 4, device=device)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        assert self.stiffness is not None
        assert self.damping is not None
        assert self.computed_effort is not None
        assert self.pre_response_effort is not None
        assert self.response_effort is not None
        assert self.post_tn_effort is not None
        assert self.applied_effort is not None
        assert self.motor_velocity is not None
        assert self.motor_command is not None
        joint_pd = self.stiffness * (cmd.position_target - cmd.pos)
        joint_pd += self.damping * (cmd.velocity_target - cmd.vel)
        joint_pd += cmd.effort_target
        self.computed_effort[:] = joint_pd
        left = self._side_motor_command(True, cmd)
        right = self._side_motor_command(False, cmd)
        # Motor order follows serial target order: left tMR/tML, right tML/tMR.
        motor_torque = torch.stack((left["tmr"], left["tml"], right["tml"], right["tmr"]), dim=1)
        motor_velocity = torch.stack((left["tmr_velocity"], left["tml_velocity"], right["tml_velocity"], right["tmr_velocity"]), dim=1)
        self.motor_velocity[:] = motor_velocity
        motor_torque = self._pre_limit_ankle(motor_torque, motor_velocity)
        self.motor_command[:] = motor_torque
        self.pre_response_effort[:] = self._motor_to_joint(motor_torque, left, right)
        motor_torque = self._current_loop_response(motor_torque)
        self.response_effort[:] = self._motor_to_joint(motor_torque, left, right)
        if self.cfg.tn_torque_limit_enabled and self.cfg.tn_limit_after_response:
            motor_torque = self._tn_clip(motor_torque, motor_velocity)
        self.post_tn_effort[:] = self._motor_to_joint(motor_torque, left, right)
        if self.cfg.kt_output_model_enabled:
            motor_torque = self._kt_map(motor_torque)
        self.applied_effort[:] = torch.nan_to_num(self._motor_to_joint(motor_torque, left, right))
        return self.applied_effort

    def _pre_limit_ankle(self, torque: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        assert self.motor_strength is not None
        if self.cfg.tn_torque_limit_enabled:
            return self._tn_clip(torque, velocity)
        limit = self.cfg.ankle_motor_torque_limit * self.motor_strength
        return torch.clamp(torque, min=-limit, max=limit)

    def _side_motor_command(self, left: bool, cmd: ActuatorCmd) -> dict[str, Any]:
        assert self.stiffness is not None
        assert self.damping is not None
        assert self.eye2 is not None
        prefix = "left" if left else "right"
        pitch_index = self.local_index[f"{prefix}_ankle_pitch_joint"]
        roll_index = self.local_index[f"{prefix}_ankle_roll_joint"]
        roll = cmd.pos[:, roll_index]
        pitch = cmd.pos[:, pitch_index]
        jacobian, valid = self._ankle_jacobian(self._side_args(left), roll, pitch)
        safe_jacobian = torch.where(valid[:, None, None], jacobian, self.eye2.expand_as(jacobian))
        inverse = torch.linalg.pinv(safe_jacobian)
        joint_velocity = torch.stack((cmd.vel[:, roll_index], cmd.vel[:, pitch_index]), dim=1)
        motor_velocity = torch.bmm(safe_jacobian, joint_velocity.unsqueeze(-1)).squeeze(-1)
        joint_damping = torch.zeros(cmd.pos.shape[0], 2, 2, device=cmd.pos.device)
        joint_damping[:, 0, 0] = self.damping[:, roll_index]
        joint_damping[:, 1, 1] = self.damping[:, pitch_index]
        motor_damping = inverse.transpose(1, 2).bmm(joint_damping).bmm(inverse)
        diagonal = torch.diagonal(motor_damping, dim1=1, dim2=2)
        diagonal = torch.nan_to_num(diagonal, nan=self.cfg.default_motor_kd)
        diagonal = torch.clamp(diagonal, min=0.0, max=self.cfg.max_motor_kd)
        cross = torch.nan_to_num(motor_damping.clone())
        cross[:, 0, 0] = 0.0
        cross[:, 1, 1] = 0.0
        stiffness_torque = torch.stack(
            (
                self.stiffness[:, roll_index] * (cmd.position_target[:, roll_index] - roll),
                self.stiffness[:, pitch_index] * (cmd.position_target[:, pitch_index] - pitch),
            ),
            dim=1,
        )
        motor_torque = torch.bmm(inverse.transpose(1, 2), stiffness_torque.unsqueeze(-1)).squeeze(-1)
        motor_torque -= torch.bmm(cross, motor_velocity.unsqueeze(-1)).squeeze(-1)
        motor_torque -= diagonal * motor_velocity
        motor_torque = torch.where(valid[:, None], motor_torque, torch.zeros_like(motor_torque))
        motor_velocity = torch.where(valid[:, None], motor_velocity, torch.zeros_like(motor_velocity))
        return {
            "pitch_index": pitch_index,
            "roll_index": roll_index,
            "jacobian": safe_jacobian,
            "valid": valid,
            "tml": motor_torque[:, 0],
            "tmr": motor_torque[:, 1],
            "tml_velocity": motor_velocity[:, 0],
            "tmr_velocity": motor_velocity[:, 1],
        }

    def _motor_to_joint(self, motor_torque: torch.Tensor, left: dict[str, Any], right: dict[str, Any]) -> torch.Tensor:
        output = torch.zeros_like(motor_torque)
        left_tml_tmr = torch.stack((motor_torque[:, 1], motor_torque[:, 0]), dim=1)
        right_tml_tmr = motor_torque[:, 2:4]
        left_joint = torch.bmm(left["jacobian"].transpose(1, 2), left_tml_tmr.unsqueeze(-1)).squeeze(-1)
        right_joint = torch.bmm(right["jacobian"].transpose(1, 2), right_tml_tmr.unsqueeze(-1)).squeeze(-1)
        left_joint = torch.where(left["valid"][:, None], left_joint, torch.zeros_like(left_joint))
        right_joint = torch.where(right["valid"][:, None], right_joint, torch.zeros_like(right_joint))
        output[:, left["roll_index"]] = left_joint[:, 0]
        output[:, left["pitch_index"]] = left_joint[:, 1]
        output[:, right["roll_index"]] = right_joint[:, 0]
        output[:, right["pitch_index"]] = right_joint[:, 1]
        return output

    def _side_args(self, left: bool) -> tuple[float, ...]:
        p = self.parameters
        zl = p["zl"] + p["z0"]
        zr = p["zr"] + p["z0"]
        if left:
            return p["d"], p["df"], zl, zr, p["l"], p["lm"], p["hl"], p["hr"], p["z0"]
        return p["d"], p["df"], zr, zl, p["l"], p["lm"], p["hr"], p["hl"], p["z0"]

    def _ankle_jacobian(self, args: tuple[float, ...], roll: torch.Tensor, pitch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_dtype = roll.dtype
        roll = roll.to(dtype=torch.float64)
        pitch = pitch.to(dtype=torch.float64)
        h = self.cfg.jacobian_epsilon
        tmr, tml, valid = self._ankle_ik(args, roll, pitch)
        tmr_roll, tml_roll, valid_roll = self._ankle_ik(args, roll + h, pitch)
        tmr_pitch, tml_pitch, valid_pitch = self._ankle_ik(args, roll, pitch + h)
        jacobian = torch.stack(
            (
                torch.stack(((tml_roll - tml) / h, (tml_pitch - tml) / h), dim=1),
                torch.stack(((tmr_roll - tmr) / h, (tmr_pitch - tmr) / h), dim=1),
            ),
            dim=1,
        )
        valid &= valid_roll & valid_pitch & torch.all(torch.isfinite(jacobian), dim=(1, 2))
        return jacobian.to(dtype=original_dtype), valid

    @staticmethod
    def _ankle_ik(args: tuple[float, ...], roll: torch.Tensor, pitch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d, df, zl, zr, link_l, lm, hl, hr, z0 = args
        cx, cy = torch.cos(roll), torch.cos(pitch)
        sx, sy = torch.sin(roll), torch.sin(pitch)
        epsilon = 1.0e-12
        al = -2.0 * d * (df * cx - z0 * sx)
        bl = 2.0 * d * (zl + link_l * sy - df * sx * cy - z0 * cx * cy)
        cl = (
            d**2
            + (lm - link_l * cy - df * sx * sy - z0 * cx * sy) ** 2
            + (df * cx - z0 * sx) ** 2
            + (zl + link_l * sy - df * sx * cy - z0 * cx * cy) ** 2
        )
        dl = hl**2 - cl
        rl = torch.sqrt(torch.clamp(al**2 + bl**2, min=0.0))
        ar = -2.0 * d * (df * cx + z0 * sx)
        br = -2.0 * d * (zr + link_l * sy + df * sx * cy - z0 * cx * cy)
        cr = (
            d**2
            + (lm - link_l * cy + df * sx * sy - z0 * cx * sy) ** 2
            + (df * cx + z0 * sx) ** 2
            + (zr + link_l * sy + df * sx * cy - z0 * cx * cy) ** 2
        )
        dr = hr**2 - cr
        rr = torch.sqrt(torch.clamp(ar**2 + br**2, min=0.0))
        sl = dl / torch.clamp(rl, min=epsilon)
        sr = dr / torch.clamp(rr, min=epsilon)
        valid_left = (rl + epsilon >= torch.abs(dl)) & (rl > epsilon) & torch.isfinite(sl)
        valid_right = (rr + epsilon >= torch.abs(dr)) & (rr > epsilon) & torch.isfinite(sr)
        tml = torch.asin(torch.clamp(sl, -1.0, 1.0)) - torch.atan2(al, bl)
        tmr = math.pi - torch.asin(torch.clamp(sr, -1.0, 1.0)) - torch.atan2(ar, br) - 2.0 * math.pi
        tml = torch.where(valid_left, tml, torch.zeros_like(tml))
        tmr = torch.where(valid_right, tmr, torch.zeros_like(tmr))
        valid = valid_left & valid_right & torch.isfinite(tml) & torch.isfinite(tmr)
        return tmr, tml, valid
