from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from humanoidverse.agents.envs.mini3_real_motor_actuator import (
    Mini3ParallelAnkleRealMotorActuator,
    Mini3RealMotorActuator,
    _PhysicsStepTorqueDelay,
)
from humanoidverse.utils.mini3_real_motor import (
    ANKLE_JOINT_NAMES,
    ANKLE_PARAMS,
    KT_OUTPUT_TABLES,
    MOTOR_SPECS,
    Mini3ParallelAnkle,
    Mini3RealMotorModel,
    MotorKtOutputModel,
    MotorTnLimit,
    TorqueCurrentLoopResponse,
    ankle_ik,
)

MINI3_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
)


class Mini3RealMotorTest(unittest.TestCase):
    def test_tn_limit_matches_peak_and_no_load_points(self) -> None:
        limiter = MotorTnLimit(name="4340p", **MOTOR_SPECS["4340p"])
        rated_speed = 36.0 * 2.0 * math.pi / 60.0
        no_load_speed = 60.0 * 2.0 * math.pi / 60.0
        _, rated_upper = limiter.bounds(rated_speed)
        no_load_lower, no_load_upper = limiter.bounds(no_load_speed)
        self.assertAlmostEqual(float(rated_upper), 27.0)
        self.assertAlmostEqual(float(no_load_upper), 0.0)
        self.assertAlmostEqual(float(no_load_lower), -27.0)

    def test_kt_mapping_hits_calibration_points_and_preserves_sign(self) -> None:
        model = MotorKtOutputModel(name="4310p", **KT_OUTPUT_TABLES["4310p"])
        np.testing.assert_allclose(model.map(np.asarray([-7.12, 0.0, 7.12])), [-6.0, 0.0, 6.0])

    def test_current_loop_output_delay_advances_in_physics_steps(self) -> None:
        response = TorqueCurrentLoopResponse(
            1,
            dt=0.002,
            kp=0.0,
            ki=100.0,
            plant_tau_s=0.004,
            delay_steps=1.0,
        )
        first = response.compute(np.asarray([10.0]))
        second_internal = None
        second = response.compute(np.asarray([10.0]))
        second_internal = response.tau_raw.copy()
        third = response.compute(np.asarray([10.0]))
        np.testing.assert_allclose(second, first)
        np.testing.assert_allclose(third, second_internal)

    def test_parallel_ankle_jacobian_mapping_preserves_instantaneous_power(self) -> None:
        ankle = Mini3ParallelAnkle(MINI3_JOINT_NAMES)
        target = np.zeros(len(MINI3_JOINT_NAMES))
        position = np.zeros_like(target)
        velocity = np.zeros_like(target)
        target[ankle.indices] = [0.10, -0.04, 0.08, 0.03]
        velocity[ankle.indices] = [0.30, -0.20, -0.15, 0.25]
        kp = np.full_like(target, 50.0)
        kd = np.full_like(target, 1.2)
        _, motor_velocity, states = ankle.motor_command(target, position, velocity, kp, kd)
        motor_torque = np.asarray([2.0, -1.0, 1.5, -0.5])
        joint_torque = ankle.motor_to_joint_torque(motor_torque, states)
        joint_velocity = velocity[ankle.indices]
        self.assertAlmostEqual(float(motor_torque @ motor_velocity), float(joint_torque @ joint_velocity), places=8)

    def test_full_model_uses_kt_output_for_serial_joint(self) -> None:
        size = len(MINI3_JOINT_NAMES)
        model = Mini3RealMotorModel(
            MINI3_JOINT_NAMES,
            kp=np.full(size, 20.0),
            kd=np.ones(size),
            effort_limit=np.full(size, 27.0),
            dt=0.002,
            response_enabled=False,
            tn_enabled=True,
            kt_enabled=True,
        )
        target = np.zeros(size)
        target[0] = 1.0
        output = model.compute(target, np.zeros(size), np.zeros(size))
        self.assertGreater(output[0], 0.0)
        self.assertLess(output[0], 20.0)
        np.testing.assert_allclose(output[[MINI3_JOINT_NAMES.index(name) for name in ANKLE_JOINT_NAMES]], 0.0)

    def test_training_response_delay_resets_environments_independently(self) -> None:
        delay = _PhysicsStepTorqueDelay(num_envs=2, num_channels=1, delay_steps=1.0, device="cpu")
        torch.testing.assert_close(delay.append_and_read(torch.tensor([[1.0], [10.0]])), torch.tensor([[1.0], [10.0]]))
        torch.testing.assert_close(delay.append_and_read(torch.tensor([[2.0], [20.0]])), torch.tensor([[1.0], [10.0]]))
        delay.reset(torch.tensor([0]))
        torch.testing.assert_close(delay.append_and_read(torch.tensor([[3.0], [30.0]])), torch.tensor([[3.0], [20.0]]))

    def test_training_tn_and_kt_match_pure_mujoco_models(self) -> None:
        actuator = object.__new__(Mini3RealMotorActuator)
        actuator.motor_type = "4340p"
        actuator.motor_strength = torch.ones(1, 3)
        table = KT_OUTPUT_TABLES["4340p"]
        actuator.kt_feedback = torch.tensor((0.0, *table["feedback_tau_nm"]))
        actuator.kt_actual = torch.tensor((0.0, *table["actual_tau_nm"]))
        torque = torch.tensor([[30.0, 20.0, -20.0]])
        velocity = torch.tensor([[0.0, math.pi, -math.pi]])
        training_tn = actuator._tn_clip(torque, velocity).numpy()
        pure_tn = MotorTnLimit(name="4340p", **MOTOR_SPECS["4340p"]).clip(torque.numpy(), velocity.numpy())
        np.testing.assert_allclose(training_tn, pure_tn, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(
            actuator._kt_map(torch.from_numpy(pure_tn)).numpy(),
            MotorKtOutputModel(name="4340p", **table).map(pure_tn),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    def test_training_parallel_ankle_ik_matches_pure_mujoco(self) -> None:
        p = ANKLE_PARAMS
        args = (p["d"], p["df"], p["zl"] + p["z0"], p["zr"] + p["z0"], p["l"], p["lm"], p["hl"], p["hr"], p["z0"])
        roll = torch.tensor([0.0, 0.1], dtype=torch.float64)
        pitch = torch.tensor([0.0, -0.08], dtype=torch.float64)
        training_tmr, training_tml, valid = Mini3ParallelAnkleRealMotorActuator._ankle_ik(args, roll, pitch)
        self.assertTrue(torch.all(valid))
        expected = np.asarray([ankle_ik(*args, float(r), float(q)) for r, q in zip(roll, pitch)])
        np.testing.assert_allclose(training_tmr.numpy(), expected[:, 0], rtol=0.0, atol=1.0e-10)
        np.testing.assert_allclose(training_tml.numpy(), expected[:, 1], rtol=0.0, atol=1.0e-10)


if __name__ == "__main__":
    unittest.main()
