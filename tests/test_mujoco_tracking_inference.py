from __future__ import annotations

import unittest

import numpy as np

from humanoidverse.mujoco_tracking_inference import (
    ActorHistory,
    PhysicsStepActionDelay,
    PhysicsStepImuDelay,
    _dc_motor_clip,
)


class MujocoTrackingInferenceTest(unittest.TestCase):
    def test_action_delay_advances_in_physics_steps(self) -> None:
        delay = PhysicsStepActionDelay(np.array([0, 2]))
        np.testing.assert_allclose(delay.push(np.array([1.0, 1.0])), [1.0, 0.0])
        np.testing.assert_allclose(delay.push(np.array([2.0, 2.0])), [2.0, 0.0])
        np.testing.assert_allclose(delay.push(np.array([3.0, 3.0])), [3.0, 1.0])

    def test_imu_delay_interpolates_between_physics_samples(self) -> None:
        delay = PhysicsStepImuDelay(physics_dt=0.002, delay_seconds=0.003, max_delay_seconds=0.004, interpolate=True)
        delay.reset(np.zeros(6))
        delay.record(np.full(6, 1.0))
        delay.record(np.full(6, 2.0))
        delay.record(np.full(6, 3.0))
        np.testing.assert_allclose(delay.read(), np.full(6, 1.5))

    def test_actor_history_is_newest_first_and_lagged(self) -> None:
        history = ActorHistory({"actions": 2, "dof_pos": 2}, {"actions": 1, "dof_pos": 1})
        np.testing.assert_allclose(history.query_flat(), [0.0, 0.0, 0.0, 0.0])
        history.add({"actions": np.array([1.0]), "dof_pos": np.array([10.0])})
        np.testing.assert_allclose(history.query_flat(), [1.0, 0.0, 10.0, 0.0])
        history.add({"actions": np.array([2.0]), "dof_pos": np.array([20.0])})
        np.testing.assert_allclose(history.query_flat(), [2.0, 1.0, 20.0, 10.0])

    def test_dc_motor_clip_matches_torque_speed_corner(self) -> None:
        effort = np.array([10.0, 10.0, 10.0])
        velocity_limit = np.array([5.0, 5.0, 5.0])
        torque = np.array([20.0, 20.0, -20.0])
        velocity = np.array([0.0, 2.5, -2.5])
        np.testing.assert_allclose(_dc_motor_clip(torque, velocity, effort, velocity_limit), [10.0, 5.0, -5.0])


if __name__ == "__main__":
    unittest.main()
