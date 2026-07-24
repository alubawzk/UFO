from __future__ import annotations

import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from humanoidverse.mujoco_tracking_inference import (
    ActorHistory,
    MotionReferenceTrajectory,
    MotionReferenceVisualizer,
    PhysicsStepActionDelay,
    PhysicsStepImuDelay,
    _dc_motor_clip,
    _future_weighted_mean,
    _joint_layout,
    parse_args,
)
from humanoidverse.utils.robot_spec import load_robot_spec


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

    def test_future_weighted_mean_uses_lookahead_and_truncates_tail(self) -> None:
        values = torch.tensor([[1.0], [2.0], [4.0], [8.0]])

        result = _future_weighted_mean(values, future_frames=3, gamma=0.5)

        torch.testing.assert_close(
            result,
            torch.tensor(
                [
                    [12.0 / 7.0],
                    [24.0 / 7.0],
                    [16.0 / 3.0],
                    [8.0],
                ]
            ),
        )

    def test_single_future_frame_keeps_latents_unchanged(self) -> None:
        values = torch.randn(4, 3)
        self.assertIs(_future_weighted_mean(values, future_frames=1, gamma=0.8), values)

    def test_reference_motion_visualization_is_enabled_by_default(self) -> None:
        with patch(
            "sys.argv",
            [
                "mujoco_tracking_inference",
                "--model-folder",
                "runs/example",
                "--data-path",
                "motion.pkl",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.show_reference_motion)

    def test_reference_visualizer_renders_selected_frame_beside_robot(self) -> None:
        robot_spec = load_robot_spec("configs/robots/mini3.yaml")
        model = mujoco.MjModel.from_xml_path(robot_spec.xml_path)
        layout = _joint_layout(model, list(robot_spec.control_joint_names))
        visualizer = MotionReferenceVisualizer(model, layout, lateral_offset=1.25, alpha=0.4)
        user_scene = mujoco.MjvScene(model, maxgeom=100)
        trajectory = MotionReferenceTrajectory(
            root_pos=np.array([[0.0, 0.0, 0.5], [2.0, 3.0, 0.6]]),
            root_quat_wxyz=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            dof_pos=np.zeros((2, len(robot_spec.control_joint_names))),
        )

        geom_count = visualizer.update(user_scene, trajectory, frame_index=1)

        self.assertGreater(geom_count, 0)
        self.assertEqual(geom_count, user_scene.ngeom)
        np.testing.assert_allclose(visualizer.data.qpos[:3], [2.0, 4.25, 0.6])
        for geom_index in range(geom_count):
            geom = user_scene.geoms[geom_index]
            np.testing.assert_allclose(geom.rgba, [0.05, 0.85, 1.0, 0.4], atol=1.0e-6)
            self.assertEqual(int(geom.category), int(mujoco.mjtCatBit.mjCAT_DECOR))


if __name__ == "__main__":
    unittest.main()
