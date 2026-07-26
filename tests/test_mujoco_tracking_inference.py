from __future__ import annotations

import math
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
    _build_mujoco_model,
    _dc_motor_clip,
    _future_weighted_mean,
    _joint_layout,
    _offset_motion_lib_joint_angles,
    _scale_motion_lib_root_xy_motion,
    _zero_motion_lib_root_xy_motion,
    parse_args,
)
from humanoidverse.utils.helpers import (
    _centered_moving_average,
    _centered_quaternion_moving_average,
)
from humanoidverse.utils.robot_spec import load_robot_spec, load_robot_training_spec


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
        self.assertFalse(args.zero_reference_root_linear_velocity)
        self.assertFalse(args.zero_reference_root_xy_motion)
        self.assertEqual(args.reference_root_xy_scale, 1.0)
        self.assertEqual(args.reference_smoothing_window, 1)
        self.assertEqual(args.reference_ankle_pitch_offset_deg, 0.0)
        self.assertIsNone(args.scene_xml)

    def test_reference_smoothing_uses_centered_replicated_window(self) -> None:
        values = torch.tensor([[0.0], [0.0], [9.0], [0.0], [0.0]])

        smoothed = _centered_moving_average(values, window=3)

        torch.testing.assert_close(smoothed, torch.tensor([[0.0], [3.0], [3.0], [3.0], [0.0]]))

    def test_reference_quaternion_smoothing_resolves_antipodal_signs(self) -> None:
        quaternions = torch.tensor(
            [
                [[0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, -1.0]],
                [[0.0, 0.0, 0.0, 1.0]],
            ]
        )

        smoothed = _centered_quaternion_moving_average(quaternions, window=3)

        torch.testing.assert_close(smoothed, torch.tensor([[[0.0, 0.0, 0.0, 1.0]]] * 3))

    def test_reference_smoothing_window_can_be_set(self) -> None:
        with patch(
            "sys.argv",
            [
                "mujoco_tracking_inference",
                "--model-folder",
                "runs/example",
                "--data-path",
                "motion.pkl",
                "--reference-smoothing-window",
                "5",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.reference_smoothing_window, 5)

    def test_reference_ankle_pitch_offset_rebuilds_joint_motion(self) -> None:
        class MeshParserStub:
            body_names = ["base_link", "left_ankle_pitch_link", "right_ankle_pitch_link"]
            mjcf_data = {
                "body_to_joint": {
                    "left_ankle_pitch_link": "left_ankle_pitch_joint",
                    "right_ankle_pitch_link": "right_ankle_pitch_joint",
                }
            }
            dof_axis = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

            @staticmethod
            def fk_batch(pose_aa, root_position, *, return_full, dt):
                del return_full, dt
                body_count = pose_aa.shape[2]
                global_translation = root_position[:, :, None, :].repeat(1, 1, body_count, 1)
                return {
                    "global_translation": global_translation,
                    "dof_pos": pose_aa[:, :, 1:, 1],
                }

        class MotionLibStub:
            mesh_parsers = MeshParserStub()
            length_starts = torch.tensor([0])
            _motion_num_frames = torch.tensor([3])
            _motion_dt = torch.tensor([0.02])
            _motion_aa = torch.zeros((3, 9))
            gts = torch.zeros((3, 3, 3))
            grs = torch.tensor([[[0.0, 0.0, 0.0, 1.0]] * 3] * 3)
            dof_pos = torch.zeros((3, 2))

        motion_lib = MotionLibStub()
        offset_rad = math.radians(-5.0)

        _offset_motion_lib_joint_angles(
            motion_lib,
            motion_id=0,
            dof_names=["left_ankle_pitch_joint", "right_ankle_pitch_joint"],
            offsets_rad={
                "left_ankle_pitch_joint": offset_rad,
                "right_ankle_pitch_joint": offset_rad,
            },
        )

        pose_aa = motion_lib._motion_aa.reshape(3, 3, 3)
        torch.testing.assert_close(pose_aa[:, 1, 1], torch.full((3,), offset_rad))
        torch.testing.assert_close(pose_aa[:, 2, 1], torch.full((3,), offset_rad))
        torch.testing.assert_close(motion_lib.dof_pos, torch.full((3, 2), offset_rad))

    def test_reference_ankle_pitch_offset_can_be_set(self) -> None:
        with patch(
            "sys.argv",
            [
                "mujoco_tracking_inference",
                "--model-folder",
                "runs/example",
                "--data-path",
                "motion.pkl",
                "--reference-ankle-pitch-offset-deg",
                "-5",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.reference_ankle_pitch_offset_deg, -5.0)

    def test_scene_xml_contains_ground_and_initial_marker(self) -> None:
        training = load_robot_training_spec("configs/robots/mini3.yaml")
        scene_path = training.robot.xml_file.with_name("scene.xml")

        model, _ = _build_mujoco_model(
            training.robot.xml_file,
            training.to_env_dict(),
            physics_hz=500.0,
            ground_friction=1.25,
            scene_xml_path=scene_path,
            initial_marker_xy=np.asarray([0.4, -0.3]),
        )

        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "initial_position_marker")
        self.assertGreaterEqual(floor_id, 0)
        self.assertGreaterEqual(marker_id, 0)
        np.testing.assert_allclose(model.geom_friction[floor_id], [1.25, 0.005, 0.0001])
        np.testing.assert_allclose(model.geom_pos[marker_id], [0.4, -0.3, 0.025])
        np.testing.assert_allclose(model.geom_rgba[marker_id], [1.0, 0.0, 0.0, 1.0])
        self.assertEqual(int(model.geom_contype[marker_id]), 0)
        self.assertEqual(int(model.geom_conaffinity[marker_id]), 0)

    def test_reference_root_linear_velocity_ablation_can_be_enabled(self) -> None:
        with patch(
            "sys.argv",
            [
                "mujoco_tracking_inference",
                "--model-folder",
                "runs/example",
                "--data-path",
                "motion.pkl",
                "--zero-reference-root-linear-velocity",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.zero_reference_root_linear_velocity)

    def test_reference_root_xy_motion_ablation_can_be_enabled(self) -> None:
        with patch(
            "sys.argv",
            [
                "mujoco_tracking_inference",
                "--model-folder",
                "runs/example",
                "--data-path",
                "motion.pkl",
                "--zero-reference-root-xy-motion",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.zero_reference_root_xy_motion)

    def test_root_xy_motion_is_zeroed_at_motion_lib_storage(self) -> None:
        class MotionLibStub:
            pass

        motion_lib = MotionLibStub()
        motion_lib.gts = torch.tensor(
            [
                [[1.0, 2.0, 0.5], [1.2, 2.3, 0.1]],
                [[4.0, 6.0, 0.6], [4.4, 6.5, 0.2]],
            ]
        )
        motion_lib.gts_t = motion_lib.gts.clone()
        motion_lib.gvs = torch.tensor(
            [
                [[0.5, 0.2, 0.1], [0.8, 0.7, 0.3]],
                [[1.0, -0.4, 0.2], [1.6, 0.1, 0.4]],
            ]
        )
        motion_lib.gvs_t = motion_lib.gvs.clone()
        motion_lib.grvs = motion_lib.gvs[:, 0].clone()
        expected_relative_positions = motion_lib.gts - motion_lib.gts[:, 0:1]
        expected_relative_velocities = motion_lib.gvs - motion_lib.gvs[:, 0:1]
        expected_root_z = motion_lib.gts[:, 0, 2].clone()
        expected_root_vz = motion_lib.gvs[:, 0, 2].clone()

        _zero_motion_lib_root_xy_motion(motion_lib)

        torch.testing.assert_close(motion_lib.gts[:, 0, 0:2], torch.zeros((2, 2)))
        torch.testing.assert_close(motion_lib.gts_t[:, 0, 0:2], torch.zeros((2, 2)))
        torch.testing.assert_close(motion_lib.gvs[:, 0, 0:2], torch.zeros((2, 2)))
        torch.testing.assert_close(motion_lib.gvs_t[:, 0, 0:2], torch.zeros((2, 2)))
        torch.testing.assert_close(motion_lib.grvs[:, 0:2], torch.zeros((2, 2)))
        torch.testing.assert_close(motion_lib.gts - motion_lib.gts[:, 0:1], expected_relative_positions)
        torch.testing.assert_close(motion_lib.gvs - motion_lib.gvs[:, 0:1], expected_relative_velocities)
        torch.testing.assert_close(motion_lib.gts[:, 0, 2], expected_root_z)
        torch.testing.assert_close(motion_lib.gvs[:, 0, 2], expected_root_vz)

    def test_root_xy_motion_is_scaled_at_motion_lib_storage(self) -> None:
        class MotionLibStub:
            pass

        motion_lib = MotionLibStub()
        motion_lib.gts = torch.tensor(
            [
                [[1.0, 2.0, 0.5], [1.2, 2.3, 0.1]],
                [[4.0, 6.0, 0.6], [4.4, 6.5, 0.2]],
            ]
        )
        motion_lib.gts_t = motion_lib.gts.clone()
        motion_lib.gvs = torch.tensor(
            [
                [[0.5, 0.2, 0.1], [0.8, 0.7, 0.3]],
                [[1.0, -0.4, 0.2], [1.6, 0.1, 0.4]],
            ]
        )
        motion_lib.gvs_t = motion_lib.gvs.clone()
        motion_lib.grvs = motion_lib.gvs[:, 0].clone()
        expected_relative_positions = motion_lib.gts - motion_lib.gts[:, 0:1]
        expected_relative_velocities = motion_lib.gvs - motion_lib.gvs[:, 0:1]
        expected_root_xy = motion_lib.gts[:, 0, 0:2].clone() * 5.0
        expected_root_vxy = motion_lib.gvs[:, 0, 0:2].clone() * 5.0
        expected_root_z = motion_lib.gts[:, 0, 2].clone()
        expected_root_vz = motion_lib.gvs[:, 0, 2].clone()

        _scale_motion_lib_root_xy_motion(motion_lib, 5.0)

        torch.testing.assert_close(motion_lib.gts[:, 0, 0:2], expected_root_xy)
        torch.testing.assert_close(motion_lib.gts_t[:, 0, 0:2], expected_root_xy)
        torch.testing.assert_close(motion_lib.gvs[:, 0, 0:2], expected_root_vxy)
        torch.testing.assert_close(motion_lib.gvs_t[:, 0, 0:2], expected_root_vxy)
        torch.testing.assert_close(motion_lib.grvs[:, 0:2], expected_root_vxy)
        torch.testing.assert_close(motion_lib.gts - motion_lib.gts[:, 0:1], expected_relative_positions)
        torch.testing.assert_close(motion_lib.gvs - motion_lib.gvs[:, 0:1], expected_relative_velocities)
        torch.testing.assert_close(motion_lib.gts[:, 0, 2], expected_root_z)
        torch.testing.assert_close(motion_lib.gvs[:, 0, 2], expected_root_vz)

    def test_reference_root_xy_scale_can_be_set(self) -> None:
        with patch(
            "sys.argv",
            [
                "mujoco_tracking_inference",
                "--model-folder",
                "runs/example",
                "--data-path",
                "motion.pkl",
                "--reference-root-xy-scale",
                "5",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.reference_root_xy_scale, 5.0)

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
