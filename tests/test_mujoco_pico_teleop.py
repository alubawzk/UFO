import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from humanoidverse.mujoco_pico_teleop import (
    SONIC_HEADER_SIZE,
    LatentSmoother,
    Mini3Reference,
    OnlineMini3ReferenceEncoder,
    PicoNpzSource,
    PicoSmplFrame,
    rotation_6d_columns_to_quat_wxyz,
    sonic_frame_to_joyin_pose,
    unpack_sonic_pose_message,
)
from humanoidverse.utils.robot_spec import load_robot_spec


class TestSonicWireFormat(unittest.TestCase):
    def test_unpack_pose_message(self) -> None:
        pose = np.arange(2 * 21 * 3, dtype=np.float32).reshape(2, 21, 3)
        frame_index = np.array([7, 8], dtype=np.int64)
        fields = [
            {"name": "smpl_pose", "dtype": "f32", "shape": list(pose.shape)},
            {"name": "frame_index", "dtype": "i64", "shape": list(frame_index.shape)},
        ]
        header = json.dumps({"v": 3, "endian": "le", "count": 1, "fields": fields}).encode()
        message = b"pose" + header.ljust(SONIC_HEADER_SIZE, b"\x00") + pose.tobytes() + frame_index.tobytes()

        decoded = unpack_sonic_pose_message(message)

        np.testing.assert_array_equal(decoded["smpl_pose"], pose)
        np.testing.assert_array_equal(decoded["frame_index"], frame_index)
        self.assertEqual(decoded["version"], 3)

    def test_rejects_truncated_payload(self) -> None:
        fields = [{"name": "value", "dtype": "f32", "shape": [2]}]
        header = json.dumps({"v": 3, "endian": "le", "fields": fields}).encode()
        message = b"pose" + header.ljust(SONIC_HEADER_SIZE, b"\x00") + np.zeros(1, dtype=np.float32).tobytes()
        with self.assertRaisesRegex(ValueError, "Truncated"):
            unpack_sonic_pose_message(message)


class TestSonicSmplAdapter(unittest.TestCase):
    def test_rotation_6d_uses_sonic_column_layout(self) -> None:
        expected = Rotation.from_euler("xyz", [0.2, -0.3, 0.4])
        encoded = expected.as_matrix()[:, :2].reshape(6)
        actual = Rotation.from_quat(rotation_6d_columns_to_quat_wxyz(encoded), scalar_first=True)
        np.testing.assert_allclose(actual.as_matrix(), expected.as_matrix(), atol=1.0e-7)

    def test_builds_global_smpl_orientations_for_joyin(self) -> None:
        pose = np.zeros((21, 3), dtype=np.float64)
        pose[0] = [0.0, 0.0, np.pi / 2.0]  # left hip, child of pelvis
        pose[3] = [np.pi / 2.0, 0.0, 0.0]  # left knee, child of left hip
        joints = np.arange(24 * 3, dtype=np.float64).reshape(24, 3) / 100.0
        translation = np.array([1.0, 2.0, 3.0])
        frame = PicoSmplFrame(
            smpl_pose=pose,
            smpl_joints=joints,
            root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            root_translation=translation,
            timestamp=0.0,
        )

        result = sonic_frame_to_joyin_pose(frame)

        left_hip = Rotation.from_quat(result["left_hip"][1], scalar_first=True)
        left_knee = Rotation.from_quat(result["left_knee"][1], scalar_first=True)
        expected_hip = Rotation.from_rotvec(pose[0])
        expected_knee = expected_hip * Rotation.from_rotvec(pose[3])
        np.testing.assert_allclose(left_hip.as_matrix(), expected_hip.as_matrix(), atol=1.0e-7)
        np.testing.assert_allclose(left_knee.as_matrix(), expected_knee.as_matrix(), atol=1.0e-7)
        np.testing.assert_allclose(result["left_knee"][0], joints[4] + translation)
        self.assertEqual(len(result), 12)


class TestPicoNpzSource(unittest.TestCase):
    def test_reads_frames_loops_and_restores_root_displacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "walking_v2.npz"
            rotations = Rotation.from_euler("z", [0.0, 0.1]).as_matrix()
            anchor = rotations[:, :, :2].reshape(2, 6).astype(np.float32)
            sparse_root = np.zeros((2, 5, 3), dtype=np.float32)
            sparse_root[1, 0] = [0.4, -0.2, 0.1]
            np.savez(
                path,
                sonic_smpl_pose=np.zeros((2, 21, 3), dtype=np.float32),
                sonic_smpl_joints=np.zeros((2, 24, 3), dtype=np.float32),
                sonic_smpl_anchor_orientation=anchor,
                body_pos_w=sparse_root,
                fps=np.array(50.0, dtype=np.float32),
                timestamp_monotonic=np.array([10.0, 10.02], dtype=np.float64),
            )
            source = PicoNpzSource(path, start_frame=0, loop=True, root_motion=True)

            first = source.receive()
            second = source.receive()
            looped = source.receive()

            self.assertTrue(first.sequence_reset)
            self.assertFalse(second.sequence_reset)
            self.assertTrue(looped.sequence_reset)
            np.testing.assert_allclose(first.root_translation, 0.0)
            np.testing.assert_allclose(second.root_translation, [0.4, -0.2, 0.1])
            np.testing.assert_allclose(second.root_quat_wxyz, Rotation.from_euler("z", 0.1).as_quat(scalar_first=True))


class TestOnlineMini3ReferenceEncoder(unittest.TestCase):
    def test_builds_training_aligned_backward_observation(self) -> None:
        robot_spec = load_robot_spec("configs/robots/mini3.yaml")
        config = OmegaConf.create(
            {
                "use_contact_in_obs_max": False,
                "obs": {
                    "use_obs_filter": True,
                    "obs_scales": {"base_ang_vel": 0.25},
                },
            }
        )
        default_angles = {name: 0.0 for name in robot_spec.control_joint_names}
        encoder = OnlineMini3ReferenceEncoder(
            Path(robot_spec.xml_path),
            list(robot_spec.control_joint_names),
            list(robot_spec.body_names),
            default_angles,
            config,
            device="cpu",
            dt=0.02,
            root_height_obs=False,
        )
        first = Mini3Reference(
            root_pos=np.array([0.0, 0.0, 0.5]),
            root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            dof_pos=np.zeros(len(robot_spec.control_joint_names)),
        )
        second_dof = np.zeros(len(robot_spec.control_joint_names))
        second_dof[0] = 0.1
        second = Mini3Reference(
            root_pos=np.array([0.01, 0.0, 0.5]),
            root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            dof_pos=second_dof,
        )

        first_obs = encoder.backward_observation(first)
        second_obs = encoder.backward_observation(second)

        self.assertEqual(first_obs["state"].shape, (1, 48))
        self.assertEqual(first_obs["last_action"].shape, (1, 21))
        self.assertEqual(first_obs["privileged_state"].shape, (1, 357))
        self.assertTrue(torch.allclose(first_obs["state"][:, 21:42], torch.zeros(1, 21)))
        self.assertAlmostEqual(second_obs["state"][0, 21].item(), 5.0, places=5)
        for value in second_obs.values():
            self.assertTrue(torch.isfinite(value).all())


class TestLatentSmoother(unittest.TestCase):
    def test_weighted_average_is_renormalized_to_fb_radius(self) -> None:
        smoother = LatentSmoother(3, 0.8, renormalize=True)
        first = torch.zeros(1, 16)
        first[0, 0] = 4.0
        second = torch.zeros(1, 16)
        second[0, 1] = 4.0
        smoother.update(first)

        result = smoother.update(second)

        self.assertAlmostEqual(torch.linalg.norm(result).item(), 4.0, places=5)
        self.assertGreater(result[0, 1].item(), result[0, 0].item())


if __name__ == "__main__":
    unittest.main()
