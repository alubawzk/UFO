from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from humanoidverse.tools.play_robot_state_npz import (
    build_qpos_trajectory,
    load_npz_motion,
    load_playback_model,
    validate_forward_kinematics,
    validate_joint_limits,
)
from humanoidverse.utils.motion_data.robot_state import RobotStateMotion
from humanoidverse.utils.robot_spec import load_robot_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = REPO_ROOT / "configs" / "robots" / "mini3.yaml"


class PlayRobotStateNpzTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.robot_spec = load_robot_spec(ROBOT_CONFIG)
        cls.scene_path, cls.model = load_playback_model(cls.robot_spec)

    def _motion(self) -> RobotStateMotion:
        frame_count = 3
        root_pos = np.asarray(
            [
                [1.0, 2.0, 0.46],
                [1.1, 2.1, 0.47],
                [1.2, 2.2, 0.48],
            ],
            dtype=np.float32,
        )
        root_quat = np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.1, 0.2, 0.3, 0.9],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        dof_pos = np.zeros((frame_count, len(self.robot_spec.control_joint_names)), dtype=np.float32)
        dof_pos[1, 0] = 0.25
        dof_pos[1, 12] = -0.5
        dof_pos[2, 16] = 0.75
        return RobotStateMotion(
            motion_key="synthetic",
            root_pos=root_pos,
            root_quat=root_quat,
            dof_pos=dof_pos,
            fps=120.0,
            joint_names=list(self.robot_spec.control_joint_names),
            source="test",
        )

    def test_qpos_mapping_uses_mujoco_wxyz_and_xml_joint_addresses(self) -> None:
        motion = self._motion()
        qpos = build_qpos_trajectory(motion, self.robot_spec, self.model)
        self.assertEqual(qpos.shape, (3, 28))

        free_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.robot_spec.free_joint)
        free_addr = int(self.model.jnt_qposadr[free_id])
        np.testing.assert_allclose(qpos[:, free_addr : free_addr + 3], motion.root_pos, atol=1.0e-7)
        expected_wxyz = np.asarray([0.9, 0.1, 0.2, 0.3], dtype=np.float64)
        expected_wxyz /= np.linalg.norm(expected_wxyz)
        np.testing.assert_allclose(qpos[1, free_addr + 3 : free_addr + 7], expected_wxyz, atol=1.0e-7)

        for dof_index, joint_name in enumerate(self.robot_spec.control_joint_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            np.testing.assert_allclose(qpos[:, qpos_addr], motion.dof_pos[:, dof_index])

        checked = validate_forward_kinematics(self.model, qpos, start_frame=0, end_frame=3, stride=2)
        self.assertEqual(checked, 2)

    def test_joint_limit_violation_is_rejected(self) -> None:
        motion = self._motion()
        motion.dof_pos[1, 0] = 10.0
        with self.assertRaisesRegex(ValueError, "left_hip_pitch_joint"):
            validate_joint_limits(motion, self.robot_spec)

    def test_npz_reader_reorders_named_joints_before_playback(self) -> None:
        motion = self._motion()
        reversed_names = list(reversed(self.robot_spec.control_joint_names))
        reverse_indices = [self.robot_spec.control_joint_names.index(name) for name in reversed_names]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reordered.npz"
            np.savez(
                path,
                root_pos=motion.root_pos,
                root_quat=motion.root_quat,
                dof_pos=motion.dof_pos[:, reverse_indices],
                joint_names=np.asarray(reversed_names),
                fps=np.asarray(120.0, dtype=np.float32),
            )
            _, loaded = load_npz_motion(path, self.robot_spec)
            self.assertEqual(loaded.joint_names, self.robot_spec.control_joint_names)
            np.testing.assert_allclose(loaded.dof_pos, motion.dof_pos)


if __name__ == "__main__":
    unittest.main()
