from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import mujoco
import numpy as np

from humanoidverse.tools.play_mini3_pkl import load_mini3_pkl_motion
from humanoidverse.tools.play_robot_state_npz import build_qpos_trajectory, load_playback_model
from humanoidverse.utils.robot_spec import load_robot_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = REPO_ROOT / "configs" / "robots" / "mini3.yaml"


class PlayMini3PklTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.robot_spec = load_robot_spec(ROBOT_CONFIG)
        _, cls.model = load_playback_model(cls.robot_spec)

    def _payload(self) -> dict[str, object]:
        root_rot = np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0],
                [-0.1, 0.2, -0.3, 0.9],
            ],
            dtype=np.float64,
        )
        root_rot /= np.linalg.norm(root_rot, axis=1, keepdims=True)
        return {
            "root_pos": np.asarray([[0.0, 0.0, 0.46], [0.1, 0.2, 0.47]], dtype=np.float64),
            "root_rot": root_rot,
            "dof_pos": np.zeros((2, 21), dtype=np.float64),
            "fps": 120.0048,
            "local_body_pos": None,
            "link_body_list": None,
        }

    def test_xyzw_root_rot_is_read_directly_and_mapped_to_mujoco_wxyz(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "motion.pkl"
            payload = self._payload()
            joblib.dump(payload, path)
            loaded = load_mini3_pkl_motion(path, self.robot_spec)

            np.testing.assert_allclose(loaded.motion.root_quat, payload["root_rot"], atol=3.0e-8, rtol=0.0)
            self.assertEqual(loaded.motion.fps, 120.0048)
            qpos = build_qpos_trajectory(loaded.motion, self.robot_spec, self.model)
            free_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.robot_spec.free_joint)
            free_addr = int(self.model.jnt_qposadr[free_id])
            expected = np.asarray(payload["root_rot"])[1, [3, 0, 1, 2]]
            np.testing.assert_allclose(qpos[1, free_addr + 3 : free_addr + 7], expected, atol=1.0e-12)

    def test_wxyz_option_only_reorders_stored_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "motion.pkl"
            payload = self._payload()
            xyzw = np.asarray(payload["root_rot"])
            payload["root_rot"] = xyzw[:, [3, 0, 1, 2]]
            joblib.dump(payload, path)
            loaded = load_mini3_pkl_motion(path, self.robot_spec, quat_order="wxyz")
            np.testing.assert_allclose(loaded.motion.root_quat, xyzw, atol=0.0)

    def test_missing_root_rot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "motion.pkl"
            payload = self._payload()
            payload.pop("root_rot")
            joblib.dump(payload, path)
            with self.assertRaisesRegex(ValueError, "root_rot"):
                load_mini3_pkl_motion(path, self.robot_spec)


if __name__ == "__main__":
    unittest.main()
