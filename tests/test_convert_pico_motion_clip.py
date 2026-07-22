from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import mujoco
import numpy as np

from humanoidverse.tools.convert_pico_motion_clip import (
    PICO_BODY_NAMES,
    convert_pico_clips,
    discover_pico_clips,
    load_pico_clip,
)
from humanoidverse.utils.motion_data.schema import validate_ufo_motion_dict
from humanoidverse.utils.robot_spec import load_robot_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = REPO_ROOT / "configs" / "robots" / "mini3.yaml"


def _write_reachable_pico_clip(path: Path, *, axes_version: int = 3) -> None:
    model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / "humanoidverse/data/robots/mini3_mjlab/mini3.xml"))
    data = mujoco.MjData(model)
    source_to_target = {
        "pelvis": ("base_link", np.zeros(3)),
        "left_ankle_roll_link": ("left_ankle_roll_link", np.zeros(3)),
        "right_ankle_roll_link": ("right_ankle_roll_link", np.zeros(3)),
        "left_wrist_yaw_link": ("left_elbow_pitch_link", np.asarray([0.107, -0.018, 0.0085])),
        "right_wrist_yaw_link": ("right_elbow_pitch_link", np.asarray([0.107, 0.018, 0.0085])),
    }
    joint_frames = [
        {
            "left_hip_pitch_joint": -0.075,
            "left_knee_pitch_joint": 0.15,
            "left_ankle_pitch_joint": -0.075,
            "right_hip_pitch_joint": -0.075,
            "right_knee_pitch_joint": 0.15,
            "right_ankle_pitch_joint": -0.075,
            "left_shoulder_pitch_joint": -0.3,
            "left_elbow_pitch_joint": 0.8,
            "right_shoulder_pitch_joint": -0.25,
            "right_elbow_pitch_joint": 0.7,
        },
        {
            "left_hip_pitch_joint": -0.3,
            "left_knee_pitch_joint": 0.6,
            "left_ankle_pitch_joint": -0.3,
            "right_hip_pitch_joint": -0.25,
            "right_knee_pitch_joint": 0.5,
            "right_ankle_pitch_joint": -0.25,
            "left_shoulder_pitch_joint": -0.5,
            "left_elbow_pitch_joint": 1.1,
            "right_shoulder_pitch_joint": -0.45,
            "right_elbow_pitch_joint": 1.0,
        },
        {
            "left_hip_pitch_joint": -0.15,
            "left_knee_pitch_joint": 0.3,
            "left_ankle_pitch_joint": -0.15,
            "right_hip_pitch_joint": -0.2,
            "right_knee_pitch_joint": 0.4,
            "right_ankle_pitch_joint": -0.2,
            "left_shoulder_pitch_joint": -0.2,
            "left_elbow_pitch_joint": 0.9,
            "right_shoulder_pitch_joint": -0.15,
            "right_elbow_pitch_joint": 0.85,
        },
    ]
    positions = np.empty((len(joint_frames), len(PICO_BODY_NAMES), 3), dtype=np.float32)
    quaternions = np.empty((len(joint_frames), len(PICO_BODY_NAMES), 4), dtype=np.float32)
    for frame_idx, joint_values in enumerate(joint_frames):
        data.qpos[:] = model.qpos0
        data.qpos[0] = 0.02 * frame_idx
        for joint_name, value in joint_values.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            data.qpos[model.jnt_qposadr[joint_id]] = value
        mujoco.mj_forward(model, data)
        for body_idx, source_name in enumerate(PICO_BODY_NAMES):
            body_name, local_point = source_to_target[source_name]
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            rotation = data.xmat[body_id].reshape(3, 3)
            positions[frame_idx, body_idx] = data.xpos[body_id] + rotation @ local_point
            quaternions[frame_idx, body_idx] = data.xquat[body_id]

    np.savez(
        path,
        body_pos_w=positions,
        body_quat_w=quaternions,
        body_names=np.asarray(PICO_BODY_NAMES),
        fps=np.float32(50.0),
        dt=np.float32(0.02),
        source=np.asarray("pico_motion_clip"),
        pico_position_axes_version=np.int32(axes_version),
        body_state_frame=np.asarray("g1_robotics_zup_v1"),
    )


class ConvertPicoMotionClipTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.robot_spec = load_robot_spec(ROBOT_CONFIG)

    def test_conversion_writes_motionlib_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "reachable.npz"
            output = root / "converted.pkl"
            _write_reachable_pico_clip(source)

            self.assertEqual(discover_pico_clips(root), [source])
            clip = load_pico_clip(source)
            self.assertEqual(clip.body_pos_w.shape, (3, 5, 3))
            results = convert_pico_clips(
                [source],
                output,
                self.robot_spec,
                scale=1.0,
                max_iterations=60,
                overwrite=True,
            )

            self.assertEqual(list(results), ["reachable"])
            result = results["reachable"]
            self.assertLess(max(result.mean_position_error_m.values()), 0.03)
            payload = validate_ufo_motion_dict(joblib.load(output), "test-output")
            motion = payload["reachable"]
            self.assertEqual(motion["root_trans_offset"].shape, (3, 3))
            self.assertEqual(motion["pose_aa"].shape, (3, len(self.robot_spec.body_names), 3))
            self.assertEqual(motion["dof_pos"].shape, (3, len(self.robot_spec.control_joint_names)))
            self.assertTrue(motion["metadata"]["approximate_retarget"])
            self.assertEqual(
                motion["metadata"]["target_mapping"]["left_wrist_yaw_link"]["target_body"],
                "left_elbow_pitch_link",
            )

    def test_loader_rejects_old_position_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "old.npz"
            _write_reachable_pico_clip(source, axes_version=2)
            with self.assertRaisesRegex(ValueError, "pico_position_axes_version=2"):
                load_pico_clip(source)


if __name__ == "__main__":
    unittest.main()
