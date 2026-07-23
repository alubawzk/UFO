import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from humanoidverse.mujoco_pico_teleop import (
    SONIC_HEADER_SIZE,
    LatentSmoother,
    Mini3Reference,
    OnlineMini3ReferenceEncoder,
    OnnxPolicyRunner,
    PicoNpzSource,
    PicoSmplFrame,
    RetargetReferenceVisualizer,
    StartupReferenceGrounder,
    rotation_6d_columns_to_quat_wxyz,
    sonic_frame_to_smplx_parameters,
    unpack_sonic_pose_message,
)
from humanoidverse.mujoco_tracking_inference import _joint_layout
from humanoidverse.utils.robot_spec import load_robot_spec
from pico_sim2sim.joyin import GeneralMotionRetargeting
from pico_sim2sim.smplx_model import NeutralSmplxBodyModel
from pico_sim2sim.sonic_protocol import pack_pose_message
from pico_sim2sim.sonic_server import XRT_JOINT_FRAME_CORRECTION, Y_UP_TO_Z_UP, body_poses_to_sonic

_TEST_SMPLX_MODEL: NeutralSmplxBodyModel | None = None


def _test_smplx_model() -> NeutralSmplxBodyModel:
    global _TEST_SMPLX_MODEL
    if _TEST_SMPLX_MODEL is None:
        _TEST_SMPLX_MODEL = NeutralSmplxBodyModel(device="cpu")
    return _TEST_SMPLX_MODEL


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

    def test_integrated_publisher_codec_round_trip(self) -> None:
        payload = {
            "smpl_pose": np.arange(2 * 21 * 3, dtype=np.float32).reshape(2, 21, 3),
            "body_quat_w": np.ones((2, 4), dtype=np.float32),
            "frame_index": np.asarray([4, 5], dtype=np.int64),
        }

        decoded = unpack_sonic_pose_message(pack_pose_message(payload))

        for name, expected in payload.items():
            np.testing.assert_array_equal(decoded[name], expected)

    def test_rejects_truncated_payload(self) -> None:
        fields = [{"name": "value", "dtype": "f32", "shape": [2]}]
        header = json.dumps({"v": 3, "endian": "le", "fields": fields}).encode()
        message = b"pose" + header.ljust(SONIC_HEADER_SIZE, b"\x00") + np.zeros(1, dtype=np.float32).tobytes()
        with self.assertRaisesRegex(ValueError, "Truncated"):
            unpack_sonic_pose_message(message)


class TestSonicSmplxAdapter(unittest.TestCase):
    def test_rotation_6d_uses_sonic_column_layout(self) -> None:
        expected = Rotation.from_euler("xyz", [0.2, -0.3, 0.4])
        encoded = expected.as_matrix()[:, :2].reshape(6)
        actual = Rotation.from_quat(rotation_6d_columns_to_quat_wxyz(encoded), scalar_first=True)
        np.testing.assert_allclose(actual.as_matrix(), expected.as_matrix(), atol=1.0e-7)

    def test_restores_standard_smplx_parameters(self) -> None:
        pose = np.arange(63, dtype=np.float64).reshape(21, 3) / 100.0
        expected_root = Rotation.from_euler("xyz", [0.2, -0.3, 0.4])
        smpl_base = Rotation.from_quat(np.full(4, 0.5), scalar_first=True)
        sonic_root = expected_root * smpl_base.inv()
        translation = np.array([1.0, 2.0, 3.0])
        frame = PicoSmplFrame(
            smpl_pose=pose,
            smpl_joints=np.arange(24 * 3, dtype=np.float64).reshape(24, 3),
            root_quat_wxyz=sonic_root.as_quat(scalar_first=True),
            root_translation=translation,
            timestamp=0.0,
        )

        parameters = sonic_frame_to_smplx_parameters(frame)

        np.testing.assert_allclose(parameters.pose_body, pose.reshape(63))
        np.testing.assert_allclose(Rotation.from_rotvec(parameters.root_orient).as_matrix(), expected_root.as_matrix(), atol=1.0e-7)
        np.testing.assert_allclose(parameters.trans, translation)
        np.testing.assert_array_equal(parameters.betas, np.zeros(16))

    def test_does_not_use_sonic_local_joint_positions_as_smplx_input(self) -> None:
        common = {
            "smpl_pose": np.zeros((21, 3)),
            "root_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
            "root_translation": np.zeros(3),
            "timestamp": 0.0,
        }
        first = sonic_frame_to_smplx_parameters(PicoSmplFrame(smpl_joints=np.zeros((24, 3)), **common))
        second = sonic_frame_to_smplx_parameters(PicoSmplFrame(smpl_joints=np.ones((24, 3)), **common))

        np.testing.assert_array_equal(first.pose_body, second.pose_body)
        np.testing.assert_array_equal(first.root_orient, second.root_orient)
        np.testing.assert_array_equal(first.trans, second.trans)

    def test_integrated_xrt_conversion_matches_sonic_root_convention(self) -> None:
        body_poses = np.zeros((24, 7), dtype=np.float64)
        body_poses[:, 6] = 1.0

        pose, joints, sonic_root = body_poses_to_sonic(body_poses, _test_smplx_model())
        parameters = sonic_frame_to_smplx_parameters(
            PicoSmplFrame(
                smpl_pose=pose,
                smpl_joints=joints,
                root_quat_wxyz=sonic_root,
                root_translation=np.zeros(3),
                timestamp=0.0,
            )
        )

        expected_root = Y_UP_TO_Z_UP * XRT_JOINT_FRAME_CORRECTION
        np.testing.assert_allclose(pose, 0.0, atol=1.0e-7)
        self.assertEqual(joints.shape, (24, 3))
        np.testing.assert_allclose(
            Rotation.from_rotvec(parameters.root_orient).as_matrix(),
            expected_root.as_matrix(),
            atol=1.0e-7,
        )


class TestIntegratedJoyIn(unittest.TestCase):
    def test_real_neutral_smplx_and_mini3_retarget(self) -> None:
        body = _test_smplx_model().forward(
            root_orient=np.zeros(3),
            pose_body=np.zeros((21, 3)),
            trans=np.zeros(3),
        )
        self.assertEqual(body.positions.shape, (22, 3))
        self.assertEqual(body.sonic_positions.shape, (24, 3))
        self.assertTrue(np.all(np.isfinite(body.positions)))
        np.testing.assert_allclose(body.rotations_wxyz[:, 0], 1.0, atol=1.0e-7)
        np.testing.assert_allclose(body.rotations_wxyz[:, 1:], 0.0, atol=1.0e-7)

        retargeter = GeneralMotionRetargeting(
            verbose=False,
            use_velocity_limit=False,
            ik_dt=0.02,
            max_iter=1,
        )
        qpos = retargeter.retarget(body.as_joyin_data(), offset_to_ground=True)

        self.assertEqual(qpos.shape, (28,))
        self.assertTrue(np.all(np.isfinite(qpos)))


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


class TestStartupReferenceGrounder(unittest.TestCase):
    def test_calibrates_once_from_lowest_foot_collision_mesh(self) -> None:
        robot_spec = load_robot_spec("configs/robots/mini3.yaml")
        model = mujoco.MjModel.from_xml_path(robot_spec.xml_path)
        layout = _joint_layout(model, list(robot_spec.control_joint_names))
        grounder = StartupReferenceGrounder(
            model,
            layout,
            list(robot_spec.feet),
            ground_height=0.0,
            enabled=True,
        )
        first = Mini3Reference(
            root_pos=np.array([0.0, 0.0, 0.5]),
            root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            dof_pos=np.zeros(len(robot_spec.control_joint_names)),
        )

        grounded_first = grounder.apply(first)

        self.assertTrue(grounder.calibrated)
        self.assertIsNotNone(grounder.z_offset)
        self.assertAlmostEqual(grounder.lowest_foot_z(grounded_first), 0.0, places=7)
        np.testing.assert_array_equal(first.root_pos, [0.0, 0.0, 0.5])
        first_offset = grounder.z_offset

        second_dof = np.zeros(len(robot_spec.control_joint_names))
        second_dof[4] = 0.2
        second = Mini3Reference(
            root_pos=np.array([0.2, -0.1, 0.6]),
            root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            dof_pos=second_dof,
        )
        grounded_second = grounder.apply(second)

        self.assertEqual(grounder.z_offset, first_offset)
        self.assertAlmostEqual(grounded_second.root_pos[2] - second.root_pos[2], first_offset, places=12)


class TestRetargetReferenceVisualizer(unittest.TestCase):
    def test_renders_translucent_reference_beside_simulated_robot(self) -> None:
        robot_spec = load_robot_spec("configs/robots/mini3.yaml")
        model = mujoco.MjModel.from_xml_path(robot_spec.xml_path)
        layout = _joint_layout(model, list(robot_spec.control_joint_names))
        visualizer = RetargetReferenceVisualizer(model, layout, lateral_offset=1.25, alpha=0.4)
        user_scene = mujoco.MjvScene(model, maxgeom=100)
        reference = Mini3Reference(
            root_pos=np.array([10.0, -5.0, 0.52]),
            root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            dof_pos=np.zeros(len(robot_spec.control_joint_names)),
        )

        geom_count = visualizer.update(user_scene, reference, np.array([2.0, 3.0, 0.48]))

        self.assertGreater(geom_count, 0)
        self.assertEqual(geom_count, user_scene.ngeom)
        np.testing.assert_allclose(visualizer.data.qpos[:3], [2.0, 4.25, 0.52])
        for geom_index in range(geom_count):
            geom = user_scene.geoms[geom_index]
            np.testing.assert_allclose(geom.rgba, [0.05, 0.85, 1.0, 0.4], atol=1.0e-6)
            self.assertEqual(int(geom.category), int(mujoco.mjtCatBit.mjCAT_DECOR))


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


class TestOnnxPolicyRunner(unittest.TestCase):
    def test_runs_pruned_backward_inputs_and_metadata_order(self) -> None:
        class ValueInfo:
            def __init__(self, name: str, shape: list[int | str]):
                self.name = name
                self.shape = shape

        class FakeSession:
            def __init__(self, path: str, providers: list[str]):
                self.is_backward = Path(path).name == "backward_encoder.onnx"
                self.providers = list(providers)

            def get_inputs(self) -> list[ValueInfo]:
                if self.is_backward:
                    # last_action is intentionally absent: ONNX export prunes
                    # inputs unused by this checkpoint's backward filter.
                    return [ValueInfo("state", ["batch", 3]), ValueInfo("privileged_state", ["batch", 6])]
                return [ValueInfo("actor_obs", ["batch", 14])]

            def get_outputs(self) -> list[ValueInfo]:
                return [ValueInfo("z" if self.is_backward else "action", ["batch", 5 if self.is_backward else 2])]

            def get_providers(self) -> list[str]:
                return self.providers

            def run(self, _outputs: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
                if self.is_backward:
                    self_outer.assertEqual(set(feed), {"state", "privileged_state"})
                    return [np.full((feed["state"].shape[0], 5), 2.0, dtype=np.float32)]
                actor_obs = feed["actor_obs"]
                self_outer.assertEqual(actor_obs.shape, (1, 14))
                return [actor_obs[:, :2].copy()]

        self_outer = self
        fake_ort = types.SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
            InferenceSession=FakeSession,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_folder = Path(temporary_directory)
            export_dir = model_folder / "exported"
            export_dir.mkdir()
            (export_dir / "TinyPolicy.onnx").write_bytes(b"actor")
            (export_dir / "backward_encoder.onnx").write_bytes(b"backward")
            (export_dir / "TinyPolicy.meta.json").write_text(
                json.dumps(
                    {
                        "actor_input_keys": ["state", "last_action", "history_actor"],
                        "actor_input_dims": {"state": 3, "last_action": 2, "history_actor": 4},
                        "actor_obs_dim": 14,
                        "z_dim": 5,
                        "output_action_dim": 2,
                        "control_joint_names": ["joint_a", "joint_b"],
                    }
                )
            )
            checkpoint_config = model_folder / "checkpoint" / "model" / "config.json"
            checkpoint_config.parent.mkdir(parents=True)
            checkpoint_config.write_text(json.dumps({"archi": {"norm_z": True}}))

            with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                runner = OnnxPolicyRunner(
                    model_folder,
                    None,
                    requested_device="cuda:0",
                    provider="auto",
                    control_joint_names=["joint_a", "joint_b"],
                )
                latent = runner.encode_reference(
                    {
                        "state": torch.ones(1, 3),
                        "last_action": torch.ones(1, 2),
                        "privileged_state": torch.ones(1, 6),
                    }
                )
                action = runner.act(
                    {
                        "state": torch.tensor([[3.0, 4.0, 5.0]]),
                        "last_action": torch.ones(1, 2),
                        "history_actor": torch.ones(1, 4),
                    },
                    latent,
                )

        self.assertEqual(runner.tensor_device, "cpu")
        self.assertTrue(runner.norm_z)
        self.assertEqual(runner.backward_input_names, ["state", "privileged_state"])
        np.testing.assert_array_equal(latent.numpy(), np.full((1, 5), 2.0, dtype=np.float32))
        np.testing.assert_array_equal(action, [3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
