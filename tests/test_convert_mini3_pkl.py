from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from omegaconf import OmegaConf

from humanoidverse.tools.convert_mini3_pkl import convert_pkl_dataset, discover_pkl_files
from humanoidverse.tools.data_inspect import inspect_data_source
from humanoidverse.utils.motion_data.adapters import load_robot_state_pkl
from humanoidverse.utils.motion_data.manifest import PER_MOTION_DIRECTORY_INDEX, prepare_motion_manifest
from humanoidverse.utils.motion_data.robot_state_readers import read_robot_state_pkl
from humanoidverse.utils.robot_spec import load_robot_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = REPO_ROOT / "configs" / "robots" / "mini3.yaml"


def _flat_payload(frame_count: int = 5, fps: float = 2.0) -> dict:
    root_rot = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.1, -0.2, 0.3, 0.9],
            [0.2, 0.1, -0.1, 0.95],
            [-0.3, 0.1, 0.2, 0.9],
            [0.0, 0.4, 0.0, 0.9],
        ],
        dtype=np.float64,
    )[:frame_count]
    root_rot /= np.linalg.norm(root_rot, axis=1, keepdims=True)
    return {
        "root_pos": np.stack([np.linspace(0.0, 0.1, frame_count), np.zeros(frame_count), np.full(frame_count, 0.46)], axis=1),
        "root_rot": root_rot,
        "dof_pos": np.zeros((frame_count, 21), dtype=np.float64),
        "fps": fps,
        "local_body_pos": None,
        "link_body_list": None,
    }


class ConvertMini3PklTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.robot_spec = load_robot_spec(ROBOT_CONFIG)

    def test_reader_and_adapter_use_root_rot_directly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "motion.pkl"
            payload = _flat_payload()
            joblib.dump(payload, path)

            loaded = read_robot_state_pkl(path, source_name="unit", robot_spec=self.robot_spec)
            motion = loaded["motion"]
            np.testing.assert_allclose(motion.root_quat, payload["root_rot"], atol=3.0e-8, rtol=0.0)
            self.assertEqual(motion.metadata["root_rotation_source"], "direct root_rot field")

            converted = load_robot_state_pkl(path, source_name="unit", robot_spec=self.robot_spec)
            np.testing.assert_array_equal(converted["motion"]["root_quat"], payload["root_rot"].astype(np.float32))
            self.assertEqual(converted["motion"]["pose_aa"].shape, (5, len(self.robot_spec.body_names), 3))

    def test_conversion_writes_lazy_training_clips_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source" / "nested"
            source_dir.mkdir(parents=True)
            source = source_dir / "motion.pkl"
            payload = _flat_payload()
            joblib.dump(payload, source)
            output_dir = root / "training"
            manifest_path = root / "mini3_pkl.yaml"

            paths = discover_pkl_files(root / "source")
            result = convert_pkl_dataset(
                paths,
                output_dir,
                manifest_path,
                self.robot_spec,
                robot_config=ROBOT_CONFIG,
                clip_seconds=2.0,
                stride_seconds=2.0,
                min_clip_seconds=0.5,
                verify_output=True,
                log_every=1,
            )

            self.assertEqual(result.written_motion_files, 2)
            first_path = output_dir / "motion__clip000.pkl"
            second_path = output_dir / "motion__clip001.pkl"
            self.assertTrue(first_path.is_file())
            self.assertTrue(second_path.is_file())
            first = joblib.load(first_path)["motion__clip000"]
            np.testing.assert_array_equal(first["root_quat"], payload["root_rot"][:4].astype(np.float32))
            self.assertEqual(first["root_trans_offset"].shape[0], 4)
            self.assertEqual(joblib.load(second_path)["motion__clip001"]["root_trans_offset"].shape[0], 1)

            index = json.loads((output_dir / PER_MOTION_DIRECTORY_INDEX).read_text())
            self.assertEqual(index["status"], "complete")
            self.assertEqual(index["motion_files"], 2)
            manifest = OmegaConf.to_container(OmegaConf.load(manifest_path), resolve=True)
            self.assertEqual(manifest["datasets"][0]["storage"], "per_motion_directory")

            prepared = prepare_motion_manifest(manifest_path, cache_root=root / "cache")
            self.assertEqual(prepared.train_data_paths, [str(output_dir.resolve())])
            self.assertEqual(Path(prepared.robot_config_path), ROBOT_CONFIG)

    def test_data_inspect_recognizes_flat_pkl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "motion.pkl"
            joblib.dump(_flat_payload(), path)
            result = inspect_data_source(
                robot_config=ROBOT_CONFIG,
                source=str(path),
                fmt="robot_state_pkl",
                clip_seconds=2.0,
                min_clip_seconds=0.5,
            )
            self.assertEqual(result.format, "robot_state_pkl")
            self.assertEqual(result.root_quat_columns, ["root_rot"])
            self.assertEqual(result.suggested_manifest["datasets"][0]["columns"]["root_quat_order"], "xyzw")
            self.assertEqual(result.estimated_clip_count, 2)


if __name__ == "__main__":
    unittest.main()
