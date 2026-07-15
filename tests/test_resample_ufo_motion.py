from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from humanoidverse.tools.resample_ufo_motion import discover_ufo_motion_files, resample_ufo_dataset
from humanoidverse.utils.motion_data.manifest import PER_MOTION_DIRECTORY_INDEX, prepare_motion_manifest
from humanoidverse.utils.motion_data.resample import resample_ufo_motion_record
from humanoidverse.utils.motion_lib.motion_lib_base import (
    MotionLibBase,
    _expert_grid_contract_from_motion_file,
    _motion_files_from_directory,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = REPO_ROOT / "configs" / "robots" / "mini3.yaml"


def _record(*, frames: int = 5, fps: float = 100.0, offset: float = 0.0) -> dict:
    times = np.arange(frames, dtype=np.float32) / fps
    angles = np.linspace(0.0, np.pi / 2.0, frames, dtype=np.float32)
    root_quat = Rotation.from_rotvec(np.stack([np.zeros(frames), np.zeros(frames), angles], axis=1)).as_quat().astype(np.float32)
    pose_aa = np.zeros((frames, 2, 3), dtype=np.float32)
    pose_aa[:, 0, 2] = angles
    pose_aa[:, 1, 1] = angles * 0.5
    return {
        "root_trans_offset": np.stack([times + offset, np.zeros(frames), np.ones(frames)], axis=1).astype(np.float32),
        "pose_aa": pose_aa,
        "fps": fps,
        "dof_pos": angles[:, None],
        "root_quat": root_quat,
        "contact": np.asarray([False, False, True, True, False][:frames]),
        "metadata": {"root_quat_order": "xyzw", "fixture": True},
    }


class ResampleUfoMotionTest(unittest.TestCase):
    def test_resamples_linear_rotation_and_nearest_fields(self) -> None:
        output = resample_ufo_motion_record(_record(), 50.0, source_name="unit")

        self.assertEqual(output["root_trans_offset"].shape, (3, 3))
        np.testing.assert_allclose(output["root_trans_offset"][:, 0], np.asarray([0.0, 0.02, 0.04]), atol=1.0e-7)
        np.testing.assert_allclose(output["dof_pos"][:, 0], np.asarray([0.0, np.pi / 4.0, np.pi / 2.0]), atol=1.0e-6)
        np.testing.assert_array_equal(output["contact"], np.asarray([False, True, False]))
        output_angles = Rotation.from_quat(output["root_quat"]).as_rotvec()[:, 2]
        np.testing.assert_allclose(output_angles, np.asarray([0.0, np.pi / 4.0, np.pi / 2.0]), atol=1.0e-6)
        np.testing.assert_allclose(output["pose_aa"][:, 1, 1], np.asarray([0.0, np.pi / 8.0, np.pi / 4.0]), atol=1.0e-6)
        self.assertEqual(output["fps"], 50.0)
        self.assertEqual(output["metadata"]["resampling"]["source_frame_count"], 5)
        self.assertEqual(output["metadata"]["resampling"]["legacy_expert_sample_count"], 2)

    def test_target_grid_does_not_extrapolate_past_source(self) -> None:
        output = resample_ufo_motion_record(_record(frames=4, fps=120.0), 50.0, source_name="unit")
        self.assertEqual(output["root_trans_offset"].shape[0], 2)
        self.assertAlmostEqual(float(output["root_trans_offset"][-1, 0]), 0.02, places=7)
        self.assertLessEqual((output["root_trans_offset"].shape[0] - 1) / 50.0, (4 - 1) / 120.0)

    def test_preserves_wxyz_quaternion_order(self) -> None:
        record = _record()
        record["root_quat"] = record["root_quat"][:, [3, 0, 1, 2]]
        record["metadata"]["root_quat_order"] = "wxyz"
        output = resample_ufo_motion_record(record, 50.0, source_name="unit")
        xyzw = output["root_quat"][:, [1, 2, 3, 0]]
        angles = Rotation.from_quat(xyzw).as_rotvec()[:, 2]
        np.testing.assert_allclose(angles, np.asarray([0.0, np.pi / 4.0, np.pi / 2.0]), atol=1.0e-6)

    def test_rejects_one_frame_and_ambiguous_time_series(self) -> None:
        with self.assertRaisesRegex(ValueError, "one-frame"):
            resample_ufo_motion_record(_record(frames=1), 50.0, source_name="unit")

        with self.assertRaisesRegex(ValueError, "only one frame"):
            resample_ufo_motion_record(_record(frames=2, fps=120.0), 50.0, source_name="unit")

        wrong_length = _record()
        wrong_length["dof_pos"] = wrong_length["dof_pos"][:-1]
        with self.assertRaisesRegex(ValueError, "Known time-series field 'dof_pos'"):
            resample_ufo_motion_record(wrong_length, 50.0, source_name="unit")

        unknown = _record()
        unknown["mystery"] = np.zeros((5, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Unknown ndarray field 'mystery'"):
            resample_ufo_motion_record(unknown, 50.0, source_name="unit")

    def test_directory_conversion_writes_stable_index_manifest_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            source.mkdir()
            joblib.dump({"b": _record(offset=1.0)}, source / "b.pkl")
            joblib.dump({"a": _record(offset=2.0)}, source / "a.pkl")
            output = root / "output"
            manifest = root / "mini3_50fps.yaml"

            paths = discover_ufo_motion_files(source)
            result = resample_ufo_dataset(
                paths,
                output,
                target_fps=50.0,
                workers=2,
                verify_output=True,
                log_every=1,
                manifest_path=manifest,
                robot_config=ROBOT_CONFIG,
            )
            self.assertEqual(result.written_motion_files, 2)
            self.assertEqual(result.reused_motion_files, 0)
            index = json.loads((output / PER_MOTION_DIRECTORY_INDEX).read_text())
            self.assertEqual(index["status"], "complete")
            self.assertEqual([item["motion_key"] for item in index["motions"]], ["a", "b"])
            self.assertEqual(index["format"], "ufo_per_motion_directory_v2")
            self.assertEqual([item["source_local_id"] for item in index["motions"]], [0, 1])
            self.assertEqual([item["relative_path"] for item in index["motions"]], ["a.pkl", "b.pkl"])
            self.assertNotIn("global_motion_id", index["motions"][0])
            self.assertTrue(index["motions"][0]["source_content_sha256"])
            self.assertEqual(index["motions"][0]["expert_control_dt_seconds"], 0.02)
            self.assertTrue(index["motion_index_sha256"])
            prepared = prepare_motion_manifest(manifest, cache_root=root / "cache")
            self.assertEqual(prepared.train_data_paths, [str(output.resolve())])

            resumed = resample_ufo_dataset(
                paths,
                output,
                target_fps=50.0,
                workers=2,
                log_every=2,
                manifest_path=manifest,
                robot_config=ROBOT_CONFIG,
            )
            self.assertEqual(resumed.written_motion_files, 0)
            self.assertEqual(resumed.reused_motion_files, 2)

            source_path = source / "a.pkl"
            original_size = source_path.stat().st_size
            joblib.dump({"a": _record(offset=7.0)}, source_path)
            self.assertEqual(source_path.stat().st_size, original_size)
            refreshed = resample_ufo_dataset(paths, output, target_fps=50.0, workers=2, log_every=2)
            self.assertEqual(refreshed.written_motion_files, 1)
            self.assertEqual(refreshed.reused_motion_files, 1)
            self.assertAlmostEqual(float(joblib.load(output / "a.pkl")["a"]["root_trans_offset"][0, 0]), 7.0)

    def test_rejects_incomplete_input_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir)
            joblib.dump({"a": _record()}, source / "a.pkl")
            (source / PER_MOTION_DIRECTORY_INDEX).write_text(
                json.dumps({"format": "ufo_per_motion_directory_v1", "status": "in_progress", "motion_files": 1})
            )
            with self.assertRaisesRegex(ValueError, "status='complete'"):
                discover_ufo_motion_files(source)

    def test_motionlib_uses_resampled_expert_grid_contract(self) -> None:
        motion_lib = MotionLibBase.__new__(MotionLibBase)
        motion_lib._sim_fps = 50.0
        motion_lib._motion_lengths = torch.tensor([9.98, 0.04], dtype=torch.float32)
        motion_lib._motion_expert_sample_counts = torch.tensor([500, -1], dtype=torch.int64)
        motion_lib._motion_expert_control_dts = torch.tensor([0.02, float("nan")], dtype=torch.float32)

        counts = motion_lib.get_expert_sample_count(control_dt=0.02)
        self.assertEqual(counts.tolist(), [500, 2])
        self.assertEqual(motion_lib.get_motion_num_steps().tolist(), [500, 2])
        self.assertEqual(int(motion_lib.get_expert_sample_count(0, control_dt=0.01)), 998)

    def test_motionlib_strictly_validates_resampling_metadata(self) -> None:
        output = resample_ufo_motion_record(_record(), 50.0, source_name="unit")
        self.assertEqual(_expert_grid_contract_from_motion_file(output, 3, 50.0), (2, 0.02))

        wrong_algorithm = deepcopy(output)
        wrong_algorithm["metadata"]["resampling"]["algorithm"] = "stale_algorithm"
        with self.assertRaisesRegex(ValueError, "algorithm"):
            _expert_grid_contract_from_motion_file(wrong_algorithm, 3, 50.0)

        wrong_count = deepcopy(output)
        wrong_count["metadata"]["resampling"]["legacy_expert_sample_count"] = 1
        with self.assertRaisesRegex(ValueError, "recomputed count"):
            _expert_grid_contract_from_motion_file(wrong_count, 3, 50.0)

        wrong_source_grid = deepcopy(output)
        wrong_source_grid["metadata"]["resampling"]["source_frame_count"] = 4
        with self.assertRaisesRegex(ValueError, "target frame count"):
            _expert_grid_contract_from_motion_file(wrong_source_grid, 3, 50.0)

    def test_motionlib_uses_v2_index_paths_and_ignores_unindexed_pkls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "nested"
            nested.mkdir()
            joblib.dump({"a": _record()}, nested / "a.pkl")
            joblib.dump({"z": _record()}, root / "z.pkl")
            joblib.dump({"extra": _record()}, root / "extra.pkl")
            (root / PER_MOTION_DIRECTORY_INDEX).write_text(
                json.dumps(
                    {
                        "format": "ufo_per_motion_directory_v2",
                        "status": "complete",
                        "motion_files": 2,
                        "motions": [
                            {"source_local_id": 0, "motion_key": "a", "relative_path": "nested/a.pkl"},
                            {"source_local_id": 1, "motion_key": "z", "relative_path": "z.pkl"},
                        ],
                    }
                )
            )

            expected = [str(nested / "a.pkl"), str(root / "z.pkl")]
            self.assertEqual(_motion_files_from_directory(root), expected)

            motion_lib = MotionLibBase.__new__(MotionLibBase)
            motion_lib._device = torch.device("cpu")
            motion_lib.load_data(str(root))
            self.assertEqual(motion_lib._motion_data_keys.tolist(), expected)

    def test_v2_index_rejects_any_missing_motion_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            joblib.dump({"a": _record()}, root / "a.pkl")
            (root / PER_MOTION_DIRECTORY_INDEX).write_text(
                json.dumps(
                    {
                        "format": "ufo_per_motion_directory_v2",
                        "status": "complete",
                        "motion_files": 2,
                        "motions": [
                            {"source_local_id": 0, "motion_key": "a", "relative_path": "a.pkl"},
                            {"source_local_id": 1, "motion_key": "missing", "relative_path": "missing.pkl"},
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "missing motion files"):
                _motion_files_from_directory(root)

    def test_programmatic_api_rejects_cross_directory_name_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            first_path = first / "a.pkl"
            second_path = second / "a.pkl"
            joblib.dump({"a": _record()}, first_path)
            joblib.dump({"a": _record()}, second_path)
            with self.assertRaisesRegex(ValueError, "same directory"):
                resample_ufo_dataset([first_path, second_path], root / "output")

    def test_rejects_in_place_resampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir)
            path = source / "a.pkl"
            joblib.dump({"a": _record()}, path)
            with self.assertRaisesRegex(ValueError, "in-place"):
                resample_ufo_dataset([path], source)


if __name__ == "__main__":
    unittest.main()
