from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanoidverse.tools.convert_mini3_csv import convert_csv_file, load_mini3_csv_motion
from humanoidverse.tools.play_mini3_csv import compare_csv_and_npz
from humanoidverse.tools.play_robot_state_npz import build_qpos_trajectory, load_npz_motion, load_playback_model
from humanoidverse.utils.robot_spec import load_robot_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = REPO_ROOT / "configs" / "robots" / "mini3.yaml"


def _write_raw_csv(path: Path, joint_names: list[str]) -> None:
    header = [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
        *(f"{name}_dof" for name in joint_names),
    ]
    dof = np.zeros((3, len(joint_names)), dtype=np.float64)
    dof[1, 0] = 15.0
    dof[1, 12] = -20.0
    dof[2, 16] = 30.0
    rows = [
        [0, 100.0, 200.0, 46.0, 10.0, 20.0, 30.0, *dof[0]],
        [1, 101.0, 201.0, 47.0, 11.0, 21.0, 31.0, *dof[1]],
        [2, 102.0, 202.0, 48.0, 12.0, 22.0, 32.0, *dof[2]],
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


class PlayMini3CsvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.robot_spec = load_robot_spec(ROBOT_CONFIG)
        _, cls.model = load_playback_model(cls.robot_spec)

    def test_raw_csv_qpos_matches_converted_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "motion.csv"
            npz_path = root / "motion.npz"
            _write_raw_csv(csv_path, self.robot_spec.control_joint_names)
            raw = load_mini3_csv_motion(csv_path, self.robot_spec)
            convert_csv_file(csv_path, npz_path, self.robot_spec, verify_output=True)
            _, converted = load_npz_motion(npz_path, self.robot_spec)

            raw_qpos = build_qpos_trajectory(raw.motion, self.robot_spec, self.model)
            converted_qpos = build_qpos_trajectory(converted, self.robot_spec, self.model)
            np.testing.assert_allclose(raw_qpos, converted_qpos, atol=1.0e-7)
            compare_csv_and_npz(raw.motion, npz_path, self.robot_spec, self.model)

    def test_comparison_rejects_modified_npz_dof(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "motion.csv"
            npz_path = root / "motion.npz"
            _write_raw_csv(csv_path, self.robot_spec.control_joint_names)
            raw = load_mini3_csv_motion(csv_path, self.robot_spec)
            convert_csv_file(csv_path, npz_path, self.robot_spec)
            with np.load(npz_path, allow_pickle=False) as source:
                payload = {name: source[name].copy() for name in source.files}
            payload["dof_pos"][1, 0] += 0.01
            np.savez(npz_path, **payload)

            with self.assertRaisesRegex(ValueError, "DOF mismatch"):
                compare_csv_and_npz(raw.motion, npz_path, self.robot_spec, self.model)


if __name__ == "__main__":
    unittest.main()
