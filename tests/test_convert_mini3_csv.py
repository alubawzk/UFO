from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from humanoidverse.tools.convert_mini3_csv import (
    Mini3CsvConversionError,
    convert_csv_file,
    discover_csv_files,
)
from humanoidverse.utils.motion_data.robot_state_readers import read_robot_state_npz
from humanoidverse.utils.robot_spec import load_robot_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = REPO_ROOT / "configs" / "robots" / "mini3.yaml"


def _write_mini3_csv(
    path: Path,
    joint_names: list[str],
    *,
    missing_column: str | None = None,
) -> None:
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
    joint_values = np.zeros((3, len(joint_names)), dtype=np.float64)
    # Exercise tolerance for tiny source-side limit rounding. This is less
    # than 0.01 degrees beyond the precise XML limit and should be clipped.
    joint_values[0, 0] = 105.005
    joint_values[1, 12] = 15.0
    joint_values[2, 16] = 30.0
    rows = [
        [0, 100.0, 200.0, 46.305, 10.0, 20.0, 30.0, *joint_values[0]],
        [1, 101.0, 202.0, 47.305, 11.0, 21.0, 31.0, *joint_values[1]],
        [2, 102.0, 204.0, 48.305, 12.0, 22.0, 32.0, *joint_values[2]],
    ]
    if missing_column is not None:
        column_index = header.index(missing_column)
        header.pop(column_index)
        rows = [row[:column_index] + row[column_index + 1 :] for row in rows]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


class ConvertMini3CsvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.robot_spec = load_robot_spec(ROBOT_CONFIG)

    def test_conversion_uses_confirmed_units_and_intrinsic_xyz(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "mini3_motion.csv"
            output = root / "mini3_motion.npz"
            _write_mini3_csv(source, self.robot_spec.control_joint_names)

            result = convert_csv_file(
                source,
                output,
                self.robot_spec,
                verify_output=True,
            )

            self.assertEqual(result.frames, 3)
            self.assertEqual(result.clipped_joint_values, 1)
            with np.load(output, allow_pickle=False) as data:
                np.testing.assert_allclose(data["root_pos"][0], [1.0, 2.0, 0.46305], atol=1.0e-6)
                expected_quat = Rotation.from_euler("XYZ", [10.0, 20.0, 30.0], degrees=True).as_quat()
                np.testing.assert_allclose(data["root_quat"][0], expected_quat, atol=1.0e-7)
                self.assertTrue(np.all(np.sum(data["root_quat"][1:] * data["root_quat"][:-1], axis=1) >= 0.0))
                self.assertEqual(data["dof_pos"].shape, (3, 21))
                self.assertAlmostEqual(
                    float(data["dof_pos"][0, 0]),
                    float(self.robot_spec.joint_ranges["left_hip_pitch_joint"][1]),
                    places=6,
                )
                self.assertAlmostEqual(float(data["dof_pos"][1, 12]), float(np.deg2rad(15.0)), places=6)
                self.assertEqual(float(data["fps"]), 120.0)
                self.assertEqual(str(data["source_euler_order"]), "intrinsic XYZ")
                self.assertEqual(str(data["root_quat_order"]), "xyzw")
                self.assertEqual(data["joint_names"].tolist(), self.robot_spec.control_joint_names)

            loaded = read_robot_state_npz(output, source_name="mini3_test", robot_spec=self.robot_spec)
            self.assertEqual(list(loaded), ["mini3_motion"])
            self.assertEqual(loaded["mini3_motion"].root_pos.shape, (3, 3))
            self.assertEqual(loaded["mini3_motion"].dof_pos.shape, (3, 21))
            self.assertEqual(loaded["mini3_motion"].fps, 120.0)

    def test_missing_required_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "missing.csv"
            _write_mini3_csv(
                source,
                self.robot_spec.control_joint_names,
                missing_column="root_rotateZ",
            )
            with self.assertRaisesRegex(Mini3CsvConversionError, "root_rotateZ"):
                convert_csv_file(source, root / "missing.npz", self.robot_spec)

    def test_discovery_can_exclude_mirrored_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ("a.csv", "a_M.csv", "nested/b.csv"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            selected = discover_csv_files(root, exclude_mirrored=True)
            self.assertEqual([path.stem for path in selected], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
