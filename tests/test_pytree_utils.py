import unittest

import torch

from humanoidverse.agents.pytree_utils import assert_finite_tensors


class AssertFiniteTensorsTest(unittest.TestCase):
    def test_accepts_finite_nested_tensors(self) -> None:
        assert_finite_tensors(
            {"observation": {"state": torch.tensor([[1.0, -2.0]])}, "terminated": torch.tensor([False])},
            label="batch",
        )

    def test_reports_nested_nonfinite_tensor_path(self) -> None:
        value = {"next": {"observation": {"state": torch.tensor([[1.0, float("nan")]])}}}

        with self.assertRaisesRegex(FloatingPointError, r"batch\.next\.observation\.state"):
            assert_finite_tensors(value, label="batch")


if __name__ == "__main__":
    unittest.main()
