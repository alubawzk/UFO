import math
import unittest

import torch

from humanoidverse.distributed import clip_grad_norm_stable_


class StableGradientClipTest(unittest.TestCase):
    def test_clips_large_finite_gradients_without_norm_overflow(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(1024))
        parameter.grad = torch.full_like(parameter, 1e20)

        total_norm = clip_grad_norm_stable_([parameter], 1.0)

        self.assertTrue(torch.isfinite(total_norm))
        self.assertAlmostEqual(float(total_norm), math.sqrt(1024) * 1e20, delta=1e15)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertAlmostEqual(float(torch.linalg.vector_norm(parameter.grad)), 1.0, places=5)

    def test_rejects_nonfinite_gradient_elements(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([1.0, float("nan")])

        with self.assertRaisesRegex(RuntimeError, "contains NaN or Inf"):
            clip_grad_norm_stable_([parameter], 1.0)


if __name__ == "__main__":
    unittest.main()
