import unittest

import torch

from humanoidverse.agents.fb_cpr.agent import _stable_gradient_norm


class StableGradientNormTest(unittest.TestCase):
    def test_zero_gradient_has_finite_second_order_derivative(self) -> None:
        gradients = torch.zeros((4, 8), dtype=torch.float32, requires_grad=True)
        penalty = ((_stable_gradient_norm(gradients) - 1.0) ** 2).mean()

        penalty.backward()

        self.assertTrue(torch.isfinite(penalty))
        self.assertTrue(torch.isfinite(gradients.grad).all())


if __name__ == "__main__":
    unittest.main()
