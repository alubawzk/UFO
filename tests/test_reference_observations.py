from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from humanoidverse.utils.reference_observations import build_clean_discriminator_state, reference_base_ang_vel
from humanoidverse.utils.torch_utils import quat_rotate_inverse

REPO_ROOT = Path(__file__).resolve().parents[1]


class ReferenceObservationsTest(unittest.TestCase):
    def test_clean_discriminator_state_has_shared_order_and_scales(self) -> None:
        dof_pos_rel = torch.tensor([[1.0, 2.0]])
        dof_vel = torch.tensor([[3.0, 4.0]])
        projected_gravity = torch.tensor([[5.0, 6.0, 7.0]])
        base_ang_vel = torch.tensor([[8.0, 9.0, 10.0]])
        obs_scales = {
            "dof_pos": 2.0,
            "dof_vel": 3.0,
            "projected_gravity": 4.0,
            "base_ang_vel": 0.25,
        }

        actual = build_clean_discriminator_state(
            dof_pos_rel,
            dof_vel,
            projected_gravity,
            base_ang_vel,
            obs_scales,
        )
        expected = torch.tensor([[2.0, 4.0, 9.0, 12.0, 20.0, 24.0, 28.0, 2.0, 2.25, 2.5]])

        self.assertTrue(torch.equal(actual, expected))

    def test_policy_discriminator_state_bypasses_policy_noise_and_imu_delay(self) -> None:
        tree = ast.parse((REPO_ROOT / "humanoidverse/agents/envs/humanoidverse_mjlab.py").read_text())
        calls = [
            node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == "build_clean_discriminator_state"
        ]

        self.assertEqual(len(calls), 1)
        arguments = [ast.unparse(argument) for argument in calls[0].args]
        self.assertEqual(
            arguments,
            [
                "self.dof_pos - self.default_dof_pos",
                "self.dof_vel",
                "self.projected_gravity",
                "self.base_ang_vel",
                "self.config.obs.obs_scales",
            ],
        )

    def test_fb_discriminator_uses_clean_state_and_normalizes_it(self) -> None:
        source = (REPO_ROOT / "humanoidverse/agents/presets/fb.py").read_text()

        self.assertIn('key=["discriminator_state", "privileged_state"]', source)
        self.assertIn('"discriminator_state": BatchNormNormalizerConfig', source)

    def test_reference_base_ang_vel_matches_simulator_contract(self) -> None:
        env = SimpleNamespace(config=SimpleNamespace(obs=SimpleNamespace(obs_scales={"base_ang_vel": 0.25})))
        sqrt_half = 2.0**-0.5
        base_quat = torch.tensor(
            [
                [0.0, 0.0, sqrt_half, sqrt_half],
                [sqrt_half, 0.0, 0.0, sqrt_half],
            ],
            dtype=torch.float32,
        )
        world_ang_vel = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [-2.0, 4.0, 1.0],
            ],
            dtype=torch.float32,
        )

        expected = quat_rotate_inverse(base_quat, world_ang_vel, w_last=True) * 0.25
        actual = reference_base_ang_vel(env, base_quat, world_ang_vel)

        self.assertTrue(torch.allclose(actual, expected))

    def test_reference_base_ang_vel_defaults_to_unit_scale(self) -> None:
        env = SimpleNamespace(config=SimpleNamespace(obs=SimpleNamespace(obs_scales={})))
        base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        world_ang_vel = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)

        actual = reference_base_ang_vel(env, base_quat, world_ang_vel)

        self.assertTrue(torch.equal(actual, world_ang_vel))

    def test_reference_paths_do_not_directly_assign_world_base_ang_vel(self) -> None:
        paths = [
            "humanoidverse/agents/envs/expert_motion_loader.py",
            "humanoidverse/utils/helpers.py",
            "humanoidverse/agents/evaluations/humanoidverse_mjlab.py",
        ]
        for relative_path in paths:
            with self.subTest(path=relative_path):
                tree = ast.parse((REPO_ROOT / relative_path).read_text())
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Assign):
                        continue
                    assigns_ref_ang_vel = any(isinstance(target, ast.Name) and target.id == "ref_ang_vel" for target in node.targets)
                    if assigns_ref_ang_vel:
                        self.assertNotEqual(ast.unparse(node.value), "ref_body_angular_vels[:, 0]", relative_path)
