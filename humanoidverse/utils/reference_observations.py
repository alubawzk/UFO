"""Reference observation helpers shared by expert and inference paths."""

from collections.abc import Mapping

import torch

from humanoidverse.utils.torch_utils import quat_rotate_inverse


def build_clean_discriminator_state(
    dof_pos_rel: torch.Tensor,
    dof_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    base_ang_vel: torch.Tensor,
    obs_scales: Mapping[str, float],
) -> torch.Tensor:
    """Build the shared clean state used for policy and expert discrimination.

    Inputs must be current, noise-free simulator/reference values. In particular,
    ``base_ang_vel`` and ``projected_gravity`` must not come from the delayed
    policy IMU path.
    """
    return torch.cat(
        [
            dof_pos_rel * float(obs_scales.get("dof_pos", 1.0)),
            dof_vel * float(obs_scales.get("dof_vel", 1.0)),
            projected_gravity * float(obs_scales.get("projected_gravity", 1.0)),
            base_ang_vel * float(obs_scales.get("base_ang_vel", 1.0)),
        ],
        dim=-1,
    )


def reference_base_ang_vel(env, base_quat: torch.Tensor, world_ang_vel: torch.Tensor) -> torch.Tensor:
    """Match simulator actor-observation preprocessing for base angular velocity."""
    local_ang_vel = quat_rotate_inverse(base_quat, world_ang_vel, w_last=True)
    scale = float(env.config.obs.obs_scales.get("base_ang_vel", 1.0))
    return local_ang_vel * scale
