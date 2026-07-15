from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from humanoidverse.agents.envs.humanoidverse_mjlab import (
    HumanoidVerseMjlabCore,
    _compose_humanoidverse_config,
    _randomize_dc_motor_strength,
    make_mjlab_ufo_env_cfg,
)
from humanoidverse.tracking_inference import (
    _expert_qpos_from_obs,
    _resolve_tracking_robot_config,
    _target_states_from_obs,
)
from humanoidverse.tracking_inference import (
    parse_args as parse_tracking_args,
)
from humanoidverse.train import _resolve_training_robot_config, build_ufo_mjlab_config
from humanoidverse.train import parse_args as parse_train_args
from humanoidverse.utils.robot_spec import load_robot_training_spec


def _write_tiny_robot_with_training(root: Path, *, missing_actuator_joint: bool = False) -> Path:
    xml_path = root / "tiny_train.xml"
    xml_path.write_text(
        """
<mujoco model="tiny_train">
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" mass="0.1"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-2 2"/>
          <geom type="sphere" size="0.03" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="joint1_motor" joint="joint1"/>
    <motor name="joint2_motor" joint="joint2"/>
  </actuator>
</mujoco>
""".strip()
    )
    joint2_block = (
        []
        if missing_actuator_joint
        else [
            "        joint2:",
            "          effort_limit: 2.0",
            "          velocity_limit: 20.0",
            "          armature: 0.02",
            "          friction: 0.002",
        ]
    )
    robot_config = root / "tiny_train.yaml"
    robot_config.write_text(
        "\n".join(
            [
                "name: tiny_train",
                "xml_path: tiny_train.xml",
                "base_body: base",
                "root_quat_order: xyzw",
                "coordinate_system: z_up",
                "dof_unit: rad",
                "control_joints:",
                "  mode: all_actuated",
                "feet: [link2]",
                "hands: []",
                "key_bodies: [base, link1, link2]",
                "default_dof_pos: {}",
                "training:",
                "  hydra_robot: g1/g1_29dof_hard_waist",
                "  hydra_overrides: []",
                "  semantics:",
                "    contact_bodies: [link2]",
                "    undesired_contact_bodies: [base]",
                "    torso_name: base",
                "    left_ankle_dof_names: []",
                "    right_ankle_dof_names: []",
                "  init_state:",
                "    pos: [0.0, 0.0, 1.0]",
                "    rot: [0.0, 0.0, 0.0, 1.0]",
                "    lin_vel: [0.0, 0.0, 0.0]",
                "    ang_vel: [0.0, 0.0, 0.0]",
                "    default_joint_angles:",
                "      joint1: 0.0",
                "      joint2: 0.0",
                "  control:",
                "    action_scale: 0.25",
                "    action_clip_value: 5.0",
                "    normalize_action_to: 5.0",
                "    effort_limit: [1.0, 2.0]",
                "    velocity_limit: [10.0, 20.0]",
                "    stiffness: {joint1: 1.0, joint2: 2.0}",
                "    damping: {joint1: 0.1, joint2: 0.2}",
                "  actuator:",
                "    source: yaml",
                "    joints:",
                "      joint1:",
                "        effort_limit: 1.0",
                "        velocity_limit: 10.0",
                "        armature: 0.01",
                "        friction: 0.001",
                *joint2_block,
            ]
        )
    )
    return robot_config


def _make_mini3_mjlab_cfg(*, disable_domain_randomization: bool = False):
    training = load_robot_training_spec("configs/robots/mini3.yaml")
    hydra_overrides = [
        f"robot={training.hydra_robot}",
        f"robot.control.action_scale={training.action_scale}",
        f"robot.control.action_clip_value={training.action_clip_value}",
        f"robot.control.normalize_action_to={training.normalize_action_to}",
        *training.hydra_overrides,
    ]
    hv_config, _ = _compose_humanoidverse_config(
        num_envs=2,
        relative_config_path="exp/bfm_zero/bfm_zero",
        hydra_overrides=hydra_overrides,
        lafan_tail_path="humanoidverse/data/mini3_pkl_ufo",
        data_mix_weights=[1.0],
        disable_obs_noise=False,
        disable_domain_randomization=disable_domain_randomization,
        max_episode_length_s=None,
        root_height_obs=True,
        robot_training=training.to_env_dict(),
    )
    mjlab_config = make_mjlab_ufo_env_cfg(
        hv_config,
        num_envs=2,
        seed=1,
        mjcf_path=training.robot.xml_path,
        auto_reset=False,
        robot_training=training.to_env_dict(),
    )
    return hv_config, mjlab_config


class RobotConfigTrainingTest(unittest.TestCase):
    def test_old_g1_default_builds_cfg(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_unit",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
        )
        self.assertTrue(str(cfg.env.robot_config_path).endswith("configs/robots/g1_29dof.yaml"))
        self.assertTrue(str(cfg.env.mjcf_path).endswith("humanoidverse/data/robots/g1_mjlab/g1_29dof.xml"))

    def test_explicit_g1_robot_config_builds_cfg(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_unit",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            robot_config="configs/robots/g1_29dof.yaml",
        )
        self.assertTrue(str(cfg.env.robot_config_path).endswith("configs/robots/g1_29dof.yaml"))

    def test_mini3_pd_gain_domain_randomization_builds_reset_event(self) -> None:
        from mjlab.envs.mdp import dr as mjlab_dr

        hv_config, mjlab_config = _make_mini3_mjlab_cfg()

        self.assertTrue(hv_config.domain_rand.randomize_pd_gain)
        self.assertEqual(list(hv_config.domain_rand.kp_range), [0.75, 1.25])
        self.assertEqual(list(hv_config.domain_rand.kd_range), [0.75, 1.25])
        event = mjlab_config.events["random_pd_gains"]
        self.assertEqual(event.mode, "reset")
        self.assertIs(event.func, mjlab_dr.pd_gains)
        self.assertEqual(event.params["kp_range"], (0.75, 1.25))
        self.assertEqual(event.params["kd_range"], (0.75, 1.25))
        self.assertEqual(event.params["operation"], "scale")

        self.assertTrue(hv_config.domain_rand.randomize_ctrl_delay)
        self.assertEqual(list(hv_config.domain_rand.ctrl_delay_step_range), [0, 1])
        self.assertTrue(hv_config.domain_rand.randomize_motor_strength)
        motor_event = mjlab_config.events["random_motor_strength"]
        self.assertEqual(motor_event.mode, "reset")
        self.assertIs(motor_event.func, _randomize_dc_motor_strength)
        self.assertEqual(motor_event.params["strength_range"], (0.9, 1.1))

    def test_disable_dr_removes_mini3_pd_gain_event(self) -> None:
        hv_config, mjlab_config = _make_mini3_mjlab_cfg(disable_domain_randomization=True)

        self.assertFalse(hv_config.domain_rand.randomize_pd_gain)
        self.assertNotIn("random_pd_gains", mjlab_config.events)
        self.assertFalse(hv_config.domain_rand.randomize_ctrl_delay)
        self.assertFalse(hv_config.domain_rand.randomize_motor_strength)
        self.assertNotIn("random_motor_strength", mjlab_config.events)

    def test_mini3_fb_aux_reward_overrides_reach_agent_config(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_unit",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            agent="fb",
            robot_config="configs/robots/mini3.yaml",
        )

        self.assertEqual(cfg.agent.aux_rewards_scaling["penalty_action_rate"], -0.2)
        self.assertEqual(cfg.agent.aux_rewards_scaling["penalty_ankle_roll"], -1.0)

    def test_manifest_robot_config_is_used_when_cli_missing(self) -> None:
        argv = [
            "train.py",
            "--agent",
            "fb",
            "--data-manifest",
            "configs/data/example_mix.yaml",
            "--gpu-ids",
            "single",
            "--smoke",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_train_args()
        self.assertTrue(str(args.robot_config).endswith("configs/robots/g1_29dof.yaml"))

    def test_cli_manifest_robot_config_mismatch_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_robot = _write_tiny_robot_with_training(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "does not match data manifest robot_config"):
                _resolve_training_robot_config(tiny_robot, "configs/robots/g1_29dof.yaml")

    def test_tracking_manifest_robot_config_is_used_when_cli_missing(self) -> None:
        argv = [
            "tracking_inference.py",
            "--model-folder",
            "/tmp/ufo_unit_model",
            "--data-manifest",
            "configs/data/example_robot_state_auto_build.yaml",
            "--dataset",
            "g1_robot_state_sample",
            "--export-onnx",
            "false",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_tracking_args()
        self.assertTrue(str(args.robot_config).endswith("configs/robots/g1_29dof.yaml"))

    def test_tracking_cli_manifest_robot_config_mismatch_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_robot = _write_tiny_robot_with_training(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "does not match data manifest robot_config"):
                _resolve_tracking_robot_config(tiny_robot, "configs/robots/g1_29dof.yaml")

    def test_aux_rewards_require_two_contact_bodies_unconditionally(self) -> None:
        core = object.__new__(HumanoidVerseMjlabCore)
        core.reward_scales = {}
        cfg = OmegaConf.create(
            {
                "robot": {
                    "contact_bodies": ["left_foot"],
                    "left_ankle_dof_names": ["left_ankle_pitch_joint", "left_ankle_roll_joint"],
                    "right_ankle_dof_names": ["right_ankle_pitch_joint", "right_ankle_roll_joint"],
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "robot.contact_bodies.*biped foot auxiliary terms unconditionally"):
            core._validate_aux_reward_semantics(cfg)

    def test_aux_ankle_reward_requires_both_ankle_fields(self) -> None:
        core = object.__new__(HumanoidVerseMjlabCore)
        core.reward_scales = {"penalty_ankle_roll": -1.0}
        cfg = OmegaConf.create(
            {
                "robot": {
                    "contact_bodies": ["left_foot", "right_foot"],
                    "left_ankle_dof_names": ["left_ankle_pitch_joint"],
                    "right_ankle_dof_names": [],
                }
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            "robot.left_ankle_dof_names, robot.right_ankle_dof_names.*penalty_ankle_roll",
        ):
            core._validate_aux_reward_semantics(cfg)

    def test_mjlab_action_input_reorders_policy_actions_to_action_term_order(self) -> None:
        core = object.__new__(HumanoidVerseMjlabCore)
        core.applied_actions = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        core.default_dof_pos_offset = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        core.action_target_scale = torch.tensor([[1.0, 2.0, 1.0, 4.0]])
        core._action_term_dof_indices = torch.tensor([2, 0, 3, 1])

        action_input = core._mjlab_action_input()

        torch.testing.assert_close(action_input, torch.tensor([[33.0, 11.0, 41.0, 21.0]]))

    def test_control_delay_applies_per_environment_action_history(self) -> None:
        core = object.__new__(HumanoidVerseMjlabCore)
        core.config = OmegaConf.create({"domain_rand": {"randomize_ctrl_delay": True, "ctrl_delay_step_range": [0, 1]}})
        core.num_envs = 2
        core.num_dof = 2
        core.device = "cpu"
        core.actions = torch.zeros(2, 2)
        core.applied_actions = torch.zeros_like(core.actions)
        core._init_control_delay()
        core.ctrl_delay_steps[:] = torch.tensor([0, 1])

        first = core._apply_control_delay(torch.tensor([[1.0, 2.0], [3.0, 4.0]])).clone()
        second = core._apply_control_delay(torch.tensor([[5.0, 6.0], [7.0, 8.0]])).clone()

        torch.testing.assert_close(first, torch.tensor([[1.0, 2.0], [0.0, 0.0]]))
        torch.testing.assert_close(second, torch.tensor([[5.0, 6.0], [3.0, 4.0]]))

    def test_yaml_actuator_missing_joint_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_robot = _write_tiny_robot_with_training(Path(tmpdir), missing_actuator_joint=True)
            with self.assertRaisesRegex(ValueError, "missing parameters for joint 'joint2'"):
                load_robot_training_spec(tiny_robot)

    def test_tracking_shapes_follow_num_dof(self) -> None:
        obs = {
            "ref_body_pos": torch.zeros(4, 1, 3),
            "ref_body_rots": torch.zeros(4, 1, 4),
            "ref_body_vels": torch.zeros(4, 1, 3),
            "ref_body_angular_vels": torch.zeros(4, 1, 3),
            "dof_pos": torch.zeros(4, 2),
            "ref_dof_vel": torch.ones(4, 2),
        }
        obs["ref_body_rots"][..., 3] = 1.0
        qpos = _expert_qpos_from_obs(obs, num_dof=2, dof_qpos_order_indices=torch.tensor([0, 1]).numpy())
        self.assertEqual(qpos.shape, (4, 9))
        target = _target_states_from_obs(obs, device="cpu", num_dof=2)
        self.assertEqual(tuple(target["dof_states"].shape), (1, 2, 2))


if __name__ == "__main__":
    unittest.main()
