from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from humanoidverse.agents.presets import build_agent_preset
from humanoidverse.train import build_ufo_mjlab_config, parse_args
from humanoidverse.training.workspace import (
    Workspace,
    _accumulate_metrics,
    _log_evaluation_metrics_to_wandb,
    _trajectory_output_keys,
)


class UpdateZCliTest(unittest.TestCase):
    def _parse(self, *args: str):
        with patch.object(sys, "argv", ["train.py", *args]), patch("sys.stderr", io.StringIO()):
            return parse_args()

    def test_agent_specific_defaults_are_preserved(self) -> None:
        self.assertEqual(self._parse("--agent", "fb").update_z_every_step, 100)
        self.assertEqual(self._parse("--agent", "tldr").update_z_every_step, 10)

    def test_tldr_cli_value_reaches_agent_config(self) -> None:
        args = self._parse("--agent", "tldr", "--update-z-every-step", "37")
        selected = build_agent_preset(
            agent=args.agent,
            device="cpu",
            compile=False,
            update_z_every_step=args.update_z_every_step,
            lr_scale=1.0,
            clip_grad_norm=0.0,
            cartwheel_aux_safe=False,
            wandb_project="test",
        )
        self.assertEqual(selected["agent_cfg"].train.update_z_every_step, 37)

    def test_programmatic_tldr_default_remains_ten(self) -> None:
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_update_z_test",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            agent="tldr",
        )
        self.assertEqual(cfg.agent.train.update_z_every_step, 10)

    def test_init_checkpoint_cli_reaches_train_config(self) -> None:
        args = self._parse("--init-checkpoint", "runs/old_dc")
        self.assertEqual(args.init_checkpoint, Path("runs/old_dc"))
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_init_checkpoint_test",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            init_checkpoint=args.init_checkpoint,
        )
        self.assertEqual(cfg.init_checkpoint, str(Path("runs/old_dc").resolve()))

    def test_checkpoint_buffer_is_enabled_by_default(self) -> None:
        args = self._parse()
        self.assertFalse(args.no_checkpoint_buffer)
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_checkpoint_buffer_default_test",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            checkpoint_buffer=not args.no_checkpoint_buffer,
        )
        self.assertTrue(cfg.checkpoint_buffer)

    def test_no_checkpoint_buffer_cli_disables_buffer_checkpoint(self) -> None:
        args = self._parse("--no-checkpoint-buffer")
        self.assertTrue(args.no_checkpoint_buffer)
        cfg = build_ufo_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_no_checkpoint_buffer_test",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            checkpoint_buffer=not args.no_checkpoint_buffer,
        )
        self.assertFalse(cfg.checkpoint_buffer)

    def test_disabled_checkpoint_buffer_still_saves_training_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Workspace.__new__(Workspace)
            workspace.cfg = SimpleNamespace(
                distributed_sync=False,
                distributed_world_size=1,
                checkpoint_buffer=False,
                agent=SimpleNamespace(train=SimpleNamespace(batch_size=32)),
            )
            workspace.work_dir = Path(tmp_dir)
            workspace._write_shared_artifacts = True
            workspace.agent = SimpleNamespace(
                save=Mock(side_effect=lambda path: Path(path).mkdir(parents=True, exist_ok=True))
            )
            replay_buffer = SimpleNamespace(save=Mock())

            workspace.save(
                local_time=100,
                global_time=100,
                optimizer_steps=10,
                replay_buffer={"train": replay_buffer},
            )

            workspace.agent.save.assert_called_once_with(str(Path(tmp_dir) / "checkpoint"))
            replay_buffer.save.assert_not_called()
            self.assertTrue((Path(tmp_dir) / "checkpoint" / "train_status.json").is_file())
            self.assertFalse((Path(tmp_dir) / "checkpoint" / "buffers").exists())

    def test_tldr_trajectory_buffer_keeps_aux_rewards(self) -> None:
        selected = build_agent_preset(
            agent="tldr",
            device="cpu",
            compile=False,
            update_z_every_step=10,
            lr_scale=1.0,
            clip_grad_norm=0.0,
            cartwheel_aux_safe=False,
            wandb_project="test",
        )
        self.assertIn("aux_rewards", _trajectory_output_keys(selected["agent_cfg"]))

    def test_metric_accumulation_accepts_tldr_phase_changes(self) -> None:
        totals, counts = _accumulate_metrics(
            None,
            {},
            {"tldr_te_loss": torch.tensor(2.0)},
        )
        totals, counts = _accumulate_metrics(
            totals,
            counts,
            {
                "tldr_te_loss": torch.tensor(4.0),
                "disc_wgan_gp_loss": torch.tensor(6.0),
            },
        )
        self.assertEqual(counts, {"tldr_te_loss": 2, "disc_wgan_gp_loss": 1})
        self.assertEqual((totals["tldr_te_loss"] / counts["tldr_te_loss"]).item(), 3.0)
        self.assertEqual((totals["disc_wgan_gp_loss"] / counts["disc_wgan_gp_loss"]).item(), 6.0)

    @patch("humanoidverse.training.workspace.wandb.log")
    def test_distributed_evaluation_does_not_block_on_wandb(self, wandb_log: Mock) -> None:
        published = _log_evaluation_metrics_to_wandb(
            enabled=True,
            distributed_sync=True,
            evaluation_name="tracking",
            metrics={"emd": 1.25},
            step=0,
        )

        self.assertFalse(published)
        wandb_log.assert_not_called()

    @patch("humanoidverse.training.workspace.wandb.log")
    def test_single_process_evaluation_still_logs_to_wandb(self, wandb_log: Mock) -> None:
        published = _log_evaluation_metrics_to_wandb(
            enabled=True,
            distributed_sync=False,
            evaluation_name="tracking",
            metrics={"emd": 1.25},
            step=42,
        )

        self.assertTrue(published)
        wandb_log.assert_called_once_with({"eval/tracking/emd": 1.25}, step=42)


if __name__ == "__main__":
    unittest.main()
