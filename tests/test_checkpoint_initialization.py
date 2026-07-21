from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import safetensors.torch
import torch

from humanoidverse.training.workspace import (
    _initialize_agent_model,
    _resolve_init_checkpoint_model_path,
    create_agent_or_load_checkpoint,
)


class CheckpointInitializationTest(unittest.TestCase):
    def test_resolves_run_checkpoint_model_and_file_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            model_file = run_dir / "checkpoint" / "model" / "model.safetensors"
            model_file.parent.mkdir(parents=True)
            model_file.touch()

            for source in (run_dir, run_dir / "checkpoint", model_file.parent, model_file):
                self.assertEqual(_resolve_init_checkpoint_model_path(source), model_file)

    def test_initialization_loads_only_model_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_file = Path(temp_dir) / "checkpoint" / "model" / "model.safetensors"
            model_file.parent.mkdir(parents=True)
            source_model = torch.nn.Linear(3, 2)
            with torch.no_grad():
                source_model.weight.fill_(1.25)
                source_model.bias.fill_(-0.5)
            safetensors.torch.save_model(source_model, model_file)

            target_model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.Adam(target_model.parameters(), lr=1.0e-4)
            agent = SimpleNamespace(_model=target_model)
            resolved = _initialize_agent_model(agent, model_file.parent.parent, "cpu")

            self.assertEqual(resolved, model_file)
            torch.testing.assert_close(target_model.weight, source_model.weight)
            torch.testing.assert_close(target_model.bias, source_model.bias)
            self.assertEqual(optimizer.state_dict()["state"], {})

    def test_init_checkpoint_rejects_existing_resume_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            checkpoint_dir = work_dir / "checkpoint"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "train_status.json").write_text("{}")
            cfg = SimpleNamespace(init_checkpoint="old_run")

            with self.assertRaisesRegex(ValueError, "fresh work_dir"):
                create_agent_or_load_checkpoint(work_dir, cfg, {})


if __name__ == "__main__":
    unittest.main()
