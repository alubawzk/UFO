# BFM-Zero MJLab Migration Notes

This project was created as an independent copy at:

- Remote target: `/home/xue/bfmzero-mjlab`
- Source copied from: `/home/xue/BFM-Zero`
- Copy command excluded only local/runtime artifacts: `.git/`, `.venv/`, `_wandb/`, `__pycache__/`, `motivo_isaac.egg-info/`, `tea_debug.log`, `=0.26.0`

The original source project `/home/xue/BFM-Zero` is treated as read-only for this migration.

## Added/Changed Files

- `humanoidverse/agents/envs/humanoidverse_mjlab.py`
  - Adds `HumanoidVerseMjlabConfig`, `HumanoidVerseMjlabCore`, and `HumanoidVerseMjlabVectorEnv`.
  - Builds an MJLab `ManagerBasedRlEnvCfg` using the G1 29DoF MJCF asset.
  - Preserves the original BFM-Zero observation/action/reward/info surface expected by `Workspace.train_online()`.
  - Reorders MJLab body/joint data to the original HumanoidVerse YAML order before computing `max_local_self`.
  - Adds MJLab event-based domain randomization for push, torso COM offset, link mass, and geom friction.
  - Recreates original default joint position offset randomization in the wrapper and applies it to both observation-relative DOF position and position targets.

- `humanoidverse/train.py`
  - Adds `HumanoidVerseMjlabConfig` to the environment union.
  - Reuses the existing FBcprAux training loop, replay buffer, expert motion loader and tracking evaluation path for MJLab envs.
  - Adds distributed rank metadata so rank0 owns shared logs/config/checkpoints while replay buffers are saved/restored per rank.
  - Uses separate distributed counters: `local_time` for per-rank rollout/update cadence, `global_time` for sample budget/checkpoint/eval/log cadence, and `optimizer_steps` for actual `agent.update()` calls.
  - Distributed training currently uses `loss_mode=local_loss_average`: each rank computes the local FB/actor/critic losses and gradients are averaged. This is synchronized training, but it is not mathematically identical to an all-gathered global-batch FB loss.

- `humanoidverse/train_mjlab.py`
  - Adds a separate BFM-Zero MJLab training entrypoint.
  - Keeps the original FBcprAux model/train hyperparameters from `train_bfm_zero()`.
  - Supports `--gpu-ids single`, `--gpu-ids all`, and comma-separated GPU lists.
  - Adds `--checkpoint-every-steps` for validation/debug runs without changing the default checkpoint cadence.
  - Adds `--disable-eval-prioritization` for validation/debug runs that need large env counts without t=0 tracking eval. The default training path still keeps evaluation and prioritization enabled.

- `tools/validate_mjlab_bfmzero.py`
  - Adds asset/order/reward-key validation.
  - Adds optional MJLab reset/step smoke validation.

- `pyproject.toml`
  - Converts this independent copy to the MJLab dependency stack by removing IsaacLab/IsaacSim runtime dependencies and adding `mjlab==1.4.0`, `mujoco~=3.8.0`, `torch>=2.7,<2.8`, and `torchrunx>=0.3.4`.

## Current Verification Commands

Asset-only validation:

```bash
cd /home/xue/bfmzero-mjlab
uv run python tools/validate_mjlab_bfmzero.py --asset-only
```

MJLab smoke validation after installing optional dependencies:

```bash
cd /home/xue/bfmzero-mjlab
uv run python tools/validate_mjlab_bfmzero.py --smoke --num-envs 16 --steps 2 --device cuda:0
```

Short training smoke:

```bash
cd /home/xue/bfmzero-mjlab
CUDA_VISIBLE_DEVICES=0 ./run_mjlab.sh --gpu-ids single --smoke --work-dir /data/xue/bfmzero-mjlab/_smoke_runs/single_gpu
```

8-GPU startup:

```bash
cd /home/xue/bfmzero-mjlab
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_mjlab.sh --gpu-ids all
```

Use `/data/xue/bfmzero-mjlab` for run outputs/checkpoints. The server root filesystem that backs `/home` is nearly full, while `/data` has sufficient space.

Use `./run_mjlab.sh` for MJLab training instead of invoking `uv run python -m humanoidverse.train_mjlab` directly. The wrapper sets `UV_CACHE_DIR`, `PYTHONPYCACHEPREFIX`, `TMPDIR`, TorchInductor, Triton, and CUDA cache paths before Python starts, so root filesystem pressure does not break long runs.

## Distributed Schedule Semantics

- `num_env_steps` is a global sample budget. With 8 ranks and `num_env_steps=384000000`, training stops at roughly 384M total transitions, not 8x that amount.
- `--num-envs` / `online_parallel_envs` is per rank. With `--num-envs 1024` and 8 ranks, the global parallel environment count is `1024 * 8 = 8192`.
- `local_time` counts transitions collected by each rank. It drives `num_seed_steps`, `update_agent_every`, rollout context timing, and the `step` passed to `agent.update()`.
- `global_time = local_time * world_size` when distributed global accounting is enabled. It drives checkpoint, eval, CSV/W&B logging, progress bars, and the legacy `time` alias in `train_status.json`.
- `optimizer_steps` counts actual `agent.update()` calls. `num_agent_updates` remains 16 by default and is not multiplied by `world_size`.
- `num_seed_steps` is a local gate. On 8 ranks, the first update begins after roughly `num_seed_steps * 8` global samples.
- Replay buffers are rank-local shards. `buffer_size` is the per-rank replay capacity; the effective aggregate replay capacity is `buffer_size * world_size`.
- New checkpoints write `train_status.json` with `local_time`, `global_time`, `optimizer_steps`, `world_size`, `loss_mode`, and `effective_batch_size`; older single-rank `{"time": ...}` checkpoints are read as legacy global-time checkpoints.
- Distributed checkpoint resume is fail-fast on `world_size` mismatch. Rank-local replay buffer shards are not automatically migrated between GPU counts.
- The training loop stops before any rollout that would exceed `num_env_steps`; a terminal checkpoint may be written at the final reached `global_time`, but no extra rollout/update is run after the global budget is reached.
- Distributed runs disable `torch.compile` for now (`compile=False`) to keep gradient synchronization outside compiled graphs.

## Known Remaining Work

- Keep the torch constraint pinned to `torch>=2.7,<2.8`. Unrestricted `torch>=2.7` resolved to `torch 2.12.1+cu130`, which could not initialize CUDA on this server. The updated lock resolves to `torch 2.7.1+cu126`, and `torch.cuda` initialization works on `cuda:0`.
- The motion file used by the MJLab project is now present at `humanoidverse/data/lafan_29dof_10s-clipped.pkl`, copied from `/data/xue/bfmzero/data/lafan_29dof_10s-clipped.pkl`.
- MJLab domain randomization is implemented and smoke-tested. `dr.body_mass` emits an MJLab advisory warning because it scales mass without inertia; this intentionally follows the original mass-scale field semantics.
- Validate longer training behavior on the intended available GPU set, NaN stability under real workload, and performance/convergence.
- If convergence differs, report measured steps/s and bottlenecks instead of claiming parity.

## Verification So Far

- `uv run python -m py_compile humanoidverse/agents/envs/humanoidverse_mjlab.py humanoidverse/train.py humanoidverse/train_mjlab.py tools/validate_mjlab_bfmzero.py`
  passed on the remote new project.
- `uv run python tools/validate_mjlab_bfmzero.py --asset-only`
  passed on the remote new project.
- `uv run python tools/validate_mjlab_bfmzero.py --smoke --num-envs 2 --steps 1 --device cuda:0`
  passed on the remote new project.
- `uv run python tools/validate_mjlab_bfmzero.py --smoke --num-envs 16 --steps 2 --device cuda:0`
  passed on the remote new project.
- `uv run python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())"`
  reports `2.7.1+cu126`, CUDA `12.6`, and `True`.
- `CUDA_VISIBLE_DEVICES=0 ./run_mjlab.sh --gpu-ids single --smoke --num-envs 2 --work-dir /home/xue/bfmzero-mjlab/_smoke_runs/single_gpu`
  reached `Starting training` and exited cleanly.
- `CUDA_VISIBLE_DEVICES=0,1 ./run_mjlab.sh --gpu-ids all --smoke --num-envs 2 --num-env-steps 32 --work-dir /home/xue/bfmzero-mjlab/_smoke_runs/two_gpu_rank0`
  completed successfully. Only rank0 wrote shared `config.json/config.yaml`; no `rank_1` config directory was created.
- `CUDA_VISIBLE_DEVICES=0,1 ./run_mjlab.sh --gpu-ids all --smoke --num-envs 2 --num-env-steps 8 --checkpoint-every-steps 2 --work-dir /data/xue/bfmzero-mjlab/_smoke_runs/two_gpu_ckpt`
  completed successfully and saved rank0 checkpoint metadata/model plus per-rank buffers:
  `checkpoint/buffers/train_rank_0` and `checkpoint/buffers/train_rank_1`.
- Re-running the same 2-GPU checkpoint smoke with `--num-env-steps 8` loaded `train_status.json` at time 8 and restored both rank buffers. Rank1 log confirms `Loading checkpointed buffer` and `Loaded buffer of size 4`.
- Original project non-invasive check:
  `cd /home/xue/BFM-Zero && python3 -m py_compile humanoidverse/train.py humanoidverse/agents/envs/humanoidverse_isaac.py`
  passed. The original project still has no `humanoidverse/train_mjlab.py` or `humanoidverse/agents/envs/humanoidverse_mjlab.py`.
- `uv run python tools/validate_mjlab_bfmzero.py --smoke --num-envs 1024 --steps 2 --device cuda:0`
  passed on `cuda:0`.
- `CUDA_VISIBLE_DEVICES=0 ./run_mjlab.sh --gpu-ids single --num-envs 1024 --num-env-steps 2048 --checkpoint-every-steps 1024 --disable-eval-prioritization --work-dir /data/xue/bfmzero-mjlab/_smoke_runs/single_1024_ckpt`
  completed successfully. Checkpoint evidence: `train_status.json` reports `{"time": 2048}`, `checkpoint/model/model.safetensors` exists, and `checkpoint/buffers/train/buffer.hdf5` exists.
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_mjlab.sh --gpu-ids all --smoke --num-envs 2 --num-env-steps 32 --work-dir /data/xue/bfmzero-mjlab/_smoke_runs/eight_gpu_start`
  completed successfully with 8 worker logs and no `rank_*` config directories.
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_mjlab.sh --gpu-ids all --smoke --num-envs 16 --num-env-steps 2048 --work-dir /data/xue/bfmzero-mjlab/_smoke_runs/eight_gpu_2048`
  completed successfully. All 8 worker logs contain `Starting training`; a log scan found no `Traceback`, `RuntimeError`, `ValueError`, `NaN`, or `nan` tokens.

- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./run_mjlab.sh --gpu-ids all --smoke --num-envs 2 --num-env-steps 8 --checkpoint-every-steps 2 --work-dir /data/xue/bfmzero-mjlab/_smoke_runs/eight_gpu_ckpt`
  completed successfully. `train_status.json` reports `{"time": 8}`, `checkpoint/model/model.safetensors` exists, and all per-rank buffers exist at `checkpoint/buffers/train_rank_0` through `checkpoint/buffers/train_rank_7`. All 8 worker logs contain `Starting training` and no `Traceback`, `RuntimeError`, `ValueError`, `NaN`, or `nan` tokens. The torchrunx main log reports workers exited without errors; it also records a post-success `TemporaryDirectory.cleanup` atexit traceback.

- A formal 8-GPU launch was stopped because GPUs 6 and 7 were already occupied by another experiment. It exposed a distributed default-path issue where non-rank0 workers had their evaluations cleared while `prioritization=True`, causing `ValueError: Prioritization requires tracking evaluation to be enabled`.
- `humanoidverse/train.py` was updated so non-rank0 workers keep evaluation objects for local prioritization, while only rank0 creates shared CSV/W&B/config artifacts. `eval()` now passes `logger=None` on non-writing ranks.
- Rank1 construction with `prioritization=True` passed with tracking evaluation enabled and no logger on the non-writing rank.
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./run_mjlab.sh --gpu-ids all --smoke --num-envs 2 --num-env-steps 8 --checkpoint-every-steps 2 --work-dir /data/xue/bfmzero-mjlab/_smoke_runs/six_gpu_ckpt`
  completed successfully. `train_status.json` reports `{"time": 8}`, `checkpoint/model/model.safetensors` exists, and all per-rank buffers exist at `checkpoint/buffers/train_rank_0` through `checkpoint/buffers/train_rank_5`. All 6 worker logs contain `Starting training` and no `Traceback`, `RuntimeError`, `ValueError`, `CUDA out of memory`, `Killed`, `NaN`, or `nan` tokens.
- A formal 6-GPU launch failed after t=0 evaluation because torch.compile/Inductor/Triton wrote compile caches to `/tmp/torchinductor_xue` on the root filesystem. The root filesystem had no free space, causing `OSError: [Errno 28] No space left on device`; the later `cannot pickle 'frame' object` message was secondary exception propagation from torchrunx.
- `humanoidverse/train_mjlab.py` now defaults compile/cache temp paths to `/data/xue/bfmzero-mjlab/cache`, and `run_mjlab.sh` was added as a wrapper that also sets `UV_CACHE_DIR` and `PYTHONPYCACHEPREFIX` before Python starts.
- Historical pre-schedule-fix run:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./run_mjlab.sh --gpu-ids all --num-envs 16 --num-env-steps 12288 --checkpoint-every-steps 4096 --disable-eval-prioritization --work-dir /data/xue/bfmzero-mjlab/_smoke_runs/six_gpu_update_cache`
  completed successfully and exercised agent update under the older distributed schedule. It is not current evidence for the fixed schedule or for distributed `torch.compile`; current distributed training sets `compile=False`.
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./run_mjlab.sh --gpu-ids all --num-envs 1024 --num-env-steps 73728 --checkpoint-every-steps 12288 --disable-eval-prioritization --work-dir /data/xue/bfmzero-mjlab/_smoke_runs/six_gpu_schedule_fix1`
  completed successfully on the schedule-fix code path. It wrote `train_status.json` with `local_time=12288`, `global_time=73728`, `optimizer_steps=16`, `world_size=6`, `loss_mode=local_loss_average`, and `effective_batch_size=6144`; `distributed_sync.json` reported `max_abs_diff_from_rank0=0.0`.
- Resuming the same `six_gpu_schedule_fix1` run to `--num-env-steps 86016` loaded `local_time=12288/global_time=73728/optimizer_steps=16` and advanced to `local_time=14336/global_time=86016/optimizer_steps=32`, with `distributed_sync.json` still reporting `max_abs_diff_from_rank0=0.0`.
- Hardfix verification after adding fail-fast `world_size` resume checks, budget-end stopping, and distributed observability:
  `python3 -m py_compile humanoidverse/train.py humanoidverse/train_mjlab.py` passed on the remote project.
- A direct `_normalize_train_status()` guard check confirmed that distributed resume fails for both mismatched checkpoint/current `world_size` and missing legacy `world_size`.
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./run_mjlab.sh --gpu-ids all --num-envs 1024 --num-env-steps 73728 --checkpoint-every-steps 12288 --disable-eval-prioritization --work-dir /data/xue/bfmzero-mjlab/_smoke_runs/six_gpu_hardfix1`
  completed successfully on the current code. It wrote `local_time=12288`, `global_time=73728`, `optimizer_steps=16`, `world_size=6`, `loss_mode=local_loss_average`, and `effective_batch_size=6144`; `distributed_sync.json` reported `max_abs_diff_from_rank0=0.0`.
- Resuming `six_gpu_hardfix1` to `--num-env-steps 86016` loaded `local_time=12288/global_time=73728/optimizer_steps=16` and advanced to `local_time=14336/global_time=86016/optimizer_steps=32`; `distributed_sync.json` again reported `max_abs_diff_from_rank0=0.0`.
- `CUDA_VISIBLE_DEVICES=0,1 ./run_mjlab.sh --gpu-ids all --smoke --num-envs 2 --num-env-steps 6 --checkpoint-every-steps 1 --work-dir /data/xue/bfmzero-mjlab/_smoke_runs/two_gpu_budget_guard_hardfix1`
  stopped at `global_time=4` and printed `Stopping before next rollout to avoid exceeding global sample budget`, proving the loop does not roll out to `global_time=8` when the global budget is 6.
- `CUDA_VISIBLE_DEVICES=0,1 ./run_mjlab.sh --gpu-ids all --num-envs 256 --num-env-steps 512 --checkpoint-every-steps 512 --work-dir /data/xue/bfmzero-mjlab/_smoke_runs/two_gpu_default_eval_hardfix1`
  completed the current default eval/prioritization path. It logged `Starting evaluation at time 0`, wrote `humanoidverse_tracking_eval.csv` with 863 lines, checkpointed at `local_time=256/global_time=512`, and `distributed_sync.json` reported `max_abs_diff_from_rank0=0.0`.
