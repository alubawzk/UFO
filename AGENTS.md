# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `humanoidverse/`. Training and inference entry points are top-level modules such as `train.py`, `tracking_inference.py`, and `reward_inference.py`; algorithms are under `humanoidverse/agents/`, environment logic under `humanoidverse/envs/`, and reusable data, robot, and math utilities under `humanoidverse/utils/`. Hydra configuration is in `humanoidverse/config/`, while portable robot and dataset definitions live in `configs/`. Keep unit tests in `tests/`, documentation in `docs/`, helper CLIs in `humanoidverse/tools/` or `tools/`, shell automation in `scripts/`, and images in `assets/`.

## Build, Test, and Development Commands

- `uv sync --dev` installs the pinned Python 3.10 environment and Ruff.
- `uv run python -m unittest discover -s tests -p 'test_*.py'` runs the unit suite.
- `uv run ruff check .` checks lint rules and import ordering; `uv run ruff format --check .` verifies formatting.
- `./run_train.sh --agent fb --data-manifest configs/data/example_mix.yaml --gpu-ids single --smoke` exercises a short training path.
- `bash scripts/smoke_release.sh` runs focused tests, compilation checks, smoke training, and a sample data build. It requires the appropriate CUDA/MJLab environment.

## Coding Style & Naming Conventions

Use four-space indentation, type hints for public interfaces, and focused modules. Ruff is configured in `pyproject.toml` with a 140-character line limit; E402 and E731 are intentionally ignored. Use `snake_case` for modules, functions, variables, and YAML files; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Preserve lazy imports where simulator startup or memory use benefits from them.

## Testing Guidelines

Tests use `unittest` conventions, including `TestCase` classes and `test_*` methods/files, and commonly isolate filesystem work with `tempfile.TemporaryDirectory`. Add regression tests beside related coverage and include validation/error cases for robot schemas, joint ordering, and motion-data conversion. No numeric coverage threshold is declared; new behavior should be exercised without requiring large downloaded datasets when a compact fixture will do.

## Commit & Pull Request Guidelines

Recent history favors short, imperative summaries, sometimes with prefixes such as `fix:` or `docs:`. Keep each commit scoped and describe the observable change (for example, `fix: preserve XML joint order`). Pull requests should explain motivation, list tested commands, link relevant issues, and call out robot/config compatibility. Include logs for training changes and screenshots or generated artifacts only when behavior is visual. Never commit W&B credentials, checkpoints, downloaded datasets, or local cache/run directories.
