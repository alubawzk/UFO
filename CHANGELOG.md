# Changelog

## v0.1.0

Initial UFO release candidate.

### Added

- Unified UFO training entrypoint with FB and TLDR unsupervised RL presets.
- Documented current `update_z_every_step` defaults: FB uses `100`, TLDR uses `10`.
- Distributed MJLab training support for Unitree G1-style humanoid policies.
- Weighted multi-source motion dataset manifests with source-level sampling weights.
- RobotState motion import pipeline for `root_pos`, `root_quat`, `dof_pos`, and `fps` or `time`.
- CSV and NPZ readers for RobotState data, plus import wizard tools:
  - `humanoidverse.tools.robot_inspect`
  - `humanoidverse.tools.data_inspect`
  - `humanoidverse.tools.data_build`
- Tracking, goal, and reward inference entrypoints with MP4/ONNX export support.
- Release smoke script at `scripts/smoke_release.sh`.

### Hardened

- Manifest dataset names must be unique.
- Manifest weights must be finite, non-negative, and sum to a positive value.
- User-facing failed-motion sampling text now uses "PMCP-inspired failed-motion prioritization".

### Known Limitations

- UFO currently supports robot-aware motion data import, but the training environment is still G1/MJLab-centered.
- `robot_inspect` generates a draft robot YAML and cannot fully infer semantic fields such as feet, hands, and key bodies without user review.
- RobotState import does not perform FPS resampling.
- RobotState-to-UFO conversion currently supports hinge control joints.
- Cross-robot shared-policy training and arbitrary-robot training environments are not part of v0.1.0.
