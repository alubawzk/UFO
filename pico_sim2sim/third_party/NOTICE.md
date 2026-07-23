# Third-party components

This directory records the external code and data incorporated into the
standalone PICO-to-Mini3 path.

## JOYIn-Retarget

- Upstream: `https://github.com/YanjieZe/JOYIn-Retarget`
- Revision used: `f58d810a8c2d606a29353c1bc03e684ebedbd4d4`
- License: MIT (`JOYIN_LICENSE.txt`)
- Integrated material: the Mini3 two-stage GMR algorithm, SMPL-X-to-Mini3 IK
  table, and an IK-only representation of the Mini3 kinematic model.
- Changes: removed unrelated robots/loaders/viewers, changed asset resolution
  to package-local paths, and removed meshes/dynamics that do not affect IK.

## Gear Sonic / GR00T-WholeBodyControl

- Upstream: `https://github.com/NVlabs/GR00T-WholeBodyControl`
- Revision inspected: `038c0c923a56a133e670d21253236e9a6978a3f2`
- License: source code is Apache-2.0; model/data assets are distributed under
  the NVIDIA Open Model License (`GEAR_SONIC_LICENSE.txt`).
- Integrated material: Sonic pose wire format, XRT-to-local-SMPL rotation
  conversion, and timestamp resampling behavior.
- Changes: excluded all G1 planner, hand IK, visualization, training, and model
  inference code. The integrated path evaluates the separately supplied
  SMPL-X body model instead of Gear Sonic's neutral-joint data.

## XRoboToolkit PC Service and Python binding

- Upstreams:
  - `https://github.com/XR-Robotics/XRoboToolkit-PC-Service`
  - `https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind`
- PC service/source version used by the installer: `v1.0.0`
- Licenses: PC Service is Apache-2.0; Python binding is MIT
  (`XROBOTOOLKIT_PYBIND_LICENSE.txt`).
- Integrated material: a reduced Python binding that exposes only body-data
  availability, body joint poses, timestamps, initialization, and shutdown.

## SMPL-X

- Upstream: `https://smpl-x.is.tue.mpg.de/`
- Integrated material: user-supplied neutral SMPL-X body model,
  `pico_sim2sim/smplx/SMPLX_NEUTRAL.pkl`.
- Runtime: `smplx` Python package v0.1.28.
- License: the model file has separate SMPL-X download and usage terms. The
  Git LFS rule is only a transport mechanism and does not grant redistribution
  rights.
