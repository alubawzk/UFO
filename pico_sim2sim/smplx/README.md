# SMPL-X model

`SMPLX_NEUTRAL.pkl` is the real neutral SMPL-X body model used at runtime by
both `pico_sim2sim.sonic_server` and
`humanoidverse.mujoco_pico_teleop`.

The file is approximately 519 MiB and is tracked through Git LFS. After a
fresh clone, verify it with:

```bash
git lfs pull --include="pico_sim2sim/smplx/SMPLX_NEUTRAL.pkl"
stat -c "%n %s bytes" pico_sim2sim/smplx/SMPLX_NEUTRAL.pkl
```

A small text file at this path is only an unresolved Git LFS pointer. Running
`bash pico_sim2sim/install.sh` also detects this condition.

The model is subject to the SMPL-X license:
<https://smpl-x.is.tue.mpg.de/modellicense.html>. Ensure you have permission
before pushing or redistributing the model through a Git remote.
