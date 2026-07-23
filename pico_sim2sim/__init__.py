"""Self-contained PICO/Sonic to Mini3 MuJoCo integration.

The package contains only the pose-streaming path used by UFO.  It does not
vendor the unrelated G1 planner, visualization, or training components from
Gear Sonic.
"""

from .smplx_model import DEFAULT_SMPLX_MODEL, NeutralSmplxBodyModel

__all__ = ["DEFAULT_SMPLX_MODEL", "NeutralSmplxBodyModel"]
