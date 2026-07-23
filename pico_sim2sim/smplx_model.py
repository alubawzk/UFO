"""Real neutral SMPL-X body-model adapter for JOYIn and Sonic.

The adapter loads the licensed ``SMPLX_NEUTRAL.pkl`` supplied with this
package and exposes the first 22 body joints in JOYIn's
``name -> (position, quaternion_wxyz)`` representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

DEFAULT_SMPLX_MODEL = Path(__file__).resolve().parent / "smplx" / "SMPLX_NEUTRAL.pkl"
JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
SMPLX_NUM_BETAS = 16
SONIC_JOINT_INDICES = np.asarray([*range(22), 39, 54], dtype=np.int64)


@dataclass(frozen=True)
class SmplxBodyOutput:
    """SMPL-X forward result used by the integrated live pipeline."""

    positions: np.ndarray
    rotations_wxyz: np.ndarray
    sonic_positions: np.ndarray

    def as_joyin_data(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {name: (self.positions[index].copy(), self.rotations_wxyz[index].copy()) for index, name in enumerate(JOINT_NAMES)}


class NeutralSmplxBodyModel:
    """Load and evaluate the real neutral SMPL-X pickle."""

    joint_names = JOINT_NAMES

    def __init__(
        self,
        model_file: Path | str = DEFAULT_SMPLX_MODEL,
        *,
        device: str | torch.device = "cpu",
        num_betas: int = SMPLX_NUM_BETAS,
    ):
        self.model_file = Path(model_file).expanduser().resolve()
        if not self.model_file.is_file():
            raise FileNotFoundError(
                f"SMPL-X model not found: {self.model_file}. Place the licensed SMPLX_NEUTRAL.pkl at pico_sim2sim/smplx/SMPLX_NEUTRAL.pkl."
            )
        self.device = torch.device(device)
        self.num_betas = int(num_betas)
        if self.num_betas <= 0:
            raise ValueError("num_betas must be positive")

        try:
            import smplx
        except ImportError as error:
            raise ImportError("The real SMPL-X model requires the 'smplx' package; run: uv sync --extra pico-teleop") from error

        # smplx.create() expects the directory above its model-type folder:
        # <model_path>/smplx/SMPLX_NEUTRAL.pkl.
        self.model = smplx.create(
            str(self.model_file.parent.parent),
            model_type="smplx",
            gender="neutral",
            ext=self.model_file.suffix.lstrip("."),
            num_betas=self.num_betas,
            use_pca=False,
            batch_size=1,
        ).to(self.device)
        self.model.eval()
        self.parents = np.asarray(self.model.parents.detach().cpu(), dtype=np.int64)
        if self.parents.shape[0] < len(JOINT_NAMES):
            raise ValueError(f"SMPL-X model has only {self.parents.shape[0]} joints; expected at least {len(JOINT_NAMES)}")

        self._zero_hand_pose = torch.zeros((1, 45), dtype=torch.float32, device=self.device)
        self._zero_face_pose = torch.zeros((1, 3), dtype=torch.float32, device=self.device)

    def forward(
        self,
        *,
        root_orient: np.ndarray,
        pose_body: np.ndarray,
        trans: np.ndarray,
        betas: np.ndarray | None = None,
    ) -> SmplxBodyOutput:
        """Run the real body model for one frame."""

        root = np.asarray(root_orient, dtype=np.float32).reshape(1, 3)
        body = np.asarray(pose_body, dtype=np.float32).reshape(1, 63)
        translation = np.asarray(trans, dtype=np.float32).reshape(1, 3)
        shape = (
            np.zeros((1, self.num_betas), dtype=np.float32)
            if betas is None
            else np.asarray(betas, dtype=np.float32).reshape(1, self.num_betas)
        )
        if not all(np.all(np.isfinite(value)) for value in (root, body, translation, shape)):
            raise ValueError("SMPL-X inputs must be finite")

        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            output = self.model(
                betas=tensor(shape),
                global_orient=tensor(root),
                body_pose=tensor(body),
                transl=tensor(translation),
                left_hand_pose=self._zero_hand_pose,
                right_hand_pose=self._zero_hand_pose,
                jaw_pose=self._zero_face_pose,
                leye_pose=self._zero_face_pose,
                reye_pose=self._zero_face_pose,
                return_full_pose=True,
            )

        joint_positions = output.joints[0].detach().cpu().numpy().astype(np.float64, copy=False)
        local_rotvec = output.full_pose[0].detach().cpu().numpy().reshape(-1, 3)[: len(JOINT_NAMES)]
        local_rotations = Rotation.from_rotvec(local_rotvec)
        global_rotations: list[Rotation] = []
        for index, local_rotation in enumerate(local_rotations):
            parent = int(self.parents[index])
            global_rotations.append(local_rotation if parent < 0 else global_rotations[parent] * local_rotation)
        rotations_wxyz = np.stack([rotation.as_quat(scalar_first=True) for rotation in global_rotations])
        return SmplxBodyOutput(
            positions=joint_positions[: len(JOINT_NAMES)].copy(),
            rotations_wxyz=rotations_wxyz,
            sonic_positions=joint_positions[SONIC_JOINT_INDICES].copy(),
        )

    def joyin_data(
        self,
        *,
        root_orient: np.ndarray,
        pose_body: np.ndarray,
        trans: np.ndarray,
        betas: np.ndarray | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return self.forward(root_orient=root_orient, pose_body=pose_body, trans=trans, betas=betas).as_joyin_data()
