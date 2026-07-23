"""JOYIn General Motion Retargeting, limited to SMPL-X -> Mini3.

This is a scoped copy of JOYIn-Retarget's MIT-licensed GMR implementation.
Asset lookup was changed to package-local files and unrelated robot formats
were removed.  The two-stage Mink IK behavior is intentionally preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ROBOT_XML = PACKAGE_DIR / "mini3_ik.xml"
DEFAULT_IK_CONFIG = PACKAGE_DIR / "smplx_to_mini3.json"


class GeneralMotionRetargeting:
    """Retarget global SMPL-X body transforms to a Mini3 free-base qpos."""

    def __init__(
        self,
        src_human: str = "smplx",
        tgt_robot: str = "mini3",
        actual_human_height: float | None = None,
        solver: str = "daqp",
        damping: float = 5.0e-1,
        verbose: bool = True,
        use_velocity_limit: bool = True,
        posture_cost: float = 0.0,
        ik_dt: float | None = None,
        max_iter: int = 10,
        *,
        robot_xml: str | Path | None = None,
        ik_config: str | Path | None = None,
    ) -> None:
        if src_human != "smplx" or tgt_robot != "mini3":
            raise ValueError("The integrated retargeter supports only src_human='smplx' and tgt_robot='mini3'")
        self.xml_file = str(Path(robot_xml or DEFAULT_ROBOT_XML).expanduser().resolve())
        self.ik_config_file = str(Path(ik_config or DEFAULT_IK_CONFIG).expanduser().resolve())
        self.model = mujoco.MjModel.from_xml_path(self.xml_file)

        self.robot_dof_names = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[index]): index for index in range(self.model.nv)
        }
        self.robot_body_names = {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, index): index for index in range(self.model.nbody)}
        self.robot_motor_names = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index): index for index in range(self.model.nu)
        }
        if verbose:
            print(f"[GMR] robot={self.xml_file}")
            print(f"[GMR] ik_config={self.ik_config_file}")
            print(f"[GMR] DoF order: {list(self.robot_dof_names)}")
            print(f"[GMR] Body order: {list(self.robot_body_names)}")
            print(f"[GMR] Motor order: {list(self.robot_motor_names)}")

        with Path(self.ik_config_file).open(encoding="utf-8") as stream:
            config = json.load(stream)
        ratio = 1.0 if actual_human_height is None else float(actual_human_height) / float(config["human_height_assumption"])
        config["human_scale_table"] = {name: float(scale) * ratio for name, scale in config["human_scale_table"].items()}

        self.ik_match_table1 = config["ik_match_table1"]
        self.ik_match_table2 = config["ik_match_table2"]
        self.human_root_name = str(config["human_root_name"])
        self.robot_root_name = str(config["robot_root_name"])
        self.use_ik_match_table1 = bool(config["use_ik_match_table1"])
        self.use_ik_match_table2 = bool(config["use_ik_match_table2"])
        self.human_scale_table = config["human_scale_table"]
        self.ground = float(config["ground_height"]) * np.asarray([0.0, 0.0, 1.0])

        self.max_iter = int(max_iter)
        self._ik_dt = None if ik_dt is None else float(ik_dt)
        self.solver = str(solver)
        self.damping = float(damping)
        self.posture_cost = float(posture_cost)

        self.human_body_to_task1: dict[str, mink.FrameTask] = {}
        self.human_body_to_task2: dict[str, mink.FrameTask] = {}
        self.pos_offsets1: dict[str, np.ndarray] = {}
        self.rot_offsets1: dict[str, Rotation] = {}
        self.pos_offsets2: dict[str, np.ndarray] = {}
        self.rot_offsets2: dict[str, Rotation] = {}

        self.ik_limits: list[object] = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            actuator_joint_names: list[str] = []
            for actuator_id in range(self.model.nu):
                if self.model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
                    continue
                joint_id = int(self.model.actuator_trnid[actuator_id, 0])
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                if name is not None:
                    actuator_joint_names.append(name)
            self.ik_limits.append(mink.VelocityLimit(self.model, {name: 3.0 * np.pi for name in actuator_joint_names}))

        self.configuration = mink.Configuration(self.model)
        self.tasks1: list[mink.FrameTask] = []
        self.tasks2: list[mink.FrameTask] = []
        self.posture_task: mink.PostureTask | None = None
        self._setup_tasks()
        self.ground_offset = 0.0
        self.scaled_human_data: dict[str, list[np.ndarray]] = {}

    def _add_tasks(
        self,
        table: dict[str, list[object]],
        body_to_task: dict[str, mink.FrameTask],
        pos_offsets: dict[str, np.ndarray],
        rot_offsets: dict[str, Rotation],
        tasks: list[mink.FrameTask],
    ) -> None:
        for frame_name, entry in table.items():
            body_name, position_cost, orientation_cost, position_offset, rotation_offset = entry
            if float(position_cost) == 0.0 and float(orientation_cost) == 0.0:
                continue
            task = mink.FrameTask(
                frame_name=frame_name,
                frame_type="body",
                position_cost=position_cost,
                orientation_cost=orientation_cost,
                lm_damping=3,
            )
            body_to_task[str(body_name)] = task
            pos_offsets[str(body_name)] = np.asarray(position_offset, dtype=np.float64) - self.ground
            rot_offsets[str(body_name)] = Rotation.from_quat(rotation_offset, scalar_first=True)
            tasks.append(task)

    def _setup_tasks(self) -> None:
        self._add_tasks(
            self.ik_match_table1,
            self.human_body_to_task1,
            self.pos_offsets1,
            self.rot_offsets1,
            self.tasks1,
        )
        self._add_tasks(
            self.ik_match_table2,
            self.human_body_to_task2,
            self.pos_offsets2,
            self.rot_offsets2,
            self.tasks2,
        )
        if self.posture_cost > 0.0:
            cost = np.zeros(self.model.nv)
            cost[3:6] = self.posture_cost
            cost[6:] = self.posture_cost * 0.001
            self.posture_task = mink.PostureTask(self.model, cost=cost)
            default_qpos = np.zeros(self.model.nq)
            default_qpos[3] = 1.0
            self.posture_task.set_target(default_qpos)

    @staticmethod
    def _copy_numpy(
        human_data: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, list[np.ndarray]]:
        return {
            name: [
                np.asarray(transform[0], dtype=np.float64).copy(),
                np.asarray(transform[1], dtype=np.float64).copy(),
            ]
            for name, transform in human_data.items()
        }

    @staticmethod
    def _scale_human_data(
        human_data: dict[str, list[np.ndarray]],
        human_root_name: str,
        scale_table: dict[str, float],
    ) -> dict[str, list[np.ndarray]]:
        root_pos, root_quat = human_data[human_root_name]
        scaled_root_pos = float(scale_table[human_root_name]) * root_pos
        result = {human_root_name: [scaled_root_pos, root_quat]}
        for body_name, (position, quaternion) in human_data.items():
            if body_name == human_root_name or body_name not in scale_table:
                continue
            local_position = (position - root_pos) * float(scale_table[body_name])
            result[body_name] = [local_position + scaled_root_pos, quaternion]
        return result

    @staticmethod
    def _offset_human_data(
        human_data: dict[str, list[np.ndarray]],
        pos_offsets: dict[str, np.ndarray],
        rot_offsets: dict[str, Rotation],
    ) -> dict[str, list[np.ndarray]]:
        result: dict[str, list[np.ndarray]] = {}
        for body_name, (position, quaternion) in human_data.items():
            result[body_name] = [position.copy(), quaternion.copy()]
            if body_name not in rot_offsets:
                continue
            updated_rotation = Rotation.from_quat(quaternion, scalar_first=True) * rot_offsets[body_name]
            updated_quaternion = updated_rotation.as_quat(scalar_first=True)
            result[body_name] = [
                position + updated_rotation.apply(pos_offsets[body_name]),
                updated_quaternion,
            ]
        return result

    @staticmethod
    def _offset_to_ground(human_data: dict[str, list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
        foot_heights = [transform[0][2] for name, transform in human_data.items() if "foot" in name.lower()]
        if not foot_heights:
            raise ValueError("JOYIn ground alignment requires at least one human foot target")
        shift = np.asarray([0.0, 0.0, -min(foot_heights) + 0.1])
        return {name: [position + shift, quaternion.copy()] for name, (position, quaternion) in human_data.items()}

    def update_targets(
        self,
        human_data: dict[str, tuple[np.ndarray, np.ndarray]],
        offset_to_ground: bool = False,
    ) -> None:
        data = self._copy_numpy(human_data)
        data = self._scale_human_data(data, self.human_root_name, self.human_scale_table)
        data = self._offset_human_data(data, self.pos_offsets1, self.rot_offsets1)
        ground_shift = np.asarray([0.0, 0.0, self.ground_offset])
        data = {name: [position - ground_shift, quaternion] for name, (position, quaternion) in data.items()}
        if offset_to_ground:
            data = self._offset_to_ground(data)
        self.scaled_human_data = data

        if self.use_ik_match_table1:
            for body_name, task in self.human_body_to_task1.items():
                position, quaternion = data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(quaternion), position))
        if self.use_ik_match_table2:
            for body_name, task in self.human_body_to_task2.items():
                position, quaternion = data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(quaternion), position))

    def _solve_stage(self, tasks: list[mink.FrameTask]) -> None:
        combined_tasks: list[object] = list(tasks)
        if self.posture_task is not None:
            combined_tasks.append(self.posture_task)
        dt = self.configuration.model.opt.timestep if self._ik_dt is None else self._ik_dt

        def error() -> float:
            return float(np.linalg.norm(np.concatenate([task.compute_error(self.configuration) for task in tasks])))

        current_error = error()
        velocity = mink.solve_ik(
            self.configuration,
            combined_tasks,
            dt,
            self.solver,
            self.damping,
            self.ik_limits,
        )
        self.configuration.integrate_inplace(velocity, dt)
        next_error = error()
        iteration = 0
        while current_error - next_error > 0.001 and iteration < self.max_iter:
            current_error = next_error
            velocity = mink.solve_ik(
                self.configuration,
                combined_tasks,
                dt,
                self.solver,
                self.damping,
                self.ik_limits,
            )
            self.configuration.integrate_inplace(velocity, dt)
            next_error = error()
            iteration += 1

    def retarget(
        self,
        human_data: dict[str, tuple[np.ndarray, np.ndarray]],
        offset_to_ground: bool = False,
    ) -> np.ndarray:
        self.update_targets(human_data, offset_to_ground=offset_to_ground)
        if self.posture_task is not None:
            target_qpos = np.zeros(self.model.nq)
            target_qpos[:3] = self.configuration.data.qpos[:3]
            target_qpos[3] = 1.0
            self.posture_task.set_target(target_qpos)
        if self.use_ik_match_table1:
            self._solve_stage(self.tasks1)
        if self.use_ik_match_table2:
            self._solve_stage(self.tasks2)
        return self.configuration.data.qpos.copy()

    def set_ground_offset(self, ground_offset: float) -> None:
        self.ground_offset = float(ground_offset)
