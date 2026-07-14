from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
from omegaconf import OmegaConf

from humanoidverse.utils.robot_spec import load_robot_training_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_CONFIG = REPO_ROOT / "configs" / "robots" / "mini3.yaml"
HYDRA_CONFIG = REPO_ROOT / "humanoidverse" / "config" / "robot" / "mini3" / "mini3_auto.yaml"
XML_PATH = REPO_ROOT / "humanoidverse" / "data" / "robots" / "mini3_mjlab" / "mini3.xml"

CONTROL_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
]

KPS = [
    60.0, 55.0, 25.0, 60.0, 50.0, 45.0,
    60.0, 55.0, 25.0, 60.0, 50.0, 45.0,
    65.0,
    30.0, 25.0, 30.0, 20.0,
    30.0, 25.0, 30.0, 20.0,
]

KDS = [
    4.5, 2.8, 1.1, 4.5, 1.2, 1.2,
    4.5, 2.8, 1.1, 4.5, 1.2, 1.2,
    3.0,
    1.0, 2.0, 1.0, 1.0,
    1.0, 2.0, 1.0, 1.0,
]


class Mini3RobotConfigTest(unittest.TestCase):
    def test_mjcf_is_self_contained_and_has_stand_keyframe(self) -> None:
        root = ET.parse(XML_PATH).getroot()
        self.assertEqual(root.attrib["model"], "mini3")
        self.assertEqual(root.find("compiler").attrib["meshdir"], "meshes")
        for mesh in root.findall("./asset/mesh"):
            self.assertEqual(Path(mesh.attrib["file"]).name, mesh.attrib["file"])

        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.assertEqual((model.nq, model.nv, model.nu, model.nbody - 1), (28, 27, 21, 24))
        self.assertEqual(model.nkey, 1)
        self.assertEqual(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, 0), "stand")
        self.assertEqual(model.key_qpos.shape[1], model.nq)

    def test_robot_training_spec_matches_mjcf_layout(self) -> None:
        training = load_robot_training_spec(ROBOT_CONFIG)
        robot = training.robot

        self.assertEqual(robot.name, "mini3")
        self.assertEqual(robot.base_body, "base_link")
        self.assertEqual(robot.free_joint, "floating_base")
        self.assertEqual(robot.control_joint_names, CONTROL_JOINTS)
        self.assertEqual(robot.actuator_joint_names, CONTROL_JOINTS)
        self.assertEqual(len(robot.body_names), 24)
        self.assertEqual(robot.feet, ["left_ankle_roll_link", "right_ankle_roll_link"])
        self.assertEqual(robot.hands, [])

        self.assertEqual(training.hydra_robot, "mini3/mini3_auto")
        self.assertEqual(training.actuator["source"], "yaml")
        self.assertEqual(len(training.effort_limits), 21)
        self.assertEqual(len(training.velocity_limits), 21)
        self.assertEqual([training.stiffness[name] for name in CONTROL_JOINTS], KPS)
        self.assertEqual([training.damping[name] for name in CONTROL_JOINTS], KDS)
        self.assertEqual(training.contact_bodies, ["left_ankle_roll_link", "right_ankle_roll_link"])
        self.assertEqual(training.left_ankle_dof_names, ["left_ankle_pitch_joint", "left_ankle_roll_joint"])
        self.assertEqual(training.right_ankle_dof_names, ["right_ankle_pitch_joint", "right_ankle_roll_joint"])

    def test_hydra_config_dimensions_and_order_match_robot_spec(self) -> None:
        cfg = OmegaConf.load(HYDRA_CONFIG).robot
        self.assertEqual(cfg.num_bodies, 24)
        self.assertEqual(cfg.dof_obs_size, 21)
        self.assertEqual(cfg.actions_dim, 21)
        self.assertEqual(cfg.lower_body_actions_dim, 12)
        self.assertEqual(cfg.upper_body_actions_dim, 9)
        self.assertEqual(list(cfg.dof_names), CONTROL_JOINTS)
        self.assertEqual(len(cfg.body_names), 24)
        self.assertEqual(len(cfg.dof_pos_lower_limit_list), 21)
        self.assertEqual(len(cfg.dof_pos_upper_limit_list), 21)
        self.assertEqual([cfg.control.stiffness[name] for name in CONTROL_JOINTS], KPS)
        self.assertEqual([cfg.control.damping[name] for name in CONTROL_JOINTS], KDS)

    def test_motionlib_uses_actuator_joint_references_and_detects_freejoint(self) -> None:
        # Import lazily because MotionLib loads the simulator/mesh stack.
        from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

        cfg = OmegaConf.create(
            {
                "asset": {"assetRoot": str(XML_PATH.parent), "assetFileName": XML_PATH.name},
                "extend_config": [],
                "has_mesh": False,
            }
        )
        humanoid = Humanoid_Batch(cfg)
        self.assertEqual(humanoid.num_dof, 21)
        self.assertTrue(humanoid.has_freejoint)
        self.assertEqual(tuple(humanoid.dof_axis.shape), (21, 3))
        self.assertEqual(len(humanoid.actuated_joints_idx), 21)


if __name__ == "__main__":
    unittest.main()
