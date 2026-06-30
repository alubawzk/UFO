#!/usr/bin/env python3
"""Validation helpers for the BFM-Zero MJLab migration."""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
G1_CFG = ROOT / "humanoidverse/config/robot/g1/g1_29dof_hard_waist.yaml"
G1_XML = ROOT / "humanoidverse/data/robots/g1_mjlab/g1-mode_13_15.xml"
REWARD_CFG = ROOT / "humanoidverse/config/rewards/reward_bfm_zero.yaml"


EXPECTED_REWARD_KEYS = {
    "penalty_torques",
    "penalty_undesired_contact",
    "penalty_action_rate",
    "penalty_ankle_roll",
    "penalty_feet_ori",
    "feet_heading_alignment",
    "penalty_slippage",
    "limits_dof_pos",
    "limits_dof_vel",
    "limits_torque",
}

REQUIRED_BODY_NAMES = {
    "pelvis",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "waist_yaw_link",
    "waist_roll_link",
}


def _reflected_inertia_from_two_stage_planetary(
    rotor_inertia: tuple[float, float, float],
    gear_ratio: tuple[float, float, float],
) -> float:
    return rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2 + rotor_inertia[1] * gear_ratio[2] ** 2 + rotor_inertia[2]


_ARMATURE_5020 = _reflected_inertia_from_two_stage_planetary((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + (46 / 18), 1 + (56 / 16)))
_ARMATURE_7520_14 = _reflected_inertia_from_two_stage_planetary((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + (48 / 22)))
_ARMATURE_7520_22 = _reflected_inertia_from_two_stage_planetary((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))
_ARMATURE_5010 = _reflected_inertia_from_two_stage_planetary((0.084e-4, 0.015e-4, 0.068e-4), (1, 4, 4))


def _g1_mjlab_mode15_actuator_params(dof_names: list[str]) -> dict[str, list[float]]:
    efforts: list[float] = []
    velocities: list[float] = []
    armatures: list[float] = []
    frictions: list[float] = []

    for joint_name in dof_names:
        if "_hip_pitch_joint" in joint_name:
            effort, velocity, armature = 139.0, 20.0, _ARMATURE_7520_22
        elif "_hip_yaw_joint" in joint_name or joint_name == "waist_yaw_joint":
            effort, velocity, armature = 88.0, 32.0, _ARMATURE_7520_14
        elif "_hip_roll_joint" in joint_name or "_knee_joint" in joint_name:
            effort, velocity, armature = 139.0, 20.0, _ARMATURE_7520_22
        elif "_ankle_pitch_joint" in joint_name or "_ankle_roll_joint" in joint_name:
            effort, velocity, armature = 50.0, 37.0, 2.0 * _ARMATURE_5020
        elif joint_name in ("waist_pitch_joint", "waist_roll_joint"):
            effort, velocity, armature = 50.0, 37.0, 2.0 * _ARMATURE_5020
        elif (
            "_shoulder_pitch_joint" in joint_name
            or "_shoulder_roll_joint" in joint_name
            or "_shoulder_yaw_joint" in joint_name
            or "_elbow_joint" in joint_name
            or "_wrist_roll_joint" in joint_name
        ):
            effort, velocity, armature = 25.0, 37.0, _ARMATURE_5020
        elif "_wrist_pitch_joint" in joint_name or "_wrist_yaw_joint" in joint_name:
            effort, velocity, armature = 13.4, 27.0, _ARMATURE_5010
        else:
            raise AssertionError(f"No G1 mode-15 actuator parameters for joint: {joint_name}")
        efforts.append(effort)
        velocities.append(velocity)
        armatures.append(armature)
        frictions.append(0.01)

    return {
        "effort_limit": efforts,
        "velocity_limit": velocities,
        "armature": armatures,
        "friction": frictions,
    }


def _load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _xml_root() -> ET.Element:
    if not G1_XML.exists():
        raise AssertionError(f"missing MJLab G1 XML: {G1_XML.relative_to(ROOT)}")
    return ET.parse(G1_XML).getroot()


def _xml_mesh_files(root: ET.Element) -> list[str]:
    files: list[str] = []
    for mesh in root.findall(".//mesh"):
        file_name = mesh.attrib.get("file")
        if file_name and file_name not in files:
            files.append(file_name)
    return files


def _xml_joint_names(root: ET.Element) -> list[str]:
    return [joint.attrib["name"] for joint in root.findall(".//joint") if joint.attrib.get("name") != "floating_base_joint"]


def _xml_body_names(root: ET.Element) -> list[str]:
    return [body.attrib["name"] for body in root.findall(".//body")]


def _match_joint_value(joint_name: str, values: dict[str, float], default: float = 0.0) -> float:
    for key, value in values.items():
        if key in joint_name:
            return float(value)
    return float(default)


def _action_scale(robot: dict) -> list[float]:
    dof_names = list(robot["dof_names"])
    stiffness = {str(k): float(v) for k, v in robot["control"]["stiffness"].items()}
    effort_limits = [float(v) for v in robot["dof_effort_limit_list"]]
    scales: list[float] = []
    for i, joint_name in enumerate(dof_names):
        scale = float(robot["control"]["action_scale"])
        if bool(robot["control"].get("action_rescale", False)):
            kp = _match_joint_value(joint_name, stiffness)
            if kp <= 0.0:
                raise AssertionError(f"invalid kp for action_rescale: {joint_name}={kp}")
            scale *= effort_limits[i] / kp
        scales.append(scale)
    return scales


def _assert_finite(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            _assert_finite(child, f"{label}[{i}]")
        return
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.dtype.kind not in "fc":
        return
    if not np.isfinite(arr).all():
        if arr.shape == ():
            raise AssertionError(f"non-finite {label}: scalar value={arr.item()}")
        bad = np.argwhere(~np.isfinite(arr))
        first = tuple(int(x) for x in bad[0]) if bad.size else ()
        raise AssertionError(f"non-finite {label}: shape={arr.shape} first_bad={first} value={arr[first] if first else arr}")


def _build_env_cfg(num_envs: int = 1):
    from humanoidverse.agents.envs.humanoidverse_mjlab import (
        G1_MJLAB_MJCF_PATH,
        _compose_humanoidverse_config,
        make_mjlab_bfmzero_env_cfg,
    )

    if Path(G1_MJLAB_MJCF_PATH) != G1_XML.relative_to(ROOT):
        raise AssertionError(f"code XML path mismatch: {G1_MJLAB_MJCF_PATH} != {G1_XML.relative_to(ROOT)}")
    hv_config, _ = _compose_humanoidverse_config(
        num_envs=num_envs,
        relative_config_path="exp/bfm_zero/bfm_zero",
        hydra_overrides=[
            "robot=g1/g1_29dof_hard_waist",
            "robot.control.action_scale=0.25",
            "robot.control.action_clip_value=5.0",
            "robot.control.normalize_action_to=5.0",
            "env.config.lie_down_init=True",
            "env.config.lie_down_init_prob=0.3",
        ],
        lafan_tail_path="humanoidverse/data/lafan_29dof_10s-clipped.pkl",
        disable_obs_noise=False,
        disable_domain_randomization=False,
        max_episode_length_s=None,
        root_height_obs=True,
    )
    return hv_config, make_mjlab_bfmzero_env_cfg(
        hv_config,
        num_envs=num_envs,
        seed=123,
        mjcf_path=G1_MJLAB_MJCF_PATH,
        auto_reset=False,
    )


def asset_alignment() -> None:
    robot = _load_yaml(G1_CFG)["robot"]
    rewards = _load_yaml(REWARD_CFG)["rewards"]["reward_scales"]
    root = _xml_root()
    xml_text = G1_XML.read_text()
    xml_joints = _xml_joint_names(root)
    xml_bodies = _xml_body_names(root)
    xml_bodies_without_toes = [name for name in xml_bodies if not name.endswith("_toe_link")]
    xml_actuator_joints = [act.attrib["joint"] for act in root.findall(".//actuator/*") if "joint" in act.attrib]

    body_names = list(robot["body_names"])
    dof_names = list(robot["dof_names"])
    motion_body_names = list(robot["motion"]["body_names"])
    bfm_effort_limits = [float(v) for v in robot["dof_effort_limit_list"]]
    effort_scale = float(robot.get("dof_effort_limit_scale", 1.0))
    scaled_effort_limits = [v * effort_scale for v in bfm_effort_limits]
    G1_actuator_params = _g1_mjlab_mode15_actuator_params(dof_names)
    effort_limits = G1_actuator_params["effort_limit"]
    velocity_limits = G1_actuator_params["velocity_limit"]
    armature = G1_actuator_params["armature"]
    friction = G1_actuator_params["friction"]
    action_scale = _action_scale(robot)

    errors: list[str] = []
    if G1_XML.relative_to(ROOT).as_posix() != "humanoidverse/data/robots/g1_mjlab/g1-mode_13_15.xml":
        errors.append(f"unexpected XML path: {G1_XML.relative_to(ROOT)}")
    if "actuatorfrcrange" in xml_text:
        errors.append("MJLab G1 XML contains actuatorfrcrange")
    missing_meshes = [file for file in _xml_mesh_files(root) if not (G1_XML.parent / "meshes" / file).exists()]
    if missing_meshes:
        errors.append(f"missing mesh files: {missing_meshes}")
    if len(dof_names) != 29:
        errors.append(f"BFM-Zero action/dof count must be 29, got {len(dof_names)}")
    if dof_names != xml_joints:
        errors.append(f"joint order mismatch: yaml={dof_names} xml={xml_joints}")
    if xml_actuator_joints:
        errors.append(f"XML must not provide actuators; found joints={xml_actuator_joints}")
    if body_names != xml_bodies_without_toes:
        errors.append("body order mismatch after dropping MJLab G1 toe helper bodies")
    missing_required_bodies = sorted(REQUIRED_BODY_NAMES - set(xml_bodies))
    if missing_required_bodies:
        errors.append(f"missing required body names: {missing_required_bodies}")
    if motion_body_names != body_names:
        errors.append("robot.motion.body_names does not match robot.body_names")
    for contact_body in robot["contact_bodies"]:
        if contact_body not in body_names:
            errors.append(f"contact body missing from BFM body order: {contact_body}")
    for parent in [item["parent_name"] for item in robot["motion"]["extend_config"]]:
        if parent not in body_names:
            errors.append(f"motion extend parent missing from BFM body order: {parent}")
    missing_rewards = sorted(EXPECTED_REWARD_KEYS - set(rewards.keys()))
    if missing_rewards:
        errors.append(f"missing reward keys: {missing_rewards}")
    if any(math.isclose(a, b, rel_tol=0.0, abs_tol=1.0e-6) for a, b in zip(bfm_effort_limits, scaled_effort_limits)):
        errors.append("effort limit validation cannot distinguish unscaled and scaled values")

    hv_config = mjlab_cfg = None
    if not errors:
        hv_config, mjlab_cfg = _build_env_cfg(num_envs=1)
        actuators = tuple(mjlab_cfg.scene.entities["robot"].articulation.actuators)
        if len(actuators) != 29:
            errors.append(f"final MJLab actuator count must be 29, got {len(actuators)}")
        final_efforts = [float(act.effort_limit) for act in actuators]
        final_saturation_efforts = [float(act.saturation_effort) for act in actuators]
        final_velocity_limits = [float(act.velocity_limit) for act in actuators]
        final_armature = [float(act.armature) for act in actuators]
        final_friction = [float(act.frictionloss) for act in actuators]
        if final_efforts != effort_limits:
            errors.append(f"final actuator effort limits mismatch G1 mode_15: {final_efforts} != {effort_limits}")
        if final_saturation_efforts != effort_limits:
            errors.append("final actuator saturation_effort mismatch G1 mode_15")
        if final_velocity_limits != velocity_limits:
            errors.append(f"final actuator velocity limits mismatch G1 mode_15: {final_velocity_limits} != {velocity_limits}")
        if final_efforts == scaled_effort_limits:
            errors.append("final actuator effort limits incorrectly include dof_effort_limit_scale")
        if final_armature != armature:
            errors.append("final actuator armature mismatch G1 mode_15")
        if final_friction != friction:
            errors.append("final actuator frictionloss mismatch G1 mode_15")
        if list(hv_config.robot.dof_names) != dof_names:
            errors.append("composed Hydra dof_names changed")
        action_cfg = mjlab_cfg.actions["actions"]
        if tuple(action_cfg.actuator_names) != tuple(dof_names):
            errors.append("MJLab action actuator_names no longer match BFM-Zero dof_names order")
        if not bool(action_cfg.preserve_order):
            errors.append("MJLab action config must preserve BFM-Zero dof order")
        if not bool(action_cfg.use_default_offset):
            errors.append("MJLab action config must use default joint position offset")
        if not isinstance(action_cfg.scale, dict):
            errors.append(f"MJLab action scale must be a per-joint dict, got {type(action_cfg.scale)}")
        else:
            final_action_scale = [float(action_cfg.scale[name]) for name in dof_names]
            if not np.allclose(final_action_scale, action_scale, rtol=0.0, atol=1.0e-8):
                errors.append(f"MJLab action scale mismatch BFM-Zero formula: {final_action_scale} != {action_scale}")

    if errors:
        for err in errors:
            print(f"FAIL: {err}")
        raise SystemExit(1)

    print("PASS asset_alignment")
    print(f"  xml={G1_XML.relative_to(ROOT)}")
    print(f"  joints={len(dof_names)} actuators=29 bodies={len(body_names)} xml_extra_bodies=['left_toe_link', 'right_toe_link']")
    print(f"  no_actuatorfrcrange=True xml_actuator_count=0")
    print(f"  dof_effort_limit_scale={effort_scale} ignored_for_mjlab_actuator_limits=True")
    print("  target_dof_pos=default_dof_pos + action * action_scale")
    print(f"  action_scale={action_scale}")
    print(f"  effort_limit={effort_limits}")
    print(f"  velocity_limit={velocity_limits}")
    print(f"  armature={armature}")
    print(f"  friction={friction}")


def smoke(num_envs: int, steps: int, device: str, torso_force_limit: float) -> None:
    from humanoidverse.agents.envs.humanoidverse_mjlab import HumanoidVerseMjlabConfig, G1_MJLAB_MJCF_PATH

    env_cfg = HumanoidVerseMjlabConfig(
        name="humanoidverse_mjlab",
        device=device,
        lafan_tail_path="humanoidverse/data/lafan_29dof_10s-clipped.pkl",
        mjcf_path=G1_MJLAB_MJCF_PATH,
        relative_config_path="exp/bfm_zero/bfm_zero",
        include_last_action=True,
        hydra_overrides=[
            "robot=g1/g1_29dof_hard_waist",
            "robot.control.action_scale=0.25",
            "robot.control.action_clip_value=5.0",
            "robot.control.normalize_action_to=5.0",
            "env.config.lie_down_init=True",
            "env.config.lie_down_init_prob=0.3",
        ],
        include_history_actor=True,
        root_height_obs=True,
        auto_reset=False,
        seed=123,
    )
    env, _ = env_cfg.build(num_envs=num_envs)
    obs, info = env.reset()
    required_obs = {"state", "privileged_state", "last_action", "history_actor", "time"}
    missing = required_obs - set(obs)
    if missing:
        raise AssertionError(f"missing observation keys: {sorted(missing)}")
    if obs["state"].shape[0] != num_envs or env.single_action_space.shape != (29,):
        raise AssertionError(f"bad shape: state={obs['state'].shape}, action={env.single_action_space.shape}")
    if info["qpos"].shape[-1] != 36 or info["qvel"].shape[-1] != 35:
        raise AssertionError(f"bad qpos/qvel shape: {info['qpos'].shape}, {info['qvel'].shape}")
    _assert_finite(obs, "reset.obs")
    _assert_finite(info, "reset.info")

    torso_force_max = 0.0
    for step_idx in range(steps):
        action = env.action_space.sample().astype(np.float32)
        _assert_finite(action, f"step{step_idx}.action")
        obs, reward, terminated, truncated, info = env.step(action)
        _assert_finite(obs, f"step{step_idx}.obs")
        _assert_finite(reward, f"step{step_idx}.reward")
        _assert_finite(terminated, f"step{step_idx}.terminated")
        _assert_finite(truncated, f"step{step_idx}.truncated")
        _assert_finite(info, f"step{step_idx}.info")
        torso_force = env.base_env.contact_forces[:, env.base_env.torso_index, :].detach().cpu().numpy()
        torso_force_max = max(torso_force_max, float(np.abs(torso_force).max(initial=0.0)))

    expected_aux = {
        "penalty_torques",
        "penalty_action_rate",
        "limits_dof_pos",
        "limits_torque",
        "penalty_undesired_contact",
        "penalty_feet_ori",
        "penalty_ankle_roll",
        "penalty_slippage",
    }
    if not expected_aux.issubset(info["aux_rewards"]):
        raise AssertionError(f"missing aux reward keys: {sorted(expected_aux - set(info['aux_rewards']))}")
    if torso_force_max > torso_force_limit:
        raise AssertionError(f"torso contact force explosion: max={torso_force_max} limit={torso_force_limit}")
    env.close()
    print(f"PASS smoke num_envs={num_envs} steps={steps} device={device} torso_contact_force_max={torso_force_max:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torso-force-limit", type=float, default=1.0e6)
    args = parser.parse_args()

    asset_alignment()
    if args.smoke:
        smoke(args.num_envs, args.steps, args.device, args.torso_force_limit)
    elif not args.asset_only:
        print("NOTE smoke not requested; pass --smoke to build MJLab env.")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        print(f"FAIL missing dependency: {exc}", file=sys.stderr)
        raise SystemExit(2)
