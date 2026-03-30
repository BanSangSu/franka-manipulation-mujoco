"""High-level helpers for creating and sampling MuJoCo scenes."""
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import yaml

from mujoco_app.mj_simulation import MjSim


def load_cfg(path: str | Path) -> Dict[str, Any]:
    """Loads a YAML configuration file and returns it as a dictionary.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the parsed configuration.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def agrobot(
    cfg_path: str | Path = "configs/test_config_mj.yaml",
    *,
    joint_angles: Optional[Sequence[float]] = None,
    settle_steps: int = 240,
    run_steps: int = 600,
    capture: bool = True,
    camera_target: Optional[Sequence[float]] = None,
) -> Tuple[MjSim, Dict[str, Any]]:
    """Creates a MuJoCo simulation, optionally sets joint angles, and captures renders.

    The 'agrobot' helper provides a repeatable pipeline: load configuration,
    initialize simulation, apply joint configuration, run settling steps,
    capture multi-camera renders, and optionally run additional simulation steps.

    Args:
        cfg_path: Path to the YAML configuration file.
        joint_angles: Optional sequence of joint angle values to set before settling.
        settle_steps: Number of simulation steps to run before capturing.
        run_steps: Number of additional simulation steps after capturing.
        capture: Whether to capture renders from static and extra cameras.
        camera_target: Optional 3D target position for the primary camera.

    Returns:
        A tuple (sim, results) where sim is the MjSim instance and results
        is a dict containing 'static_camera', 'robot_pose', and 'extra_cameras'.
    """
    cfg_path = Path(cfg_path)
    cfg = copy.deepcopy(load_cfg(cfg_path))
    cfg.setdefault("_config_dir", str(cfg_path.parent.resolve()))
    cfg.setdefault("mujoco", {})
    if camera_target is not None:
        cfg["mujoco"].setdefault("camera", {})["target"] = list(camera_target)

    sim = MjSim(cfg)
    sim.reset()

    if joint_angles is not None:
        sim.set_arm_joint_positions(joint_angles)

    if settle_steps > 0:
        sim.step(settle_steps)

    results: Dict[str, Any] = {}

    if capture:
        cam_cfg = cfg["mujoco"]["camera"]
        rgb, depth, seg, P, V = sim.get_static_renders(
            width=cam_cfg["width"],
            height=cam_cfg["height"],
            near=cam_cfg["near"],
            far=cam_cfg["far"],
            fovy=cam_cfg["fovy"],
        )
        results["static_camera"] = {
            "rgb": rgb,
            "depth": depth,
            "seg": seg,
            "projection": P,
            "view": V,
        }
        results["robot_pose"] = sim.robot.get_ee_pose()
        results["extra_cameras"] = sim.render_additional_cameras()

    if run_steps > 0:
        sim.step(run_steps)

    return sim, results
