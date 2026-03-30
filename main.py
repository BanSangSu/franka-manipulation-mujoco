import argparse
from typing import Any, Dict

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml

from mujoco_app.mj_simulation import MjSim


# Metric for success
def check_object_in_basket(sim: MjSim) -> dict:
    """Check if the grasp object is inside the basket.

    Returns a dict with keys:
        - in_basket: bool - True if object is in basket
        - in_x, in_y, in_z: bool - Individual bound checks
        - object_pos: list - [x, y, z] position of object
    """
    # Get basket parameters
    basket_center = sim.ids.get("basket_center")
    basket_dims = sim.ids.get("basket_dims")
    basket_height = sim.ids.get("basket_height")

    # Default if basket not configured
    if basket_center is None or basket_dims is None or basket_height is None:
        return {
            "in_basket": False,
            "in_x": False,
            "in_y": False,
            "in_z": False,
            "object_pos": None,
        }

    # Get grasp object body ID
    grasp_obj_info = sim.ids.get("grasp_object", {})
    obj_body_name = grasp_obj_info.get("body_name", "sample_object")

    obj_body_id = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_BODY, obj_body_name
    )

    if obj_body_id < 0:
        return {
            "in_basket": False,
            "in_x": False,
            "in_y": False,
            "in_z": False,
            "object_pos": None,
        }

    # Get object position
    obj_pos = sim.data.xpos[obj_body_id].copy()

    # Check bounds
    in_x = abs(obj_pos[0] - basket_center[0]) < basket_dims[0] / 2.0
    in_y = abs(obj_pos[1] - basket_center[1]) < basket_dims[1] / 2.0
    basket_bottom = basket_center[2] - basket_height / 2.0
    in_z = obj_pos[2] > basket_bottom

    return {
        "in_basket": in_x and in_y and in_z,
        "in_x": in_x,
        "in_y": in_y,
        "in_z": in_z,
        "object_pos": obj_pos.tolist(),
    }


# point projection from 3D to 2D
def project_points(X_world, K, R_cw, t_cw):
    X_cam = (R_cw @ X_world.T + t_cw[:, None]).T
    x = (K @ X_cam.T).T
    return x[:, :2] / x[:, 2:3]


# Helper method to visualize RGB and DEPTH image
def show_rgb_depth(
    rgb,
    depth,
    cam_name,
    near=None,
    far=None,
    cmap="viridis",
    figsize=(10, 4),
    title_rgb="RGB",
    title_depth="Depth",
):
    """
    Display RGB and depth images side-by-side.

    Args:
        rgb (H, W, 3): uint8 RGB image
        depth (H, W): float depth image (meters)
        cam_name (str): name of the camera
        near (float, optional): near clip for visualization
        far (float, optional): far clip for visualization
        cmap (str): colormap for depth
        figsize (tuple): matplotlib figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title(cam_name + "_" + title_rgb)
    axes[0].axis("off")

    # Depth processing
    depth_vis = depth.astype(np.float32)

    if near is not None and far is not None:
        depth_vis = np.clip(depth_vis, near, far)

    # Hide invalid depth
    depth_vis[~np.isfinite(depth_vis)] = np.nan

    axes[1].set_title(cam_name + "_" + title_depth)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


# Experiment runner example
def runner(config: Dict[str, Any], num_experiments: int):
    # Configure in YAML
    cam_cfg = config.get("mujoco", {}).get("camera", {})
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)
    near = cam_cfg.get("near", 0.01)
    far = cam_cfg.get("far", 5.0)
    fovy = cam_cfg.get("fovy", 58.0)

    sim = MjSim(config)
    for _ in range(num_experiments):
        sim.reset()

        # For sim stabilization
        for _ in range(1000):
            sim.step()

        # lower iterations per step for reaching the target pose
        print("Moving to target pose...")
        for t in range(100000):
            sim.step()

            # Showcasing some operations that can be done with the simulation
            rgb, depth, intrinsic, extrinsic = sim.render_camera(
                "side_cam",
                width=width,
                height=height,
                near=near,
                far=far,
                fovy=fovy,
            )
            ee_body_name = sim.robot_settings.get("ee_body_name", "hand")
            ee_body_id = mujoco.mj_name2id(
                sim.model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name
            )
            ee_pos = sim.data.xpos[ee_body_id].copy()
            # Robot should not collide with obstacles
            # This condition must be there
            if sim.check_robot_obstacle_collision():
                print("Collision!")
                break
    sim.close()
    print("Simulation completed.")


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # You can make runner for one experiment
    runner(config, 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/test_config_mj.yaml"
    )
    args = parser.parse_args()
    main(config_path=args.config)
