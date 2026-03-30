#!/usr/bin/env python3
"""
run_pick_and_place.py  –  End-to-end pick-and-place demo.

Usage:
    python run_pick_and_place.py --config configs/test_config_mj.yaml
    python run_pick_and_place.py --config configs/test_config_mj.yaml --perception-only
    python run_pick_and_place.py --config configs/test_config_mj.yaml --viz
    python run_pick_and_place.py --config configs/test_config_mj.yaml --no-obstacle-avoidance
"""

from __future__ import annotations

import argparse
import logging
from logging import config
import sys
from pathlib import Path

import yaml

# Ensure src/ is on the path
ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mujoco_app.mj_simulation import MjSim
from pick_and_place.pipeline import PickAndPlacePipeline


def main():
    parser = argparse.ArgumentParser(description="Pick-and-Place Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/test_config_mj.yaml",
        help="Path to the MuJoCo config YAML.",
    )
    parser.add_argument(
        "--perception-only",
        action="store_true",
        help="Run only the perception stage (skip planning & control).",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Able Open3D visualisation windows.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.005,
        help="Voxel grid size for point-cloud downsampling (metres).",
    )
    parser.add_argument(
        "--depth-trunc",
        type=float,
        default=3.0,
        help="Maximum depth to use (metres).",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["user_cam", "side_cam"],
        # default=["user_cam"],
        # default=["static", "user_cam"],
        # default=["static", "user_cam", "side_cam"],
        help="List of camera names to use (e.g. --cameras side_cam user_cam) to reduce point cloud density.",
    )
    parser.add_argument(
        "--num-grasps",
        type=int,
        default=200,
        help="Number of grasp samples to generate.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Keep at most this many grasps.",
    )
    # ---- Obstacle avoidance options ----
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolo26n.pt",
        help="YOLO model for obstacle detection (e.g. yolo26n.pt, yolo26n-seg.pt).",
    )
    parser.add_argument(
        "--detection-camera",
        type=str,
        default="static",
        help="Camera to use for real-time obstacle detection.",
    )
    parser.add_argument(
        "--no-obstacle-avoidance",
        action="store_true",
        help="Disable real-time obstacle avoidance during MPPI execution.",
    )
    parser.add_argument(
        "--num-experiments",
        type=int,
        default=10,
        help="Number of experiments to run (i.e. number of resets).",
    )
    args = parser.parse_args()

    # ---- Logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ---- Load config & create simulation ----
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    target =""
    # Extract target object from the config if the user didn't provide a manual override
    xml_path = config.get("mujoco", {}).get("grasp_object", {}).get("xml", "")

    # Extract the object name (e.g., 'YcbMasterChefCan')
    if xml_path:
        folder_name = xml_path.split('/')[-2]
        
        # Clean up the name (e.g., "YcbMasterChefCan" -> "master chef can")
        import re
        clean_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', folder_name.replace('Ycb', '')).lower().strip()
        
        target = clean_name + "."

    logging.info(f"Targeting extracted object: {target}")

    logging.info("Creating MuJoCo simulation …")
    sim = MjSim(config)
    
    logging.info("Simulation settled.")
    for _ in range(args.num_experiments):
        sim.reset()

        # Let physics settle
        for _ in range(500): # for the test 1s
        # for _ in range(2000): # for the eval 4s
            sim.step()

        # ---- Build pipeline ----
        pipeline = PickAndPlacePipeline(
            sim=sim,
            target_object=target,
            camera_names=args.cameras,
            voxel_size=args.voxel_size,
            depth_trunc=args.depth_trunc,
            num_grasps=args.num_grasps,
            topk=args.topk,
            visualize=args.viz,
            yolo_model=args.yolo_model,
            detection_camera=args.detection_camera,
            enable_obstacle_avoidance=not args.no_obstacle_avoidance,
        )

        # ---- Execute ----
        if args.perception_only:
            logging.info("Running perception only …")
            target_pts, target_clr, scene_pts, scene_clr = pipeline.run_perception()
            logging.info("Target points: %d", target_pts.shape[0])
            logging.info("Scene  points: %d", scene_pts.shape[0])
        else:
            logging.info("Running full pipeline …")
            result = pipeline.run()
            logging.info("Pipeline finished.  Grasps found: %d", result.grasp_poses.shape[0])
            if result.best_grasp is not None:
                logging.info("Best grasp pose:\n%s", result.best_grasp)
            logging.info("Grasp success: %s", result.grasp_success)
            logging.info("Place success: %s", result.place_success)
            logging.info("Overall success: %s", result.success)

        # Stop obstacle tracker thread if it was started
        if hasattr(pipeline, 'motion_planner') and hasattr(pipeline.motion_planner, '_obstacle_tracker'):
            tracker = pipeline.motion_planner._obstacle_tracker
            if tracker is not None and hasattr(tracker, 'stop'):
                tracker.stop()
        
        for _ in range(1000): # for the test 2s
            sim.step()

            if sim.check_robot_obstacle_collision():
                logging.warning("Collision detected")
                break
    
    sim.close()
    logging.info("Done.")


if __name__ == "__main__":
    main()