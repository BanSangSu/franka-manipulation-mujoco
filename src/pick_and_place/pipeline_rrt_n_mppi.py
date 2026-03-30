"""
PickAndPlacePipeline – top-level orchestrator that connects
Perception → Planning → Control → Placing.

Full pipeline:
  1. PERCEPTION: detect target object, build point clouds, generate grasps
  2. PLANNING:   IK + RRT-Connect for collision-free path to grasp
  3. CONTROL:    MPPI tracks path with real-time obstacle avoidance (YOLO + Kalman)
  4. GRASP:      close gripper
  5. PLACE:      find basket, plan path to it (still avoiding obstacles), release

This class is the single entry point that a user script calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from .perception import PerceptionPipeline
from .planning import MotionPlanner

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Stores the output of one pick-and-place cycle."""

    # Perception outputs
    target_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    target_colors: Optional[np.ndarray] = None
    scene_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    scene_colors: Optional[np.ndarray] = None

    # Planning outputs
    grasp_poses: np.ndarray = field(default_factory=lambda: np.empty((0, 4, 4)))
    grasp_confidences: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    best_grasp: Optional[np.ndarray] = None  # (4, 4)
    trajectory: List[np.ndarray] = field(default_factory=list)  # Joint space path

    # Control outputs
    execution_result: Optional[dict] = None
    grasp_success: bool = False

    # Placing outputs
    place_target: Optional[np.ndarray] = None  # (3,) world position
    place_trajectory: List[np.ndarray] = field(default_factory=list)
    place_result: Optional[dict] = None
    place_success: bool = False

    # Overall
    success: bool = False


class PickAndPlacePipeline:
    """Full pick-and-place: Perception → Planning → Control → Place.

    Parameters
    ----------
    sim : MjSim
        Initialised MuJoCo simulation.
    target_object : str
        What to grasp (e.g. "mustard bottle").
    camera_names : list[str] | None
    voxel_size, depth_trunc : float
        Point-cloud builder settings.
    florence_model : str
        Florence-2 model id.
    grasp_config : str | None
        Path to GraspGen gripper config YAML.
    num_grasps, topk : int
        GraspGen inference settings.
    visualize : bool
        Pop up Open3D windows for intermediate results.
    yolo_model : str
        YOLO model for obstacle detection.
    detection_camera : str
        Camera to use for real-time obstacle detection.
    enable_obstacle_avoidance : bool
        Enable real-time obstacle avoidance during MPPI execution.
    """

    def __init__(
        self,
        sim: Any,
        target_object: str = "mustard bottle",
        camera_names: list[str] | None = None,
        voxel_size: float = 0.005,
        depth_trunc: float = 3.0,
        florence_model: str = "microsoft/Florence-2-base-ft",
        grasp_config: str | None = None,
        num_grasps: int = 200,
        topk: int = 50,
        visualize: bool = True,
        yolo_model: str = "yolo26n.pt",
        detection_camera: str = "static",
        enable_obstacle_avoidance: bool = True,
    ):
        self.sim = sim
        self.visualize = visualize
        self.yolo_model = yolo_model
        self.detection_camera = detection_camera
        self.enable_obstacle_avoidance = enable_obstacle_avoidance

        # ------ Stage 1: Perception ------
        self.perception = PerceptionPipeline(
            sim=sim,
            target_object=target_object,
            camera_names=camera_names,
            voxel_size=voxel_size,
            depth_trunc=depth_trunc,
            florence_model=florence_model,
            grasp_config=grasp_config,
            num_grasps=num_grasps,
            topk=topk,
            visualize=False,  # we control viz at pipeline level
        )

        # ------ Stage 2 & 3: Planning + Control ------
        self.motion_planner = MotionPlanner(
            robot=sim.robot,
            sim=sim,
        )

        # Enable obstacle tracking if requested
        if enable_obstacle_avoidance:
            self.motion_planner.enable_obstacle_tracking(
                yolo_model=yolo_model,
                detection_camera=detection_camera,
            )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run(self) -> PipelineResult:
        """Execute the full pick-and-place cycle.

        Returns
        -------
        PipelineResult with all intermediate and final outputs.
        """
        result = PipelineResult()

        # ---- PERCEPTION & AFFORDANCE ----
        logger.info("═══ Stage 1: PERCEPTION ═══")
        grasps, confs, target_pts, scene_pts = self.perception.run(
            visualize=self.visualize
        )
        result.target_points = target_pts
        result.scene_points = scene_pts
        result.grasp_poses = grasps
        result.grasp_confidences = confs

        if grasps.shape[0] == 0:
            logger.error("Perception returned no grasp candidates – aborting.")
            return result

        # ---- PLANNING (IK + RRT-Connect) ----
        logger.info("═══ Stage 2: PLANNING ═══")

        # Try grasps in order of confidence until one succeeds
        import trimesh.transformations as tra
        sorted_indices = np.argsort(-confs)  # highest confidence first
        max_grasp_attempts = min(10, grasps.shape[0])

        traj = []
        for rank, grasp_idx in enumerate(sorted_indices[:max_grasp_attempts]):
            grasp = grasps[grasp_idx]
            logger.info(
                "Trying grasp %d/%d (index=%d, conf=%.4f)...",
                rank + 1, max_grasp_attempts, grasp_idx, confs[grasp_idx],
            )

            target_pos = grasp[:3, 3]
            target_quat_wxyz = tra.quaternion_from_matrix(grasp)
            target_quat = np.array([
                target_quat_wxyz[1], target_quat_wxyz[2],
                target_quat_wxyz[3], target_quat_wxyz[0],
            ])

            traj = self.motion_planner.plan_to_pose(target_pos, target_quat)
            if traj:
                result.best_grasp = grasp
                result.trajectory = traj
                logger.info(
                    "Planning succeeded for grasp %d (conf=%.4f) → %d waypoints",
                    grasp_idx, confs[grasp_idx], len(traj),
                )
                break
            else:
                logger.warning("Grasp %d failed planning, trying next...", grasp_idx)

        if not traj:
            logger.error("Motion planning failed for all %d grasp candidates.", max_grasp_attempts)
            return result

        # Use direct follow for the grasp phase (no MPPI)
        logger.info("Executing trajectory with direct joint control...")
        exec_result = self.motion_planner.follow_trajectory(traj)
        result.execution_result = exec_result
        result.grasp_success = exec_result.get("success", False)

        if result.grasp_success:
            logger.info("✓ Robot reached the grasp pose successfully!")
        else:
            logger.warning(
                "✗ Execution did not fully converge (error=%.4f)",
                exec_result.get("final_error", float("inf")),
            )
            return result

        # ---- GRASP: Close gripper ----
        logger.info("═══ Stage 4: GRASPING ═══")
        self._close_gripper()
        logger.info("✓ Gripper closed.")

        # Weld the object to the end-effector (as requested)
        self.motion_planner.weld_object_to_ee(self.perception.target_object)

        # ---- PLACE: Find basket and move there (MPPI with obstacle avoidance) ----
        logger.info("═══ Stage 5: PLACING ═══")
        
        # Start obstacle tracker for the placing phase
        tracker = self.motion_planner._obstacle_tracker
        if tracker is not None:
            logger.info("Starting obstacle tracker thread for placing stage...")
            tracker.start()

        place_pos = self._find_place_target()
        result.place_target = place_pos

        if place_pos is not None:
            logger.info(
                "Place target: (%.3f, %.3f, %.3f)",
                place_pos[0], place_pos[1], place_pos[2],
            )

            # Plan path to place position (tracker thread still running!)
            place_traj = self.motion_planner.plan_to_position(place_pos)
            result.place_trajectory = place_traj

            if place_traj:
                logger.info("Place trajectory: %d waypoints", len(place_traj))
                place_exec = self.motion_planner.execute(place_traj)
                result.place_result = place_exec
                result.place_success = place_exec.get("success", False)

                if result.place_success:
                    logger.info("✓ Robot reached the place position!")
                    # Open gripper to release
                    self._open_gripper()
                    logger.info("✓ Gripper opened – object placed.")
                else:
                    logger.warning(
                        "✗ Place execution failed (error=%.4f)",
                        place_exec.get("final_error", float("inf")),
                    )
            else:
                logger.error("Planning to place position failed.")
        else:
            logger.warning("Could not determine place target – skipping place.")

        # Stop the obstacle tracker thread (done with MPPI execution)
        if tracker is not None:
            logger.info("Stopping obstacle tracker thread...")
            tracker.stop()

        result.success = result.grasp_success and result.place_success
        return result

    # ------------------------------------------------------------------
    # Gripper control
    # ------------------------------------------------------------------
    def _close_gripper(self, steps: int = 200):
        """Close the gripper and let physics settle."""
        self.sim._set_gripper_opening(0.0)
        for _ in range(steps):
            self.sim.step()

    def _open_gripper(self, steps: int = 200):
        """Open the gripper and let physics settle."""
        default_opening = self.sim.robot_settings.get("default_gripper_opening", 0.04)
        self.sim._set_gripper_opening(float(default_opening))
        for _ in range(steps):
            self.sim.step()

    # ------------------------------------------------------------------
    # Find placing target (basket position)
    # ------------------------------------------------------------------
    def _find_place_target(self) -> Optional[np.ndarray]:
        """Determine the placing target position from the config (basket).

        The basket position is defined in the config YAML and gives us
        the world-frame position to place the object.

        Returns
        -------
        place_pos : (3,) or None
        """
        import yaml
        try:
            # Read basket position from config
            basket_cfg = self.sim.cfg.get("basket", {})
            if basket_cfg:
                basket_pos = np.array(basket_cfg.get("pos", [0.5, 0.52, 0.7]))
                basket_height = float(basket_cfg.get("height", 0.08))

                # Place target is above the basket centre
                place_pos = basket_pos.copy()
                # Table height + basket wall height + some clearance
                table_cfg = self.sim.cfg.get("table", {})
                table_pos = np.array(table_cfg.get("pos", [0.6, 0.0, 0.7]))
                table_size = np.array(table_cfg.get("size", [0.65, 0.95, 0.025]))

                # Place above the basket: table top + basket height + clearance
                place_pos[2] = table_pos[2] + table_size[2] + basket_height + 0.10

                logger.info(
                    "Basket place target: (%.3f, %.3f, %.3f)",
                    place_pos[0], place_pos[1], place_pos[2],
                )
                return place_pos

            # Fallback: try to find basket from MuJoCo model
            import mujoco
            basket_body_id = mujoco.mj_name2id(
                self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "basket_base"
            )
            if basket_body_id >= 0:
                basket_pos = self.sim.data.xpos[basket_body_id].copy()
                basket_pos[2] += 0.15  # above basket
                return basket_pos

            logger.warning("No basket found in config or model.")
            return None

        except Exception as e:
            logger.error("Failed to find place target: %s", e)
            return None

    # ------------------------------------------------------------------
    # Individual stages (for debugging / step-by-step usage)
    # ------------------------------------------------------------------
    def run_perception(
        self, visualize: bool | None = None
    ) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
        """Run perception only."""
        viz = visualize if visualize is not None else self.visualize
        return self.perception.run(visualize=viz)

    def run_planning(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
    ) -> List[np.ndarray]:
        """Run planning only (IK + RRT-Connect)."""
        return self.motion_planner.plan_to_pose(target_pos, target_quat)

    def run_execution(
        self,
        trajectory: List[np.ndarray],
    ) -> dict:
        """Execute a pre-planned trajectory with MPPI."""
        return self.motion_planner.execute(trajectory)