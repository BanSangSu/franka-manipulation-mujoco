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
            visualize=self.visualize,  # we control viz at pipeline level
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
                device = "cuda", # cpu, cuda
                visualize=self.visualize,
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
        grasps, confs, target_pts, scene_pts = self.run_perception(
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


            # manually rotate 90 degrees around local z-axis
            grasp_offset = tra.rotation_matrix(-np.pi/2, [0, 0, 1])
            corrected_grasp = grasp @ grasp_offset

            target_pos = corrected_grasp[:3, 3]
            target_quat_wxyz = tra.quaternion_from_matrix(corrected_grasp)
            # target_pos = grasp[:3, 3]
            # target_quat_wxyz = tra.quaternion_from_matrix(grasp)
            target_quat = np.array([
                target_quat_wxyz[1], target_quat_wxyz[2],
                target_quat_wxyz[3], target_quat_wxyz[0],
            ])

            traj = self.run_planning(target_pos, target_quat)
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

        # For debugging
        # Debug-only weld hook. Keep disabled for normal placing.
        grasp_body_name = self.sim.ids.get("grasp_object", {}).get("body_name", "sample_object")
        self.motion_planner.weld_object_to_ee(grasp_body_name)

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

            # Artificial Potential Field (APF) planner to place position (with obstacle avoidance)
            place_exec = self.motion_planner.execute_apf(place_pos)
            
            # RRT-Connect to place position (no obstacle avoidance, just path planning)
            # place_quat = np.array([0.7071068, 0, 0, 0])
            # rrt_traj = self.run_planning(place_pos, place_quat)
            # place_exec = self.motion_planner.follow_trajectory(rrt_traj)
            

            # MPPI-only placing: no RRT trajectory, just direct goal tracking with avoidance.
            # place_exec = self.motion_planner.execute_to_pose_mppi(place_pos, place_quat)

            result.place_result = place_exec
            result.place_success = place_exec.get("success", False)

            if result.place_success:
                logger.info("✓ Robot reached the place position!")
                
                # For debugging.
                self.motion_planner.release_welded_object()
                
                self._open_gripper()
                for _ in range(250):
                    self.sim.step()
                basket_check = self.sim.check_object_in_basket()
                result.place_success = bool(basket_check.get("in_basket", False))
                result.place_result["basket_check"] = basket_check
                if result.place_success:
                    logger.info("✓ Gripper opened – object placed in basket.")
                else:
                    logger.warning(
                        "Object not inside basket after release: %s",
                        basket_check,
                    )
            else:
                logger.warning(
                    "✗ Place execution failed (error=%.4f)",
                    place_exec.get("final_error", float("inf")),
                )
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
    def _close_gripper(self, steps: int = 1000):
        """Close the gripper and let physics settle."""
        self.sim._set_gripper_opening(0.0)
        for _ in range(steps):
            self.sim.step()

    def _open_gripper(self, steps: int = 1000):
        """Open the gripper and let physics settle."""
        default_opening = self.sim.robot_settings.get("default_gripper_opening", 0.04)
        self.sim._set_gripper_opening(float(default_opening))
        for _ in range(steps):
            self.sim.step()

    # ------------------------------------------------------------------
    # Find placing target (basket position)
    # ------------------------------------------------------------------
    def _find_place_target(self) -> Optional[np.ndarray]:
        """Determine the placing target position from basket segmentation.

        Florence-2 segments the basket directly from the camera images using
        the open-vocabulary prompt ``basket``. We then back-project the masked
        depth pixels into the world frame and estimate a place pose from the
        basket cloud centre and top surface height.

        Returns
        -------
        place_pos : (3,) or None
        """
        try:
            frame = self.perception.camera_mgr.capture_single("static")
            mask = self.perception.florence2.segment(frame.rgb, target_object="basket")
            if mask is None or not np.any(mask):
                logger.warning("Could not segment basket in static camera view.")
                return None

            basket_points, _ = self.perception.pc_builder.fuse(
                [frame],
                [mask.astype(bool)],
                [frame.rgb],
            )
            if basket_points.shape[0] == 0:
                logger.warning("Basket segmentation produced no valid 3D points.")
                return None

            basket_center = np.median(basket_points, axis=0)
            basket_top = float(np.percentile(basket_points[:, 2], 95))

            place_pos = basket_center.copy()
            place_pos[2] = basket_top + 0.10

            logger.info(
                "Basket place target from Florence2 segmentation: "
                "center=(%.3f, %.3f, %.3f), top=%.3f, place=(%.3f, %.3f, %.3f)",
                basket_center[0], basket_center[1], basket_center[2],
                basket_top,
                place_pos[0], place_pos[1], place_pos[2],
            )
            return place_pos

        except Exception as e:
            logger.error("Failed to find place target: %s", e)
            return None

    # ------------------------------------------------------------------
    # Individual stages (for debugging / step-by-step usage)
    # ------------------------------------------------------------------
    def run_perception(
        self, visualize: bool | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
