"""
MotionPlanner – orchestrates IK, RRT-Connect (global path), and MPPI (local control).

Flow:
    1. Solve IK for pre-grasp and grasp configurations.
    2. RRT-Connect generates a collision-free joint-space path.
    3. Path is smoothed and resampled.
    4. MPPI controller tracks the path while avoiding dynamic obstacles
       detected in real-time via camera-based YOLO + Kalman.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MotionPlanner:
    """Generates and executes paths from current robot state to target poses.

    Parameters
    ----------
    robot : MjRobot
        The robot wrapper with IK solver and kinematic utilities.
    sim : MjSim
        The simulation for stepping and collision checking.
    rrt_max_iterations : int
        Maximum RRT-Connect iterations.
    rrt_max_time : float
        Maximum RRT-Connect planning time (seconds).
    rrt_step_size : float
        RRT extension step size in joint space (radians).
    mppi_samples : int
        Number of MPPI trajectory samples.
    mppi_horizon : int
        MPPI planning horizon.
    mppi_dt : float
        MPPI control timestep.
    """

    def __init__(
        self,
        robot: Any,
        sim: Any,
        rrt_max_iterations: int = 5000,
        rrt_max_time: float = 30.0,
        rrt_step_size: float = 0.15,
        mppi_samples: int = 256,
        mppi_horizon: int = 20,
        mppi_dt: float = 0.02,
    ):
        self.robot = robot
        self.sim = sim
        self.model = sim.model
        self.data = sim.data

        # RRT settings
        self.rrt_max_iterations = rrt_max_iterations
        self.rrt_max_time = rrt_max_time
        self.rrt_step_size = rrt_step_size

        # MPPI settings
        self.mppi_samples = mppi_samples
        self.mppi_horizon = mppi_horizon
        self.mppi_dt = mppi_dt

        # Lazy-init planner components
        self._collision_checker = None
        self._js_config = None
        self._mppi = None
        self._obstacle_tracker = None

        # Weld info
        self._welded_object: Optional[str] = None
        self._weld_rel_pos: Optional[np.ndarray] = None
        self._weld_rel_mat: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------
    def _init_rrt(self):
        """Initialize RRT-Connect components."""
        if self._collision_checker is not None:
            return
        from .rrt_connect import JointSpaceConfig, MuJoCoCollisionChecker

        self._js_config = JointSpaceConfig.from_robot(self.robot)
        self._js_config.step_size = self.rrt_step_size
        self._collision_checker = MuJoCoCollisionChecker(
            self.model, self.data, self.robot,
            self_collision=True,
            env_collision=False,  # We only avoid obstacles via MPPI
        )
        logger.info("RRT-Connect initialized (step_size=%.3f)", self.rrt_step_size)

    def _init_mppi(self):
        """Initialize MPPI controller."""
        if self._mppi is not None:
            return
        from .mppi_controller import MPPIController

        self._mppi = MPPIController(
            model=self.model,
            data=self.data,
            robot=self.robot,
            num_samples=self.mppi_samples,
            horizon=self.mppi_horizon,
            dt=self.mppi_dt,
        )
        logger.info("MPPI controller initialized (K=%d, T=%d)", self.mppi_samples, self.mppi_horizon)

    # ------------------------------------------------------------------
    # Obstacle Tracking (threaded YOLO26 + Kalman)
    # ------------------------------------------------------------------
    def enable_obstacle_tracking(
        self,
        yolo_model: str = "yolo26n.pt",
        detection_camera: str = "static",
        device: str = "cpu", # cpu, cuda
    ):
        """Enable real-time camera-based obstacle tracking in a background thread.

        A background thread continuously:
          1. Captures frames from the MuJoCo camera
          2. Runs YOLO26 .track(persist=True) for detection + tracking IDs
          3. Maps BBox → depth → 3D sphere (centre + radius)
          4. Updates per-obstacle Kalman filters

        MPPI reads the latest state via thread-safe accessors.

        Parameters
        ----------
        yolo_model : str
            YOLO model path (e.g. "yolo26n.pt" or "yolo26n-seg.pt").
        detection_camera : str
            Camera name to use for obstacle detection.
        device : str
            Device for YOLO inference.
        """
        self._init_mppi()
        from ..perception.obstacle_tracker import ObstacleTracker

        self._obstacle_tracker = ObstacleTracker(
            sim=self.sim,
            yolo_model=yolo_model,
            detection_camera=detection_camera,
            dt=self.mppi_dt,
            device=device,
        )
        self._mppi.set_tracker(self._obstacle_tracker)

        logger.info(
            "Obstacle tracking prepared (YOLO: %s, camera: %s) — thread starts on execute()",
            yolo_model, detection_camera,
        )

    # ------------------------------------------------------------------
    # IK helpers
    # ------------------------------------------------------------------
    def _get_current_arm_q(self) -> np.ndarray:
        """Get current arm joint positions."""
        return np.array([self.data.qpos[qidx] for qidx, _ in self.robot.arm_pairs])

    def _solve_ik(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        max_attempts: int = 20,
    ) -> Optional[np.ndarray]:
        """Solve IK with multiple random restarts.

        Parameters
        ----------
        target_pos : (3,) target position.
        target_quat : (4,) target quaternion (xyzw).
        max_attempts : int – number of IK attempts with random seeds.

        Returns
        -------
        q : (n_joints,) joint positions, or None if all attempts fail.
        """
        import mujoco

        # First try from current configuration
        result = self.robot.solve_ik(
            target_pos, target_quat, max_steps=500, tol=1e-3, pos_tol=0.01
        )
        if result is not None:
            return result

        # Try with random starting configurations
        q_orig = self.data.qpos.copy()
        for attempt in range(max_attempts):
            # Random starting configuration
            for qidx, _ in self.robot.arm_pairs:
                if qidx in self.robot._joint_lims:
                    lo, hi = self.robot._joint_lims[qidx]
                    self.data.qpos[qidx] = np.random.uniform(lo, hi)
            mujoco.mj_forward(self.model, self.data)

            result = self.robot.solve_ik(
                target_pos, target_quat, max_steps=500, tol=1e-3, pos_tol=0.01
            )
            if result is not None:
                self.data.qpos[:] = q_orig
                mujoco.mj_forward(self.model, self.data)
                logger.info("IK solved on attempt %d (with orientation)", attempt + 1)
                return result

        # Fallback: try position-only IK (ignore orientation)
        logger.info("Full 6-DoF IK failed, trying position-only fallback...")
        self.data.qpos[:] = q_orig
        mujoco.mj_forward(self.model, self.data)
        result = self.robot.solve_ik(
            target_pos, None, max_steps=500, tol=1e-3, pos_tol=0.01
        )
        if result is not None:
            logger.info("Position-only IK succeeded (orientation not matched)")
            return result

        for attempt in range(max_attempts):
            for qidx, _ in self.robot.arm_pairs:
                if qidx in self.robot._joint_lims:
                    lo, hi = self.robot._joint_lims[qidx]
                    self.data.qpos[qidx] = np.random.uniform(lo, hi)
            mujoco.mj_forward(self.model, self.data)
            result = self.robot.solve_ik(
                target_pos, None, max_steps=500, tol=1e-3, pos_tol=0.01
            )
            if result is not None:
                self.data.qpos[:] = q_orig
                mujoco.mj_forward(self.model, self.data)
                logger.info("Position-only IK solved on attempt %d", attempt + 1)
                return result

        # Restore state
        self.data.qpos[:] = q_orig
        mujoco.mj_forward(self.model, self.data)
        return None

    # ------------------------------------------------------------------
    # Planning: IK → RRT-Connect
    # ------------------------------------------------------------------
    def plan_to_pose(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        approach_offset: float = 0.08,
    ) -> List[np.ndarray]:
        """Plan a collision-free joint-space path to a target Cartesian pose.

        Steps:
        1. Solve IK for pre-grasp (retracted along approach axis).
        2. Solve IK for the final grasp pose.
        3. RRT-Connect: current → pre-grasp → grasp.
        4. Smooth and resample.

        Parameters
        ----------
        target_pos : (3,) – target position in world frame.
        target_quat : (4,) – target orientation (xyzw).
        approach_offset : float – distance (m) to offset for pre-grasp along approach.

        Returns
        -------
        trajectory : list[ndarray] – joint-space path (may be empty on failure).
        """
        self._init_rrt()

        # Current arm state
        start_q = self._get_current_arm_q()

        # 1. Compute pre-grasp pose
        import trimesh.transformations as tra
        rot_mat = tra.quaternion_matrix(
            [target_quat[3], target_quat[0], target_quat[1], target_quat[2]]
        )[:3, :3]
        approach_vec = rot_mat[:, 2]  # local Z axis
        pre_grasp_pos = target_pos - approach_vec * approach_offset

        # 2. Solve IK
        logger.info("Solving IK for pre-grasp pose...")
        pre_grasp_q = self._solve_ik(pre_grasp_pos, target_quat)
        if pre_grasp_q is None:
            logger.error("IK failed for pre-grasp pose")
            return []

        logger.info("Solving IK for grasp pose...")
        grasp_q = self._solve_ik(target_pos, target_quat)
        if grasp_q is None:
            logger.error("IK failed for grasp pose")
            return []

        logger.info("IK solutions found. Starting RRT-Connect planning...")

        # 3. RRT-Connect: start → pre-grasp
        from .rrt_connect import rrt_connect, smooth_path, resample_path

        path_to_pregrasp = rrt_connect(
            q_start=start_q,
            q_goal=pre_grasp_q,
            collision_fn=self._collision_checker,
            js=self._js_config,
            max_iterations=self.rrt_max_iterations,
            max_time=self.rrt_max_time,
            step_size=self.rrt_step_size,
        )

        if path_to_pregrasp is None:
            logger.error("RRT-Connect failed: start → pre-grasp")
            # Fallback: simple linear interpolation
            logger.info("Falling back to linear interpolation...")
            path_to_pregrasp = self._linear_interpolate(start_q, pre_grasp_q, n_steps=30)

        # Smooth the path
        path_to_pregrasp = smooth_path(path_to_pregrasp, self._collision_checker, max_iterations=100)

        # 4. RRT-Connect: pre-grasp → grasp (shorter, usually straight line)
        path_approach = rrt_connect(
            q_start=pre_grasp_q,
            q_goal=grasp_q,
            collision_fn=self._collision_checker,
            js=self._js_config,
            max_iterations=1000,
            max_time=5.0,
            step_size=self.rrt_step_size * 0.5,  # finer steps for approach
        )

        if path_approach is None:
            logger.warning("RRT failed for approach phase, using linear interpolation")
            path_approach = self._linear_interpolate(pre_grasp_q, grasp_q, n_steps=15)

        # 5. Combine and resample
        full_path = path_to_pregrasp + path_approach[1:]  # avoid duplication at pre-grasp
        full_path = resample_path(full_path, resolution=0.05)

        logger.info("Planning complete: %d waypoints in trajectory", len(full_path))
        return full_path

    def plan_to_joint_config(
        self,
        target_q: np.ndarray,
    ) -> List[np.ndarray]:
        """Plan a path from current config to a target joint config.

        Parameters
        ----------
        target_q : (n_joints,) target joint positions.

        Returns
        -------
        trajectory : list[ndarray] – joint-space path.
        """
        self._init_rrt()
        start_q = self._get_current_arm_q()

        from .rrt_connect import rrt_connect, smooth_path, resample_path

        path = rrt_connect(
            q_start=start_q,
            q_goal=target_q,
            collision_fn=self._collision_checker,
            js=self._js_config,
            max_iterations=self.rrt_max_iterations,
            max_time=self.rrt_max_time,
            step_size=self.rrt_step_size,
        )

        if path is None:
            logger.warning("RRT-Connect failed, using linear interpolation")
            path = self._linear_interpolate(start_q, target_q, n_steps=50)

        path = smooth_path(path, self._collision_checker,  max_iterations=100)
        path = resample_path(path, resolution=0.05)
        return path

    def plan_to_position(
        self,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Plan a path to a Cartesian position (orientation optional).

        Parameters
        ----------
        target_pos : (3,) target position in world frame.
        target_quat : (4,) target quaternion (xyzw) or None for position-only.

        Returns
        -------
        trajectory : list[ndarray] – joint-space path.
        """
        self._init_rrt()
        start_q = self._get_current_arm_q()

        # Solve IK
        target_q = self._solve_ik(target_pos, target_quat)
        if target_q is None:
            logger.error("IK failed for target position")
            return []

        from .rrt_connect import rrt_connect, smooth_path, resample_path

        path = rrt_connect(
            q_start=start_q,
            q_goal=target_q,
            collision_fn=self._collision_checker,
            js=self._js_config,
            max_iterations=self.rrt_max_iterations,
            max_time=self.rrt_max_time,
            step_size=self.rrt_step_size,
        )

        if path is None:
            logger.warning("RRT-Connect failed, using linear interpolation")
            path = self._linear_interpolate(start_q, target_q, n_steps=50)

        path = smooth_path(path, self._collision_checker, max_iterations=100)
        path = resample_path(path, resolution=0.05)
        return path

    # ------------------------------------------------------------------
    # Execution: MPPI tracking
    # ------------------------------------------------------------------
    def execute(
        self,
        trajectory: List[np.ndarray],
        max_steps: int = 2000,
    ) -> dict:
        """Execute a planned trajectory using MPPI for tracking + obstacle avoidance.

        Parameters
        ----------
        trajectory : list[ndarray] – reference joint-space path from `plan_to_pose`.
        max_steps : int – max simulation steps.

        Returns
        -------
        result : dict with 'success', 'steps', 'final_error', 'actual_trajectory'.
        """
        if not trajectory:
            logger.error("Cannot execute empty trajectory")
            return {"success": False, "steps": 0, "final_error": float("inf"), "actual_trajectory": []}

        self._init_mppi()
        return self._mppi.execute_trajectory(
            sim=self.sim,
            reference_path=trajectory,
            max_steps=max_steps,
            step_callback=self.weld_step,
        )

    def follow_trajectory(
        self,
        trajectory: List[np.ndarray],
    ) -> dict:
        """Follow a trajectory directly by setting joint positions.

        This is a simple controller used before the grasp (no obstacle avoidance).
        It advances through the trajectory waypoints at a fixed rate.

        Parameters
        ----------
        trajectory : list[ndarray]
            Reference joint-space path.

        Returns
        -------
        result : dict with 'success', 'final_error'.
        """
        if not trajectory:
            return {"success": False, "final_error": float("inf")}

        logger.info("Directly following trajectory (%d waypoints)...", len(trajectory))

        actual_traj = []
        for q in trajectory:
            # Set joint positions directly
            self.sim.set_arm_joint_positions(q.tolist(), clamp=True, sync=False)

            # Let physics settle briefly for each waypoint
            # (Higher quality simulation would interpolate, but this follows waypoints)
            self.sim.step(10)  # 10 steps * 0.002s = 20ms per waypoint

            q_current = self._get_current_arm_q()
            actual_traj.append(q_current)

        final_q = self._get_current_arm_q()
        goal_error = np.linalg.norm(final_q - trajectory[-1])

        success = goal_error < 0.05
        logger.info("Direct follow complete: success=%s, error=%.4f", success, goal_error)

        return {
            "success": success,
            "final_error": goal_error,
            "actual_trajectory": actual_traj,
        }

    def weld_object_to_ee(self, object_name: str):
        """Welds an object to the robot's end-effector.

        This is a 'metaphysical' weld for simulation stability:
        1. Finds the object body ID.
        2. Disables its gravity (gravcomp=1).
        3. Records the relative transformation from EE to object.
        """
        import mujoco
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if obj_id < 0:
            logger.error("Weld failed: body '%s' not found", object_name)
            return

        # Disable gravity for the object to make it weightless in the hand
        # This is a bit of a hack but very effective for MuJoCo grasping
        self.model.body_gravcomp[obj_id] = 1.0

        # Compute relative transform from EE to object
        ee_pos = self.data.xpos[self.robot.ee_id]
        ee_mat = self.data.xmat[self.robot.ee_id].reshape(3, 3)
        obj_pos = self.data.xpos[obj_id]
        obj_mat = self.data.xmat[obj_id].reshape(3, 3)

        # R_ee^T * (t_obj - t_ee)
        self._welded_object = object_name
        self._weld_rel_pos = ee_mat.T @ (obj_pos - ee_pos)
        self._weld_rel_mat = ee_mat.T @ obj_mat

        logger.info("Object '%s' welded to end-effector.", object_name)

    def weld_step(self):
        """Update the pose of the welded object to match the EE.
        
        This should be called every simulation step when an object is welded.
        """
        if self._welded_object is None or self._weld_rel_pos is None:
            return

        import mujoco
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self._welded_object)
        if obj_id < 0:
            return

        # Find the free joint for this body
        joint_adr = self.model.body_jntadr[obj_id]
        if joint_adr < 0 or self.model.jnt_type[joint_adr] != mujoco.mjtJoint.mjJNT_FREE:
            # Fallback for complex models: check all joints
            for j in range(self.model.njnt):
                if self.model.jnt_bodyid[j] == obj_id and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    joint_adr = j
                    break
        
        if joint_adr < 0:
            return

        qadr = self.model.jnt_qposadr[joint_adr]

        # New object pose = EE pose * R_rel + t_rel
        ee_pos = self.data.xpos[self.robot.ee_id]
        ee_mat = self.data.xmat[self.robot.ee_id].reshape(3, 3)

        new_obj_pos = ee_pos + ee_mat @ self._weld_rel_pos
        new_obj_mat = ee_mat @ self._weld_rel_mat

        # Convert mat to quat (wxyz)
        new_obj_quat = np.zeros(4)
        mujoco.mju_mat2Quat(new_obj_quat, new_obj_mat.flatten())

        self.data.qpos[qadr : qadr + 3] = new_obj_pos
        self.data.qpos[qadr + 3 : qadr + 7] = new_obj_quat

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _linear_interpolate(
        q_start: np.ndarray, q_goal: np.ndarray, n_steps: int = 50
    ) -> List[np.ndarray]:
        """Simple linear interpolation in joint space."""
        path = []
        for i in range(n_steps + 1):
            alpha = i / n_steps
            path.append(q_start * (1 - alpha) + q_goal * alpha)
        return path