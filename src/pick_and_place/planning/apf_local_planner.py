"""Artificial potential field local planner for workspace obstacle avoidance."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

import mujoco
import numpy as np

logger = logging.getLogger(__name__)


class APFLocalPlanner:
    """Workspace APF controller using tracked spherical obstacles."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot: Any,
        tracker: Optional[Any] = None,
        dt: float = 0.02,
        carried_body_name: Optional[str] = None,
        k_att: float = 4.0,
        k_rep: float = 0.10,
        d0: float = 0.20,
        max_rep_force: float = 0.20,
        max_step: float = 0.03,
        table_clearance: float = 0.07,
        goal_tolerance: float = 0.03,
        close_goal_radius: float = 0.08,
        stall_window: int = 20,
        stall_tolerance: float = 0.002,
    ):
        self.model = model
        self.data = data
        self.robot = robot
        self.tracker = tracker
        self.dt = dt
        self.carried_body_name = carried_body_name
        self.k_att = k_att
        self.k_rep = k_rep
        self.d0 = d0
        self.max_rep_force = max_rep_force
        self.max_step = max_step
        self.table_clearance = table_clearance
        self.goal_tolerance = goal_tolerance
        self.close_goal_radius = close_goal_radius
        self.stall_window = stall_window
        self.stall_tolerance = stall_tolerance

    def attractive_force(self, position: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return -self.k_att * (position - goal)

    def repulsive_force(
        self,
        position: np.ndarray,
        obstacles: List[dict],
    ) -> np.ndarray:
        force = np.zeros(3, dtype=float)
        for obs in obstacles:
            center = np.asarray(obs["position"], dtype=float)
            radius = float(obs.get("radius", 0.0))
            diff = position - center
            dist = np.linalg.norm(diff)
            signed_dist = dist - radius
            if signed_dist <= 1e-6:
                direction = diff / max(dist, 1e-6)
                force += direction * self.max_rep_force
                continue
            if signed_dist < self.d0:
                direction = diff / dist
                rep = self.k_rep * ((1.0 / signed_dist) - (1.0 / self.d0)) * (1.0 / (signed_dist ** 2)) * direction
                rep_norm = np.linalg.norm(rep)
                if rep_norm > self.max_rep_force:
                    rep = rep / rep_norm * self.max_rep_force
                force += rep
        return force

    def total_force(
        self,
        position: np.ndarray,
        goal: np.ndarray,
        obstacles: List[dict],
    ) -> np.ndarray:
        return self.attractive_force(position, goal) + self.repulsive_force(position, obstacles)

    def _current_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.robot.get_ee_pose()

    def _get_dynamic_obstacles(self) -> List[dict]:
        if self.tracker is None:
            return []
        return list(self.tracker.get_obstacle_states())

    def _capture_tracker_frame(self):
        if self.tracker is None or not getattr(self.tracker, "is_running", False):
            return
        self.tracker.capture_and_submit()

    def _plan_joint_trajectory(
        self,
        sim: Any,
        goal_pos: np.ndarray,
        goal_quat: Optional[np.ndarray] = None,
        max_steps: int = 800,
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate an APF-guided joint-space trajectory without executing dynamics."""
        qpos_orig = self.data.qpos.copy()
        qvel_orig = self.data.qvel.copy()

        joint_traj: List[np.ndarray] = [np.array(
            [self.data.qpos[qidx] for qidx, _ in self.robot.arm_pairs],
            dtype=float,
        )]
        ee_traj: List[np.ndarray] = []
        goal_error_history: List[float] = []

        sim_cfg = getattr(sim, "cfg", {})
        table_top = float(sim_cfg.get("table", {}).get("pos", [0.6, 0.0, 0.7])[2])
        table_half_h = float(sim_cfg.get("table", {}).get("size", [0.65, 0.95, 0.025])[2])
        z_min = table_top + table_half_h + self.table_clearance

        success = False
        try:
            for step in range(max_steps):
                self._capture_tracker_frame()
                ee_pos, ee_quat = self._current_ee_pose()
                ee_traj.append(ee_pos.copy())

                goal_error = np.linalg.norm(ee_pos - goal_pos)
                goal_error_history.append(goal_error)
                if goal_error < self.goal_tolerance:
                    logger.info(
                        "APF planner generated trajectory in %d steps (err=%.4f)",
                        step,
                        goal_error,
                    )
                    success = True
                    break

                obstacles = self._get_dynamic_obstacles()
                force = self.total_force(ee_pos, goal_pos, obstacles)

                if len(goal_error_history) > self.stall_window:
                    progress = goal_error_history[-self.stall_window] - goal_error_history[-1]
                    if progress < self.stall_tolerance:
                        force = goal_pos - ee_pos

                force_norm = np.linalg.norm(force)
                if force_norm < 1e-8:
                    next_pos = goal_pos.copy()
                else:
                    step_vec = force * self.dt
                    step_norm = np.linalg.norm(step_vec)
                    if step_norm > self.max_step:
                        step_vec = step_vec / step_norm * self.max_step
                    next_pos = ee_pos + step_vec

                next_pos[2] = max(next_pos[2], z_min)
                if goal_error < self.close_goal_radius:
                    next_pos = goal_pos.copy()
                    next_pos[2] = max(next_pos[2], z_min)

                q_next = self.robot.solve_ik(
                    next_pos,
                    goal_quat if goal_quat is not None else ee_quat,
                    max_steps=200,
                    tol=1e-3,
                    pos_tol=0.01,
                )
                if q_next is None:
                    q_next = self.robot.solve_ik(
                        next_pos,
                        None,
                        max_steps=200,
                        tol=1e-3,
                        pos_tol=0.01,
                    )
                if q_next is None:
                    logger.warning("APF planner could not solve IK at step %d", step)
                    break

                self.robot.set_arm_joint_positions(q_next.tolist(), clamp=True, sync=True)
                joint_traj.append(np.asarray(q_next, dtype=float).copy())

                if hasattr(sim, "check_robot_obstacle_collision") and sim.check_robot_obstacle_collision():
                    logger.warning("APF planner generated a colliding configuration at step %d", step)
                    joint_traj.pop()
                    break

            if not success:
                final_pos, _ = self._current_ee_pose()
                final_error = np.linalg.norm(final_pos - goal_pos)
                logger.warning("APF planner stopped with error %.4f", final_error)
        finally:
            self.data.qpos[:] = qpos_orig
            self.data.qvel[:] = qvel_orig
            mujoco.mj_forward(self.model, self.data)

        return joint_traj, ee_traj

    def plan(
        self,
        sim: Any,
        goal_pos: np.ndarray,
        goal_quat: Optional[np.ndarray] = None,
        max_steps: int = 800,
    ) -> List[np.ndarray]:
        """Generate an APF-guided joint-space trajectory."""
        joint_traj, _ = self._plan_joint_trajectory(
            sim=sim,
            goal_pos=goal_pos,
            goal_quat=goal_quat,
            max_steps=max_steps,
        )
        if len(joint_traj) <= 1:
            return []
        return joint_traj

    def execute(
        self,
        sim: Any,
        goal_pos: np.ndarray,
        goal_quat: Optional[np.ndarray] = None,
        max_steps: int = 800,
        step_callback: Optional[Callable[[], None]] = None,
    ) -> dict:
        joint_traj, actual_traj = self._plan_joint_trajectory(
            sim=sim,
            goal_pos=goal_pos,
            goal_quat=goal_quat,
            max_steps=max_steps,
        )
        if len(joint_traj) <= 1:
            final_pos, _ = self._current_ee_pose()
            final_error = np.linalg.norm(final_pos - goal_pos)
            logger.warning("APF local planner stopped with error %.4f", final_error)
            return {
                "success": final_error < max(self.goal_tolerance, 0.05),
                "steps": 0,
                "final_error": final_error,
                "actual_trajectory": actual_traj,
            }

        for q_next in joint_traj[1:]:
            sim.set_arm_joint_positions(q_next.tolist(), clamp=True, sync=False)

            n_substeps = max(1, int(self.dt / self.model.opt.timestep))
            for _ in range(n_substeps):
                sim.step(1)

                if sim.check_robot_obstacle_collision():
                    logging.warning("Collision detected")
                    sim.reset()

                if step_callback is not None:
                    step_callback()
                self._capture_tracker_frame()

        final_pos, _ = self._current_ee_pose()
        final_error = np.linalg.norm(final_pos - goal_pos)
        if final_error < max(self.goal_tolerance, 0.05):
            logger.info("APF local planner reached goal with error %.4f", final_error)
        else:
            logger.warning("APF local planner stopped with error %.4f", final_error)
        return {
            "success": final_error < max(self.goal_tolerance, 0.05),
            "steps": len(joint_traj) - 1,
            "final_error": final_error,
            "actual_trajectory": actual_traj,
        }
