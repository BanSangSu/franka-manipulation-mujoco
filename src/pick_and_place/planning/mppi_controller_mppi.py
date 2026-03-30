"""
MPPI Controller – wraps ``pytorch_mppi.MPPI`` for local trajectory tracking
and dynamic-obstacle avoidance on the Franka Panda in MuJoCo.

We define:
  • **dynamics** – batched Euler integration q_{t+1} = clamp(q_t + u_t * dt)
  • **running_cost** – tracking + obstacle-proximity + smoothness + joint-limit
  • **execute_trajectory** – receding-horizon loop that steps the MuJoCo sim.

The obstacle cost now uses camera-derived obstacle data (via YOLO + Kalman)
rather than reading from MuJoCo directly:
  - Obstacle positions & velocities from Kalman filter
  - Per-obstacle radii from YOLO BBox → point-cloud sphere estimation
  - Predicted trajectories over the MPPI horizon
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Dynamics wrapper (batched, pure-PyTorch)
# ──────────────────────────────────────────────────────────────────────
class FrankaDynamics:
    """Simple Euler dynamics: q_{t+1} = clamp(q_t + u_t * dt).

    State  x = (n_joints,)  – joint positions.
    Action u = (n_joints,)  – joint velocities.
    """

    def __init__(
        self,
        dt: float,
        joint_lower: torch.Tensor,
        joint_upper: torch.Tensor,
    ):
        self.dt = dt
        self.q_lo = joint_lower   # (nj,)
        self.q_hi = joint_upper   # (nj,)

    def __call__(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Batched forward: (K×nx), (K×nu) → (K×nx)."""
        next_q = state + action * self.dt
        next_q = torch.clamp(next_q, self.q_lo, self.q_hi)
        return next_q


# ──────────────────────────────────────────────────────────────────────
# Running-cost functor
# ──────────────────────────────────────────────────────────────────────
class TrackingCost:
    """Cost = w_track * ||q - q_ref||²
            + w_obs * obstacle_cost (per-obstacle, using predicted trajectory & per-obstacle radius)
            + w_smooth * ||u||²
            + w_limit * joint_limit_cost

    ``q_ref`` is updated externally each MPPI step via :meth:`set_reference`.
    ``obstacle data`` is updated from camera-based detection each step.
    """

    def __init__(
        self,
        dt: float,
        joint_lower: torch.Tensor,
        joint_upper: torch.Tensor,
        w_tracking: float = 10.0,
        w_obstacle: float = 200.0,
        w_smooth: float = 0.5,
        w_limit: float = 50.0,
        safety_margin: float = 0.08,
        device: str = "cuda",
    ):
        self.dt = dt
        self.q_lo = joint_lower
        self.q_hi = joint_upper
        self.w_tracking = w_tracking
        self.w_obstacle = w_obstacle
        self.w_smooth = w_smooth
        self.w_limit = w_limit
        self.safety_margin = safety_margin
        self.device = device

        # To be set each MPPI step
        self._ref: Optional[torch.Tensor] = None          # (T, nj)
        self._step_idx: int = 0                            # current horizon step
        self._obstacle_traj: Optional[torch.Tensor] = None  # (T, n_obs, 3)
        self._obstacle_radii: Optional[torch.Tensor] = None  # (n_obs,)

        # FK callback (set by MPPIController after init)
        self._fk_fn = None

    # -- external setters ---------------------------------------------------
    def set_reference(self, ref_slice: torch.Tensor):
        """ref_slice : (T, nj) reference joint positions for the horizon."""
        self._ref = ref_slice
        self._step_idx = 0

    def set_obstacles(
        self,
        trajectory: np.ndarray,
        radii: Optional[np.ndarray] = None,
    ):
        """Set obstacle predictions for the current horizon.

        Parameters
        ----------
        trajectory : (T, n_obs, 3) predicted world-frame positions.
        radii : (n_obs,) per-obstacle radii. If None, uses default 0.06m.
        """
        if trajectory is not None and len(trajectory) > 0 and trajectory.shape[1] > 0:
            self._obstacle_traj = torch.tensor(
                trajectory, dtype=torch.float32, device=self.device
            )
            if radii is not None:
                self._obstacle_radii = torch.tensor(
                    radii, dtype=torch.float32, device=self.device
                )
            else:
                n_obs = trajectory.shape[1]
                self._obstacle_radii = torch.full(
                    (n_obs,), 0.06, dtype=torch.float32, device=self.device
                )
        else:
            self._obstacle_traj = None
            self._obstacle_radii = None

    def set_fk_fn(self, fk_fn):
        """Set a callable fk_fn(q_batch) -> ee_pos_batch (K, 3)."""
        self._fk_fn = fk_fn

    # -- cost ---------------------------------------------------------------
    def __call__(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Batched running cost.  state: (K, nx), action: (K, nu) → (K,)."""
        K = state.shape[0]
        cost = torch.zeros(K, device=state.device)

        # 1) Tracking cost
        if self._ref is not None:
            t = min(self._step_idx, self._ref.shape[0] - 1)
            ref_q = self._ref[t].unsqueeze(0)        # (1, nj)
            diff = state - ref_q                       # (K, nj)
            cost = cost + self.w_tracking * (diff * diff).sum(dim=-1)
            self._step_idx += 1

        # 2) Obstacle proximity cost
        #    Uses FK to get EE position, then computes distance to each
        #    obstacle's predicted position at the current horizon step.
        #    The penalty uses per-obstacle radius + safety margin.
        if self._fk_fn is not None and self._obstacle_traj is not None:
            ee_pos = self._fk_fn(state)  # (K, 3)
            t = max(0, min(self._step_idx - 1, self._obstacle_traj.shape[0] - 1))

            obs_at_t = self._obstacle_traj[t]  # (n_obs, 3)
            n_obs = obs_at_t.shape[0]

            for oi in range(n_obs):
                obs_pos = obs_at_t[oi]  # (3,)
                obs_r = self._obstacle_radii[oi] if self._obstacle_radii is not None else 0.06

                d = torch.norm(ee_pos - obs_pos.unsqueeze(0), dim=-1)  # (K,)

                # Safety radius = obstacle radius + safety margin
                safe_r = obs_r + self.safety_margin

                # Quadratic penalty when inside safety radius
                penetration = torch.clamp(safe_r - d, min=0.0)
                cost = cost + self.w_obstacle * penetration * penetration

                # Additional exponential penalty for very close approaches
                # This gives a smoother gradient for the MPPI optimiser
                close_penalty = torch.exp(-10.0 * torch.clamp(d - obs_r, min=0.0))
                cost = cost + self.w_obstacle * 0.1 * close_penalty

        # 3) Smoothness cost (penalise large velocities)
        cost = cost + self.w_smooth * (action * action).sum(dim=-1)

        # 4) Joint-limit cost (soft barrier near limits)
        margin = 0.05
        lower_viol = torch.clamp(self.q_lo + margin - state, min=0.0)
        upper_viol = torch.clamp(state - self.q_hi + margin, min=0.0)
        cost = cost + self.w_limit * (
            (lower_viol * lower_viol).sum(dim=-1)
            + (upper_viol * upper_viol).sum(dim=-1)
        )

        return cost


# ──────────────────────────────────────────────────────────────────────
# Approximate batched FK using MuJoCo (not truly batched – done on CPU)
# ──────────────────────────────────────────────────────────────────────
class BatchedFK:
    """Approximate batched FK: for each of K configs, set joints in MuJoCo
    and read out the EE position.

    This is used only when obstacle_positions are supplied.
    To keep MPPI fast, we subsample the K samples.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot: Any,
        max_fk_samples: int = 64,
    ):
        self.model = model
        self.data = data
        self.robot = robot
        self.q_indices = [p[0] for p in robot.arm_pairs]
        self.ee_id = robot.ee_id
        self.max_fk_samples = max_fk_samples

    def __call__(self, q_batch: torch.Tensor) -> torch.Tensor:
        """q_batch: (K, nj) → ee_pos: (K, 3)."""
        K, nj = q_batch.shape
        device = q_batch.device
        q_np = q_batch.detach().cpu().numpy()

        ee_positions = np.zeros((K, 3))
        q_orig = self.data.qpos.copy()

        # Subsample if K is too large
        if K > self.max_fk_samples:
            indices = np.random.choice(K, self.max_fk_samples, replace=False)
        else:
            indices = np.arange(K)

        for idx in indices:
            for i, qidx in enumerate(self.q_indices):
                self.data.qpos[qidx] = q_np[idx, i]
            mujoco.mj_forward(self.model, self.data)
            ee_positions[idx] = self.data.xpos[self.ee_id].copy()

        # For non-sampled indices, use nearest sampled FK
        if K > self.max_fk_samples:
            sampled_q = q_np[indices]
            for k in range(K):
                if k not in indices:
                    nearest = indices[np.argmin(np.linalg.norm(sampled_q - q_np[k], axis=1))]
                    ee_positions[k] = ee_positions[nearest]

        # Restore
        self.data.qpos[:] = q_orig
        mujoco.mj_forward(self.model, self.data)

        return torch.tensor(ee_positions, dtype=torch.float32, device=device)


# ──────────────────────────────────────────────────────────────────────
# Main controller
# ──────────────────────────────────────────────────────────────────────
class MPPIController:
    """Wraps ``pytorch_mppi.MPPI`` for the Franka Panda.

    Parameters
    ----------
    model, data : MuJoCo model/data.
    robot : MjRobot.
    num_samples : K trajectories to sample.
    horizon : T steps per trajectory.
    dt : control timestep (s).
    lambda_ : MPPI temperature.
    noise_sigma : per-joint velocity noise std-dev (rad/s).
    device : "cpu" or "cuda".
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot: Any,
        num_samples: int = 512,
        horizon: int = 20,
        dt: float = 0.02,
        lambda_: float = 1.0,
        noise_sigma: float = 0.5,
        w_tracking: float = 10.0,
        w_obstacle: float = 200.0,
        w_smooth: float = 0.5,
        w_limit: float = 50.0,
        device: str = "cuda",
    ):
        self.model = model
        self.data = data
        self.robot = robot
        self.dt = dt
        self.device = device

        nj = len(robot.arm_pairs)
        self.n_joints = nj
        self.q_indices = [p[0] for p in robot.arm_pairs]

        # Joint limits
        lower = np.array([
            robot._joint_lims.get(q, (-2 * np.pi, 2 * np.pi))[0]
            for q, _ in robot.arm_pairs
        ])
        upper = np.array([
            robot._joint_lims.get(q, (-2 * np.pi, 2 * np.pi))[1]
            for q, _ in robot.arm_pairs
        ])
        q_lo = torch.tensor(lower, dtype=torch.float32, device=device)
        q_hi = torch.tensor(upper, dtype=torch.float32, device=device)

        # Velocity limits (Franka Panda)
        vel_lim = torch.tensor(
            [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61],
            dtype=torch.float32, device=device,
        )

        # Build dynamics & cost
        self._dynamics = FrankaDynamics(dt, q_lo, q_hi)
        self._cost = TrackingCost(
            dt=dt,
            joint_lower=q_lo,
            joint_upper=q_hi,
            w_tracking=w_tracking,
            w_obstacle=w_obstacle,
            w_smooth=w_smooth,
            w_limit=w_limit,
            device=device,
        )

        # FK for obstacle cost
        self._fk = BatchedFK(model, data, robot, max_fk_samples=64)
        self._cost.set_fk_fn(self._fk)

        # Build MPPI
        from pytorch_mppi import MPPI

        sigma = noise_sigma * torch.eye(nj, dtype=torch.float32, device=device)

        self._mppi = MPPI(
            dynamics=self._dynamics,
            running_cost=self._cost,
            nx=nj,
            noise_sigma=sigma,
            num_samples=num_samples,
            horizon=horizon,
            device=device,
            lambda_=lambda_,
            u_min=-vel_lim,
            u_max=vel_lim,
        )

        self._lower = lower
        self._upper = upper

        # Dynamic Obstacle Tracker (threaded, camera-based)
        self._tracker: Optional[Any] = None

    # ------------------------------------------------------------------
    def set_tracker(self, tracker: Any):
        """Set the threaded ObstacleTracker.
        
        The tracker runs its own background thread for detection.
        MPPI just reads the latest state each step (thread-safe).
        """
        self._tracker = tracker

    def _get_current_q(self) -> np.ndarray:
        return np.array([self.data.qpos[qidx] for qidx in self.q_indices])

    def _capture_tracker_frame(self):
        if self._tracker is None or not getattr(self._tracker, "is_running", False):
            return
        self._tracker.capture_and_submit()

    # ------------------------------------------------------------------
    # Execute trajectory
    # ------------------------------------------------------------------
    def execute_trajectory(
        self,
        sim: Any,
        reference_path: List[np.ndarray],
        steps_per_waypoint: int = 10,
        max_steps: int = 2000,
    ) -> dict:
        """Follow a reference path using MPPI for local control.

        Obstacle data comes from the **threaded** ObstacleTracker which
        runs YOLO detection + Kalman filtering in a background thread.
        MPPI just reads the latest obstacle state each step (thread-safe).

        Parameters
        ----------
        sim : MjSim
        reference_path : list of (n_joints,) joint configs.
        max_steps : absolute step cap.

        Returns
        -------
        dict with 'success', 'steps', 'final_error', 'actual_trajectory'.
        """
        self._mppi.reset()
        ref = np.array(reference_path)
        ref_t = torch.tensor(ref, dtype=torch.float32, device=self.device)

        actual_traj = []
        q_current = self._get_current_q()
        actual_traj.append(q_current.copy())

        total_steps = 0
        ref_idx = 0
        goal = ref[-1]
        T = self._mppi.T

        logger.info("MPPI executing trajectory (%d waypoints, horizon=%d)...", len(ref), T)

        while total_steps < max_steps:
            self._capture_tracker_frame()
            # 1. Build reference slice for the horizon
            ref_slice = torch.zeros(T, self.n_joints, device=self.device)
            for t in range(T):
                idx = min(ref_idx + t + 1, len(ref) - 1)
                ref_slice[t] = ref_t[idx]
            self._cost.set_reference(ref_slice)

            # 2. Read latest obstacle state from threaded tracker
            #    (tracker runs detection in background — this is just a read)
            if self._tracker is not None:
                obs_traj = self._tracker.get_obstacle_predictions(
                    horizon=T, dt=self.dt
                )
                obs_radii = self._tracker.get_obstacle_radii()
                self._cost.set_obstacles(obs_traj, obs_radii)
            else:
                self._cost.set_obstacles(np.empty((T, 0, 3)), None)

            # 3. MPPI command
            state_t = torch.tensor(q_current, dtype=torch.float32, device=self.device)
            action = self._mppi.command(state_t)
            action_np = action.detach().cpu().numpy()

            # 4. Apply: q_next = q + u * dt
            q_target = q_current + action_np * self.dt
            q_target = np.clip(q_target, self._lower, self._upper)

            sim.set_arm_joint_positions(q_target.tolist(), clamp=True, sync=False)

            # 5. Step simulation
            n_substeps = max(1, int(self.dt / self.model.opt.timestep))
            for _ in range(n_substeps):
                sim.step(1)
                self._capture_tracker_frame()

            # 6. Read back
            q_current = self._get_current_q()
            actual_traj.append(q_current.copy())
            total_steps += 1

            # 7. Advance reference index (keep it ahead of the robot)
            dist_to_ref = np.linalg.norm(q_current - ref[min(ref_idx, len(ref) - 1)])
            if dist_to_ref < 0.15 and ref_idx < len(ref) - 1:
                ref_idx += 1

            # 8. Check goal
            goal_error = np.linalg.norm(q_current - goal)
            if goal_error < 0.05:
                logger.info("MPPI ✓ reached goal! steps=%d error=%.4f", total_steps, goal_error)
                return {
                    "success": True,
                    "steps": total_steps,
                    "final_error": goal_error,
                    "actual_trajectory": actual_traj,
                }

            if total_steps % 100 == 0:
                logger.info(
                    "MPPI step %d: ref=%d/%d goal_err=%.4f",
                    total_steps, ref_idx, len(ref), goal_error,
                )

        final_error = np.linalg.norm(q_current - goal)
        success = final_error < 0.1
        logger.info("MPPI done: steps=%d error=%.4f success=%s", total_steps, final_error, success)
        return {
            "success": success,
            "steps": total_steps,
            "final_error": final_error,
            "actual_trajectory": actual_traj,
        }
