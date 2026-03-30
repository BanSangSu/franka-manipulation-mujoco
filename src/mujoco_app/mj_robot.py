import mujoco
import numpy as np


class MjRobot:
    """Minimal robot wrapper in MuJoCo Menagerie.

    Provides kinematic control utilities including forward kinematics,
    Jacobian-based inverse kinematics for position and full 6D pose,
    and joint position management with automatic limit enforcement.

    Attributes:
        model: MuJoCo model instance.
        data: MuJoCo data instance.
        ee_id: Integer ID of the end-effector body.
        nq: Number of generalized positions in the model.
        nv: Number of generalized velocities in the model.
        arm_pairs: List of (qpos_index, dof_index) tuples for controllable arm joints.
    """

    def __init__(
        self, model: mujoco.MjModel, data: mujoco.MjData, ee_body_name: str
    ):
        """Initializes the robot wrapper.

        Args:
            model: MuJoCo model containing the robot.
            data: MuJoCo data instance for state storage.
            ee_body_name: Name of the end-effector body in the model.

        Raises:
            ValueError: If the end-effector body is not found in the model.
        """
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name
        )
        if self.ee_id < 0:
            raise ValueError(
                f"End-effector body '{ee_body_name}' not found in model"
            )

        # Select only robot arm joints (exclude free bodies like the ball)
        self.nq = model.nq
        self.nv = model.nv
        self.arm_pairs = []  # list of (qpos_index, dof_index) for controllable joints
        self._joint_lims = {}  # map qpos_index -> (min,max) if limited
        for i in range(1, 8):
            jname = f"joint{i}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            jtype = int(model.jnt_type[jid])
            qadr = int(model.jnt_qposadr[jid])
            dadr = int(model.jnt_dofadr[jid])
            # Only include hinge/slide (1 DoF) joints for IK updates
            if jtype in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                self.arm_pairs.append((qadr, dadr))
                # Cache limits if available (autolimits enabled in menagerie)
                rng = self.model.jnt_range[jid]
                if (
                    np.isfinite(rng[0])
                    and np.isfinite(rng[1])
                    and rng[0] < rng[1]
                ):
                    self._joint_lims[qadr] = (float(rng[0]), float(rng[1]))

    def get_qpos(self):
        """Returns a copy of the current generalized position vector.

        Returns:
            A numpy array containing the current qpos.
        """
        return self.data.qpos.copy()

    def set_qpos(self, q):
        """Sets the generalized position vector and updates kinematics.

        Args:
            q: New qpos vector to apply.
        """
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)

    def set_arm_joint_positions(
        self, joint_positions, clamp: bool = True, sync: bool = True
    ):
        """Sets the controllable arm joints (joint1..joint7) to the provided positions.

        Args:
            joint_positions: Iterable of joint angle values.
            clamp: Whether to enforce joint limits.
            sync: Whether to run forward kinematics after setting positions.

        Raises:
            ValueError: If the number of positions doesn't match the arm's DoF.
        """
        joint_positions = list(joint_positions)
        if len(joint_positions) != len(self.arm_pairs):
            raise ValueError(
                f"Expected {len(self.arm_pairs)} joint values, got {len(joint_positions)}"
            )
        for value, (qidx, _) in zip(joint_positions, self.arm_pairs):
            new_q = float(value)
            if clamp and qidx in self._joint_lims:
                lo, hi = self._joint_lims[qidx]
                new_q = min(max(new_q, lo), hi)
            self.data.qpos[qidx] = new_q
        if sync:
            mujoco.mj_forward(self.model, self.data)

    def get_ee_pose(self):
        """Computes the current end-effector pose.

        Returns:
            A tuple (position, quaternion_xyzw) where position is a 3D numpy array
            and quaternion_xyzw is a 4D numpy array in [x, y, z, w] order.
        """
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.xpos[self.ee_id].copy()
        xmat = self.data.xmat[self.ee_id].reshape(3, 3)
        # convert 3x3 to quaternion (wxyz) using MuJoCo helper
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, xmat.flatten())
        # MuJoCo returns (w,x,y,z)
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
        return pos, quat_xyzw

    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        max_steps: int = 200,
        tol: float = 1e-3,
        pos_tol: float = 0.005,
        step_size: float = 1.0,
    ) -> np.ndarray | None:
        """Solves inverse kinematics for a target end-effector pose.

        Uses damped least squares on the Jacobian with adaptive damping,
        step-size control, and best-effort return.

        Args:
            target_pos: Desired 3D position in world frame.
            target_quat: Desired orientation as (x,y,z,w) quaternion.
            max_steps: Maximum Newton iterations.
            tol: Combined error threshold for exact convergence.
            pos_tol: Position-only tolerance for best-effort return (m).
            step_size: Initial step multiplier (<=1.0 for stability).

        Returns:
            Joint positions as a numpy array, or None if failed.
        """
        q_orig = self.data.qpos.copy()

        q_idxs = [p[0] for p in self.arm_pairs]
        v_idxs = [p[1] for p in self.arm_pairs]
        n_arm = len(v_idxs)

        best_err = float("inf")
        best_pos_err = float("inf")
        best_q = None

        damping = 1e-4
        alpha = step_size
        prev_err_norm = float("inf")

        for it in range(max_steps):
            mujoco.mj_forward(self.model, self.data)

            curr_pos = self.data.xpos[self.ee_id]
            curr_mat = self.data.xmat[self.ee_id].reshape(3, 3)

            # Position error
            pos_err = target_pos - curr_pos
            pos_err_norm = float(np.linalg.norm(pos_err))

            # Orientation error
            rot_err = np.zeros(3)
            if target_quat is not None:
                target_quat_wxyz = np.array([
                    target_quat[3], target_quat[0],
                    target_quat[1], target_quat[2],
                ])
                curr_quat_wxyz = np.zeros(4)
                mujoco.mju_mat2Quat(curr_quat_wxyz, curr_mat.flatten())
                neg_curr_quat = np.zeros(4)
                mujoco.mju_negQuat(neg_curr_quat, curr_quat_wxyz)
                diff_quat = np.zeros(4)
                mujoco.mju_mulQuat(diff_quat, target_quat_wxyz, neg_curr_quat)
                mujoco.mju_quat2Vel(rot_err, diff_quat, 1.0)

            # Weighted error (position weighted more than orientation)
            err = np.zeros(6)
            err[:3] = pos_err
            err[3:] = 0.3 * rot_err  # lower weight for orientation

            err_norm = float(np.linalg.norm(err))

            # Track best solution
            if err_norm < best_err:
                best_err = err_norm
                best_pos_err = pos_err_norm
                best_q = [self.data.qpos[idx] for idx in q_idxs]

            # Check convergence
            if err_norm < tol:
                self.data.qpos[:] = q_orig
                mujoco.mj_forward(self.model, self.data)
                return np.array(best_q)

            # Adaptive damping: increase when error grows, decrease when shrinking
            if err_norm > prev_err_norm:
                damping = min(damping * 5.0, 1.0)
                alpha = max(alpha * 0.5, 0.1)
            else:
                damping = max(damping * 0.5, 1e-6)
                alpha = min(alpha * 1.1, 1.0)
            prev_err_norm = err_norm

            # Jacobian
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)

            # Weight rows to match the error weighting
            J = np.vstack([jacp[:, v_idxs], 0.3 * jacr[:, v_idxs]])

            # Damped least squares
            reg = damping * np.eye(n_arm)
            dq = np.linalg.solve(J.T @ J + reg, J.T @ err)

            # Limit step magnitude
            dq_norm = np.linalg.norm(dq)
            max_dq = 0.2
            if dq_norm > max_dq:
                dq = dq * (max_dq / dq_norm)

            # Apply step
            q_new = self.data.qpos.copy()
            for i, idx in enumerate(q_idxs):
                q_new[idx] += alpha * dq[i]
                if idx in self._joint_lims:
                    lo, hi = self._joint_lims[idx]
                    q_new[idx] = np.clip(q_new[idx], lo, hi)
            self.data.qpos[:] = q_new

        # Best-effort: return if position is close enough
        self.data.qpos[:] = q_orig
        mujoco.mj_forward(self.model, self.data)

        if best_q is not None and best_pos_err < pos_tol:
            return np.array(best_q)

        return None
