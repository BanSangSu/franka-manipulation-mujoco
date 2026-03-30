"""MuJoCo simulation wrapper used by the project."""

import importlib
import os
import random
from typing import Any, Dict, Iterable, Tuple

import mujoco
import numpy as np

from mujoco_app.mj_robot import MjRobot
from mujoco_app.scene_builder import build_scene

GRAY_LABEL = 44


class MjSim:
    """Wrapper for MuJoCo simulation with rendering and robot control utilities.

    Manages the simulation lifecycle, coordinates rendering from multiple cameras,
    provides high-level grasp and collision APIs, and animates moving obstacles.

    Attributes:
        cfg: Configuration dictionary passed at initialization.
        model: MuJoCo model instance.
        data: MuJoCo data instance containing simulation state.
        ids: Dictionary of named scene elements (cameras, bodies, etc.).
        renderer: MuJoCo Renderer instance for off-screen rendering.
        robot: MjRobot instance wrapping the arm kinematics.
        robot_settings: Robot-specific configuration from cfg.
        extra_cameras: Dict mapping camera names to their MuJoCo IDs.
        extra_specs: Dict mapping camera names to their specification dicts.
    """

    def __init__(self, cfg: dict):
        """Initializes the simulation from a configuration dictionary.

        Args:
            cfg: Configuration dict with keys 'mujoco', 'robot_settings', etc.
        """
        self.cfg = cfg
        self.sim_seed = cfg.get("mujoco", {}).get("seed", 42)
        self._set_seed(self.sim_seed)
        artifacts = build_scene(cfg)
        self.obstacle_toggle = cfg.get("mujoco", {}).get(
            "obstacle_toggle", False
        )
        self.model: mujoco.MjModel = artifacts.model
        self.data: mujoco.MjData = artifacts.data
        self.ids: Dict[str, Dict[str, Any]] = artifacts.ids

        cam_cfg = dict(cfg.get("mujoco", {}).get("camera", {}))
        self._primary_camera = cam_cfg.get("name", "static")
        width = int(cam_cfg.get("width", 640))
        height = int(cam_cfg.get("height", 480))

        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        robot_settings = dict(cfg.get("robot_settings", {}))
        ee_body_name = robot_settings.get(
            "ee_body_name", cfg.get("mujoco", {}).get("ee_body_name", "hand")
        )
        self.robot = MjRobot(
            self.model,
            self.data,
            ee_body_name=ee_body_name,
        )
        self.robot_settings = dict(cfg.get("robot_settings", {}))
        self._finger_indices = self._detect_gripper_actuators()
        self._arm_actuator_indices = self._detect_arm_actuators()
        self.jitter = np.zeros(3)
        self.extra_cameras: Dict[str, int] = dict(
            self.ids.get("extra_cameras", {})
        )
        self.extra_specs: Dict[str, Dict[str, Any]] = dict(
            self.ids.get("extra_camera_specs", {})
        )

        self._viewer = None
        if bool(cfg.get("mujoco", {}).get("gui", False)):
            self._viewer = self._maybe_launch_viewer()
        self._sim_time: float = 0.0

    def _set_seed(self, seed: int):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

    # Lifecycle
    def reset(self) -> None:
        """Resets the simulation state to initial conditions.

        Resets MuJoCo data, applies default joint positions and gripper opening
        from robot_settings, and syncs the viewer if one is active.
        """
        self.jitter = np.zeros(3)
        mujoco.mj_resetData(self.model, self.data)
        default_joints = self.robot_settings.get("default_joint_positions")
        if default_joints is not None:
            self.set_arm_joint_positions(default_joints, clamp=True, sync=True)
        default_opening = self.robot_settings.get("default_gripper_opening")
        if default_opening is not None:
            self._set_gripper_opening(float(default_opening))
        if self._viewer is not None:
            self._viewer.sync()

    def step(self, n: int = 1) -> None:
        """Advances the simulation by n timesteps.

        Args:
            n: Number of simulation steps to execute.
        """
        for _ in range(n):
            if self.obstacle_toggle:
                self._animate_moving_obstacles()
            mujoco.mj_step(self.model, self.data)
            self._sim_time += float(self.model.opt.timestep)
            if self._viewer is not None:
                self._viewer.sync()

    def close(self) -> None:
        """Closes the viewer window if one was launched."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    #
    def get_camera_pose(self, id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the position and orientation of a camera.

        Args:
            id: Index of the camera.

        Returns:
            A tuple containing the position and orientation of the camera.
        """
        cam_pos = self.data.cam_xpos[id]
        cam_rot = self.data.cam_xmat[id].reshape(3, 3)
        return cam_pos, cam_rot

    def get_intrinsic_mat(
        self, fov: float, width: int, height: int
    ) -> np.ndarray:
        """Returns the intrinsic matrix of a camera.

        Args:
            fov: Field of view of the camera in degrees.
            width: Width of the image in pixels.
            height: Height of the image in pixels.

        Returns:
            The intrinsic matrix of the camera.
        """
        theta = np.deg2rad(fov)
        cx = width / 2.0
        cy = height / 2.0
        fx = -(1.0 / np.tan(theta / 2)) * cy
        fy = (1.0 / np.tan(theta / 2)) * cy
        # don't use different x and y focals if you want non skewed results
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def get_extrinsic_mat(
        self, cam_pos: np.ndarray, cam_rot: np.ndarray
    ) -> np.ndarray:
        """Returns the extrinsic matrix of a camera.

        Args:
            cam_pos: Position of the camera in world coordinates.
            cam_rot: Rotation of the camera in world coordinates.

        Returns:
            The extrinsic matrix of the camera.
        """
        R_cw = cam_rot.T
        t_cw = -R_cw @ cam_pos

        # Construct the final 4x4 Extrinsic Matrix E_wc
        E_wc = np.eye(4)
        E_wc[:3, :3] = R_cw
        E_wc[:3, 3] = t_cw.flatten()

        return E_wc

    def get_static_renders(
        self, width: int, height: int, near: float, far: float, fovy: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Renders RGB, depth, intrinsic and extrinsic matrices
           from the primary camera.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            near: Near clipping plane distance.
            far: Far clipping plane distance.
            fovy: Vertical field-of-view in degrees.

        Returns:
            A tuple (rgb, depth, intrinsic, extrinsic).
        """
        cam_id = int(self.ids.get("cam_id", 0))
        rgb, depth = self._render_color_depth(cam_id, width, height)
        depth = np.clip(depth, near, far)
        cam_pos, cam_rot = self.get_camera_pose(cam_id)

        K = self.get_intrinsic_mat(fovy, width, height)
        E = self.get_extrinsic_mat(cam_pos, cam_rot)
        return rgb, depth, K, E

    def render_camera(
        self,
        cam_name: str,
        width: int,
        height: int,
        near: float,
        far: float,
        fovy: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Renders RGB, depth, intrinsic and extrinsic matrices from a named camera.

        Args:
            cam_name: Name of the camera in the MuJoCo model.
            width: Image width in pixels.
            height: Image height in pixels.
            near: Near clipping plane distance.
            far: Far clipping plane distance.
            fovy: Vertical field-of-view in degrees.

        Returns:
            A tuple (rgb, depth, intrinsic_matrix, extrinsic_matrix).

        Raises:
            ValueError: If the camera name is not found in the model.
        """
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name
        )
        if cam_id < 0:
            raise ValueError(f"Camera '{cam_name}' not found in model")
        rgb, depth = self._render_color_depth(cam_id, width, height)
        depth = np.clip(depth, near, far)
        cam_pos, cam_rot = self.get_camera_pose(cam_id)
        K = self.get_intrinsic_mat(fovy, width, height)
        E = self.get_extrinsic_mat(cam_pos, cam_rot)
        return rgb, depth, K, E

    def render_additional_cameras(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Renders outputs from all extra cameras defined in the configuration.

        Returns:
            A dict mapping camera names to dicts containing 'rgb', 'depth',
            'intrinsic', and 'extrinsic' keys.
        """
        outputs: Dict[str, Dict[str, np.ndarray]] = {}
        if not self.extra_cameras:
            return outputs
        base_cfg = dict(self.cfg.get("mujoco", {}).get("camera", {}))
        for name, cam_id in self.extra_cameras.items():
            specs = dict(self.extra_specs.get(name, {}))
            width = int(specs.get("width", base_cfg.get("width", 640)))
            height = int(specs.get("height", base_cfg.get("height", 480)))
            near = float(specs.get("near", base_cfg.get("near", 0.01)))
            far = float(specs.get("far", base_cfg.get("far", 5.0)))
            fovy = float(specs.get("fovy", base_cfg.get("fovy", 58.0)))
            rgb, depth = self._render_color_depth(cam_id, width, height)
            depth = np.clip(depth, near, far)
            cam_pos, cam_rot = self.get_camera_pose(cam_id)
            K = self.get_intrinsic_mat(fovy, width, height)
            E = self.get_extrinsic_mat(cam_pos, cam_rot)
            outputs[name] = {
                "rgb": rgb,
                "depth": depth,
                "intrinsic": K,
                "extrinsic": E,
            }
        return outputs

    # ------------------------------------------------------------------
    # Robot helpers
    # ------------------------------------------------------------------

    def set_arm_joint_positions(
        self,
        joint_positions: Iterable[float],
        *,
        clamp: bool = True,
        sync: bool = True,
    ) -> None:
        """Applies joint positions and updates actuator targets to match."""
        joint_positions = list(joint_positions)
        self.robot.set_arm_joint_positions(
            joint_positions, clamp=clamp, sync=sync
        )
        self._set_arm_joint_targets(joint_positions)

    def contacts(self) -> list[dict]:
        """Returns a list of all active contacts in the simulation.

        Returns:
            A list of dicts with keys 'geom1', 'geom2', 'dist', 'pos', and 'frame'.
        """
        contacts = []
        for i in range(int(self.data.ncon)):
            con = self.data.contact[i]
            contacts.append(
                {
                    "geom1": int(con.geom1),
                    "geom2": int(con.geom2),
                    "dist": float(con.dist),
                    "pos": np.array(con.pos, dtype=float).copy(),
                    "frame": np.array(con.frame, dtype=float).copy(),
                }
            )
        return contacts

    def bodies_colliding(self, body1: str, body2: str) -> bool:
        """Checks whether two named bodies are currently in contact.

        Args:
            body1: Name of the first body.
            body2: Name of the second body.

        Returns:
            True if any geom from body1 is in contact with any geom from body2.
        """
        b1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body1)
        b2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body2)
        if b1 < 0 or b2 < 0:
            return False
        geoms_b1 = set(np.where(self.model.geom_bodyid == b1)[0].tolist())
        geoms_b2 = set(np.where(self.model.geom_bodyid == b2)[0].tolist())
        for contact in self.contacts():
            g1 = contact["geom1"]
            g2 = contact["geom2"]
            if (g1 in geoms_b1 and g2 in geoms_b2) or (
                g2 in geoms_b1 and g1 in geoms_b2
            ):
                return True
        return False

    def check_robot_obstacle_collision(
        self, robot_body_prefix: str = "panda"
    ) -> bool:
        """Checks if the robot arm is colliding with any moving obstacles.

        Args:
            robot_body_prefix: Prefix of robot body names (e.g., "panda" for Franka Panda).
                              Also checks for common robot parts: hand, finger, gripper.

        Returns:
            True if robot is colliding with any obstacle, False otherwise.
        """
        # Get all moving obstacle names
        mobjs = self.ids.get("moving_obstacles", {})
        if not mobjs:
            return False

        # Get all robot body IDs (check for links, hand, fingers, gripper)
        robot_body_ids = set()
        robot_keywords = [
            robot_body_prefix.lower(),
            "hand",
            "finger",
            "gripper",
        ]
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, i
            )
            if body_name and any(
                keyword in body_name.lower() for keyword in robot_keywords
            ):
                robot_body_ids.add(i)

        # Get all robot geoms
        robot_geoms = set()
        for body_id in robot_body_ids:
            geoms = np.where(self.model.geom_bodyid == body_id)[0]
            robot_geoms.update(geoms.tolist())

        # Get all obstacle body IDs and their geoms
        obstacle_geoms = set()
        for obstacle_name in mobjs.keys():
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, obstacle_name
            )
            if body_id >= 0:
                geoms = np.where(self.model.geom_bodyid == body_id)[0]
                obstacle_geoms.update(geoms.tolist())

        # Check contacts
        for contact in self.contacts():
            g1 = contact["geom1"]
            g2 = contact["geom2"]
            if (g1 in robot_geoms and g2 in obstacle_geoms) or (
                g2 in robot_geoms and g1 in obstacle_geoms
            ):
                return True

        return False

    def check_object_in_basket(self) -> dict:
        """Check whether the configured grasp object is inside the basket bounds."""
        basket_center = self.ids.get("basket_center")
        basket_dims = self.ids.get("basket_dims")
        basket_height = self.ids.get("basket_height")

        if basket_center is None or basket_dims is None or basket_height is None:
            return {
                "in_basket": False,
                "in_x": False,
                "in_y": False,
                "in_z": False,
                "object_pos": None,
            }

        grasp_obj_info = self.ids.get("grasp_object", {})
        obj_body_name = grasp_obj_info.get("body_name", "sample_object")

        obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, obj_body_name
        )
        if obj_body_id < 0:
            return {
                "in_basket": False,
                "in_x": False,
                "in_y": False,
                "in_z": False,
                "object_pos": None,
            }

        obj_pos = self.data.xpos[obj_body_id].copy()

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

    def check_object_slip(
        self,
        object_body_name: str,
        gripper_body_name: str = "hand",
        velocity_threshold: float = 0.05,
    ) -> bool:
        """Checks if a grasped object is slipping from the gripper.

        Args:
            object_body_name: Name of the object body to check.
            gripper_body_name: Name of the gripper/hand body.
            velocity_threshold: Relative velocity threshold (m/s) to consider as slipping.

        Returns:
            True if object is slipping, False otherwise.
        """
        # Get body IDs
        obj_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, object_body_name
        )
        gripper_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, gripper_body_name
        )

        if obj_id < 0 or gripper_id < 0:
            return False

        # Check if gripper is in contact with object
        in_contact = self.bodies_colliding(object_body_name, gripper_body_name)
        if not in_contact:
            # Not grasping, so not slipping
            return False

        # Get velocities
        obj_vel = self._get_body_velocity(obj_id)
        gripper_vel = self._get_body_velocity(gripper_id)

        if obj_vel is None or gripper_vel is None:
            return False

        # Calculate relative velocity
        rel_vel = np.linalg.norm(obj_vel - gripper_vel)

        # If relative velocity is high while in contact, object is slipping
        return rel_vel > velocity_threshold

    def _get_body_velocity(self, body_id: int) -> np.ndarray | None:
        """Get linear velocity of a body.

        Args:
            body_id: MuJoCo body ID.

        Returns:
            3D velocity vector or None if body has no free joint.
        """
        if body_id < 0 or body_id >= self.model.nbody:
            return None

        # Find the free joint for this body (if any)
        joint_adr = self.model.body_jntadr[body_id]
        if joint_adr < 0:
            return None

        joint_type = self.model.jnt_type[joint_adr]

        # Only handle free joints (which have 6 DOFs: 3 linear + 3 angular)
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            dof_adr = self.model.jnt_dofadr[joint_adr]
            linear_vel = self.data.qvel[dof_adr : dof_adr + 3].copy()
            return linear_vel

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _maybe_launch_viewer(self):
        viewer_spec = importlib.util.find_spec("mujoco.viewer")
        if viewer_spec is None:
            return None
        viewer_module = importlib.import_module("mujoco.viewer")
        return viewer_module.launch_passive(self.model, self.data)

    def _render_color_depth(
        self, cam_id: int, width: int, height: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.renderer.update_scene(self.data, camera=cam_id)

        color_buffer = np.empty((height, width, 3), dtype=np.uint8)
        color = self.renderer.render(out=color_buffer)

        if hasattr(self.renderer, "enable_depth_rendering"):
            self.renderer.enable_depth_rendering()
            depth_buffer = np.empty((height, width), dtype=np.float32)
            depth = self.renderer.render(out=depth_buffer)
            if hasattr(self.renderer, "disable_depth_rendering"):
                self.renderer.disable_depth_rendering()
        else:
            depth = np.zeros((height, width), dtype=np.float32)

        return np.asarray(color), np.asarray(depth, dtype=np.float32)

    # floating obstacles
    def _animate_moving_obstacles(self) -> None:
        mobjs = self.ids.get("moving_obstacles")
        if not mobjs:
            return
        t = self._sim_time
        for meta in mobjs.values():
            qadr = int(meta["qadr"])  # free joint qpos start idx
            center = np.asarray(meta["center"], dtype=float)
            axis = meta["axis"]
            amp = float(meta["amplitude"])
            freq = float(meta["frequency"])
            phase = float(meta["phase"])
            jitter_scale = float(meta["jitter_scale"])
            jitter_smooth = float(meta["jitter_smooth"])
            # sinusoidal displacement
            delta = amp * np.sin(2.0 * np.pi * freq * t + phase)

            noise = np.random.normal(scale=jitter_scale, size=3)
            self.jitter = (
                jitter_smooth * self.jitter + (1.0 - jitter_smooth) * noise
            )
            pos = center.copy()
            if axis == "x":
                pos[0] += delta
            elif axis == "y":
                pos[1] += delta
            else:  # "z"
                pos[2] += delta
            # qpos layout for free joint: [x y z qw qx qy qz]
            pos += self.jitter
            self.data.qpos[qadr : qadr + 3] = pos
        # Note: mj_forward is not needed here since step() calls mj_step which includes forward kinematics

    def _project(
        self,
        point: Iterable[float],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        V: np.ndarray,
    ) -> Tuple[int, int] | None:
        pw = np.array([point[0], point[1], point[2], 1.0], dtype=float)
        pc = V @ pw
        z = pc[2]
        if abs(z) < 1e-6:
            return None
        u = int((pc[0] * fx / z) + cx)
        v = int((pc[1] * fy / z) + cy)
        return (u, v)

    def _detect_gripper_actuators(self) -> Tuple[int, ...]:
        indices = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
            )
            if name and ("finger" in name or "gripper" in name):
                indices.append(i)
        return tuple(indices)

    def _detect_arm_actuators(self) -> Tuple[int, ...]:
        """Returns actuator indices that drive the arm joints in arm_pairs order."""
        qpos_to_act = {}
        for act_idx in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[act_idx, 0])
            if joint_id < 0 or joint_id >= self.model.njnt:
                continue
            qadr = int(self.model.jnt_qposadr[joint_id])
            qpos_to_act[qadr] = act_idx
        arm_actuators: list[int] = []
        for qidx, _ in self.robot.arm_pairs:
            act_idx = qpos_to_act.get(int(qidx))
            if act_idx is not None:
                arm_actuators.append(act_idx)
        return tuple(arm_actuators)

    def _set_gripper_opening(self, opening: float) -> None:
        for idx in self._finger_indices:
            lo, hi = self.model.actuator_ctrlrange[idx]
            print(f"Opening: {opening}, Lo: {lo}, Hi: {hi}")
            value = (
                float(np.clip(opening, lo, hi))
                if np.isfinite(lo) and np.isfinite(hi)
                else opening
            )
            self.data.ctrl[idx] = value

    def _set_arm_joint_targets(self, joint_positions: Iterable[float]) -> None:
        """Sets actuator targets to match provided joint positions."""
        if not self._arm_actuator_indices:
            return
        for value, act_idx in zip(joint_positions, self._arm_actuator_indices):
            lo, hi = self.model.actuator_ctrlrange[act_idx]
            target = float(value)
            if np.isfinite(lo) and np.isfinite(hi):
                target = float(np.clip(target, lo, hi))
            self.data.ctrl[act_idx] = target

    def _resolve_gripper_opening(self, requested: bool | float) -> float:
        if isinstance(requested, bool):
            if requested:
                return float(
                    self.robot_settings.get("default_gripper_opening", 0.04)
                )
            return 0.0
        return float(requested)
