"""
GraspPlanner – wraps GraspGen to generate grasp candidates from
a segmented target-object point cloud (already in world frame).

Usage:
    planner = GraspPlanner()
    grasps, confs = planner.plan(target_points)
    planner.visualize(target_points, grasps)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve GraspGen import path
# ---------------------------------------------------------------------------
_GRASPGEN_ROOT = Path(__file__).resolve().parents[3] / "GraspGen"
if str(_GRASPGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GRASPGEN_ROOT))


class GraspPlanner:
    """Generate and rank 6-DoF grasps for a Franka Panda gripper.

    Parameters
    ----------
    gripper_config : str | Path
        Path to the GraspGen YAML config (e.g.
        ``GraspGen/GraspGenModels/checkpoints/graspgen_franka_panda.yml``).
    num_grasps : int
        How many grasp samples to generate per batch.
    topk : int
        Return at most this many grasps (sorted by confidence descending).
    grasp_threshold : float
        Minimum discriminator score to keep a grasp.
        -1 → use ``topk`` based ranking only.
    device : str | None
        Torch device.  ``None`` → auto (cuda if available).
    """

    _DEFAULT_CONFIG = (
        _GRASPGEN_ROOT / "GraspGenModels" / "checkpoints" / "graspgen_franka_panda.yml"
    )

    def __init__(
        self,
        gripper_config: str | Path | None = None,
        num_grasps: int = 200,
        topk: int = 50,
        grasp_threshold: float = -1.0,
        device: str | None = None,
    ):
        self.gripper_config = Path(gripper_config or self._DEFAULT_CONFIG)
        self.num_grasps = num_grasps
        self.topk = topk
        self.grasp_threshold = grasp_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._sampler: Optional[object] = None
        self._gripper_collision_mesh = None
        self._gripper_visual_mesh = None

    # ------------------------------------------------------------------
    # Lazy-load model
    # ------------------------------------------------------------------
    def _load(self):
        if self._sampler is not None:
            return
        from grasp_gen.grasp_server import load_grasp_cfg, GraspGenSampler
        from grasp_gen.robot import get_gripper_info

        logger.info("Loading GraspGen model from %s …", self.gripper_config)
        cfg = load_grasp_cfg(str(self.gripper_config))
        self._sampler = GraspGenSampler(cfg)
        
        gripper_name = cfg.data.gripper_name
        gripper_info = get_gripper_info(gripper_name)
        self._gripper_collision_mesh = gripper_info.collision_mesh
        self._gripper_visual_mesh = gripper_info.visual_mesh
        
        logger.info("GraspGen ready on %s", self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan(
        self, target_points: np.ndarray, scene_points: np.ndarray | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate grasp candidates for a target point cloud.

        Parameters
        ----------
        target_points : (N, 3) float
            XYZ coordinates in *world frame*.

        Returns
        -------
        grasps : (K, 4, 4) float – SE(3) grasp poses in world frame.
        confidences : (K,) float – discriminator confidence scores.
        """
        self._load()
        from grasp_gen.grasp_server import GraspGenSampler

        pc_tensor = torch.from_numpy(target_points).float().to(self.device)

        grasps, confs = GraspGenSampler.run_inference(
            object_pc=pc_tensor,
            grasp_sampler=self._sampler,
            grasp_threshold=self.grasp_threshold,
            num_grasps=self.num_grasps,
            topk_num_grasps=self.topk,
        )

        grasps_np = grasps.cpu().numpy() if len(grasps) > 0 else np.empty((0, 4, 4))
        confs_np = confs.cpu().numpy() if len(confs) > 0 else np.empty((0,))

        if scene_points is not None and grasps_np.shape[0] > 0 and hasattr(self, "_gripper_collision_mesh"):
            from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps
            
            # Downsample scene points internally to speed up collision checking
            max_scene_points = 8192
            if len(scene_points) > max_scene_points:
                indices = np.random.choice(len(scene_points), max_scene_points, replace=False)
                scene_downsampled = scene_points[indices]
            else:
                scene_downsampled = scene_points
                
            collision_free_mask = filter_colliding_grasps(
                scene_pc=scene_downsampled,
                grasp_poses=grasps_np,
                gripper_collision_mesh=self._gripper_collision_mesh,
                collision_threshold=0.005,
            )
            grasps_np = grasps_np[collision_free_mask]
            confs_np = confs_np[collision_free_mask]
            logger.info("Filtered %d colliding grasps, %d remaining", len(collision_free_mask) - sum(collision_free_mask), grasps_np.shape[0])

        logger.info(
            "GraspGen returned %d collision-free grasps  (conf range: %.3f – %.3f)",
            grasps_np.shape[0],
            confs_np.min() if len(confs_np) else 0,
            confs_np.max() if len(confs_np) else 0,
        )
        return grasps_np, confs_np

    # ------------------------------------------------------------------
    # Visualise grasps in Open3D
    # ------------------------------------------------------------------
    def visualize(
        self,
        target_points: np.ndarray,
        grasps: np.ndarray,
        confidences: np.ndarray | None = None,
        scene_points: np.ndarray | None = None,
        scene_colors: np.ndarray | None = None,
        top_n: int = 10,
    ) -> None:
        """Show full scene cloud + target cloud + top-N gripper meshes in Open3D.

        Parameters
        ----------
        target_points : (N, 3) – object point cloud.
        grasps : (K, 4, 4) – SE(3) grasp poses.
        confidences : (K,) – optional scores.
        scene_points : (M, 3) – full scene point cloud (optional).
        scene_colors : (M, 3) – per-point RGB colours in [0, 1] (optional).
        top_n : int – how many grasps to draw.
        """
        import open3d as o3d

        geometries = []

        # World frame
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(coord)

        self._load()

        # Full scene cloud (grey if no colours provided)
        if scene_points is not None and scene_points.shape[0] > 0:
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
            if scene_colors is not None and scene_colors.shape[0] == scene_points.shape[0]:
                scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors)
            else:
                scene_pcd.paint_uniform_color([0.6, 0.6, 0.6])
            geometries.append(scene_pcd)

        # Target object cloud (green)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(target_points)
        pcd.paint_uniform_color([0.1, 0.7, 0.2])
        geometries.append(pcd)

        assert self._gripper_visual_mesh is not None
        gripper_mesh_o3d = o3d.geometry.TriangleMesh()
        import typing
        v_mesh = typing.cast(typing.Any, self._gripper_visual_mesh)
        gripper_mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(v_mesh.vertices))
        gripper_mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(v_mesh.faces))
        gripper_mesh_o3d.compute_vertex_normals()
        gripper_mesh_o3d.paint_uniform_color([0.8, 0.2, 0.2])

        # Draw top-N gripper meshes
        if grasps.shape[0] > 0:
            if confidences is not None and len(confidences) > 0:
                order = np.argsort(-confidences)
            else:
                order = np.arange(grasps.shape[0])

            for rank, idx in enumerate(order[:top_n]):
                T = grasps[idx]
                mesh_copied = o3d.geometry.TriangleMesh(gripper_mesh_o3d)
                mesh_copied.transform(T)
                geometries.append(mesh_copied)

                gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.04
                )
                gripper_frame.transform(T)
                geometries.append(gripper_frame)

        o3d.visualization.draw_geometries(
            geometries,
            window_name="Grasp Candidates",
        )
