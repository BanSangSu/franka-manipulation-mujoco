"""
PointCloudBuilder – back-projects depth images through per-camera intrinsics /
extrinsics and fuses the results into a single world-frame point cloud.

Provides optional Open3D visualisation.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from .camera_manager import CameraFrame

logger = logging.getLogger(__name__)


class PointCloudBuilder:
    """Build and fuse point clouds from multiple depth + mask pairs.

    Parameters
    ----------
    voxel_size : float
        Down-sampling voxel size (metres) applied after fusion.
        Set to 0 to skip down-sampling.
    depth_trunc : float
        Ignore depth values beyond this distance (metres).
    """

    def __init__(self, voxel_size: float = 0.002, depth_trunc: float = 3.0):
        self.voxel_size = voxel_size
        self.depth_trunc = depth_trunc

    # ------------------------------------------------------------------
    # Core: depth + mask → world-frame points
    # ------------------------------------------------------------------
    @staticmethod
    def deproject_to_camera_frame(
        depth: np.ndarray,
        K: np.ndarray,
        mask: np.ndarray | None = None,
        depth_trunc: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Back-project depth map into 3-D points in *camera* frame.

        Parameters
        ----------
        depth : (H, W) float32 – metres.
        K : (3, 3) intrinsic matrix.
        mask : (H, W) bool – only project pixels where True.
        depth_trunc : float – ignore depths above this.

        Returns
        -------
        points_cam : (N, 3) float64 – xyz in camera frame.
        pixel_indices : (N, 2) int – (row, col) of each point.
        """
        H, W = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Build pixel grid
        u_map, v_map = np.meshgrid(np.arange(W), np.arange(H))  # u=col, v=row

        # Combined valid mask
        valid = (depth > 0) & (depth < depth_trunc) & np.isfinite(depth)
        if mask is not None:
            valid &= mask

        us = u_map[valid].astype(np.float64)
        vs = v_map[valid].astype(np.float64)
        zs = depth[valid].astype(np.float64)

        # MuJoCo convention: camera looks along -Z, but the depth values are
        # positive distances along the optical axis.
        # We want Z in camera frame to be in front of the camera (-Z).
        z_cam = -zs
        
        # Given fx is negative and fy is positive in get_intrinsic_mat:
        # x_cam = (us - cx) * z_cam / fx -> right is +X
        # y_cam = (vs - cy) * z_cam / fy -> up is +Y
        xs = (us - cx) * z_cam / fx
        ys = (vs - cy) * z_cam / fy

        points_cam = np.stack([xs, ys, z_cam], axis=-1)  # (N, 3)
        pixel_indices = np.stack([vs.astype(int), us.astype(int)], axis=-1)
        return points_cam, pixel_indices

    @staticmethod
    def camera_to_world(
        points_cam: np.ndarray, E: np.ndarray
    ) -> np.ndarray:
        """Transform points from camera frame to world frame.

        Parameters
        ----------
        points_cam : (N, 3)
        E : (4, 4) extrinsic matrix  (world→camera).
            We invert it to get camera→world.

        Returns
        -------
        points_world : (N, 3)
        """
        # E maps world→camera : p_cam = E @ p_world
        # Inverse:  p_world = E_inv @ p_cam
        E_inv = np.linalg.inv(E)
        ones = np.ones((points_cam.shape[0], 1), dtype=np.float64)
        pts_h = np.hstack([points_cam, ones])  # (N, 4)
        pts_world_h = (E_inv @ pts_h.T).T       # (N, 4)
        return pts_world_h[:, :3]

    # ------------------------------------------------------------------
    # Multi-camera fusion
    # ------------------------------------------------------------------
    def fuse(
        self,
        frames: List[CameraFrame],
        masks: List[np.ndarray],
        rgb_list: List[np.ndarray] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Fuse masked point clouds from multiple cameras into one world-frame cloud.

        Parameters
        ----------
        frames : list[CameraFrame]
        masks : list[ndarray (H,W) bool]
            One binary mask per frame – only masked pixels are projected.
        rgb_list : list[ndarray] | None
            If supplied, corresponding RGB values are carried along.

        Returns
        -------
        fused_points : (M, 3) float64 – world-frame xyz.
        fused_colors : (M, 3) float64 | None – RGB normalised to [0, 1].
        """
        all_pts: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []

        for i, (frame, mask) in enumerate(zip(frames, masks)):
            pts_cam, px_idx = self.deproject_to_camera_frame(
                frame.depth, frame.intrinsic, mask=mask, depth_trunc=self.depth_trunc
            )
            if pts_cam.shape[0] == 0:
                logger.warning(
                    "Camera '%s': no valid points after masking", frame.cam_name
                )
                continue

            pts_world = self.camera_to_world(pts_cam, frame.extrinsic)
            all_pts.append(pts_world)
            logger.info(
                "Camera '%s': %d pts projected to world frame",
                frame.cam_name,
                pts_world.shape[0],
            )

            if rgb_list is not None:
                rgb = rgb_list[i]
                colors = rgb[px_idx[:, 0], px_idx[:, 1]].astype(np.float64) / 255.0
                all_colors.append(colors)

        if not all_pts:
            return np.empty((0, 3)), None

        fused_points = np.concatenate(all_pts, axis=0)
        fused_colors = np.concatenate(all_colors, axis=0) if all_colors else None

        logger.info("Fused cloud: %d points before downsampling", fused_points.shape[0])

        # Optional voxel down-sampling
        if self.voxel_size > 0 and fused_points.shape[0] > 0:
            fused_points, fused_colors = self._voxel_downsample(
                fused_points, fused_colors
            )
            logger.info(
                "After voxel downsampling (%.4f m): %d points",
                self.voxel_size,
                fused_points.shape[0],
            )

        return fused_points, fused_colors

    # ------------------------------------------------------------------
    # Visualisation (Open3D)
    # ------------------------------------------------------------------
    @staticmethod
    def visualize(
        points: np.ndarray,
        colors: np.ndarray | None = None,
        target_points: np.ndarray | None = None,
        target_colors: np.ndarray | None = None,
        window_name: str = "Fused Point Cloud",
        point_size: float = 2.0,
        coordinate_frame_size: float = 0.15,
    ) -> None:
        """Open an Open3D window showing the fused (scene) cloud and optional target cloud.

        Parameters
        ----------
        points : (N, 3) – full scene cloud.
        colors : (N, 3) or None – per-point RGB in [0, 1].
        target_points : (M, 3) or None – target object cloud (highlighted).
        target_colors : (M, 3) or None – if None and target_points given, use green.
        window_name : str
        point_size : float
        coordinate_frame_size : float
        """
        import open3d as o3d

        geometries = []

        # World coordinate frame
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=coordinate_frame_size
        )
        geometries.append(coord)

        # Scene cloud
        if points.shape[0] > 0:
            pcd_scene = o3d.geometry.PointCloud()
            pcd_scene.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                pcd_scene.colors = o3d.utility.Vector3dVector(colors)
            else:
                # Grey
                pcd_scene.paint_uniform_color([0.6, 0.6, 0.6])
            geometries.append(pcd_scene)

        # Target object cloud (highlighted)
        if target_points is not None and target_points.shape[0] > 0:
            pcd_target = o3d.geometry.PointCloud()
            pcd_target.points = o3d.utility.Vector3dVector(target_points)
            if target_colors is not None:
                pcd_target.colors = o3d.utility.Vector3dVector(target_colors)
            else:
                pcd_target.paint_uniform_color([0.0, 1.0, 0.0])  # green
            geometries.append(pcd_target)

        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _voxel_downsample(
        self, points: np.ndarray, colors: np.ndarray | None
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Simple grid-based voxel down-sampling using numpy."""
        quantized = np.floor(points / self.voxel_size).astype(np.int64)
        # Unique voxel keys
        _, unique_idx = np.unique(quantized, axis=0, return_index=True)
        unique_idx.sort()
        ds_pts = points[unique_idx]
        ds_colors = colors[unique_idx] if colors is not None else None
        return ds_pts, ds_colors
