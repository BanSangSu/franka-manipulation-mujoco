"""
PerceptionPipeline – ties CameraManager, Florence2Segmentor, and
PointCloudBuilder together into a single call:

    pipeline.run()  →  (target_points, target_colors, scene_points, scene_colors)

Optionally visualises the result via Open3D.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np

from .camera_manager import CameraManager, CameraFrame
from .florence2 import Florence2Segmentor
from .point_cloud_builder import PointCloudBuilder
from .grasp_planner import GraspPlanner

logger = logging.getLogger(__name__)


class PerceptionPipeline:
    """End-to-end perception: capture → segment → fuse → grasp estimate.

    Parameters
    ----------
    sim : MjSim
        Initialised simulation.
    target_object : str
        Natural-language label of the object to segment (e.g. "mustard bottle").
    camera_names : list[str] | None
    voxel_size, depth_trunc : float
        Point-cloud builder settings.
    grasp_config : str | None
        GraspGen config path.
    num_grasps, topk : int
        GraspGen sampling settings.
    visualize : bool
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
    ):
        self.camera_mgr = CameraManager(sim, camera_names)
        self.florence2 = Florence2Segmentor(
            target_object=target_object,
            model_id=florence_model,
        )
        self.pc_builder = PointCloudBuilder(
            voxel_size=voxel_size, depth_trunc=depth_trunc
        )
        self.grasp_planner = GraspPlanner(
            gripper_config=grasp_config,
            num_grasps=num_grasps,
            topk=topk,
        )
        self.target_object = target_object
        self._visualize = visualize

    def run(
        self,
        visualize: bool | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute the full perception pipeline.

        Returns
        -------
        grasps : (K, 4, 4) – grasp candidates.
        confs : (K,) – confidence scores.
        target_points : (M, 3) – object cloud.
        scene_points : (N, 3) – full scene cloud.
        """
        do_viz = visualize if visualize is not None else self._visualize

        # 1. Capture & Segment
        frames = self.camera_mgr.capture_all()
        cropped_frames = []
        cropped_rgbs = []
        scene_masks = []
        import copy
        
        if do_viz:
            import matplotlib.pyplot as plt
            
        for f in frames:
            H, W = f.rgb.shape[:2]
            bbox = self.florence2.detect(f.rgb, self.target_object)
            
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                
                c_f = copy.copy(f)
                c_f.rgb = f.rgb[y1:y2, x1:x2].copy()
                c_f.depth = f.depth[y1:y2, x1:x2].copy()
                
                c_K = f.intrinsic.copy()
                c_K[0, 2] -= x1
                c_K[1, 2] -= y1
                c_f.intrinsic = c_K
                
                cropped_frames.append(c_f)
                cropped_rgbs.append(c_f.rgb)
                
                m = np.ones((H, W), dtype=bool)
                x1_d = max(0, x1 - 15)
                y1_d = max(0, y1 - 15)
                x2_d = min(W, x2 + 15)
                y2_d = min(H, y2 + 15)
                m[y1_d:y2_d, x1_d:x2_d] = False
                scene_masks.append(m)
            else:
                scene_masks.append(np.ones((H, W), dtype=bool))
                
            # Show RGB detection visualization without requiring OpenCV
            if do_viz:
                import os
                overlay = f.rgb.copy()
                if bbox is not None:
                    thickness = 2
                    overlay[max(0, y1-thickness):min(H, y1+thickness), max(0, x1-thickness):min(W, x2+thickness)] = [0, 255, 0]
                    overlay[max(0, y2-thickness):min(H, y2+thickness), max(0, x1-thickness):min(W, x2+thickness)] = [0, 255, 0]
                    overlay[max(0, y1-thickness):min(H, y2+thickness), max(0, x1-thickness):min(W, x1+thickness)] = [0, 255, 0]
                    overlay[max(0, y1-thickness):min(H, y2+thickness), max(0, x2-thickness):min(W, x2+thickness)] = [0, 255, 0]
                    
                plt.figure(f"Detection - {f.cam_name}")
                plt.imshow(overlay)
                plt.title(f"Detection for {self.target_object} - {f.cam_name}")
                plt.axis('off')
                
                os.makedirs("debug_output", exist_ok=True)
                save_path = f"debug_output/det_{f.cam_name}.png"
                plt.savefig(save_path, bbox_inches='tight')
                logger.info(f"Saved detection visualization to {save_path}")
                
        if do_viz:
            logger.info("Displaying RGB detection windows...")
            plt.show(block=False)
            plt.pause(2.0)
            plt.close('all')

        # 2. Build Point Clouds
        target_points, target_colors = self.pc_builder.fuse(
            cropped_frames, [None] * len(cropped_frames), cropped_rgbs
        )
        
        # Build scene points EXCLUDING the target object so the grasps don't collide with the target itself
        scene_points, scene_colors = self.pc_builder.fuse(
            frames, scene_masks, [f.rgb for f in frames]
        )

        # 3. Clean target cloud (Outlier Removal)
        if target_points.shape[0] > 0:
            import torch
            from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal_with_color
            
            # Move to torch for the optimized KNN filtering
            tp_torch = torch.from_numpy(target_points).float()
            tc_torch = torch.from_numpy(target_colors).float() if target_colors is not None else torch.zeros_like(tp_torch)
            
            filtered_pts, _, filtered_clr, _ = point_cloud_outlier_removal_with_color(tp_torch, tc_torch)
            
            target_points = filtered_pts.cpu().numpy()
            if target_colors is not None:
                target_colors = filtered_clr.cpu().numpy()

        # --- ICP ALIGNMENT WITH FULL 3D MODEL ---
        if target_points.shape[0] > 0:
            try:
                import open3d as o3d
                import yaml
                import os
                
                with open("configs/test_config_mj.yaml", "r") as f:
                    config = yaml.safe_load(f)
                
                xml_path = config.get("mujoco", {}).get("grasp_object", {}).get("xml", "")
                if xml_path:
                    obj_path = os.path.join(os.path.dirname(xml_path), "textured.obj")
                    if os.path.exists(obj_path):
                        mesh = o3d.io.read_triangle_mesh(obj_path)
                        mesh.compute_vertex_normals()
                        
                        num_points = max(5000, target_points.shape[0] * 2)
                        model_pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
                        
                        target_pcd = o3d.geometry.PointCloud()
                        target_pcd.points = o3d.utility.Vector3dVector(target_points)
                        
                        # --- PRE-PROCESS TARGET CLOUD ---
                        # 1. Deterministic Table Removal: Z > 0.73 filters the ~0.725 table completely without RANSAC randomness
                        pts = np.asarray(target_pcd.points)
                        valid_idx = np.where(pts[:, 2] > 0.73)[0]
                        if len(valid_idx) > 10:
                            target_pcd = target_pcd.select_by_index(valid_idx)
                            
                        # 2. DBSCAN clustering to keep only the object
                        labels = np.array(target_pcd.cluster_dbscan(eps=0.04, min_points=10, print_progress=False))
                        if len(labels) > 0 and labels.max() >= 0:
                            largest_cluster_idx = np.bincount(labels[labels >= 0]).argmax()
                            target_pcd = target_pcd.select_by_index(np.where(labels == largest_cluster_idx)[0])
                        # --------------------------------
                        
                        target_center = target_pcd.get_center()
                        model_center = model_pcd.get_center()
                        
                        best_fitness = -1.0
                        best_transformation = np.eye(4)
                        
                        threshold = 0.05
                        
                        # Generate robust alignment initializations (all 90-deg orientations)
                        rotations = []
                        for axis_idx in range(3):
                            for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
                                if axis_idx > 0 and angle == 0: continue # Skip duplicate identity
                                r3 = np.eye(3)
                                c, s = np.cos(angle), np.sin(angle)
                                if axis_idx == 0:   # X
                                    r3[1,1], r3[1,2], r3[2,1], r3[2,2] = c, -s, s, c
                                elif axis_idx == 1: # Y
                                    r3[0,0], r3[0,2], r3[2,0], r3[2,2] = c, s, -s, c
                                else:               # Z
                                    r3[0,0], r3[0,1], r3[1,0], r3[1,1] = c, -s, s, c
                                
                                r4 = np.eye(4)
                                r4[:3, :3] = r3
                                rotations.append(r4)
                        
                        for r in rotations:
                            # Align centroids with the rotation
                            t1 = np.eye(4)
                            t1[0, 3] = -target_center[0]
                            t1[1, 3] = -target_center[1]
                            t1[2, 3] = -target_center[2]
                            t2 = np.eye(4); t2[:3, 3] = model_center
                            
                            init_T = t2 @ r @ t1
                            
                            reg = o3d.pipelines.registration.registration_icp(
                                target_pcd, model_pcd, threshold, init_T,
                                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
                            )
                            if reg.fitness > best_fitness:
                                best_fitness = reg.fitness
                                best_transformation = reg.transformation
                                
                        # Invert transformation to map model -> target world space
                        final_T = np.linalg.inv(best_transformation)
                        model_pcd.transform(final_T)
                        
                        target_points = np.asarray(model_pcd.points)
                        
                        if target_colors is not None:
                            new_colors = np.zeros_like(target_points)
                            new_colors[:, 2] = 1.0  # Paint the full model blue
                            target_colors = new_colors
                            
                        logger.info(f"ICP alignment successful. Replaced target cloud with {target_points.shape[0]} points.")
                    else:
                        logger.warning(f"Could not find model file {obj_path} for ICP. Using partial cloud.")
            except Exception as e:
                logger.error(f"ICP alignment failed: {e}. Using partial cloud.")

        # --- Synthesize floor patch beneath target object ---
        if target_points.shape[0] > 0:
            xy_min = target_points[:, :2].min(axis=0) - 0.15
            xy_max = target_points[:, :2].max(axis=0) + 0.15

            grid_res = 0.01
            xs = np.arange(xy_min[0], xy_max[0], grid_res)
            ys = np.arange(xy_min[1], xy_max[1], grid_res)
            xx, yy = np.meshgrid(xs, ys)

            floor_z = 0.73
            floor_layers = []
            for dz in [0.0, -0.01, -0.02]:   # 3 layers so voxel downsampling keeps at least one
                layer = np.stack([xx.ravel(), yy.ravel(),
                                  np.full(xx.size, floor_z + dz)], axis=1)
                floor_layers.append(layer)

            floor_pts = np.concatenate(floor_layers, axis=0).astype(np.float32)
            scene_points = np.concatenate([scene_points, floor_pts], axis=0)
            if scene_colors is not None:
                floor_colors = np.full((len(floor_pts), 3), 0.5, dtype=np.float32)
                scene_colors = np.concatenate([scene_colors, floor_colors], axis=0)

        # 4. Visualise before planning
        if do_viz:
            # Keep original colors for the scene, but paint the target object solid blue
            blue_target_colors = np.zeros_like(target_points)
            blue_target_colors[:, 2] = 1.0  # RGB: [0, 0, 1] is blue
            self.pc_builder.visualize(scene_points, scene_colors, target_points, blue_target_colors)

        # 5. Plan Grasps (the perception of affordances)
        grasps = np.empty((0, 4, 4))
        confs = np.empty((0,))
        if target_points.shape[0] > 0:
            grasps, confs = self.grasp_planner.plan(target_points, scene_points=scene_points)
            if do_viz and grasps.shape[0] > 0:
                self.grasp_planner.visualize(
                    target_points, grasps, confs,
                    scene_points=scene_points, scene_colors=scene_colors,
                )

        return grasps, confs, target_points, scene_points

