"""
YOLOObstacleDetector – uses YOLO26 (ultralytics) for real-time obstacle
detection from MuJoCo camera frames.

Detection classes:
    - 'sports ball'  → red obstacle
    - 'orange'       → orange obstacle

Pipeline per frame:
  1. YOLO detection (or seg if seg model) → BBox + optional mask
  2. BBox pixels mapped to depth → 3D point cloud in camera frame
  3. Point cloud → world frame via extrinsic
  4. Estimate sphere centre + radius from mapped 3D points
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import os
import cv2

logger = logging.getLogger(__name__)


class YOLOObstacleDetector:
    """Detect obstacles in RGB images using YOLO and map them to 3D spheres.

    Parameters
    ----------
    model_path : str
        Path or model name for ultralytics YOLO (e.g. "yolo26n.pt",
        "yolo26n-seg.pt").  Segmentation models are preferred.
    conf_threshold : float
        Minimum detection confidence.
    target_classes : list[str] | None
        COCO class names to treat as obstacles.  Defaults to
        ['sports ball', 'orange', 'apple'].
    device : str
        "cuda" or "cpu".
    """

    # Default COCO class names that look like the moving obstacles
    # sports ball = red sphere, orange = orange sphere
    DEFAULT_OBSTACLE_CLASSES = ["sports ball", "orange", "apple"]

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        conf_threshold: float = 0.25,
        target_classes: Optional[List[str]] = None,
        device: str = "cpu", # cpu, cuda
        fps=30, 
        record_seconds=5,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.target_classes = target_classes or self.DEFAULT_OBSTACLE_CLASSES
        self.device = device

        self._model = None
        self._is_seg = False  # True if segmentation model
        
        self.video_writer = None
        self.fps = fps
        self.max_frames = self.fps * record_seconds
        self.frame_count = 0
        self.recording_finished = False

    # ------------------------------------------------------------------
    # Lazy-load YOLO
    # ------------------------------------------------------------------
    def _load_model(self):
        if self._model is not None:
            return
        from ultralytics import YOLO

        logger.info("Loading YOLO model: %s ...", self.model_path)
        self._model = YOLO(self.model_path)
        self._is_seg = "seg" in self.model_path.lower()
        logger.info(
            "YOLO loaded (seg=%s) on %s", self._is_seg, self.device
        )

    @property
    def model(self):
        """Access the underlying YOLO model (lazy-loaded)."""
        self._load_model()
        return self._model

    # ------------------------------------------------------------------
    # Single-frame detection (used by the threaded tracker)
    # ------------------------------------------------------------------
    def detect(self, rgb: np.ndarray, use_track: bool = True, save_path: Optional[str] = None, video_save: bool = False) -> List[dict]:
        """Detect obstacles in an RGB image.

        Parameters
        ----------
        rgb : (H, W, 3) uint8
        use_track : bool
            If True, use model.track() with persist=True for tracking IDs.
        save_path : str | None
            Path to save the annotated image.

        Returns
        -------
        detections : list[dict]
            Each dict has:
                'bbox'     : [x1, y1, x2, y2] int
                'conf'     : float
                'cls'      : str  (class name)
                'track_id' : int | None (only if use_track=True)
                'mask'     : (H, W) bool | None  (only if seg model)
        """
        self._load_model()
        H, W = rgb.shape[:2]

        # ## Debug: Save input frame
        # if save_path is not None:
        #     os.makedirs(save_path, exist_ok=True)
        #     # OpenCV는 BGR을 쓰므로 저장 시 RGB -> BGR 변환이 필수입니다.
        #     original_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        #     input_fn = f"{save_path}/raw_frame_{self.frame_count}.png"
        #     cv2.imwrite(input_fn, original_bgr)

        bgr_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # YOLO expects BGR if passing np.ndarray, but ultralytics can handle RGB directly. We will keep it in RGB for detection and convert to BGR for saving if needed.
        if use_track:
            results = self._model.track(
                bgr_image,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
                persist=True,
            )
        else:
            results = self._model.predict(
                bgr_image,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )
        
        ## Debug
        # # # Save image with detections for debugging
        # if save_path and len(results) > 0:
        #     os.makedirs(save_path, exist_ok=True)
        #     for i, result in enumerate(results):
        #         plotted_img = result.plot().copy()
        #         # 스레드 안전을 위해 유니크한 파일명 생성
        #         fn = f"{save_path}/frame_{self.frame_count}_{i}.png"
        #         cv2.imwrite(fn, plotted_img)

        # Video recording for debugging
        if video_save and not self.recording_finished:
            if self.video_writer is None:
                os.makedirs(save_path, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                path = f"{save_path}/tracking_5s.mp4"
                self.video_writer = cv2.VideoWriter(path, fourcc, self.fps, (W, H))
                if not self.video_writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer for {path}")
                print(f"Started recording to {path}")

            if self.frame_count < self.max_frames:
                annotated_frame = results[0].plot()
                annotated_frame = np.ascontiguousarray(annotated_frame, dtype=np.uint8)
                if (annotated_frame.shape[1], annotated_frame.shape[0]) != (W, H):
                    annotated_frame = cv2.resize(annotated_frame, (W, H))
                self.video_writer.write(annotated_frame)
                self.frame_count += 1
            else:
                self.close_video_writer()
                print("Finished saving 5-second video.")

        detections = []
        if len(results) == 0:
            return detections

        result = results[0]
        names = result.names  # {class_id: class_name}

        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]

            # Filter by target class
            if cls_name not in self.target_classes:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            # Tracking ID (from model.track with persist=True)
            # We will use our color classification instead
            # track_id = None
            # if use_track and box.id is not None:
            #     track_id = int(box.id[0])

            # Segmentation mask (if available)
            mask = None
            if self._is_seg and result.masks is not None:
                seg_mask = result.masks.data[i].cpu().numpy()

                mask = cv2.resize(
                    seg_mask, (W, H), interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            # Classify color and assign ID (orange=1, red=2)
            track_id = self._classify_color_id(rgb, [x1, y1, x2, y2], mask)

            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "conf": conf,
                "cls": cls_name,
                "track_id": track_id,
                "mask": mask,
            })

        logger.debug("YOLO detected %d obstacles", len(detections))
        return detections

    def close_video_writer(self):
        """Flush and close the debug video writer."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.recording_finished = True

    def _classify_color_id(self, rgb: np.ndarray, bbox: list, mask: Optional[np.ndarray]) -> int:
        """Classify if the detected obstacle is orange or red.
        Returns:
            1 if orange
            2 if red
        """
        import cv2
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return 1  # Fallback

        crop = rgb[y1:y2, x1:x2]
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

        # Red: 0-10 and 160-179
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([179, 255, 255])

        # Orange: 11-25
        lower_orange = np.array([11, 70, 50])
        upper_orange = np.array([25, 255, 255])

        mask_red1 = cv2.inRange(crop_hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(crop_hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        mask_orange = cv2.inRange(crop_hsv, lower_orange, upper_orange)

        if mask is not None:
            mask_crop = mask[y1:y2, x1:x2].astype(np.uint8) * 255
            mask_red = cv2.bitwise_and(mask_red, mask_crop)
            mask_orange = cv2.bitwise_and(mask_orange, mask_crop)

        red_count = np.count_nonzero(mask_red)
        orange_count = np.count_nonzero(mask_orange)

        # ID 1 for orange, 2 for red
        if orange_count >= red_count:
            return 1
        else:
            return 2

    # ------------------------------------------------------------------
    # 3D Sphere estimation from BBox + Depth + Intrinsic/Extrinsic
    # ------------------------------------------------------------------
    def estimate_obstacle_spheres(
        self,
        depth: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        detections: List[dict],
    ) -> List[dict]:
        """For each detected obstacle, map BBox to a 3D sphere in world frame.

        Steps (per detection):
        - BBox region → select depth pixels → back-project to camera frame
        - Camera frame → world frame via extrinsic inverse
        - From world-frame points: diameter = max(z) - min(z), radius = diameter/2
        - Centre = frontmost point + radius along viewing direction

        Parameters
        ----------
        depth : (H, W) float32 – depth in metres
        intrinsic : (3, 3) camera intrinsic matrix
        extrinsic : (4, 4) world-to-camera extrinsic matrix
        detections : list[dict] from detect()

        Returns
        -------
        obstacles : list[dict]
            Each dict has:
                'center_world' : (3,) centre in world frame
                'radius'       : float
                'bbox'         : [x1, y1, x2, y2]
                'cls'          : str
                'conf'         : float
                'track_id'     : int | None
        """
        if not detections:
            return []

        E_inv = np.linalg.inv(extrinsic)
        fx, fy = abs(intrinsic[0, 0]), abs(intrinsic[1, 1]) # why the instric z is negative in MuJoCo?
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        obstacles = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            # --- Get depth within bbox (use mask if available) ---
            depth_region = depth[y1:y2, x1:x2].copy()

            if det.get("mask") is not None:
                mask_region = det["mask"][y1:y2, x1:x2]
                depth_region[~mask_region] = 0.0

            valid = (
                (depth_region > 0)
                & (depth_region < 3.0)
                & np.isfinite(depth_region)
            )

            if valid.sum() < 10:
                logger.warning("Too few valid depth points in bbox for obstacle")
                continue

            # --- Robust center depth: median of valid pixels ---
            center_depth = float(np.median(depth_region[valid]))

            # --- Radius from angular size (pinhole geometry) ---
            # A sphere of radius r at depth d projects to half-width = fx * r / d
            # Inverting: r = half_width_px * d / fx
            bbox_width_px = x2 - x1
            bbox_height_px = y2 - y1


            # # DEBUG — remove once fixed
            # print(f"[DEBUG] cls={det['cls']} bbox=({x1},{y1},{x2},{y2}) "
            #     f"w={bbox_width_px}px h={bbox_height_px}px "
            #     f"center_depth={center_depth:.3f}m "
            #     f"fx={fx:.1f} fy={fy:.1f} "
            #     f"raw_r_w={bbox_width_px/2 * center_depth / fx:.4f} "
            #     f"raw_r_h={bbox_height_px/2 * center_depth / fy:.4f}", flush=True)


            half_width = (bbox_width_px / 2.0) * center_depth / fx
            half_height = (bbox_height_px / 2.0) * center_depth / fy
            radius = (half_width + half_height) / 2.0  # average of both axes
            radius = np.clip(radius, 0.01, 0.3)

            # --- Back-project bbox center pixel to world ---
            u_center = (x1 + x2) / 2.0
            v_center = (y1 + y2) / 2.0
            z_cam = center_depth 
            x_cam = (u_center - cx) * z_cam / fx
            y_cam = (v_center - cy) * z_cam / fy

            pt_cam_h = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float64)
            center_world = (E_inv @ pt_cam_h)[:3]

            obstacles.append({
                "center_world": center_world,
                "radius": float(radius),
                "bbox": det["bbox"],
                "cls": det["cls"],
                "conf": det["conf"],
                "track_id": det.get("track_id"),
            })

            logger.debug(
                "Obstacle '%s' (track=%s): centre=(%.3f, %.3f, %.3f) r=%.3f",
                det["cls"], det.get("track_id"),
                center_world[0], center_world[1], center_world[2], radius,
            )

        return obstacles
