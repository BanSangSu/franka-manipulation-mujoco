"""
Obstacle Tracker – threaded real-time detection + 3D sphere + Kalman filter.

Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │  Main Thread                                             │
    │    1. Capture frame from MuJoCo static camera            │
    │    2. Submit the captured frame to the tracker           │
    └──────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────┐
    │  Background Thread (runs continuously)                   │
    │    1. Wait for latest submitted frame                    │
    │    2. YOLO26 .track(frame, persist=True) → detections    │
    │    3. BBox + depth → 3D sphere (centre + radius)         │
    │    4. Data association → Kalman filter update             │
    │    5. Store latest state under a threading.Lock           │
    └──────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────┐
    │  MPPI Controller (main thread)                           │
    │    Reads latest obstacle state (position, velocity,      │
    │    radius) via get_obstacle_predictions() — thread-safe  │
    └──────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .camera_manager import CameraManager, CameraFrame

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 3D Constant-Velocity Kalman Filter
# ──────────────────────────────────────────────────────────────────────
class KalmanFilter3D:
    """3D Constant-Velocity Kalman Filter.

    State x = [x, y, z, vx, vy, vz]^T
    Measurement z = [x, y, z]^T  (position only)
    """

    def __init__(
        self,
        dt: float = 0.02,
        process_noise: float = 0.5,
        measurement_noise: float = 0.02,
        initial_p: float = 1.0,
    ):
        self.dt = dt
        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)

        # Process noise (tuned for moving spheres with jitter)
        self.Q = np.zeros((6, 6))
        self.Q[:3, :3] = np.eye(3) * process_noise * (dt ** 2)
        self.Q[:3, 3:] = np.eye(3) * process_noise * dt
        self.Q[3:, :3] = np.eye(3) * process_noise * dt
        self.Q[3:, 3:] = np.eye(3) * process_noise

        self.R = np.eye(3) * measurement_noise
        self.P = np.eye(6) * initial_p

        self.state = np.zeros(6)
        self.initialized = False

    def reset(self, initial_pos: np.ndarray):
        self.state[:3] = initial_pos
        self.state[3:] = 0.0
        self.P = np.eye(6) * 1.0
        self.initialized = True

    def predict(self) -> np.ndarray:
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:3].copy()

    def update(self, measurement: np.ndarray):
        if not self.initialized:
            self.reset(measurement)
            return
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    @property
    def position(self) -> np.ndarray:
        return self.state[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:].copy()

    def predict_horizon(self, steps: int, dt: float) -> np.ndarray:
        preds = []
        curr = self.state.copy()
        F_h = np.eye(6)
        F_h[0, 3] = dt
        F_h[1, 4] = dt
        F_h[2, 5] = dt
        for _ in range(steps):
            curr = F_h @ curr
            preds.append(curr[:3].copy())
        return np.array(preds)


# ──────────────────────────────────────────────────────────────────────
# Tracked Obstacle
# ──────────────────────────────────────────────────────────────────────
class TrackedObstacle:
    """State for a single tracked obstacle."""

    def __init__(self, obs_id: int, dt: float = 0.02):
        self.obs_id = obs_id
        self.kf = KalmanFilter3D(dt=dt)
        self.radius: float = 0.06
        self.last_seen: int = 0
        self.cls: str = "unknown"
        self.track_id: Optional[int] = None  # YOLO track ID

    @property
    def position(self) -> np.ndarray:
        return self.kf.position

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.velocity

    def predict_horizon(self, steps: int, dt: float) -> np.ndarray:
        return self.kf.predict_horizon(steps, dt)


# ──────────────────────────────────────────────────────────────────────
# Threaded Obstacle Tracker
# ──────────────────────────────────────────────────────────────────────
class ObstacleTracker:
    """Tracks multiple dynamic obstacles in a background worker thread.

    The main thread:
        1. Captures a frame from the MuJoCo camera
        2. Submits the frame via ``capture_and_submit()`` or ``submit_frame()``

    The background thread continuously:
        1. Waits for the latest submitted frame
        2. Runs YOLO26 .track(frame, persist=True) for detections + tracking IDs
        3. Maps BBox → depth → 3D sphere (centre + radius)
        4. Associates detections with existing Kalman tracks
        5. Updates Kalman filters

    The MPPI controller reads the latest state via thread-safe accessors.

    Parameters
    ----------
    sim : MjSim
        The simulation (for camera access).
    yolo_model : str
        YOLO model path (e.g. "yolo26n.pt" or "yolo26n-seg.pt").
    detection_camera : str
        Camera name to use for detection.
    dt : float
        Time step for Kalman filter.
    max_association_dist : float
        Max distance (m) to associate detection with existing track.
    max_lost_frames : int
        Remove a track after this many frames without detection.
    device : str
        Device for YOLO inference.
    """

    def __init__(
        self,
        sim: Any,
        yolo_model: str = "yolo26n.pt",
        detection_camera: str = "static",
        dt: float = 0.02,
        max_association_dist: float = 0.3,
        max_lost_frames: int = 50,
        device: str = "cuda", # cpu, cuda
        visualize: bool = False,
    ):
        from .yolo_obstacle_detector import YOLOObstacleDetector

        self.sim = sim
        self.visualize = visualize

        self.detector = YOLOObstacleDetector(
            model_path=yolo_model,
            device=device,
        )
        self.detection_camera = detection_camera
        self.dt = dt
        self.max_association_dist = max_association_dist
        self.max_lost_frames = max_lost_frames

        # Camera manager for capturing frames
        self._cam_mgr = CameraManager(sim, [detection_camera])

        # Tracking state (protected by lock)
        self._lock = threading.Lock()
        self._tracks: Dict[int, TrackedObstacle] = {}
        self._next_id = 0
        self._frame_count = 0

        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_ready = threading.Condition()
        self._pending_frames: deque[CameraFrame] = deque(maxlen=1)

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------
    def start(self):
        """Start the background detection thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Tracker thread already running.")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._detection_loop,
            daemon=True,
            name="ObstacleTracker",
        )
        self._thread.start()
        logger.info(
            "ObstacleTracker started (camera=%s, model=%s)",
            self.detection_camera, self.detector.model_path,
        )

    def stop(self):
        """Stop the background detection thread."""
        self._running = False
        with self._frame_ready:
            self._frame_ready.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.detector.close_video_writer()
        logger.info("ObstacleTracker stopped.")

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Background detection loop
    # ------------------------------------------------------------------
    def submit_frame(self, frame: CameraFrame):
        """Submit a camera frame captured on the main thread for async processing."""
        frame_copy = CameraFrame(
            cam_name=frame.cam_name,
            rgb=frame.rgb.copy(),
            depth=frame.depth.copy(),
            intrinsic=frame.intrinsic.copy(),
            extrinsic=frame.extrinsic.copy(),
            width=frame.width,
            height=frame.height,
        )
        with self._frame_ready:
            self._pending_frames.clear()
            self._pending_frames.append(frame_copy)
            self._frame_ready.notify()

    def capture_and_submit(self):
        """Capture on the caller's thread, then hand the frame to the worker."""
        frame = self._cam_mgr.capture_single(self.detection_camera)
        self.submit_frame(frame)

    def _get_next_frame(self) -> Optional[CameraFrame]:
        with self._frame_ready:
            while self._running and not self._pending_frames:
                self._frame_ready.wait(timeout=0.1)
            if not self._running:
                return None
            return self._pending_frames.pop()

    def _detection_loop(self):
        """Runs in a background thread. Processes frames captured on the main thread."""
        logger.info("Detection loop started.")

        while self._running:
            try:
                frame = self._get_next_frame()
                if frame is None:
                    break

                # 1. YOLO track with persist=True for consistent track IDs
                detections = self.detector.detect(
                    frame.rgb, use_track=True,
                    # save_path="",
                    save_path="yolo_tracking_outputs",
                    video_save=self.visualize,
                )

                # 2. Estimate 3D spheres from BBox + depth
                spheres = self.detector.estimate_obstacle_spheres(
                    depth=frame.depth,
                    intrinsic=frame.intrinsic,
                    extrinsic=frame.extrinsic,
                    detections=detections,
                )

                # 3. Update tracking state (under lock)
                with self._lock:
                    self._update_tracks(spheres)

            except Exception as e:
                logger.error("Detection loop error: %s", e, exc_info=True)

            time.sleep(0.001)

        logger.info("Detection loop ended.")

    # ------------------------------------------------------------------
    # Track management (called under self._lock)
    # ------------------------------------------------------------------
    def _update_tracks(self, spheres: List[dict]):
        """Associate detections with existing tracks using the custom IDs (1=orange, 2=red)."""
        self._frame_count += 1

        # Predict all existing tracks forward
        for track in self._tracks.values():
            track.kf.predict()

        if not spheres:
            self._remove_stale_tracks()
            return

        # Group detections by custom ID and keep only the highest confidence one per ID
        # (ID 1 for orange, 2 for red)
        best_detections = {}
        for s in spheres:
            tid = s.get("track_id")
            if tid is None:
                continue
            if tid not in best_detections or s["conf"] > best_detections[tid]["conf"]:
                best_detections[tid] = s

        # Update or create tracks based on these IDs
        for tid, s in best_detections.items():
            if tid in self._tracks:
                # Update existing track
                track = self._tracks[tid]
                track.kf.update(s["center_world"])
                track.radius = s["radius"]
                track.last_seen = self._frame_count
                track.cls = s["cls"]
                logger.debug(
                    "Updated track #%d (%s) conf(%.3f): pos=(%.3f, %.3f, %.3f) r=%.3f",
                    tid, s["cls"], s["conf"], *track.position, track.radius
                )
            else:
                # Create new track using the detection ID as the track ID
                new_track = TrackedObstacle(tid, dt=self.dt)
                new_track.track_id = tid
                new_track.kf.reset(s["center_world"])
                new_track.radius = s["radius"]
                new_track.last_seen = self._frame_count
                new_track.cls = s["cls"]
                self._tracks[tid] = new_track
                logger.info(
                    "New ID-based track #%d (%s): pos=(%.3f, %.3f, %.3f) r=%.3f",
                    tid, s["cls"], *s["center_world"], s["radius"]
                )

        self._remove_stale_tracks()

    def _remove_stale_tracks(self):
        """Remove tracks not seen for too long."""
        stale = [
            tid for tid, t in self._tracks.items()
            if (self._frame_count - t.last_seen) > self.max_lost_frames
        ]
        for tid in stale:
            logger.info("Removing stale track #%d", tid)
            del self._tracks[tid]

    # ------------------------------------------------------------------
    # Thread-safe accessors for MPPI controller
    # ------------------------------------------------------------------
    def get_obstacle_predictions(self, horizon: int, dt: float) -> np.ndarray:
        """Get predicted obstacle trajectories for the MPPI horizon.

        Thread-safe. Called from the main MPPI thread.

        Returns
        -------
        predictions : (T, n_obs, 3)
        """
        with self._lock:
            if not self._tracks:
                return np.empty((horizon, 0, 3))
            preds = []
            for track in self._tracks.values():
                pred = track.predict_horizon(horizon, dt)
                preds.append(pred)
            return np.stack(preds, axis=1)

    def get_obstacle_radii(self) -> np.ndarray:
        """Get radii of all tracked obstacles.  Thread-safe.

        Returns
        -------
        radii : (n_obs,)
        """
        with self._lock:
            if not self._tracks:
                return np.empty((0,))
            return np.array([t.radius for t in self._tracks.values()])

    def get_obstacle_states(self) -> List[dict]:
        """Get current obstacle states.  Thread-safe.

        Returns
        -------
        list[dict] each with 'position', 'velocity', 'radius'.
        """
        with self._lock:
            return [
                {
                    "position": t.position,
                    "velocity": t.velocity,
                    "radius": t.radius,
                }
                for t in self._tracks.values()
            ]

    @property
    def num_tracked(self) -> int:
        with self._lock:
            return len(self._tracks)
