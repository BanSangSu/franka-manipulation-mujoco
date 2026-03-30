"""
CameraManager – captures RGB, depth, intrinsic K, and extrinsic E from
all configured MuJoCo cameras (static, user_cam, side_cam).

Each camera output is a dict:
    {"rgb": (H,W,3 uint8), "depth": (H,W float32 metres),
     "K": (3,3), "E": (4,4), "cam_name": str}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class CameraFrame:
    """Container for a single camera's output."""

    cam_name: str
    rgb: np.ndarray           # (H, W, 3) uint8
    depth: np.ndarray         # (H, W) float32 metres
    intrinsic: np.ndarray     # (3, 3)
    extrinsic: np.ndarray     # (4, 4)  world-to-camera
    width: int = 0
    height: int = 0


class CameraManager:
    """Grabs frames from every camera in the MuJoCo simulation.

    Parameters
    ----------
    sim : MjSim
        An already-initialised MuJoCo simulation wrapper.
    camera_names : list[str] | None
        Names of cameras to capture.  ``None`` → use all 3 defaults
        (static, user_cam, side_cam).
    """

    # Camera names that ship in the default project config
    DEFAULT_CAMERAS = ("static", "user_cam", "side_cam")

    def __init__(self, sim: Any, camera_names: List[str] | None = None):
        self.sim = sim
        self.camera_names: List[str] = list(camera_names or self.DEFAULT_CAMERAS)
        # Pull default camera params from config
        base_cfg = dict(sim.cfg.get("mujoco", {}).get("camera", {}))
        self._default_width = int(base_cfg.get("width", 640))
        self._default_height = int(base_cfg.get("height", 480))
        self._default_near = float(base_cfg.get("near", 0.01))
        self._default_far = float(base_cfg.get("far", 5.0))
        self._default_fovy = float(base_cfg.get("fovy", 58.0))

    def _camera_params(self, cam_name: str) -> Dict[str, Any]:
        """Return per-camera (width, height, near, far, fovy), falling back to defaults."""
        specs = dict(self.sim.extra_specs.get(cam_name, {}))
        return {
            "width": int(specs.get("width", self._default_width)),
            "height": int(specs.get("height", self._default_height)),
            "near": float(specs.get("near", self._default_near)),
            "far": float(specs.get("far", self._default_far)),
            "fovy": float(specs.get("fovy", self._default_fovy)),
        }

    def capture_all(self) -> List[CameraFrame]:
        """Render every configured camera and return a list of CameraFrame."""
        frames: List[CameraFrame] = []
        for cam_name in self.camera_names:
            params = self._camera_params(cam_name)
            rgb, depth, K, E = self.sim.render_camera(
                cam_name,
                width=params["width"],
                height=params["height"],
                near=params["near"],
                far=params["far"],
                fovy=params["fovy"],
            )
            frames.append(
                CameraFrame(
                    cam_name=cam_name,
                    rgb=rgb,
                    depth=depth,
                    intrinsic=K,
                    extrinsic=E,
                    width=params["width"],
                    height=params["height"],
                )
            )
        return frames

    def capture_single(self, cam_name: str) -> CameraFrame:
        """Render a single camera by name."""
        params = self._camera_params(cam_name)
        rgb, depth, K, E = self.sim.render_camera(
            cam_name,
            width=params["width"],
            height=params["height"],
            near=params["near"],
            far=params["far"],
            fovy=params["fovy"],
        )
        return CameraFrame(
            cam_name=cam_name,
            rgb=rgb,
            depth=depth,
            intrinsic=K,
            extrinsic=E,
            width=params["width"],
            height=params["height"],
        )
