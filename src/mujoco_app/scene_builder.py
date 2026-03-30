"""Scene construction utilities for the MuJoCo simulation."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import mujoco
import numpy as np
import trimesh

from mujoco_app.transformations import (
    camera_xyaxes,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    rpy_to_quat_wxyz,
)


@dataclass
class RobotInfo:
    xml_path: Path
    base_dir: Path
    extra_specs: Dict[str, dict]


@dataclass
class SceneArtifacts:
    """Container for the compiled MuJoCo scene and associated metadata.

    Attributes:
        model: The compiled MuJoCo model.
        data: MuJoCo data instance initialized from the model.
        ids: Dictionary mapping semantic names to MuJoCo object IDs or other metadata.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    ids: Dict[str, object]


class SceneBuilder:
    """Builds the MuJoCo scene graph from a configuration dictionary.

    Programmatically constructs MuJoCo XML from YAML configuration, adding
    the robot, table, cameras, objects, and moving obstacles. Returns a
    compiled SceneArtifacts object with model, data, and semantic ID mappings.

    Attributes:
        cfg: Full configuration dictionary.
        mcfg: Subset of cfg under the 'mujoco' key.
        cfg_dir: Directory path of the loaded configuration file.
        root: Root XML element for the MuJoCo model.
        asset: XML 'asset' element for meshes and textures.
        world: XML 'worldbody' element for spatial scene graph.
        ids: Dict mapping semantic names to MuJoCo IDs or metadata.
        freejoint_initializers: List of (body_name, pos, quat) for free-joint bodies.
        object_names: List of object body names added to the scene.
    """

    def __init__(self, cfg: dict):
        """Initializes the scene builder with a configuration dictionary.

        Args:
            cfg: Configuration dict with keys like 'mujoco', 'table', 'robot_settings', etc.
        """
        self.cfg = cfg
        self.mcfg: dict = dict(cfg.get("mujoco", {}))
        self.obstacle_toggle = self.mcfg.get("obstacle_toggle", False)
        self.cfg_dir = Path(cfg.get("_config_dir", "configs")).resolve()
        self.root = ET.Element(
            "mujoco",
            attrib={"model": self.mcfg.get("model_name", "irobman_scene")},
        )
        ET.SubElement(self.root, "option", attrib=self._option_attributes())
        self.asset = ET.SubElement(self.root, "asset")
        self.world = ET.SubElement(self.root, "worldbody")
        self.ids: Dict[str, object] = {}
        self.freejoint_initializers: List[
            Tuple[str, np.ndarray, np.ndarray]
        ] = []
        self.object_names: List[str] = []
        # store metadata for moving obstacles to resolve qpos addresses
        self._moving_obstacles_meta: List[dict] = []

    def build(self) -> SceneArtifacts:
        """Constructs the complete MuJoCo scene from the configuration.

        Sequentially adds ground, table, lights, robot, cameras, basket, objects,
        and moving obstacles, then finalizes and compiles the XML into a MuJoCo model.

        Returns:
            A SceneArtifacts object containing the compiled model, data, and IDs.
        """
        self._add_ground_plane()
        table_center = self._add_table()
        self._add_lights()
        robot_info = self._add_robot(table_center)
        self._add_primary_camera(table_center)
        self._add_user_camera(table_center)
        self._add_extra_cameras(table_center)
        self._add_basket(table_center)
        self._add_objects(table_center)
        self._add_grasp_object(table_center)
        # moving obstacles last
        if self.obstacle_toggle:
            self._add_moving_obstacles(table_center)

        model, data = self._finalize_model(robot_info.base_dir)
        self.ids.setdefault("extra_camera_specs", robot_info.extra_specs)
        return SceneArtifacts(model=model, data=data, ids=self.ids)

    # ------------------------------------------------------------------
    # Option helpers
    # ------------------------------------------------------------------
    def _option_attributes(self) -> Dict[str, str]:
        timestep = float(self.mcfg.get("timestep", 0.002))
        gravity = " ".join(
            str(v) for v in self.mcfg.get("gravity", [0.0, 0.0, -9.81])
        )
        integrator = self.mcfg.get("integrator", "implicitfast")
        iterations = int(self.mcfg.get("iterations", 100))
        tolerance = float(self.mcfg.get("tolerance", 1e-10))
        return {
            "timestep": f"{timestep:g}",
            "gravity": gravity,
            "integrator": integrator,
            "iterations": str(iterations),
            "tolerance": f"{tolerance:g}",
        }

    # ------------------------------------------------------------------
    # Geometry components
    # ------------------------------------------------------------------
    def _add_ground_plane(self) -> None:
        ET.SubElement(
            self.world,
            "geom",
            attrib={
                "name": "ground",
                "type": "plane",
                "size": "3 3 0.1",
                "rgba": "0.9 0.9 0.9 1",
                "contype": "1",
                "conaffinity": "1",
                "condim": "3",
                "friction": "1.0 0.005 0.0001",
                "solimp": "0.9 0.99 0.001",
                "solref": "0.02 1",
            },
        )

    def _add_table(self) -> np.ndarray:
        cfg = dict(self.cfg.get("table", {}))
        size = np.asarray(cfg.get("size", [0.35, 0.55, 0.02]), dtype=float)
        pos = np.asarray(cfg.get("pos", [0.6, 0.0, 0.7]), dtype=float)
        attrs = {"name": "table", "pos": f"{pos[0]} {pos[1]} {pos[2]}"}
        if not bool(cfg.get("gravity", True)):
            attrs["gravcomp"] = "1"
        table_body = ET.SubElement(self.world, "body", attrib=attrs)
        ET.SubElement(
            table_body,
            "geom",
            attrib={
                "type": "box",
                "size": f"{size[0]} {size[1]} {size[2]}",
                "rgba": cfg.get("rgba", "0.8 0.7 0.6 1"),
                "contype": "1",
                "conaffinity": "1",
                "condim": "3",
                "friction": cfg.get("friction", "1.0 0.005 0.0001"),
                "solimp": cfg.get("solimp", "0.9 0.99 0.001"),
                "solref": cfg.get("solref", "0.02 1"),
            },
        )
        self._add_table_feet(table_body, cfg, size)
        center = np.array([pos[0], pos[1], pos[2] + size[2]], dtype=float)
        self.ids["table_center"] = center
        self.ids["table_top"] = pos[2] + size[2]
        self.ids["table_size"] = size
        return center

    def _add_table_feet(
        self, table_body: ET.Element, cfg: dict, size: np.ndarray
    ) -> None:
        foot_cfg = cfg.get("feet")
        if not foot_cfg:
            return
        foot_size = np.asarray(
            foot_cfg.get("size", [0.035, 0.035, 0.35]), dtype=float
        )
        rgba = foot_cfg.get("rgba", "0.3 0.25 0.2 1")
        offsets = foot_cfg.get("offsets")
        if offsets is None:
            offsets = [
                [size[0] - foot_size[0], size[1] - foot_size[1]],
                [size[0] - foot_size[0], -size[1] + foot_size[1]],
                [-size[0] + foot_size[0], size[1] - foot_size[1]],
                [-size[0] + foot_size[0], -size[1] + foot_size[1]],
            ]
        for idx, (ox, oy) in enumerate(offsets):
            pos = [ox, oy, -(size[2] + foot_size[2])]
            ET.SubElement(
                table_body,
                "geom",
                attrib={
                    "name": f"table_leg_{idx}",
                    "type": "box",
                    "size": f"{foot_size[0]} {foot_size[1]} {foot_size[2]}",
                    "pos": f"{pos[0]} {pos[1]} {pos[2]}",
                    "rgba": rgba,
                    "contype": "1",
                    "conaffinity": "1",
                    "condim": "3",
                    "friction": cfg.get("friction", "1.0 0.005 0.0001"),
                    "solimp": cfg.get("solimp", "0.9 0.99 0.001"),
                    "solref": cfg.get("solref", "0.02 1"),
                },
            )

    # ------------------------------------------------------------------
    # Lights and cameras
    # ------------------------------------------------------------------
    def _add_lights(self) -> None:
        for lcfg in self.mcfg.get("lights", []):
            attrib = {
                "name": lcfg.get("name", "scene_light"),
                "pos": "{} {} {}".format(*lcfg.get("pos", [0.6, -0.5, 1.6])),
                "dir": "{} {} {}".format(*lcfg.get("dir", [0.0, 0.5, -1.0])),
                "diffuse": "{} {} {}".format(
                    *lcfg.get("diffuse", [1.0, 1.0, 1.0])
                ),
                "specular": "{} {} {}".format(
                    *lcfg.get("specular", [0.2, 0.2, 0.2])
                ),
                "attenuation": "{} {} {}".format(
                    *lcfg.get("attenuation", [1.0, 0.0, 0.0])
                ),
            }
            if "cutoff" in lcfg:
                attrib["cutoff"] = f"{float(lcfg['cutoff']):g}"
            if "exponent" in lcfg:
                attrib["exponent"] = f"{float(lcfg['exponent']):g}"
            ET.SubElement(self.world, "light", attrib=attrib)

    def _add_primary_camera(self, target_hint: np.ndarray) -> None:
        cam_cfg = dict(self.mcfg.get("camera", {}))
        if not cam_cfg:
            raise ValueError("Primary camera configuration is required")
        pos = np.asarray(
            cam_cfg.get("pos", target_hint + np.array([0.0, -0.8, 0.8])),
            dtype=float,
        )
        xyaxes = self._camera_xyaxes(cam_cfg, pos, target_hint)
        cam_body = ET.SubElement(
            self.world,
            "body",
            attrib={
                "name": cam_cfg.get("body_name", "static_cam_body"),
                "pos": f"{pos[0]} {pos[1]} {pos[2]}",
            },
        )
        ET.SubElement(
            cam_body,
            "camera",
            attrib={
                "name": cam_cfg.get("name", "static"),
                "mode": cam_cfg.get("mode", "fixed"),
                "xyaxes": xyaxes,
                "fovy": f"{float(cam_cfg.get('fovy', 58.0)):g}",
            },
        )
        cam_name = cam_cfg.get("name", "static")
        self.ids["cam_name"] = cam_name

    def _add_user_camera(self, target_hint: np.ndarray) -> None:
        user_cfg = dict(self.mcfg.get("user_camera", {}))
        if not user_cfg or not user_cfg.get("enable", True):
            return
        pos = np.asarray(
            user_cfg.get("pos", target_hint + np.array([0.0, -1.0, 1.0])),
            dtype=float,
        )
        xyaxes = self._camera_xyaxes(user_cfg, pos, target_hint)
        body_name = user_cfg.get(
            "body_name", f"{user_cfg.get('name', 'user_cam')}_body"
        )
        cam_body = ET.SubElement(
            self.world,
            "body",
            attrib={"name": body_name, "pos": f"{pos[0]} {pos[1]} {pos[2]}"},
        )
        cam_name = user_cfg.get("name", "user_cam")
        ET.SubElement(
            cam_body,
            "camera",
            attrib={
                "name": cam_name,
                "mode": user_cfg.get("mode", "fixed"),
                "xyaxes": xyaxes,
                "fovy": f"{float(user_cfg.get('fovy', self.mcfg.get('camera', {}).get('fovy', 58.0))):g}",
            },
        )
        self.ids.setdefault("extra_cameras", {})[cam_name] = None

    def _add_extra_cameras(self, target_hint: np.ndarray) -> None:
        extras_cfg = self.mcfg.get("extra_cameras", [])
        if not extras_cfg:
            return
        specs: Dict[str, dict] = {}
        for ecfg in extras_cfg:
            pos = np.asarray(
                ecfg.get("pos", target_hint + np.array([0.0, -0.5, 0.6])),
                dtype=float,
            )
            xyaxes = self._camera_xyaxes(ecfg, pos, target_hint)
            body_name = ecfg.get(
                "body_name", f"{ecfg.get('name', 'extra_cam')}_body"
            )
            cam_body = ET.SubElement(
                self.world,
                "body",
                attrib={
                    "name": body_name,
                    "pos": f"{pos[0]} {pos[1]} {pos[2]}",
                },
            )
            cam_name = ecfg.get("name", "extra_cam")
            ET.SubElement(
                cam_body,
                "camera",
                attrib={
                    "name": cam_name,
                    "mode": ecfg.get("mode", "fixed"),
                    "xyaxes": xyaxes,
                    "fovy": f"{float(ecfg.get('fovy', self.mcfg.get('camera', {}).get('fovy', 58.0))):g}",
                },
            )
            specs[cam_name] = {
                "fovy": float(
                    ecfg.get(
                        "fovy", self.mcfg.get("camera", {}).get("fovy", 58.0)
                    )
                ),
                "near": float(
                    ecfg.get(
                        "near", self.mcfg.get("camera", {}).get("near", 0.01)
                    )
                ),
                "far": float(
                    ecfg.get("far", self.mcfg.get("camera", {}).get("far", 5.0))
                ),
            }
            self.ids.setdefault("extra_cameras", {})[cam_name] = None
        self.ids["extra_camera_specs"] = specs

    def _camera_xyaxes(
        self, cam_cfg: dict, pos: np.ndarray, target_hint: np.ndarray
    ) -> str:
        if "xyaxes" in cam_cfg:
            axis = cam_cfg["xyaxes"]
            if isinstance(axis, str):
                return axis
            flat = " ".join(str(v) for v in axis)
            return flat
        if "target" in cam_cfg:
            target = np.asarray(cam_cfg["target"], dtype=float)
        else:
            target = target_hint
        xyaxes, _ = camera_xyaxes(pos, target, (0.0, 0.0, -1.0))
        return xyaxes

    def _add_robot(self, table_center: np.ndarray) -> RobotInfo:
        robot_settings = dict(self.cfg.get("robot_settings", {}))
        base_body = robot_settings.get(
            "robot_base_body", self.mcfg.get("robot_base_body", "link0")
        )
        ee_body = robot_settings.get(
            "ee_body_name", self.mcfg.get("ee_body_name", "hand")
        )
        base_margin = float(
            robot_settings.get("base_margin", self.mcfg.get("base_margin", 0.1))
        )
        base_yaw = float(
            robot_settings.get("base_yaw", self.mcfg.get("base_yaw", 0.0))
        )
        table_size = self.ids["table_size"]
        base_pos = np.array(
            [
                table_center[0] - table_size[0] + base_margin,
                table_center[1],
                table_center[2],
            ]
        )

        robot_xml_path = self._resolve_path(
            robot_settings.get("robot_xml", self.mcfg.get("robot_xml"))
        )
        patched_xml = self._patch_robot_xml(
            robot_xml_path,
            base_body=base_body,
            base_position=base_pos,
            base_yaw=base_yaw,
            ee_body=ee_body,
            wrist_cfg=self.mcfg.get("wrist_camera", {}),
        )
        ET.SubElement(self.root, "include", attrib={"file": str(patched_xml)})
        return RobotInfo(
            xml_path=patched_xml, base_dir=patched_xml.parent, extra_specs={}
        )

    def _patch_robot_xml(
        self,
        xml_path: Path,
        base_body: str,
        base_position: np.ndarray,
        base_yaw: float,
        ee_body: str,
        wrist_cfg: dict,
    ) -> Path:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(root, "compiler")
        compiler.set("meshdir", str(xml_path.parent / "assets"))
        world = root.find("worldbody")
        if world is None:
            raise ValueError("Robot XML missing worldbody element")
        base = world.find(f".//body[@name='{base_body}']")
        if base is None:
            raise ValueError(f"Base body '{base_body}' not found in robot XML")
        base.set(
            "pos", f"{base_position[0]} {base_position[1]} {base_position[2]}"
        )
        if abs(base_yaw) > 1e-9:
            quat = rpy_to_quat_wxyz((0.0, 0.0, base_yaw))
            base.set("quat", f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")
        if wrist_cfg.get("enable", False):
            ee = world.find(f".//body[@name='{ee_body}']")
            if ee is None:
                raise ValueError(
                    f"End-effector body '{ee_body}' missing for wrist camera attachment"
                )
            cam_name = wrist_cfg.get("name", "wrist_cam")
            cam_pos = wrist_cfg.get("pos", [0.0, 0.0, 0.05])
            xyaxes = wrist_cfg.get("xyaxes", "1 0 0 0 -1 0")
            fovy = wrist_cfg.get(
                "fovy", self.mcfg.get("camera", {}).get("fovy", 58.0)
            )
            ET.SubElement(
                ee,
                "camera",
                attrib={
                    "name": cam_name,
                    "pos": "{} {} {}".format(*cam_pos),
                    "xyaxes": xyaxes
                    if isinstance(xyaxes, str)
                    else " ".join(str(v) for v in xyaxes),
                    "fovy": f"{float(fovy):g}",
                },
            )
        patched_path = xml_path.parent / "_irobman_robot_patched.xml"
        tree.write(patched_path)
        return patched_path

    # ------------------------------------------------------------------
    # Add basket and objects
    # ------------------------------------------------------------------
    def _add_basket(self, table_center: np.ndarray) -> None:
        basket_cfg = dict(self.cfg.get("basket", {}))
        if not basket_cfg:
            return
        inner = np.asarray(basket_cfg.get("inner", [0.22, 0.18]), dtype=float)
        height = float(basket_cfg.get("height", 0.06))
        thickness = float(basket_cfg.get("thickness", 0.005))
        place = basket_cfg.get("place", "table_right")
        pos = np.asarray(basket_cfg.get("pos", None), dtype=float)
        margin = float(basket_cfg.get("margin", 0.02))
        table_size = self.ids["table_size"]
        center = table_center.copy()
        if pos is not None:
            center = pos.copy()
        elif place == "table_right":
            center[0] = (
                table_center[0] + table_size[0] - inner[0] / 2.0 - margin
            )
        elif place == "table_left":
            center[0] = (
                table_center[0] - table_size[0] + inner[0] / 2.0 + margin
            )
        center[2] = table_center[2] + height / 2.0
        body_attrs = {
            "name": basket_cfg.get("name", "basket"),
            "pos": f"{center[0]} {center[1]} {center[2]}",
        }
        if not bool(basket_cfg.get("gravity", True)):
            body_attrs["gravcomp"] = "1"
        basket = ET.SubElement(self.world, "body", attrib=body_attrs)
        rgba_base = basket_cfg.get("rgba_base", "0.25 0.25 0.25 1")
        rgba_wall = basket_cfg.get("rgba_wall", "0.3 0.3 0.3 1")
        ET.SubElement(
            basket,
            "geom",
            attrib={
                "type": "box",
                "size": f"{inner[0] / 2.0} {inner[1] / 2.0} {thickness / 2.0}",
                "pos": f"0 0 {-height / 2.0 + thickness / 2.0}",
                "rgba": rgba_base,
            },
        )
        wall_specs = [
            (
                inner[0] / 2.0,
                thickness / 2.0,
                "0 {} 0".format(inner[1] / 2.0 - thickness / 2.0),
            ),
            (
                inner[0] / 2.0,
                thickness / 2.0,
                "0 {} 0".format(-inner[1] / 2.0 + thickness / 2.0),
            ),
            (
                thickness / 2.0,
                inner[1] / 2.0,
                "{} 0 0".format(inner[0] / 2.0 - thickness / 2.0),
            ),
            (
                thickness / 2.0,
                inner[1] / 2.0,
                "{} 0 0".format(-inner[0] / 2.0 + thickness / 2.0),
            ),
        ]
        for half_x, half_y, pos_str in wall_specs:
            ET.SubElement(
                basket,
                "geom",
                attrib={
                    "type": "box",
                    "size": f"{half_x} {half_y} {height / 2.0}",
                    "pos": pos_str,
                    "rgba": rgba_wall,
                },
            )
        self.ids["basket_center"] = center
        self.ids["basket_dims"] = inner
        self.ids["basket_height"] = height
        self.ids["basket_thickness"] = thickness

    def _add_objects(self, table_center: np.ndarray) -> None:
        objects = self.mcfg.get("ycb_objects", [])
        if not objects:
            return
        for obj_cfg in objects:
            name = obj_cfg["name"]
            body = self._spawn_primitive_object(name, obj_cfg, table_center)
            if obj_cfg.get("dynamic", True):
                joint_name = f"{name}_free"
                ET.SubElement(body, "freejoint", attrib={"name": joint_name})
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                pos = self._object_resting_position(obj_cfg, table_center)
                self.freejoint_initializers.append((joint_name, pos, quat))
            self.object_names.append(name)

    def _spawn_primitive_object(
        self, name: str, obj_cfg: dict, table_center: np.ndarray
    ) -> ET.Element:
        pos = self._object_resting_position(obj_cfg, table_center)
        body_attrs = {"name": name, "pos": f"{pos[0]} {pos[1]} {pos[2]}"}
        if not bool(obj_cfg.get("gravity", True)):
            body_attrs["gravcomp"] = "1"
        body = ET.SubElement(self.world, "body", attrib=body_attrs)
        rgba = obj_cfg.get("rgba", "1 0 0 1")
        geom_kwargs = {
            "name": f"{name}_geom",
            "rgba": rgba,
            "contype": "1",
            "conaffinity": "1",
            "condim": "3",
            "friction": obj_cfg.get("friction", "1.0 0.01 0.001"),
            "solimp": obj_cfg.get("solimp", "0.95 0.99 0.001"),
            "solref": obj_cfg.get("solref", "0.02 1"),
        }
        if obj_cfg.get("mesh"):
            mesh_name = f"{name}_mesh"
            mesh_path = self._resolve_path(obj_cfg["mesh"])
            ET.SubElement(
                self.asset,
                "mesh",
                attrib={"name": mesh_name, "file": str(mesh_path)},
            )
            geom_kwargs.update({"type": "mesh", "mesh": mesh_name})
        else:
            primitive = obj_cfg.get("type", obj_cfg.get("primitive", "sphere"))
            if primitive == "box":
                size = obj_cfg.get("size", [0.03, 0.03, 0.03])
                geom_kwargs.update(
                    {"type": "box", "size": "{} {} {}".format(*size)}
                )
            elif primitive == "capsule":
                size = obj_cfg.get("size", [0.02, 0.06])
                geom_kwargs.update(
                    {"type": "capsule", "size": "{} {}".format(*size)}
                )
            else:
                radius = float(obj_cfg.get("size", 0.03))
                geom_kwargs.update({"type": "sphere", "size": f"{radius}"})
        ET.SubElement(body, "geom", attrib=geom_kwargs)
        return body

    # ------------------------------------------------------------------
    # Moving obstacles (free bodies animated by the simulator)
    # ------------------------------------------------------------------
    def _add_moving_obstacles(self, table_center: np.ndarray) -> None:
        cfg_list = list(self.mcfg.get("moving_obstacles", []))
        # If none specified, create two sensible defaults
        if not cfg_list:
            table_top = float(self.ids.get("table_top", table_center[2]))
            table_size = np.asarray(
                self.ids.get("table_size", [0.4, 0.6, 0.02]), dtype=float
            )
            default_half = np.array(
                [0.06, 0.06, 0.06], dtype=float
            )  # Larger spheres for reliable collision testing
            travel_margin = 0.02  # keep a small clearance from table edges
            # Move red sphere center to right side, away from robot base
            lr_center = [
                float(table_center[0] + 0.15),
                float(table_center[1]),
                float(table_top + 0.25),
            ]
            # yellow obstacle sweeps across table width (y-axis) near the user edge
            height_offset = 0.35
            tb_center = [
                float(table_center[0]),
                float(table_center[1]),
                float(table_top + height_offset),
            ]
            lr_center[2] = float(table_top + height_offset)
            tb_amplitude = float(
                max(0.0, table_size[1] - default_half[1] - travel_margin)
            )
            cfg_list = [
                {
                    "name": "obstacle_lr",
                    "size": default_half.tolist(),  # sphere radius
                    "rgba": "1.0 0.0 0.0 1",  # red
                    "center": lr_center,
                    "axis": "x",
                    "amplitude": 0.12,  # Reduced to stay away from robot base
                    "frequency": 0.35,  # Faster movement
                    "phase": 0.0,
                    "gravity": False,
                    "jitter_scale": 0.0,  # meters
                    "jitter_smooth": 1.0,  # closer to 1 = smoother
                },
                {
                    "name": "obstacle_tb",
                    "size": default_half.tolist(),  # sphere radius
                    "rgba": "1.0 0.5 0.0 1",  # orange
                    "center": tb_center,
                    "axis": "y",
                    "amplitude": tb_amplitude * 1.2,  # 20% more range
                    "frequency": 0.30,  # Faster movement
                    "phase": 0.0,
                    "gravity": False,
                    "jitter_scale": 0.0,  # meters
                    "jitter_smooth": 1.0,  # closer to 1 = smoother
                },
            ]

        for ocfg in cfg_list:
            name = ocfg.get(
                "name", f"obstacle_{len(self._moving_obstacles_meta)}"
            )
            half = np.asarray(ocfg.get("size", [0.05, 0.05, 0.01]), dtype=float)
            center = np.asarray(
                ocfg.get(
                    "center",
                    [table_center[0], table_center[1], table_center[2] + 0.25],
                ),
                dtype=float,
            )
            axis = str(ocfg.get("axis", "x")).lower()
            amp = float(ocfg.get("amplitude", 0.15))
            freq = float(ocfg.get("frequency", 0.25))
            phase = float(ocfg.get("phase", 0.0))
            rgba = ocfg.get("rgba", "0.2 0.8 0.2 1")
            jitter_scale = float(ocfg.get("jitter_scale", 0.0))
            jitter_smooth = float(ocfg.get("jitter_smooth", 1.0))

            body_attrs = {
                "name": name,
                "pos": f"{center[0]} {center[1]} {center[2]}",
            }
            if not bool(ocfg.get("gravity", True)):
                body_attrs["gravcomp"] = "1"
            body = ET.SubElement(self.world, "body", attrib=body_attrs)
            ET.SubElement(
                body,
                "geom",
                attrib={
                    "name": f"{name}_geom",
                    "type": "sphere",
                    "size": f"{half[0]}",
                    "rgba": rgba,
                    "contype": "1",
                    "conaffinity": "1",
                    "condim": "3",
                    "friction": ocfg.get("friction", "1.0 0.01 0.001"),
                    "solimp": ocfg.get("solimp", "0.95 0.99 0.001"),
                    "solref": ocfg.get("solref", "0.02 1"),
                },
            )
            jname = f"{name}_free"
            ET.SubElement(body, "freejoint", attrib={"name": jname})
            # initialize pose later in _finalize_model
            quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            self.freejoint_initializers.append(
                (jname, center.copy(), quat_wxyz.copy())
            )
            # remember animation params for runtime
            self._moving_obstacles_meta.append(
                {
                    "name": name,
                    "joint": jname,
                    "center": center.copy(),
                    "axis": axis,
                    "amplitude": amp,
                    "frequency": freq,
                    "phase": phase,
                    "jitter_scale": jitter_scale,
                    "jitter_smooth": jitter_smooth,
                }
            )

    def _object_resting_position(
        self, obj_cfg: dict, table_center: np.ndarray
    ) -> np.ndarray:
        if "pos" in obj_cfg:
            pos = np.asarray(obj_cfg["pos"], dtype=float)
            return pos
        table_top = self.ids["table_top"]
        placement = obj_cfg.get("place", "table_center")
        horizontal = np.array([table_center[0], table_center[1]], dtype=float)
        table_size = self.ids["table_size"]
        margin = float(obj_cfg.get("margin", 0.02))
        clearance = self._object_clearance(obj_cfg)
        # small spawn offsetfor preventing initial collision with table
        spawn_offset = float(
            obj_cfg.get("spawn_offset", 0.010)
        )  # 5mm default offset
        if placement == "table_right":
            horizontal[0] = table_center[0] + table_size[0] - clearance - margin
        elif placement == "table_left":
            horizontal[0] = table_center[0] - table_size[0] + clearance + margin
        elif placement == "basket_center" and "basket_center" in self.ids:
            center = self.ids["basket_center"]
            height = self.ids["basket_height"]
            thickness = self.ids["basket_thickness"]
            return np.array(
                [
                    center[0],
                    center[1],
                    center[2]
                    - height / 2.0
                    + thickness
                    + clearance
                    + spawn_offset,
                ]
            )
        z = table_top + clearance + spawn_offset
        return np.array([horizontal[0], horizontal[1], z])

    def _object_clearance(self, obj_cfg: dict) -> float:
        if obj_cfg.get("mesh"):
            mesh_path = self._resolve_path(obj_cfg["mesh"])
            mesh = trimesh.load_mesh(mesh_path, process=False)
            bounds = mesh.bounds
            extent = bounds[1] - bounds[0]
            return float(extent[2] * 0.5)
        primitive = obj_cfg.get("type", obj_cfg.get("primitive", "sphere"))
        if primitive == "box":
            size = np.asarray(
                obj_cfg.get("size", [0.03, 0.03, 0.03]), dtype=float
            )
            return float(size[2])
        if primitive == "capsule":
            size = obj_cfg.get("size", [0.02, 0.06])
            return float(size[1] * 0.5)
        return float(obj_cfg.get("size", 0.03))

    def _add_grasp_object(self, table_center: np.ndarray) -> None:
        grasp_cfg = self.mcfg.get("grasp_object")
        if not grasp_cfg:
            return
        xml_path = self._resolve_path(grasp_cfg["xml"])
        pos = np.asarray(
            grasp_cfg.get(
                "pos", self._object_resting_position(grasp_cfg, table_center)
            ),
            dtype=float,
        )
        xy_jitter = np.random.uniform(-0.2, 0.2, size=2)
        pos[:2] += xy_jitter
        quat_xyzw = grasp_cfg.get("quat_xyzw")
        if quat_xyzw is None and "rpy" in grasp_cfg:
            quat = rpy_to_quat_wxyz(grasp_cfg["rpy"])
            quat_xyzw = quat_wxyz_to_xyzw(quat)
        if quat_xyzw is None:
            quat_xyzw = [0.0, 0.0, 0.0, 1.0]
        body_name = grasp_cfg.get("body", grasp_cfg.get("name", "grasp_object"))
        patched = self._patch_object_xml(xml_path, body_name, pos, quat_xyzw)
        ET.SubElement(self.root, "include", attrib={"file": str(patched)})
        self.object_names.append(body_name)
        self.ids.setdefault("grasp_object", {})["body_name"] = body_name
        self.ids.setdefault("grasp_object", {})["pos"] = pos
        self.ids.setdefault("grasp_object", {})["quat_xyzw"] = np.asarray(
            quat_xyzw, dtype=float
        )
        init_cfg = grasp_cfg.get("initial_pose", {})
        if init_cfg:
            joint_name = init_cfg.get("freejoint")
            if joint_name:
                quat_wxyz = rpy_to_quat_wxyz(
                    init_cfg.get("rpy", (0.0, 0.0, 0.0))
                )
                self.freejoint_initializers.append((joint_name, pos, quat_wxyz))

    # ------------------------------------------------------------------
    # Finalize model
    # ------------------------------------------------------------------
    def _finalize_model(
        self, base_dir: Path
    ) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        xml_str = ET.tostring(self.root, encoding="unicode")
        model = mujoco.MjModel.from_xml_string(xml_str)
        if hasattr(model.opt, "collision"):
            if hasattr(mujoco, "mjtCollision") and hasattr(
                mujoco.mjtCollision, "mjCOL_ALL"
            ):
                model.opt.collision = mujoco.mjtCollision.mjCOL_ALL
            else:
                model.opt.collision = 2
        data = mujoco.MjData(model)
        for joint_name, pos, quat in self.freejoint_initializers:
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if jid < 0:
                continue
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            qadr = model.jnt_qposadr[jid]
            dad = model.jnt_dofadr[jid]
            data.qpos[qadr : qadr + 7] = np.array(
                [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]
            )
            data.qvel[dad : dad + 6] = 0.0
        mujoco.mj_forward(model, data)
        self.ids["cam_id"] = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            self.ids.get("cam_name", "static"),
        )
        if self.object_names:
            obj_ids: Dict[str, int] = {}
            for name in self.object_names:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid >= 0:
                    obj_ids[name] = bid
            if obj_ids:
                self.ids["obj_bodies"] = obj_ids
        if "extra_cameras" in self.ids:
            extras = {}
            for name in list(self.ids["extra_cameras"].keys()):
                cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                extras[name] = cid
            self.ids["extra_cameras"] = extras
        # resolve moving obstacle joint qpos addresses
        if self._moving_obstacles_meta:
            mobjs: Dict[str, dict] = {}
            for meta in self._moving_obstacles_meta:
                jname = meta["joint"]
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    continue
                if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                    continue
                qadr = int(model.jnt_qposadr[jid])
                mobjs[meta["name"]] = {
                    "qadr": qadr,
                    "center": np.asarray(meta["center"], dtype=float),
                    "axis": meta["axis"],
                    "amplitude": float(meta["amplitude"]),
                    "frequency": float(meta["frequency"]),
                    "phase": float(meta["phase"]),
                    "jitter_scale": float(meta["jitter_scale"]),
                    "jitter_smooth": float(meta["jitter_smooth"]),
                }
            if mobjs:
                self.ids["moving_obstacles"] = mobjs
        return model, data

    # ------------------------------------------------------------------
    # Path helper
    # ------------------------------------------------------------------
    def _resolve_path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        candidate = path
        if candidate.exists():
            return candidate
        cfg_candidate = (self.cfg_dir / path).resolve()
        if cfg_candidate.exists():
            return cfg_candidate
        raise FileNotFoundError(f"Cannot resolve path: {relative}")

    def _patch_object_xml(
        self,
        xml_path: Path,
        body_name: str,
        pos: np.ndarray,
        quat_xyzw: Iterable[float],
    ) -> Path:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        world = root.find("worldbody")
        if world is None:
            raise ValueError("Object XML missing worldbody element")
        body = world.find(f".//body[@name='{body_name}']")
        if body is None:
            raise ValueError(f"Body '{body_name}' not found in object XML")
        quat_wxyz = quat_xyzw_to_wxyz(quat_xyzw)
        body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
        body.set("quat", "{} {} {} {}".format(*quat_wxyz))
        # Include source filename to avoid collisions when different objects share the same body name
        for geom in body.findall(".//geom"):
            group = geom.get("group")
            if group == "3":
                geom.set("friction", "2.0 0.01 0.001")
                geom.set("solimp",   "0.99 0.995 0.01 0.5 2.0")
                geom.set("solref",   "0.01 1.0")
                geom.set("contype",     geom.get("contype",     "1"))
                geom.set("conaffinity", geom.get("conaffinity", "1"))
            elif group == "2":
                geom.set("contype",     "0")
                geom.set("conaffinity", "0")

        source_stem = xml_path.stem
        patched_path = (
            xml_path.parent / f"_irobman_{source_stem}_{body_name}_patched.xml"
        )
        tree.write(patched_path)
        return patched_path


def build_scene(cfg: dict) -> SceneArtifacts:
    """Convenience function to build a complete MuJoCo scene from configuration.

    Args:
        cfg: Configuration dictionary with scene parameters.

    Returns:
        A SceneArtifacts object containing the compiled model, data, and IDs.
    """
    builder = SceneBuilder(cfg)
    return builder.build()
