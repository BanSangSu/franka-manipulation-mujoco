# Robot Sensors Cheat Sheet

## Quick Start
- Init: `sim = MjSim(config); sim.reset()`
- Access sensor data through `sim.data`, rendering methods, and collision APIs
- All sensor data updates automatically after each `sim.step()`

## Camera Data

### Primary Camera (RGB, Depth)
```python
# Configure in YAML
cam_cfg = sim.cfg.get("mujoco", {}).get("camera", {})
width = cam_cfg.get("width", 640)
height = cam_cfg.get("height", 480)
near = cam_cfg.get("near", 0.01)
far = cam_cfg.get("far", 5.0)
fovy = cam_cfg.get("fovy", 58.0)

# Get renders from primary camera
rgb, depth, intrinsic, extrinsic = sim.get_static_renders(
    width=width, height=height, near=near, far=far, fovy=fovy
)
# rgb: (H,W,3) uint8 RGB image
# depth: (H,W) float32 depth map in meters
# intrinsic: (3,3) intrinsic matrix
# extrinsic: (4,4) extrinsic matrix
```

### Named Camera
```python
rgb, depth, intrinsic, extrinsic = sim.render_camera(
    "wrist_cam",  # camera name
    width=640, height=480,
    near=0.01, far=5.0, fovy=58.0
)
```

### All Extra Cameras
```python
outputs = sim.render_additional_cameras()
for cam_name, data in outputs.items():
    rgb = data["rgb"]        # (H,W,3) uint8
    depth = data["depth"]    # (H,W) float32
    K = data["intrinsic"]    # (3,3)
    E = data["extrinsic"]    # (4,4)
```

## Joint Sensors

### Arm Joint States
```python
# Joint positions (radians)
arm_positions = [sim.data.qpos[qidx] for qidx, _ in sim.robot.arm_pairs]

# Joint velocities (rad/s)
arm_velocities = [sim.data.qvel[vidx] for _, vidx in sim.robot.arm_pairs]

# Joint efforts/torques
arm_efforts = [sim.data.actuator_force[i] for i in sim._arm_actuator_indices]
```

### Gripper State
```python
# Gripper joint positions
finger_positions = [sim.data.qpos[qidx] for qidx, _ in sim.robot.finger_pairs]

# Total gripper opening (meters)
gripper_opening = sum(abs(q) for q in finger_positions)
```

## End-Effector Sensors

### EE Pose (Position + Orientation)
```python
import mujoco

# Get end-effector body ID
ee_body_name = sim.robot_settings.get("ee_body_name", "hand")
ee_body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)

# Position (x,y,z)
ee_pos = sim.data.xpos[ee_body_id].copy()

# Rotation matrix (3x3)
ee_rot = sim.data.xmat[ee_body_id].reshape(3, 3).copy()

# Or use robot helper
ee_pos, ee_rot = sim.robot.get_ee_pose()
```

### EE Velocity
```python
# Get body velocity (linear + angular)
ee_vel = sim.data.cvel[ee_body_id].copy()  # (6,) [angular_vel, linear_vel]
linear_vel = ee_vel[3:6]   # (vx, vy, vz)
angular_vel = ee_vel[0:3]  # (wx, wy, wz)
```

## Contact Sensors

### All Contacts
```python
contacts = sim.contacts()
# Returns list of dicts with keys:
# - 'geom1', 'geom2': geom IDs in contact
# - 'pos': contact position
# - 'frame': contact frame orientation
# - 'dist': penetration distance
# - 'force': contact force (6D)
# - 'normal': contact normal
# - 'friction': friction coefficients

num_contacts = len(contacts)
```

### Robot-Obstacle Collision Detection
```python
# Check if robot collides with moving obstacles
collision = sim.check_robot_obstacle_collision(robot_body_prefix="link")
if collision:
    print("Robot collided with obstacle!")
```

### Object Slip Detection
```python
# Check if object is slipping from gripper
object_name = "sample_object"
is_slipping = sim.check_object_slip(
    object_name,
    gripper_body_name="hand",
    velocity_threshold=0.05  # m/s
)
```

### Specific Contact Check
```python
for contact in sim.contacts():
    geom1_name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, contact['geom1'])
    geom2_name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, contact['geom2'])

    # Check if specific bodies are in contact
    if "finger" in geom1_name and "object" in geom2_name:
        contact_force = contact['force']  # (6,) wrench
        contact_pos = contact['pos']      # (3,) position
```

## Object State Sensors

### Object Pose (for objects with freejoint)
```python
# Get object body ID
obj_name = "sample_object"
obj_body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)

# Position
obj_pos = sim.data.xpos[obj_body_id].copy()

# Orientation (rotation matrix)
obj_rot = sim.data.xmat[obj_body_id].reshape(3, 3).copy()

# Or from freejoint qpos directly
obj_joint_name = f"{obj_name}_free"
jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, obj_joint_name)
if jid >= 0:
    qadr = sim.model.jnt_qposadr[jid]
    obj_pos = sim.data.qpos[qadr:qadr+3].copy()        # xyz
    obj_quat_wxyz = sim.data.qpos[qadr+3:qadr+7].copy() # w,x,y,z
```

### Object Velocity
```python
# Linear and angular velocity
obj_vel = sim.data.cvel[obj_body_id].copy()  # (6,)
obj_linear_vel = obj_vel[3:6]
obj_angular_vel = obj_vel[0:3]
```

### Basket Detection
```python
# Check if object is inside basket
basket_center = sim.ids.get("basket_center")
basket_dims = sim.ids.get("basket_dims")  # [inner_x, inner_y]
basket_height = sim.ids.get("basket_height")

obj_pos = sim.data.xpos[obj_body_id].copy()

# Check horizontal bounds
in_x = abs(obj_pos[0] - basket_center[0]) < basket_dims[0] / 2
in_y = abs(obj_pos[1] - basket_center[1]) < basket_dims[1] / 2

# Check vertical position (object above basket bottom)
in_z = obj_pos[2] > (basket_center[2] - basket_height/2)

in_basket = in_x and in_y and in_z
```

## Simulation Time
```python
# Current simulation time (seconds)
current_time = sim.data.time

# Internal time accumulator (if needed)
sim_time = sim._sim_time
```

## Sensor Logging Template
```python
from pathlib import Path
import json

class SensorLogger:
    def __init__(self, sim, log_dir):
        self.sim = sim
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0
        self.data_log = []

    def log_frame(self):
        # Joint states
        arm_pos = [self.sim.data.qpos[qidx] for qidx, _ in self.sim.robot.arm_pairs]
        arm_vel = [self.sim.data.qvel[vidx] for _, vidx in self.sim.robot.arm_pairs]

        # EE pose
        ee_pos, ee_rot = self.sim.robot.get_ee_pose()

        # Camera (example: save stats, not full images)
        rgb, depth, _, _ = self.sim.get_static_renders(width=640, height=480,
                                                       near=0.01, far=5.0, fovy=58.0)

        # Contacts
        contacts = self.sim.contacts()

        self.data_log.append({
            "frame": self.frame_count,
            "time": self.sim.data.time,
            "arm_positions": arm_pos,
            "arm_velocities": arm_vel,
            "ee_position": ee_pos.tolist(),
            "ee_rotation": ee_rot.tolist(),
            "depth_mean": float(depth.mean()),
            "num_contacts": len(contacts),
        })

        self.frame_count += 1

    def save_logs(self):
        with open(self.log_dir / "sensor_log.json", "w") as f:
            json.dump(self.data_log, f, indent=2)
        print(f"Saved {self.frame_count} frames to {self.log_dir}")

# Usage
logger = SensorLogger(sim, "sensor_logs")
for _ in range(1000):
    sim.step()
    if _ % 10 == 0:  # Log every 10 steps
        logger.log_frame()
logger.save_logs()
```

## Tips
- Use `sim.data.time` for consistent time tracking across all sensors
- Contact normals point from geom2 towards geom1
- Depth values are in meters; use `near` and `far` for scaling
