# General Information

### Available Objects

The following YCB objects are available (case-sensitive):
- YcbBanana
- YcbCrackerBox
- YcbFoamBrick
- YcbMasterChefCan
- YcbMustardBottle
- YcbPear
- YcbPowerDrill
- YcbStrawberry
- YcbTennisBall
- YcbTomatoSoupCan

### Sensor Logging Output

When `--save-sensors` is enabled, logs are saved to `sensor_logs/<ObjectName>_<timestamp>/`:
- `camera_log.json` – RGB/depth statistics per frame
- `joint_log.json` – Robot arm joint positions and velocities
- `ee_log.json` – End-effector position and rotation matrix
- `contact_log.json` – Number of active contacts per frame
- `rgb_frame_*.npy` / `depth_frame_*.npy` – (if `--save-images`) Raw image arrays saved every 30 frames

## Platform-Specific Graphics Settings

**macOS:**
```bash
# mjpython if gui is turned on
# Use EGL for offscreen rendering (recommended)
export MUJOCO_GL=egl
python view_object_with_sensors.py YcbBanana

# Or use software rendering
export MUJOCO_GL=osmesa
python view_object_with_sensors.py YcbBanana
```

**Linux:**
```bash
# EGL is auto-set for Linux in most cases
# If you encounter issues, explicitly set it:
export MUJOCO_GL=egl
python view_object_with_sensors.py YcbChipsCan

# For headless servers without GPU:
export MUJOCO_GL=osmesa
python view_object_with_sensors.py YcbChipsCan
```

**Windows:**
```powershell
# Windows uses native OpenGL by default - no environment variable needed
python view_object_with_sensors.py YcbBanana

# If you encounter rendering issues, try:
$env:MUJOCO_GL="osmesa"
python view_object_with_sensors.py YcbBanana
```

## Controls in Viewer
- **Mouse**: Rotate view (drag), Pan (Shift+drag), Zoom (scroll or right-drag)
- **Ctrl+C**: Exit viewer and save logs

## Configuring Scenes (`mujoco_app/configs/test_config_mj.yaml`)

Key sections:

- **`mujoco.camera`** – main camera intrinsics/extrinsics (`pos`, `target`, `fovy`, `near`, `far`).
- **`mujoco.user_camera` / `mujoco.extra_cameras`** – optional additional viewpoints.
- **`mujoco.grasp_object`** – include a standalone object XML and place it on the table. Example object XMLs live under `mujoco_app/assets/objects/`.
- **`mujoco.ycb_objects`** – quick primitive objects placed via presets (`table_left`, `table_right`, `basket_center`, `table_center`).
- **`table` / `basket`** – geometric and visual properties for furniture.
- **`robot_settings`** – default joint angles and gripper opening after `reset()`.

The YAML file is the single source of truth for spatial transforms; all conversions in code go through `transformations.py`.

Note on default object:
- To use the YCB strawberry (`assets/mujoco_objects/YcbStrawberry/textured.xml`), ensure the meshes exist under `assets/mujoco_objects/YcbStrawberry/` or update the XML paths accordingly.

## Developing & Extending

- Use `scene_builder.py` to extend asset generation (e.g., more props, materials, or lighting schemes).
- For new kinematic utilities, add them to `transformations.py` to keep frame math consistent.
- Additional helper scripts can import `MjSim` directly for custom control loops.

## Troubleshooting

### General Issues

- **Images are blank** – disable the GUI (`gui: false`) for purely off-screen rendering, or ensure EGL/OSMesa is available.
- **Objects sinking into the table** – verify the `size`/`place` fields in the YAML so placement logic computes the right clearance.
- **Missing assets** – update `xml` paths relative to the project root (`mujoco_app/assets` is a good place for custom meshes and materials).
    - If you switch to YCB objects, make sure the referenced OBJ files exist locally, or keep the default `sample_object.xml`.

### Platform-Specific Issues

**macOS (Darwin):**
- **OpenGL Context Issues**: If you encounter GL errors or blank renders, set the MuJoCo GL backend:
  ```bash
  # For offscreen rendering
  export MUJOCO_GL=egl
  # Or use software rendering
  export MUJOCO_GL=osmesa

  # Then run with UV
  ```
- **Apple Silicon (M1/M2/M3)**: UV automatically installs ARM64-native wheels for better performance.

**Windows:**
- **PowerShell Execution Policy**: If UV installation fails, ensure execution policy allows scripts:
  ```powershell
  Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
  ```
- **PATH Issues**: Restart PowerShell or your terminal after UV installation to refresh PATH.
- **Visual C++ Redistributable**: Some packages may require Microsoft Visual C++ 14.0 or greater. Install from [Microsoft's website](https://visualstudio.microsoft.com/downloads/).

**Linux:**
- **Missing OpenGL Libraries**: Install system dependencies for OpenGL:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install libgl1-mesa-glx libglu1-mesa libosmesa6

  # Fedora/RHEL
  sudo dnf install mesa-libGL mesa-libGLU mesa-libOSMesa
  ```
- **Headless Servers**: Use EGL or OSMesa for rendering without a display:
  ```bash
  export MUJOCO_GL=egl
  # Or
  export MUJOCO_GL=osmesa
  ```

## UV-Specific Issues

- **Lock file conflicts**: If switching between platforms, regenerate the lock:
  ```bash
  uv lock --upgrade
  ```
- **Cache issues**: Clear UV cache if you encounter package resolution problems:
  ```bash
  uv cache clean
  ```

## MjSim issues:

- If your simulation seems slow it can be due to the time of computations for your methods. Check once with only `sim.step()` in a for loop.
- You can also increase the number of physics steps per simulation step:
  ```python
  sim.step(10)  # Implies 10 physics steps per simulation step
  ```
- You have been given an example script in `main.py` showcasing how to use the simulator.
- Usually its a good idea to do a few thousand sim.steps() before you start acting on the sim. This ensures all the objects have settled down.
- Always after ending an episode with a given object, reset the simulation:
  ```python
  sim.reset()
  ```
  Do not exit the code and start again as this will not randomize the objects correctly and will spawn it in the same location as before.
- If your object seems too wobbly and not settling down even after a lot of steps inform the TA's and we will update that particular model or provide you with a different model.
- If you encounter any other issues, please refer to the [official documentation](https://mujoco.readthedocs.io/en/latest/) or contact the TA's for assistance.
