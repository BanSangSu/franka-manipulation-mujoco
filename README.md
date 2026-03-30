# iRobMan WiSe2526 – 🏗️ Pick and Place Task in MuJoCo

This repository implements an autonomous Pick and Place task in a dynamic environment, utilizing [iRobMan project](https://gitlab.pearl.informatik.tu-darmstadt.de/teaching/irobman2526-finalproject) as a baseline environment. The core objective is to pick and place the target object to the goal location while avoiding moving obstacles within the MuJoCo engine.

![demo](imgs/demo_l.gif)

---

## 1. Prerequisites

1. **Python** == `3.11`.
2. Install the MuJoCo runtime and Python package. Easiest paths:
    - With uv (fast, recommended)
    - With pip (if uv does not work well on your device)
3. (Optional, GUI issues) Set `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa` when rendering headless or on macOS with multiple GL stacks.
4. Clone the repository with all submodules.

---

## 2. Docker Setup

You can run the robot pipeline in a containerized environment using Docker. This approach **skips all installation steps** (uv sync, pip install, etc.).

```bash
# Allow X11 forwarding for visualization (Linux/macOS)
xhost +local:docker

# Pull the Docker image
docker pull bandi0605/irobman2526project:robot_pipeline_0.1

# Run the container with GPU support and X11 display
docker run -it --gpus all \
  --name irobman2526project \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  bandi0605/irobman2526project:robot_pipeline_0.1
```

**Note:** You can now directly use `python3 run_pick_and_place.py` and other commands inside the container without any setup.

---

## 3. Quick Start

### Using Docker (Recommended)

Docker is the easiest way to get started. See [Docker Setup](#2-docker-setup) section above for detailed instructions.

---

## 4. Pick and Place Testing

Test the pick and place manipulation pipeline with point cloud visualization.

### Prerequisites

Activate the virtual environment first:

```bash
source /workspace/irobman2526-finalproject/.venv/bin/activate
```

### Basic Usage

```bash
# Run pick and place with default configuration
python3 run_pick_and_place.py --config configs/test_config_mj.yaml

# Run with point cloud visualization
python3 run_pick_and_place.py --config configs/test_config_mj.yaml --viz
```

### Arguments

- `--config`: Path to the MuJoCo configuration file (default: `configs/test_config_mj.yaml`)
- `--viz`: Enable visualization of point cloud images during execution

---

## 5. Robot Pipeline

The robot pipeline orchestrates the manipulation stack with integrated pick and place capabilities.

### Main Components

- **`run_pick_and_place.py`**: Entry point for executing pick and place tasks. Handles robot motion planning and object manipulation with optional point cloud visualization.
- **`pipeline.py`**: Core pipeline orchestration. Manages the integration of simulation, perception, and control modules for coordinated robot operation.

### Basic Workflow

1. Load configuration from `configs/test_config_mj.yaml`
2. Initialize MuJoCo simulation environment
3. Execute pick and place sequence
4. (Optional) Visualize point cloud data with `--viz` flag
---

## 6. Key Technologies Used

- **Florence-2**: A prompt-based vision-language model (VLM) providing task flexibility.
- **YOLOv23n**: Real-time object detection and tracking for moving objects.
- **ICP**: Point cloud registration and 3D pose estimation.
- **GraspGen**: Diffusion-based generation of optimal grasp points.
- **RRTConnect**: Collision-free global motion planning in complex environments.
- **Artificial Potential Field**: Local planner for real-time dynamic obstacle avoidance.
