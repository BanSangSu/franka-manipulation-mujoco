# Robot Control Cheat Sheet

Minimal, action-focused guide for controlling the Panda in MuJoCo.

## Quick Start
- Init: `sim = MjSim(config); sim.reset()` then loop `sim.step()`.
- Joint space: `sim.set_arm_joint_positions([...])`

## Joint Control
- Set joints (radians, 7 values):
```python
joint_positions = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
sim.set_arm_joint_positions(joint_positions, clamp=True, sync=True)
```
- Read state:
```python
q = [sim.data.qpos[qidx] for qidx, _ in sim.robot.arm_pairs]
qd = [sim.data.qvel[vidx] for _, vidx in sim.robot.arm_pairs]
```
- Direct control (advanced):
```python
for idx, pos in zip(sim._arm_actuator_indices, joint_positions):
    sim.data.ctrl[idx] = pos
```

## Gripper
- Set opening (meters):
```python
sim._set_gripper_opening(0.04)   # open
sim._set_gripper_opening(0.00)   # close
sim._set_gripper_opening(0.02)   # partial
```

## Control Loop (template)
```python
sim = MjSim(config); sim.reset()
for _ in range(1000):
    q = [sim.data.qpos[qidx] for qidx, _ in sim.robot.arm_pairs]
    target_q = plan(q)  # your planner/controller
    sim.set_arm_joint_positions(target_q)
    sim.step()
    if sim.check_robot_obstacle_collision():
        print("Collision!"); break
sim.close()
```

## Examples
- Joint trajectory (sin on J1):
```python
import numpy as np
for t in range(1000):
    qp = [0.5*np.sin(2*np.pi*t/1000), -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    sim.set_arm_joint_positions(qp); sim.step()
```

## Tips
- Check collisions with `sim.check_robot_obstacle_collision()` and slip with `sim.check_object_slip()`.
