"""
RRT-Connect – Bi-directional Rapidly-exploring Random Tree planner for
joint-space path planning in MuJoCo.

Reference: Kuffner & LaValle, "RRT-Connect: An Efficient Approach to
Single-Query Path Planning", ICRA 2000.

Changes vs. original
---------------------
* MuJoCoCollisionChecker
  - Properly filters base-link / floor contacts (was commented out but
    never implemented).
  - Accepts an optional ``target_body_name`` for the object to be grasped;
    end-effector ↔ target contacts are *allowed* inside
    ``pre_grasp_distance`` of the goal so the planner can reach all the way
    in without being trapped.
  - Collision mode can be overridden per-call via ``allow_target_contact``.
* _connect bug-fix: now correctly threads the *returned* node through
  repeated _extend calls instead of always reading tree[-1].
* smooth_path: honours the same collision checker so pre-grasp contacts
  are not incorrectly cut short.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Set, Tuple

import mujoco
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------
class TreeNode:
    """A single node in the RRT tree."""

    __slots__ = ("config", "parent")

    def __init__(self, config: np.ndarray, parent: Optional["TreeNode"] = None):
        self.config = config
        self.parent = parent

    def retrace(self) -> List[np.ndarray]:
        """Walk back to root and return the path (root → self)."""
        path: List[np.ndarray] = []
        node: Optional[TreeNode] = self
        while node is not None:
            path.append(node.config)
            node = node.parent
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# Configuration-space helpers
# ---------------------------------------------------------------------------
@dataclass
class JointSpaceConfig:
    """Stores joint limits and helpers for the robot arm."""

    lower: np.ndarray
    upper: np.ndarray
    n_joints: int = 7
    step_size: float = 0.1  # radians – max extension per RRT step

    @classmethod
    def from_robot(cls, robot: Any) -> "JointSpaceConfig":
        lowers, uppers = [], []
        for qidx, _ in robot.arm_pairs:
            if qidx in robot._joint_lims:
                lo, hi = robot._joint_lims[qidx]
            else:
                lo, hi = -2 * np.pi, 2 * np.pi
            lowers.append(lo)
            uppers.append(hi)
        return cls(
            lower=np.array(lowers),
            upper=np.array(uppers),
            n_joints=len(robot.arm_pairs),
        )


def _distance(q1: np.ndarray, q2: np.ndarray) -> float:
    return float(np.linalg.norm(q2 - q1))


def _sample(js: JointSpaceConfig, goal: np.ndarray, goal_bias: float = 0.1) -> np.ndarray:
    if np.random.random() < goal_bias:
        return goal.copy()
    return np.random.uniform(js.lower, js.upper)


def _steer(q_from: np.ndarray, q_to: np.ndarray, step_size: float) -> np.ndarray:
    diff = q_to - q_from
    dist = np.linalg.norm(diff)
    if dist <= step_size:
        return q_to.copy()
    return q_from + diff * (step_size / dist)


def _interpolate(q1: np.ndarray, q2: np.ndarray, resolution: float = 0.05) -> List[np.ndarray]:
    dist = np.linalg.norm(q2 - q1)
    n_steps = max(int(np.ceil(dist / resolution)), 1)
    return [q1 * (1.0 - i / n_steps) + q2 * (i / n_steps) for i in range(1, n_steps + 1)]


# ---------------------------------------------------------------------------
# Collision checking via MuJoCo  (FIXED + EXTENDED)
# ---------------------------------------------------------------------------
class MuJoCoCollisionChecker:
    """Check collisions by temporarily setting joint positions in MuJoCo.

    Parameters
    ----------
    model, data : MuJoCo model / data pair.
    robot        : MjRobot-like object with ``arm_pairs``.
    self_collision  : whether to check arm self-collisions.
    env_collision   : whether to check collisions with environment objects.
    target_body_name: name of the object that will be grasped.  Contacts
                      between the end-effector and this body are *ignored*
                      within ``pre_grasp_distance`` joint-space radius of
                      the goal, so the planner can approach without getting
                      trapped by the object it is reaching for.
    ee_body_name    : name of the end-effector body (fingertip / hand).
    base_body_names : body names whose ground contacts should be ignored
                      (robot base, heavy link1, etc.).  Pass an empty list
                      to disable.
    """

    # Keywords used to identify robot bodies when no explicit list is given.
    _ROBOT_KEYWORDS = ("panda", "hand", "finger", "link")

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot: Any,
        self_collision: bool = True,
        env_collision: bool = True,
        target_body_name: Optional[str] = None,
        ee_body_name: Optional[str] = "panda_hand",
        base_body_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.data = data
        self.robot = robot
        self._self_collision = self_collision
        self._env_collision = env_collision
        self._attached_geom_ids: Set[int] = set()

        # --- robot geometry IDs ----------------------------------------
        self._robot_geom_ids: Set[int] = self._get_robot_geom_ids()
        self._adjacent_pairs: Set[Tuple[int, int]] = self._get_adjacent_pairs()

        # --- base-link geoms (contacts with floor/table are ignored) -----
        self._base_geom_ids: Set[int] = set()
        if base_body_names is None:
            # Default: ignore floor contacts for the first two link bodies.
            base_body_names = ["panda_link0", "panda_link1"]
        for bname in base_body_names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid >= 0:
                geoms = np.where(model.geom_bodyid == bid)[0]
                self._base_geom_ids.update(geoms.tolist())

        # --- environment (non-robot) geoms --------------------------------
        self._env_geom_ids: Set[int] = (
            set(range(model.ngeom)) - self._robot_geom_ids
        )

        # --- floor geoms (named "floor" or type mjGEOM_PLANE) ------------
        self._floor_geom_ids: Set[int] = self._get_floor_geom_ids()

        # --- target object geoms (for pre-grasp relaxation) --------------
        self._target_geom_ids: Set[int] = set()
        self._allow_target_contact = False  # toggled externally
        if target_body_name:
            tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
            if tid >= 0:
                geoms = np.where(model.geom_bodyid == tid)[0]
                self._target_geom_ids.update(geoms.tolist())
            else:
                logger.warning("target_body_name '%s' not found in model", target_body_name)

        # --- end-effector geoms (fingertip contacts with target OK) ------
        self._ee_geom_ids: Set[int] = set()
        if ee_body_name:
            eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
            if eid >= 0:
                geoms = np.where(model.geom_bodyid == eid)[0]
                self._ee_geom_ids.update(geoms.tolist())
            # Also include finger bodies one level down.
            for i in range(model.nbody):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                if name and "finger" in name.lower():
                    geoms = np.where(model.geom_bodyid == i)[0]
                    self._ee_geom_ids.update(geoms.tolist())

    def set_attached_body_name(self, body_name: Optional[str]):
        """Treat the given body as attached to the hand during planning."""
        self._attached_geom_ids.clear()
        if not body_name:
            return
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid >= 0:
            geoms = np.where(self.model.geom_bodyid == bid)[0]
            self._attached_geom_ids.update(geoms.tolist())
        else:
            logger.warning("attached body '%s' not found in model", body_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_robot_geom_ids(self) -> Set[int]:
        robot_bodies: Set[int] = set()
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and any(kw in name.lower() for kw in self._ROBOT_KEYWORDS):
                robot_bodies.add(i)
        geom_ids: Set[int] = set()
        for bid in robot_bodies:
            geoms = np.where(self.model.geom_bodyid == bid)[0]
            geom_ids.update(geoms.tolist())
        return geom_ids

    def _get_adjacent_pairs(self) -> Set[Tuple[int, int]]:
        pairs: Set[Tuple[int, int]] = set()
        body_to_geoms: dict = {}
        for gid in self._robot_geom_ids:
            bid = int(self.model.geom_bodyid[gid])
            body_to_geoms.setdefault(bid, set()).add(gid)
        for bid in body_to_geoms:
            parent = int(self.model.body_parentid[bid])
            if parent in body_to_geoms:
                for g1 in body_to_geoms[bid]:
                    for g2 in body_to_geoms[parent]:
                        pairs.add((min(g1, g2), max(g1, g2)))
        return pairs

    def _get_floor_geom_ids(self) -> Set[int]:
        ids: Set[int] = set()
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if name and "floor" in name.lower():
                ids.add(gid)
            # Also catch un-named plane geoms (type 0 == mjGEOM_PLANE)
            elif int(self.model.geom_type[gid]) == 0:
                ids.add(gid)
        return ids

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(
        self,
        q: np.ndarray,
        allow_target_contact: Optional[bool] = None,
    ) -> bool:
        """Return True if config ``q`` is in collision.

        Parameters
        ----------
        allow_target_contact : override the instance-level flag for this
            single call.  Pass True near the goal to let the hand touch
            the grasp target without being flagged as a collision.
        """
        allow_target = (
            allow_target_contact
            if allow_target_contact is not None
            else self._allow_target_contact
        )

        # Save state
        q_orig = self.data.qpos.copy()

        # Set arm joints
        for i, (qidx, _) in enumerate(self.robot.arm_pairs):
            self.data.qpos[qidx] = q[i]
        mujoco.mj_forward(self.model, self.data)

        in_collision = False

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)

            # Ignore separated contacts
            if con.dist > 0:
                continue

            pair = (min(g1, g2), max(g1, g2))
            both_robot = g1 in self._robot_geom_ids and g2 in self._robot_geom_ids

            if both_robot:
                # ---- Self-collision ----------------------------------
                if not self._self_collision:
                    continue
                # Skip adjacent-link contacts (they always overlap slightly)
                if pair in self._adjacent_pairs:
                    continue
                in_collision = True
                break

            else:
                # ---- Environment collision ---------------------------
                if not self._env_collision:
                    continue

                robot_in = g1 in self._robot_geom_ids or g2 in self._robot_geom_ids
                if not robot_in:
                    continue

                # Ignore base links touching the floor / table plane
                base_involved = g1 in self._base_geom_ids or g2 in self._base_geom_ids
                floor_involved = g1 in self._floor_geom_ids or g2 in self._floor_geom_ids
                if base_involved and floor_involved:
                    continue

                # Ignore end-effector ↔ target during pre-grasp approach
                if allow_target and self._target_geom_ids:
                    ee_involved = g1 in self._ee_geom_ids or g2 in self._ee_geom_ids
                    tgt_involved = g1 in self._target_geom_ids or g2 in self._target_geom_ids
                    if ee_involved and tgt_involved:
                        continue

                # Ignore robot ↔ attached-object contacts while carrying.
                if self._attached_geom_ids:
                    attached_involved = (
                        g1 in self._attached_geom_ids or g2 in self._attached_geom_ids
                    )
                    if attached_involved:
                        continue

                in_collision = True
                break

        # Restore state
        self.data.qpos[:] = q_orig
        mujoco.mj_forward(self.model, self.data)

        return in_collision

    def check_edge(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        resolution: float = 0.05,
        allow_target_contact: Optional[bool] = None,
    ) -> bool:
        """Return True if the edge q1→q2 contains a collision."""
        for q in _interpolate(q1, q2, resolution):
            if self(q, allow_target_contact=allow_target_contact):
                return True
        return False


# ---------------------------------------------------------------------------
# RRT-Connect core
# ---------------------------------------------------------------------------
def _nearest(tree: List[TreeNode], q: np.ndarray) -> TreeNode:
    dists = [_distance(n.config, q) for n in tree]
    return tree[int(np.argmin(dists))]


def _extend(
    tree: List[TreeNode],
    q_target: np.ndarray,
    step_size: float,
    collision_fn: MuJoCoCollisionChecker,
    js: JointSpaceConfig,
    allow_target_contact: bool = False,
) -> Tuple[TreeNode, str]:
    nearest = _nearest(tree, q_target)
    q_new = np.clip(_steer(nearest.config, q_target, step_size), js.lower, js.upper)

    if collision_fn(q_new, allow_target_contact=allow_target_contact):
        return nearest, "trapped"
    if collision_fn.check_edge(nearest.config, q_new, allow_target_contact=allow_target_contact):
        return nearest, "trapped"

    new_node = TreeNode(q_new, parent=nearest)
    tree.append(new_node)

    status = "reached" if _distance(q_new, q_target) < 1e-3 else "advanced"
    return new_node, status


def _connect(
    tree: List[TreeNode],
    q_target: np.ndarray,
    step_size: float,
    collision_fn: MuJoCoCollisionChecker,
    js: JointSpaceConfig,
    allow_target_contact: bool = False,
) -> Tuple[TreeNode, str]:
    """Greedily extend towards q_target until reached or trapped.

    BUG FIX: the original always read tree[-1] for the current node;
    we now correctly thread the node returned by each _extend call.
    """
    node, status = _extend(tree, q_target, step_size, collision_fn, js, allow_target_contact)
    while status == "advanced":
        node, status = _extend(tree, q_target, step_size, collision_fn, js, allow_target_contact)
    return node, status


def rrt_connect(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    collision_fn: MuJoCoCollisionChecker,
    js: JointSpaceConfig,
    max_iterations: int = 5000,
    max_time: float = 30.0,
    goal_bias: float = 0.1,
    step_size: Optional[float] = None,
    pre_grasp_distance: float = 0.5,
) -> Optional[List[np.ndarray]]:
    """Bi-directional RRT-Connect in joint space.

    Parameters
    ----------
    q_start, q_goal : start and goal joint configs.
    collision_fn     : MuJoCoCollisionChecker instance.
    js               : joint-space limits / step size.
    max_iterations   : max RRT iterations.
    max_time         : wall-clock timeout in seconds.
    goal_bias        : probability of sampling the goal directly.
    step_size        : override js.step_size if given.
    pre_grasp_distance : joint-space radius around the goal within which
        end-effector ↔ grasp-target contacts are *not* treated as
        collisions.  Set to 0 to disable.
    """
    if step_size is None:
        step_size = js.step_size

    t0 = time.time()

    if collision_fn(q_start):
        logger.warning("RRT: start configuration is in collision – proceeding anyway.")
    if collision_fn(q_goal, allow_target_contact=True):
        logger.warning("RRT: goal configuration is in collision (excluding target contact).")

    tree_a: List[TreeNode] = [TreeNode(q_start)]
    tree_b: List[TreeNode] = [TreeNode(q_goal)]

    # Track which tree contains the goal so we know when pre-grasp rules apply.
    goal_tree_index = 1  # tree_b starts as the goal tree

    for iteration in range(max_iterations):
        if time.time() - t0 > max_time:
            logger.warning("RRT-Connect timed out after %.1fs", max_time)
            break

        # Swap trees for balanced growth; track which tree holds the goal.
        if len(tree_a) > len(tree_b):
            tree_a, tree_b = tree_b, tree_a
            goal_tree_index = 1 - goal_tree_index  # 0 ↔ 1

        # tree_b_is_goal_tree: True when tree_b is rooted at q_goal
        tree_b_is_goal = goal_tree_index == 1

        # Sample towards tree_b root (for goal bias)
        q_rand = _sample(js, goal=tree_b[0].config, goal_bias=goal_bias)

        # Extend tree_a
        node_a, status_a = _extend(tree_a, q_rand, step_size, collision_fn, js)

        if status_a != "trapped":
            # Allow touching the target when connecting into the goal tree
            allow_contact = (
                tree_b_is_goal
                and pre_grasp_distance > 0
                and _distance(node_a.config, q_goal) < pre_grasp_distance
            )
            node_b, status_b = _connect(
                tree_b,
                node_a.config,
                step_size,
                collision_fn,
                js,
                allow_target_contact=allow_contact,
            )

            if status_b == "reached":
                path_a = node_a.retrace()
                path_b = node_b.retrace()

                # Ensure path runs start → goal
                if _distance(path_a[0], q_start) > _distance(path_b[0], q_start):
                    path_a, path_b = path_b, path_a

                full_path = path_a + path_b[::-1]
                logger.info(
                    "RRT-Connect found path: %d nodes, %d iterations, %.2fs",
                    len(full_path),
                    iteration + 1,
                    time.time() - t0,
                )
                return full_path

    logger.warning("RRT-Connect failed after %d iterations", max_iterations)
    return None


# ---------------------------------------------------------------------------
# Path smoothing / resampling
# ---------------------------------------------------------------------------
def smooth_path(
    path: List[np.ndarray],
    collision_fn: MuJoCoCollisionChecker,
    max_iterations: int = 200,
    pre_grasp_distance: float = 0.5,
    q_goal: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Shortcut-based path smoothing.

    Shortcuts near the goal respect pre-grasp contact relaxation so the
    smoother doesn't break the final approach.
    """
    if len(path) <= 2:
        return path

    path = [q.copy() for q in path]

    for _ in range(max_iterations):
        if len(path) <= 2:
            break

        i = np.random.randint(0, len(path) - 2)
        j = np.random.randint(i + 2, len(path))

        # Allow target contact for shortcuts that reach into the pre-grasp zone
        allow_contact = (
            q_goal is not None
            and pre_grasp_distance > 0
            and _distance(path[j], q_goal) < pre_grasp_distance
        )

        if not collision_fn.check_edge(
            path[i], path[j], resolution=0.03, allow_target_contact=allow_contact
        ):
            path = path[: i + 1] + path[j:]

    logger.info("Path smoothed to %d waypoints", len(path))
    return path


def resample_path(path: List[np.ndarray], resolution: float = 0.05) -> List[np.ndarray]:
    """Resample a path to roughly uniform joint-space spacing."""
    if len(path) <= 1:
        return path

    resampled = [path[0].copy()]
    for i in range(1, len(path)):
        resampled.extend(_interpolate(path[i - 1], path[i], resolution))

    logger.info("Path resampled to %d waypoints (resolution=%.3f)", len(resampled), resolution)
    return resampled
