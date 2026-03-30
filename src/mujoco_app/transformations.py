"""Utility functions for coordinate frame and pose conversions."""

from math import cos, sin
from typing import Iterable, Tuple

import numpy as np


def rpy_to_quat_wxyz(rpy: Iterable[float]) -> np.ndarray:
    """Converts roll-pitch-yaw angles (radians) to a quaternion in wxyz order.

    Args:
        rpy: An iterable of three floats (roll, pitch, yaw) in radians.

    Returns:
        A 4-element numpy array [w, x, y, z] representing the rotation quaternion.
    """
    roll, pitch, yaw = rpy
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=float)


def quat_xyzw_to_matrix(quat_xyzw: Iterable[float]) -> np.ndarray:
    """Converts a quaternion (xyzw order) to a 3×3 rotation matrix.

    Args:
        quat_xyzw: An iterable of four floats [x, y, z, w].

    Returns:
        A 3×3 numpy array representing the rotation matrix.
    """
    x, y, z, w = quat_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )
    return rot


def compose_pose(
    position: Iterable[float], quat_xyzw: Iterable[float]
) -> np.ndarray:
    """Composes a 4×4 homogeneous transform from position and quaternion.

    Args:
        position: An iterable of three floats [x, y, z].
        quat_xyzw: An iterable of four floats [x, y, z, w] representing rotation.

    Returns:
        A 4×4 homogeneous transformation matrix.
    """
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = quat_xyzw_to_matrix(quat_xyzw)
    pose[:3, 3] = np.asarray(tuple(position), dtype=float)
    return pose


def look_at_matrix(
    eye: Iterable[float],
    target: Iterable[float],
    up: Iterable[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Returns a rotation matrix whose columns are the camera axes (X right, Y up, Z forward).

    Args:
        eye: 3D position of the camera.
        target: 3D position the camera is looking at.
        up: Up direction hint (defaults to +Z).

    Returns:
        A 3×3 rotation matrix.

    Raises:
        ValueError: If eye and target are identical or too close.
    """
    eye_vec = np.asarray(tuple(eye), dtype=float)
    target_vec = np.asarray(tuple(target), dtype=float)
    up_vec = np.asarray(tuple(up), dtype=float)

    forward = target_vec - eye_vec
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        raise ValueError("Eye and target must be distinct points")
    forward /= norm

    up_unit = up_vec / (np.linalg.norm(up_vec) + 1e-8)
    right = np.cross(up_unit, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # Choose an arbitrary orthogonal axis if up aligns with forward
        if abs(forward[0]) < 0.9:
            arbitrary = np.array([1.0, 0.0, 0.0])
        else:
            arbitrary = np.array([0.0, 1.0, 0.0])
        right = np.cross(arbitrary, forward)
        right /= np.linalg.norm(right)
    else:
        right /= right_norm

    up_cam = np.cross(forward, right)
    up_cam /= np.linalg.norm(up_cam)

    rot = np.column_stack((right, up_cam, forward))
    return rot


def camera_xyaxes(
    eye: Iterable[float],
    target: Iterable[float],
    up: Iterable[float] = (0.0, 0.0, 1.0),
) -> Tuple[str, np.ndarray]:
    """Computes the MuJoCo camera xyaxes string and rotation matrix from a look-at target.

    Args:
        eye: 3D position of the camera.
        target: 3D position the camera is looking at.
        up: Up direction hint (defaults to +Z).

    Returns:
        A tuple (xyaxes_string, rotation_matrix) where xyaxes_string is suitable
        for use in MuJoCo camera XML and rotation_matrix is the 3×3 rotation.
    """
    rot = look_at_matrix(eye, target, up)
    # MuJoCo cameras look along -Z, so flip the third column and Y axis accordingly.
    x_axis = rot[:, 0]
    y_axis = -rot[:, 1]
    xyaxes = "{} {} {} {} {} {}".format(
        x_axis[0], x_axis[1], x_axis[2], y_axis[0], y_axis[1], y_axis[2]
    )
    return xyaxes, rot


def quat_wxyz_to_xyzw(quat_wxyz: Iterable[float]) -> np.ndarray:
    """Converts a quaternion from wxyz ordering to xyzw ordering.

    Args:
        quat_wxyz: An iterable of four floats [w, x, y, z].

    Returns:
        A 4-element numpy array [x, y, z, w].
    """
    w, x, y, z = quat_wxyz
    return np.array([x, y, z, w], dtype=float)


def quat_xyzw_to_wxyz(quat_xyzw: Iterable[float]) -> np.ndarray:
    """Converts a quaternion from xyzw ordering to wxyz ordering.

    Args:
        quat_xyzw: An iterable of four floats [x, y, z, w].

    Returns:
        A 4-element numpy array [w, x, y, z].
    """
    x, y, z, w = quat_xyzw
    return np.array([w, x, y, z], dtype=float)


def normalize_vector(vec: Iterable[float]) -> np.ndarray:
    """Normalizes a vector to unit length.

    Args:
        vec: An iterable of floats representing a vector.

    Returns:
        A numpy array with the same direction but unit norm.

    Raises:
        ValueError: If the vector norm is too small to normalize.
    """
    arr = np.asarray(tuple(vec), dtype=float)
    norm = np.linalg.norm(arr)
    if norm < 1e-9:
        raise ValueError("Vector norm is too small to normalize")
    return arr / norm
