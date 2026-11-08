"""Vectorized math helpers shared by the physics engine."""

import numpy as np


def cross(a, b):
    """
    Compute the cross product of two arrays of 3D vectors.

    This is several times faster than ``np.cross`` for the small batches the
    solver handles.

    Parameters
    ----------
    a, b : ndarray, shape (..., 3)
        Input vectors.

    Returns
    -------
    ndarray, shape (..., 3)
        Cross products.
    """
    return np.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def dot(a, b):
    """
    Compute the dot product along the last axis.

    Parameters
    ----------
    a, b : ndarray, shape (..., K)
        Input vectors.

    Returns
    -------
    ndarray, shape (...)
        Dot products.
    """
    return (a * b).sum(-1)


def quat_to_matrix(q):
    """
    Convert quaternions to rotation matrices.

    Parameters
    ----------
    q : ndarray, shape (N, 4)
        Unit quaternions in ``(x, y, z, w)`` order.

    Returns
    -------
    ndarray, shape (N, 3, 3)
        Rotation matrices.
    """
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    matrix = np.empty((len(q), 3, 3))
    matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[:, 0, 1] = 2 * (x * y - z * w)
    matrix[:, 0, 2] = 2 * (x * z + y * w)
    matrix[:, 1, 0] = 2 * (x * y + z * w)
    matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[:, 1, 2] = 2 * (y * z - x * w)
    matrix[:, 2, 0] = 2 * (x * z - y * w)
    matrix[:, 2, 1] = 2 * (y * z + x * w)
    matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def integrate_quaternions(q, angle):
    """
    Rotate quaternions in place by small world-space rotation vectors.

    Parameters
    ----------
    q : ndarray, shape (N, 4)
        Quaternions in ``(x, y, z, w)`` order, modified in place.
    angle : ndarray, shape (N, 3)
        Rotation vectors, in radians.
    """
    wx, wy, wz = angle[:, 0], angle[:, 1], angle[:, 2]
    qx, qy, qz, qw = q[:, 0].copy(), q[:, 1].copy(), q[:, 2].copy(), q[:, 3].copy()
    q[:, 0] = qx + 0.5 * (wx * qw + wy * qz - wz * qy)
    q[:, 1] = qy + 0.5 * (wy * qw + wz * qx - wx * qz)
    q[:, 2] = qz + 0.5 * (wz * qw + wx * qy - wy * qx)
    q[:, 3] = qw + 0.5 * (-wx * qx - wy * qy - wz * qz)
    q /= np.linalg.norm(q, axis=1, keepdims=True)


def world_inverse_inertia(rotation, inverse_inertia):
    """
    Rotate diagonal body-space inverse inertia tensors into world space.

    Parameters
    ----------
    rotation : ndarray, shape (N, 3, 3)
        Body rotation matrices.
    inverse_inertia : ndarray, shape (N, 3)
        Diagonal inverse inertia in body space.

    Returns
    -------
    ndarray, shape (N, 3, 3)
        World-space inverse inertia tensors.
    """
    return (rotation * inverse_inertia[:, None, :]) @ rotation.transpose(0, 2, 1)
