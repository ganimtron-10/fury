"""Collision shapes and their mass properties."""

import numpy as np

SPHERE = 0

DYNAMIC = 0
STATIC = 1
KINEMATIC = 2


class Shape:
    """Base class for collision shapes."""

    kind = -1

    def sizes(self, count):
        """
        Return per-body size parameters.

        Parameters
        ----------
        count : int
            Number of bodies using this shape.

        Returns
        -------
        ndarray, shape (count, 3)
            Size parameters stored by the engine.
        """
        raise NotImplementedError


def _broadcast_positive(values, count, *, name, width):
    """
    Broadcast sizes to ``(count, width)`` and reject non-positive entries.

    Parameters
    ----------
    values : float or array_like
        Sizes to broadcast.
    count : int
        Number of bodies.
    name : str
        Parameter name used in error messages.
    width : int
        Number of components per body.

    Returns
    -------
    ndarray, shape (count, width)
        Broadcast sizes.
    """
    arr = np.asarray(values, dtype=float)
    try:
        if width == 1:
            arr = np.broadcast_to(arr.reshape(-1, 1), (count, 1))
        else:
            arr = np.broadcast_to(arr, (count, width))
    except ValueError as err:
        raise ValueError(
            f"{name} must be a scalar or have one value per body."
        ) from err
    if np.any(arr <= 0) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be positive and finite.")
    return arr.copy()


class Sphere(Shape):
    """
    A sphere.

    Parameters
    ----------
    radius : float or array_like
        Radius, or one radius per body.
    """

    kind = SPHERE

    def __init__(self, radius):
        self.radius = radius

    def sizes(self, count):
        """
        Return the radius in the first column.

        Parameters
        ----------
        count : int
            Number of bodies using this shape.

        Returns
        -------
        ndarray, shape (count, 3)
            Radii repeated across all three columns.
        """
        radii = _broadcast_positive(self.radius, count, name="radius", width=1)
        return np.repeat(radii, 3, axis=1)


def bounding_radius(size):
    """
    Return the radius of a sphere enclosing each body.

    Parameters
    ----------
    size : ndarray, shape (N, 3)
        Size parameters.

    Returns
    -------
    ndarray, shape (N,)
        Bounding radii.
    """
    return size[:, 0].copy()


def volume(size):
    """
    Return the volume of each body.

    Parameters
    ----------
    size : ndarray, shape (N, 3)
        Size parameters.

    Returns
    -------
    ndarray, shape (N,)
        Volume per body.
    """
    return 4 / 3 * np.pi * size[:, 0] ** 3


def inertia(size, mass):
    """
    Return the diagonal body-space inertia of each body.

    Parameters
    ----------
    size : ndarray, shape (N, 3)
        Size parameters.
    mass : ndarray, shape (N,)
        Body masses.

    Returns
    -------
    ndarray, shape (N, 3)
        Principal moments of inertia.
    """
    return np.repeat((0.4 * mass * size[:, 0] ** 2)[:, None], 3, axis=1)
