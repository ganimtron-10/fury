"""Generate contact points for candidate body pairs."""

from collections import namedtuple

import numpy as np

from fury.physics.shapes import SPHERE


class Contacts(
    namedtuple("Contacts", ["a", "b", "points", "normals", "separation", "feature"])
):
    """
    Contact points, one entry per point.

    Attributes
    ----------
    a, b : ndarray, shape (M,)
        Body indices. The normal points from ``a`` to ``b``.
    points : ndarray, shape (M, 3)
        World-space contact points.
    normals : ndarray, shape (M, 3)
        Unit contact normals.
    separation : ndarray, shape (M,)
        Signed gap; negative values are penetrating.
    feature : ndarray, shape (M,)
        Identifier that stays stable while the same features touch.
    """

    __slots__ = ()


def empty_contacts():
    """
    Return a contact set with no points.

    Returns
    -------
    Contacts
        Empty contact arrays.
    """
    integers = np.empty(0, dtype=np.intp)
    return Contacts(
        integers,
        integers,
        np.empty((0, 3)),
        np.empty((0, 3)),
        np.empty(0),
        integers,
    )


def merge_contacts(parts):
    """
    Concatenate contact sets.

    Parameters
    ----------
    parts : list of Contacts
        Contact sets to merge.

    Returns
    -------
    Contacts
        The merged contacts.
    """
    if not parts:
        return empty_contacts()
    return Contacts(*(np.concatenate(column) for column in zip(*parts, strict=True)))


def sphere_sphere(positions, size, i, j, margin):
    """
    Generate contacts between pairs of spheres.

    Parameters
    ----------
    positions : ndarray, shape (B, 3)
        Body positions.
    size : ndarray, shape (B, 3)
        Size parameters; radii are in the first column.
    i, j : ndarray, shape (N,)
        Pair indices.
    margin : float
        Largest gap that still yields a contact.

    Returns
    -------
    Contacts
        Generated contacts.
    """
    delta = positions[j] - positions[i]
    distance = np.linalg.norm(delta, axis=1)
    gap = distance - size[i, 0] - size[j, 0]
    keep = gap <= margin
    i, j, delta, distance, gap = (
        i[keep],
        j[keep],
        delta[keep],
        distance[keep],
        gap[keep],
    )

    normals = np.zeros_like(delta)
    separated = distance > 1e-9
    normals[separated] = delta[separated] / distance[separated, None]
    normals[~separated] = (0.0, 1.0, 0.0)
    points = positions[i] + normals * (size[i, 0] + 0.5 * gap)[:, None]
    return Contacts(i, j, points, normals, gap, np.zeros(i.size, dtype=np.intp))


def generate_contacts(world, rotation, i, j):
    """
    Generate contacts for candidate pairs of a world.

    Parameters
    ----------
    world : World
        The world being stepped.
    rotation : ndarray, shape (B, 3, 3)
        Body rotation matrices.
    i, j : ndarray, shape (N,)
        Candidate pair indices.

    Returns
    -------
    Contacts
        Contacts for all pairs.
    """
    parts = []
    both_spheres = (world._shape[i] == SPHERE) & (world._shape[j] == SPHERE)
    if both_spheres.any():
        parts.append(
            sphere_sphere(
                world._positions,
                world._size,
                i[both_spheres],
                j[both_spheres],
                world.margin,
            )
        )
    return merge_contacts(parts)
