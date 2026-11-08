"""
Find pairs of bodies that may be touching.

Bodies whose radii differ by less than ``SIZE_CLASS_RATIO`` share a search
tree, so one big floor does not inflate the search radius for everything else.
"""

import numpy as np
from scipy.spatial import cKDTree

SIZE_CLASS_RATIO = 4.0


def _aabb_filter(i, j, positions, extents, margin):
    """
    Keep candidate pairs whose bounding boxes overlap.

    Parameters
    ----------
    i, j : ndarray, shape (N,)
        Candidate pair indices.
    positions : ndarray, shape (B, 3)
        Body positions.
    extents : ndarray, shape (B, 3)
        Bounding-box half widths.
    margin : float
        Extra distance at which a pair is kept.

    Returns
    -------
    tuple of ndarray
        Filtered ``(i, j)``.
    """
    apart = np.abs(positions[i] - positions[j]) - extents[i] - extents[j]
    keep = (apart <= margin).all(axis=1)
    return i[keep], j[keep]


def find_pairs(positions, extents, radii, candidates, margin):
    """
    Find every pair of candidate bodies with overlapping bounds.

    Parameters
    ----------
    positions : ndarray, shape (B, 3)
        Body positions.
    extents : ndarray, shape (B, 3)
        Bounding-box half widths.
    radii : ndarray, shape (B,)
        Bounding-sphere radii.
    candidates : ndarray, shape (B,)
        Boolean mask of the bodies to consider.
    margin : float
        Extra distance at which a pair is kept.

    Returns
    -------
    tuple of ndarray
        Arrays ``(i, j)`` of body indices with ``i < j``.
    """
    index = np.flatnonzero(candidates)
    if index.size < 2:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty

    body_radii = radii[index]
    level = np.log(body_radii / body_radii.min()) / np.log(SIZE_CLASS_RATIO)
    level = np.floor(level).astype(np.intp)
    groups = [index[level == value] for value in np.unique(level)]
    trees = [cKDTree(positions[group]) for group in groups]
    reaches = [radii[group].max() for group in groups]

    found_i, found_j = [], []
    for first, (group, tree, reach) in enumerate(
        zip(groups, trees, reaches, strict=True)
    ):
        pairs = tree.query_pairs(2.0 * reach + margin, output_type="ndarray")
        found_i.append(group[pairs[:, 0]])
        found_j.append(group[pairs[:, 1]])
        for second in range(first + 1, len(groups)):
            near = tree.sparse_distance_matrix(
                trees[second], reach + reaches[second] + margin, output_type="ndarray"
            )
            found_i.append(group[near["i"]])
            found_j.append(groups[second][near["j"]])

    i, j = np.concatenate(found_i), np.concatenate(found_j)
    low, high = np.minimum(i, j), np.maximum(i, j)
    return _aabb_filter(low, high, positions, extents, margin)
