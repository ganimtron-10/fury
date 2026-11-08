"""Generate contact points for candidate body pairs."""

from collections import namedtuple

import numpy as np

from fury.physics._math import cross, dot
from fury.physics.shapes import BOX, SPHERE


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


def box_sphere(positions, rotation, size, box, sphere, margin):
    """
    Generate contacts between boxes and spheres.

    Parameters
    ----------
    positions : ndarray, shape (B, 3)
        Body positions.
    rotation : ndarray, shape (B, 3, 3)
        Body rotation matrices.
    size : ndarray, shape (B, 3)
        Size parameters.
    box, sphere : ndarray, shape (N,)
        Pair indices. Normals point from the box to the sphere.
    margin : float
        Largest gap that still yields a contact.

    Returns
    -------
    Contacts
        Generated contacts.
    """
    axes = rotation[box]
    half = size[box]
    local = (axes.transpose(0, 2, 1) @ (positions[sphere] - positions[box])[..., None])[
        ..., 0
    ]
    outside = local - np.clip(local, -half, half)
    distance = np.linalg.norm(outside, axis=1)
    normal_local = outside / np.maximum(distance, 1e-9)[:, None]
    gap = distance - size[sphere, 0]

    inside = np.flatnonzero(distance <= 1e-9)
    if inside.size:
        depth = half[inside] - np.abs(local[inside])
        axis = depth.argmin(axis=1)
        normal_local[inside] = 0.0
        normal_local[inside, axis] = np.where(local[inside, axis] >= 0, 1.0, -1.0)
        gap[inside] = -depth[np.arange(inside.size), axis] - size[sphere[inside], 0]

    keep = gap <= margin
    box, sphere, gap = box[keep], sphere[keep], gap[keep]
    normals = (axes[keep] @ normal_local[keep][..., None])[..., 0]
    points = positions[sphere] - normals * (size[sphere, 0] + 0.5 * gap)[:, None]
    return Contacts(
        box, sphere, points, normals, gap, np.zeros(box.size, dtype=np.intp)
    )


_EDGE_A = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
_EDGE_B = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
_EDGE_A1, _EDGE_A2 = (_EDGE_A + 1) % 3, (_EDGE_A + 2) % 3
_EDGE_B1, _EDGE_B2 = (_EDGE_B + 1) % 3, (_EDGE_B + 2) % 3
_SUPPORT_U = np.array([1.0, 1.0, -1.0, -1.0])
_SUPPORT_V = np.array([1.0, -1.0, 1.0, -1.0])
_EDGE_AXIS_MIN_ADVANTAGE = 0.01
_CLIP_EPSILON = 1e-6
_EDGE_FEATURE_OFFSET = 100


def _box_face_contacts(
    positions, rotation, size, reference, incident, normals, axis, margin
):
    """
    Clip incident faces against reference faces.

    Rather than clipping each pair in a loop, every point the clipped polygon
    could contain is listed up front: 4 incident corners, 4 reference corners
    and 16 edge crossings. Masks then pick the real ones.

    Parameters
    ----------
    positions : ndarray, shape (B, 3)
        Body positions.
    rotation : ndarray, shape (B, 3, 3)
        Body rotation matrices.
    size : ndarray, shape (B, 3)
        Half extents.
    reference, incident : ndarray, shape (N,)
        Box indices. Normals point from reference to incident.
    normals : ndarray, shape (N, 3)
        Contact normals.
    axis : ndarray, shape (N,)
        Local axis of the reference face.
    margin : float
        Largest gap that still yields a contact.

    Returns
    -------
    Contacts
        Up to four contacts per pair.
    """
    pairs = reference.size
    every = np.arange(pairs)
    ref_axes, inc_axes = rotation[reference], rotation[incident]
    ref_half, inc_half = size[reference], size[incident]
    ref_pos, inc_pos = positions[reference], positions[incident]

    axis_u, axis_v = (axis + 1) % 3, (axis + 2) % 3
    u, v = ref_axes[every, :, axis_u], ref_axes[every, :, axis_v]
    half_n = ref_half[every, axis]
    half_u, half_v = ref_half[every, axis_u], ref_half[every, axis_v]

    alignment = (inc_axes.transpose(0, 2, 1) @ normals[..., None])[..., 0]
    inc_axis = np.abs(alignment).argmax(axis=1)
    inc_sign = -np.sign(alignment[every, inc_axis])
    inc_sign[inc_sign == 0] = 1.0
    axis_p, axis_q = (inc_axis + 1) % 3, (inc_axis + 2) % 3
    center = (
        inc_pos
        + inc_axes[every, :, inc_axis] * (inc_sign * inc_half[every, inc_axis])[:, None]
    )
    span_p = inc_axes[every, :, axis_p] * inc_half[every, axis_p][:, None]
    span_q = inc_axes[every, :, axis_q] * inc_half[every, axis_q][:, None]
    corners = np.stack(
        [
            center + span_p + span_q,
            center - span_p + span_q,
            center - span_p - span_q,
            center + span_p - span_q,
        ],
        axis=1,
    )

    relative = corners - ref_pos[:, None]
    pu = dot(relative, u[:, None])
    pv = dot(relative, v[:, None])
    height = dot(relative, normals[:, None]) - half_n[:, None]
    limit_u, limit_v = half_u[:, None], half_v[:, None]

    inside_ref = (np.abs(pu) <= limit_u + _CLIP_EPSILON) & (
        np.abs(pv) <= limit_v + _CLIP_EPSILON
    )

    corner_u = np.stack([half_u, -half_u, -half_u, half_u], axis=1)
    corner_v = np.stack([half_v, half_v, -half_v, -half_v], axis=1)
    edge_u, edge_v = np.roll(pu, -1, 1) - pu, np.roll(pv, -1, 1) - pv
    side = edge_u[:, :, None] * (corner_v[:, None] - pv[:, :, None])
    side = side - edge_v[:, :, None] * (corner_u[:, None] - pu[:, :, None])
    inside_inc = (side >= -_CLIP_EPSILON).all(1) | (side <= _CLIP_EPSILON).all(1)

    e1u, e1v, e1h = (
        pu[:, 1] - pu[:, 0],
        pv[:, 1] - pv[:, 0],
        height[:, 1] - height[:, 0],
    )
    e2u, e2v, e2h = (
        pu[:, 3] - pu[:, 0],
        pv[:, 3] - pv[:, 0],
        height[:, 3] - height[:, 0],
    )
    determinant = e1u * e2v - e2u * e1v
    solvable = np.abs(determinant) > 1e-9
    determinant = np.where(solvable, determinant, 1.0)[:, None]
    du, dv = corner_u - pu[:, 0, None], corner_v - pv[:, 0, None]
    alpha = (du * e2v[:, None] - dv * e2u[:, None]) / determinant
    beta = (dv * e1u[:, None] - du * e1v[:, None]) / determinant
    corner_height = height[:, 0, None] + alpha * e1h[:, None] + beta * e2h[:, None]
    inside_inc &= solvable[:, None]

    start_u, end_u = pu, np.roll(pu, -1, 1)
    start_v, end_v = pv, np.roll(pv, -1, 1)
    start_h, end_h = height, np.roll(height, -1, 1)
    lines = np.stack([half_u, -half_u, half_v, -half_v], axis=1)[:, :, None]
    along_u = np.array([True, True, False, False])[None, :, None]
    a0 = np.where(along_u, start_u[:, None], start_v[:, None])
    a1 = np.where(along_u, end_u[:, None], end_v[:, None])
    b0 = np.where(along_u, start_v[:, None], start_u[:, None])
    b1 = np.where(along_u, end_v[:, None], end_u[:, None])
    bound = np.where(along_u, half_v[:, None, None], half_u[:, None, None])
    span = a1 - a0
    crosses = np.abs(span) > 1e-9
    t = (lines - a0) / np.where(crosses, span, 1.0)
    other = b0 + t * (b1 - b0)
    on_edge = crosses & (t > 0) & (t < 1) & (np.abs(other) <= bound + _CLIP_EPSILON)
    edge_height = start_h[:, None] + t * (end_h - start_h)[:, None]
    edge_u_pos = np.where(along_u, lines, other).reshape(pairs, -1)
    edge_v_pos = np.where(along_u, other, lines).reshape(pairs, -1)

    all_u = np.concatenate([pu, corner_u, edge_u_pos], axis=1)
    all_v = np.concatenate([pv, corner_v, edge_v_pos], axis=1)
    all_h = np.concatenate([height, corner_height, edge_height.reshape(pairs, -1)], 1)
    valid = np.concatenate([inside_ref, inside_inc, on_edge.reshape(pairs, -1)], 1)
    valid &= all_h <= margin

    score = all_u[:, :, None] * _SUPPORT_U + all_v[:, :, None] * _SUPPORT_V
    score = np.where(valid[:, :, None], score, -np.inf)
    chosen = np.empty((pairs, 4), dtype=np.intp)
    for direction in range(4):
        chosen[:, direction] = score[:, :, direction].argmax(axis=1)
        score[every, chosen[:, direction]] = -np.inf
    chosen = np.sort(chosen, axis=1)
    use = np.ones_like(chosen, dtype=bool)
    use[:, 1:] = chosen[:, 1:] != chosen[:, :-1]
    use &= valid[every[:, None], chosen]

    row, column = np.nonzero(use)
    candidate = chosen[row, column]
    gap = all_h[row, candidate]
    points = (
        ref_pos[row]
        + u[row] * all_u[row, candidate][:, None]
        + v[row] * all_v[row, candidate][:, None]
        + normals[row] * (half_n[row] + 0.5 * gap)[:, None]
    )
    return Contacts(reference[row], incident[row], points, normals[row], gap, candidate)


def _box_edge_contacts(positions, rotation, size, ia, ib, delta, best, separation):
    """
    Generate one contact for each pair touching edge to edge.

    Parameters
    ----------
    positions : ndarray, shape (B, 3)
        Body positions.
    rotation : ndarray, shape (B, 3, 3)
        Body rotation matrices.
    size : ndarray, shape (B, 3)
        Half extents.
    ia, ib : ndarray, shape (N,)
        Box indices.
    delta : ndarray, shape (N, 3)
        Offset from the first box to the second.
    best : ndarray, shape (N,)
        Index of the separating edge pair.
    separation : ndarray, shape (N,)
        Signed gap along that edge axis.

    Returns
    -------
    Contacts
        One contact per pair.
    """
    every = np.arange(ia.size)
    axes_a, axes_b = rotation[ia], rotation[ib]
    edge_a, edge_b = _EDGE_A[best], _EDGE_B[best]
    dir_a, dir_b = axes_a[every, :, edge_a], axes_b[every, :, edge_b]
    normals = cross(dir_a, dir_b)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    normals[dot(normals, delta) < 0] *= -1.0

    sign_a = np.sign((axes_a.transpose(0, 2, 1) @ normals[..., None])[..., 0])
    sign_b = np.sign((axes_b.transpose(0, 2, 1) @ normals[..., None])[..., 0])
    sign_a[every, edge_a] = 0.0
    sign_b[every, edge_b] = 0.0
    point_a = positions[ia] + (axes_a @ (sign_a * size[ia])[..., None])[..., 0]
    point_b = positions[ib] - (axes_b @ (sign_b * size[ib])[..., None])[..., 0]

    offset = point_a - point_b
    cosine = dot(dir_a, dir_b)
    project_a, project_b = dot(dir_a, offset), dot(dir_b, offset)
    denominator = np.maximum(1.0 - cosine * cosine, 1e-9)
    along_a = (cosine * project_b - project_a) / denominator
    along_b = (project_b - cosine * project_a) / denominator
    points = 0.5 * (
        point_a + dir_a * along_a[:, None] + point_b + dir_b * along_b[:, None]
    )
    return Contacts(ia, ib, points, normals, separation, _EDGE_FEATURE_OFFSET + best)


def box_box(positions, rotation, size, ia, ib, margin):
    """
    Generate contacts between pairs of boxes with the separating-axis test.

    Parameters
    ----------
    positions : ndarray, shape (B, 3)
        Body positions.
    rotation : ndarray, shape (B, 3, 3)
        Body rotation matrices.
    size : ndarray, shape (B, 3)
        Half extents.
    ia, ib : ndarray, shape (N,)
        Pair indices.
    margin : float
        Largest gap that still yields a contact.

    Returns
    -------
    Contacts
        Generated contacts.
    """
    axes_a, axes_b, half_a, half_b = rotation[ia], rotation[ib], size[ia], size[ib]
    delta = positions[ib] - positions[ia]
    transpose_a = axes_a.transpose(0, 2, 1)
    relative = transpose_a @ axes_b
    magnitude = np.abs(relative) + 1e-6
    offset_a = (transpose_a @ delta[..., None])[..., 0]
    offset_b = (relative.transpose(0, 2, 1) @ offset_a[..., None])[..., 0]

    face_a = np.abs(offset_a) - half_a - (magnitude @ half_b[..., None])[..., 0]
    face_b = np.abs(offset_b) - half_b
    face_b -= (magnitude.transpose(0, 2, 1) @ half_a[..., None])[..., 0]

    distance = np.abs(
        offset_a[:, _EDGE_A2] * relative[:, _EDGE_A1, _EDGE_B]
        - offset_a[:, _EDGE_A1] * relative[:, _EDGE_A2, _EDGE_B]
    )
    reach_a = (
        half_a[:, _EDGE_A1] * magnitude[:, _EDGE_A2, _EDGE_B]
        + half_a[:, _EDGE_A2] * magnitude[:, _EDGE_A1, _EDGE_B]
    )
    reach_b = (
        half_b[:, _EDGE_B1] * magnitude[:, _EDGE_A, _EDGE_B2]
        + half_b[:, _EDGE_B2] * magnitude[:, _EDGE_A, _EDGE_B1]
    )
    length = np.sqrt(np.maximum(1.0 - relative[:, _EDGE_A, _EDGE_B] ** 2, 0.0))
    edge = np.where(
        length > 1e-3,
        (distance - reach_a - reach_b) / np.maximum(length, 1e-3),
        -np.inf,
    )

    max_a, max_b, max_edge = face_a.max(1), face_b.max(1), edge.max(1)
    touching = (max_a <= margin) & (max_b <= margin) & (max_edge <= margin)
    if not touching.any():
        return empty_contacts()
    is_edge = touching & (
        max_edge > np.maximum(max_a, max_b) + _EDGE_AXIS_MIN_ADVANTAGE
    )
    is_face = touching & ~is_edge

    parts = []
    if is_face.any():
        face = np.flatnonzero(is_face)
        use_b = max_b[face] > max_a[face] + 1e-3
        axis = np.where(use_b, face_b[face].argmax(1), face_a[face].argmax(1))
        which = np.arange(face.size)
        reference = np.where(use_b, ib[face], ia[face])
        incident = np.where(use_b, ia[face], ib[face])
        sign = np.where(
            use_b, -np.sign(offset_b[face, axis]), np.sign(offset_a[face, axis])
        )
        sign[sign == 0] = 1.0
        normals = rotation[reference][which, :, axis] * sign[:, None]
        parts.append(
            _box_face_contacts(
                positions, rotation, size, reference, incident, normals, axis, margin
            )
        )
    if is_edge.any():
        pick = np.flatnonzero(is_edge)
        parts.append(
            _box_edge_contacts(
                positions,
                rotation,
                size,
                ia[pick],
                ib[pick],
                delta[pick],
                edge[pick].argmax(1),
                max_edge[pick],
            )
        )
    return merge_contacts(parts)


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
    x, size, margin = world._positions, world._size, world.margin
    shape_i, shape_j = world._shape[i], world._shape[j]

    both_spheres = (shape_i == SPHERE) & (shape_j == SPHERE)
    if both_spheres.any():
        parts.append(sphere_sphere(x, size, i[both_spheres], j[both_spheres], margin))

    mixed = ((shape_i == BOX) & (shape_j == SPHERE)) | (
        (shape_i == SPHERE) & (shape_j == BOX)
    )
    if mixed.any():
        first, second = i[mixed], j[mixed]
        swap = shape_i[mixed] == SPHERE
        box = np.where(swap, second, first)
        sphere = np.where(swap, first, second)
        parts.append(box_sphere(x, rotation, size, box, sphere, margin))

    both_boxes = (shape_i == BOX) & (shape_j == BOX)
    if both_boxes.any():
        parts.append(box_box(x, rotation, size, i[both_boxes], j[both_boxes], margin))
    return merge_contacts(parts)
