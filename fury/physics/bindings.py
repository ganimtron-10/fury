"""
Connect physics bodies to the FURY actors that display them.

A flat square becomes a thin box whose thickness is ``_FLAT_THICKNESS`` of its
smaller side.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from fury.lib import Buffer, normals_from_vertices
from fury.physics._math import quat_to_matrix
from fury.physics.shapes import Box, Sphere

_FLAT_THICKNESS = 0.05
_IDENTITY_QUATERNION = np.array([0.0, 0.0, 0.0, 1.0])


class ActorDescription:
    """
    What the engine learned from an actor.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Body centres.
    rotations : ndarray, shape (N, 4)
        Body orientations.
    shape : Shape
        The collision shape.
    attach : callable
        Prepares the actor for simulation and returns the object that writes
        body state back to it. Nothing about the actor changes before this is
        called, so a failed add leaves the actor as it was.
    shape_kind : str
        ``"box"`` or ``"sphere"``.
    """

    def __init__(self, *, positions, rotations, shape, attach, shape_kind):
        self.positions = positions
        self.rotations = rotations
        self.shape = shape
        self.attach = attach
        self.shape_kind = shape_kind


class _Binding:
    """Base class for objects that update an actor from its bodies."""

    bodies = None

    def sync(self):
        """Write body state to the actor."""
        raise NotImplementedError

    def before_step(self, dt):
        """
        Prepare for a physics step.

        Parameters
        ----------
        dt : float
            Step duration.
        """


class TransformBinding(_Binding):
    """
    Update one actor through its transform.

    Parameters
    ----------
    actor : WorldObject
        The actor to update.
    rest_rotation : ndarray, shape (4,)
        Orientation baked into the actor's vertices.
    """

    def __init__(self, actor, rest_rotation):
        self.actor = actor
        self._inverse_rest = Rotation.from_quat(rest_rotation).inv()

    def sync(self):
        """Write position and rotation to the actor's transform."""
        world, row = self.bodies.world, self.bodies.start
        turned = Rotation.from_quat(world._rotations[row]) * self._inverse_rest
        self.actor.world.rotation = tuple(turned.as_quat())
        self.actor.world.position = tuple(world._positions[row])


class MeshBatchBinding(_Binding):
    """
    Update a mesh holding many primitives by rewriting its vertices.

    Parameters
    ----------
    actor : Mesh
        The merged mesh.
    centers : ndarray, shape (N, 3)
        Primitive centres.
    rest_rotations : ndarray, shape (N, 3, 3)
        Rotation baked into each primitive's vertices.
    """

    def __init__(self, actor, centers, rest_rotations):
        self.actor = actor
        geometry = actor.geometry
        count = len(centers)
        self._per_primitive = geometry.positions.data.shape[0] // count

        inverse = np.repeat(rest_rotations.transpose(0, 2, 1), self._per_primitive, 0)
        rest = geometry.positions.data.astype(float)
        rest -= np.repeat(centers, self._per_primitive, axis=0)
        self._rest = (inverse @ rest[..., None])[..., 0]

        self._rest_normals = None
        if not getattr(actor.material, "flat_shading", False):
            self._rest_normals = (inverse @ self._normals(geometry)[..., None])[..., 0]

    def _normals(self, geometry):
        """
        Return vertex normals, creating the buffer when the mesh has none.

        The renderer builds missing normals once from the first positions, so
        a mesh that rotates needs its own buffer.

        Parameters
        ----------
        geometry : Geometry
            The mesh geometry.

        Returns
        -------
        ndarray, shape (V, 3)
            Vertex normals.
        """
        normals = getattr(geometry, "normals", None)
        if normals is None:
            data = normals_from_vertices(
                geometry.positions.data, geometry.indices.data
            ).astype(np.float32)
            normals = Buffer(data)
            geometry.normals = normals
        return normals.data.astype(float)

    def sync(self):
        """Rewrite the mesh's vertices from body state."""
        bodies, geometry = self.bodies, self.actor.geometry
        world, span = bodies.world, bodies._range
        repeat = self._per_primitive
        rotation = np.repeat(quat_to_matrix(world._rotations[span]), repeat, axis=0)
        positions = (rotation @ self._rest[..., None])[..., 0]
        positions += np.repeat(world._positions[span], repeat, axis=0)
        geometry.positions.data[:] = positions.astype(np.float32)
        geometry.positions.update_full()
        info = self.actor.primitive_info
        info.centers = world._positions[span].copy()
        info.rotations = world._rotations[span].copy()
        if self._rest_normals is not None:
            normals = (rotation @ self._rest_normals[..., None])[..., 0]
            geometry.normals.data[:] = normals.astype(np.float32)
            geometry.normals.update_full()


class BillboardBinding(_Binding):
    """
    Update impostor spheres, which only need their centres written.

    Parameters
    ----------
    actor : Billboard
        The billboard actor.
    """

    def __init__(self, actor):
        self.actor = actor
        self._per_billboard = actor.geometry.positions.data.shape[0] // len(
            actor.billboard_centers
        )

    def sync(self):
        """Write sphere centres to the actor."""
        bodies, actor = self.bodies, self.actor
        centers = bodies.world._positions[bodies._range].astype(np.float32)
        actor.geometry.positions.data[:] = np.repeat(
            centers, self._per_billboard, axis=0
        )
        actor.geometry.positions.update_full()
        actor.billboard_centers[:] = centers


class FollowBinding(_Binding):
    """
    Make a kinematic body copy its actor's pose.

    The body's velocity comes from the change in pose, because velocity is
    what pushes other bodies. Implausibly large jumps are treated as teleports.

    Parameters
    ----------
    inner : TransformBinding
        The binding describing the actor.
    bodies : BodySet
        The kinematic body.
    """

    def __init__(self, inner, bodies):
        self.inner = inner
        self.bodies = bodies

    def sync(self):
        """Leave the actor alone, since the actor drives the body."""

    def before_step(self, dt):
        """
        Set the body's velocity to reach the actor's pose during this step.

        Parameters
        ----------
        dt : float
            Step duration.
        """
        world, row = self.bodies.world, self.bodies.start
        actor = self.inner.actor
        target = np.asarray(actor.world.position, dtype=float)
        rest = Rotation.from_quat(self.inner._inverse_rest.inv().as_quat())
        wanted = Rotation.from_quat(np.asarray(actor.world.rotation)) * rest

        linear = (target - world._positions[row]) / dt
        turn = wanted * Rotation.from_quat(world._rotations[row]).inv()
        angular = turn.as_rotvec() / dt
        if np.linalg.norm(linear) > world.max_follow_speed:
            world._positions[row] = target
            world._rotations[row] = wanted.as_quat()
            linear, angular = np.zeros(3), np.zeros(3)
        world._velocities[row, :3] = linear
        world._velocities[row, 3:] = angular


def direction_quaternions(directions):
    """
    Return the rotation ``repeat_primitive`` applies for orientation vectors.

    Parameters
    ----------
    directions : ndarray, shape (N, 3)
        Orientation vectors; the unrotated direction is +X.

    Returns
    -------
    ndarray, shape (N, 4)
        Quaternions in ``(x, y, z, w)`` order. A direction of exactly -X is
        baked in as a reflection, which no rotation can express, and yields the
        identity so the reflected vertices stay in the rest shape.
    """
    quats = np.tile(_IDENTITY_QUATERNION, (len(directions), 1))
    length = np.linalg.norm(directions, axis=1)
    use = length > 0
    unit = directions[use] / length[use, None]
    cosine = np.clip(unit[:, 0], -1.0, 1.0)
    axis = np.stack([np.zeros(unit.shape[0]), -unit[:, 2], unit[:, 1]], axis=1)
    half = np.sqrt(np.maximum(2.0 * (1.0 + cosine), 1e-18))
    rotated = np.concatenate([axis / half[:, None], 0.5 * half[:, None]], axis=1)
    rotated[cosine <= -1.0 + 1e-12] = _IDENTITY_QUATERNION
    quats[use] = rotated
    return quats


def _decompose(actor):
    """
    Split an actor's world transform into scale and rotation.

    Parameters
    ----------
    actor : WorldObject
        The actor.

    Returns
    -------
    matrix : ndarray, shape (4, 4)
        World transform.
    scale : ndarray, shape (3,)
        Scale along each axis.
    rotation : ndarray, shape (4,)
        Orientation quaternion.
    """
    matrix = np.asarray(actor.world.matrix, dtype=float)
    scale = np.linalg.norm(matrix[:3, :3], axis=0)
    if not np.all(scale > 1e-12) or np.linalg.det(matrix[:3, :3]) < 0:
        raise ValueError("Actors must have a positive, non-mirrored scale.")
    rotation = Rotation.from_matrix(matrix[:3, :3] / scale).as_quat()
    return matrix, scale, rotation


def _is_identity(matrix):
    """
    Test whether a transform does nothing.

    Parameters
    ----------
    matrix : ndarray, shape (4, 4)
        The transform.

    Returns
    -------
    bool
        True when the matrix is the identity.
    """
    return np.allclose(matrix, np.eye(4), atol=1e-9)


def _require_uniform_parent_scale(actor):
    """
    Refuse actors under a parent that stretches or shears some axes.

    A turning child of such a parent would be sheared, which a body cannot be.

    Parameters
    ----------
    actor : WorldObject
        The actor.
    """
    parent = getattr(actor, "parent", None)
    if parent is None:
        return
    linear = np.asarray(parent.world.matrix, dtype=float)[:3, :3]
    gram = linear.T @ linear
    if not np.allclose(gram, np.trace(gram) / 3.0 * np.eye(3), atol=1e-9):
        raise ValueError("Actors must not sit under a parent with non-uniform scale.")


def _require_plain_parent(actor):
    """
    Refuse actors whose vertices would be baked under a transformed parent.

    Parameters
    ----------
    actor : WorldObject
        The actor.
    """
    parent = getattr(actor, "parent", None)
    if parent is not None and not _is_identity(np.asarray(parent.world.matrix)):
        raise ValueError(
            "Actors holding many primitives must not sit under a transformed "
            "parent when added to a world."
        )


def _apply_matrix(points, matrix):
    """
    Transform points.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Input points.
    matrix : ndarray, shape (4, 4)
        Transform.

    Returns
    -------
    ndarray, shape (N, 3)
        Transformed points.
    """
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _inferred_shape(info):
    """
    Infer a collision shape from primitive metadata.

    Parameters
    ----------
    info : PrimitiveInfo
        What the actor was built from.

    Returns
    -------
    Shape or None
        The inferred shape, or None when the primitive has no known form.
    """
    if info.kind == "box":
        return Box(0.5 * info.scales)
    if info.kind == "sphere":
        return Sphere(info.radii)
    if info.kind == "disk":
        return Sphere(info.radii * info.scales[:, 0])
    if info.kind == "square":
        half = 0.5 * info.scales.copy()
        half[:, 2] = _FLAT_THICKNESS * half[:, :2].min(axis=1)
        return Box(half)
    return None


def _flat(shape):
    """
    Give a box the thickness of a flat square, whatever its scale along z.

    Parameters
    ----------
    shape : Box
        The square's box.

    Returns
    -------
    Box
        The box with its thickness reset.
    """
    half = np.array(shape.half_extents, dtype=float, ndmin=2)
    half[:, 2] = _FLAT_THICKNESS * half[:, :2].min(axis=1)
    return Box(half)


def _scaled(shape, scale, rest_rotation):
    """
    Return a shape with a transform's scale applied.

    The actor's scale acts along the actor's axes, while the shape is measured
    along each primitive's own axes, so a non-uniform scale is only possible
    when every primitive is lined up with the actor's axes.

    Parameters
    ----------
    shape : Shape
        The input shape.
    scale : ndarray, shape (3,)
        Scale along each of the actor's axes.
    rest_rotation : ndarray, shape (N, 3, 3)
        Rotation from each primitive's axes to the actor's axes.

    Returns
    -------
    Shape
        The scaled shape.
    """
    if np.allclose(scale, 1.0):
        return shape
    if isinstance(shape, Sphere):
        if not np.allclose(scale, scale[0]):
            raise ValueError("Spheres need uniform scaling.")
        return Sphere(np.asarray(shape.radius) * scale[0])
    along = _scale_along(scale, rest_rotation)
    half = np.asarray(shape.half_extents, dtype=float)
    return Box(np.broadcast_to(half, along.shape) * along)


def _scale_along(scale, rest_rotation):
    """
    Express an actor's scale along each primitive's own axes.

    Parameters
    ----------
    scale : ndarray, shape (3,)
        Scale along each of the actor's axes.
    rest_rotation : ndarray, shape (N, 3, 3)
        Rotation from each primitive's axes to the actor's axes.

    Returns
    -------
    ndarray, shape (N, 3)
        Scale along each primitive's axes.
    """
    count = len(rest_rotation)
    if np.allclose(scale, scale[0]):
        return np.full((count, 3), float(scale[0]))
    frame = np.abs(rest_rotation)
    if not np.allclose(frame, np.round(frame), atol=1e-6):
        raise ValueError(
            "A non-uniform scale needs primitives lined up with the actor's axes."
        )
    return np.round(frame).transpose(0, 2, 1) @ scale


def _kind_of(shape):
    """
    Name a shape's type.

    Parameters
    ----------
    shape : Shape
        The shape.

    Returns
    -------
    str
        ``"box"`` or ``"sphere"``.
    """
    return "box" if isinstance(shape, Box) else "sphere"


def _rebase_single(actor, center, rest_rotation, matrix, scale, rotation):
    """
    Plan moving a one-primitive mesh's origin to its centre, so it turns in place.

    Parameters
    ----------
    actor : Mesh
        The actor to change.
    center : ndarray, shape (3,)
        Primitive centre in the actor's vertices.
    rest_rotation : ndarray, shape (4,)
        Orientation baked into the vertices.
    matrix : ndarray, shape (4, 4)
        Actor's world transform.
    scale : ndarray, shape (3,)
        Actor's scale.
    rotation : ndarray, shape (4,)
        Actor's orientation.

    Returns
    -------
    position : ndarray, shape (1, 3)
        Body centre.
    body_rotation : ndarray, shape (1, 4)
        Body orientation.
    attach : callable
        Moves the vertices and returns the actor's binding.
    """
    position = _apply_matrix(center[None], matrix)
    combined = Rotation.from_quat(rotation) * Rotation.from_quat(rest_rotation)

    def attach():
        """
        Prepare the actor and return its binding.

        Returns
        -------
        _Binding
            The object that writes body state to the actor.
        """
        positions = actor.geometry.positions
        moved = positions.data.astype(float) - center
        positions.data[:] = moved.astype(np.float32)
        positions.update_full()
        actor.primitive_info.centers = np.zeros((1, 3))
        actor.world.scale = tuple(scale)
        actor.world.rotation = tuple(rotation)
        actor.world.position = tuple(position[0])
        return TransformBinding(actor, rest_rotation)

    return position, combined.as_quat()[None], attach


def describe_actor(actor, shape, *, single=False):
    """
    Work out how to simulate an actor.

    Parameters
    ----------
    actor : WorldObject
        The actor.
    shape : Shape or None
        An explicit collision shape.
    single : bool, optional
        Require the actor to be exactly one body.

    Returns
    -------
    ActorDescription
        Body data and a binding for the actor.
    """
    _require_uniform_parent_scale(actor)
    matrix, scale, rotation = _decompose(actor)

    if hasattr(actor, "billboard_centers"):
        return _describe_billboard(actor, shape, matrix, scale, single)

    info = getattr(actor, "primitive_info", None)
    if info is not None:
        return _describe_primitives(actor, info, shape, matrix, scale, rotation, single)

    if shape is None:
        raise TypeError(
            f"Cannot work out a collision shape for {type(actor).__name__}. "
            "Pass one with shape=physics.Box(...) or shape=physics.Sphere(...)."
        )
    return ActorDescription(
        positions=matrix[:3, 3][None],
        rotations=rotation[None],
        shape=_scaled(shape, scale, np.eye(3)[None]),
        attach=lambda: TransformBinding(actor, _IDENTITY_QUATERNION),
        shape_kind=_kind_of(shape),
    )


def _describe_billboard(actor, shape, matrix, scale, single):
    """
    Describe an impostor-sphere actor.

    Parameters
    ----------
    actor : Billboard
        The actor.
    shape : Shape or None
        An explicit collision shape.
    matrix : ndarray, shape (4, 4)
        World transform.
    scale : ndarray, shape (3,)
        Actor scale.
    single : bool
        Require exactly one body.

    Returns
    -------
    ActorDescription
        Body data and a binding for the actor.
    """
    if single:
        raise TypeError("Impostor spheres cannot be driven by their actor.")
    if not np.allclose(scale, 1.0):
        raise ValueError("Impostor spheres cannot be added with a scale applied.")
    centers = np.asarray(actor.billboard_centers, dtype=float)
    moved = centers
    if not _is_identity(matrix):
        _require_plain_parent(actor)
        moved = _apply_matrix(centers, matrix)
    if shape is None:
        radii = getattr(actor, "billboard_radii", None)
        if radii is None:
            raise TypeError(
                "Cannot work out a collision shape for these billboards. "
                "Pass one with shape=physics.Sphere(...)."
            )
        shape = Sphere(np.asarray(radii, dtype=float))

    def attach():
        """
        Prepare the actor and return its binding.

        Returns
        -------
        _Binding
            The object that writes body state to the actor.
        """
        if moved is not centers:
            positions = actor.geometry.positions
            repeat = positions.data.shape[0] // len(centers)
            positions.data[:] = np.repeat(moved, repeat, axis=0).astype(np.float32)
            positions.update_full()
            actor.billboard_centers[:] = moved.astype(np.float32)
            actor.local.matrix = np.eye(4)
        return BillboardBinding(actor)

    return ActorDescription(
        positions=moved,
        rotations=np.tile(_IDENTITY_QUATERNION, (len(centers), 1)),
        shape=shape,
        attach=attach,
        shape_kind=_kind_of(shape),
    )


def _describe_primitives(actor, info, shape, matrix, scale, rotation, single):
    """
    Describe an actor built by ``actor_from_primitive``.

    Parameters
    ----------
    actor : Mesh
        The actor.
    info : PrimitiveInfo
        What the actor was built from.
    shape : Shape or None
        An explicit collision shape.
    matrix : ndarray, shape (4, 4)
        World transform.
    scale : ndarray, shape (3,)
        Actor scale.
    rotation : ndarray, shape (4,)
        Actor orientation.
    single : bool
        Require exactly one body.

    Returns
    -------
    ActorDescription
        Body data and a binding for the actor.
    """
    inferred = shape is None
    shape = shape if shape is not None else _inferred_shape(info)
    if shape is None:
        raise TypeError(
            f"Cannot work out a collision shape for this {type(actor).__name__}. "
            "Pass one with shape=physics.Box(...) or shape=physics.Sphere(...)."
        )
    rest = getattr(info, "rotations", None)
    if rest is None:
        rest = direction_quaternions(info.directions)
    rest = np.asarray(rest, dtype=float)
    shape = _scaled(shape, scale, quat_to_matrix(rest))
    if inferred and info.kind == "square":
        shape = _flat(shape)
    count = len(info.centers)
    if single and count != 1:
        raise TypeError("This actor holds many primitives; expected exactly one.")

    if count == 1:
        position, body_rotation, attach = _rebase_single(
            actor, info.centers[0], rest[0], matrix, scale, rotation
        )
        return ActorDescription(
            positions=position,
            rotations=body_rotation,
            shape=shape,
            attach=attach,
            shape_kind=_kind_of(shape),
        )

    centers = np.asarray(info.centers, dtype=float).copy()
    baked = not _is_identity(matrix)
    body_centers, body_rest = centers, rest
    if baked:
        _require_plain_parent(actor)
        along = _scale_along(scale, quat_to_matrix(rest))
        body_centers = _apply_matrix(centers, matrix)
        body_rest = (Rotation.from_quat(rotation) * Rotation.from_quat(rest)).as_quat()

    def attach():
        """
        Prepare the actor and return its binding.

        Returns
        -------
        _Binding
            The object that writes body state to the actor.
        """
        if baked:
            _bake_transform(actor, matrix, rotation)
            info.centers = body_centers.copy()
            info.rotations = body_rest.copy()
            info.scales = np.asarray(info.scales, dtype=float) * along
            if info.kind == "sphere" and info.radii is not None:
                info.radii = np.asarray(info.radii, dtype=float) * along[:, 0]
        return MeshBatchBinding(actor, body_centers, quat_to_matrix(body_rest))

    return ActorDescription(
        positions=body_centers,
        rotations=body_rest,
        shape=shape,
        attach=attach,
        shape_kind=_kind_of(shape),
    )


def _bake_transform(actor, matrix, rotation):
    """
    Move an actor's transform into its vertices and normals.

    Parameters
    ----------
    actor : Mesh
        The actor.
    matrix : ndarray, shape (4, 4)
        Actor's world transform.
    rotation : ndarray, shape (4,)
        Actor's orientation.
    """
    geometry = actor.geometry
    moved = geometry.positions.data.astype(float) @ matrix[:3, :3].T
    geometry.positions.data[:] = (moved + matrix[:3, 3]).astype(np.float32)
    geometry.positions.update_full()
    normals = getattr(geometry, "normals", None)
    if normals is not None:
        turn = quat_to_matrix(rotation[None])[0]
        normals.data[:] = (normals.data.astype(float) @ turn.T).astype(np.float32)
        normals.update_full()
    actor.local.matrix = np.eye(4)
