"""The physics world: body state and stepping."""

import logging

import numpy as np

from fury.physics._math import cross, quat_to_matrix, world_inverse_inertia
from fury.physics.bindings import FollowBinding, describe_actor
from fury.physics.bodies import BodySet
from fury.physics.broadphase import find_pairs
from fury.physics.kinematics import Container, KinematicBodySet
from fury.physics.narrowphase import (
    container_contacts,
    empty_contacts,
    generate_contacts,
    merge_contacts,
)
from fury.physics.shapes import (
    BOX,
    CONTAINER,
    DYNAMIC,
    KINEMATIC,
    STATIC,
    Shape,
    bounding_radius,
    inertia,
    volume,
)
from fury.physics.solver import Solver

_KINDS = {"dynamic": DYNAMIC, "static": STATIC, "kinematic": KINEMATIC}


def _default_state(count):
    """
    Return default per-body state arrays.

    Parameters
    ----------
    count : int
        Number of bodies.

    Returns
    -------
    dict
        Arrays by attribute name.
    """
    return {
        "_positions": np.zeros((count, 3)),
        "_rotations": np.tile([0.0, 0.0, 0.0, 1.0], (count, 1)),
        "_velocities": np.zeros((count, 6)),
        "_force": np.zeros((count, 6)),
        "_inverse_mass": np.zeros(count),
        "_inverse_inertia": np.zeros((count, 3)),
        "_shape": np.zeros(count, dtype=np.intp),
        "_size": np.zeros((count, 3)),
        "_radius": np.zeros(count),
        "_kind": np.zeros(count, dtype=np.intp),
        "_friction": np.zeros(count),
        "_restitution": np.zeros(count),
        "_linear_damping": np.zeros(count),
        "_angular_damping": np.zeros(count),
        "_gravity_scale": np.ones(count),
        "_holder": np.full(count, -1, dtype=np.intp),
    }


def _per_body(value, count, name):
    """
    Broadcast a per-body option.

    Parameters
    ----------
    value : float or array_like
        A scalar, or one value per body.
    count : int
        Number of bodies.
    name : str
        Name used in error messages.

    Returns
    -------
    ndarray, shape (count,)
        One value per body.
    """
    try:
        return np.broadcast_to(np.asarray(value, dtype=float), (count,)).copy()
    except ValueError as err:
        raise ValueError(
            f"{name} must be a scalar or have one value per body."
        ) from err


class World:
    """
    A collection of rigid bodies stepped together.

    Small, fast bodies can pass through each other between steps: a pair is only
    tested where it is at the start of each step.

    Parameters
    ----------
    gravity : array_like, shape (3,), optional
        Gravitational acceleration.
    dt : float, optional
        Duration of one physics step, in seconds.
    substeps : int, optional
        Solver substeps per step. More substeps are more accurate and cost
        roughly proportionally more.
    max_colors : int, optional
        Contact batches solved one after another. More batches give stiffer
        stacks; zero solves everything at once.
    contact_margin : float, optional
        Gap at which contacts are created. By default it is a tenth of the
        median body radius.
    contact_hertz : float, optional
        Stiffness of contact recovery. Capped at a quarter of the substep rate,
        above which the solver is unstable.
    contact_damping : float, optional
        Damping ratio of contact recovery.
    max_push_speed : float, optional
        Largest speed at which overlapping bodies are pushed apart.
    restitution_threshold : float, optional
        Approach speed below which contacts do not bounce.
    """

    def __init__(
        self,
        *,
        gravity=(0.0, -9.81, 0.0),
        dt=1 / 60,
        substeps=4,
        max_colors=8,
        contact_margin=None,
        contact_hertz=30.0,
        contact_damping=10.0,
        max_push_speed=3.0,
        restitution_threshold=1.0,
    ):
        if not dt > 0:
            raise ValueError("dt must be positive.")
        if substeps < 1:
            raise ValueError("substeps must be at least 1.")
        if max_colors < 0:
            raise ValueError("max_colors must not be negative.")
        if not contact_hertz > 0:
            raise ValueError("contact_hertz must be positive.")
        if contact_damping < 0:
            raise ValueError("contact_damping must not be negative.")
        if max_push_speed <= 0:
            raise ValueError("max_push_speed must be positive.")
        if contact_margin is not None and not contact_margin >= 0:
            raise ValueError("contact_margin must not be negative.")
        if not restitution_threshold >= 0:
            raise ValueError("restitution_threshold must not be negative.")
        self.gravity = np.asarray(gravity, dtype=float)
        if self.gravity.shape != (3,) or not np.all(np.isfinite(self.gravity)):
            raise ValueError("gravity must contain three finite values.")
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.max_colors = int(max_colors)
        self.contact_hertz = float(contact_hertz)
        self.contact_damping = float(contact_damping)
        self.max_push_speed = float(max_push_speed)
        self.max_follow_speed = 100.0
        self.restitution_threshold = float(restitution_threshold)
        self._user_margin = contact_margin
        self.margin = 0.04 if contact_margin is None else float(contact_margin)

        self.time = 0.0
        self.step_count = 0
        self._sets = []
        self._containers = []
        self._container_serials = 0
        self._bindings = []
        self.contacts = empty_contacts()
        self._solver = Solver()
        self._allocate(0)

    def _allocate(self, count):
        """
        Create empty state arrays.

        Parameters
        ----------
        count : int
            Number of bodies.
        """
        for name, rows in _default_state(count).items():
            setattr(self, name, rows)

    @property
    def sets(self):
        """
        Return every set of bodies in the world.

        Returns
        -------
        tuple of BodySet
            Every set of bodies in the world.
        """
        return tuple(self._sets)

    @property
    def containers(self):
        """
        Return every container in the world.

        Returns
        -------
        tuple of Container
            Every container in the world.
        """
        return tuple(self._containers)

    @property
    def bindings(self):
        """
        Return the objects that connect bodies to actors.

        Returns
        -------
        tuple
            Every actor binding in the world.
        """
        return tuple(self._bindings)

    @property
    def count(self):
        """
        Return number of bodies in the world.

        Returns
        -------
        int
            Number of bodies in the world.
        """
        return self._positions.shape[0]

    def _append(self, **columns):
        """
        Append rows to every state array.

        Parameters
        ----------
        **columns
            New rows, by state array name; omitted arrays get default rows.

        Returns
        -------
        int
            Index of the first new body.
        """
        start = self.count
        added = len(columns["_positions"])
        for name, default in _default_state(added).items():
            rows = columns.get(name, default)
            setattr(self, name, np.concatenate([getattr(self, name), rows]))
        self._solver.clear_cache()
        return start

    def _update_margin(self):
        """Refresh the automatic contact margin after bodies change."""
        if self._user_margin is None and self.count:
            bodies = self._radius[self._kind == DYNAMIC]
            if bodies.size:
                self.margin = 0.1 * float(np.median(bodies))

    def add_bodies(
        self,
        positions,
        shape,
        *,
        kind="dynamic",
        density=None,
        mass=None,
        velocities=None,
        restitution=0.2,
        friction=0.5,
        linear_damping=None,
        angular_damping=None,
        gravity_scale=None,
    ):
        """
        Add bodies that share a shape description.

        Parameters
        ----------
        positions : array_like, shape (N, 3)
            Body centres.
        shape : Shape
            Collision shape; its sizes may be per-body.
        kind : {"dynamic", "static", "kinematic"}, optional
            ``"dynamic"`` bodies respond to forces, ``"static"`` bodies never
            move, and ``"kinematic"`` bodies move only as they are told to.
        density : float or array_like, optional
            Mass per unit volume, 1.0 by default. Ignored, with a warning, when
            ``mass`` is given or the bodies are not dynamic.
        mass : float or array_like, optional
            Mass of each body. Ignored, with a warning, unless the bodies are
            dynamic.
        velocities : array_like, shape (3,) or (N, 3), optional
            Initial velocities. Not allowed for static bodies.
        restitution : float or array_like, optional
            Bounciness, from 0 to 1.
        friction : float or array_like, optional
            Friction coefficient.
        linear_damping, angular_damping : float or array_like, optional
            Velocity loss per second, 0.0 and 0.05 by default. Ignored, with a
            warning, unless the bodies are dynamic.
        gravity_scale : float or array_like, optional
            Multiplier applied to gravity, 1.0 by default. Ignored, with a
            warning, unless the bodies are dynamic.

        Returns
        -------
        BodySet
            A handle to the new bodies.
        """
        positions = np.array(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3).")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions must be finite.")
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {sorted(_KINDS)}.")
        count = len(positions)
        if count == 0:
            raise ValueError("At least one body is required.")

        if not isinstance(shape, Shape):
            raise TypeError("shape must be a fury.physics Shape, such as Sphere.")
        code = np.full(count, shape.kind, dtype=np.intp)
        size = shape.sizes(count)
        kind_code = _KINDS[kind]

        im, ii = np.zeros(count), np.zeros((count, 3))
        options = {
            "mass": mass,
            "density": density,
            "linear_damping": linear_damping,
            "angular_damping": angular_damping,
            "gravity_scale": gravity_scale,
        }
        ignored = sorted(key for key, value in options.items() if value is not None)
        if kind_code != DYNAMIC and ignored:
            logging.warning("%s ignored for %s bodies.", ", ".join(ignored), kind)
        if kind_code == DYNAMIC:
            if mass is not None and density is not None:
                logging.warning("density is ignored because mass was given.")
            if mass is None:
                density = 1.0 if density is None else density
                body_mass = _per_body(density, count, "density") * volume(code, size)
            else:
                body_mass = _per_body(mass, count, "mass")
            if np.any(body_mass <= 0) or not np.all(np.isfinite(body_mass)):
                raise ValueError("density and mass must be positive and finite.")
            im = 1.0 / body_mass
            ii = 1.0 / inertia(code, size, body_mass)

        frictions = _per_body(friction, count, "friction")
        if np.any(frictions < 0) or not np.all(np.isfinite(frictions)):
            raise ValueError("friction must not be negative.")
        restitutions = _per_body(restitution, count, "restitution")
        if not np.all((restitutions >= 0) & (restitutions <= 1)):
            raise ValueError("restitution must be between 0 and 1.")
        linear_loss = _per_body(
            0.0 if linear_damping is None else linear_damping, count, "linear_damping"
        )
        angular_loss = _per_body(
            0.05 if angular_damping is None else angular_damping,
            count,
            "angular_damping",
        )
        if not np.all((linear_loss >= 0) & (angular_loss >= 0)):
            raise ValueError("Damping must not be negative.")

        gravity_scales = _per_body(
            1.0 if gravity_scale is None else gravity_scale, count, "gravity_scale"
        )
        if not np.all(np.isfinite(gravity_scales)):
            raise ValueError("gravity_scale must be finite.")

        velocity = np.zeros((count, 6))
        if velocities is not None:
            if kind_code == STATIC:
                raise ValueError("Static bodies cannot be given a velocity.")
            try:
                velocity[:, :3] = np.broadcast_to(
                    np.asarray(velocities, dtype=float), (count, 3)
                )
            except ValueError as err:
                raise ValueError(
                    "velocities must have shape (3,) or one row per body."
                ) from err
            if not np.all(np.isfinite(velocity)):
                raise ValueError("velocities must be finite.")

        start = self._append(
            _positions=positions,
            _velocities=velocity,
            _inverse_mass=im,
            _inverse_inertia=ii,
            _shape=code,
            _size=size,
            _radius=bounding_radius(code, size),
            _kind=np.full(count, kind_code, dtype=np.intp),
            _friction=frictions,
            _restitution=restitutions,
            _linear_damping=linear_loss,
            _angular_damping=angular_loss,
            _gravity_scale=gravity_scales,
        )
        kind_class = KinematicBodySet if kind_code == KINEMATIC else BodySet
        bodies = kind_class(self, start, count)
        self._sets.append(bodies)
        self._update_margin()
        return bodies

    def add(self, actor, *, shape=None, follow_actor=False, **body_options):
        """
        Add a FURY actor to the simulation.

        Sizes come from the actor. An actor holding many primitives, such as
        ``actor.box(centers=...)``, becomes one body per primitive.

        Parameters
        ----------
        actor : WorldObject
            The actor to simulate.
        shape : Shape, optional
            Collision shape, overriding what would be inferred. Required for
            actors whose form cannot be inferred.
        follow_actor : bool, optional
            Make a kinematic body copy the actor's pose every step, so that
            ``fury.motion`` animations can drive it. An actor holding one
            primitive is recentred on that primitive when added, so its
            position then gives the primitive's centre; set animation
            keyframes after adding it.
        **body_options
            Passed to :meth:`add_bodies`.

        Returns
        -------
        BodySet
            A handle to the new bodies.
        """
        if follow_actor and body_options.get("kind", "dynamic") != "kinematic":
            raise ValueError('follow_actor requires kind="kinematic".')
        description = describe_actor(actor, shape, single=follow_actor)
        bodies = self.add_bodies(
            description.positions, description.shape, **body_options
        )
        self._rotations[bodies._range] = description.rotations
        self._finish_binding(bodies, description, follow_actor)
        return bodies

    def _finish_binding(self, bodies, description, follow_actor):
        """
        Attach the actor described by ``description`` to ``bodies``.

        Parameters
        ----------
        bodies : BodySet
            The bodies driving or driven by the actor.
        description : ActorDescription
            What was learned from the actor.
        follow_actor : bool
            Whether the actor drives the body.
        """
        binding = description.attach()
        binding.bodies = bodies
        if follow_actor:
            binding = FollowBinding(binding, bodies)
        self._bindings.append(binding)

    def add_container(
        self,
        *,
        actor=None,
        half_extents=None,
        center=None,
        open_top=False,
        friction=0.5,
        restitution=0.2,
        follow_actor=False,
    ):
        """
        Add a hollow box that keeps dynamic bodies inside it.

        Parameters
        ----------
        actor : WorldObject, optional
            A box actor giving the container's size and position, which then
            moves with it.
        half_extents : float or array_like, shape (3,), optional
            Half widths of the interior. Required without ``actor``.
        center : array_like, shape (3,), optional
            Centre of the container; the origin when not given. Taken from
            ``actor`` when one is given.
        open_top : bool, optional
            Whether to leave out the wall on the local +Y side.
        friction : float, optional
            Friction of the walls.
        restitution : float, optional
            Bounciness of the walls, from 0 to 1.
        follow_actor : bool, optional
            Copy the actor's pose every step, so that ``fury.motion``
            animations can drive the container.

        Returns
        -------
        Container
            A handle that can shake, tilt, spin or move the container.
        """
        description = None
        if actor is not None:
            if half_extents is not None or center is not None:
                raise ValueError(
                    "Give either an actor or half_extents and center, not both."
                )
            info = getattr(actor, "primitive_info", None)
            if info is not None and info.kind not in (None, "box"):
                raise TypeError("A container actor must be a single box.")
            description = describe_actor(actor, None, single=True)
            if description.shape_kind != "box":
                raise TypeError("A container actor must be a single box.")
            half_extents = description.shape.half_extents
            center = description.positions[0]
        elif half_extents is None:
            raise ValueError("Give an actor or half_extents.")
        elif follow_actor:
            raise ValueError("follow_actor requires an actor.")
        if center is None:
            center = np.zeros(3)
        size = np.broadcast_to(np.asarray(half_extents, dtype=float), (1, 3)).copy()
        if np.any(size <= 0) or not np.all(np.isfinite(size)):
            raise ValueError("half_extents must be positive and finite.")
        center = np.asarray(center, dtype=float)
        if center.shape != (3,):
            raise ValueError("center must contain three values.")
        if friction < 0:
            raise ValueError("friction must not be negative.")
        if not 0 <= restitution <= 1:
            raise ValueError("restitution must be between 0 and 1.")
        start = self._append(
            _positions=center[None],
            _shape=np.array([CONTAINER], dtype=np.intp),
            _size=size,
            _kind=np.array([KINEMATIC], dtype=np.intp),
            _friction=np.array([float(friction)]),
            _restitution=np.array([float(restitution)]),
        )
        container = Container(self, start, open_top=open_top)
        self._sets.append(container)
        self._containers.append(container)
        if description is not None:
            self._rotations[container._range] = description.rotations
            self._finish_binding(container, description, follow_actor)
        return container

    def remove(self, bodies):
        """
        Remove a set of bodies from the world.

        Parameters
        ----------
        bodies : BodySet
            The set to remove.
        """
        if bodies not in self._sets:
            raise ValueError("These bodies do not belong to this world.")
        keep = np.ones(self.count, dtype=bool)
        keep[bodies._range] = False
        for name in _default_state(0):
            setattr(self, name, getattr(self, name)[keep])
        self._sets.remove(bodies)
        if bodies in self._containers:
            self._containers.remove(bodies)
            self._holder[self._holder == bodies._serial] = -1
        self._bindings = [b for b in self._bindings if b.bodies is not bodies]
        for other in self._sets:
            if other._start > bodies._start:
                other._start -= bodies._count
        bodies._world = None
        self.contacts = empty_contacts()
        self._solver.clear_cache()
        self._update_margin()

    def _state_changed(self, bodies, *, row=None):
        """
        Note that user code changed a set's state.

        Parameters
        ----------
        bodies : BodySet
            The set that changed.
        row : int, optional
            The one body that changed, when not the whole set.
        """
        if bodies in self._containers:
            self._holder[self._holder == bodies._serial] = -1
        else:
            self._holder[bodies._range if row is None else row] = -1

    def _spin_about(self, body, vectors, at):
        """
        Compute the torque or angular impulse of an off-centre application.

        Parameters
        ----------
        body : ndarray
            Body indices.
        vectors : ndarray, shape (K, 3)
            Force or impulse per body.
        at : array_like or None
            World-space application points.

        Returns
        -------
        ndarray, shape (K, 3)
            Torque or angular impulse.
        """
        if at is None:
            return np.zeros_like(vectors)
        at = np.broadcast_to(np.asarray(at, dtype=float), vectors.shape)
        if not np.all(np.isfinite(at)):
            raise ValueError("Application points must be finite.")
        return cross(at - self._positions[body], vectors)

    def _apply_impulse(self, body, impulse, at):
        """
        Apply impulses to selected bodies.

        Parameters
        ----------
        body : ndarray
            Body indices.
        impulse : ndarray, shape (K, 3)
            Linear impulses.
        at : array_like or None
            World-space application points.
        """
        spin = self._spin_about(body, impulse, at)
        rotation = quat_to_matrix(self._rotations[body])
        inv_inertia = world_inverse_inertia(rotation, self._inverse_inertia[body])
        np.add.at(
            self._velocities[:, :3], body, impulse * self._inverse_mass[body, None]
        )
        np.add.at(
            self._velocities[:, 3:], body, (inv_inertia @ spin[..., None])[..., 0]
        )

    def _apply_force(self, body, force, at):
        """
        Accumulate forces on selected bodies for the next step.

        Parameters
        ----------
        body : ndarray
            Body indices.
        force : ndarray, shape (K, 3)
            Forces.
        at : array_like or None
            World-space application points.
        """
        np.add.at(self._force[:, :3], body, force)
        np.add.at(self._force[:, 3:], body, self._spin_about(body, force, at))

    def _apply_torque(self, body, torque):
        """
        Accumulate torques on selected bodies for the next step.

        Parameters
        ----------
        body : ndarray
            Body indices.
        torque : ndarray, shape (K, 3)
            Torques.
        """
        np.add.at(self._force[:, 3:], body, torque)

    def _solver_dynamic_mask(self):
        """
        Return the bodies the solver may move.

        Returns
        -------
        ndarray, shape (B,)
            Boolean mask.
        """
        return self._kind == DYNAMIC

    def _active_inverse_mass(self):
        """
        Return per-axis inverse masses.

        Returns
        -------
        ndarray, shape (B, 1)
            Inverse masses.
        """
        return self._inverse_mass[:, None]

    def _active_inverse_inertia(self):
        """
        Return body-space inverse inertia.

        Returns
        -------
        ndarray, shape (B, 3)
            Inverse inertia.
        """
        return self._inverse_inertia

    def _substep_damping(self, h):
        """
        Return the factor velocities are multiplied by each substep.

        Parameters
        ----------
        h : float
            Substep duration.

        Returns
        -------
        ndarray, shape (B, 6)
            Damping factors; kinematic rows are left unchanged.
        """
        factor = np.empty((self.count, 6))
        factor[:, :3] = 1.0 / (1.0 + h * self._linear_damping[:, None])
        factor[:, 3:] = 1.0 / (1.0 + h * self._angular_damping[:, None])
        factor[self._kind != DYNAMIC] = 1.0
        return factor

    def _acceleration(self, rotation):
        """
        Compute the acceleration of every body.

        Parameters
        ----------
        rotation : ndarray, shape (B, 3, 3)
            Body rotation matrices.

        Returns
        -------
        ndarray, shape (B, 6)
            Linear and angular acceleration.
        """
        acceleration = np.zeros((self.count, 6))
        dynamic = self._kind == DYNAMIC
        acceleration[dynamic, :3] = (
            self.gravity * self._gravity_scale[dynamic, None]
            + self._force[dynamic, :3] * self._inverse_mass[dynamic, None]
        )
        if self._force[:, 3:].any():
            inv_inertia = world_inverse_inertia(rotation, self._inverse_inertia)
            acceleration[:, 3:] = (inv_inertia @ self._force[:, 3:, None])[..., 0]
        return acceleration

    def _extents(self, rotation):
        """
        Compute bounding-box half widths.

        Parameters
        ----------
        rotation : ndarray, shape (B, 3, 3)
            Body rotation matrices.

        Returns
        -------
        ndarray, shape (B, 3)
            Half widths.
        """
        extents = self._radius[:, None] * np.ones((self.count, 3))
        boxes = self._shape == BOX
        extents[boxes] = (np.abs(rotation[boxes]) @ self._size[boxes, :, None])[..., 0]
        return extents

    def _candidates(self):
        """
        Return the bodies that take part in pair finding.

        Returns
        -------
        ndarray, shape (B,)
            Boolean mask.
        """
        return self._shape != CONTAINER

    def _new_container_serial(self):
        """
        Return a number that identifies a new container for its whole life.

        Returns
        -------
        int
            The container's serial number.
        """
        self._container_serials += 1
        return self._container_serials

    def _contact_mask(self, row):
        """
        Return the bodies that can touch a kinematic body or container.

        Parameters
        ----------
        row : int
            Index of the kinematic body.

        Returns
        -------
        ndarray, shape (B,)
            Boolean mask.
        """
        return self._kind == DYNAMIC

    def _filter_pairs(self, i, j):
        """
        Drop pairs that cannot produce a useful contact.

        Parameters
        ----------
        i, j : ndarray, shape (N,)
            Pair indices.

        Returns
        -------
        tuple of ndarray
            Filtered ``(i, j)``.
        """
        keep = (self._kind[i] == DYNAMIC) | (self._kind[j] == DYNAMIC)
        return i[keep], j[keep]

    def _find_contacts(self, rotation):
        """
        Run collision detection.

        Parameters
        ----------
        rotation : ndarray, shape (B, 3, 3)
            Body rotation matrices.

        Returns
        -------
        Contacts
            Contacts for this step.
        """
        i, j = find_pairs(
            self._positions,
            self._extents(rotation),
            self._radius,
            self._candidates(),
            self.margin,
        )
        i, j = self._filter_pairs(i, j)
        parts = [generate_contacts(self, rotation, i, j)]
        parts.extend(
            container_contacts(self, rotation, container)
            for container in self._containers
        )
        return merge_contacts(parts)

    def sync(self):
        """Write the current state into every bound actor."""
        for binding in self._bindings:
            binding.sync()

    def step(self, count):
        """
        Advance the simulation by whole steps.

        Parameters
        ----------
        count : int
            Number of steps.
        """
        for _ in range(count):
            self._step_once()

    def _step_once(self):
        """Advance the simulation by one step."""
        for bodies in self._sets:
            if isinstance(bodies, KinematicBodySet):
                bodies._update(self.time, self.dt)
        for binding in self._bindings:
            binding.before_step(self.dt)
        rotation = quat_to_matrix(self._rotations)
        self.contacts = self._find_contacts(rotation)
        self._solver.solve(self, self.contacts, rotation, self._acceleration(rotation))
        self._force.fill(0.0)
        self.time += self.dt
        self.step_count += 1
