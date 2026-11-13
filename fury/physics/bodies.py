"""User-facing handles for bodies in a world."""

import numpy as np

from fury.physics.shapes import DYNAMIC, STATIC


def _snapshot(values):
    """
    Return a read-only copy of engine state.

    Parameters
    ----------
    values : ndarray
        Engine state.

    Returns
    -------
    ndarray
        A copy that cannot be written to.
    """
    copied = values.copy()
    copied.flags.writeable = False
    return copied


def _finite_rows(value, count, width):
    """
    Broadcast new state values to one row per body and check them.

    Parameters
    ----------
    value : array_like
        New values, broadcastable to ``(count, width)``.
    count : int
        Number of bodies.
    width : int
        Components per body.

    Returns
    -------
    ndarray, shape (count, width)
        The broadcast values.
    """
    try:
        rows = np.broadcast_to(np.asarray(value, dtype=float), (count, width))
    except ValueError as err:
        raise ValueError(
            f"Values must have shape ({width},) or one row per body."
        ) from err
    if not np.all(np.isfinite(rows)):
        raise ValueError("Values must be finite.")
    return rows


class BodySet:
    """
    A block of bodies added to a world together.

    Array properties return read-only snapshots of the current state. Assign to
    the property to change it.

    Parameters
    ----------
    world : World
        The owning world.
    start : int
        Index of the first body.
    count : int
        Number of bodies.
    """

    def __init__(self, world, start, count):
        self._world = world
        self._start = start
        self._count = count

    @property
    def world(self):
        """
        Return the world these bodies belong to.

        Returns
        -------
        World
            The world these bodies belong to.
        """
        if self._world is None:
            raise ValueError("These bodies have been removed from their world.")
        return self._world

    @property
    def start(self):
        """
        Return index of the first body in the world's arrays.

        Returns
        -------
        int
            Index of the first body in the world's arrays.
        """
        return self._start

    @property
    def count(self):
        """
        Return number of bodies.

        Returns
        -------
        int
            Number of bodies.
        """
        return self._count

    @property
    def _range(self):
        """
        Return location of this set in the world's arrays.

        Returns
        -------
        slice
            Location of this set in the world's arrays.
        """
        return slice(self.start, self.start + self.count)

    def __len__(self):
        """
        Return the number of bodies.

        Returns
        -------
        int
            Number of bodies in the set.
        """
        return self.count

    def __getitem__(self, index):
        """
        Return one body of the set.

        Parameters
        ----------
        index : int
            Body position within the set; negative values count from the end.

        Returns
        -------
        Body
            A handle to that body.
        """
        index = int(index)
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError("Body index out of range.")
        return Body(self, index)

    def __iter__(self):
        """
        Iterate over the bodies of the set.

        Yields
        ------
        Body
            A handle to each body.
        """
        for index in range(self.count):
            yield Body(self, index)

    def _require_movable(self):
        """Reject velocity changes that a static body could never honour."""
        if self.world._kind[self.start] == STATIC:
            raise ValueError("Static bodies cannot be given a velocity.")

    def _require_dynamic(self):
        """Reject forces, which only dynamic bodies respond to."""
        if self.world._kind[self.start] != DYNAMIC:
            raise ValueError("Only dynamic bodies respond to forces and impulses.")

    def _assign(self, array, value, width):
        """
        Write values into a state array.

        Parameters
        ----------
        array : ndarray
            Engine state array.
        value : array_like
            New values, broadcastable to the set.
        width : int
            Components per body.
        """
        array[self._range] = _finite_rows(value, self.count, width)
        self.world._state_changed(self)

    @property
    def positions(self):
        """
        Return body positions.

        Returns
        -------
        ndarray, shape (N, 3)
            Body positions.
        """
        return _snapshot(self.world._positions[self._range])

    @positions.setter
    def positions(self, value):
        """
        Replace positions.

        Parameters
        ----------
        value : array_like
            The new value, broadcastable to the bodies.
        """
        self._assign(self.world._positions, value, 3)

    @property
    def rotations(self):
        """
        Return quaternions, in ``(x, y, z, w)`` order.

        Returns
        -------
        ndarray, shape (N, 4)
            Quaternions, in ``(x, y, z, w)`` order.
        """
        return _snapshot(self.world._rotations[self._range])

    @rotations.setter
    def rotations(self, value):
        """
        Replace rotations.

        Parameters
        ----------
        value : array_like
            The new value, broadcastable to the bodies.
        """
        value = np.asarray(value, dtype=float)
        norm = np.linalg.norm(value, axis=-1, keepdims=True)
        if not np.all(norm >= 1e-12):
            raise ValueError("Quaternions must be finite and non-zero.")
        self._assign(self.world._rotations, value / norm, 4)

    @property
    def velocities(self):
        """
        Return linear velocities.

        Returns
        -------
        ndarray, shape (N, 3)
            Linear velocities.
        """
        return _snapshot(self.world._velocities[self._range, :3])

    @velocities.setter
    def velocities(self, value):
        """
        Replace velocities.

        Parameters
        ----------
        value : array_like
            The new value, broadcastable to the bodies.
        """
        self._require_movable()
        self.world._velocities[self._range, :3] = _finite_rows(value, self.count, 3)
        self.world._state_changed(self)

    @property
    def angular_velocities(self):
        """
        Return angular velocities, in radians per second.

        Returns
        -------
        ndarray, shape (N, 3)
            Angular velocities, in radians per second.
        """
        return _snapshot(self.world._velocities[self._range, 3:])

    @angular_velocities.setter
    def angular_velocities(self, value):
        """
        Replace angular velocities.

        Parameters
        ----------
        value : array_like
            The new value, broadcastable to the bodies.
        """
        self._require_movable()
        self.world._velocities[self._range, 3:] = _finite_rows(value, self.count, 3)
        self.world._state_changed(self)

    @property
    def masses(self):
        """
        Return masses; zero for static and kinematic bodies.

        Returns
        -------
        ndarray, shape (N,)
            Masses; zero for static and kinematic bodies.
        """
        inverse = self.world._inverse_mass[self._range]
        return _snapshot(
            np.divide(1.0, inverse, out=np.zeros_like(inverse), where=inverse > 0)
        )

    def _selection(self, where):
        """
        Convert a selection into world body indices.

        Parameters
        ----------
        where : array_like or None
            Boolean mask or integer indices within the set; None selects all.

        Returns
        -------
        ndarray
            World body indices.
        """
        indices = np.arange(self.start, self.start + self.count)
        if where is None:
            return indices
        where = np.atleast_1d(where)
        return indices[where if where.size else where.astype(np.intp)]

    def apply_impulse(self, impulse, *, where=None, at=None):
        """
        Change momentum instantly.

        Parameters
        ----------
        impulse : array_like, shape (3,) or (K, 3)
            Linear impulse for each selected body.
        where : array_like, optional
            Boolean mask or indices selecting bodies within this set.
        at : array_like, shape (3,) or (K, 3), optional
            World-space application points; bodies spin when off-centre.
        """
        self._require_dynamic()
        world = self.world
        body = self._selection(where)
        impulse = _finite_rows(impulse, body.size, 3)
        world._apply_impulse(body, impulse, at)

    def apply_force(self, force, *, where=None, at=None):
        """
        Apply a force during the next step.

        Parameters
        ----------
        force : array_like, shape (3,) or (K, 3)
            Force for each selected body.
        where : array_like, optional
            Boolean mask or indices selecting bodies within this set.
        at : array_like, shape (3,) or (K, 3), optional
            World-space application points; bodies spin when off-centre.
        """
        self._require_dynamic()
        world = self.world
        body = self._selection(where)
        force = _finite_rows(force, body.size, 3)
        world._apply_force(body, force, at)

    def apply_torque(self, torque, *, where=None):
        """
        Apply a torque during the next step.

        Parameters
        ----------
        torque : array_like, shape (3,) or (K, 3)
            Torque for each selected body.
        where : array_like, optional
            Boolean mask or indices selecting bodies within this set.
        """
        self._require_dynamic()
        body = self._selection(where)
        torque = _finite_rows(torque, body.size, 3)
        self.world._apply_torque(body, torque)


class Body:
    """
    A handle to one body of a :class:`BodySet`.

    Parameters
    ----------
    bodies : BodySet
        The owning set.
    index : int
        Position within the set.
    """

    def __init__(self, bodies, index):
        self._bodies = bodies
        self._index = index

    @property
    def bodies(self):
        """
        Return the set this body belongs to.

        Returns
        -------
        BodySet
            The owning set.
        """
        return self._bodies

    @property
    def index(self):
        """
        Return this body's position within its set.

        Returns
        -------
        int
            Position within the set.
        """
        return self._index

    @property
    def _id(self):
        """
        Return index of this body in the world's arrays.

        Returns
        -------
        int
            Index of this body in the world's arrays.
        """
        return self.bodies.start + self.index

    def _write(self, array, value):
        """
        Write one body's state.

        Parameters
        ----------
        array : ndarray
            Engine state selected to this body's columns.
        value : array_like
            New value.
        """
        array[:] = _finite_rows(value, 1, array.shape[-1])[0]
        self.bodies.world._state_changed(self.bodies, row=self._id)

    @property
    def position(self):
        """
        Return position.

        Returns
        -------
        ndarray, shape (3,)
            Position.
        """
        return _snapshot(self.bodies.world._positions[self._id])

    @position.setter
    def position(self, value):
        """
        Replace position.

        Parameters
        ----------
        value : array_like
            The new value, broadcastable to the bodies.
        """
        self._write(self.bodies.world._positions[self._id], value)

    @property
    def rotation(self):
        """
        Return quaternion, in ``(x, y, z, w)`` order.

        Returns
        -------
        ndarray, shape (4,)
            Quaternion, in ``(x, y, z, w)`` order.
        """
        return _snapshot(self.bodies.world._rotations[self._id])

    @rotation.setter
    def rotation(self, value):
        """
        Replace rotation.

        Parameters
        ----------
        value : array_like
            The new value, broadcastable to the bodies.
        """
        value = np.asarray(value, dtype=float)
        norm = np.linalg.norm(value)
        if not norm >= 1e-12:
            raise ValueError("Quaternions must be finite and non-zero.")
        self._write(self.bodies.world._rotations[self._id], value / norm)

    @property
    def velocity(self):
        """
        Return linear velocity.

        Returns
        -------
        ndarray, shape (3,)
            Linear velocity.
        """
        return _snapshot(self.bodies.world._velocities[self._id, :3])

    @velocity.setter
    def velocity(self, value):
        """
        Replace velocity.

        Parameters
        ----------
        value : array_like
            The new value, broadcastable to the bodies.
        """
        self.bodies._require_movable()
        self._write(self.bodies.world._velocities[self._id, :3], value)

    @property
    def angular_velocity(self):
        """
        Return angular velocity, in radians per second.

        Returns
        -------
        ndarray, shape (3,)
            Angular velocity, in radians per second.
        """
        return _snapshot(self.bodies.world._velocities[self._id, 3:])

    @angular_velocity.setter
    def angular_velocity(self, value):
        """
        Replace angular velocity.

        Parameters
        ----------
        value : array_like
            The new value, broadcastable to the bodies.
        """
        self.bodies._require_movable()
        self._write(self.bodies.world._velocities[self._id, 3:], value)

    def apply_impulse(self, impulse, *, at=None):
        """
        Change this body's momentum instantly.

        Parameters
        ----------
        impulse : array_like, shape (3,)
            Linear impulse.
        at : array_like, shape (3,), optional
            World-space application point.
        """
        self.bodies.apply_impulse(impulse, where=[self.index], at=at)

    def apply_force(self, force, *, at=None):
        """
        Apply a force during the next step.

        Parameters
        ----------
        force : array_like, shape (3,)
            Force.
        at : array_like, shape (3,), optional
            World-space application point.
        """
        self.bodies.apply_force(force, where=[self.index], at=at)

    def apply_torque(self, torque):
        """
        Apply a torque during the next step.

        Parameters
        ----------
        torque : array_like, shape (3,)
            Torque.
        """
        self.bodies.apply_torque(torque, where=[self.index])
