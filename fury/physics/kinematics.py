"""Scripted motion for kinematic bodies and containers."""

import logging

import numpy as np

from fury.physics.bodies import BodySet
from fury.physics.shapes import DYNAMIC


def _unit(axis):
    """
    Normalize an axis.

    Parameters
    ----------
    axis : array_like, shape (3,)
        Axis to normalize.

    Returns
    -------
    ndarray, shape (3,)
        Unit vector.
    """
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,):
        raise ValueError("axis must contain three values.")
    length = np.linalg.norm(axis)
    if length < 1e-12:
        raise ValueError("axis must be non-zero.")
    return axis / length


def _positive(value, name):
    """
    Validate a strictly positive number.

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Name used in error messages.

    Returns
    -------
    float
        The validated value.
    """
    value = float(value)
    if value <= 0 or not np.isfinite(value):
        raise ValueError(f"{name} must be positive and finite.")
    return value


def _smoothstep(fraction):
    """
    Return the smoothstep curve, which eases in and out.

    Parameters
    ----------
    fraction : float
        Progress through the motion, from 0 to 1.

    Returns
    -------
    float
        Eased progress, from 0 to 1.
    """
    fraction = min(max(fraction, 0.0), 1.0)
    return fraction * fraction * (3.0 - 2.0 * fraction)


class Motion:
    """
    A velocity that depends on time since the motion started.

    Parameters
    ----------
    duration : float or None
        Length of the motion; None runs until stopped.
    """

    def __init__(self, *, duration=None):
        self.duration = None if duration is None else _positive(duration, "duration")
        self.started = None

    def sample(self, start, end, positions, dt):
        """
        Return the velocities that cover one step's share of the motion.

        Parameters
        ----------
        start : float
            Seconds since the motion began, at the start of the step.
        end : float
            Seconds since the motion began, at the end of the step, capped
            at the motion's duration.
        positions : ndarray, shape (N, 3)
            Current body positions.
        dt : float
            Step duration.

        Returns
        -------
        linear : ndarray, shape (3,) or (N, 3)
            Linear velocity.
        angular : ndarray, shape (3,)
            Angular velocity.
        """
        raise NotImplementedError


class _Shake(Motion):
    """
    Sinusoidal translation.

    Parameters
    ----------
    amplitude : float
        Largest distance from the start.
    frequency : float
        Cycles per second.
    axis : array_like, shape (3,)
        Direction of motion.
    duration : float or None
        Seconds to run, rounded up to whole cycles.
    """

    def __init__(self, amplitude, frequency, axis, duration):
        self.omega = 2.0 * np.pi * _positive(frequency, "frequency")
        self.direction = _unit(axis) * _positive(amplitude, "amplitude")
        if duration is not None:
            period = 1.0 / frequency
            cycles = np.ceil(_positive(duration, "duration") / period - 1e-9)
            duration = max(cycles, 1.0) * period
        super().__init__(duration=duration)

    def sample(self, start, end, positions, dt):
        """
        Return the velocities that cover one step's share of the motion.

        Parameters
        ----------
        start : float
            Seconds since the motion began, at the start of the step.
        end : float
            Seconds since the motion began, at the end of the step, capped
            at the motion's duration.
        positions : ndarray, shape (N, 3)
            Current body positions.
        dt : float
            Step duration.

        Returns
        -------
        linear : ndarray, shape (3,) or (N, 3)
            Linear velocity.
        angular : ndarray, shape (3,)
            Angular velocity.
        """
        travel = np.sin(self.omega * end) - np.sin(self.omega * start)
        return self.direction * travel / dt, np.zeros(3)


class _Rock(Motion):
    """
    Sinusoidal rotation about each body's centre.

    Parameters
    ----------
    axis : array_like, shape (3,)
        Rotation axis.
    angle : float
        Largest rotation, in radians.
    frequency : float
        Cycles per second.
    duration : float or None
        Seconds to run, rounded up to whole cycles.
    """

    def __init__(self, axis, angle, frequency, duration):
        self.omega = 2.0 * np.pi * _positive(frequency, "frequency")
        self.axis = _unit(axis) * _positive(angle, "angle")
        if duration is not None:
            period = 1.0 / frequency
            cycles = np.ceil(_positive(duration, "duration") / period - 1e-9)
            duration = max(cycles, 1.0) * period
        super().__init__(duration=duration)

    def sample(self, start, end, positions, dt):
        """
        Return the velocities that cover one step's share of the motion.

        Parameters
        ----------
        start : float
            Seconds since the motion began, at the start of the step.
        end : float
            Seconds since the motion began, at the end of the step, capped
            at the motion's duration.
        positions : ndarray, shape (N, 3)
            Current body positions.
        dt : float
            Step duration.

        Returns
        -------
        linear : ndarray, shape (3,) or (N, 3)
            Linear velocity.
        angular : ndarray, shape (3,)
            Angular velocity.
        """
        turn = np.sin(self.omega * end) - np.sin(self.omega * start)
        return np.zeros(3), self.axis * turn / dt


class _Spin(Motion):
    """
    Constant rotation, optionally orbiting a pivot.

    Parameters
    ----------
    axis : array_like, shape (3,)
        Rotation axis.
    speed : float
        Angular speed, in radians per second.
    pivot : array_like, shape (3,) or None
        Point to orbit, or None to spin in place.
    duration : float or None
        Seconds to run.
    """

    def __init__(self, axis, speed, pivot, duration):
        self.axis = _unit(axis)
        self.speed = float(speed)
        if not np.isfinite(self.speed):
            raise ValueError("speed must be finite.")
        self.pivot = None if pivot is None else np.asarray(pivot, dtype=float)
        if self.pivot is not None and (
            self.pivot.shape != (3,) or not np.all(np.isfinite(self.pivot))
        ):
            raise ValueError("pivot must contain three finite values.")
        super().__init__(duration=duration)

    def sample(self, start, end, positions, dt):
        """
        Return the velocities that cover one step's share of the motion.

        Parameters
        ----------
        start : float
            Seconds since the motion began, at the start of the step.
        end : float
            Seconds since the motion began, at the end of the step, capped
            at the motion's duration.
        positions : ndarray, shape (N, 3)
            Current body positions.
        dt : float
            Step duration.

        Returns
        -------
        linear : ndarray, shape (3,) or (N, 3)
            Linear velocity.
        angular : ndarray, shape (3,)
            Angular velocity.
        """
        angle = self.speed * (end - start)
        angular = self.axis * angle / dt
        if self.pivot is None:
            return np.zeros(3), angular
        arm = positions - self.pivot
        parallel = self.axis * (arm @ self.axis)[:, None]
        radial = arm - parallel
        turned = radial * np.cos(angle) + np.cross(self.axis, radial) * np.sin(angle)
        return (turned - radial) / dt, angular


class _Slide(Motion):
    """
    Eased translation by a fixed offset.

    Parameters
    ----------
    offset : array_like, shape (3,)
        Distance to move.
    duration : float
        Seconds to take.
    """

    def __init__(self, offset, duration):
        self.offset = np.asarray(offset, dtype=float)
        if self.offset.shape != (3,):
            raise ValueError("offset must contain three values.")
        super().__init__(duration=_positive(duration, "duration"))

    def sample(self, start, end, positions, dt):
        """
        Return the velocities that cover one step's share of the motion.

        Parameters
        ----------
        start : float
            Seconds since the motion began, at the start of the step.
        end : float
            Seconds since the motion began, at the end of the step, capped
            at the motion's duration.
        positions : ndarray, shape (N, 3)
            Current body positions.
        dt : float
            Step duration.

        Returns
        -------
        linear : ndarray, shape (3,) or (N, 3)
            Linear velocity.
        angular : ndarray, shape (3,)
            Angular velocity.
        """
        share = _smoothstep(end / self.duration) - _smoothstep(start / self.duration)
        return self.offset * share / dt, np.zeros(3)


class _Tilt(Motion):
    """
    Eased rotation by a fixed angle.

    Parameters
    ----------
    axis : array_like, shape (3,)
        Rotation axis.
    angle : float
        Total rotation, in radians.
    duration : float
        Seconds to take.
    """

    def __init__(self, axis, angle, duration):
        if not np.isfinite(angle):
            raise ValueError("angle must be finite.")
        self.axis = _unit(axis) * float(angle)
        super().__init__(duration=_positive(duration, "duration"))

    def sample(self, start, end, positions, dt):
        """
        Return the velocities that cover one step's share of the motion.

        Parameters
        ----------
        start : float
            Seconds since the motion began, at the start of the step.
        end : float
            Seconds since the motion began, at the end of the step, capped
            at the motion's duration.
        positions : ndarray, shape (N, 3)
            Current body positions.
        dt : float
            Step duration.

        Returns
        -------
        linear : ndarray, shape (3,) or (N, 3)
            Linear velocity.
        angular : ndarray, shape (3,)
            Angular velocity.
        """
        share = _smoothstep(end / self.duration) - _smoothstep(start / self.duration)
        return np.zeros(3), self.axis * share / dt


class _Path(Motion):
    """
    Translation along a user-supplied function of time.

    Parameters
    ----------
    path : callable
        Maps seconds since the motion began to a position.
    duration : float or None
        Seconds to follow.
    """

    def __init__(self, path, duration):
        if not callable(path):
            raise TypeError("path must be callable.")
        start = np.asarray(path(0.0), dtype=float)
        if start.shape != (3,) or not np.all(np.isfinite(start)):
            raise ValueError("path must return three finite values.")
        self.path = path
        super().__init__(duration=duration)

    def sample(self, start, end, positions, dt):
        """
        Return the velocities that cover one step's share of the motion.

        Parameters
        ----------
        start : float
            Seconds since the motion began, at the start of the step.
        end : float
            Seconds since the motion began, at the end of the step, capped
            at the motion's duration.
        positions : ndarray, shape (N, 3)
            Current body positions.
        dt : float
            Step duration.

        Returns
        -------
        linear : ndarray, shape (3,) or (N, 3)
            Linear velocity.
        angular : ndarray, shape (3,)
            Angular velocity.
        """
        ahead = np.asarray(self.path(end), dtype=float)
        behind = np.asarray(self.path(start), dtype=float)
        for point in (ahead, behind):
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                raise ValueError("path must return three finite values.")
        return (ahead - behind) / dt, np.zeros(3)


class KinematicBodySet(BodySet):
    """
    Bodies that move only as scripted, and push everything they touch.

    Motions run together and add up. Velocity is what transfers momentum to
    the bodies being pushed, so motions are expressed as velocities.

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
        super().__init__(world, start, count)
        self._motions = []
        self._warned = False

    def _begin(self, motion):
        """
        Start a motion.

        Parameters
        ----------
        motion : Motion
            The motion to start.

        Returns
        -------
        KinematicBodySet
            This set, so calls can be chained.
        """
        self._motions.append(motion)
        return self

    def _reach(self):
        """
        Return how far the surfaces of these bodies reach from their centres.

        Returns
        -------
        float
            Largest bounding radius.
        """
        return float(self.world._radius[self._range].max())

    def _check_speed(self, linear, *, angular=0.0, pivot=None):
        """
        Warn when a motion moves surfaces too far in one step.

        Bodies then overlap visibly and, in a container, can be pushed through
        the walls; a smaller ``dt`` or a slower motion avoids this.

        Parameters
        ----------
        linear : float
            Peak linear speed of the motion.
        angular : float, optional
            Peak angular speed of the motion, in radians per second.
        pivot : ndarray, shape (3,), optional
            Point the bodies turn about, if not their own centres.
        """
        world = self.world
        reach = self._reach()
        if pivot is not None:
            positions = world._positions[self._range]
            reach += float(np.linalg.norm(positions - pivot, axis=1).max())
        fast = self._warn_if_fast(abs(linear) + abs(angular) * reach)
        self._warned = self._warned or fast

    def _warn_if_fast(self, peak):
        """
        Log the speed warning when surfaces move too far in one step.

        Parameters
        ----------
        peak : float
            Fastest surface speed.

        Returns
        -------
        bool
            Whether the warning was logged.
        """
        world = self.world
        dynamic = world._radius[world._kind == DYNAMIC]
        if not (dynamic.size and peak * world.dt > 0.5 * dynamic.min()):
            return False
        logging.warning(
            "Surfaces move up to %.1f per second, more than half of the "
            "smallest body in one step, so bodies may overlap visibly or be "
            "pushed through container walls. Slow the motion or reduce dt.",
            peak,
        )
        return True

    def shake(self, *, amplitude=0.5, frequency=2.0, axis=(1, 0, 0), duration=None):
        """
        Move back and forth along an axis.

        Parameters
        ----------
        amplitude : float, optional
            Largest distance from the start.
        frequency : float, optional
            Cycles per second.
        axis : array_like, shape (3,), optional
            Direction of motion.
        duration : float, optional
            Seconds to shake, rounded up to a whole number of cycles so that
            the bodies end where they began. None shakes until stopped.

        Returns
        -------
        KinematicBodySet
            This set, so calls can be chained.
        """
        motion = _Shake(amplitude, frequency, axis, duration)
        self._check_speed(amplitude * motion.omega)
        return self._begin(motion)

    def rock(self, *, axis=(0, 0, 1), angle=0.3, frequency=1.0, duration=None):
        """
        Rotate back and forth about each body's centre.

        Parameters
        ----------
        axis : array_like, shape (3,), optional
            Rotation axis.
        angle : float, optional
            Largest rotation, in radians.
        frequency : float, optional
            Cycles per second.
        duration : float, optional
            Seconds to rock, rounded up to whole cycles. None runs until stopped.

        Returns
        -------
        KinematicBodySet
            This set, so calls can be chained.
        """
        motion = _Rock(axis, angle, frequency, duration)
        self._check_speed(0.0, angular=np.linalg.norm(motion.axis) * motion.omega)
        return self._begin(motion)

    def spin(self, *, axis=(0, 1, 0), speed=1.0, pivot=None, duration=None):
        """
        Rotate at a constant rate.

        Parameters
        ----------
        axis : array_like, shape (3,), optional
            Rotation axis.
        speed : float, optional
            Angular speed, in radians per second.
        pivot : array_like, shape (3,), optional
            Point to orbit. By default each body spins about its own centre.
        duration : float, optional
            Seconds to spin. None runs until stopped.

        Returns
        -------
        KinematicBodySet
            This set, so calls can be chained.
        """
        motion = _Spin(axis, speed, pivot, duration)
        self._check_speed(0.0, angular=motion.speed, pivot=motion.pivot)
        return self._begin(motion)

    def tilt(self, *, axis=(0, 0, 1), angle=0.5, duration=1.0):
        """
        Rotate smoothly by a fixed angle, then stop.

        Parameters
        ----------
        axis : array_like, shape (3,), optional
            Rotation axis.
        angle : float, optional
            Total rotation, in radians.
        duration : float, optional
            Seconds to take.

        Returns
        -------
        KinematicBodySet
            This set, so calls can be chained.
        """
        motion = _Tilt(axis, angle, duration)
        self._check_speed(
            0.0, angular=1.5 * np.linalg.norm(motion.axis) / motion.duration
        )
        return self._begin(motion)

    def move_by(self, offset, *, duration=1.0):
        """
        Translate smoothly by an offset, then stop.

        Parameters
        ----------
        offset : array_like, shape (3,)
            Distance to move.
        duration : float, optional
            Seconds to take.

        Returns
        -------
        KinematicBodySet
            This set, so calls can be chained.
        """
        motion = _Slide(offset, duration)
        self._check_speed(1.5 * np.linalg.norm(motion.offset) / motion.duration)
        return self._begin(motion)

    def move_to(self, position, *, duration=1.0):
        """
        Translate smoothly so the first body ends at a position.

        The offset is measured when this is called, and running motions still
        add to it, so start a second ``move_to`` after the first has finished.

        Parameters
        ----------
        position : array_like, shape (3,)
            Target for the first body; the others keep their offsets.
        duration : float, optional
            Seconds to take.

        Returns
        -------
        KinematicBodySet
            This set, so calls can be chained.
        """
        offset = np.asarray(position, dtype=float) - self.positions[0]
        return self.move_by(offset, duration=duration)

    def follow_path(self, path, *, duration=None):
        """
        Translate along a function of time.

        Parameters
        ----------
        path : callable
            Maps seconds since the motion began to a position. Only changes in
            the position matter, so the path need not start at the body.
        duration : float, optional
            Seconds to follow. None runs until stopped.

        Returns
        -------
        KinematicBodySet
            This set, so calls can be chained.
        """
        return self._begin(_Path(path, duration))

    def stop(self):
        """
        Cancel all motions and bring the bodies to rest.

        Returns
        -------
        KinematicBodySet
            This set, so calls can be chained.
        """
        self._motions.clear()
        self._warned = False
        self.world._velocities[self._range] = 0.0
        return self

    @property
    def is_moving(self):
        """
        Return whether a scripted motion is running.

        Returns
        -------
        bool
            Whether a scripted motion is running.
        """
        return bool(self._motions)

    def _update(self, time, dt):
        """
        Set this step's velocity from the running motions.

        Parameters
        ----------
        time : float
            World time at the start of the step.
        dt : float
            Step duration.
        """
        if not self._motions:
            return
        world = self.world
        linear = np.zeros((self.count, 3))
        angular = np.zeros((self.count, 3))
        positions = world._positions[self._range]
        running = []
        for motion in self._motions:
            if motion.started is None:
                motion.started = time
            start = time - motion.started
            end = start + dt
            if motion.duration is not None:
                if start >= motion.duration - 1e-9 * dt:
                    continue
                end = min(end, motion.duration)
            step_linear, step_angular = motion.sample(start, end, positions, dt)
            linear += step_linear
            angular += step_angular
            running.append(motion)
        self._motions = running
        world._velocities[self._range, :3] = linear
        world._velocities[self._range, 3:] = angular
        surface = np.linalg.norm(linear, axis=1)
        surface += np.linalg.norm(angular, axis=1) * self._reach()
        if running and not self._warned:
            self._warned = self._warn_if_fast(float(surface.max()))
        if not running:
            self._warned = False


class Container(KinematicBodySet):
    """
    A hollow box that keeps bodies inside it.

    Its walls are inward-facing half-spaces rather than thin solid plates, so
    bodies pushed a little way through a wall by fast motion are pulled back
    in. Motions that move the walls more than about half the smallest body's
    radius in one step log a warning, because bodies can then end up far
    outside; reduce ``dt`` for such motions. The walls
    act only on bodies the container holds, so bodies outside are not
    affected. A body is taken in once its centre is inside, including when a
    moving container sweeps over it, and let go when its centre rises above an
    open top, when its position or velocity is set by hand while it is outside,
    or when the container is removed.

    Parameters
    ----------
    world : World
        The owning world.
    start : int
        Index of the container's row.
    open_top : bool
        Whether to leave out the wall on the local +Y side.
    """

    def __init__(self, world, start, *, open_top=False):
        super().__init__(world, start, 1)
        self.open_top = bool(open_top)
        self._serial = world._new_container_serial()

    def _reach(self):
        """
        Return how far the walls reach from the container's centre.

        Returns
        -------
        float
            Distance from the centre to a corner.
        """
        return float(np.linalg.norm(self.world._size[self.start]))

    @property
    def walls(self):
        """
        Return enabled walls, as +X, +Y, +Z, -X, -Y, -Z.

        Returns
        -------
        ndarray, shape (6,)
            Enabled walls, as +X, +Y, +Z, -X, -Y, -Z.
        """
        walls = np.ones(6, dtype=bool)
        walls[1] = not self.open_top
        return walls

    @property
    def half_extents(self):
        """
        Return half widths of the interior.

        Returns
        -------
        ndarray, shape (3,)
            Half widths of the interior.
        """
        return self.world._size[self.start].copy()
