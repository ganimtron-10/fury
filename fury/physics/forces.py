"""Force fields for physics worlds."""

import numpy as np


def _vector(value, name):
    """
    Validate a 3D vector.

    Parameters
    ----------
    value : array_like
        The vector.
    name : str
        Name used in error messages.

    Returns
    -------
    ndarray, shape (3,)
        The vector as floats.
    """
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain three finite values.")
    return value


def _number(value, name, *, positive=False, non_negative=False):
    """
    Validate a finite scalar.

    Parameters
    ----------
    value : float
        The number.
    name : str
        Name used in error messages.
    positive : bool, optional
        Require a value above zero.
    non_negative : bool, optional
        Require a value of at least zero.

    Returns
    -------
    float
        The number.
    """
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if positive and not value > 0:
        raise ValueError(f"{name} must be positive.")
    if non_negative and value < 0:
        raise ValueError(f"{name} must not be negative.")
    return value


def _relax(world, target, coefficient):
    """
    Return forces pulling velocities towards a target for one step.

    The force is chosen so the step's velocity change matches exact
    exponential decay, which stays stable however light a body is.

    Parameters
    ----------
    world : World
        The world being stepped.
    target : ndarray, shape (3,)
        Velocity the bodies are pulled towards.
    coefficient : float
        Force per unit of speed difference.

    Returns
    -------
    ndarray, shape (B, 3)
        Force on each body.
    """
    inverse = world._inverse_mass
    share = -np.expm1(-coefficient * inverse * world.dt)
    scale = _masses(world) * share / world.dt
    return scale[:, None] * (target - world._velocities[:, :3])


def _masses(world):
    """
    Return body masses, with zero for immovable bodies.

    Parameters
    ----------
    world : World
        The world.

    Returns
    -------
    ndarray, shape (B,)
        Masses.
    """
    inverse = world._inverse_mass
    return np.divide(1.0, inverse, out=np.zeros_like(inverse), where=inverse > 0)


class Relaxation:
    """
    A force field that pulls velocities towards a target, such as drag or wind.

    The world adds all such fields together and applies them as one exact
    exponential decay per step, so any mix of them stays stable however light
    a body is. Under a strong constant force, such as gravity on a very light
    body, the speed it settles at is somewhat higher than the exact one.

    Parameters
    ----------
    coefficient : float
        Force per unit of speed difference.
    target : ndarray, shape (3,)
        Velocity the bodies are pulled towards.
    """

    def __init__(self, coefficient, target):
        self.coefficient = coefficient
        self.target = target

    def __call__(self, world):
        """
        Compute this field's force on every body, on its own.

        Parameters
        ----------
        world : World
            The world being stepped.

        Returns
        -------
        ndarray, shape (B, 3)
            Force on each body.
        """
        return _relax(world, self.target, self.coefficient)


def combined_relaxation(world, fields):
    """
    Return the force of several relaxation fields applied together.

    Parameters
    ----------
    world : World
        The world being stepped.
    fields : list of Relaxation
        The fields to combine.

    Returns
    -------
    ndarray, shape (B, 3)
        Force on each body.
    """
    total = sum(field.coefficient for field in fields)
    if total <= 0:
        return np.zeros((world.count, 3))
    target = sum(field.coefficient * field.target for field in fields) / total
    return _relax(world, target, total)


def uniform(force):
    """
    Create a constant force on every body.

    Parameters
    ----------
    force : array_like, shape (3,)
        The force.

    Returns
    -------
    callable
        A force field for :meth:`World.add_force`.
    """
    force = _vector(force, "force")

    def field(world):
        """
        Compute this field's force on every body.

        Parameters
        ----------
        world : World
            The world being stepped.

        Returns
        -------
        ndarray, shape (B, 3)
            Force on each body.
        """
        return np.broadcast_to(force, (world.count, 3))

    return field


def drag(coefficient):
    """
    Create a force resisting motion, proportional to speed.

    It is applied implicitly, so even very light bodies slow down smoothly.

    Parameters
    ----------
    coefficient : float
        Force per unit of speed.

    Returns
    -------
    Relaxation
        A force field for :meth:`World.add_force`.
    """
    coefficient = _number(coefficient, "coefficient", non_negative=True)

    return Relaxation(coefficient, np.zeros(3))


def wind(velocity, *, coefficient=1.0):
    """
    Create a force dragging bodies towards a wind's velocity.

    Parameters
    ----------
    velocity : array_like, shape (3,)
        Velocity of the air.
    coefficient : float, optional
        Force per unit of speed difference.

    Returns
    -------
    Relaxation
        A force field for :meth:`World.add_force`.
    """
    velocity = _vector(velocity, "velocity")
    coefficient = _number(coefficient, "coefficient", non_negative=True)

    return Relaxation(coefficient, velocity)


def attractor(point, *, strength=10.0, softening=0.5):
    """
    Create a pull towards a point that fades with distance squared.

    Every body accelerates equally, as in gravity. A negative strength pushes.

    Parameters
    ----------
    point : array_like, shape (3,)
        The attracting point.
    strength : float, optional
        Scale of the acceleration ``strength * r / (r**2 + softening**2)**1.5``
        at distance ``r``.
    softening : float, optional
        Length that keeps the pull finite near the point.

    Returns
    -------
    callable
        A force field for :meth:`World.add_force`.
    """
    point = _vector(point, "point")
    strength = _number(strength, "strength")
    softening = _number(softening, "softening", positive=True)

    def field(world):
        """
        Compute this field's force on every body.

        Parameters
        ----------
        world : World
            The world being stepped.

        Returns
        -------
        ndarray, shape (B, 3)
            Force on each body.
        """
        offset = point - world._positions
        distance2 = (offset**2).sum(1) + softening**2
        return (strength * offset / distance2[:, None] ** 1.5) * _masses(world)[:, None]

    return field


def vortex(*, center=(0.0, 0.0, 0.0), axis=(0.0, 1.0, 0.0), strength=5.0, pull=0.0):
    """
    Create a swirl around an axis.

    Parameters
    ----------
    center : array_like, shape (3,), optional
        A point on the axis.
    axis : array_like, shape (3,), optional
        The swirl axis.
    strength : float, optional
        Sideways acceleration at unit distance; it fades with distance.
    pull : float, optional
        Acceleration towards the axis.

    Returns
    -------
    callable
        A force field for :meth:`World.add_force`.
    """
    center, axis = _vector(center, "center"), _vector(axis, "axis")
    length = np.linalg.norm(axis)
    if length < 1e-12:
        raise ValueError("axis must be non-zero.")
    axis = axis / length
    strength, pull = _number(strength, "strength"), _number(pull, "pull")

    def field(world):
        """
        Compute this field's force on every body.

        Parameters
        ----------
        world : World
            The world being stepped.

        Returns
        -------
        ndarray, shape (B, 3)
            Force on each body.
        """
        offset = world._positions - center
        radial = offset - axis * (offset @ axis)[:, None]
        distance2 = (radial**2).sum(1)[:, None] + 1e-3
        swirl = np.cross(axis, radial) * strength / distance2
        inward = -radial * pull / np.sqrt(distance2)
        return (swirl + inward) * _masses(world)[:, None]

    return field


def mutual_gravity(*, strength=1.0, softening=0.2):
    """
    Create gravity between every pair of dynamic bodies.

    The cost grows with the square of the body count.

    Parameters
    ----------
    strength : float, optional
        The gravitational constant.
    softening : float, optional
        Length that keeps close encounters finite.

    Returns
    -------
    callable
        A force field for :meth:`World.add_force`.
    """
    strength = _number(strength, "strength")
    softening = _number(softening, "softening", positive=True)

    def field(world):
        """
        Compute this field's force on every body.

        Parameters
        ----------
        world : World
            The world being stepped.

        Returns
        -------
        ndarray, shape (B, 3)
            Force on each body.
        """
        masses = _masses(world)
        force = np.zeros((world.count, 3))
        bodies = np.flatnonzero(masses > 0)
        if bodies.size < 2:
            return force
        offset = world._positions[bodies][None] - world._positions[bodies][:, None]
        distance2 = (offset**2).sum(2) + softening**2
        weight = masses[bodies][None] / distance2**1.5
        np.fill_diagonal(weight, 0.0)
        pull = (offset * weight[..., None]).sum(1)
        force[bodies] = strength * pull * masses[bodies][:, None]
        return force

    return field
