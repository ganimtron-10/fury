"""
Substepped soft-contact solver, written as NumPy array operations.

Simultaneous bounces are resolved with at most ``_RESTITUTION_SWEEPS`` sweeps,
stopping once the velocity change falls below ``_RESTITUTION_TOLERANCE``.
"""

import numpy as np

from fury.physics._math import cross, integrate_quaternions, world_inverse_inertia

_RESTITUTION_SWEEPS = 32
_RESTITUTION_TOLERANCE = 1e-9


def color_contacts(a, b, is_dynamic, max_colors):
    """
    Split contacts into batches in which no dynamic body appears twice.

    Bodies in a batch can be updated with a single array assignment. Contacts
    that do not fit in ``max_colors`` batches are returned separately, to be
    solved together with mass splitting.

    Parameters
    ----------
    a, b : ndarray, shape (M,)
        Body indices of each contact.
    is_dynamic : ndarray, shape (B,)
        Boolean mask of dynamic bodies.
    max_colors : int
        Maximum number of batches.

    Returns
    -------
    batches : list of ndarray
        Contact indices for each batch.
    leftover : ndarray
        Contact indices that did not fit.
    """
    remaining = np.arange(a.size)
    batches = []
    first_seen = np.empty(is_dynamic.size, dtype=np.intp)
    while remaining.size and len(batches) < max_colors:
        priority = np.arange(remaining.size)
        first_seen.fill(np.iinfo(np.intp).max)
        body_a, body_b = a[remaining], b[remaining]
        np.minimum.at(first_seen, body_a, priority)
        np.minimum.at(first_seen, body_b, priority)
        selected = (~is_dynamic[body_a] | (first_seen[body_a] == priority)) & (
            ~is_dynamic[body_b] | (first_seen[body_b] == priority)
        )
        batches.append(remaining[selected])
        remaining = remaining[~selected]
    return batches, remaining


class _Batch:
    """Everything one batch needs, sliced once per step."""

    __slots__ = (
        "index",
        "count",
        "a",
        "b",
        "jac_a",
        "jac_b",
        "w_a",
        "w_b",
        "eff",
        "friction",
        "tangent_rows",
        "jacobi",
    )


class Solver:
    """Resolve contacts and integrate a world, remembering impulses across steps."""

    def __init__(self):
        self.clear_cache()
        self.peak_normal = np.empty(0)

    def clear_cache(self):
        """Forget impulses remembered from previous steps."""
        self._keys = np.empty(0, dtype=np.int64)
        self._impulses = np.empty((0, 3))

    def _softness(self, world, h):
        """
        Compute constraint softness for a substep.

        Parameters
        ----------
        world : World
            The world being stepped.
        h : float
            Substep duration.
        """
        hertz = min(world.contact_hertz, 0.25 / h)
        omega = 2.0 * np.pi * hertz
        damped = 2.0 * world.contact_damping + h * omega
        stiffness = h * omega * damped
        softening = 1.0 / (1.0 + stiffness)
        self._bias_rate = omega / damped
        self._mass_scale = stiffness * softening
        self._impulse_scale = softening

    def _batch(self, index, a, b, jac, w, *, split=None):
        """
        Build the arrays used to solve a set of contacts.

        Parameters
        ----------
        index : ndarray
            Contact indices of the batch.
        a, b : ndarray
            Body indices of all contacts.
        jac, w : ndarray, shape (3M, 12)
            Jacobians and inverse-mass-weighted Jacobians.
        split : ndarray, optional
            Per-body contact counts for mass splitting.

        Returns
        -------
        _Batch
            The prepared batch.
        """
        contacts = a.size
        rows = np.concatenate([index, index + contacts, index + 2 * contacts])
        batch = _Batch()
        batch.index = index
        batch.count = index.size
        batch.a, batch.b = np.tile(a[index], 3), np.tile(b[index], 3)
        batch.jac_a, batch.jac_b = jac[rows, :6], jac[rows, 6:]
        batch.w_a, batch.w_b = w[rows, :6], w[rows, 6:]
        diag_a = (batch.jac_a * batch.w_a).sum(1)
        diag_b = (batch.jac_b * batch.w_b).sum(1)
        if split is not None:
            diag_a = diag_a * split[batch.a]
            diag_b = diag_b * split[batch.b]
        diagonal = diag_a + diag_b
        batch.eff = np.divide(
            1.0, diagonal, out=np.zeros_like(diagonal), where=diagonal > 0
        )
        batch.friction = self._friction[index]
        batch.tangent_rows = rows[batch.count :]
        batch.jacobi = split is not None
        return batch

    def _solve_batch(self, batch, velocity, displacement, inv_h, *, use_bias, push):
        """
        Run one solver sweep over a batch.

        Parameters
        ----------
        batch : _Batch
            The batch to solve.
        velocity : ndarray, shape (B, 6)
            Body velocities, updated in place.
        displacement : ndarray, shape (B, 6)
            Motion accumulated so far this step.
        inv_h : float
            Inverse substep duration.
        use_bias : bool
            Whether to push penetrating contacts apart.
        push : float
            Largest speed at which contacts are pushed apart.
        """
        count, body_a, body_b = batch.count, batch.a, batch.b
        jac_a, jac_b, w_a, w_b, eff = (
            batch.jac_a,
            batch.jac_b,
            batch.w_a,
            batch.w_b,
            batch.eff,
        )
        index, impulse = batch.index, self._lambda

        relative = (jac_a * velocity[body_a]).sum(1) + (jac_b * velocity[body_b]).sum(1)
        separation = (
            self._separation[index]
            + (jac_a[:count] * displacement[body_a[:count]]).sum(1)
            + (jac_b[:count] * displacement[body_b[:count]]).sum(1)
        )
        speculative = separation > 0
        if use_bias:
            bias = np.where(
                speculative,
                separation * inv_h,
                np.maximum(self._bias_rate * separation, -push),
            )
            mass_scale = np.where(speculative, 1.0, self._mass_scale)
            impulse_scale = np.where(speculative, 0.0, self._impulse_scale)
        else:
            bias = np.where(speculative, separation * inv_h, 0.0)
            mass_scale, impulse_scale = 1.0, 0.0

        old = impulse[index]
        delta = -eff[:count] * mass_scale * (relative[:count] + bias)
        delta -= impulse_scale * old
        new = np.maximum(old + delta, 0.0)
        impulse[index] = new
        self.peak_normal[index] = np.maximum(self.peak_normal[index], new)

        tangent_rows = batch.tangent_rows
        old_tangent = impulse[tangent_rows]
        new_tangent = old_tangent - eff[count:] * relative[count:]
        magnitude = np.hypot(new_tangent[:count], new_tangent[count:])
        limit = batch.friction * new
        scale = np.divide(
            limit, magnitude, out=np.ones_like(limit), where=magnitude > limit
        )
        new_tangent = new_tangent * np.tile(scale, 2)
        impulse[tangent_rows] = new_tangent

        change = np.concatenate([new - old, new_tangent - old_tangent])[:, None]
        if batch.jacobi:
            np.add.at(velocity, body_a, change * w_a)
            np.add.at(velocity, body_b, change * w_b)
        else:
            velocity[body_a[:count]] += (change * w_a).reshape(3, count, 6).sum(0)
            velocity[body_b[:count]] += (change * w_b).reshape(3, count, 6).sum(0)

    def _integrate(self, world, velocity, displacement, h):
        """
        Advance positions and orientations by one substep.

        Parameters
        ----------
        world : World
            The world being stepped.
        velocity : ndarray, shape (B, 6)
            Body velocities.
        displacement : ndarray, shape (B, 6)
            Accumulated motion, updated in place.
        h : float
            Substep duration.
        """
        step = velocity * h
        world._positions += step[:, :3]
        displacement += step
        integrate_quaternions(world._rotations, step[:, 3:])

    def _prepare_rows(self, world, contacts, inv_inertia):
        """
        Build Jacobians for every contact.

        Parameters
        ----------
        world : World
            The world being stepped.
        contacts : Contacts
            Contacts of this step.
        inv_inertia : ndarray, shape (B, 3, 3)
            World-space inverse inertia.

        Returns
        -------
        tuple of ndarray
            Jacobians, weighted Jacobians, and repeated body indices.
        """
        a, b, points, normals = contacts[:4]
        lever_a, lever_b = points - world._positions[a], points - world._positions[b]

        helper = np.where(
            np.abs(normals[:, 0, None]) < 0.6, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]
        )
        tangent1 = cross(normals, helper)
        tangent1 /= np.linalg.norm(tangent1, axis=1, keepdims=True)
        tangent2 = cross(normals, tangent1)
        directions = np.concatenate([normals, tangent1, tangent2])

        body_a, body_b = np.tile(a, 3), np.tile(b, 3)
        lever_a, lever_b = np.tile(lever_a, (3, 1)), np.tile(lever_b, (3, 1))
        jac = np.concatenate(
            [
                -directions,
                -cross(lever_a, directions),
                directions,
                cross(lever_b, directions),
            ],
            axis=1,
        )
        inv_mass = world._active_inverse_mass()
        w = np.concatenate(
            [
                jac[:, :3] * inv_mass[body_a],
                (inv_inertia[body_a] @ jac[:, 3:6, None])[..., 0],
                jac[:, 6:9] * inv_mass[body_b],
                (inv_inertia[body_b] @ jac[:, 9:, None])[..., 0],
            ],
            axis=1,
        )
        return jac, w, body_a, body_b

    def _warm_start_impulses(self, keys):
        """
        Look up impulses remembered from the previous step.

        Parameters
        ----------
        keys : ndarray, shape (M,)
            Contact identifiers.

        Returns
        -------
        ndarray, shape (3M,)
            Starting impulses, by row.
        """
        count = keys.size
        impulses = np.zeros(3 * count)
        if self._keys.size == 0 or count == 0:
            return impulses
        found = np.searchsorted(self._keys, keys).clip(0, self._keys.size - 1)
        hit = self._keys[found] == keys
        for row in range(3):
            impulses[row * count : (row + 1) * count][hit] = self._impulses[
                found[hit], row
            ]
        return impulses

    def _apply_restitution(
        self, a, b, jac, w, velocity, approach, threshold, is_dynamic
    ):
        """
        Apply bounce to contacts that were approaching quickly.

        A contact's final impulse is often zero by this point, because warm
        starting can separate the bodies early. The peak impulse tells whether
        the contact actually fired. A body with several firing contacts needs
        repeated sweeps before every contact reaches its target speed.

        Parameters
        ----------
        a, b : ndarray, shape (M,)
            Body indices.
        jac, w : ndarray, shape (3M, 12)
            Jacobians and weighted Jacobians.
        velocity : ndarray, shape (B, 6)
            Body velocities, updated in place.
        approach : ndarray, shape (M,)
            Normal velocity at the start of the step.
        threshold : float
            Approach speed below which contacts do not bounce.
        is_dynamic : ndarray, shape (B,)
            Boolean mask of bodies the solver may move.
        """
        fired = np.flatnonzero(
            (approach < -threshold) & (self.peak_normal > 0) & (self._restitution > 0)
        )
        if fired.size == 0:
            return
        groups, _ = color_contacts(a[fired], b[fired], is_dynamic, fired.size)
        groups = [fired[group] for group in groups]
        for _ in range(_RESTITUTION_SWEEPS):
            largest = 0.0
            for index in groups:
                body_a, body_b = a[index], b[index]
                jac_a, jac_b = jac[index, :6], jac[index, 6:]
                w_a, w_b = w[index, :6], w[index, 6:]
                relative = (jac_a * velocity[body_a]).sum(1)
                relative += (jac_b * velocity[body_b]).sum(1)
                diagonal = (jac_a * w_a).sum(1) + (jac_b * w_b).sum(1)
                delta = -(relative + self._restitution[index] * approach[index])
                delta = np.divide(
                    delta, diagonal, out=np.zeros_like(delta), where=diagonal > 0
                )
                old = self._lambda[index]
                new = np.maximum(old + delta, 0.0)
                self._lambda[index] = new
                change = (new - old)[:, None]
                gain_a, gain_b = change * w_a, change * w_b
                velocity[body_a] += gain_a
                velocity[body_b] += gain_b
                largest = max(
                    largest, float(np.abs(gain_a).max()), float(np.abs(gain_b).max())
                )
            if largest < _RESTITUTION_TOLERANCE:
                break

    def solve(self, world, contacts, rotation, acceleration):
        """
        Advance a world by one step.

        Parameters
        ----------
        world : World
            The world to advance.
        contacts : Contacts
            Contacts found for this step.
        rotation : ndarray, shape (B, 3, 3)
            Body rotation matrices at the start of the step.
        acceleration : ndarray, shape (B, 6)
            Linear and angular acceleration applied each substep.
        """
        h = world.dt / world.substeps
        inv_h = 1.0 / h
        velocity = world._velocities
        displacement = np.zeros_like(velocity)
        damping = world._substep_damping(h)

        count = contacts.a.size
        self.peak_normal = np.zeros(count)
        if count == 0:
            for _ in range(world.substeps):
                velocity += acceleration * h
                velocity *= damping
                self._integrate(world, velocity, displacement, h)
            self.clear_cache()
            return

        self._softness(world, h)
        a, b = contacts.a, contacts.b
        inv_inertia = world_inverse_inertia(rotation, world._active_inverse_inertia())
        jac, w, body_a, body_b = self._prepare_rows(world, contacts, inv_inertia)

        self._separation = contacts.separation
        self._friction = np.sqrt(world._friction[a] * world._friction[b])
        self._restitution = np.maximum(world._restitution[a], world._restitution[b])

        keys = (a * (world.count + 2) + b) * 4096 + contacts.feature
        self._lambda = self._warm_start_impulses(keys)

        batches, leftover = color_contacts(
            a, b, world._solver_dynamic_mask(), world.max_colors
        )
        solve_batches = [self._batch(index, a, b, jac, w) for index in batches]
        if leftover.size:
            split = np.zeros(world.count)
            np.add.at(split, a[leftover], 1)
            np.add.at(split, b[leftover], 1)
            split = np.maximum(split, 1)
            solve_batches.append(self._batch(leftover, a, b, jac, w, split=split))

        approach = (jac[:count, :6] * velocity[a]).sum(1)
        approach += (jac[:count, 6:] * velocity[b]).sum(1)

        for _ in range(world.substeps):
            velocity += acceleration * h
            velocity *= damping
            warm = self._lambda[:, None]
            np.add.at(velocity, body_a, warm * w[:, :6])
            np.add.at(velocity, body_b, warm * w[:, 6:])
            for batch in solve_batches:
                self._solve_batch(
                    batch,
                    velocity,
                    displacement,
                    inv_h,
                    use_bias=True,
                    push=world.max_push_speed,
                )
            self._integrate(world, velocity, displacement, h)
            for batch in solve_batches:
                self._solve_batch(
                    batch,
                    velocity,
                    displacement,
                    inv_h,
                    use_bias=False,
                    push=world.max_push_speed,
                )

        self._apply_restitution(
            a,
            b,
            jac,
            w,
            velocity,
            approach,
            world.restitution_threshold,
            world._solver_dynamic_mask(),
        )

        order = keys.argsort()
        self._keys = keys[order]
        self._impulses = np.stack(
            [
                self._lambda[:count],
                self._lambda[count : 2 * count],
                self._lambda[2 * count :],
            ],
            axis=1,
        )[order]
