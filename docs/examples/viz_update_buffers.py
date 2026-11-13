"""
===============================================
Animate NumPy Data by Updating Buffers in Place
===============================================

Many simulations produce new NumPy arrays every frame. The simplest way to
show them is to remove the old actor and create a new one, but that rebuilds
the geometry, the material and the GPU buffers each time.

This tutorial shows the faster pattern: create each actor once, then write the
new values into its existing buffers and tell FURY to upload them. We animate
a swirling cloud of particles and their trails, and colour the particles by
speed.
"""

###############################################################################
# Import the modules we need.

import time

import numpy as np

from fury import actor, window

###############################################################################
# Some particles moving in a vortex ring, like a smoke ring. Each particle
# swirls around the z axis and also circles the ring's core, a circle of
# radius ``RING_RADIUS`` in the xy plane. Particles far from the core move
# faster, so colouring by speed shows the structure of the flow.

N_PARTICLES = 2000
TRAIL_LENGTH = 12
DT = 0.01
RING_RADIUS = 4.0

rng = np.random.default_rng(0)
angle = rng.uniform(0.0, 2.0 * np.pi, N_PARTICLES)
core_angle = rng.uniform(0.0, 2.0 * np.pi, N_PARTICLES)
core_distance = rng.uniform(0.3, 2.5, N_PARTICLES)
ring = RING_RADIUS + core_distance * np.cos(core_angle)
positions = np.stack(
    (
        ring * np.cos(angle),
        ring * np.sin(angle),
        core_distance * np.sin(core_angle),
    ),
    axis=1,
)


def velocity(points):
    """
    Evaluate the vortex ring flow.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Particle positions.

    Returns
    -------
    ndarray, shape (N, 3)
        Velocity at each position.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    radius = np.sqrt(x**2 + y**2)
    radial = np.stack((x / radius, y / radius, np.zeros_like(x)), axis=1)
    around_z = np.stack((-y / radius, x / radius, np.zeros_like(x)), axis=1)
    up = np.array([0.0, 0.0, 1.0])
    around_core = 2.0 * (-z[:, None] * radial + (radius - RING_RADIUS)[:, None] * up)
    return 1.5 * around_z + around_core


###############################################################################
# Create the actors once. A point actor stores one position and one colour
# per particle. A line actor stores all trails in one buffer, with a row of
# NaN between trails so they are not joined.

trails = np.repeat(positions[:, None, :], TRAIL_LENGTH, axis=1)

points_actor = actor.point(positions, colors=(1.0, 1.0, 1.0), size=5.0)
trails_actor = actor.line(list(trails), colors=(0.3, 0.5, 0.9), opacity=0.5)

scene = window.Scene()
scene.background = (0.03, 0.03, 0.06)
scene.add(trails_actor, points_actor)

###############################################################################
# The buffers live on ``actor.geometry``. Each one wraps a NumPy array in
# ``.data``. Its shape is fixed when the actor is created, so keep the number
# of particles constant and only change the values.

print("positions buffer:", points_actor.geometry.positions.data.shape)
print("colors buffer:", points_actor.geometry.colors.data.shape)
print("trail buffer:", trails_actor.geometry.positions.data.shape)

###############################################################################
# To update a buffer, assign into ``.data[:]`` and call ``update_full()`` so
# the new values are uploaded on the next draw. Use ``update_range(offset,
# size)`` instead when only a slice of the buffer changed.
#
# The trail buffer is filled from ``trail_rows``, a padded
# ``(N, TRAIL_LENGTH + 1, 3)`` array. Its last row for each trail,
# ``trail_rows[:, TRAIL_LENGTH]``, stays NaN and acts as the separator. The line
# buffer has no separator after the final trail, so the last row is dropped
# with ``[:-1]``.

trail_rows = np.full((N_PARTICLES, TRAIL_LENGTH + 1, 3), np.nan, dtype=np.float32)


def speed_colors(speed):
    """
    Map speeds to colours from blue (slow) to orange (fast).

    Parameters
    ----------
    speed : ndarray, shape (N,)
        Particle speeds.

    Returns
    -------
    ndarray, shape (N, 3)
        RGB colours.
    """
    t = np.clip((speed - 2.0) / 4.5, 0.0, 1.0)[:, None]
    return (1.0 - t) * np.array([0.2, 0.5, 1.0]) + t * np.array([1.0, 0.6, 0.1])


def update_actors(points, flow):
    """
    Write new particle data into the existing actor buffers.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        New particle positions.
    flow : ndarray, shape (N, 3)
        Particle velocities, used for the colours.
    """
    points_actor.geometry.positions.data[:] = points
    points_actor.geometry.positions.update_full()

    colors = points_actor.geometry.colors.data
    colors[:, :3] = speed_colors(np.linalg.norm(flow, axis=1))
    points_actor.geometry.colors.update_full()

    trails[:, 1:] = trails[:, :-1]
    trails[:, 0] = points
    trail_rows[:, :TRAIL_LENGTH] = trails
    trails_actor.geometry.positions.data[:] = trail_rows.reshape(-1, 3)[:-1]
    trails_actor.geometry.positions.update_full()


###############################################################################
# How much does this save? Compare rebuilding the actors with updating them
# in place for the same data. Both timings measure CPU time only; the GPU
# upload and the draw happen later and are not included.

repeats = 20
start = time.perf_counter()
for _ in range(repeats):
    actor.point(positions, colors=speed_colors(np.zeros(N_PARTICLES)), size=5.0)
    actor.line(list(trails), colors=(0.3, 0.5, 0.9), opacity=0.5)
rebuild_ms = (time.perf_counter() - start) / repeats * 1000

start = time.perf_counter()
for _ in range(repeats):
    update_actors(positions, np.zeros_like(positions))
update_ms = (time.perf_counter() - start) / repeats * 1000

print(f"rebuild actors: {rebuild_ms:.2f} ms of CPU time per frame")
print(f"update buffers: {update_ms:.2f} ms of CPU time per frame")

###############################################################################
# Finally, advance the particles every frame with a timer callback and push
# the new values into the buffers.


def step():
    """Advance the particles by one time step and update the actors."""
    flow = velocity(positions)
    positions[:] += DT * flow
    update_actors(positions, flow)


show_manager = window.ShowManager(
    scene=scene, size=(800, 800), title="Update Buffers In Place"
)
show_manager.screens[0].camera.local.position = (0.0, -13.0, 8.0)
show_manager.screens[0].camera.look_at((0.0, 0.0, 0.0))
show_manager.register_callback(step, 1 / 60, True, "update_buffers")
show_manager.start()
