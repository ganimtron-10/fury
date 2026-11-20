"""
===============================
GPU physics: a million spheres
===============================

:class:`fury.physics.GPUWorld` simulates spheres with WGSL compute shaders. The
sphere state never leaves the GPU: the compute kernels move the spheres and
the renderer draws them from the same buffer.

This example pours spheres into a box, shakes it, stirs them with a vortex
and finally blows them out of the top. Raise ``SPHERES`` to a million on a
desktop GPU.
"""

import numpy as np

from fury import physics, window

SPHERES = 50_000
RADIUS = 0.05

###############################################################################
# Size the box so the spheres fill about a third of it once settled, and scatter
# them through it to fall.

side = (SPHERES * (2 * RADIUS) ** 3 / 0.45) ** (1 / 3)
height = 1.6 * side
rng = np.random.default_rng(1)
inner = side / 2 - RADIUS
centers = rng.uniform(
    (-inner, RADIUS, -inner), (inner, height - RADIUS, inner), (SPHERES, 3)
)
colors = np.column_stack(
    (
        0.2 + 0.8 * centers[:, 1] / height,
        0.35 + 0.2 * rng.random(SPHERES),
        1.0 - 0.8 * centers[:, 1] / height,
    )
)

###############################################################################
# Build the world. ``add_spheres`` returns the actor that draws them.

world = physics.GPUWorld(substeps=6, friction=0.3)
spheres = world.add_spheres(centers, radii=RADIUS, colors=colors)
box = world.set_container(
    center=(0, height / 2, 0), size=(side, height, side), open_top=True
)

###############################################################################
# Script the show: shake the box, add a vortex, then an explosion.

vortex = physics.vortex(center=(0, 0, 0), strength=4.0, pull=1.0)


def direct(world):
    """
    Start each part of the show at its time.

    Parameters
    ----------
    world : fury.physics.GPUWorld
        The world being stepped.
    """
    if abs(world.time - 3.0) < world.dt / 2:
        box.shake(amplitude=0.15 * side, frequency=1.5, duration=2.0)
    elif abs(world.time - 5.5) < world.dt / 2:
        world.add_force(vortex)
    elif abs(world.time - 8.0) < world.dt / 2:
        world.remove_force(vortex)
        world.explode((0, 0, 0), strength=12.0, radius=side)


world.on_step(direct)

###############################################################################
# Show it. The physics advances in step with drawing.

scene = window.Scene(background=(0.06, 0.07, 0.09))
scene.add(spheres)
show_manager = window.ShowManager(
    scene=scene, size=(900, 700), title="FURY GPU physics"
)
show_manager.screens[0].camera.local.position = (side * 1.6, height * 1.4, side * 1.9)
show_manager.screens[0].camera.look_at((0, height * 0.3, 0))
show_manager.add_physics(world)

if __name__ == "__main__":
    show_manager.start()
