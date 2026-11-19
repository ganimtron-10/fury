"""Physics demos for ``fury.physics`` (PRs 01-06).

Open one in a window (close the window to stop)::

    python demos/physics_demos.py 06-forces

Record a GIF of one or all scenes with ``ShowManager.record_physics``::

    python demos/physics_demos.py --record 06-forces
    python demos/physics_demos.py --record            # all scenes

GIFs go to ``demos/out/``. Scenes: 01-core, 02-boxes, 03-containers,
04-actors, 05-loop, 06-forces.
"""

import os
import sys

import numpy as np

from fury import actor, physics, window
from fury.physics import forces

RNG = np.random.default_rng(4)


def _show(scene, eye, target=(0, 0, 0), *, record=False):
    show = window.ShowManager(
        scene=scene,
        size=(400, 300) if record else (1100, 800),
        window_type="offscreen" if record else "default",
        title="FURY physics",
    )
    camera = show.screens[0].camera
    camera.local.position = eye
    camera.look_at(target)
    return show


def _scene():
    scene = window.Scene()
    scene.background = (0.97, 0.97, 0.99)
    return scene


def _colors(n):
    return RNG.uniform(0.15, 0.95, (n, 3))


def _wire_box(half, center=(0, 0, 0)):
    return actor.box(
        np.asarray([center], float),
        scales=2 * np.asarray([half], float),
        colors=(0.2, 0.2, 0.25),
        wireframe=True,
        material="basic",
    )


def core(scene, world):
    """PR 01: spheres only. Balls rain into a bowl built from static spheres."""
    bowl = []
    for ring, (polar, count) in enumerate([(0.0, 1), (0.45, 8), (0.9, 14), (1.3, 18)]):
        for k in range(count):
            azimuth = 2 * np.pi * (k + 0.5 * ring) / count
            bowl.append(
                4.0
                * np.array(
                    [
                        np.sin(polar) * np.cos(azimuth),
                        -np.cos(polar),
                        np.sin(polar) * np.sin(azimuth),
                    ]
                )
            )
    bowl = np.array(bowl)
    shell = actor.sphere(bowl, radii=0.95, colors=(0.55, 0.57, 0.62))
    world.add(shell, kind="static", friction=0.6)
    count = 140
    drops = np.c_[
        RNG.uniform(-1.5, 1.5, count),
        np.linspace(1, 12, count),
        RNG.uniform(-1.5, 1.5, count),
    ]
    balls = actor.sphere(drops, radii=RNG.uniform(0.2, 0.3, count), colors=_colors(count))
    world.add(balls, restitution=0.5, friction=0.4)
    scene.add(shell, balls)
    return (0, 5, 11), (0, -1.5, 0), 7.0


def boxes(scene, world):
    """PR 02: boxes and spheres tumble onto a floor and a box stack."""
    floor = actor.box(
        np.array([[0, -2.5, 0]], float), scales=(12, 1, 8), colors=(0.8, 0.8, 0.84)
    )
    world.add(floor, kind="static", friction=0.7)
    stack = actor.box(
        np.array([[0, -1.5 + i * 1.001, 0] for i in range(4)], float),
        scales=(1, 1, 1),
        colors=(0.85, 0.45, 0.2),
    )
    world.add(stack, friction=0.7, restitution=0.0)
    count = 30
    centers = np.c_[
        RNG.uniform(-3, 3, count), np.linspace(4, 16, count), RNG.uniform(-2, 2, count)
    ]
    crates = actor.box(
        centers[: count // 2], scales=RNG.uniform(0.4, 0.9, (count // 2, 3)),
        colors=_colors(count // 2),
    )
    balls = actor.sphere(
        centers[count // 2 :], radii=RNG.uniform(0.25, 0.4, count - count // 2),
        colors=_colors(count - count // 2),
    )
    world.add(crates, friction=0.6, restitution=0.1)
    world.add(balls, friction=0.4, restitution=0.4)
    scene.add(floor, stack, crates, balls)
    return (2, 4, 13), (0, -0.5, 0), 6.0


def containers(scene, world):
    """PR 03: a box of balls and crates, settled, shaken, then tilted."""
    half = np.array([3.0, 2.5, 2.0])
    tank = _wire_box(half)
    holder = world.add_container(actor=tank, restitution=0.3)
    count = 70
    centers = np.c_[
        RNG.uniform(-2.5, 2.5, count),
        RNG.uniform(-2.0, 2.0, count),
        RNG.uniform(-1.5, 1.5, count),
    ]
    balls = actor.sphere(
        centers[:40], radii=RNG.uniform(0.2, 0.32, 40), colors=_colors(40)
    )
    crates = actor.box(
        centers[40:], scales=RNG.uniform(0.35, 0.6, (30, 3)), colors=_colors(30)
    )
    world.add(balls, restitution=0.35, friction=0.4)
    world.add(crates, restitution=0.2, friction=0.5)
    scene.add(tank, balls, crates)

    def schedule(world):
        if abs(world.time - 1.5) < world.dt / 2:
            holder.shake(amplitude=0.5, frequency=2.5, axis=(1, 0.3, 0), duration=2.0)
        if abs(world.time - 4.5) < world.dt / 2:
            holder.tilt(axis=(0, 0, 1), angle=0.6, duration=1.0)
        if abs(world.time - 6.5) < world.dt / 2:
            holder.tilt(axis=(0, 0, 1), angle=-0.6, duration=1.0)

    world.on_step(schedule)
    return (0, 3, 13), (0, -0.5, 0), 8.5


def actors(scene, world):
    """PR 04: FURY actors as bodies, plus a container driven by its actor."""
    half = np.array([3.5, 2.5, 2.5])
    drum = _wire_box(half)
    holder = world.add_container(actor=drum, restitution=0.2)
    count = 60
    centers = np.c_[
        RNG.uniform(-3, 3, count), RNG.uniform(-2, 2, count), RNG.uniform(-2, 2, count)
    ]
    cubes = actor.box(
        centers[:30], scales=0.6, colors=_colors(30)
    )
    balls = actor.sphere(centers[30:], radii=0.3, colors=_colors(30))
    world.add(cubes, friction=0.6)
    world.add(balls, friction=0.3, restitution=0.4)
    scene.add(drum, cubes, balls)

    def schedule(world):
        if abs(world.time - 1.0) < world.dt / 2:
            holder.spin(axis=(0, 0, 1), speed=1.2)

    world.on_step(schedule)
    return (0, 1.5, 14), (0, 0, 0), 7.0


def loop(scene, world):
    """PR 05: a domino run recorded with record_physics at a fixed step."""
    floor = actor.box(
        np.array([[0, -0.5, 0]], float), scales=(20, 1, 6), colors=(0.82, 0.82, 0.86)
    )
    world.add(floor, kind="static", friction=0.8)
    count = 22
    x = np.linspace(-7.5, 7.5, count)
    z = 1.2 * np.sin(x / 2.5)
    dominoes = actor.box(
        np.c_[x, np.full(count, 0.8), z],
        scales=(0.15, 1.6, 0.8),
        colors=np.c_[np.linspace(0.9, 0.2, count), np.full(count, 0.4),
                     np.linspace(0.2, 0.9, count)],
    )
    tiles = world.add(dominoes, friction=0.6, restitution=0.0)
    ball = actor.sphere(np.array([[-10.0, 0.35, 0.0]]), radii=0.35,
                        colors=(0.2, 0.2, 0.2))
    world.add(ball, velocities=(6.0, 0, 0), mass=2.0)
    scene.add(floor, dominoes, ball)
    del tiles
    return (0, 7, 12), (0, 0, 0), 6.0


def force_fields(scene, world):
    """PR 06: a vortex stirs a cloud of balls, then an explosion scatters them."""
    world.gravity = np.zeros(3)
    half = np.array([5.0, 5.0, 5.0])
    tank = _wire_box(half)
    world.add_container(actor=tank, restitution=0.6)
    count = 160
    centers = np.c_[
        RNG.uniform(-4, 4, count), RNG.uniform(-4, 4, count), RNG.uniform(-4, 4, count)
    ]
    balls = actor.sphere(centers, radii=RNG.uniform(0.15, 0.3, count),
                         colors=_colors(count))
    world.add(balls, restitution=0.5, friction=0.2)
    scene.add(tank, balls)
    swirl = forces.vortex(axis=(0, 1, 0), strength=8.0, pull=4.0)
    world.add_force(swirl)

    def schedule(world):
        if abs(world.time - 4.0) < world.dt / 2:
            world.remove_force(swirl)
            world.explode((0, 0, 0), strength=40.0, radius=6.0)

    world.on_step(schedule)
    return (6, 6, 14), (0, 0, 0), 7.0


SCENES = {
    "01-core": core,
    "02-boxes": boxes,
    "03-containers": containers,
    "04-actors": actors,
    "05-loop": loop,
    "06-forces": force_fields,
}


def run(name, record):
    """Build one scene and show it in a window, or record it to a GIF."""
    scene = _scene()
    world = physics.World()
    eye, target, seconds = SCENES[name](scene, world)
    show = _show(scene, eye, target, record=record)
    show.add_physics(world)
    if not record:
        show.start()
        return
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"physics-{name}.gif")
    frames = show.record_physics(path, duration=seconds, fps=15)
    print(name, frames, "frames ->", path)


if __name__ == "__main__":
    args = sys.argv[1:]
    record = "--record" in args
    names = [a for a in args if a != "--record"]
    if not names and not record:
        print(__doc__)
        sys.exit(1)
    for name in names or list(SCENES):
        if name not in SCENES:
            sys.exit(f"Unknown scene {name!r}. Choose from {', '.join(SCENES)}.")
        run(name, record)
