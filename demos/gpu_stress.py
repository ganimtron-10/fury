"""Stress test for fury.physics.GPUWorld.

For each sphere count it fills a box with spheres, lets them fall and settle,
then reports:

- step time (simulation only, GPU work waited for),
- frame time (simulation step plus an offscreen render),
- correctness after settling: NaNs, spheres outside the box, worst overlap
  between neighbours (as a fraction of the diameter), and grid overflow.

Usage::

    python gpu_stress.py                      # 10k 50k 100k 250k 500k 1M
    python gpu_stress.py 1000000 --steps 300  # one count, longer settle
    python gpu_stress.py --render 0           # skip the render timing

On a machine without a GPU this runs on Mesa's software Vulkan (lavapipe),
so the numbers measure CPU emulation, not GPU speed.
"""

import argparse
import json
import time

import numpy as np
from scipy.spatial import cKDTree

from fury import physics, window
from fury.lib import gfx_wgpu


def build(count, radius, seed=0):
    side = (count * (2 * radius) ** 3 / 0.45) ** (1 / 3)
    height = side * 1.6
    world = physics.GPUWorld(substeps=6, iterations=2, friction=0.3)
    rng = np.random.default_rng(seed)
    centers = rng.uniform(
        (-side / 2 + radius, radius, -side / 2 + radius),
        (side / 2 - radius, height, side / 2 - radius),
        (count, 3),
    )
    colors = np.column_stack(
        (
            centers[:, 1] / height,
            0.4 + 0.2 * rng.random(count),
            1 - centers[:, 1] / height,
        )
    )
    spheres = world.add_spheres(centers, radii=radius, colors=colors)
    world.set_container(
        center=(0, height / 2, 0), size=(side, height, side), open_top=True
    )
    return world, spheres, side, height


def wait(world):
    world.overflow  # a readback waits for every submitted kernel


def measure(count, *, radius, settle, render, size, frames):
    world, spheres, side, height = build(count, radius)
    start = time.perf_counter()
    world.step(1)
    wait(world)
    setup = time.perf_counter() - start

    timed = min(20, settle)
    start = time.perf_counter()
    world.step(timed)
    wait(world)
    step_ms = (time.perf_counter() - start) / timed * 1000

    frame_ms = None
    if render:
        scene = window.Scene()
        scene.add(spheres)
        show_manager = window.ShowManager(
            scene=scene, size=size, window_type="offscreen", pixel_ratio=1
        )
        show_manager.add_physics(world)
        show_manager.render()
        show_manager.window.draw()
        start = time.perf_counter()
        for _ in range(frames):
            world.step(1)
            show_manager.render()
            show_manager.window.draw()
            show_manager.snapshot()
        frame_ms = (time.perf_counter() - start) / frames * 1000

    world.step(max(settle - timed - 1, 0))
    positions = world.positions
    velocities = world.velocities
    finite = bool(np.isfinite(positions).all() and np.isfinite(velocities).all())
    half = side / 2 - radius + 1e-3
    outside = int(
        (
            (np.abs(positions[:, 0]) > half)
            | (np.abs(positions[:, 2]) > half)
            | (positions[:, 1] < radius - 1e-3)
        ).sum()
    )
    sample = min(count, 200_000)
    tree = cKDTree(positions)
    distances, _ = tree.query(positions[:sample], k=2)
    overlap = float(max(0.0, 1 - distances[:, 1].min() / (2 * radius)))
    mean_overlap = float(np.clip(1 - distances[:, 1] / (2 * radius), 0, None).mean())
    return {
        "count": count,
        "setup_s": round(setup, 2),
        "step_ms": round(step_ms, 1),
        "steps_per_s": round(1000 / step_ms, 1),
        "frame_ms": None if frame_ms is None else round(frame_ms, 1),
        "settled_steps": settle,
        "finite": finite,
        "outside_box": outside,
        "worst_overlap": round(overlap, 3),
        "mean_overlap": round(mean_overlap, 4),
        "mean_speed": round(float(np.linalg.norm(velocities, axis=1).mean()), 3),
        "grid_overflow": world.overflow,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "counts",
        nargs="*",
        type=int,
        default=[10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000],
    )
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    info = gfx_wgpu.get_shared().adapter.info
    print(f"adapter: {info.get('device')} ({info.get('backend_type')})")
    results = []
    for count in args.counts:
        result = measure(
            count,
            radius=args.radius,
            settle=args.steps,
            render=bool(args.render),
            size=(args.size, args.size),
            frames=args.frames,
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"adapter": dict(info), "results": results}, handle, indent=2)


if __name__ == "__main__":
    main()
